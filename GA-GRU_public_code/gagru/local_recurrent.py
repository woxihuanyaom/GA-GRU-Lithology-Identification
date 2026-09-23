from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn

from .residual_bigru import make_loader
from .search import GRUSearchCandidate
from .training import set_reproducible_seed


STANDARD_ARCHITECTURES = ("vanilla_rnn", "lstm", "gru")
CENTER_AWARE_ARCHITECTURES = (
    "center_residual_gru",
    "local_residual_gru",
    "center_bigru",
)
ARCHITECTURES = (*STANDARD_ARCHITECTURES, *CENTER_AWARE_ARCHITECTURES)


class LocalRecurrentClassifier(nn.Module):
    def __init__(
        self,
        architecture: str,
        input_size: int,
        output_size: int,
        candidate: GRUSearchCandidate,
    ) -> None:
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f"Unknown local recurrent architecture: {architecture}")
        base_architecture = (
            architecture if architecture in STANDARD_ARCHITECTURES else "gru"
        )
        recurrent_class = {
            "vanilla_rnn": nn.RNN,
            "lstm": nn.LSTM,
            "gru": nn.GRU,
        }[base_architecture]
        bidirectional = architecture == "center_bigru"
        kwargs = {
            "input_size": input_size,
            "hidden_size": candidate.hidden_size,
            "num_layers": candidate.num_layers,
            "batch_first": True,
            "dropout": candidate.dropout if candidate.num_layers > 1 else 0.0,
            "bidirectional": bidirectional,
        }
        if base_architecture == "vanilla_rnn":
            kwargs["nonlinearity"] = "tanh"
        self.architecture = architecture
        self.recurrent = recurrent_class(**kwargs)
        recurrent_width = candidate.hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Linear(recurrent_width, output_size)
        self.local_classifier: nn.Linear | None = None
        if architecture == "center_residual_gru":
            self.local_classifier = nn.Linear(input_size, output_size)
        elif architecture == "local_residual_gru":
            self.local_classifier = nn.Linear(3 * input_size, output_size)
        if self.local_classifier is not None:
            nn.init.zeros_(self.local_classifier.weight)
            nn.init.zeros_(self.local_classifier.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError(
                "Recurrent input must have [batch, sequence, features] shape"
            )
        output, _ = self.recurrent(values)
        center = values.shape[1] // 2
        if self.architecture == "center_bigru":
            return self.classifier(output[:, center, :])
        logits = self.classifier(output[:, -1, :])
        if self.architecture == "center_residual_gru":
            if self.local_classifier is None:
                raise RuntimeError("Center residual classifier was not initialized")
            return logits + self.local_classifier(values[:, center, :])
        if self.architecture == "local_residual_gru":
            if values.shape[1] < 3:
                raise ValueError("Local residual GRU requires at least three samples")
            if self.local_classifier is None:
                raise RuntimeError("Local residual classifier was not initialized")
            local = values[:, center - 1 : center + 2, :].flatten(start_dim=1)
            return logits + self.local_classifier(local)
        return logits

    @property
    def trainable_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


@dataclass
class LocalRecurrentFitResult:
    state_dict: dict[str, torch.Tensor]
    best_epoch: int
    epochs_completed: int
    validation_macro_f1: float
    validation_accuracy: float
    validation_balanced_accuracy: float
    best_validation_loss: float
    runtime_seconds: float
    trainable_parameters: int
    history: list[dict[str, float | int]]


def build_local_recurrent(
    architecture: str,
    candidate: GRUSearchCandidate,
    X: np.ndarray,
    *,
    output_size: int,
    device: torch.device,
) -> LocalRecurrentClassifier:
    if X.ndim != 3:
        raise ValueError("X must have [samples, sequence, features] shape")
    return LocalRecurrentClassifier(
        architecture,
        input_size=int(X.shape[2]),
        output_size=output_size,
        candidate=candidate,
    ).to(device)


def predict_local_recurrent_logits(
    model: nn.Module,
    X: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = make_loader(
        X,
        np.zeros(len(X), dtype=np.int64),
        batch_size=batch_size,
        seed=0,
        shuffle=False,
        device=device,
    )
    parts = []
    model.eval()
    with torch.inference_mode():
        for X_batch, _ in loader:
            parts.append(model(X_batch.to(device, non_blocking=True)).cpu().numpy())
    return np.concatenate(parts)


def _cross_entropy_criterion(
    output_size: int,
    class_weights: np.ndarray | None,
    device: torch.device,
) -> nn.CrossEntropyLoss:
    if class_weights is None:
        return nn.CrossEntropyLoss()
    weights = np.asarray(class_weights, dtype=np.float32)
    if weights.shape != (output_size,):
        raise ValueError(f"class_weights must have shape ({output_size},)")
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("class_weights must be finite and positive")
    return nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))


def _training_classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.CrossEntropyLoss,
    *,
    focal_gamma: float,
) -> torch.Tensor:
    if focal_gamma < 0:
        raise ValueError("focal_gamma must be non-negative")
    if focal_gamma == 0:
        return criterion(logits, targets)
    cross_entropy = nn.functional.cross_entropy(
        logits,
        targets,
        weight=criterion.weight,
        reduction="none",
    )
    target_probability = (
        torch.softmax(logits, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
    )
    return (((1.0 - target_probability) ** focal_gamma) * cross_entropy).mean()


def fit_local_recurrent_with_validation(
    architecture: str,
    candidate: GRUSearchCandidate,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    batch_size: int,
    max_epochs: int,
    patience: int,
    class_weights: np.ndarray | None = None,
    focal_gamma: float = 0.0,
    validation_metric_labels: np.ndarray | None = None,
) -> LocalRecurrentFitResult:
    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs and patience must be positive")
    if focal_gamma < 0:
        raise ValueError("focal_gamma must be non-negative")
    set_reproducible_seed(seed, deterministic=True)
    output_size = int(max(np.max(y_train), np.max(y_validation))) + 1
    model = build_local_recurrent(
        architecture,
        candidate,
        X_train,
        output_size=output_size,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=candidate.learning_rate,
        weight_decay=candidate.weight_decay,
    )
    criterion = _cross_entropy_criterion(output_size, class_weights, device)
    loader = make_loader(
        X_train,
        y_train,
        batch_size=batch_size,
        seed=seed,
        shuffle=True,
        device=device,
    )
    labels = (
        np.arange(output_size)
        if validation_metric_labels is None
        else np.asarray(validation_metric_labels, dtype=np.int64)
    )
    if (
        labels.ndim != 1
        or not len(labels)
        or len(np.unique(labels)) != len(labels)
        or labels.min() < 0
        or labels.max() >= output_size
    ):
        raise ValueError("validation_metric_labels must be unique local class IDs")
    if not set(np.unique(y_validation)).issubset(set(labels.tolist())):
        raise ValueError("validation_metric_labels omit an observed validation class")
    best_state: dict[str, torch.Tensor] | None = None
    best_rank = (-np.inf, -np.inf, -np.inf)
    best_epoch = 0
    best_loss = np.inf
    stale = 0
    history: list[dict[str, float | int]] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_sum = 0.0
        samples = 0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = _training_classification_loss(
                model(X_batch),
                y_batch,
                criterion,
                focal_gamma=focal_gamma,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss in local recurrent training")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.item()) * len(y_batch)
            samples += len(y_batch)

        logits = predict_local_recurrent_logits(
            model, X_validation, batch_size=batch_size, device=device
        )
        prediction = logits.argmax(axis=1).astype(np.int64)
        macro = float(
            f1_score(
                y_validation,
                prediction,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        )
        accuracy = float(accuracy_score(y_validation, prediction))
        balanced = float(balanced_accuracy_score(y_validation, prediction))
        validation_loss = float(
            nn.functional.cross_entropy(
                torch.from_numpy(logits), torch.from_numpy(y_validation)
            ).item()
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / samples,
                "validation_loss": validation_loss,
                "validation_macro_f1": macro,
                "validation_accuracy": accuracy,
                "validation_balanced_accuracy": balanced,
            }
        )
        rank = (macro, accuracy, balanced)
        if rank > best_rank:
            best_rank = rank
            best_epoch = epoch
            best_loss = validation_loss
            best_state = deepcopy(
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
            )
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if best_state is None:
        raise RuntimeError("Local recurrent model produced no validation checkpoint")
    return LocalRecurrentFitResult(
        state_dict=best_state,
        best_epoch=best_epoch,
        epochs_completed=len(history),
        validation_macro_f1=float(best_rank[0]),
        validation_accuracy=float(best_rank[1]),
        validation_balanced_accuracy=float(best_rank[2]),
        best_validation_loss=float(best_loss),
        runtime_seconds=float(time.perf_counter() - started),
        trainable_parameters=model.trainable_parameters,
        history=history,
    )


def fit_local_recurrent_fixed_epochs(
    architecture: str,
    candidate: GRUSearchCandidate,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    output_size: int,
    device: torch.device,
    seed: int,
    batch_size: int,
    epochs: int,
    class_weights: np.ndarray | None = None,
    focal_gamma: float = 0.0,
) -> tuple[LocalRecurrentClassifier, list[dict[str, float | int]], float]:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if focal_gamma < 0:
        raise ValueError("focal_gamma must be non-negative")
    set_reproducible_seed(seed, deterministic=True)
    model = build_local_recurrent(
        architecture,
        candidate,
        X_train,
        output_size=output_size,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=candidate.learning_rate,
        weight_decay=candidate.weight_decay,
    )
    criterion = _cross_entropy_criterion(output_size, class_weights, device)
    loader = make_loader(
        X_train,
        y_train,
        batch_size=batch_size,
        seed=seed,
        shuffle=True,
        device=device,
    )
    history: list[dict[str, float | int]] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        samples = 0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = _training_classification_loss(
                model(X_batch),
                y_batch,
                criterion,
                focal_gamma=focal_gamma,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss in local recurrent refit")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.item()) * len(y_batch)
            samples += len(y_batch)
        history.append({"epoch": epoch, "train_loss": loss_sum / samples})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return model, history, float(time.perf_counter() - started)
