from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .search import GRUSearchCandidate
from .training import set_reproducible_seed


@dataclass
class ResidualFitResult:
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


class ResidualBiGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        window_length: int,
        candidate: GRUSearchCandidate,
        output_size: int = 10,
    ) -> None:
        super().__init__()
        hidden = candidate.hidden_size
        recurrent_dropout = candidate.dropout if candidate.num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=candidate.num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=True,
        )
        self.attention = nn.Sequential(
            nn.Linear(2 * hidden, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.window_branch = nn.Sequential(
            nn.Flatten(),
            nn.Linear(window_length * input_size, 2 * hidden),
            nn.LayerNorm(2 * hidden),
            nn.GELU(),
            nn.Dropout(candidate.dropout),
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(3 * hidden, 128),
            nn.GELU(),
            nn.Dropout(candidate.dropout),
            nn.Linear(128, output_size),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.gru(values)
        weights = torch.softmax(self.attention(sequence).squeeze(-1), dim=1)
        context = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        residual = self.window_branch(values)
        return self.classifier(torch.cat((context, residual), dim=1))


def physical_features(values: np.ndarray) -> np.ndarray:
    physical = values.astype(np.float64, copy=True)
    if physical.ndim != 3 or physical.shape[2] < 3:
        raise ValueError("Physical inputs must have [samples, sequence, >=3 features]")
    if np.any(physical[..., :3] <= 0):
        raise ValueError("Resistivity values must be positive before log10 transformation")
    physical[..., :3] = np.log10(physical[..., :3])
    return physical


def derived_features(values: np.ndarray) -> np.ndarray:
    physical = physical_features(values)
    separation = np.stack(
        (
            physical[..., 2] - physical[..., 1],
            physical[..., 1] - physical[..., 0],
            physical[..., 2] - physical[..., 0],
        ),
        axis=-1,
    )
    gradient = np.diff(physical, axis=1, prepend=physical[:, :1, :])
    return np.concatenate((physical, separation, gradient), axis=-1)


def center_conditioned_features(
    values: np.ndarray, *, include_contrast: bool = False
) -> np.ndarray:
    physical = physical_features(values)
    engineered = derived_features(values)
    center = physical[:, physical.shape[1] // 2 : physical.shape[1] // 2 + 1, :]
    repeated_center = np.broadcast_to(center, physical.shape)
    additions = [repeated_center]
    if include_contrast:
        additions.append(physical - repeated_center)
    return np.concatenate((engineered, *additions), axis=-1)


def center_contrast_features(values: np.ndarray) -> np.ndarray:
    physical = physical_features(values)
    center = physical[:, physical.shape[1] // 2 : physical.shape[1] // 2 + 1, :]
    return np.concatenate((derived_features(values), physical - center), axis=-1)


def prepare_per_well_inputs(
    fit: Any,
    *others: Any,
    representation: str = "engineered",
) -> tuple[np.ndarray, ...]:
    batches = (fit, *others)
    transformers = {
        "physical": physical_features,
        "engineered": derived_features,
        "center_repeated": center_conditioned_features,
        "center_contrast": center_contrast_features,
        "center_repeated_contrast": lambda values: center_conditioned_features(
            values, include_contrast=True
        ),
    }
    if representation not in transformers:
        raise ValueError(
            "representation must be physical, engineered, center_repeated, "
            "center_contrast, or center_repeated_contrast"
        )
    arrays = [transformers[representation](batch.X) for batch in batches]
    fit_wells = set(fit.wells.astype(str).tolist())
    for batch in others:
        unknown = set(batch.wells.astype(str).tolist()).difference(fit_wells)
        if unknown:
            raise ValueError(f"Per-well standardization has no fit rows for wells: {sorted(unknown)}")

    for well_id in dict.fromkeys(fit.wells.tolist()):
        fit_mask = fit.wells == well_id
        center = arrays[0][fit_mask].mean(axis=(0, 1), keepdims=True)
        scale = arrays[0][fit_mask].std(axis=(0, 1), keepdims=True)
        scale[scale == 0] = 1.0
        for values, batch in zip(arrays, batches, strict=True):
            mask = batch.wells == well_id
            values[mask] = (values[mask] - center) / scale
    return tuple(values.astype(np.float32) for values in arrays)


def combine_windows(first: Any, second: Any) -> Any:
    return type(first)(
        X=np.concatenate((first.X, second.X)),
        y=np.concatenate((first.y, second.y)),
        wells=np.concatenate((first.wells, second.wells)),
        groups=np.concatenate((first.groups, second.groups)),
        depths=np.concatenate((first.depths, second.depths)),
        center_ids=np.concatenate((first.center_ids, second.center_ids)),
        context_ids=first.context_ids + second.context_ids,
    )


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    seed: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y.astype(np.int64))),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )


def predict_logits(
    model: nn.Module,
    X: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    model.eval()
    dummy = np.zeros(len(X), dtype=np.int64)
    loader = make_loader(
        X, dummy, batch_size=batch_size, seed=0, shuffle=False, device=device
    )
    with torch.inference_mode():
        for X_batch, _ in loader:
            parts.append(model(X_batch.to(device, non_blocking=True)).cpu().numpy())
    return np.concatenate(parts)


def per_well_macro_f1(
    y_true: np.ndarray, y_pred: np.ndarray, wells: np.ndarray
) -> tuple[float, dict[str, float]]:
    scores: dict[str, float] = {}
    for well_id in dict.fromkeys(wells.tolist()):
        mask = wells == well_id
        labels = np.unique(y_true[mask])
        scores[str(well_id)] = float(
            f1_score(
                y_true[mask],
                y_pred[mask],
                labels=labels,
                average="macro",
                zero_division=0,
            )
        )
    return float(np.mean(list(scores.values()))), scores


def build_model(
    candidate: GRUSearchCandidate,
    X: np.ndarray,
    device: torch.device,
    *,
    output_size: int = 10,
) -> ResidualBiGRU:
    return ResidualBiGRU(
        input_size=X.shape[2],
        window_length=X.shape[1],
        candidate=candidate,
        output_size=output_size,
    ).to(device)


def fit_with_validation(
    candidate: GRUSearchCandidate,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    validation_wells: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    batch_size: int,
    max_epochs: int,
    patience: int,
) -> ResidualFitResult:
    set_reproducible_seed(seed, deterministic=True)
    output_size = int(max(np.max(y_train), np.max(y_validation))) + 1
    model = build_model(candidate, X_train, device, output_size=output_size)
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=candidate.learning_rate,
        weight_decay=candidate.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    train_loader = make_loader(
        X_train,
        y_train,
        batch_size=batch_size,
        seed=seed,
        shuffle=True,
        device=device,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_score = -np.inf
    best_accuracy = -np.inf
    best_balanced = -np.inf
    best_validation_loss = np.inf
    history: list[dict[str, float | int]] = []
    stale = 0
    started = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.item()) * len(y_batch)
            count += len(y_batch)

        validation_logits = predict_logits(
            model, X_validation, batch_size=batch_size, device=device
        )
        prediction = validation_logits.argmax(axis=1).astype(np.int64)
        score, _ = per_well_macro_f1(y_validation, prediction, validation_wells)
        accuracy = float(np.mean(y_validation == prediction))
        balanced = float(balanced_accuracy_score(y_validation, prediction))
        validation_loss = float(
            nn.functional.cross_entropy(
                torch.from_numpy(validation_logits), torch.from_numpy(y_validation)
            ).item()
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / max(1, count),
                "validation_loss": validation_loss,
                "validation_macro_f1": score,
                "validation_accuracy": accuracy,
                "validation_balanced_accuracy": balanced,
            }
        )
        rank = (score, accuracy, balanced)
        best_rank = (best_score, best_accuracy, best_balanced)
        if rank > best_rank:
            best_state = deepcopy(
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
            )
            best_epoch = epoch
            best_score = score
            best_accuracy = accuracy
            best_balanced = balanced
            best_validation_loss = validation_loss
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("Residual BiGRU did not produce a validation checkpoint")
    return ResidualFitResult(
        state_dict=best_state,
        best_epoch=best_epoch,
        epochs_completed=len(history),
        validation_macro_f1=float(best_score),
        validation_accuracy=float(best_accuracy),
        validation_balanced_accuracy=float(best_balanced),
        best_validation_loss=float(best_validation_loss),
        runtime_seconds=float(runtime),
        trainable_parameters=int(trainable_parameters),
        history=history,
    )


def fit_fixed_epochs(
    candidate: GRUSearchCandidate,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    batch_size: int,
    epochs: int,
) -> tuple[ResidualBiGRU, list[dict[str, float | int]], float]:
    set_reproducible_seed(seed, deterministic=True)
    output_size = int(np.max(y_train)) + 1
    model = build_model(candidate, X_train, device, output_size=output_size)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=candidate.learning_rate,
        weight_decay=candidate.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    train_loader = make_loader(
        X_train,
        y_train,
        batch_size=batch_size,
        seed=seed,
        shuffle=True,
        device=device,
    )
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for epoch in range(1, max(1, epochs) + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.item()) * len(y_batch)
            count += len(y_batch)
        history.append({"epoch": epoch, "train_loss": loss_sum / max(1, count)})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return model, history, float(time.perf_counter() - started)
