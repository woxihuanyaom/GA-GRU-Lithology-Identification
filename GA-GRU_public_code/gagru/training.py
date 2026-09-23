from __future__ import annotations

import os
import random
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .errors import DataValidationError
from .folds import PreparedSplit
from .metrics import ClassificationMetrics, classification_metrics
from .model import GRUClassifier, GRUModelConfig


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 200
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4
    gradient_clip_norm: float = 1.0
    num_workers: int = 0
    deterministic: bool = True

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer parameters")
        if self.batch_size < 1 or self.max_epochs < 1:
            raise ValueError("batch_size and max_epochs must be positive")
        if self.early_stopping_patience < 1 or self.early_stopping_min_delta < 0:
            raise ValueError("Invalid early-stopping parameters")
        if self.gradient_clip_norm <= 0 or self.num_workers < 0:
            raise ValueError("Invalid gradient clip or worker count")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    train_loss: float
    validation_loss: float
    validation_selection_score: float
    epoch_seconds: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class TrainingResult:
    model: GRUClassifier
    history: list[EpochRecord]
    best_epoch: int
    best_validation_loss: float
    best_metrics: ClassificationMetrics
    epochs_completed: int
    stopped_early: bool
    runtime_seconds: float
    seed: int
    device: str
    deterministic_algorithms: bool

    def summary_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "device": self.device,
            "deterministic_algorithms": self.deterministic_algorithms,
            "runtime_seconds": self.runtime_seconds,
            "epochs_completed": self.epochs_completed,
            "best_epoch": self.best_epoch,
            "stopped_early": self.stopped_early,
            "best_validation_loss": self.best_validation_loss,
            "best_metrics": self.best_metrics.to_dict(),
            "trainable_parameters": self.model.trainable_parameters,
        }


def set_reproducible_seed(seed: int, *, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic


def resolve_device(requested: str = "auto") -> torch.device:
    normalized = requested.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch build cannot use CUDA")
    if normalized not in {"cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    return torch.device(normalized)


def _loader(
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(X, dtype=np.float32)),
        torch.from_numpy(np.asarray(y, dtype=np.int64)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        generator=generator,
    )


def _evaluate(
    model: GRUClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            batch_size = len(y_batch)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            labels.append(y_batch.cpu().numpy())
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    return (
        total_loss / total_samples,
        np.concatenate(labels),
        np.concatenate(predictions),
    )


def train_gru(
    split: PreparedSplit,
    model_config: GRUModelConfig,
    training_config: TrainingConfig,
    *,
    seed: int,
    device: str = "auto",
    progress_callback: Callable[[EpochRecord], None] | None = None,
) -> TrainingResult:
    if split.train.X.shape[2] != model_config.input_size:
        raise DataValidationError("Model input size differs from the prepared features")
    if split.train.window_length != split.evaluation.window_length:
        raise DataValidationError("Training and evaluation window lengths differ")
    if set(split.train_wells).intersection(split.evaluation_wells):
        raise DataValidationError("Training/evaluation wells overlap")

    set_reproducible_seed(seed, deterministic=training_config.deterministic)
    target_device = resolve_device(device)
    pin_memory = target_device.type == "cuda"
    training_loader = _loader(
        split.train.X,
        split.train.y,
        batch_size=training_config.batch_size,
        shuffle=True,
        seed=seed,
        num_workers=training_config.num_workers,
        pin_memory=pin_memory,
    )
    validation_loader = _loader(
        split.evaluation.X,
        split.evaluation.y,
        batch_size=training_config.batch_size,
        shuffle=False,
        seed=seed,
        num_workers=training_config.num_workers,
        pin_memory=pin_memory,
    )

    model = GRUClassifier(model_config).to(target_device)
    class_weights = torch.as_tensor(
        split.class_weights, dtype=torch.float32, device=target_device
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    best_score = -np.inf
    best_epoch = 0
    best_loss = np.inf
    best_metrics: ClassificationMetrics | None = None
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[EpochRecord] = []
    validation_wells = split.evaluation.metadata["well_id"].astype(str).to_numpy()
    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)
        torch.cuda.synchronize(target_device)
    training_started = time.perf_counter()

    for epoch in range(1, training_config.max_epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        running_loss = 0.0
        samples_seen = 0
        for X_batch, y_batch in training_loader:
            X_batch = X_batch.to(target_device, non_blocking=True)
            y_batch = y_batch.to(target_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip_norm)
            optimizer.step()
            batch_count = len(y_batch)
            running_loss += float(loss.item()) * batch_count
            samples_seen += batch_count

        validation_loss, y_true, y_pred = _evaluate(
            model, validation_loader, criterion, target_device
        )
        metrics = classification_metrics(
            y_true,
            y_pred,
            validation_wells,
            num_classes=model_config.output_size,
        )
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        history.append(
            EpochRecord(
                epoch=epoch,
                train_loss=running_loss / samples_seen,
                validation_loss=validation_loss,
                validation_selection_score=metrics.selection_score,
                epoch_seconds=time.perf_counter() - epoch_started,
            )
        )
        if progress_callback is not None:
            progress_callback(history[-1])

        if metrics.selection_score > best_score + training_config.early_stopping_min_delta:
            best_score = metrics.selection_score
            best_epoch = epoch
            best_loss = validation_loss
            best_metrics = metrics
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= training_config.early_stopping_patience:
            break

    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    runtime_seconds = time.perf_counter() - training_started
    if best_state is None or best_metrics is None:
        raise RuntimeError("Training completed without a valid validation checkpoint")
    model.load_state_dict(best_state)
    return TrainingResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_validation_loss=float(best_loss),
        best_metrics=best_metrics,
        epochs_completed=len(history),
        stopped_early=len(history) < training_config.max_epochs,
        runtime_seconds=float(runtime_seconds),
        seed=seed,
        device=str(target_device),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
    )
