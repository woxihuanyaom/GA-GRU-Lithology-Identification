from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .errors import DataValidationError
from .model import GRUClassifier, GRUModelConfig
from .preprocessing import CurvePreprocessor
from .residual_bigru import ResidualBiGRU
from .search import GRUSearchCandidate
from .training import set_reproducible_seed
from .windows import WindowedDataset, build_windows


@dataclass(frozen=True)
class PreparedWithinWellTask:
    well_id: str
    feature_names: tuple[str, ...]
    window_length: int
    global_classes: tuple[int, ...]
    train: WindowedDataset
    validation: WindowedDataset
    y_train_local: np.ndarray
    y_validation_local: np.ndarray
    imputation_medians: dict[str, float]
    class_weights: np.ndarray


@dataclass
class FixedGRUValidationResult:
    best_epoch: int
    epochs_completed: int
    validation_macro_f1: float
    validation_accuracy: float
    validation_balanced_accuracy: float
    runtime_seconds: float
    trainable_parameters: int
    prediction_local: np.ndarray
    history: list[dict[str, float | int]]


def augment_standardized_windows(X: np.ndarray) -> np.ndarray:
    """Add resistivity separations and first differences to standardized curves."""

    values = np.asarray(X, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] < 3:
        raise DataValidationError("Augmented windows require at least three curves")
    separation = np.stack(
        (
            values[..., 2] - values[..., 1],
            values[..., 1] - values[..., 0],
            values[..., 2] - values[..., 0],
        ),
        axis=-1,
    )
    gradient = np.diff(values, axis=1, prepend=values[:, :1, :])
    return np.concatenate((values, separation, gradient), axis=-1).astype(
        np.float32, copy=False
    )


def _read_partition(protocol_dir: Path, well_id: str, split: str) -> pd.DataFrame:
    path = protocol_dir / "data" / well_id / f"{well_id}_{split}.csv"
    if not path.is_file():
        raise DataValidationError(f"Frozen partition does not exist: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if set(frame["well_id"].astype(str)) != {well_id}:
        raise DataValidationError(f"Partition contains another well: {well_id}/{split}")
    if set(frame["split"].astype(str)) != {split}:
        raise DataValidationError(f"Partition split marker changed: {well_id}/{split}")
    return frame


def prepare_within_well_task(
    protocol_dir: Path,
    well_id: str,
    features: Sequence[str],
    window_length: int,
    global_classes: Sequence[int],
    *,
    require_validation_all_classes: bool = True,
) -> PreparedWithinWellTask:
    """Load training/validation only and fit all preprocessing on training data."""

    feature_names = tuple(features)
    classes = tuple(int(value) for value in global_classes)
    train_frame = _read_partition(protocol_dir, well_id, "train")
    validation_frame = _read_partition(protocol_dir, well_id, "validation")
    return prepare_within_well_frames(
        train_frame,
        validation_frame,
        well_id=well_id,
        features=feature_names,
        window_length=window_length,
        global_classes=classes,
        require_validation_all_classes=require_validation_all_classes,
    )


def prepare_within_well_frames(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    *,
    well_id: str,
    features: Sequence[str],
    window_length: int,
    global_classes: Sequence[int],
    require_validation_all_classes: bool = True,
) -> PreparedWithinWellTask:
    """Prepare one task from already isolated train and validation frames."""

    feature_names = tuple(features)
    classes = tuple(int(value) for value in global_classes)
    train_frame = train_frame.copy()
    validation_frame = validation_frame.copy()
    overlap = set(train_frame["source_row_id"].astype(int)).intersection(
        validation_frame["source_row_id"].astype(int)
    )
    if overlap:
        raise DataValidationError(f"{well_id} train/validation source rows overlap")

    medians: dict[str, float] = {}
    for feature in feature_names:
        train_frame[feature] = pd.to_numeric(train_frame[feature], errors="coerce")
        validation_frame[feature] = pd.to_numeric(
            validation_frame[feature], errors="coerce"
        )
        median = float(train_frame[feature].median())
        if not np.isfinite(median):
            raise DataValidationError(f"{well_id} has no training median for {feature}")
        medians[feature] = median
        train_frame[feature] = train_frame[feature].fillna(median)
        validation_frame[feature] = validation_frame[feature].fillna(median)

    train_raw = build_windows(train_frame, feature_names, window_length)
    validation_raw = build_windows(validation_frame, feature_names, window_length)
    if not len(train_raw.y) or not len(validation_raw.y):
        raise DataValidationError(
            f"{well_id} produced an empty train/validation window set"
        )
    if not set(classes).issubset(set(int(value) for value in np.unique(train_raw.y))):
        raise DataValidationError(f"{well_id} training windows lost a frozen class")
    if require_validation_all_classes and not set(classes).issubset(
        set(int(value) for value in np.unique(validation_raw.y))
    ):
        raise DataValidationError(f"{well_id} validation windows lost a frozen class")

    resistivity = tuple(
        name for name in ("MSFL", "LLS", "LLD") if name in feature_names
    )
    preprocessor = CurvePreprocessor.fit(
        train_raw.X,
        feature_names,
        resistivity,
        training_wells=[well_id],
    )
    train = train_raw.with_features(preprocessor.transform(train_raw.X))
    validation = validation_raw.with_features(preprocessor.transform(validation_raw.X))

    local = {global_id: local_id for local_id, global_id in enumerate(classes)}
    try:
        y_train_local = np.asarray(
            [local[int(value)] for value in train.y], dtype=np.int64
        )
        y_validation_local = np.asarray(
            [local[int(value)] for value in validation.y], dtype=np.int64
        )
    except KeyError as exc:
        raise DataValidationError(
            f"{well_id} partition contains a class outside the frozen local task"
        ) from exc
    counts = np.bincount(y_train_local, minlength=len(classes)).astype(float)
    if np.any(counts == 0):
        raise DataValidationError(f"{well_id} local training labels are incomplete")
    weights = 1.0 / counts
    weights /= weights.mean()
    return PreparedWithinWellTask(
        well_id=well_id,
        feature_names=feature_names,
        window_length=window_length,
        global_classes=classes,
        train=train,
        validation=validation,
        y_train_local=y_train_local,
        y_validation_local=y_validation_local,
        imputation_medians=medians,
        class_weights=weights.astype(np.float32),
    )


def _loader(
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
        TensorDataset(
            torch.from_numpy(np.asarray(X, dtype=np.float32)),
            torch.from_numpy(np.asarray(y, dtype=np.int64)),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )


def _predict(
    model: nn.Module,
    X: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = _loader(
        X,
        np.zeros(len(X), dtype=np.int64),
        batch_size=batch_size,
        seed=0,
        shuffle=False,
        device=device,
    )
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for X_batch, _ in loader:
            logits = model(X_batch.to(device, non_blocking=True))
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions).astype(np.int64)


def fit_fixed_gru_validation(
    task: PreparedWithinWellTask,
    *,
    seed: int,
    device: torch.device,
    hidden_size: int = 64,
    num_layers: int = 2,
    learning_rate: float = 1e-3,
    dropout: float = 0.2,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    max_epochs: int = 80,
    patience: int = 10,
    class_weighted: bool = True,
) -> FixedGRUValidationResult:
    set_reproducible_seed(seed, deterministic=True)
    config = GRUModelConfig(
        input_size=len(task.feature_names),
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=len(task.global_classes),
        dropout=dropout,
    )
    model = GRUClassifier(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), learning_rate, weight_decay=weight_decay
    )
    weights = (
        torch.as_tensor(task.class_weights, dtype=torch.float32, device=device)
        if class_weighted
        else None
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    train_loader = _loader(
        task.train.X,
        task.y_train_local,
        batch_size=batch_size,
        seed=seed,
        shuffle=True,
        device=device,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_rank = (-np.inf, -np.inf, -np.inf)
    best_epoch = 0
    stale = 0
    history: list[dict[str, float | int]] = []
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

        prediction = _predict(
            model, task.validation.X, batch_size=batch_size, device=device
        )
        labels = list(range(len(task.global_classes)))
        macro_f1 = float(
            f1_score(
                task.y_validation_local,
                prediction,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        )
        accuracy = float(accuracy_score(task.y_validation_local, prediction))
        balanced = float(balanced_accuracy_score(task.y_validation_local, prediction))
        rank = (macro_f1, accuracy, balanced)
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / max(1, count),
                "validation_macro_f1": macro_f1,
                "validation_accuracy": accuracy,
                "validation_balanced_accuracy": balanced,
            }
        )
        if rank > best_rank:
            best_rank = rank
            best_epoch = epoch
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
    runtime = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("Fixed GRU did not create a validation checkpoint")
    model.load_state_dict(best_state)
    prediction = _predict(
        model, task.validation.X, batch_size=batch_size, device=device
    )
    return FixedGRUValidationResult(
        best_epoch=best_epoch,
        epochs_completed=len(history),
        validation_macro_f1=float(best_rank[0]),
        validation_accuracy=float(best_rank[1]),
        validation_balanced_accuracy=float(best_rank[2]),
        runtime_seconds=float(runtime),
        trainable_parameters=model.trainable_parameters,
        prediction_local=prediction,
        history=history,
    )


def fit_residual_bigru_validation(
    task: PreparedWithinWellTask,
    *,
    seed: int,
    device: torch.device,
    candidate: GRUSearchCandidate,
    batch_size: int = 256,
    max_epochs: int = 80,
    patience: int = 10,
    feature_representation: str = "augmented",
    weight_mode: str = "none",
) -> FixedGRUValidationResult:
    set_reproducible_seed(seed, deterministic=True)
    if feature_representation == "augmented":
        X_train = augment_standardized_windows(task.train.X)
        X_validation = augment_standardized_windows(task.validation.X)
    elif feature_representation == "raw":
        X_train = task.train.X
        X_validation = task.validation.X
    else:
        raise ValueError("feature_representation must be 'raw' or 'augmented'")

    model = ResidualBiGRU(
        input_size=X_train.shape[2],
        window_length=task.window_length,
        candidate=candidate,
        output_size=len(task.global_classes),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=candidate.learning_rate,
        weight_decay=candidate.weight_decay,
    )
    if weight_mode == "none":
        weights = None
    elif weight_mode == "inverse":
        weights = torch.as_tensor(
            task.class_weights, dtype=torch.float32, device=device
        )
    elif weight_mode == "sqrt_inverse":
        sqrt_weights = np.sqrt(task.class_weights.astype(np.float64))
        sqrt_weights /= sqrt_weights.mean()
        weights = torch.as_tensor(sqrt_weights, dtype=torch.float32, device=device)
    else:
        raise ValueError("weight_mode must be none, sqrt_inverse, or inverse")
    criterion = nn.CrossEntropyLoss(weight=weights)
    train_loader = _loader(
        X_train,
        task.y_train_local,
        batch_size=batch_size,
        seed=seed,
        shuffle=True,
        device=device,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_rank = (-np.inf, -np.inf, -np.inf)
    best_epoch = 0
    stale = 0
    history: list[dict[str, float | int]] = []
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

        prediction = _predict(model, X_validation, batch_size=batch_size, device=device)
        labels = list(range(len(task.global_classes)))
        macro_f1 = float(
            f1_score(
                task.y_validation_local,
                prediction,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        )
        accuracy = float(accuracy_score(task.y_validation_local, prediction))
        balanced = float(balanced_accuracy_score(task.y_validation_local, prediction))
        rank = (macro_f1, accuracy, balanced)
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / max(1, count),
                "validation_macro_f1": macro_f1,
                "validation_accuracy": accuracy,
                "validation_balanced_accuracy": balanced,
            }
        )
        if rank > best_rank:
            best_rank = rank
            best_epoch = epoch
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
    runtime = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("Residual BiGRU did not create a validation checkpoint")
    model.load_state_dict(best_state)
    prediction = _predict(model, X_validation, batch_size=batch_size, device=device)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return FixedGRUValidationResult(
        best_epoch=best_epoch,
        epochs_completed=len(history),
        validation_macro_f1=float(best_rank[0]),
        validation_accuracy=float(best_rank[1]),
        validation_balanced_accuracy=float(best_rank[2]),
        runtime_seconds=float(runtime),
        trainable_parameters=int(trainable_parameters),
        prediction_local=prediction,
        history=history,
    )


def local_to_global(prediction_local: np.ndarray, classes: Sequence[int]) -> np.ndarray:
    lookup = np.asarray(tuple(classes), dtype=np.int64)
    values = np.asarray(prediction_local, dtype=np.int64)
    if values.size and (values.min() < 0 or values.max() >= len(lookup)):
        raise DataValidationError("Local prediction falls outside the class lookup")
    return lookup[values]


def validation_metrics(
    y_true_local: np.ndarray, prediction_local: np.ndarray, classes: Sequence[int]
) -> dict[str, Any]:
    labels = list(range(len(tuple(classes))))
    return {
        "samples": int(len(y_true_local)),
        "correct": int(np.sum(y_true_local == prediction_local)),
        "accuracy": float(accuracy_score(y_true_local, prediction_local)),
        "macro_f1": float(
            f1_score(
                y_true_local,
                prediction_local,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true_local, prediction_local)
        ),
    }
