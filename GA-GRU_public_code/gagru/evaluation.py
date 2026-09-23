"""Leakage-safe fixed-epoch refitting and outer-well evaluation helpers.

The formal search code deliberately remains unchanged.  This module is used
after inner model selection: the selected epoch count is fixed from the inner
split, the model is refit on all outer-training wells, and only then is the
outer test well loaded for prediction.
"""

from __future__ import annotations

import csv
import gc
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import DataRepository, file_sha256
from .errors import DataValidationError
from .metrics import classification_metrics
from .model import GRUClassifier, GRUModelConfig
from .preprocessing import CurvePreprocessor, compute_class_weights
from .protocol import FrozenProtocol
from .training import TrainingConfig, resolve_device, set_reproducible_seed
from .windows import WindowedDataset, build_windows


@dataclass(frozen=True)
class TrainingOnlyData:
    dataset: WindowedDataset
    preprocessor: CurvePreprocessor
    class_weights: np.ndarray
    wells: tuple[str, ...]


@dataclass(frozen=True)
class FixedEpochResult:
    model: GRUClassifier
    train_loss: tuple[float, ...]
    epochs_completed: int
    runtime_seconds: float
    seed: int
    device: str
    deterministic_algorithms: bool
    peak_gpu_memory_bytes: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_loss": list(self.train_loss),
            "epochs_completed": self.epochs_completed,
            "runtime_seconds": self.runtime_seconds,
            "seed": self.seed,
            "device": self.device,
            "deterministic_algorithms": self.deterministic_algorithms,
            "peak_gpu_memory_bytes": self.peak_gpu_memory_bytes,
            "trainable_parameters": self.model.trainable_parameters,
        }


def sha256_file(path: str | Path) -> str:
    """Return a deterministic SHA-256 digest for an output artifact."""

    return file_sha256(Path(path))


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(text.encode(encoding))
    os.replace(temporary, destination)
    return destination


def atomic_write_json(path: str | Path, value: Any) -> Path:
    return atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2),
    )


def prepare_training_only(
    repository: DataRepository,
    well_ids: Iterable[str],
    *,
    window_length: int,
) -> TrainingOnlyData:
    """Build and standardize windows without loading any evaluation well."""

    wells = tuple(well_ids)
    if not wells or len(set(wells)) != len(wells):
        raise DataValidationError("Training wells must be non-empty and unique")
    repository.protocol.assert_access_allowed(wells)
    frame = repository.load_wells(wells)
    features = repository.protocol.feature_names
    raw = build_windows(frame, features, window_length)
    if len(raw.y) == 0:
        raise DataValidationError("Training-only preparation produced no windows")
    preprocessor = CurvePreprocessor.fit(
        raw.X,
        features,
        repository.protocol.resistivity_names,
        training_wells=wells,
        locked_external_wells=repository.protocol.locked_external_wells,
    )
    transformed = raw.with_features(preprocessor.transform(raw.X))
    weights = compute_class_weights(
        transformed.y,
        num_classes=len(repository.protocol.class_names),
    )
    return TrainingOnlyData(
        dataset=transformed,
        preprocessor=preprocessor,
        class_weights=weights,
        wells=wells,
    )


def prepare_evaluation_only(
    repository: DataRepository,
    well_ids: Iterable[str],
    *,
    window_length: int,
    preprocessor: CurvePreprocessor,
    protocol: FrozenProtocol,
) -> WindowedDataset:
    """Load and transform evaluation wells after the model has been trained."""

    wells = tuple(well_ids)
    if not wells or len(set(wells)) != len(wells):
        raise DataValidationError("Evaluation wells must be non-empty and unique")
    repository.protocol.assert_access_allowed(wells)
    frame = repository.load_wells(wells)
    raw = build_windows(frame, protocol.feature_names, window_length)
    if len(raw.y) == 0:
        raise DataValidationError("Evaluation preparation produced no windows")
    return raw.with_features(preprocessor.transform(raw.X))


def _training_loader(
    dataset: WindowedDataset,
    *,
    batch_size: int,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    tensor_dataset = TensorDataset(
        torch.from_numpy(np.asarray(dataset.X, dtype=np.float32)),
        torch.from_numpy(np.asarray(dataset.y, dtype=np.int64)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        tensor_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        generator=generator,
    )


def train_gru_fixed_epochs(
    training: TrainingOnlyData,
    model_config: GRUModelConfig,
    training_config: TrainingConfig,
    *,
    epochs: int,
    seed: int,
    device: str = "cuda",
    progress_callback: Callable[[int, float], None] | None = None,
) -> FixedEpochResult:
    """Fit a GRU for a preselected number of epochs without test feedback."""

    if epochs < 1:
        raise ValueError("epochs must be positive")
    if training.dataset.X.shape[2] != model_config.input_size:
        raise DataValidationError("Model input size differs from prepared features")

    set_reproducible_seed(seed, deterministic=training_config.deterministic)
    target_device = resolve_device(device)
    pin_memory = target_device.type == "cuda"
    loader = _training_loader(
        training.dataset,
        batch_size=training_config.batch_size,
        seed=seed,
        num_workers=training_config.num_workers,
        pin_memory=pin_memory,
    )
    model = GRUClassifier(model_config).to(target_device)
    weights = torch.as_tensor(
        training.class_weights,
        dtype=torch.float32,
        device=target_device,
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    if target_device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(target_device)
        torch.cuda.synchronize(target_device)
    started = time.perf_counter()
    losses: list[float] = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        samples_seen = 0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(target_device, non_blocking=True)
            y_batch = y_batch.to(target_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss in fixed-epoch refit")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip_norm)
            optimizer.step()
            count = len(y_batch)
            running_loss += float(loss.item()) * count
            samples_seen += count
        epoch_loss = running_loss / samples_seen
        losses.append(epoch_loss)
        if progress_callback is not None:
            progress_callback(epoch, epoch_loss)

    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    runtime = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated(target_device)) if target_device.type == "cuda" else None
    return FixedEpochResult(
        model=model,
        train_loss=tuple(losses),
        epochs_completed=epochs,
        runtime_seconds=float(runtime),
        seed=seed,
        device=str(target_device),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        peak_gpu_memory_bytes=peak,
    )


def predict_gru(
    model: GRUClassifier,
    dataset: WindowedDataset,
    *,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return true labels, predicted labels, and maximum class probability."""

    target_device = resolve_device(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(dataset.X.astype(np.float32, copy=False))),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=target_device.type == "cuda",
    )
    model = model.to(target_device)
    model.eval()
    predictions: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    with torch.inference_mode():
        for (X_batch,) in loader:
            logits = model(X_batch.to(target_device, non_blocking=True))
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted = probabilities.max(dim=1)
            predictions.append(predicted.cpu().numpy())
            confidences.append(confidence.cpu().numpy())
    return (
        dataset.y.astype(np.int64, copy=False),
        np.concatenate(predictions).astype(np.int64, copy=False),
        np.concatenate(confidences).astype(np.float64, copy=False),
    )


def detailed_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    well_ids: Sequence[str],
    *,
    num_classes: int,
) -> dict[str, Any]:
    """Serialize the protocol metrics plus fixed-class details and confusion matrices."""

    base = classification_metrics(
        y_true,
        y_pred,
        well_ids,
        num_classes=num_classes,
    ).to_dict()
    labels = list(range(num_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    base["per_class_metrics"] = [
        {
            "class_id": index,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index in labels
    ]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )
    base["confusion_matrix_counts"] = matrix.astype(int).tolist()
    base["confusion_matrix_true_normalized"] = normalized.tolist()
    return base


def save_model_state(model: GRUClassifier, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, destination)
    return destination


def save_predictions(
    dataset: WindowedDataset,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    *,
    path: str | Path,
) -> Path:
    if len(y_pred) != len(dataset.y) or len(confidence) != len(dataset.y):
        raise DataValidationError("Prediction length differs from evaluation windows")
    frame = dataset.metadata.copy()
    frame["true_class_id"] = dataset.y.astype(np.int64)
    frame["predicted_class_id"] = y_pred.astype(np.int64)
    frame["prediction_confidence"] = confidence.astype(float)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8-sig")
    return destination


def write_hash_sidecar(path: str | Path) -> Path:
    destination = Path(path)
    # Match the frozen project's convention: ``summary.json`` ->
    # ``summary.sha256``.
    sidecar = destination.with_suffix(".sha256")
    atomic_write_text(sidecar, sha256_file(destination) + "\n", encoding="ascii")
    return sidecar


def write_training_history(path: str | Path, losses: Sequence[float]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("epoch", "train_loss"))
        writer.writeheader()
        for epoch, loss in enumerate(losses, start=1):
            writer.writerow({"epoch": epoch, "train_loss": float(loss)})
    return destination


def preprocessor_dict(preprocessor: CurvePreprocessor) -> dict[str, Any]:
    return preprocessor.to_dict()


def protocol_code_fingerprint(project_dir: str | Path) -> dict[str, str]:
    """Hash evaluation-facing source files for the run manifest."""

    project = Path(project_dir)
    paths = (
        project / "gagru" / "evaluation.py",
        project / "gagru" / "metrics.py",
        project / "gagru" / "model.py",
        project / "gagru" / "preprocessing.py",
        project / "gagru" / "training.py",
        project / "gagru" / "windows.py",
    )
    return {path.name: sha256_file(path) for path in paths}
