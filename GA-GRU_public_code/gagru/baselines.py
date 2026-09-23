from __future__ import annotations

import gc
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .errors import DataValidationError
from .folds import PreparedSplit
from .metrics import ClassificationMetrics, classification_metrics
from .training import EpochRecord, TrainingConfig, resolve_device, set_reproducible_seed
from .windows import WindowedDataset


NEURAL_BASELINES = ("mlp", "vanilla_rnn", "lstm")
CLASSICAL_BASELINES = ("random_forest", "xgboost")
BASELINE_MODELS = (*CLASSICAL_BASELINES, *NEURAL_BASELINES)


@dataclass(frozen=True)
class BaselineModelConfig:
    architecture: str
    input_size: int = 5
    window_length: int = 9
    output_size: int = 10
    hidden_size: int | None = None
    num_layers: int | None = None
    hidden_layers: tuple[int, ...] = ()
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.architecture not in NEURAL_BASELINES:
            raise ValueError(f"Unknown neural baseline: {self.architecture}")
        if self.input_size < 1 or self.window_length < 1 or self.output_size < 2:
            raise ValueError("Baseline model dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.architecture == "mlp":
            if not self.hidden_layers or any(value < 1 for value in self.hidden_layers):
                raise ValueError("MLP hidden layers must be non-empty and positive")
            if self.hidden_size is not None or self.num_layers is not None:
                raise ValueError("MLP must not define recurrent dimensions")
        else:
            if self.hidden_size is None or self.hidden_size < 1:
                raise ValueError("A recurrent baseline needs a positive hidden_size")
            if self.num_layers is None or self.num_layers < 1:
                raise ValueError("A recurrent baseline needs a positive num_layers")
            if self.hidden_layers:
                raise ValueError("A recurrent baseline must not define MLP hidden layers")

    @property
    def effective_recurrent_dropout(self) -> float:
        if self.architecture == "mlp" or self.num_layers == 1:
            return 0.0
        return self.dropout

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["hidden_layers"] = list(self.hidden_layers)
        values["effective_recurrent_dropout"] = self.effective_recurrent_dropout
        return values


class BaselineClassifier(nn.Module):
    """MLP or unidirectional many-to-one recurrent baseline."""

    def __init__(self, config: BaselineModelConfig) -> None:
        super().__init__()
        self.config = config
        if config.architecture == "mlp":
            modules: list[nn.Module] = []
            width = config.window_length * config.input_size
            for hidden in config.hidden_layers:
                modules.extend((nn.Linear(width, hidden), nn.ReLU()))
                if config.dropout > 0:
                    modules.append(nn.Dropout(config.dropout))
                width = hidden
            modules.append(nn.Linear(width, config.output_size))
            self.network = nn.Sequential(*modules)
            self.recurrent = None
            self.classifier = None
        else:
            recurrent_class = nn.RNN if config.architecture == "vanilla_rnn" else nn.LSTM
            recurrent_kwargs: dict[str, Any] = {
                "input_size": config.input_size,
                "hidden_size": int(config.hidden_size),
                "num_layers": int(config.num_layers),
                "batch_first": True,
                "dropout": config.effective_recurrent_dropout,
                "bidirectional": False,
            }
            if config.architecture == "vanilla_rnn":
                recurrent_kwargs["nonlinearity"] = "tanh"
            self.recurrent = recurrent_class(**recurrent_kwargs)
            self.classifier = nn.Linear(int(config.hidden_size), config.output_size)
            self.network = None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        expected = (self.config.window_length, self.config.input_size)
        if X.ndim != 3 or tuple(X.shape[1:]) != expected:
            raise ValueError(
                f"Expected [batch, {expected[0]}, {expected[1]}], got {tuple(X.shape)}"
            )
        if self.config.architecture == "mlp":
            assert self.network is not None
            return self.network(torch.flatten(X, start_dim=1))
        assert self.recurrent is not None and self.classifier is not None
        output, _ = self.recurrent(X)
        return self.classifier(output[:, -1, :])

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


@dataclass
class BaselineTrainingResult:
    model: BaselineClassifier
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
    peak_gpu_memory_bytes: int | None

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
            "peak_gpu_memory_bytes": self.peak_gpu_memory_bytes,
            "trainable_parameters": self.model.trainable_parameters,
        }


@dataclass
class FixedBaselineResult:
    model: BaselineClassifier
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


def _loader(
    X: np.ndarray,
    y: np.ndarray | None,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    tensors = [torch.from_numpy(np.asarray(X, dtype=np.float32))]
    if y is not None:
        tensors.append(torch.from_numpy(np.asarray(y, dtype=np.int64)))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        generator=generator,
    )


def _evaluate_neural(
    model: BaselineClassifier,
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
            count = len(y_batch)
            total_loss += float(loss.item()) * count
            total_samples += count
            labels.append(y_batch.cpu().numpy())
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    return total_loss / total_samples, np.concatenate(labels), np.concatenate(predictions)


def _validate_neural_inputs(
    split: PreparedSplit,
    model_config: BaselineModelConfig,
) -> None:
    if split.train.X.shape[2] != model_config.input_size:
        raise DataValidationError("Model input size differs from the prepared features")
    if split.train.window_length != model_config.window_length:
        raise DataValidationError("Model window length differs from the prepared windows")
    if split.train.window_length != split.evaluation.window_length:
        raise DataValidationError("Training and evaluation window lengths differ")
    if set(split.train_wells).intersection(split.evaluation_wells):
        raise DataValidationError("Training/evaluation wells overlap")


def train_neural_baseline(
    split: PreparedSplit,
    model_config: BaselineModelConfig,
    training_config: TrainingConfig,
    *,
    seed: int,
    device: str = "auto",
    progress_callback: Callable[[EpochRecord], None] | None = None,
) -> BaselineTrainingResult:
    _validate_neural_inputs(split, model_config)
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
    model = BaselineClassifier(model_config).to(target_device)
    weights = torch.as_tensor(split.class_weights, dtype=torch.float32, device=target_device)
    criterion = nn.CrossEntropyLoss(weight=weights)
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
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(target_device)
        torch.cuda.synchronize(target_device)
    started = time.perf_counter()

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
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss in neural baseline training")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip_norm)
            optimizer.step()
            count = len(y_batch)
            running_loss += float(loss.item()) * count
            samples_seen += count

        validation_loss, y_true, y_pred = _evaluate_neural(
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
        record = EpochRecord(
            epoch=epoch,
            train_loss=running_loss / samples_seen,
            validation_loss=float(validation_loss),
            validation_selection_score=metrics.selection_score,
            epoch_seconds=time.perf_counter() - epoch_started,
        )
        history.append(record)
        if progress_callback is not None:
            progress_callback(record)
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
    runtime = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated(target_device)) if target_device.type == "cuda" else None
    if best_state is None or best_metrics is None:
        raise RuntimeError("Neural baseline training produced no valid checkpoint")
    model.load_state_dict(best_state)
    return BaselineTrainingResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_validation_loss=float(best_loss),
        best_metrics=best_metrics,
        epochs_completed=len(history),
        stopped_early=len(history) < training_config.max_epochs,
        runtime_seconds=float(runtime),
        seed=seed,
        device=str(target_device),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        peak_gpu_memory_bytes=peak,
    )


def train_neural_fixed_epochs(
    dataset: WindowedDataset,
    class_weights: np.ndarray,
    model_config: BaselineModelConfig,
    training_config: TrainingConfig,
    *,
    epochs: int,
    seed: int,
    device: str = "cuda",
    progress_callback: Callable[[int, float], None] | None = None,
) -> FixedBaselineResult:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if dataset.X.shape[2] != model_config.input_size:
        raise DataValidationError("Model input size differs from prepared features")
    if dataset.window_length != model_config.window_length:
        raise DataValidationError("Model window length differs from prepared windows")
    set_reproducible_seed(seed, deterministic=training_config.deterministic)
    target_device = resolve_device(device)
    loader = _loader(
        dataset.X,
        dataset.y,
        batch_size=training_config.batch_size,
        shuffle=True,
        seed=seed,
        num_workers=training_config.num_workers,
        pin_memory=target_device.type == "cuda",
    )
    model = BaselineClassifier(model_config).to(target_device)
    weights = torch.as_tensor(class_weights, dtype=torch.float32, device=target_device)
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
                raise RuntimeError("Non-finite loss in fixed-epoch baseline refit")
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
    return FixedBaselineResult(
        model=model,
        train_loss=tuple(losses),
        epochs_completed=epochs,
        runtime_seconds=float(runtime),
        seed=seed,
        device=str(target_device),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        peak_gpu_memory_bytes=peak,
    )


def predict_neural_baseline(
    model: BaselineClassifier,
    dataset: WindowedDataset,
    *,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_device = resolve_device(device)
    loader = _loader(
        dataset.X,
        None,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
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


def center_features(dataset: WindowedDataset) -> np.ndarray:
    if dataset.window_length % 2 == 0:
        raise DataValidationError("A center-point baseline needs an odd window length")
    return np.ascontiguousarray(dataset.X[:, dataset.window_length // 2, :], dtype=np.float32)


def flattened_features(dataset: WindowedDataset) -> np.ndarray:
    return np.ascontiguousarray(dataset.X.reshape(len(dataset.y), -1), dtype=np.float32)


def class_sample_weights(y: np.ndarray, class_weights: np.ndarray) -> np.ndarray:
    labels = np.asarray(y, dtype=np.int64)
    weights = np.asarray(class_weights, dtype=np.float64)
    if labels.ndim != 1 or weights.ndim != 1 or labels.min() < 0 or labels.max() >= len(weights):
        raise DataValidationError("Invalid labels or class weights")
    return weights[labels]


def make_random_forest(candidate: dict[str, Any], *, seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=int(candidate["n_estimators"]),
        max_depth=(None if candidate["max_depth"] is None else int(candidate["max_depth"])),
        min_samples_leaf=int(candidate["min_samples_leaf"]),
        max_features=candidate["max_features"],
        class_weight=candidate["class_weight"],
        random_state=seed,
        n_jobs=-1,
    )


def make_xgboost(
    candidate: dict[str, Any],
    *,
    seed: int,
    n_estimators: int = 1000,
    early_stopping_rounds: int | None = None,
    device: str = "cuda",
):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("XGBoost is required for the frozen baseline experiment") from exc
    return XGBClassifier(
        objective="multi:softprob",
        num_class=10,
        n_estimators=int(n_estimators),
        max_depth=int(candidate["max_depth"]),
        learning_rate=float(candidate["learning_rate"]),
        subsample=float(candidate["subsample"]),
        colsample_bytree=float(candidate["colsample_bytree"]),
        min_child_weight=float(candidate["min_child_weight"]),
        reg_lambda=float(candidate["reg_lambda"]),
        eval_metric="mlogloss",
        early_stopping_rounds=early_stopping_rounds,
        tree_method="hist",
        device=device,
        random_state=seed,
        n_jobs=8,
        verbosity=0,
    )


def predict_classical(model: Any, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    predicted = np.asarray(model.predict(X), dtype=np.int64)
    probabilities = np.asarray(model.predict_proba(X), dtype=np.float64)
    if probabilities.ndim != 2 or len(probabilities) != len(predicted):
        raise RuntimeError("Classical baseline returned invalid probabilities")
    return predicted, probabilities.max(axis=1)


def random_forest_complexity(model: RandomForestClassifier) -> int:
    return int(sum(estimator.tree_.node_count for estimator in model.estimators_))


def xgboost_rounds(model: Any) -> int:
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is not None:
        return int(best_iteration) + 1
    return int(model.get_booster().num_boosted_rounds())
