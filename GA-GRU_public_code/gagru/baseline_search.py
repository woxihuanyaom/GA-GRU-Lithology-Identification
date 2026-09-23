from __future__ import annotations

import csv
import gc
import importlib.metadata
import json
import os
import platform
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .baselines import (
    BASELINE_MODELS,
    CLASSICAL_BASELINES,
    NEURAL_BASELINES,
    BaselineModelConfig,
    center_features,
    class_sample_weights,
    make_random_forest,
    make_xgboost,
    predict_classical,
    random_forest_complexity,
    train_neural_baseline,
    xgboost_rounds,
)
from .data import file_sha256
from .errors import SearchValidationError
from .folds import PreparedSplit
from .metrics import classification_metrics
from .pilot import protocol_training_configs
from .protocol import FrozenProtocol
from .search import ActiveSearchBudget
from .training import TrainingConfig, resolve_device


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding=encoding)
    os.replace(temporary, path)


def _write_json_with_hash(path: Path, value: Any) -> None:
    _atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    _atomic_write_text(path.with_suffix(".sha256"), file_sha256(path) + "\n", encoding="ascii")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def baseline_search_space(model_name: str) -> dict[str, Any]:
    if model_name == "random_forest":
        return {
            "n_estimators": {"type": "int", "min": 300, "max": 1000},
            "max_depth": {"type": "categorical", "values": [None, 8, 12, 20]},
            "min_samples_leaf": {"type": "categorical", "values": [1, 2, 5, 10]},
            "max_features": {"type": "categorical", "values": ["sqrt", 0.6, 1.0]},
            "class_weight": {"type": "categorical", "values": [None, "balanced"]},
        }
    if model_name == "xgboost":
        return {
            "max_depth": {"type": "int", "min": 3, "max": 10},
            "learning_rate": {"type": "log_uniform", "min": 1e-3, "max": 0.2},
            "subsample": {"type": "uniform", "min": 0.6, "max": 1.0},
            "colsample_bytree": {"type": "uniform", "min": 0.6, "max": 1.0},
            "min_child_weight": {"type": "int", "min": 1, "max": 10},
            "reg_lambda": {"type": "log_uniform", "min": 1e-3, "max": 10.0},
            "maximum_estimators": 1000,
            "early_stopping_rounds": 20,
        }
    if model_name == "mlp":
        return {
            "hidden_layers": {
                "type": "categorical",
                "values": [[64], [128], [128, 64], [256, 128]],
            },
            "dropout": {"type": "categorical", "values": [0.0, 0.2, 0.4]},
            "learning_rate": {"type": "log_uniform", "min": 1e-4, "max": 5e-3},
            "weight_decay": {"type": "log_uniform", "min": 1e-6, "max": 1e-3},
        }
    if model_name in {"vanilla_rnn", "lstm"}:
        return {
            "hidden_size": {"type": "int_multiple", "min": 16, "max": 192, "step": 8},
            "num_layers": {"type": "categorical", "values": [1, 2, 3]},
            "learning_rate": {"type": "log_uniform", "min": 1e-4, "max": 5e-3},
            "dropout": {
                "type": "categorical",
                "values": [0.0, 0.1, 0.2, 0.3, 0.4],
                "force_zero_when_num_layers_is_one": True,
            },
            "weight_decay": {"type": "log_uniform", "min": 1e-6, "max": 1e-3},
        }
    raise SearchValidationError(f"Unknown baseline model: {model_name}")


def validate_baseline_candidate(model_name: str, candidate: dict[str, Any]) -> None:
    space = baseline_search_space(model_name)
    if model_name == "random_forest":
        if not 300 <= int(candidate["n_estimators"]) <= 1000:
            raise SearchValidationError("RF n_estimators is outside the frozen range")
        if candidate["max_depth"] not in (None, 8, 12, 20):
            raise SearchValidationError("RF max_depth is outside the frozen choices")
        if int(candidate["min_samples_leaf"]) not in (1, 2, 5, 10):
            raise SearchValidationError("RF min_samples_leaf is outside the frozen choices")
        if candidate["max_features"] not in ("sqrt", 0.6, 1.0):
            raise SearchValidationError("RF max_features is outside the frozen choices")
        if candidate["class_weight"] not in (None, "balanced"):
            raise SearchValidationError("RF class_weight is outside the frozen choices")
    elif model_name == "xgboost":
        for key in ("max_depth", "min_child_weight"):
            limits = space[key]
            if not int(limits["min"]) <= int(candidate[key]) <= int(limits["max"]):
                raise SearchValidationError(f"XGBoost {key} is outside the frozen range")
        for key in ("learning_rate", "subsample", "colsample_bytree", "reg_lambda"):
            limits = space[key]
            if not float(limits["min"]) <= float(candidate[key]) <= float(limits["max"]):
                raise SearchValidationError(f"XGBoost {key} is outside the frozen range")
    elif model_name == "mlp":
        hidden = list(candidate["hidden_layers"])
        if hidden not in space["hidden_layers"]["values"]:
            raise SearchValidationError("MLP hidden layers are outside the frozen choices")
        if float(candidate["dropout"]) not in (0.0, 0.2, 0.4):
            raise SearchValidationError("MLP dropout is outside the frozen choices")
        for key in ("learning_rate", "weight_decay"):
            limits = space[key]
            if not float(limits["min"]) <= float(candidate[key]) <= float(limits["max"]):
                raise SearchValidationError(f"MLP {key} is outside the frozen range")
    elif model_name in {"vanilla_rnn", "lstm"}:
        hidden = int(candidate["hidden_size"])
        if hidden < 16 or hidden > 192 or hidden % 8:
            raise SearchValidationError("Recurrent hidden_size is outside the frozen range")
        layers = int(candidate["num_layers"])
        if layers not in (1, 2, 3):
            raise SearchValidationError("Recurrent num_layers is outside the frozen choices")
        dropout = float(candidate["dropout"])
        if dropout not in (0.0, 0.1, 0.2, 0.3, 0.4) or (layers == 1 and dropout != 0.0):
            raise SearchValidationError("Recurrent dropout violates the frozen choices")
        for key in ("learning_rate", "weight_decay"):
            limits = space[key]
            if not float(limits["min"]) <= float(candidate[key]) <= float(limits["max"]):
                raise SearchValidationError(f"Recurrent {key} is outside the frozen range")
    else:
        raise SearchValidationError(f"Unknown baseline model: {model_name}")


def _suggest_candidate(trial: Any, model_name: str) -> dict[str, Any]:
    if model_name == "random_forest":
        depth = trial.suggest_categorical("max_depth_choice", ["none", "8", "12", "20"])
        features = trial.suggest_categorical("max_features_choice", ["sqrt", "0.6", "1.0"])
        weight = trial.suggest_categorical("class_weight_choice", ["none", "balanced"])
        candidate = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
            "max_depth": None if depth == "none" else int(depth),
            "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [1, 2, 5, 10]),
            "max_features": "sqrt" if features == "sqrt" else float(features),
            "class_weight": None if weight == "none" else weight,
        }
    elif model_name == "xgboost":
        candidate = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    elif model_name == "mlp":
        layout = trial.suggest_categorical("hidden_layers_choice", ["64", "128", "128-64", "256-128"])
        candidate = {
            "hidden_layers": [int(value) for value in layout.split("-")],
            "dropout": trial.suggest_categorical("dropout", [0.0, 0.2, 0.4]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        }
    elif model_name in {"vanilla_rnn", "lstm"}:
        layers = trial.suggest_categorical("num_layers", [1, 2, 3])
        dropout = 0.0 if layers == 1 else trial.suggest_categorical(
            "dropout", [0.0, 0.1, 0.2, 0.3, 0.4]
        )
        candidate = {
            "hidden_size": trial.suggest_int("hidden_size", 16, 192, step=8),
            "num_layers": int(layers),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
            "dropout": float(dropout),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        }
    else:
        raise SearchValidationError(f"Unknown baseline model: {model_name}")
    validate_baseline_candidate(model_name, candidate)
    return candidate


def neural_configs(
    protocol: FrozenProtocol,
    model_name: str,
    candidate: dict[str, Any],
) -> tuple[BaselineModelConfig, TrainingConfig]:
    if model_name not in NEURAL_BASELINES:
        raise ValueError(f"Not a neural baseline: {model_name}")
    validate_baseline_candidate(model_name, candidate)
    _, base_training = protocol_training_configs()
    if model_name == "mlp":
        model = BaselineModelConfig(
            architecture="mlp",
            input_size=len(protocol.feature_names),
            window_length=9,
            output_size=len(protocol.class_names),
            hidden_layers=tuple(int(value) for value in candidate["hidden_layers"]),
            dropout=float(candidate["dropout"]),
        )
    else:
        model = BaselineModelConfig(
            architecture=model_name,
            input_size=len(protocol.feature_names),
            window_length=9,
            output_size=len(protocol.class_names),
            hidden_size=int(candidate["hidden_size"]),
            num_layers=int(candidate["num_layers"]),
            dropout=float(candidate["dropout"]),
        )
    training = replace(
        base_training,
        learning_rate=float(candidate["learning_rate"]),
        weight_decay=float(candidate["weight_decay"]),
    )
    return model, training


def _history(path: Path, records: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].to_dict()))
        writer.writeheader()
        writer.writerows(record.to_dict() for record in records)
    os.replace(temporary, path)


class PersistentBaselineEvaluator:
    def __init__(
        self,
        *,
        model_name: str,
        split: PreparedSplit,
        protocol: FrozenProtocol,
        training_seed: int,
        device: str,
        output_dir: Path,
    ) -> None:
        if model_name not in BASELINE_MODELS:
            raise SearchValidationError(f"Unknown baseline model: {model_name}")
        requested = set(split.train_wells).union(split.evaluation_wells)
        if requested.intersection(protocol.locked_external_wells):
            raise SearchValidationError("A locked external well entered a baseline search")
        self.model_name = model_name
        self.split = split
        self.protocol = protocol
        self.training_seed = training_seed
        self.device = device
        self.output_dir = output_dir
        self.log_path = output_dir / "candidate_evaluations.jsonl"
        self.history_dir = output_dir / "histories"
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.existing = self._load_existing()
        self.replay_position = 0

    def _load_existing(self) -> list[dict[str, Any]]:
        if not self.log_path.is_file():
            return []
        values: list[dict[str, Any]] = []
        seen: set[str] = set()
        with self.log_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SearchValidationError(f"Invalid baseline log line {line_number}") from exc
                if int(value.get("candidate_index", -1)) != len(values) + 1:
                    raise SearchValidationError("Baseline candidate indices are not contiguous")
                if value.get("model") != self.model_name:
                    raise SearchValidationError("Baseline candidate log contains another model")
                candidate = dict(value["candidate"])
                validate_baseline_candidate(self.model_name, candidate)
                key = _canonical_json(candidate)
                if value.get("candidate_key") != key or key in seen:
                    raise SearchValidationError("Baseline candidate key is invalid or duplicated")
                if int(value.get("training_seed", -1)) != self.training_seed:
                    raise SearchValidationError("Baseline candidate uses another training seed")
                for metric in ("selection_score", "balanced_accuracy", "runtime_seconds"):
                    if not np.isfinite(float(value[metric])):
                        raise SearchValidationError(f"Non-finite baseline value: {metric}")
                history_file = value.get("history_file")
                if self.model_name in NEURAL_BASELINES:
                    if not history_file or not (self.output_dir / history_file).is_file():
                        raise SearchValidationError("Neural baseline history is missing")
                seen.add(key)
                values.append(value)
        return values

    def evaluate(self, candidate: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
        validate_baseline_candidate(self.model_name, candidate)
        key = _canonical_json(candidate)
        if self.replay_position < len(self.existing):
            expected = self.existing[self.replay_position]
            if expected["candidate_key"] != key:
                raise SearchValidationError("Baseline TPE replay diverged from the saved log")
            self.replay_position += 1
            print(
                f"Replayed {self.model_name} candidate {expected['candidate_index']}: "
                f"score={expected['selection_score']:.6f}",
                flush=True,
            )
            return expected
        if any(value["candidate_key"] == key for value in self.existing):
            raise SearchValidationError("Baseline search requested a duplicate candidate")

        index = len(self.existing) + 1
        history_file: str | None = None
        selected_epochs: int | None = None
        selected_iterations: int | None = None
        peak_gpu_memory: int | None = None
        best_validation_loss: float | None = None
        if self.model_name in NEURAL_BASELINES:
            model_config, training_config = neural_configs(
                self.protocol, self.model_name, candidate
            )
            if self.device == "cuda":
                gc.collect()
                torch.cuda.empty_cache()
            result = train_neural_baseline(
                self.split,
                model_config,
                training_config,
                seed=self.training_seed,
                device=self.device,
            )
            metrics = result.best_metrics.to_dict()
            runtime = result.runtime_seconds
            complexity = result.model.trainable_parameters
            selected_epochs = result.best_epoch
            best_validation_loss = result.best_validation_loss
            peak_gpu_memory = result.peak_gpu_memory_bytes
            history_path = self.history_dir / f"candidate_{index:03d}.csv"
            _history(history_path, result.history)
            history_file = str(history_path.relative_to(self.output_dir))
            epochs_completed = result.epochs_completed
            model_config_value: dict[str, Any] | None = model_config.to_dict()
            training_config_value: dict[str, Any] | None = training_config.to_dict()
            del result
        else:
            X_train = center_features(self.split.train)
            X_validation = center_features(self.split.evaluation)
            started = time.perf_counter()
            if self.model_name == "random_forest":
                model = make_random_forest(candidate, seed=self.training_seed)
                model.fit(X_train, self.split.train.y)
                complexity = random_forest_complexity(model)
            else:
                model = make_xgboost(
                    candidate,
                    seed=self.training_seed,
                    n_estimators=1000,
                    early_stopping_rounds=20,
                    device=self.device,
                )
                model.fit(
                    X_train,
                    self.split.train.y,
                    sample_weight=class_sample_weights(
                        self.split.train.y, self.split.class_weights
                    ),
                    eval_set=[(X_validation, self.split.evaluation.y)],
                    verbose=False,
                )
                selected_iterations = xgboost_rounds(model)
                complexity = selected_iterations
            runtime = time.perf_counter() - started
            y_pred, _ = predict_classical(model, X_validation)
            metrics = classification_metrics(
                self.split.evaluation.y,
                y_pred,
                self.split.evaluation.metadata["well_id"].astype(str).to_numpy(),
                num_classes=len(self.protocol.class_names),
            ).to_dict()
            epochs_completed = None
            model_config_value = None
            training_config_value = None
            del model, X_train, X_validation

        value = {
            "candidate_index": index,
            "candidate_key": key,
            "candidate": candidate,
            "model": self.model_name,
            "training_seed": self.training_seed,
            "selection_score": float(metrics["selection_score"]),
            "balanced_accuracy": float(metrics["balanced_accuracy"]),
            "model_complexity": int(complexity),
            "runtime_seconds": float(runtime),
            "peak_gpu_memory_bytes": peak_gpu_memory,
            "epochs_completed": epochs_completed,
            "selected_epochs": selected_epochs,
            "selected_iterations": selected_iterations,
            "best_validation_loss": best_validation_loss,
            "metrics": metrics,
            "model_config": model_config_value,
            "training_config": training_config_value,
            "history_file": history_file,
            "provenance": provenance,
        }
        with self.log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(_canonical_json(value) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.existing.append(value)
        self.replay_position += 1
        print(
            f"Completed {self.model_name} candidate {index}: "
            f"score={value['selection_score']:.6f}, runtime={runtime:.2f} s",
            flush=True,
        )
        if self.device == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
        return value

    def assert_replay_complete(self) -> None:
        if self.replay_position != len(self.existing):
            raise SearchValidationError("Saved baseline candidates were not fully replayed")


def _evaluation_rank(value: dict[str, Any]) -> tuple[float, float, int, float]:
    return (
        float(value["selection_score"]),
        float(value["balanced_accuracy"]),
        -int(value["model_complexity"]),
        -float(value["runtime_seconds"]),
    )


def _run_tpe(
    model_name: str,
    *,
    budget: int,
    seed: int,
    warmup: int,
    evaluator: PersistentBaselineEvaluator,
) -> list[dict[str, Any]]:
    try:
        import optuna
    except ImportError as exc:
        raise SearchValidationError("Optuna is required for baseline TPE search") from exc
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=warmup),
    )
    seen: set[str] = set()
    evaluations: list[dict[str, Any]] = []
    duplicate_trials = 0
    while len(evaluations) < budget:
        trial = study.ask()
        candidate = _suggest_candidate(trial, model_name)
        key = _canonical_json(candidate)
        if key in seen:
            duplicate_trials += 1
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            if duplicate_trials > 1000:
                raise SearchValidationError("Baseline TPE repeatedly generated duplicate candidates")
            continue
        seen.add(key)
        evaluation = evaluator.evaluate(
            candidate,
            {
                "method": "tpe",
                "trial_number": int(trial.number),
                "unique_candidate": len(evaluations) + 1,
            },
        )
        study.tell(trial, float(evaluation["selection_score"]))
        evaluations.append(evaluation)
    evaluator.assert_replay_complete()
    return evaluations


def _code_hashes(project_dir: Path) -> dict[str, str]:
    names = (
        "baseline_search.py",
        "baselines.py",
        "data.py",
        "folds.py",
        "metrics.py",
        "preprocessing.py",
        "protocol.py",
        "training.py",
        "windows.py",
    )
    return {name: file_sha256(project_dir / "gagru" / name) for name in names}


def run_baseline_search(
    *,
    project_dir: str | Path,
    protocol: FrozenProtocol,
    split: PreparedSplit,
    model_name: str,
    output_dir: str | Path,
    budget: int,
    search_seed: int,
    training_seed: int,
    device: str,
    budget_source: ActiveSearchBudget,
) -> dict[str, Any]:
    if model_name not in BASELINE_MODELS:
        raise SearchValidationError(f"Unknown baseline model: {model_name}")
    expected_budget = min(budget_source.budget, 24) if model_name in CLASSICAL_BASELINES else budget_source.budget
    if budget != expected_budget:
        raise SearchValidationError(
            f"{model_name} requires {expected_budget} candidates under the frozen GPU budget"
        )
    resolved_device = "cpu" if model_name == "random_forest" else resolve_device(device).type
    if model_name != "random_forest" and resolved_device != "cuda":
        raise SearchValidationError("Formal neural/XGBoost baseline search requires CUDA")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    project = Path(project_dir).resolve()
    manifest = {
        "schema_version": "1.0",
        "run_kind": "formal_baseline_inner_search",
        "model": model_name,
        "search_method": "tpe",
        "protocol_id": protocol.raw["protocol"]["id"],
        "protocol_version": protocol.raw["protocol"]["version"],
        "dataset_aggregate_sha256": protocol.raw["dataset"]["aggregate_well_file_sha256"],
        "split_name": split.split_name,
        "training_wells": list(split.train_wells),
        "validation_wells": list(split.evaluation_wells),
        "training_windows": len(split.train.y),
        "validation_windows": len(split.evaluation.y),
        "input_representation": (
            "center_point_five_features"
            if model_name in CLASSICAL_BASELINES
            else "flattened_L9_five_features"
            if model_name == "mlp"
            else "ordered_L9_five_features"
        ),
        "features": list(split.train.feature_names),
        "window_length": split.train.window_length,
        "budget_unique_candidates": budget,
        "search_seed": search_seed,
        "candidate_training_seed": training_seed,
        "search_space": baseline_search_space(model_name),
        "selection_metric": protocol.raw["training"]["selection_metric"],
        "tie_breakers": list(protocol.raw["search"]["tie_breakers"]),
        "tpe_random_warmup_candidates": int(protocol.raw["search"]["tpe_random_warmup_candidates"]),
        "imbalance_handling": (
            "candidate_class_weight_none_or_balanced"
            if model_name == "random_forest"
            else "training_fold_inverse_frequency_sample_weight_mean_one"
            if model_name == "xgboost"
            else "training_fold_class_weighted_cross_entropy_mean_one"
        ),
        "xgboost_early_stopping_metric": "unweighted_validation_mlogloss" if model_name == "xgboost" else None,
        "device": resolved_device,
        "device_name": torch.cuda.get_device_name(0) if resolved_device == "cuda" else platform.processor(),
        "python_executable": os.path.realpath(os.sys.executable),
        "python_version": platform.python_version(),
        "packages": {
            "numpy": _package_version("numpy"),
            "scikit-learn": _package_version("scikit-learn"),
            "torch": _package_version("torch"),
            "xgboost": _package_version("xgboost"),
            "optuna": _package_version("optuna"),
        },
        "code_sha256": _code_hashes(project),
        "budget_source": {
            "runtime_pilot_report": str(budget_source.report_path),
            "runtime_pilot_sha256": budget_source.report_sha256,
            "median_runtime_T_seconds": budget_source.median_runtime_seconds,
            "candidate_budget_B": budget_source.budget,
        },
        "locked_external_wells_not_read": sorted(protocol.locked_external_wells),
    }
    manifest_path = output / "search_manifest.json"
    if manifest_path.is_file():
        sidecar = manifest_path.with_suffix(".sha256")
        if not sidecar.is_file() or file_sha256(manifest_path) != sidecar.read_text(encoding="ascii").strip():
            raise SearchValidationError("Existing baseline manifest failed hash verification")
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _canonical_json(existing_manifest) != _canonical_json(manifest):
            raise SearchValidationError(f"Existing baseline manifest conflicts: {manifest_path}")
    else:
        if any((output / name).exists() for name in ("candidate_evaluations.jsonl", "search_summary.json")):
            raise SearchValidationError("Baseline search artifacts exist without a manifest")
        _write_json_with_hash(manifest_path, manifest)
    manifest_hash = file_sha256(manifest_path)
    summary_path = output / "search_summary.json"
    if summary_path.is_file():
        sidecar = summary_path.with_suffix(".sha256")
        if not sidecar.is_file() or file_sha256(summary_path) != sidecar.read_text(encoding="ascii").strip():
            raise SearchValidationError("Existing baseline summary failed hash verification")
        completed = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            completed.get("status") != "COMPLETE"
            or completed.get("manifest_sha256") != manifest_hash
            or int(completed.get("unique_candidates_evaluated", -1)) != budget
        ):
            raise SearchValidationError("Existing baseline summary conflicts with its manifest")
        return completed

    evaluator = PersistentBaselineEvaluator(
        model_name=model_name,
        split=split,
        protocol=protocol,
        training_seed=training_seed,
        device=resolved_device,
        output_dir=output,
    )
    started = time.perf_counter()
    evaluations = _run_tpe(
        model_name,
        budget=budget,
        seed=search_seed,
        warmup=int(protocol.raw["search"]["tpe_random_warmup_candidates"]),
        evaluator=evaluator,
    )
    best = max(evaluations, key=_evaluation_rank)
    summary = {
        "status": "COMPLETE",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": model_name,
        "search_method": "tpe",
        "manifest_sha256": manifest_hash,
        "unique_candidates_evaluated": len(evaluations),
        "best_candidate_index": int(best["candidate_index"]),
        "best_candidate": dict(best["candidate"]),
        "best_selection_score": float(best["selection_score"]),
        "best_balanced_accuracy": float(best["balanced_accuracy"]),
        "best_model_complexity": int(best["model_complexity"]),
        "best_runtime_seconds": float(best["runtime_seconds"]),
        "best_selected_epochs": best["selected_epochs"],
        "best_selected_iterations": best["selected_iterations"],
        "search_training_runtime_seconds": float(
            sum(float(value["runtime_seconds"]) for value in evaluations)
        ),
        "wall_runtime_seconds_this_invocation": float(time.perf_counter() - started),
        "candidate_log": evaluator.log_path.name,
        "candidate_results": evaluations,
        "locked_external_wells_not_read": sorted(protocol.locked_external_wells),
    }
    _write_json_with_hash(summary_path, summary)
    return summary
