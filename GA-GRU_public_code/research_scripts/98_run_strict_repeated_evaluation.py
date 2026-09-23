"""Run the frozen strict complete-interval evaluation with checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


PROJECT_DIR = Path(__file__).resolve().parent
PROTOCOL_DIR = PROJECT_DIR / "experiment_protocol_v7_strict_interval"
PROTOCOL_PATH = PROTOCOL_DIR / "strict_interval_protocol_v7.json"
OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "independent_wells_v7_strict_interval"
    / "final_evaluation"
)
PLAN_PATH = OUTPUT_DIR / "strict_evaluation_plan.json"
RUNS_DIR = OUTPUT_DIR / "runs"
COMPLETION_PATH = OUTPUT_DIR / "execution_completion.json"

sys.path.insert(0, str(PROJECT_DIR))

from gagru.local_recurrent import (  # noqa: E402
    fit_local_recurrent_fixed_epochs,
    fit_local_recurrent_with_validation,
    predict_local_recurrent_logits,
)
from gagru.random_center import (  # noqa: E402
    combine_random_center_windows,
    impute_from_training_windows,
    remap_labels,
)
from gagru.random_center_ablation import apply_imbalance_strategy  # noqa: E402
from gagru.residual_bigru import prepare_per_well_inputs  # noqa: E402
from gagru.search import GRUSearchCandidate  # noqa: E402
from gagru.strict_interval import build_strict_partition_windows  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def partition_path(split_seed: int, well_id: str, split: str) -> Path:
    return (
        PROTOCOL_DIR
        / "partitions"
        / f"seed_{split_seed}"
        / well_id
        / f"{well_id}_{split}.csv"
    )


def load_partition(
    protocol: dict[str, Any], well_id: str, split_seed: int, split: str
) -> pd.DataFrame:
    path = partition_path(split_seed, well_id, split)
    expected = protocol["well_protocols"][well_id]["split_seeds"][str(split_seed)][
        "partition_sha256"
    ][split]
    if sha256_file(path) != expected:
        raise RuntimeError(f"Frozen strict partition hash mismatch: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if set(frame["well_id"].astype(str)) != {well_id}:
        raise RuntimeError(f"Strict partition contains another well: {path}")
    if set(frame["split"].astype(str)) != {split}:
        raise RuntimeError(f"Strict partition has an invalid split marker: {path}")
    return frame


def classification_report(
    y_true: np.ndarray,
    y_prediction: np.ndarray,
    global_classes: tuple[int, ...],
    family_map: np.ndarray,
) -> dict[str, Any]:
    labels = np.arange(len(global_classes), dtype=np.int64)
    supported = np.unique(y_true).astype(np.int64)
    precision, recall, class_f1, support = precision_recall_fscore_support(
        y_true,
        y_prediction,
        labels=labels,
        zero_division=0,
    )
    lookup = np.asarray(global_classes, dtype=np.int64)
    true_global = lookup[y_true]
    predicted_global = lookup[y_prediction]
    true_family = family_map[true_global]
    predicted_family = family_map[predicted_global]
    observed_families = np.unique(true_family).astype(np.int64)
    per_class = [
        {
            "local_class_id": int(local_id),
            "global_class_id": int(global_classes[local_id]),
            "precision": float(precision[local_id]),
            "recall": float(recall[local_id]),
            "f1": float(class_f1[local_id]),
            "support": int(support[local_id]),
        }
        for local_id in labels
    ]
    return {
        "accuracy": float(accuracy_score(y_true, y_prediction)),
        "supported_macro_f1": float(
            f1_score(
                y_true,
                y_prediction,
                labels=supported,
                average="macro",
                zero_division=0,
            )
        ),
        "fixed_local_macro_f1": float(
            f1_score(
                y_true,
                y_prediction,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_prediction)),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_prediction,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "family_accuracy": float(accuracy_score(true_family, predicted_family)),
        "family_macro_f1": float(
            f1_score(
                true_family,
                predicted_family,
                labels=observed_families,
                average="macro",
                zero_division=0,
            )
        ),
        "family_balanced_accuracy": float(
            balanced_accuracy_score(true_family, predicted_family)
        ),
        "supported_local_class_ids": supported.astype(int).tolist(),
        "supported_global_class_ids": lookup[supported].astype(int).tolist(),
        "observed_family_ids": observed_families.astype(int).tolist(),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_prediction, labels=labels)
        .astype(int)
        .tolist(),
        "confusion_matrix_global_class_order": [int(value) for value in global_classes],
    }


def run_directory(
    split_seed: int, well_id: str, model_id: str, training_seed: int
) -> Path:
    return (
        RUNS_DIR / f"split_{split_seed}" / well_id / model_id / f"seed_{training_seed}"
    )


def validate_existing_result(
    result: dict[str, Any],
    *,
    model_id: str,
    well_id: str,
    split_seed: int,
    training_seed: int,
) -> None:
    expected = {
        "model_id": model_id,
        "well_id": well_id,
        "split_seed": split_seed,
        "training_seed": training_seed,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise RuntimeError(f"Saved strict result conflicts for {key}")
    if result.get("status") != "COMPLETE":
        raise RuntimeError("Saved strict result is incomplete")
    for file_key, hash_key in (
        ("predictions_file", "predictions_sha256"),
        ("selection_history_file", "selection_history_sha256"),
        ("refit_history_file", "refit_history_sha256"),
    ):
        relative = result.get(file_key)
        expected_hash = result.get(hash_key)
        if relative is None:
            if expected_hash is not None:
                raise RuntimeError(f"Saved strict result has orphaned {hash_key}")
            continue
        path = PROJECT_DIR / str(relative)
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"Saved strict artifact hash mismatch: {path}")


def prepare_task(
    protocol: dict[str, Any],
    plan: dict[str, Any],
    well_id: str,
    split_seed: int,
    balance_seed: int,
) -> dict[str, Any]:
    record = protocol["well_protocols"][well_id]
    global_classes = tuple(int(value) for value in record["included_classes"])
    features = tuple(str(value) for value in plan["features"])
    window_length = int(plan["window_length"])
    frames = {
        split: load_partition(protocol, well_id, split_seed, split)
        for split in ("train", "validation", "test")
    }
    raw = {
        split: build_strict_partition_windows(frame, features, window_length)
        for split, frame in frames.items()
    }
    expected_counts = record["split_seeds"][str(split_seed)]["windows"]
    actual_counts = {split: int(len(raw[split].y)) for split in raw}
    if actual_counts != {key: int(value) for key, value in expected_counts.items()}:
        raise RuntimeError(f"Strict window counts differ for {well_id}/{split_seed}")

    selection_windows, selection_medians = impute_from_training_windows(
        {"train": raw["train"], "validation": raw["validation"]}
    )
    selection_y_train, mapping = remap_labels(
        selection_windows["train"], global_classes
    )
    selection_y_validation, _ = remap_labels(
        selection_windows["validation"], global_classes
    )
    selection_X_train, selection_X_validation = prepare_per_well_inputs(
        selection_windows["train"],
        selection_windows["validation"],
        representation=str(plan["representation"]),
    )
    selection_balanced = apply_imbalance_strategy(
        selection_X_train,
        selection_y_train,
        str(plan["imbalance_strategy"]),
        random_state=balance_seed,
    )

    refit_raw = combine_random_center_windows(raw["train"], raw["validation"])
    refit_windows, refit_medians = impute_from_training_windows(
        {"train": refit_raw, "test": raw["test"]}
    )
    refit_y, refit_mapping = remap_labels(refit_windows["train"], global_classes)
    test_y, _ = remap_labels(refit_windows["test"], global_classes)
    if mapping != refit_mapping:
        raise RuntimeError("Strict local class mapping changed during refit")
    refit_X, test_X = prepare_per_well_inputs(
        refit_windows["train"],
        refit_windows["test"],
        representation=str(plan["representation"]),
    )
    refit_balanced = apply_imbalance_strategy(
        refit_X,
        refit_y,
        str(plan["imbalance_strategy"]),
        random_state=balance_seed,
    )
    test_interval_lookup = (
        frames["test"].set_index("source_row_id")["lithology_interval_id"].astype(str)
    )
    return {
        "global_classes": global_classes,
        "mapping": mapping,
        "selection_X_train": selection_balanced.X,
        "selection_y_train": selection_balanced.y,
        "selection_class_weights": selection_balanced.class_weights,
        "selection_X_validation": selection_X_validation,
        "selection_y_validation": selection_y_validation,
        "validation_metric_labels": np.unique(selection_y_validation).astype(np.int64),
        "refit_X": refit_balanced.X,
        "refit_y": refit_balanced.y,
        "refit_class_weights": refit_balanced.class_weights,
        "test_X": test_X,
        "test_y": test_y,
        "test_windows": refit_windows["test"],
        "test_interval_ids": test_interval_lookup.loc[
            refit_windows["test"].center_ids
        ].to_numpy(str),
        "audit": {
            "window_counts": actual_counts,
            "selection_training_medians": selection_medians,
            "refit_training_validation_medians": refit_medians,
            "selection_balance": selection_balanced.audit,
            "refit_balance": refit_balanced.audit,
            "balance_seed": balance_seed,
            "validation_supported_global_classes": [
                int(global_classes[value])
                for value in np.unique(selection_y_validation)
            ],
            "test_supported_global_classes": [
                int(global_classes[value]) for value in np.unique(test_y)
            ],
            "global_to_local_class_mapping": {
                str(key): int(value) for key, value in mapping.items()
            },
            "strict_test_used_during_epoch_selection": False,
        },
    }


def save_predictions(
    path: Path,
    task: dict[str, Any],
    prediction: np.ndarray,
    family_map: np.ndarray,
    *,
    model_id: str,
    well_id: str,
    split_seed: int,
    training_seed: int,
) -> None:
    lookup = np.asarray(task["global_classes"], dtype=np.int64)
    true_global = lookup[task["test_y"]]
    predicted_global = lookup[prediction]
    test_windows = task["test_windows"]
    frame = pd.DataFrame(
        {
            "model_id": model_id,
            "well_id": well_id,
            "split_seed": split_seed,
            "training_seed": training_seed,
            "depth": test_windows.depths,
            "center_row_id": test_windows.center_ids,
            "lithology_interval_id": task["test_interval_ids"],
            "true_local_class_id": task["test_y"],
            "predicted_local_class_id": prediction,
            "true_global_class_id": true_global,
            "predicted_global_class_id": predicted_global,
            "true_family_id": family_map[true_global],
            "predicted_family_id": family_map[predicted_global],
        }
    )
    write_csv_atomic(path, frame)


def recurrent_run(
    model_spec: dict[str, Any],
    task: dict[str, Any],
    *,
    device: torch.device,
    training_seed: int,
    batch_size: int,
    maximum_epochs: int,
    patience: int,
    directory: Path,
) -> dict[str, Any]:
    candidate = GRUSearchCandidate.from_dict(model_spec["candidate"])
    architecture = str(model_spec["architecture"])
    selection = fit_local_recurrent_with_validation(
        architecture,
        candidate,
        task["selection_X_train"],
        task["selection_y_train"],
        task["selection_X_validation"],
        task["selection_y_validation"],
        device=device,
        seed=training_seed,
        batch_size=batch_size,
        max_epochs=maximum_epochs,
        patience=patience,
        class_weights=task["selection_class_weights"],
        validation_metric_labels=task["validation_metric_labels"],
    )
    selection_history_path = directory / "selection_history.csv"
    write_csv_atomic(selection_history_path, pd.DataFrame(selection.history))

    model, refit_history, refit_runtime = fit_local_recurrent_fixed_epochs(
        architecture,
        candidate,
        task["refit_X"],
        task["refit_y"],
        output_size=len(task["global_classes"]),
        device=device,
        seed=training_seed,
        batch_size=batch_size,
        epochs=selection.best_epoch,
        class_weights=task["refit_class_weights"],
    )
    refit_history_path = directory / "refit_history.csv"
    write_csv_atomic(refit_history_path, pd.DataFrame(refit_history))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_started = time.perf_counter()
    prediction = (
        predict_local_recurrent_logits(
            model,
            task["test_X"],
            batch_size=batch_size,
            device=device,
        )
        .argmax(axis=1)
        .astype(np.int64)
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_runtime = time.perf_counter() - inference_started
    result = {
        "prediction": prediction,
        "selected_epoch": selection.best_epoch,
        "selection_epochs_completed": selection.epochs_completed,
        "selection_validation_supported_macro_f1": selection.validation_macro_f1,
        "selection_validation_accuracy": selection.validation_accuracy,
        "selection_validation_balanced_accuracy": (
            selection.validation_balanced_accuracy
        ),
        "selection_runtime_seconds": selection.runtime_seconds,
        "refit_runtime_seconds": refit_runtime,
        "inference_runtime_seconds": inference_runtime,
        "trainable_parameters": model.trainable_parameters,
        "tree_nodes": None,
        "selection_history_file": str(selection_history_path.relative_to(PROJECT_DIR)),
        "selection_history_sha256": sha256_file(selection_history_path),
        "refit_history_file": str(refit_history_path.relative_to(PROJECT_DIR)),
        "refit_history_sha256": sha256_file(refit_history_path),
    }
    del model
    torch.cuda.empty_cache()
    return result


def extra_trees_run(
    model_spec: dict[str, Any],
    task: dict[str, Any],
    *,
    training_seed: int,
) -> dict[str, Any]:
    parameters = dict(model_spec["parameters"])
    parameters["random_state"] = training_seed
    model = ExtraTreesClassifier(**parameters)
    started = time.perf_counter()
    model.fit(task["refit_X"].reshape(len(task["refit_X"]), -1), task["refit_y"])
    refit_runtime = time.perf_counter() - started
    inference_started = time.perf_counter()
    prediction = model.predict(task["test_X"].reshape(len(task["test_X"]), -1)).astype(
        np.int64
    )
    inference_runtime = time.perf_counter() - inference_started
    tree_nodes = int(sum(estimator.tree_.node_count for estimator in model.estimators_))
    return {
        "prediction": prediction,
        "selected_epoch": None,
        "selection_epochs_completed": None,
        "selection_validation_supported_macro_f1": None,
        "selection_validation_accuracy": None,
        "selection_validation_balanced_accuracy": None,
        "selection_runtime_seconds": 0.0,
        "refit_runtime_seconds": refit_runtime,
        "inference_runtime_seconds": inference_runtime,
        "trainable_parameters": None,
        "tree_nodes": tree_nodes,
        "selection_history_file": None,
        "selection_history_sha256": None,
        "refit_history_file": None,
        "refit_history_sha256": None,
    }


def main() -> None:
    if not PLAN_PATH.is_file():
        raise RuntimeError("Run 97_freeze_strict_evaluation_plan.py first")
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    if plan["status"] != "FROZEN_BEFORE_ANY_STRICT_TEST_PARTITION_IS_OPENED":
        raise RuntimeError("Strict evaluation plan has an invalid status")
    if sha256_file(Path(__file__)) != plan["source_hashes"]["runner_code"]:
        raise RuntimeError("Strict runner changed after the plan was frozen")
    if sha256_file(PROTOCOL_PATH) != plan["source_hashes"]["strict_protocol"]:
        raise RuntimeError("Strict protocol changed after the plan was frozen")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Strict repeated recurrent evaluation requires CUDA")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    family_map = np.asarray(
        [
            int(protocol["class_reporting"]["family_map_by_global_class"][str(value)])
            for value in range(10)
        ],
        dtype=np.int64,
    )
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    wells = tuple(str(value) for value in plan["wells"])
    models = tuple(plan["models"])
    split_seeds = tuple(int(value) for value in plan["split_seeds"])
    training_seeds = tuple(int(value) for value in plan["training_seeds"])

    expected_runs = int(plan["expected_runs"])
    completed = 0
    started = time.perf_counter()
    result_paths: list[Path] = []
    for split_index, split_seed in enumerate(split_seeds):
        for well_index, well_id in enumerate(wells):
            balance_seed = 37101 + 100 * split_index + well_index
            print(
                f"Preparing strict split={split_seed}, well={well_id}, "
                f"balance_seed={balance_seed}",
                flush=True,
            )
            task = prepare_task(protocol, plan, well_id, split_seed, balance_seed)
            for model_spec in models:
                model_id = str(model_spec["model_id"])
                for training_seed in training_seeds:
                    directory = run_directory(
                        split_seed, well_id, model_id, training_seed
                    )
                    result_path = directory / "result.json"
                    result_paths.append(result_path)
                    if result_path.is_file():
                        saved = json.loads(result_path.read_text(encoding="utf-8"))
                        validate_existing_result(
                            saved,
                            model_id=model_id,
                            well_id=well_id,
                            split_seed=split_seed,
                            training_seed=training_seed,
                        )
                        completed += 1
                        print(
                            f"Replayed {completed}/{expected_runs}: "
                            f"{split_seed}/{well_id}/{model_id}/{training_seed}, "
                            f"accuracy={saved['metrics']['accuracy']:.4f}",
                            flush=True,
                        )
                        continue

                    directory.mkdir(parents=True, exist_ok=True)
                    if model_spec["family"] == "recurrent_neural_network":
                        run = recurrent_run(
                            model_spec,
                            task,
                            device=device,
                            training_seed=training_seed,
                            batch_size=int(plan["batch_size"]),
                            maximum_epochs=int(plan["maximum_selection_epochs"]),
                            patience=int(plan["selection_patience"]),
                            directory=directory,
                        )
                    elif model_spec["family"] == "classical_machine_learning":
                        run = extra_trees_run(
                            model_spec, task, training_seed=training_seed
                        )
                    else:
                        raise RuntimeError(
                            f"Unknown strict model family: {model_spec['family']}"
                        )

                    prediction = np.asarray(run.pop("prediction"), dtype=np.int64)
                    report = classification_report(
                        task["test_y"],
                        prediction,
                        task["global_classes"],
                        family_map,
                    )
                    predictions_path = directory / "test_predictions.csv"
                    save_predictions(
                        predictions_path,
                        task,
                        prediction,
                        family_map,
                        model_id=model_id,
                        well_id=well_id,
                        split_seed=split_seed,
                        training_seed=training_seed,
                    )
                    result = {
                        "status": "COMPLETE",
                        "model_id": model_id,
                        "display_name": model_spec["display_name"],
                        "model_family": model_spec["family"],
                        "architecture": model_spec["architecture"],
                        "model_specification": model_spec,
                        "well_id": well_id,
                        "split_seed": split_seed,
                        "training_seed": training_seed,
                        "global_classes": list(task["global_classes"]),
                        "training_samples_after_balance": int(
                            len(task["selection_y_train"])
                        ),
                        "validation_samples": int(len(task["selection_y_validation"])),
                        "refit_samples_after_balance": int(len(task["refit_y"])),
                        "test_samples": int(len(task["test_y"])),
                        "metrics": report,
                        **run,
                        "predictions_file": str(
                            predictions_path.relative_to(PROJECT_DIR)
                        ),
                        "predictions_sha256": sha256_file(predictions_path),
                        "data_preparation_audit": task["audit"],
                        "strict_test_metrics_used_for_any_selection": False,
                    }
                    write_json_atomic(result_path, result)
                    completed += 1
                    print(
                        f"Completed {completed}/{expected_runs}: "
                        f"{split_seed}/{well_id}/{model_id}/{training_seed}, "
                        f"accuracy={report['accuracy']:.4f}, "
                        f"supported macro-F1={report['supported_macro_f1']:.4f}, "
                        f"family accuracy={report['family_accuracy']:.4f}",
                        flush=True,
                    )
            del task
            torch.cuda.empty_cache()

    if completed != expected_runs or len(result_paths) != expected_runs:
        raise RuntimeError("Strict evaluation run count is incomplete")
    for path in result_paths:
        if not path.is_file():
            raise RuntimeError(f"Missing strict result: {path}")
    completion = {
        "status": "COMPLETE_PENDING_ANALYSIS",
        "runs": completed,
        "expected_runs": expected_runs,
        "strict_test_partitions_opened_only_after_plan_freeze": True,
        "strict_test_metrics_used_for_any_selection": False,
        "models": [str(model["model_id"]) for model in models],
        "wells": list(wells),
        "split_seeds": list(split_seeds),
        "training_seeds": list(training_seeds),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "wall_runtime_seconds_this_invocation": time.perf_counter() - started,
        "plan_sha256": sha256_file(PLAN_PATH),
        "result_files": [str(path.relative_to(PROJECT_DIR)) for path in result_paths],
    }
    write_json_atomic(COMPLETION_PATH, completion)
    print("Strict repeated evaluation: COMPLETE PENDING ANALYSIS")
    print(f"Runs: {completed}")
    print(f"Completion: {COMPLETION_PATH}")


if __name__ == "__main__":
    main()
