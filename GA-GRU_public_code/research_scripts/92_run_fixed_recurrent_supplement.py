"""Run frozen fixed RNN/LSTM supplemental baselines with checkpoints."""

from __future__ import annotations

import hashlib
import importlib.util
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


PROJECT_DIR = Path(__file__).resolve().parent
PROTOCOL_DIR = PROJECT_DIR / "experiment_protocol_v5_random_center"
PROTOCOL_PATH = PROTOCOL_DIR / "random_center_protocol_v5.json"
OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "independent_wells_v5"
    / "fixed_recurrent_supplement"
)
PLAN_PATH = OUTPUT_DIR / "supplement_plan.json"
RUNS_DIR = OUTPUT_DIR / "runs"
COMPLETION_PATH = OUTPUT_DIR / "execution_completion.json"
PRIMARY_RUNNER_PATH = PROJECT_DIR / "80_run_final_repeated_evaluation.py"

sys.path.insert(0, str(PROJECT_DIR))


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


def load_primary_runner() -> Any:
    spec = importlib.util.spec_from_file_location(
        "frozen_primary_runner", PRIMARY_RUNNER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the frozen primary runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_directory(
    split_seed: int, well_id: str, model_id: str, training_seed: int
) -> Path:
    return (
        RUNS_DIR
        / f"split_{split_seed}"
        / well_id
        / model_id
        / f"seed_{training_seed}"
    )


def main() -> None:
    if not PLAN_PATH.is_file():
        raise RuntimeError("Run 91_freeze_fixed_recurrent_supplement.py first")
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    if plan["status"] != "FROZEN_BEFORE_SUPPLEMENTAL_MODEL_TEST_RUNS":
        raise RuntimeError("Supplemental plan has an invalid status")
    if sha256_file(Path(__file__)) != plan["source_hashes"][
        "supplement_runner_code"
    ]:
        raise RuntimeError("Supplemental runner changed after plan freeze")
    if sha256_file(PRIMARY_RUNNER_PATH) != plan["source_hashes"][
        "primary_runner_code"
    ]:
        raise RuntimeError("Frozen primary runner changed")
    if sha256_file(PROTOCOL_PATH) != plan["source_hashes"]["protocol"]:
        raise RuntimeError("Protocol changed after plan freeze")

    primary = load_primary_runner()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Supplemental recurrent evaluation requires CUDA")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    wells = tuple(str(value) for value in plan["wells"])
    models = tuple(plan["models"])
    split_seeds = tuple(int(value) for value in plan["split_seeds"])
    training_seeds = tuple(int(value) for value in plan["training_seeds"])

    frames = {}
    for well_id in wells:
        record = protocol["well_protocols"][well_id]
        path = PROTOCOL_DIR / str(record["source_snapshot"])
        if sha256_file(path) != record["source_snapshot_sha256"]:
            raise RuntimeError(f"Frozen source hash mismatch: {path}")
        frames[well_id] = pd.read_csv(path, encoding="utf-8-sig")

    expected_runs = int(plan["expected_runs"])
    completed = 0
    started = time.perf_counter()
    result_paths = []
    for split_index, split_seed in enumerate(split_seeds):
        for well_index, well_id in enumerate(wells):
            balance_seed = 27101 + 100 * split_index + well_index
            print(
                f"Preparing split={split_seed}, well={well_id}, "
                f"balance_seed={balance_seed}",
                flush=True,
            )
            task = primary.prepare_task(
                protocol,
                plan,
                frames[well_id],
                well_id,
                split_seed,
                balance_seed,
            )
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
                        primary.validate_existing_result(
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
                    run = primary.recurrent_run(
                        model_spec,
                        task,
                        device=device,
                        training_seed=training_seed,
                        batch_size=int(plan["batch_size"]),
                        maximum_epochs=int(plan["maximum_selection_epochs"]),
                        patience=int(plan["selection_patience"]),
                        directory=directory,
                    )
                    prediction = np.asarray(run.pop("prediction"), dtype=np.int64)
                    report = primary.classification_report(
                        task["test_y"], prediction, task["global_classes"]
                    )
                    predictions_path = directory / "test_predictions.csv"
                    primary.save_predictions(
                        predictions_path,
                        task,
                        prediction,
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
                        "validation_samples": int(
                            len(task["selection_y_validation"])
                        ),
                        "refit_samples_after_balance": int(len(task["refit_y"])),
                        "test_samples": int(len(task["test_y"])),
                        "metrics": report,
                        **run,
                        "predictions_file": str(
                            predictions_path.relative_to(PROJECT_DIR)
                        ),
                        "predictions_sha256": sha256_file(predictions_path),
                        "data_preparation_audit": task["audit"],
                        "primary_results_available_before_supplement": True,
                        "test_metrics_used_for_model_or_configuration_selection": False,
                    }
                    write_json_atomic(result_path, result)
                    completed += 1
                    print(
                        f"Completed {completed}/{expected_runs}: "
                        f"{split_seed}/{well_id}/{model_id}/{training_seed}, "
                        f"accuracy={report['accuracy']:.4f}, "
                        f"macro-F1={report['macro_f1']:.4f}",
                        flush=True,
                    )
            del task
            torch.cuda.empty_cache()

    if completed != expected_runs or len(result_paths) != expected_runs:
        raise RuntimeError("Supplemental run count is incomplete")
    for path in result_paths:
        if not path.is_file():
            raise RuntimeError(f"Missing supplemental result: {path}")
    completion = {
        "status": "COMPLETE_PENDING_ANALYSIS",
        "runs": completed,
        "expected_runs": expected_runs,
        "primary_results_available_before_supplement": True,
        "test_metrics_used_for_model_or_configuration_selection": False,
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
    print("Fixed recurrent supplement: COMPLETE PENDING ANALYSIS")
    print(f"Runs: {completed}")
    print(f"Completion: {COMPLETION_PATH}")


if __name__ == "__main__":
    main()
