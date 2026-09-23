from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from .data import DataRepository, file_sha256
from .folds import prepare_inner_fold
from .model import GRUModelConfig
from .protocol import load_frozen_protocol
from .training import TrainingConfig, resolve_device, train_gru


def candidate_budget(median_runtime_seconds: float) -> int:
    if median_runtime_seconds <= 60.0:
        return 36
    if median_runtime_seconds <= 180.0:
        return 24
    return 12


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _code_hashes() -> dict[str, str]:
    package_dir = Path(__file__).resolve().parent
    names = (
        "data.py",
        "folds.py",
        "metrics.py",
        "model.py",
        "preprocessing.py",
        "protocol.py",
        "training.py",
        "windows.py",
        "pilot.py",
    )
    return {name: file_sha256(package_dir / name) for name in names}


def protocol_training_configs(max_epochs: int | None = None) -> tuple[GRUModelConfig, TrainingConfig]:
    protocol = load_frozen_protocol()
    fixed = protocol.raw["fixed_gru"]
    training = protocol.raw["training"]
    model_config = GRUModelConfig(
        input_size=len(protocol.feature_names),
        hidden_size=int(fixed["hidden_size"]),
        num_layers=int(fixed["num_layers"]),
        output_size=len(protocol.class_names),
        dropout=float(fixed["dropout"]),
    )
    training_config = TrainingConfig(
        learning_rate=float(fixed["learning_rate"]),
        weight_decay=float(fixed["weight_decay"]),
        batch_size=int(training["batch_size"]),
        max_epochs=int(max_epochs or training["max_epochs"]),
        early_stopping_patience=int(training["early_stopping_patience"]),
        early_stopping_min_delta=float(training["early_stopping_min_delta"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
    )
    return model_config, training_config


def run_training_smoke_test(
    output_path: str | Path,
    *,
    device: str = "auto",
    epochs: int = 2,
) -> dict[str, Any]:
    protocol = load_frozen_protocol()
    repository = DataRepository(protocol)
    split = prepare_inner_fold(repository, 1, window_length=9)
    model_config, training_config = protocol_training_configs(max_epochs=epochs)
    result = train_gru(
        split,
        model_config,
        training_config,
        seed=int(protocol.raw["training_seeds"][0]),
        device=device,
    )
    report = {
        "status": "PASS",
        "purpose": "implementation_smoke_test_not_used_for_scientific_conclusions",
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "fold": 1,
        "training_wells": list(split.train_wells),
        "validation_wells": list(split.evaluation_wells),
        "training_windows": len(split.train.y),
        "validation_windows": len(split.evaluation.y),
        "epochs_requested": epochs,
        "epochs_completed": result.epochs_completed,
        "runtime_seconds": result.runtime_seconds,
        "device": result.device,
        "trainable_parameters": result.model.trainable_parameters,
        "finite_losses": all(
            record.train_loss == record.train_loss
            and record.validation_loss == record.validation_loss
            for record in result.history
        ),
        "locked_external_wells_not_read": sorted(protocol.locked_external_wells),
    }
    if set(repository._cache).intersection(protocol.locked_external_wells):
        raise RuntimeError("A locked external well entered the smoke-test cache")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def run_runtime_pilot(
    output_dir: str | Path,
    *,
    device: str = "auto",
) -> dict[str, Any]:
    protocol = load_frozen_protocol()
    repository = DataRepository(protocol)
    preparation_started = time.perf_counter()
    split = prepare_inner_fold(repository, 1, window_length=9)
    preparation_seconds = time.perf_counter() - preparation_started
    model_config, training_config = protocol_training_configs()
    repetitions = int(protocol.raw["search"]["pilot_repetitions"])
    seeds = tuple(int(seed) for seed in protocol.raw["training_seeds"][:repetitions])
    if len(seeds) != repetitions:
        raise RuntimeError("Not enough frozen training seeds for the runtime pilot")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_summaries: list[dict[str, Any]] = []
    for run_number, seed in enumerate(seeds, start=1):
        print(f"Starting runtime pilot run {run_number}/{repetitions}, seed={seed} ...")

        def show_progress(record: Any) -> None:
            if record.epoch == 1 or record.epoch % 10 == 0:
                print(
                    f"  run {run_number}: epoch {record.epoch}/"
                    f"{training_config.max_epochs}, {record.epoch_seconds:.2f} s"
                )

        result = train_gru(
            split,
            model_config,
            training_config,
            seed=seed,
            device=device,
            progress_callback=show_progress,
        )
        print(
            f"Completed run {run_number}/{repetitions}: "
            f"{result.epochs_completed} epochs, {result.runtime_seconds:.2f} s"
        )
        history_path = output / f"history_run_{run_number}_seed_{seed}.csv"
        with history_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result.history[0].to_dict()))
            writer.writeheader()
            writer.writerows(record.to_dict() for record in result.history)
        summary = result.summary_dict()
        summary["run_number"] = run_number
        summary["history_file"] = history_path.name
        run_summaries.append(summary)

    runtimes = [float(run["runtime_seconds"]) for run in run_summaries]
    median_runtime = float(statistics.median(runtimes))
    budget = candidate_budget(median_runtime)
    selected_device = resolve_device(device)
    device_name = (
        torch.cuda.get_device_name(selected_device)
        if selected_device.type == "cuda"
        else platform.processor() or "CPU"
    )
    report = {
        "status": "COMPLETE",
        "purpose": "runtime_only_budget_selection_no_scientific_conclusion",
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol_id": protocol.raw["protocol"]["id"],
        "protocol_version": protocol.raw["protocol"]["version"],
        "dataset_aggregate_sha256": protocol.raw["dataset"]["aggregate_well_file_sha256"],
        "fold": 1,
        "training_wells": list(split.train_wells),
        "validation_wells": list(split.evaluation_wells),
        "training_windows": len(split.train.y),
        "validation_windows": len(split.evaluation.y),
        "window_length": split.train.window_length,
        "features": list(split.train.feature_names),
        "model_config": model_config.to_dict(),
        "training_config": training_config.to_dict(),
        "pilot_seeds": list(seeds),
        "data_preparation_seconds_excluded_from_T": preparation_seconds,
        "runs": run_summaries,
        "runtime_seconds": runtimes,
        "median_runtime_T_seconds": median_runtime,
        "candidate_budget_B": budget,
        "budget_rule": "T<=60:36; 60<T<=180:24; T>180:12",
        "performance_metrics_used_to_choose_B": False,
        "device": str(selected_device),
        "device_name": device_name,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_threads": torch.get_num_threads(),
        "packages": {
            "numpy": _version("numpy"),
            "pandas": _version("pandas"),
            "scikit-learn": _version("scikit-learn"),
            "torch": _version("torch"),
        },
        "code_sha256": _code_hashes(),
        "locked_external_wells_not_read": sorted(protocol.locked_external_wells),
    }
    if set(repository._cache).intersection(protocol.locked_external_wells):
        raise RuntimeError("A locked external well entered the runtime-pilot cache")
    summary_path = output / "runtime_pilot_summary.json"
    summary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    digest_path = output / "runtime_pilot_summary.sha256"
    digest_path.write_text(
        hashlib.sha256(summary_path.read_bytes()).hexdigest() + "\n", encoding="ascii"
    )
    return report
