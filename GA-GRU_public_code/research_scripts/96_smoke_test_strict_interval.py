"""Smoke-test strict train/validation preparation while keeping test locked."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch


PROJECT_DIR = Path(__file__).resolve().parent
PROTOCOL_DIR = PROJECT_DIR / "experiment_protocol_v7_strict_interval"
PROTOCOL_PATH = PROTOCOL_DIR / "strict_interval_protocol_v7.json"
VALIDATION_REPORT = (
    PROJECT_DIR
    / "outputs"
    / "independent_wells_v7_strict_interval"
    / "preflight"
    / "protocol_validation.json"
)
OUTPUT_DIR = (
    PROJECT_DIR / "outputs" / "independent_wells_v7_strict_interval" / "smoke_test"
)
REPORT_PATH = OUTPUT_DIR / "strict_training_smoke_test.json"

sys.path.insert(0, str(PROJECT_DIR))

from gagru.within_well import (  # noqa: E402
    fit_fixed_gru_validation,
    prepare_within_well_frames,
)


def read_development_partition(seed: int, well_id: str, split: str) -> pd.DataFrame:
    if split not in {"train", "validation"}:
        raise RuntimeError("The strict smoke test may read only train and validation")
    path = (
        PROTOCOL_DIR
        / "partitions"
        / f"seed_{seed}"
        / well_id
        / f"{well_id}_{split}.csv"
    )
    return pd.read_csv(path, encoding="utf-8-sig")


def main() -> None:
    if not VALIDATION_REPORT.is_file():
        raise RuntimeError("Run 95_validate_strict_interval_protocol.py first")
    validation = json.loads(VALIDATION_REPORT.read_text(encoding="utf-8"))
    if validation["status"] != "PASS":
        raise RuntimeError("Strict protocol validation did not pass")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    well_id = str(protocol["research_scope"]["main_well"])
    seed = int(protocol["split"]["split_seeds"][0])
    features = tuple(str(value) for value in protocol["features"]["curves"])
    classes = tuple(
        int(value) for value in protocol["well_protocols"][well_id]["included_classes"]
    )
    window_length = int(protocol["windows"]["length"])

    train = read_development_partition(seed, well_id, "train")
    validation_frame = read_development_partition(seed, well_id, "validation")
    task = prepare_within_well_frames(
        train,
        validation_frame,
        well_id=well_id,
        features=features,
        window_length=window_length,
        global_classes=classes,
        require_validation_all_classes=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = fit_fixed_gru_validation(
        task,
        seed=17,
        device=device,
        hidden_size=32,
        num_layers=1,
        dropout=0.0,
        batch_size=512,
        max_epochs=2,
        patience=2,
        class_weighted=True,
    )
    supported_validation = sorted(set(task.validation.y.astype(int).tolist()))
    output = {
        "status": "PASS",
        "protocol": str(PROTOCOL_PATH),
        "well_id": well_id,
        "split_seed": seed,
        "device": str(device),
        "features": list(features),
        "window_length": window_length,
        "training_windows": int(len(task.train.y)),
        "validation_windows": int(len(task.validation.y)),
        "modeled_classes": list(classes),
        "validation_supported_classes": supported_validation,
        "epochs_completed": result.epochs_completed,
        "runtime_seconds": result.runtime_seconds,
        "trainable_parameters": result.trainable_parameters,
        "test_partition_read": False,
        "test_metrics_computed": False,
        "result_role": "pipeline smoke test only; not a reported model estimate",
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Strict train/validation smoke test: PASS")
    print(f"Well: {well_id}; split seed: {seed}; device: {device}")
    print(f"Windows: train={len(task.train.y)}, validation={len(task.validation.y)}")
    print(f"Epochs completed: {result.epochs_completed}")
    print("Test partition was not read and no test metric was computed.")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
