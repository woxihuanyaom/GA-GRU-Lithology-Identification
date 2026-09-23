"""Select fair RNN, LSTM, and plain-GRU baselines without test access."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


PROJECT_DIR = Path(__file__).resolve().parent
PROTOCOL_DIR = PROJECT_DIR / "experiment_protocol_v5_random_center"
PROTOCOL_PATH = PROTOCOL_DIR / "random_center_protocol_v5.json"
OPTIMIZER_FREEZE_PATH = (
    PROJECT_DIR
    / "outputs"
    / "independent_wells_v5"
    / "optimizer_comparison_audit"
    / "optimizer_winner_freeze.json"
)
OUTPUT_DIR = (
    PROJECT_DIR / "outputs" / "independent_wells_v5" / "recurrent_baseline_screen"
)

ARCHITECTURES = ("vanilla_rnn", "lstm", "gru")
TRAINING_SEED = 1701
BATCH_SIZE = 512
MAX_EPOCHS = 60
PATIENCE = 8

sys.path.insert(0, str(PROJECT_DIR))

from gagru.local_recurrent import fit_local_recurrent_with_validation  # noqa: E402
from gagru.random_center import (  # noqa: E402
    build_random_center_windows,
    impute_from_training_windows,
    remap_labels,
)
from gagru.residual_bigru import prepare_per_well_inputs  # noqa: E402
from gagru.search import GRUSearchCandidate  # noqa: E402


def fixed_candidate() -> GRUSearchCandidate:
    return GRUSearchCandidate(96, 2, 7e-4, 0.2, 1e-4)


def transferred_candidate() -> GRUSearchCandidate:
    return GRUSearchCandidate(
        128, 3, 0.0029380455401443366, 0.3, 6.532858135218046e-05
    )


def candidate_set(freeze: dict[str, Any]) -> list[tuple[str, GRUSearchCandidate]]:
    candidates = [
        ("fixed", fixed_candidate()),
        ("transferred_development", transferred_candidate()),
    ]
    for method in ("genetic_algorithm", "random_search", "tpe"):
        candidates.append(
            (
                f"{method}_winner",
                GRUSearchCandidate.from_dict(freeze["methods"][method]["candidate"]),
            )
        )
    keys = [candidate.key for _, candidate in candidates]
    if len(set(keys)) != len(keys):
        raise RuntimeError("The five frozen recurrent candidates are not unique")
    return candidates


def read_development_assignments(well_id: str, split_seed: int) -> pd.DataFrame:
    return pd.concat(
        [
            pd.read_csv(
                PROTOCOL_DIR
                / "center_assignments"
                / f"seed_{split_seed}"
                / well_id
                / f"{well_id}_{split}_centers.csv",
                encoding="utf-8-sig",
            )
            for split in ("train", "validation")
        ],
        ignore_index=True,
    )


def read_existing(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid baseline log line {line_number}") from exc
    return records


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Formal recurrent baseline screen requires CUDA")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    freeze = json.loads(OPTIMIZER_FREEZE_PATH.read_text(encoding="utf-8"))
    split_seed = int(protocol["split"]["primary_model_selection_split_seed"])
    wells = tuple(str(value) for value in protocol["research_scope"]["wells"])
    features = tuple(str(value) for value in protocol["features"]["extended_seven"])
    window_length = int(protocol["windows"]["length"])
    candidates = candidate_set(freeze)

    tasks: dict[str, dict[str, Any]] = {}
    for well_index, well_id in enumerate(wells):
        record = protocol["well_protocols"][well_id]
        classes = tuple(int(value) for value in record["included_classes"])
        frame = pd.read_csv(
            PROTOCOL_DIR / str(record["source_snapshot"]), encoding="utf-8-sig"
        )
        assignment = read_development_assignments(well_id, split_seed)
        windows = build_random_center_windows(
            frame,
            assignment,
            features,
            window_length,
            requested_splits=("train", "validation"),
        )
        windows, _ = impute_from_training_windows(windows)
        y_train, _ = remap_labels(windows["train"], classes)
        y_validation, _ = remap_labels(windows["validation"], classes)
        X_train, X_validation = prepare_per_well_inputs(
            windows["train"], windows["validation"]
        )
        tasks[well_id] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_validation": X_validation,
            "y_validation": y_validation,
            "training_seed": TRAINING_SEED + well_index,
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    histories = OUTPUT_DIR / "histories"
    histories.mkdir(exist_ok=True)
    log_path = OUTPUT_DIR / "configuration_evaluations.jsonl"
    manifest_path = OUTPUT_DIR / "screen_manifest.json"
    summary_path = OUTPUT_DIR / "screen_summary.json"
    manifest = {
        "status": "FROZEN_BEFORE_SCREEN",
        "scope": "training_and_validation_centers_only",
        "test_assignment_files_opened": False,
        "architectures": list(ARCHITECTURES),
        "candidates": [
            {"source": source, "candidate": candidate.to_dict()}
            for source, candidate in candidates
        ],
        "configurations": len(ARCHITECTURES) * len(candidates),
        "split_seed": split_seed,
        "training_seed_policy": "1701 plus zero-based well index",
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "selection_metric": "mean of six per-well supported macro-F1 values",
        "features": list(features),
        "derived_channels": "three resistivity separations plus first differences",
        "window_length": window_length,
        "wells": list(wells),
        "samples_shared_between_well_models": False,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
    }
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError("Existing recurrent baseline manifest conflicts")
    else:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    existing = read_existing(log_path)
    position = 0
    records: list[dict[str, Any]] = []
    for architecture in ARCHITECTURES:
        for candidate_source, candidate in candidates:
            configuration_index = position + 1
            if position < len(existing):
                saved = existing[position]
                if (
                    saved["configuration_index"] != configuration_index
                    or saved["architecture"] != architecture
                    or saved["candidate_key"] != candidate.key
                ):
                    raise RuntimeError("Recurrent baseline deterministic replay diverged")
                records.append(saved)
                position += 1
                print(
                    f"Replayed {configuration_index}/{manifest['configurations']}: "
                    f"{architecture}/{candidate_source}, "
                    f"macro-F1={saved['mean_per_well_macro_f1']:.4f}",
                    flush=True,
                )
                continue

            per_well = []
            total_runtime = 0.0
            for well_id, task in tasks.items():
                torch.cuda.empty_cache()
                fit = fit_local_recurrent_with_validation(
                    architecture,
                    candidate,
                    task["X_train"],
                    task["y_train"],
                    task["X_validation"],
                    task["y_validation"],
                    device=device,
                    seed=int(task["training_seed"]),
                    batch_size=BATCH_SIZE,
                    max_epochs=MAX_EPOCHS,
                    patience=PATIENCE,
                )
                row = {
                    "well_id": well_id,
                    "validation_samples": int(len(task["y_validation"])),
                    "macro_f1": fit.validation_macro_f1,
                    "accuracy": fit.validation_accuracy,
                    "balanced_accuracy": fit.validation_balanced_accuracy,
                    "best_epoch": fit.best_epoch,
                    "epochs_completed": fit.epochs_completed,
                    "runtime_seconds": fit.runtime_seconds,
                    "trainable_parameters": fit.trainable_parameters,
                }
                per_well.append(row)
                total_runtime += fit.runtime_seconds
                pd.DataFrame(fit.history).to_csv(
                    histories
                    / f"configuration_{configuration_index:02d}_{well_id}.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
            well_frame = pd.DataFrame(per_well)
            record = {
                "configuration_index": configuration_index,
                "architecture": architecture,
                "candidate_source": candidate_source,
                "candidate_key": candidate.key,
                "candidate": candidate.to_dict(),
                "mean_per_well_macro_f1": float(well_frame["macro_f1"].mean()),
                "mean_per_well_accuracy": float(well_frame["accuracy"].mean()),
                "pooled_accuracy": float(
                    np.average(
                        well_frame["accuracy"],
                        weights=well_frame["validation_samples"],
                    )
                ),
                "mean_per_well_balanced_accuracy": float(
                    well_frame["balanced_accuracy"].mean()
                ),
                "runtime_seconds": total_runtime,
                "per_well": per_well,
            }
            with log_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            records.append(record)
            position += 1
            print(
                f"Completed {configuration_index}/{manifest['configurations']}: "
                f"{architecture}/{candidate_source}, "
                f"macro-F1={record['mean_per_well_macro_f1']:.4f}, "
                f"accuracy={record['mean_per_well_accuracy']:.4f}",
                flush=True,
            )

    completed_log = read_existing(log_path)
    if (
        position != int(manifest["configurations"])
        or len(completed_log) != int(manifest["configurations"])
    ):
        raise RuntimeError("Baseline log does not contain every frozen configuration")
    winners = {}
    summary_rows = []
    for architecture in ARCHITECTURES:
        architecture_records = [
            record for record in records if record["architecture"] == architecture
        ]
        winner = max(
            architecture_records,
            key=lambda record: (
                record["mean_per_well_macro_f1"],
                record["mean_per_well_balanced_accuracy"],
                record["mean_per_well_accuracy"],
                -record["runtime_seconds"],
            ),
        )
        winners[architecture] = {
            "candidate_source": winner["candidate_source"],
            "candidate": winner["candidate"],
            "validation_mean_per_well_macro_f1": winner[
                "mean_per_well_macro_f1"
            ],
            "validation_mean_per_well_accuracy": winner[
                "mean_per_well_accuracy"
            ],
            "validation_pooled_accuracy": winner["pooled_accuracy"],
            "test_metrics_used_for_selection": False,
        }
        for record in architecture_records:
            summary_rows.append(
                {
                    key: value
                    for key, value in record.items()
                    if key not in {"per_well", "candidate_key"}
                }
            )
    summary = {
        "status": "COMPLETE",
        "test_assignment_files_opened": False,
        "test_metrics_used_for_selection": False,
        "samples_shared_between_well_models": False,
        "winners": winners,
    }
    pd.DataFrame(summary_rows).to_csv(
        OUTPUT_DIR / "configuration_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(
        [
            pd.DataFrame(record["per_well"]).assign(
                architecture=record["architecture"],
                candidate_source=record["candidate_source"],
            )
            for record in records
        ],
        ignore_index=True,
    ).to_csv(
        OUTPUT_DIR / "configuration_per_well_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nRecurrent baseline winners")
    print(json.dumps(winners, ensure_ascii=False, indent=2))
    print("No test assignment file was opened or scored.")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
