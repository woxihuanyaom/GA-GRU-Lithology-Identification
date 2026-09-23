"""Validate the frozen v5 center assignments without fitting a model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
PROTOCOL_DIR = PROJECT_DIR / "experiment_protocol_v5_random_center"
PROTOCOL_PATH = PROTOCOL_DIR / "random_center_protocol_v5.json"
HASH_PATH = PROTOCOL_DIR / "protocol_file_sha256.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "independent_wells_v5" / "preflight"
REPORT_PATH = OUTPUT_DIR / "protocol_validation.json"

sys.path.insert(0, str(PROJECT_DIR))

from gagru.data import file_sha256  # noqa: E402
from gagru.random_center import (  # noqa: E402
    SPLITS,
    build_random_center_windows,
    center_and_context_overlap_audit,
    eligible_center_ids,
    impute_from_training_windows,
)


def validate_hashes() -> int:
    manifest = pd.read_csv(HASH_PATH, encoding="utf-8-sig")
    if manifest.empty or manifest["file"].duplicated().any():
        raise RuntimeError("Protocol hash manifest is empty or contains duplicates")
    for row in manifest.itertuples(index=False):
        path = PROTOCOL_DIR / str(row.file)
        if not path.is_file():
            raise RuntimeError(f"Frozen protocol file is missing: {path}")
        if file_sha256(path) != str(row.sha256).lower():
            raise RuntimeError(f"Frozen protocol file hash changed: {path}")
    return int(len(manifest))


def validate_well_seed(
    protocol: dict[str, object], well_id: str, seed: int
) -> dict[str, object]:
    well_record = protocol["well_protocols"][well_id]
    classes = tuple(int(value) for value in well_record["included_classes"])
    features = tuple(str(value) for value in protocol["features"]["extended_seven"])
    window_length = int(protocol["windows"]["length"])
    source_path = PROTOCOL_DIR / str(well_record["source_snapshot"])
    frame = pd.read_csv(source_path, encoding="utf-8-sig")
    if file_sha256(source_path) != str(well_record["source_snapshot_sha256"]):
        raise RuntimeError(f"{well_id} source snapshot hash differs from protocol")
    if set(frame["well_id"].astype(str)) != {well_id}:
        raise RuntimeError(f"{well_id} source snapshot contains another well")
    if set(frame["class_id"].astype(int)) != set(classes):
        raise RuntimeError(f"{well_id} source snapshot class set changed")
    if frame["source_row_id"].duplicated().any():
        raise RuntimeError(f"{well_id} source snapshot duplicates source rows")

    assignments = []
    split_counts: dict[str, int] = {}
    class_counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        path = (
            PROTOCOL_DIR
            / "center_assignments"
            / f"seed_{seed}"
            / well_id
            / f"{well_id}_{split}_centers.csv"
        )
        assignment = pd.read_csv(path, encoding="utf-8-sig")
        if set(assignment["split"].astype(str)) != {split}:
            raise RuntimeError(f"{well_id}/{seed}/{split} split marker changed")
        if set(assignment["well_id"].astype(str)) != {well_id}:
            raise RuntimeError(f"{well_id}/{seed}/{split} contains another well")
        if set(assignment["class_id"].astype(int)) != set(classes):
            raise RuntimeError(f"{well_id}/{seed}/{split} lost a frozen class")
        assignments.append(assignment)
        split_counts[split] = int(len(assignment))
        class_counts[split] = {
            str(class_id): int((assignment["class_id"].astype(int) == class_id).sum())
            for class_id in classes
        }
    all_assignments = pd.concat(assignments, ignore_index=True)
    if all_assignments["source_row_id"].duplicated().any():
        raise RuntimeError(f"{well_id}/{seed} shares target centers between splits")
    eligible = set(eligible_center_ids(frame, window_length).astype(int).tolist())
    assigned = set(all_assignments["source_row_id"].astype(int).tolist())
    if assigned != eligible:
        raise RuntimeError(f"{well_id}/{seed} assignment does not cover eligible centers")
    source_labels = frame.set_index("source_row_id")["class_id"].astype(int)
    assigned_labels = all_assignments.set_index("source_row_id")["class_id"].astype(int)
    if not source_labels.loc[assigned_labels.index].equals(assigned_labels):
        raise RuntimeError(f"{well_id}/{seed} assignment labels differ from source")

    windows = build_random_center_windows(
        frame, all_assignments, features, window_length
    )
    filled, medians = impute_from_training_windows(windows)
    if any(len(filled[split].y) != split_counts[split] for split in SPLITS):
        raise RuntimeError(f"{well_id}/{seed} center and window counts differ")
    overlap = center_and_context_overlap_audit(filled)
    for split in SPLITS:
        if set(filled[split].y.astype(int).tolist()) != set(classes):
            raise RuntimeError(f"{well_id}/{seed}/{split} windows lost a class")
    return {
        "centers": split_counts,
        "class_counts": class_counts,
        "training_window_medians": medians,
        "overlap": overlap,
    }


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol["status"] != "FROZEN_BEFORE_RANDOM_CENTER_MODEL_SCREENING":
        raise RuntimeError("Unexpected v5 protocol status")
    scope = protocol["research_scope"]
    if not scope["one_independent_model_per_well"]:
        raise RuntimeError("Protocol does not require independent well models")
    if scope["samples_shared_between_well_models"]:
        raise RuntimeError("Protocol permits sample sharing between well models")
    if scope["cross_well_training_or_testing"]:
        raise RuntimeError("Protocol permits cross-well modeling")
    wells = tuple(str(value) for value in scope["wells"])
    seeds = tuple(int(value) for value in protocol["split"]["split_seeds"])
    hashed_files = validate_hashes()
    reports: dict[str, object] = {}
    overlap_values: list[float] = []
    for seed in seeds:
        seed_report = {}
        for well_id in wells:
            report = validate_well_seed(protocol, well_id, seed)
            seed_report[well_id] = report
            overlap_values.append(
                float(
                    report["overlap"]["pairs"]["train_to_test"][
                        "second_context_overlap_fraction"
                    ]
                )
            )
        reports[str(seed)] = seed_report

    output = {
        "status": "PASS",
        "protocol": str(PROTOCOL_PATH),
        "hashed_files_verified": hashed_files,
        "wells": list(wells),
        "split_seeds": list(seeds),
        "models_fitted": False,
        "test_metrics_computed": False,
        "target_center_overlap": 0,
        "train_test_context_overlap_fraction_range": [
            float(np.min(overlap_values)),
            float(np.max(overlap_values)),
        ],
        "context_overlap_interpretation": (
            "Observed curve rows are reused across neighboring windows; target labels remain "
            "disjoint. This is a same-well interpolation protocol, not an independent-window "
            "or unseen-well generalization test."
        ),
        "reports": reports,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Random-center protocol validation: PASS")
    print(f"Frozen files verified: {hashed_files}")
    print(f"Wells: {len(wells)}; split seeds: {len(seeds)}")
    print("Target-center overlap across train/validation/test: 0")
    print(
        "Train/test observed-curve context overlap range: "
        f"{min(overlap_values):.4f}-{max(overlap_values):.4f}"
    )
    print("No model was fitted and no test metric was computed.")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
