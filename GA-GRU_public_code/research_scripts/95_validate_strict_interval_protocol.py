"""Audit the frozen strict-interval protocol without fitting any model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
PROTOCOL_DIR = PROJECT_DIR / "experiment_protocol_v7_strict_interval"
PROTOCOL_PATH = PROTOCOL_DIR / "strict_interval_protocol_v7.json"
HASH_PATH = PROTOCOL_DIR / "protocol_file_sha256.csv"
OUTPUT_DIR = (
    PROJECT_DIR / "outputs" / "independent_wells_v7_strict_interval" / "preflight"
)
REPORT_PATH = OUTPUT_DIR / "protocol_validation.json"

sys.path.insert(0, str(PROJECT_DIR))

from gagru.data import file_sha256  # noqa: E402
from gagru.single_well import window_class_counts  # noqa: E402
from gagru.strict_interval import (  # noqa: E402
    SPLITS,
    audit_strict_partition_isolation,
)
from gagru.within_well import prepare_within_well_frames  # noqa: E402


def validate_hashes() -> int:
    manifest = pd.read_csv(HASH_PATH, encoding="utf-8-sig")
    if manifest.empty or manifest["file"].duplicated().any():
        raise RuntimeError("Protocol hash manifest is empty or contains duplicates")
    for row in manifest.itertuples(index=False):
        path = PROTOCOL_DIR / str(row.file)
        if not path.is_file():
            raise RuntimeError(f"Frozen strict-protocol file is missing: {path}")
        if file_sha256(path) != str(row.sha256).lower():
            raise RuntimeError(f"Frozen strict-protocol file hash changed: {path}")
    return int(len(manifest))


def load_partitions(seed: int, well_id: str) -> dict[str, pd.DataFrame]:
    result = {}
    for split in SPLITS:
        path = (
            PROTOCOL_DIR
            / "partitions"
            / f"seed_{seed}"
            / well_id
            / f"{well_id}_{split}.csv"
        )
        frame = pd.read_csv(path, encoding="utf-8-sig")
        if set(frame["well_id"].astype(str)) != {well_id}:
            raise RuntimeError(f"{well_id}/{seed}/{split} contains another well")
        if set(frame["split"].astype(str)) != {split}:
            raise RuntimeError(f"{well_id}/{seed}/{split} has an invalid split marker")
        result[split] = frame
    return result


def validate_well_seed(
    protocol: dict[str, object], well_id: str, seed: int
) -> dict[str, object]:
    record = protocol["well_protocols"][well_id]
    seed_record = record["split_seeds"][str(seed)]
    classes = tuple(int(value) for value in record["included_classes"])
    features = tuple(str(value) for value in protocol["features"]["curves"])
    window_length = int(protocol["windows"]["length"])
    purge_m = float(protocol["split"]["purge_each_side_m"])

    source_path = PROTOCOL_DIR / str(record["source_snapshot"])
    if file_sha256(source_path) != str(record["source_snapshot_sha256"]):
        raise RuntimeError(f"{well_id} strict source snapshot hash changed")
    source = pd.read_csv(source_path, encoding="utf-8-sig")
    if source["source_row_id"].duplicated().any():
        raise RuntimeError(f"{well_id} strict source duplicates source-row IDs")
    if source["lithology_interval_id"].nunique() != int(
        record["complete_lithology_intervals"]
    ):
        raise RuntimeError(f"{well_id} complete-interval count changed")

    assignment_path = PROTOCOL_DIR / str(seed_record["assignment_file"])
    assignment = pd.read_csv(assignment_path, encoding="utf-8-sig")
    if file_sha256(assignment_path) != str(seed_record["assignment_sha256"]):
        raise RuntimeError(f"{well_id}/{seed} interval assignment hash changed")
    if assignment["lithology_interval_id"].duplicated().any():
        raise RuntimeError(f"{well_id}/{seed} assigns an interval more than once")
    if set(assignment["lithology_interval_id"].astype(str)) != set(
        source["lithology_interval_id"].astype(str)
    ):
        raise RuntimeError(f"{well_id}/{seed} assignment does not cover all intervals")
    if set(assignment["split"].astype(str)) != set(SPLITS):
        raise RuntimeError(f"{well_id}/{seed} assignment lacks a partition")

    guard_path = PROTOCOL_DIR / str(seed_record["guard_file"])
    guard = pd.read_csv(guard_path, encoding="utf-8-sig")
    if file_sha256(guard_path) != str(seed_record["guard_sha256"]):
        raise RuntimeError(f"{well_id}/{seed} guard-row hash changed")
    partitions = load_partitions(seed, well_id)
    retained = pd.concat(partitions.values(), ignore_index=True)
    retained_ids = set(retained["source_row_id"].astype(int))
    guard_ids = set(guard["source_row_id"].astype(int))
    source_ids = set(source["source_row_id"].astype(int))
    if retained_ids.intersection(guard_ids):
        raise RuntimeError(f"{well_id}/{seed} guard and retained rows overlap")
    if retained_ids.union(guard_ids) != source_ids:
        raise RuntimeError(
            f"{well_id}/{seed} retained and guard rows do not cover source"
        )

    interval_to_split = assignment.set_index("lithology_interval_id")["split"].astype(
        str
    )
    for split, frame in {**partitions, "guard": guard}.items():
        expected = frame["lithology_interval_id"].map(interval_to_split)
        observed = frame["split"].astype(str)
        if not expected.astype(str).equals(observed):
            raise RuntimeError(f"{well_id}/{seed}/{split} breaks interval assignment")

    audit = audit_strict_partition_isolation(
        partitions,
        window_length=window_length,
        purge_each_side_m=purge_m,
    )
    required_audits = protocol["required_audits"]
    audit_keys = {
        "target_center_overlap": "target_center_overlap_across_splits",
        "source_row_overlap": "source_row_overlap_across_splits",
        "input_context_overlap": "input_context_overlap_across_splits",
        "complete_interval_overlap": "complete_interval_overlap_across_splits",
    }
    for protocol_key, audit_key in audit_keys.items():
        if int(audit[audit_key]) != int(required_audits[protocol_key]):
            raise RuntimeError(f"{well_id}/{seed} failed {protocol_key}")

    counts = window_class_counts(retained, window_length).loc[:, list(classes)]
    observed_windows = {split: int(counts.loc[split].sum()) for split in SPLITS}
    if observed_windows != {
        split: int(seed_record["windows"][split]) for split in SPLITS
    }:
        raise RuntimeError(f"{well_id}/{seed} frozen window counts changed")
    minimum_training = int(
        protocol["split"]["minimum_training_windows_per_modeled_class"]
    )
    if (counts.loc["train"] < minimum_training).any():
        raise RuntimeError(
            f"{well_id}/{seed} has fewer than {minimum_training} training windows "
            "for a modeled class"
        )

    task = prepare_within_well_frames(
        partitions["train"],
        partitions["validation"],
        well_id=well_id,
        features=features,
        window_length=window_length,
        global_classes=classes,
        require_validation_all_classes=False,
    )
    return {
        "rows": {split: int(len(partitions[split])) for split in SPLITS},
        "guard_rows": int(len(guard)),
        "windows": observed_windows,
        "supported_classes": {
            split: [
                class_id for class_id in classes if int(counts.loc[split, class_id]) > 0
            ]
            for split in SPLITS
        },
        "training_only_imputation_medians": task.imputation_medians,
        "isolation": audit,
    }


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol["status"] != "FROZEN_BEFORE_STRICT_MODEL_SELECTION":
        raise RuntimeError("Unexpected strict protocol status")
    scope = protocol["research_scope"]
    if not scope["one_independent_model_per_well"]:
        raise RuntimeError("Strict protocol does not require independent well models")
    if scope["samples_shared_between_well_models"]:
        raise RuntimeError("Strict protocol permits sample sharing between well models")
    if scope["cross_well_training_or_testing"]:
        raise RuntimeError("Strict protocol permits cross-well modeling")

    hashed_files = validate_hashes()
    wells = tuple(str(value) for value in scope["wells"])
    seeds = tuple(int(value) for value in protocol["split"]["split_seeds"])
    reports: dict[str, object] = {}
    for seed in seeds:
        reports[str(seed)] = {
            well_id: validate_well_seed(protocol, well_id, seed) for well_id in wells
        }

    output = {
        "status": "PASS",
        "protocol": str(PROTOCOL_PATH),
        "hashed_files_verified": hashed_files,
        "wells": list(wells),
        "split_seeds": list(seeds),
        "models_fitted": False,
        "test_metrics_computed": False,
        "test_used_for_preprocessing_or_model_selection": False,
        "target_center_overlap": 0,
        "source_row_overlap": 0,
        "input_context_overlap": 0,
        "complete_interval_overlap": 0,
        "reports": reports,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Strict complete-interval protocol validation: PASS")
    print(f"Frozen files verified: {hashed_files}")
    print(f"Wells: {len(wells)}; split seeds: {len(seeds)}")
    print("Target-center overlap: 0")
    print("Source-row overlap: 0")
    print("Input-context overlap: 0")
    print("Complete-interval overlap: 0")
    print("Preprocessing was fitted on training partitions only.")
    print("No model was fitted and no test metric was computed.")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
