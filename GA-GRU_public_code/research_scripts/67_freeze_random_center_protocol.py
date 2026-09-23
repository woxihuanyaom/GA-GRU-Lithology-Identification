"""Freeze repeated independent single-well random-center interpolation tasks."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_PROTOCOL_DIR = PROJECT_DIR / "experiment_protocol_v4_randomized_within_well"
SOURCE_PROTOCOL_PATH = SOURCE_PROTOCOL_DIR / "randomized_within_well_protocol_v4.json"
OUTPUT_DIR = PROJECT_DIR / "experiment_protocol_v5_random_center"

WINDOW_LENGTH = 9
FRACTIONS = (0.70, 0.15, 0.15)
SPLIT_SEEDS = (20260917, 20260918, 20260919)
PRIMARY_SPLIT_SEED = SPLIT_SEEDS[0]

sys.path.insert(0, str(PROJECT_DIR))

from gagru.data import file_sha256  # noqa: E402
from gagru.random_center import (  # noqa: E402
    SPLITS,
    add_source_row_ids,
    build_random_center_windows,
    center_and_context_overlap_audit,
    stratified_center_assignment,
)
from gagru.single_well import resegment_after_class_filter  # noqa: E402


def main() -> None:
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise RuntimeError(
            f"Protocol directory is not empty; refusing to overwrite: {OUTPUT_DIR}"
        )
    source_protocol = json.loads(SOURCE_PROTOCOL_PATH.read_text(encoding="utf-8"))
    wells = tuple(str(value) for value in source_protocol["research_scope"]["wells"])
    features = tuple(str(value) for value in source_protocol["features"]["extended_seven"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    data_dir = OUTPUT_DIR / "data"
    split_dir = OUTPUT_DIR / "center_assignments"
    data_dir.mkdir()
    split_dir.mkdir()
    files_to_hash: list[Path] = []
    well_records: dict[str, object] = {}
    count_parts: list[pd.DataFrame] = []
    overlap_records: list[dict[str, object]] = []

    for well_id in wells:
        source_record = source_protocol["well_protocols"][well_id]
        source_path = Path(str(source_record["source_path"]))
        expected_hash = str(source_record["source_sha256"]).lower()
        if file_sha256(source_path) != expected_hash:
            raise RuntimeError(f"Staged source hash changed for {well_id}")
        classes = tuple(int(value) for value in source_record["included_classes"])
        raw = pd.read_csv(source_path, encoding="utf-8-sig")
        filtered = resegment_after_class_filter(
            raw.loc[raw["class_id"].isin(classes)].copy()
        )
        frame = add_source_row_ids(filtered)
        well_data_dir = data_dir / well_id
        well_data_dir.mkdir()
        snapshot_path = well_data_dir / f"{well_id}_eligible_source.csv"
        frame.to_csv(snapshot_path, index=False, encoding="utf-8-sig")
        files_to_hash.append(snapshot_path)

        seed_records: dict[str, object] = {}
        for seed in SPLIT_SEEDS:
            assignment = stratified_center_assignment(
                frame,
                window_length=WINDOW_LENGTH,
                seed=seed,
                fractions=FRACTIONS,
            )
            windows = build_random_center_windows(
                frame, assignment, features, WINDOW_LENGTH
            )
            audit = center_and_context_overlap_audit(windows)
            assignment_dir = split_dir / f"seed_{seed}" / well_id
            assignment_dir.mkdir(parents=True)
            counts = []
            split_hashes: dict[str, str] = {}
            for split in SPLITS:
                split_frame = assignment.loc[assignment["split"].eq(split)].copy()
                path = assignment_dir / f"{well_id}_{split}_centers.csv"
                split_frame.to_csv(path, index=False, encoding="utf-8-sig")
                files_to_hash.append(path)
                split_hashes[split] = file_sha256(path)
                by_class = split_frame.groupby("class_id").size()
                for class_id in classes:
                    counts.append(
                        {
                            "split_seed": seed,
                            "well_id": well_id,
                            "split": split,
                            "class_id": class_id,
                            "centers": int(by_class.get(class_id, 0)),
                        }
                    )
            count_parts.append(pd.DataFrame(counts))
            test_overlap = audit["pairs"]["train_to_test"]
            overlap_records.append(
                {
                    "split_seed": seed,
                    "well_id": well_id,
                    "train_centers": len(windows["train"].y),
                    "validation_centers": len(windows["validation"].y),
                    "test_centers": len(windows["test"].y),
                    "target_center_overlap": audit["center_overlap_across_splits"],
                    "train_test_shared_context_rows": test_overlap[
                        "shared_context_rows"
                    ],
                    "test_context_rows": test_overlap["second_context_rows"],
                    "test_context_overlap_fraction": test_overlap[
                        "second_context_overlap_fraction"
                    ],
                }
            )
            seed_records[str(seed)] = {
                "centers": {
                    split: int(len(windows[split].y)) for split in SPLITS
                },
                "classes_per_split": {
                    split: sorted(set(windows[split].y.astype(int).tolist()))
                    for split in SPLITS
                },
                "center_overlap_across_splits": audit[
                    "center_overlap_across_splits"
                ],
                "context_overlap": audit["pairs"],
                "assignment_sha256": split_hashes,
            }
        well_records[well_id] = {
            "role": source_record["role"],
            "included_classes": list(classes),
            "source_snapshot": str(snapshot_path.relative_to(OUTPUT_DIR)),
            "source_snapshot_sha256": file_sha256(snapshot_path),
            "source_rows": int(len(frame)),
            "split_seeds": seed_records,
        }
        print(
            f"Frozen {well_id}: rows={len(frame)}, classes={len(classes)}, "
            f"eligible L9 centers={sum(seed_records[str(PRIMARY_SPLIT_SEED)]['centers'].values())}",
            flush=True,
        )

    counts_path = OUTPUT_DIR / "center_class_counts.csv"
    pd.concat(count_parts, ignore_index=True).to_csv(
        counts_path, index=False, encoding="utf-8-sig"
    )
    files_to_hash.append(counts_path)
    overlap_path = OUTPUT_DIR / "context_overlap_audit.csv"
    pd.DataFrame(overlap_records).to_csv(
        overlap_path, index=False, encoding="utf-8-sig"
    )
    files_to_hash.append(overlap_path)

    protocol = {
        "schema_version": "5.0",
        "protocol_id": "HAILAR-GAGRU-INDEPENDENT-RANDOM-CENTER-2026-01",
        "status": "FROZEN_BEFORE_RANDOM_CENTER_MODEL_SCREENING",
        "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "research_scope": {
            "task": "within-well supervised random-center interpolation",
            "wells": list(wells),
            "main_well": source_protocol["research_scope"]["main_well"],
            "one_independent_model_per_well": True,
            "samples_shared_between_well_models": False,
            "cross_well_training_or_testing": False,
            "cross_well_generalization_claim": False,
        },
        "classes": {
            "eligibility_rule_inherited_from_v3_v4": True,
            "minimum_rows_per_well": source_protocol["class_eligibility"][
                "minimum_rows_per_well"
            ],
            "minimum_independent_labeled_intervals_per_well": source_protocol[
                "class_eligibility"
            ]["minimum_independent_labeled_intervals_per_well"],
        },
        "features": {
            "core_five": source_protocol["features"]["core_five"],
            "extended_seven": list(features),
            "primary_candidate": "extended_seven",
            "PE": "excluded because it is unavailable in part of the six-well dataset",
            "missing_values": "training-window median only",
        },
        "split": {
            "method": "per-well class-stratified random center assignment",
            "fractions": dict(zip(SPLITS, FRACTIONS, strict=True)),
            "split_seeds": list(SPLIT_SEEDS),
            "primary_model_selection_split_seed": PRIMARY_SPLIT_SEED,
            "target_center_overlap_across_partitions": 0,
            "input_context_overlap_across_partitions": (
                "expected, quantified, and disclosed because the full log trace is observed"
            ),
            "test_role": "held out until all model and search choices are frozen",
        },
        "windows": {
            "length": WINDOW_LENGTH,
            "sampling_interval_m": 0.125,
            "task": "many-to-one center classification",
            "labels_used_only_at_window_centers": True,
            "neighboring_context_contains_curves_only_not_labels": True,
        },
        "model_selection": {
            "primary_metric": "mean per-well supported macro-F1 on validation centers",
            "secondary_metric": "pooled validation accuracy",
            "test_metrics_available_during_selection": False,
            "training_seeds_for_final_comparison": [17, 29, 43],
        },
        "interpretation_limits": {
            "supported": (
                "interpolation of unlabelled centers in a well whose complete log curves "
                "and some lithology labels are available"
            ),
            "unsupported": "zero-shot transfer to an entirely unseen well",
            "not_an_independent_window_test": True,
            "context_overlap_must_be_reported": True,
            "v4_source_row_disjoint_result_role": "sensitivity analysis only",
        },
        "well_protocols": well_records,
    }
    protocol_path = OUTPUT_DIR / "random_center_protocol_v5.json"
    protocol_path.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    files_to_hash.append(protocol_path)
    hash_path = OUTPUT_DIR / "protocol_file_sha256.csv"
    pd.DataFrame(
        [
            {"file": str(path.relative_to(OUTPUT_DIR)), "sha256": file_sha256(path)}
            for path in files_to_hash
        ]
    ).to_csv(hash_path, index=False, encoding="utf-8-sig")

    print("Independent random-center protocol frozen: PASS")
    print("Target centers are disjoint; observed-curve context overlap is disclosed.")
    print("No model was fitted and no test metric was computed.")
    print(f"Protocol: {protocol_path}")


if __name__ == "__main__":
    main()
