"""Freeze the complete-interval, zero-context-overlap sensitivity protocol."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_PROTOCOL_DIR = PROJECT_DIR / "experiment_protocol_v5_random_center"
SOURCE_PROTOCOL_PATH = SOURCE_PROTOCOL_DIR / "random_center_protocol_v5.json"
OUTPUT_DIR = PROJECT_DIR / "experiment_protocol_v7_strict_interval"

WINDOW_LENGTH = 9
SAMPLING_INTERVAL_M = 0.125
PURGE_EACH_SIDE_M = 1.0
FRACTIONS = (0.64, 0.16, 0.20)
SPLIT_SEEDS = (20260917, 20260918, 20260919)
MAIN_WELL_TRIALS = 32768
REPLICATION_WELL_TRIALS = 4096
FAMILY_MAP = (0, 1, 1, 1, 0, 0, 2, 1, 2, 1)

sys.path.insert(0, str(PROJECT_DIR))

from gagru.data import file_sha256  # noqa: E402
from gagru.strict_interval import (  # noqa: E402
    SPLITS,
    add_complete_interval_ids,
    audit_strict_partition_isolation,
    materialize_strict_candidate,
    search_strict_interval_split,
)


def main() -> None:
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise RuntimeError(
            f"Protocol directory is not empty; refusing to overwrite: {OUTPUT_DIR}"
        )
    source_protocol = json.loads(SOURCE_PROTOCOL_PATH.read_text(encoding="utf-8"))
    wells = tuple(str(value) for value in source_protocol["research_scope"]["wells"])
    main_well = str(source_protocol["research_scope"]["main_well"])
    features = tuple(
        str(value) for value in source_protocol["features"]["extended_seven"]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    source_root = OUTPUT_DIR / "source_snapshots"
    partition_root = OUTPUT_DIR / "partitions"
    assignment_root = OUTPUT_DIR / "interval_assignments"
    guard_root = OUTPUT_DIR / "guard_rows"
    boundary_root = OUTPUT_DIR / "split_boundaries"
    for path in (
        source_root,
        partition_root,
        assignment_root,
        guard_root,
        boundary_root,
    ):
        path.mkdir()

    files_to_hash: list[Path] = []
    well_records: dict[str, object] = {}
    count_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []
    overlap_records: list[dict[str, object]] = []

    for well_id in wells:
        source_record = source_protocol["well_protocols"][well_id]
        source_path = SOURCE_PROTOCOL_DIR / str(source_record["source_snapshot"])
        if file_sha256(source_path) != str(source_record["source_snapshot_sha256"]):
            raise RuntimeError(f"Frozen v5 source snapshot changed for {well_id}")
        raw = pd.read_csv(source_path, encoding="utf-8-sig")
        strict_frame = add_complete_interval_ids(
            raw, sampling_interval_m=SAMPLING_INTERVAL_M
        )
        strict_source_path = source_root / f"{well_id}_strict_source.csv"
        strict_frame.to_csv(strict_source_path, index=False, encoding="utf-8-sig")
        files_to_hash.append(strict_source_path)

        classes = tuple(int(value) for value in source_record["included_classes"])
        trials = MAIN_WELL_TRIALS if well_id == main_well else REPLICATION_WELL_TRIALS
        seed_records: dict[str, object] = {}
        print(
            f"Freezing {well_id}: rows={len(strict_frame)}, classes={classes}, "
            f"complete_intervals={strict_frame['lithology_interval_id'].nunique()}",
            flush=True,
        )
        for seed in SPLIT_SEEDS:
            candidate = search_strict_interval_split(
                strict_frame,
                seed=seed,
                required_classes=classes,
                window_length=WINDOW_LENGTH,
                fractions=FRACTIONS,
                purge_each_side_m=PURGE_EACH_SIDE_M,
                sampling_interval_m=SAMPLING_INTERVAL_M,
                trials=trials,
                minimum_train_windows_per_class=2,
            )
            partitions, purged, boundaries, assignment = materialize_strict_candidate(
                strict_frame,
                candidate,
                purge_each_side_m=PURGE_EACH_SIDE_M,
            )
            audit = audit_strict_partition_isolation(
                partitions,
                window_length=WINDOW_LENGTH,
                purge_each_side_m=PURGE_EACH_SIDE_M,
            )

            seed_partition_dir = partition_root / f"seed_{seed}" / well_id
            seed_partition_dir.mkdir(parents=True)
            partition_hashes: dict[str, str] = {}
            total_windows = int(
                candidate.window_counts.loc[:, list(classes)].to_numpy().sum()
            )
            for split in SPLITS:
                path = seed_partition_dir / f"{well_id}_{split}.csv"
                partitions[split].to_csv(path, index=False, encoding="utf-8-sig")
                files_to_hash.append(path)
                partition_hashes[split] = file_sha256(path)
                windows = candidate.window_counts.loc[split, list(classes)]
                rows = candidate.row_counts.loc[split, list(classes)]
                for class_id in classes:
                    count_records.append(
                        {
                            "split_seed": seed,
                            "well_id": well_id,
                            "split": split,
                            "class_id": class_id,
                            "retained_rows": int(rows.loc[class_id]),
                            "windows": int(windows.loc[class_id]),
                        }
                    )
                split_windows = int(windows.sum())
                summary_records.append(
                    {
                        "split_seed": seed,
                        "well_id": well_id,
                        "split": split,
                        "retained_rows": int(len(partitions[split])),
                        "windows": split_windows,
                        "window_fraction": split_windows / max(1, total_windows),
                        "assigned_complete_intervals": int(
                            assignment["split"].eq(split).sum()
                        ),
                        "supported_classes": int((windows > 0).sum()),
                        "modeled_classes": len(classes),
                    }
                )

            seed_assignment_dir = assignment_root / f"seed_{seed}"
            seed_assignment_dir.mkdir(exist_ok=True)
            assignment_path = seed_assignment_dir / f"{well_id}_interval_assignment.csv"
            assignment.to_csv(assignment_path, index=False, encoding="utf-8-sig")
            files_to_hash.append(assignment_path)

            seed_guard_dir = guard_root / f"seed_{seed}"
            seed_guard_dir.mkdir(exist_ok=True)
            guard_path = seed_guard_dir / f"{well_id}_guard_rows.csv"
            purged.to_csv(guard_path, index=False, encoding="utf-8-sig")
            files_to_hash.append(guard_path)

            seed_boundary_dir = boundary_root / f"seed_{seed}"
            seed_boundary_dir.mkdir(exist_ok=True)
            boundary_path = seed_boundary_dir / f"{well_id}_boundaries.csv"
            boundaries.to_csv(boundary_path, index=False, encoding="utf-8-sig")
            files_to_hash.append(boundary_path)

            for pair_name, pair in audit["pairs"].items():
                overlap_records.append(
                    {
                        "split_seed": seed,
                        "well_id": well_id,
                        "partition_pair": pair_name,
                        **pair,
                    }
                )
            seed_records[str(seed)] = {
                "search_trials": trials,
                "selected_trial": candidate.trial,
                "label_depth_objective": candidate.objective,
                "missing_evaluation_class_slots": (
                    candidate.missing_evaluation_classes
                ),
                "split_boundaries": candidate.split_boundaries,
                "purged_rows": candidate.purged_rows,
                "retained_rows": candidate.retained_rows,
                "partition_sha256": partition_hashes,
                "assignment_file": str(assignment_path.relative_to(OUTPUT_DIR)),
                "assignment_sha256": file_sha256(assignment_path),
                "guard_file": str(guard_path.relative_to(OUTPUT_DIR)),
                "guard_sha256": file_sha256(guard_path),
                "boundary_file": str(boundary_path.relative_to(OUTPUT_DIR)),
                "boundary_sha256": file_sha256(boundary_path),
                "windows": {
                    split: int(candidate.window_counts.loc[split, list(classes)].sum())
                    for split in SPLITS
                },
                "supported_classes": {
                    split: [
                        class_id
                        for class_id in classes
                        if int(candidate.window_counts.loc[split, class_id]) > 0
                    ]
                    for split in SPLITS
                },
                "isolation_audit": audit,
            }
            print(
                f"  seed={seed}: windows={seed_records[str(seed)]['windows']}, "
                f"supported={seed_records[str(seed)]['supported_classes']}, "
                f"guard_rows={candidate.purged_rows}",
                flush=True,
            )

        well_records[well_id] = {
            "role": source_record["role"],
            "included_classes": list(classes),
            "source_snapshot": str(strict_source_path.relative_to(OUTPUT_DIR)),
            "source_snapshot_sha256": file_sha256(strict_source_path),
            "source_rows": int(len(strict_frame)),
            "complete_lithology_intervals": int(
                strict_frame["lithology_interval_id"].nunique()
            ),
            "split_seeds": seed_records,
        }

    artifact_frames = {
        "strict_interval_class_counts.csv": pd.DataFrame(count_records),
        "strict_partition_summary.csv": pd.DataFrame(summary_records),
        "strict_overlap_audit.csv": pd.DataFrame(overlap_records),
    }
    for name, artifact in artifact_frames.items():
        path = OUTPUT_DIR / name
        artifact.to_csv(path, index=False, encoding="utf-8-sig")
        files_to_hash.append(path)

    protocol = {
        "schema_version": "7.0",
        "protocol_id": "HAILAR-GAGRU-STRICT-COMPLETE-INTERVAL-2026-02",
        "status": "FROZEN_BEFORE_STRICT_MODEL_SELECTION",
        "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_protocol": str(SOURCE_PROTOCOL_PATH),
        "source_protocol_sha256": file_sha256(SOURCE_PROTOCOL_PATH),
        "research_scope": {
            "task": "strict within-well complete-interval sensitivity analysis",
            "wells": list(wells),
            "main_well": main_well,
            "one_independent_model_per_well": True,
            "samples_shared_between_well_models": False,
            "cross_well_training_or_testing": False,
            "cross_well_generalization_claim": False,
        },
        "features": {
            "curves": list(features),
            "missing_values": "training-partition median only",
            "resistivity_transform_and_scaling": "fit on training windows only",
            "feature_selection_uses_test": False,
        },
        "split": {
            "method": "per-well class-stratified complete labeled intervals",
            "target_fractions": dict(zip(SPLITS, FRACTIONS, strict=True)),
            "split_seeds": list(SPLIT_SEEDS),
            "selection_inputs": "labels, interval identities, and depths only",
            "selection_uses_model_performance": False,
            "purge_each_side_m": PURGE_EACH_SIDE_M,
            "minimum_training_windows_per_modeled_class": 2,
            "entire_interval_assigned_to_one_partition": True,
            "test_role": "locked until all strict model and search choices are frozen",
        },
        "windows": {
            "length": WINDOW_LENGTH,
            "sampling_interval_m": SAMPLING_INTERVAL_M,
            "task": "many-to-one center classification",
            "constructed_only_after_partition_and_purge": True,
            "cross_partition_windows": False,
        },
        "required_audits": {
            "target_center_overlap": 0,
            "source_row_overlap": 0,
            "input_context_overlap": 0,
            "complete_interval_overlap": 0,
        },
        "class_reporting": {
            "global_classes": list(range(10)),
            "per_well_metrics": "supported classes present in the evaluated partition",
            "pooled_metrics": "all observed global classes",
            "family_map_by_global_class": {
                str(class_id): int(family_id)
                for class_id, family_id in enumerate(FAMILY_MAP)
            },
            "family_names": {
                "0": "mud-rich",
                "1": "sandstone-clastic",
                "2": "tuffaceous",
            },
        },
        "model_selection": {
            "primary_metric": "mean per-well supported macro-F1 on validation intervals",
            "secondary_metric": "pooled validation accuracy",
            "test_metrics_available_during_selection": False,
            "final_training_seeds": [17, 29, 43],
        },
        "interpretation_limits": {
            "role": "strict sensitivity analysis complementing the primary random-center interpolation result",
            "supported": "generalization to held-out complete labeled intervals within the same well",
            "unsupported": "zero-shot transfer to an entirely unseen well",
            "not_directly_comparable_to_random_center_accuracy": True,
        },
        "well_protocols": well_records,
    }
    protocol_path = OUTPUT_DIR / "strict_interval_protocol_v7.json"
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

    print("Strict complete-interval protocol frozen: PASS")
    print("Target-center, source-row, input-context, and interval overlap: 0")
    print("No model was fitted and no test metric was computed.")
    print(f"Protocol: {protocol_path}")


if __name__ == "__main__":
    main()
