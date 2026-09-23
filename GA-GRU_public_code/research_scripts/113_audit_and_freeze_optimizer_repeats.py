"""Audit the development data and freeze repeated optimizer searches.

This step does not alter any source CSV or inspect any test metric. It records
the evidence-supported pipeline that GA, random search, and TPE must share.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_PROTOCOL_DIR = PROJECT_DIR / "experiment_protocol_v5_random_center"
SOURCE_PROTOCOL_PATH = SOURCE_PROTOCOL_DIR / "random_center_protocol_v5.json"
EXTENDED_AUDIT_DIR = (
    PROJECT_DIR / "outputs" / "extended_curves" / "data_audit_v1"
)
DATA_REPORT_DIR = (
    PROJECT_DIR.parent / "测井文件集" / "整理后数据_20260910" / "00_依据与报告"
)
PROTOCOL_DIR = PROJECT_DIR / "experiment_protocol_v8_optimizer_repeats"
PROTOCOL_PATH = PROTOCOL_DIR / "optimizer_repeat_protocol_v8.json"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "optimizer_repeats_v1" / "data_audit"

FEATURES = ("MSFL", "LLS", "LLD", "DEN", "DT", "GR", "NPHI")
SEARCH_BASE_SEEDS = (17, 29, 43)
SEARCH_BUDGET = 24
INITIAL_CANDIDATES = 6

sys.path.insert(0, str(PROJECT_DIR))

from gagru.search import GRUSearchSpace  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text_atomic(path: Path, value: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding=encoding)
    os.replace(temporary, path)


def write_json_atomic(path: Path, value: object) -> None:
    write_text_atomic(
        path,
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def search_space() -> GRUSearchSpace:
    return GRUSearchSpace(
        hidden_sizes=tuple(range(64, 161, 16)),
        num_layers=(1, 2, 3),
        learning_rate_min=3e-4,
        learning_rate_max=4e-3,
        dropout_values=(0.0, 0.1, 0.2, 0.3),
        weight_decay_min=1e-6,
        weight_decay_max=3e-4,
        force_zero_dropout_for_one_layer=True,
    )


def assignment_path(well_id: str, split: str, split_seed: int) -> Path:
    return (
        SOURCE_PROTOCOL_DIR
        / "center_assignments"
        / f"seed_{split_seed}"
        / well_id
        / f"{well_id}_{split}_centers.csv"
    )


def staged_extended_path(well_id: str) -> Path:
    return (
        EXTENDED_AUDIT_DIR
        / "staged_labeled_development_and_supplementary"
        / f"{well_id}_五曲线加GR_NPHI_PE.csv"
    )


def finite_summary(values: pd.Series) -> dict[str, float | int | None]:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    finite = numeric[np.isfinite(numeric)]
    if not len(finite):
        return {
            "finite": 0,
            "missing": int(len(numeric)),
            "minimum": None,
            "q01": None,
            "median": None,
            "q99": None,
            "maximum": None,
        }
    q01, median, q99 = np.quantile(finite, [0.01, 0.50, 0.99])
    return {
        "finite": int(len(finite)),
        "missing": int(len(numeric) - len(finite)),
        "minimum": float(np.min(finite)),
        "q01": float(q01),
        "median": float(median),
        "q99": float(q99),
        "maximum": float(np.max(finite)),
    }


def compare_staged_values(frame: pd.DataFrame, well_id: str) -> dict[str, Any]:
    path = staged_extended_path(well_id)
    if not path.is_file():
        raise RuntimeError(f"Missing staged seven-curve source: {path}")
    staged = pd.read_csv(path, encoding="utf-8-sig")
    left = frame.loc[:, ["depth", *FEATURES]].copy()
    right = staged.loc[:, ["depth", *FEATURES]].copy()
    merged = left.merge(
        right,
        on="depth",
        how="left",
        suffixes=("_frozen", "_staged"),
        validate="one_to_one",
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        raise RuntimeError(f"{well_id} frozen rows are not a subset of staged rows")
    mismatches: dict[str, int] = {}
    for feature in FEATURES:
        frozen = pd.to_numeric(merged[f"{feature}_frozen"], errors="coerce").to_numpy(float)
        source = pd.to_numeric(merged[f"{feature}_staged"], errors="coerce").to_numpy(float)
        equal = np.isclose(frozen, source, rtol=0.0, atol=1e-12, equal_nan=True)
        mismatches[feature] = int((~equal).sum())
    return {
        "staged_file": str(path.relative_to(PROJECT_DIR)),
        "staged_file_sha256": sha256_file(path),
        "matched_rows": int(len(merged)),
        "value_mismatches": mismatches,
    }


def audit_well(
    protocol: dict[str, Any], well_id: str, split_seed: int
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    record = protocol["well_protocols"][well_id]
    source_path = SOURCE_PROTOCOL_DIR / str(record["source_snapshot"])
    if not source_path.is_file():
        raise RuntimeError(f"Missing frozen source snapshot: {source_path}")
    actual_hash = sha256_file(source_path)
    if actual_hash != str(record["source_snapshot_sha256"]):
        raise RuntimeError(f"Frozen source hash changed for {well_id}")
    frame = pd.read_csv(source_path, encoding="utf-8-sig")
    required = {
        "well_id",
        "depth",
        "class_id",
        "segment_id",
        "source_row_id",
        *FEATURES,
    }
    missing_columns = sorted(required.difference(frame.columns))
    if missing_columns:
        raise RuntimeError(f"{well_id} lacks columns: {missing_columns}")
    if len(frame) != int(record["source_rows"]):
        raise RuntimeError(f"Frozen row count changed for {well_id}")
    if frame["source_row_id"].duplicated().any():
        raise RuntimeError(f"Duplicate source_row_id in {well_id}")
    if frame["depth"].duplicated().any():
        raise RuntimeError(f"Duplicate depth in {well_id}")
    if set(frame["well_id"].astype(str)) != {well_id}:
        raise RuntimeError(f"Unexpected well_id value in {well_id}")
    observed_classes = sorted(frame["class_id"].astype(int).unique().tolist())
    expected_classes = sorted(int(value) for value in record["included_classes"])
    if observed_classes != expected_classes:
        raise RuntimeError(f"Class coverage changed for {well_id}")
    resistivity = frame.loc[:, ["MSFL", "LLS", "LLD"]].to_numpy(float)
    if not np.isfinite(resistivity).all() or np.any(resistivity <= 0):
        raise RuntimeError(f"{well_id} contains invalid resistivity values")

    discontinuities = 0
    nonmonotonic_segments = 0
    for _, segment in frame.groupby("segment_id", sort=False):
        depths = segment["depth"].to_numpy(float)
        differences = np.diff(depths)
        nonmonotonic_segments += int(np.any(differences <= 0))
        discontinuities += int((~np.isclose(differences, 0.125, atol=1e-9)).sum())
    if nonmonotonic_segments or discontinuities:
        raise RuntimeError(f"Depth continuity failed for {well_id}")

    train_path = assignment_path(well_id, "train", split_seed)
    validation_path = assignment_path(well_id, "validation", split_seed)
    train = pd.read_csv(train_path, encoding="utf-8-sig")
    validation = pd.read_csv(validation_path, encoding="utf-8-sig")
    if set(train["source_row_id"]).intersection(validation["source_row_id"]):
        raise RuntimeError(f"Train/validation centers overlap for {well_id}")
    indexed_labels = frame.set_index("source_row_id")["class_id"].astype(int)
    for split_name, assignment in (("train", train), ("validation", validation)):
        assigned_ids = assignment["source_row_id"].astype(np.int64)
        if not set(assigned_ids).issubset(set(indexed_labels.index.astype(int))):
            raise RuntimeError(f"Unknown {split_name} center in {well_id}")
        observed = indexed_labels.loc[assigned_ids].to_numpy(int)
        if not np.array_equal(observed, assignment["class_id"].to_numpy(int)):
            raise RuntimeError(f"Frozen {split_name} labels changed for {well_id}")

    staged_check = compare_staged_values(frame, well_id)
    if any(staged_check["value_mismatches"].values()):
        raise RuntimeError(f"Seven-curve values differ from staged source for {well_id}")

    feature_rows: list[dict[str, Any]] = []
    for feature in FEATURES:
        feature_rows.append(
            {"well_id": well_id, "feature": feature, **finite_summary(frame[feature])}
        )

    pair_rows: list[dict[str, Any]] = []
    for first_index, first in enumerate(FEATURES):
        for second in FEATURES[first_index + 1 :]:
            values = frame.loc[:, [first, second]].apply(
                pd.to_numeric, errors="coerce"
            )
            valid = values.notna().all(axis=1)
            first_values = values.loc[valid, first].to_numpy(float)
            second_values = values.loc[valid, second].to_numpy(float)
            exact_fraction = (
                float(np.mean(first_values == second_values)) if len(first_values) else None
            )
            correlation = (
                float(np.corrcoef(first_values, second_values)[0, 1])
                if len(first_values) > 1
                and np.std(first_values) > 0
                and np.std(second_values) > 0
                else None
            )
            pair_rows.append(
                {
                    "well_id": well_id,
                    "first_feature": first,
                    "second_feature": second,
                    "common_finite_rows": int(valid.sum()),
                    "exact_equal_fraction": exact_fraction,
                    "pearson_r": correlation,
                }
            )

    summary = {
        "well_id": well_id,
        "rows": int(len(frame)),
        "segments": int(frame["segment_id"].nunique()),
        "classes": observed_classes,
        "source_snapshot": str(source_path.relative_to(PROJECT_DIR)),
        "source_snapshot_sha256": actual_hash,
        "train_centers": int(len(train)),
        "validation_centers": int(len(validation)),
        "train_assignment_sha256": sha256_file(train_path),
        "validation_assignment_sha256": sha256_file(validation_path),
        "feature_missing_values": {
            feature: int(pd.to_numeric(frame[feature], errors="coerce").isna().sum())
            for feature in FEATURES
        },
        "gr_zero_rows": int((pd.to_numeric(frame["GR"], errors="coerce") == 0).sum()),
        "gr_above_300_rows": int(
            (pd.to_numeric(frame["GR"], errors="coerce") > 300).sum()
        ),
        "nphi_outside_minus_0_2_to_1_2_rows": int(
            (
                (pd.to_numeric(frame["NPHI"], errors="coerce") < -0.2)
                | (pd.to_numeric(frame["NPHI"], errors="coerce") > 1.2)
            ).sum()
        ),
        "staged_source_check": staged_check,
    }
    return summary, feature_rows, pair_rows


def make_shared_initial_candidates(
    space: GRUSearchSpace, base_seed: int
) -> list[dict[str, int | float]]:
    rng = np.random.default_rng(base_seed + 1_000_000)
    candidates = []
    seen: set[str] = set()
    while len(candidates) < INITIAL_CANDIDATES:
        candidate = space.sample(rng)
        if candidate.key in seen:
            continue
        seen.add(candidate.key)
        candidates.append(candidate.to_dict())
    return candidates


def main() -> None:
    protocol = json.loads(SOURCE_PROTOCOL_PATH.read_text(encoding="utf-8"))
    wells = tuple(str(value) for value in protocol["research_scope"]["wells"])
    if len(wells) != 6:
        raise RuntimeError("Optimizer repeats require the frozen six-well task")
    if tuple(protocol["features"]["extended_seven"]) != FEATURES:
        raise RuntimeError("Frozen seven-curve order changed")
    split_seed = int(protocol["split"]["primary_model_selection_split_seed"])

    well_audits: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for well_id in wells:
        well_audit, well_features, well_pairs = audit_well(
            protocol, well_id, split_seed
        )
        well_audits.append(well_audit)
        feature_rows.extend(well_features)
        pair_rows.extend(well_pairs)

    suspicious_exact_pairs = [
        row
        for row in pair_rows
        if row["exact_equal_fraction"] is not None
        and float(row["exact_equal_fraction"]) >= 0.999
    ]
    prior_validation_path = DATA_REPORT_DIR / "自动验证结果.csv"
    if not prior_validation_path.is_file():
        raise RuntimeError(f"Missing prior source-data validation: {prior_validation_path}")
    prior_validation = pd.read_csv(prior_validation_path, encoding="utf-8-sig")
    if not (prior_validation["结果"].astype(str) == "通过").all():
        raise RuntimeError("Prior source-data validation contains a failure")

    audit = {
        "status": "PASS_WITH_DOCUMENTED_RAW_VALUE_FLAGS",
        "audited_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_csv_files_modified": False,
        "labels_modified": False,
        "rows_removed": False,
        "test_assignment_files_opened": False,
        "test_metrics_read": False,
        "wells": list(wells),
        "features": list(FEATURES),
        "well_audits": well_audits,
        "suspicious_near_identical_feature_pairs": suspicious_exact_pairs,
        "documented_flags": {
            "GR_equal_zero_rows": int(sum(row["gr_zero_rows"] for row in well_audits)),
            "GR_above_300_rows": int(
                sum(row["gr_above_300_rows"] for row in well_audits)
            ),
            "NPHI_outside_minus_0_2_to_1_2_rows": int(
                sum(row["nphi_outside_minus_0_2_to_1_2_rows"] for row in well_audits)
            ),
            "action": (
                "Retain values. The source-depth audit proves provenance, and no "
                "independent correction rule establishes that these values are errors."
            ),
        },
        "prior_source_validation_file": str(prior_validation_path),
        "prior_source_validation_sha256": sha256_file(prior_validation_path),
        "decision": (
            "No result-directed cleaning is warranted. Use the frozen seven-curve "
            "snapshots, training-only imputation and scaling, and training-only "
            "SMOTE-Tomek in every candidate evaluation."
        ),
    }

    space = search_space()
    repeats = []
    for repeat_id, base_seed in enumerate(SEARCH_BASE_SEEDS, start=1):
        repeats.append(
            {
                "repeat_id": repeat_id,
                "base_search_seed": base_seed,
                "shared_initial_generation_seed": base_seed + 1_000_000,
                "method_search_seed": base_seed,
                "shared_initial_candidates": make_shared_initial_candidates(
                    space, base_seed
                ),
            }
        )

    source_hashes = {
        str(Path(row["source_snapshot"])): row["source_snapshot_sha256"]
        for row in well_audits
    }
    protocol_v8 = {
        "schema_version": "8.0",
        "protocol_id": "HAILAR-GAGRU-OPTIMIZER-REPEATS-2026-01",
        "status": "FROZEN_BEFORE_REPEATED_OPTIMIZER_SEARCH",
        "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "purpose": (
            "Compare independent GA, random-search, and TPE searches under the "
            "same complete validation pipeline; quantify stability, cost, and "
            "configuration variability without choosing data from outcomes."
        ),
        "source_protocol": str(SOURCE_PROTOCOL_PATH.relative_to(PROJECT_DIR)),
        "source_protocol_sha256": sha256_file(SOURCE_PROTOCOL_PATH),
        "data_audit": str((OUTPUT_DIR / "data_audit.json").relative_to(PROJECT_DIR)),
        "research_task": "within-well supervised random-center interpolation",
        "wells": list(wells),
        "one_independent_model_per_well": True,
        "features": list(FEATURES),
        "pe_policy": (
            "Excluded: PE gave negligible and inconsistent validation change, and "
            "reliable source PE is absent in three available wells."
        ),
        "window_length": 9,
        "context_span_m": 1.0,
        "representation": "engineered",
        "representation_definition": (
            "log10(MSFL, LLS, LLD), three resistivity separations, and first "
            "differences of all seven physical channels"
        ),
        "preprocessing": (
            "Imputation medians and per-well standardization parameters are fitted "
            "on training windows only; validation windows are transform-only."
        ),
        "imbalance_strategy": "smote_tomek",
        "imbalance_scope": "training windows only",
        "balance_seed_by_well": {
            well_id: 27101 + index for index, well_id in enumerate(wells)
        },
        "split_seed": split_seed,
        "training_seed_by_well": {
            well_id: 1701 + index for index, well_id in enumerate(wells)
        },
        "model": "unidirectional many-to-one GRU",
        "batch_size": 512,
        "maximum_epochs": 60,
        "early_stopping_patience": 8,
        "selection_metric": "unweighted mean of six per-well supported macro-F1 values",
        "ranking_tiebreaks": [
            "mean_per_well_balanced_accuracy",
            "mean_per_well_accuracy",
            "fewer_total_parameters",
            "shorter_charged_training_runtime",
        ],
        "methods": ["genetic_algorithm", "random_search", "tpe"],
        "candidate_budget_per_method_per_repeat": SEARCH_BUDGET,
        "shared_initial_candidates_per_repeat": INITIAL_CANDIDATES,
        "shared_initial_candidates_count_toward_budget": True,
        "search_space": space.to_dict(),
        "ga": {
            "population_size": 6,
            "tournament_size": 3,
            "crossover_probability": 0.8,
            "per_gene_mutation_probability": 0.2,
            "elites": 2,
        },
        "tpe": {"startup_candidates": INITIAL_CANDIDATES},
        "repeats": repeats,
        "deterministic_candidate_evaluation": True,
        "cache_policy": (
            "An identical candidate under the identical frozen data, balance, and "
            "training seeds may reuse its deterministic evaluation. Its original "
            "training runtime is charged to every method for fair cost comparison."
        ),
        "reporting": [
            "best validation macro-F1 distribution",
            "paired repeat-level differences",
            "best-so-far values at candidates 6, 12, 18, and 24",
            "charged training runtime",
            "winning configuration variability",
        ],
        "inference_restrictions": (
            "Three repeats support descriptive stability claims only. Do not claim "
            "statistical superiority solely from n=3 searches."
        ),
        "test_policy": (
            "No test assignment, prediction, or metric may be read by the repeated "
            "search or its analysis. Existing final test results are not used to "
            "choose data, search space, candidates, or winners."
        ),
        "data_decisions": {
            "source_csv_files_modified": False,
            "labels_modified": False,
            "rows_removed": False,
            "gr_flags_retained": True,
            "boundary_rows_retained": True,
            "justification": (
                "No independently verified error supports changing these rows; "
                "boundary difficulty is a geological result rather than a cleaning target."
            ),
        },
        "source_snapshot_sha256": source_hashes,
        "code_sha256": {
            "113_audit_and_freeze_optimizer_repeats.py": sha256_file(Path(__file__)),
            "114_run_repeated_optimizer_searches.py": sha256_file(
                PROJECT_DIR / "114_run_repeated_optimizer_searches.py"
            ),
            "115_analyze_repeated_optimizer_searches.py": sha256_file(
                PROJECT_DIR / "115_analyze_repeated_optimizer_searches.py"
            ),
            "gagru/random_center.py": sha256_file(
                PROJECT_DIR / "gagru" / "random_center.py"
            ),
            "gagru/random_center_ablation.py": sha256_file(
                PROJECT_DIR / "gagru" / "random_center_ablation.py"
            ),
            "gagru/residual_bigru.py": sha256_file(
                PROJECT_DIR / "gagru" / "residual_bigru.py"
            ),
            "gagru/local_recurrent.py": sha256_file(
                PROJECT_DIR / "gagru" / "local_recurrent.py"
            ),
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json_atomic(OUTPUT_DIR / "data_audit.json", audit)
    write_csv_atomic(OUTPUT_DIR / "well_audit.csv", pd.DataFrame(well_audits))
    write_csv_atomic(OUTPUT_DIR / "feature_distributions.csv", pd.DataFrame(feature_rows))
    write_csv_atomic(OUTPUT_DIR / "feature_pair_checks.csv", pd.DataFrame(pair_rows))
    write_json_atomic(PROTOCOL_PATH, protocol_v8)
    write_text_atomic(
        PROTOCOL_PATH.with_suffix(".sha256"),
        sha256_file(PROTOCOL_PATH) + "\n",
        encoding="ascii",
    )

    print("Development-data audit: PASS WITH DOCUMENTED RAW-VALUE FLAGS")
    print(f"Wells: {len(wells)}")
    print(f"Rows: {sum(row['rows'] for row in well_audits)}")
    print(f"Source CSVs modified: {audit['source_csv_files_modified']}")
    print(f"Labels modified: {audit['labels_modified']}")
    print(f"GR=0 retained rows: {audit['documented_flags']['GR_equal_zero_rows']}")
    print(f"GR>300 retained rows: {audit['documented_flags']['GR_above_300_rows']}")
    print(f"Frozen repeated-search protocol: {PROTOCOL_PATH}")
    print("Next: run 114_run_repeated_optimizer_searches.py")


if __name__ == "__main__":
    main()
