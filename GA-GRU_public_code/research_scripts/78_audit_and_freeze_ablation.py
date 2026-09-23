"""Audit validation ablations and freeze preprocessing for final testing."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = (
    PROJECT_DIR
    / "experiment_protocol_v5_random_center"
    / "random_center_protocol_v5.json"
)
OPTIMIZER_FREEZE_PATH = (
    PROJECT_DIR
    / "outputs"
    / "independent_wells_v5"
    / "plain_gru_searches"
    / "audit"
    / "plain_gru_optimizer_winner_freeze.json"
)
OUTPUT_DIR = (
    PROJECT_DIR / "outputs" / "independent_wells_v5" / "validation_ablation"
)
MANIFEST_PATH = OUTPUT_DIR / "ablation_manifest.json"
RUN_LOG_PATH = OUTPUT_DIR / "well_evaluations.jsonl"
STAGE1_FREEZE_PATH = OUTPUT_DIR / "stage1_feature_representation_freeze.json"
COMPLETION_PATH = OUTPUT_DIR / "ablation_completion.json"
WINNER_FREEZE_PATH = OUTPUT_DIR / "ablation_winner_freeze.json"
AUDIT_PATH = OUTPUT_DIR / "audit.json"
STAGE2_STRATEGIES = ("class_weighted", "smote_tomek")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_hash(path: Path) -> None:
    path.with_suffix(path.suffix + ".sha256").write_text(
        sha256_file(path), encoding="ascii"
    )


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)
    write_hash(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)
    write_hash(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSONL line {line_number}: {path}") from exc
    return records


def stage2_configurations(winner: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            **{
                key: value
                for key, value in winner.items()
                if key
                in {
                    "feature_set",
                    "features",
                    "representation",
                    "input_channels",
                }
            },
            "configuration_id": (
                f"{winner['feature_set']}__{winner['representation']}__{strategy}"
            ),
            "stage": "imbalance",
            "imbalance_strategy": strategy,
        }
        for strategy in STAGE2_STRATEGIES
    ]


def summarize(
    configuration: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    frame = pd.DataFrame(records)
    return {
        **configuration,
        "wells": int(len(frame)),
        "validation_samples": int(frame["validation_samples"].sum()),
        "mean_per_well_macro_f1": float(frame["macro_f1"].mean()),
        "mean_per_well_balanced_accuracy": float(
            frame["balanced_accuracy"].mean()
        ),
        "mean_per_well_accuracy": float(frame["accuracy"].mean()),
        "pooled_accuracy": float(
            np.average(frame["accuracy"], weights=frame["validation_samples"])
        ),
        "runtime_seconds": float(frame["runtime_seconds"].sum()),
        "mean_best_epoch": float(frame["best_epoch"].mean()),
    }


def rank(summary: dict[str, Any]) -> tuple[float, float, float, int, str]:
    return (
        float(summary["mean_per_well_macro_f1"]),
        float(summary["mean_per_well_balanced_accuracy"]),
        float(summary["mean_per_well_accuracy"]),
        -int(summary["input_channels"]),
        str(summary["configuration_id"]),
    )


def assert_summary_equal(
    observed: dict[str, Any], expected: dict[str, Any], label: str
) -> None:
    for key in (
        "configuration_id",
        "feature_set",
        "representation",
        "imbalance_strategy",
        "input_channels",
        "wells",
        "validation_samples",
    ):
        if observed[key] != expected[key]:
            raise RuntimeError(f"{label} differs for {key}")
    for key in (
        "mean_per_well_macro_f1",
        "mean_per_well_balanced_accuracy",
        "mean_per_well_accuracy",
        "pooled_accuracy",
        "runtime_seconds",
        "mean_best_epoch",
    ):
        if not math.isclose(
            float(observed[key]), float(expected[key]), rel_tol=0, abs_tol=1e-12
        ):
            raise RuntimeError(f"{label} differs for {key}")


def main() -> None:
    required = (
        PROTOCOL_PATH,
        OPTIMIZER_FREEZE_PATH,
        MANIFEST_PATH,
        RUN_LOG_PATH,
        STAGE1_FREEZE_PATH,
        COMPLETION_PATH,
        OUTPUT_DIR / "mutual_information_per_well.csv",
        OUTPUT_DIR / "mutual_information_summary.csv",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing ablation artifacts: {missing}")

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    optimizer_freeze = json.loads(
        OPTIMIZER_FREEZE_PATH.read_text(encoding="utf-8")
    )
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    stage1_freeze = json.loads(STAGE1_FREEZE_PATH.read_text(encoding="utf-8"))
    completion = json.loads(COMPLETION_PATH.read_text(encoding="utf-8"))
    records = read_jsonl(RUN_LOG_PATH)
    wells = tuple(str(value) for value in manifest["wells"])
    if tuple(protocol["research_scope"]["wells"]) != wells:
        raise RuntimeError("Manifest wells differ from the frozen protocol")
    if manifest["candidate"] != optimizer_freeze["methods"]["ga"]["candidate"]:
        raise RuntimeError("Ablation did not use the frozen GA-GRU candidate")
    if sha256_file(PROTOCOL_PATH) != manifest["protocol_sha256"]:
        raise RuntimeError("Protocol changed after the ablation manifest was frozen")
    if sha256_file(OPTIMIZER_FREEZE_PATH) != manifest["optimizer_freeze_sha256"]:
        raise RuntimeError("Optimizer freeze changed after ablation registration")
    if manifest["test_assignment_files_opened"] is not False:
        raise RuntimeError("Manifest does not certify that test assignments stayed closed")

    stage1 = list(manifest["stage1_configurations"])
    recomputed_stage1 = []
    records_by_configuration: dict[str, list[dict[str, Any]]] = {}
    seen_keys = set()
    for record in records:
        run_key = str(record["run_key"])
        if run_key in seen_keys:
            raise RuntimeError(f"Duplicate run key: {run_key}")
        seen_keys.add(run_key)
        if record["test_assignment_file_opened"] is not False:
            raise RuntimeError(f"Test access was reported for {run_key}")
        if record["preprocessing_fit_scope"] != "training_windows_only":
            raise RuntimeError(f"Invalid preprocessing scope for {run_key}")
        if record["balance_audit"]["resampling_scope"] != "training_windows_only":
            raise RuntimeError(f"Invalid balancing scope for {run_key}")
        history = PROJECT_DIR / str(record["history_file"])
        if not history.is_file():
            raise RuntimeError(f"Missing history for {run_key}")
        records_by_configuration.setdefault(
            str(record["configuration_id"]), []
        ).append(record)

    for configuration in stage1:
        configuration_records = records_by_configuration.get(
            str(configuration["configuration_id"]), []
        )
        if {str(record["well_id"]) for record in configuration_records} != set(wells):
            raise RuntimeError(
                f"Incomplete wells for {configuration['configuration_id']}"
            )
        recomputed_stage1.append(summarize(configuration, configuration_records))
    stage1_winner = max(recomputed_stage1, key=rank)
    assert_summary_equal(
        stage1_winner, stage1_freeze["winner"], "stage-1 winner freeze"
    )
    for observed, expected in zip(
        recomputed_stage1, stage1_freeze["all_stage1_summaries"], strict=True
    ):
        assert_summary_equal(observed, expected, "stage-1 summary")

    stage2 = stage2_configurations(stage1_winner)
    recomputed_stage2 = [stage1_winner]
    for configuration in stage2:
        configuration_records = records_by_configuration.get(
            str(configuration["configuration_id"]), []
        )
        if {str(record["well_id"]) for record in configuration_records} != set(wells):
            raise RuntimeError(
                f"Incomplete wells for {configuration['configuration_id']}"
            )
        recomputed_stage2.append(summarize(configuration, configuration_records))

    all_summaries = recomputed_stage1 + recomputed_stage2[1:]
    expected_configuration_ids = {
        str(summary["configuration_id"]) for summary in all_summaries
    }
    if set(records_by_configuration) != expected_configuration_ids:
        raise RuntimeError("Unexpected configuration exists in the run log")
    if len(records) != int(manifest["expected_well_runs"]):
        raise RuntimeError("Run-log length differs from the frozen expectation")
    final_winner = max(recomputed_stage2, key=rank)
    assert_summary_equal(
        final_winner, completion["stage2_winner"], "completion winner"
    )

    reference_id = "extended_seven__engineered__unweighted"
    reference = next(
        summary
        for summary in recomputed_stage1
        if summary["configuration_id"] == reference_id
    )
    frozen_ga = optimizer_freeze["methods"]["ga"]
    reference_checks = {
        "mean_per_well_macro_f1": "validation_mean_per_well_macro_f1",
        "mean_per_well_accuracy": "validation_mean_per_well_accuracy",
        "pooled_accuracy": "validation_pooled_accuracy",
    }
    for observed_key, frozen_key in reference_checks.items():
        if not math.isclose(
            float(reference[observed_key]),
            float(frozen_ga[frozen_key]),
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"Reference rerun differs from GA freeze for {observed_key}"
            )

    split_seed = str(manifest["split_seed"])
    for record in records:
        well_id = str(record["well_id"])
        expected_validation = int(
            protocol["well_protocols"][well_id]["split_seeds"][split_seed][
                "centers"
            ]["validation"]
        )
        if int(record["validation_samples"]) != expected_validation:
            raise RuntimeError(f"Validation count differs for {record['run_key']}")
        strategy = str(record["imbalance_strategy"])
        weights = record["balance_audit"]["class_weights"]
        if (strategy == "class_weighted") != (weights is not None):
            raise RuntimeError(f"Class-weight audit differs for {record['run_key']}")
        if strategy in {"unweighted", "class_weighted"} and int(
            record["training_samples_before_balance"]
        ) != int(record["training_samples_after_balance"]):
            raise RuntimeError(f"Unexpected sample-count change for {record['run_key']}")

    mi_frame = pd.read_csv(
        OUTPUT_DIR / "mutual_information_per_well.csv", encoding="utf-8-sig"
    )
    mi_summary = pd.read_csv(
        OUTPUT_DIR / "mutual_information_summary.csv", encoding="utf-8-sig"
    )
    expected_features = set(protocol["features"]["extended_seven"])
    if len(mi_frame) != len(wells) * len(expected_features):
        raise RuntimeError("Mutual-information row count is incomplete")
    if set(mi_frame["well_id"].astype(str)) != set(wells):
        raise RuntimeError("Mutual-information wells are incomplete")
    if set(mi_summary["feature"].astype(str)) != expected_features:
        raise RuntimeError("Mutual-information features are incomplete")
    if (mi_frame["mutual_information"] < 0).any():
        raise RuntimeError("Mutual information cannot be negative")

    per_well_frame = pd.DataFrame(records).sort_values(
        ["configuration_id", "well_id"], kind="stable"
    )
    per_well_columns = [
        "configuration_id",
        "stage",
        "feature_set",
        "representation",
        "input_channels",
        "imbalance_strategy",
        "well_id",
        "training_samples_before_balance",
        "training_samples_after_balance",
        "validation_samples",
        "macro_f1",
        "balanced_accuracy",
        "accuracy",
        "best_epoch",
        "epochs_completed",
        "runtime_seconds",
        "trainable_parameters",
        "training_seed",
    ]
    summary_frame = pd.DataFrame(all_summaries).sort_values(
        ["mean_per_well_macro_f1", "mean_per_well_balanced_accuracy"],
        ascending=False,
        kind="stable",
    )
    summary_path = OUTPUT_DIR / "configuration_summary.csv"
    per_well_path = OUTPUT_DIR / "per_well_metrics.csv"
    write_csv(summary_path, summary_frame)
    write_csv(per_well_path, per_well_frame.loc[:, per_well_columns])

    winner_freeze = {
        "status": "FROZEN_AFTER_VALIDATION_ABLATION_BEFORE_FINAL_TESTING",
        "selection_uses": "training_and_validation centers only",
        "test_assignment_files_opened": False,
        "test_metrics_used_for_selection": False,
        "model": "unidirectional_many_to_one_GRU",
        "optimizer": "genetic_algorithm",
        "candidate": manifest["candidate"],
        "selected_configuration": final_winner,
        "feature_transform": (
            "log10 is applied to MSFL, LLS, and LLD before per-well "
            "training-window standardization"
        ),
        "engineered_definition": (
            "three resistivity separations and first differences of each physical "
            "channel; applicable only when representation is engineered"
        ),
        "final_split_seeds": optimizer_freeze["final_split_seeds"],
        "final_training_seeds": optimizer_freeze["final_training_seeds"],
        "final_refit_rule": (
            "select the epoch on train/validation, then refit from scratch for that "
            "epoch count on their union; compute class weights or SMOTE-Tomek only "
            "from the refit set; score each untouched test-center set once"
        ),
        "interpretation": (
            "within-well supervised interpolation with shared observed curve context; "
            "not unseen-well generalization"
        ),
    }
    write_json(WINNER_FREEZE_PATH, winner_freeze)

    audit = {
        "status": "PASS",
        "well_runs": len(records),
        "unique_configurations": len(all_summaries),
        "six_independent_well_tasks": True,
        "samples_shared_between_well_models": False,
        "training_validation_only_selection": True,
        "test_assignment_files_opened": False,
        "test_metrics_used_for_selection": False,
        "frozen_ga_reference_exactly_reproduced": True,
        "stage1_selection_recomputed": True,
        "stage2_selection_recomputed": True,
        "training_only_mutual_information_verified": True,
        "training_only_imbalance_handling_verified": True,
        "selected_configuration_id": final_winner["configuration_id"],
        "artifacts": {
            "configuration_summary": sha256_file(summary_path),
            "per_well_metrics": sha256_file(per_well_path),
            "winner_freeze": sha256_file(WINNER_FREEZE_PATH),
            "run_log": sha256_file(RUN_LOG_PATH),
            "mutual_information_per_well": sha256_file(
                OUTPUT_DIR / "mutual_information_per_well.csv"
            ),
            "mutual_information_summary": sha256_file(
                OUTPUT_DIR / "mutual_information_summary.csv"
            ),
        },
    }
    write_json(AUDIT_PATH, audit)
    print("Validation ablation audit: PASS")
    print(summary_frame.to_string(index=False))
    print(f"Frozen configuration: {final_winner['configuration_id']}")
    print(f"Winner freeze: {WINNER_FREEZE_PATH}")


if __name__ == "__main__":
    main()
