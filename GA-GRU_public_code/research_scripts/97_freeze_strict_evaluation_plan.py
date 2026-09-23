"""Freeze the strict sensitivity model plan before opening test partitions."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = (
    PROJECT_DIR
    / "experiment_protocol_v7_strict_interval"
    / "strict_interval_protocol_v7.json"
)
PREFLIGHT_PATH = (
    PROJECT_DIR
    / "outputs"
    / "independent_wells_v7_strict_interval"
    / "preflight"
    / "protocol_validation.json"
)
PRIMARY_PLAN_PATH = (
    PROJECT_DIR
    / "outputs"
    / "independent_wells_v5"
    / "final_evaluation"
    / "final_evaluation_plan.json"
)
OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "independent_wells_v7_strict_interval"
    / "final_evaluation"
)
PLAN_PATH = OUTPUT_DIR / "strict_evaluation_plan.json"
RUNNER_PATH = PROJECT_DIR / "98_run_strict_repeated_evaluation.py"
ANALYSIS_PATH = PROJECT_DIR / "99_analyze_strict_evaluation.py"


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


def main() -> None:
    required = (
        PROTOCOL_PATH,
        PREFLIGHT_PATH,
        PRIMARY_PLAN_PATH,
        RUNNER_PATH,
        ANALYSIS_PATH,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Cannot freeze strict evaluation; missing: {missing}")

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    preflight = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
    primary_plan = json.loads(PRIMARY_PLAN_PATH.read_text(encoding="utf-8"))
    if protocol["status"] != "FROZEN_BEFORE_STRICT_MODEL_SELECTION":
        raise RuntimeError("Strict protocol is not frozen")
    if preflight["status"] != "PASS" or preflight["test_metrics_computed"]:
        raise RuntimeError("Strict preflight is invalid or contains test metrics")
    if primary_plan["status"] != "FROZEN_BEFORE_ANY_FINAL_TEST_ASSIGNMENT_IS_OPENED":
        raise RuntimeError("Primary model plan is not the pre-test freeze")

    models = primary_plan["models"]
    wells = [str(value) for value in protocol["research_scope"]["wells"]]
    split_seeds = [int(value) for value in protocol["split"]["split_seeds"]]
    training_seeds = [17, 29, 43]
    expected_runs = len(models) * len(wells) * len(split_seeds) * len(training_seeds)
    plan = {
        "status": "FROZEN_BEFORE_ANY_STRICT_TEST_PARTITION_IS_OPENED",
        "protocol_id": protocol["protocol_id"],
        "research_task": protocol["research_scope"]["task"],
        "reporting_role": (
            "strict complete-interval sensitivity analysis; complements but does not "
            "replace the primary within-well random-center interpolation experiment"
        ),
        "wells": wells,
        "models": models,
        "primary_method": "ga_gru",
        "model_configuration_rule": (
            "reuse every model specification frozen for the primary experiment; no "
            "strict-test result or strict-test metric informed any configuration"
        ),
        "split_seeds": split_seeds,
        "training_seeds": training_seeds,
        "features": list(protocol["features"]["curves"]),
        "representation": primary_plan["representation"],
        "input_channels": int(primary_plan["input_channels"]),
        "imbalance_strategy": primary_plan["imbalance_strategy"],
        "window_length": int(protocol["windows"]["length"]),
        "split_fractions": protocol["split"]["target_fractions"],
        "purge_each_side_m": float(protocol["split"]["purge_each_side_m"]),
        "batch_size": int(primary_plan["batch_size"]),
        "maximum_selection_epochs": int(primary_plan["maximum_selection_epochs"]),
        "selection_patience": int(primary_plan["selection_patience"]),
        "validation_metric": (
            "macro-F1 over classes with windows in the corresponding validation "
            "partition; accuracy and balanced accuracy are checkpoint tie-breakers"
        ),
        "refit_rule": (
            "select epochs on strict train/validation partitions, refit from scratch "
            "for that epoch count on their union, and score the untouched strict test "
            "partition exactly once"
        ),
        "preprocessing_rule": (
            "selection imputation and engineered-feature standardization are fitted on "
            "strict training windows only; final versions are refitted on strict "
            "train+validation windows; test is transform-only"
        ),
        "balance_rule": (
            "SMOTE-Tomek is fitted only on selection training windows or final "
            "train+validation windows; one deterministic balanced set per split/well "
            "is shared across models and training seeds"
        ),
        "test_use_rule": (
            "each frozen run predicts its strict test partition once; no test metric "
            "may alter a model, epoch, feature, balance, or reporting choice"
        ),
        "reported_metrics": {
            "per_well": [
                "accuracy",
                "supported_macro_f1",
                "balanced_accuracy",
                "weighted_f1",
                "family_accuracy",
                "family_macro_f1",
            ],
            "pooled_global_ten_class": [
                "accuracy",
                "fixed_10_class_macro_f1",
                "supported_10_class_macro_f1",
                "balanced_accuracy",
            ],
            "pooled_three_family": ["accuracy", "macro_f1", "balanced_accuracy"],
            "family_map": protocol["class_reporting"]["family_map_by_global_class"],
        },
        "statistical_plan": {
            "descriptive": (
                "average nine split/training-seed repeats within each well, then report "
                "the mean and standard deviation over six wells"
            ),
            "paired_unit": "well mean over nine repeats",
            "tests": (
                "two-sided paired Wilcoxon signed-rank for GA-GRU versus each baseline"
            ),
            "multiplicity": "Holm adjustment across all planned comparisons",
            "confidence_interval": (
                "95% well-cluster bootstrap interval for mean paired gain, 10000 "
                "resamples, seed 82017"
            ),
            "primary_metrics": [
                "supported_macro_f1",
                "accuracy",
                "family_accuracy",
            ],
            "low_power_disclosure": "only six independent well-level paired units",
        },
        "expected_runs": expected_runs,
        "expected_recurrent_runs": (
            sum(model["family"] == "recurrent_neural_network" for model in models)
            * len(wells)
            * len(split_seeds)
            * len(training_seeds)
        ),
        "expected_extra_trees_runs": (
            sum(model["family"] == "classical_machine_learning" for model in models)
            * len(wells)
            * len(split_seeds)
            * len(training_seeds)
        ),
        "strict_test_partition_files_opened_while_freezing_plan": False,
        "strict_test_metrics_available_while_freezing_plan": False,
        "source_hashes": {
            "strict_protocol": sha256_file(PROTOCOL_PATH),
            "strict_preflight": sha256_file(PREFLIGHT_PATH),
            "primary_pre_test_model_plan": sha256_file(PRIMARY_PLAN_PATH),
            "runner_code": sha256_file(RUNNER_PATH),
            "analysis_code": sha256_file(ANALYSIS_PATH),
        },
    }
    if PLAN_PATH.is_file():
        if json.loads(PLAN_PATH.read_text(encoding="utf-8")) != plan:
            raise RuntimeError("Existing strict evaluation plan conflicts")
    else:
        write_json_atomic(PLAN_PATH, plan)

    print("Strict complete-interval evaluation plan: FROZEN")
    print(f"Models: {len(models)}")
    print(f"Wells: {len(wells)}")
    print(f"Split seeds: {split_seeds}")
    print(f"Training seeds: {training_seeds}")
    print(f"Expected runs: {expected_runs}")
    print("Strict test partition files opened: False")
    print(f"Plan: {PLAN_PATH}")


if __name__ == "__main__":
    main()
