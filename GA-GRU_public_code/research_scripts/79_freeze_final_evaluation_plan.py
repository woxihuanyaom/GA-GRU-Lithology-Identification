"""Freeze every final-evaluation choice before opening test assignments."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


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
ABLATION_FREEZE_PATH = (
    PROJECT_DIR
    / "outputs"
    / "independent_wells_v5"
    / "validation_ablation"
    / "ablation_winner_freeze.json"
)
RECURRENT_FREEZE_PATH = (
    PROJECT_DIR
    / "outputs"
    / "independent_wells_v5"
    / "recurrent_baseline_screen"
    / "screen_summary.json"
)
OUTPUT_DIR = PROJECT_DIR / "outputs" / "independent_wells_v5" / "final_evaluation"
PLAN_PATH = OUTPUT_DIR / "final_evaluation_plan.json"
RUNNER_PATH = PROJECT_DIR / "80_run_final_repeated_evaluation.py"
ANALYSIS_PATH = PROJECT_DIR / "81_analyze_final_evaluation.py"


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


def recurrent_model(
    model_id: str,
    display_name: str,
    architecture: str,
    candidate: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "display_name": display_name,
        "family": "recurrent_neural_network",
        "architecture": architecture,
        "candidate": candidate,
        "candidate_source": source,
    }


def main() -> None:
    required = (
        PROTOCOL_PATH,
        OPTIMIZER_FREEZE_PATH,
        ABLATION_FREEZE_PATH,
        RECURRENT_FREEZE_PATH,
        RUNNER_PATH,
        ANALYSIS_PATH,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Cannot freeze final plan; missing files: {missing}")

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    optimizers = json.loads(OPTIMIZER_FREEZE_PATH.read_text(encoding="utf-8"))
    ablation = json.loads(ABLATION_FREEZE_PATH.read_text(encoding="utf-8"))
    recurrent = json.loads(RECURRENT_FREEZE_PATH.read_text(encoding="utf-8"))
    if optimizers["test_metrics_used_for_selection"] is not False:
        raise RuntimeError("Optimizer freeze is not test-blind")
    if ablation["test_metrics_used_for_selection"] is not False:
        raise RuntimeError("Ablation freeze is not test-blind")
    if recurrent["test_metrics_used_for_selection"] is not False:
        raise RuntimeError("Recurrent baseline freeze is not test-blind")

    fixed_candidate = {
        "hidden_size": 96,
        "num_layers": 2,
        "learning_rate": 0.0007,
        "dropout": 0.2,
        "weight_decay": 0.0001,
    }
    models = [
        recurrent_model(
            "ga_gru",
            "GA-GRU",
            "gru",
            optimizers["methods"]["ga"]["candidate"],
            "equal-budget GA validation winner",
        ),
        recurrent_model(
            "random_search_gru",
            "Random-search GRU",
            "gru",
            optimizers["methods"]["random"]["candidate"],
            "equal-budget random-search validation winner",
        ),
        recurrent_model(
            "tpe_gru",
            "TPE-GRU",
            "gru",
            optimizers["methods"]["tpe"]["candidate"],
            "equal-budget TPE validation winner",
        ),
        recurrent_model(
            "fixed_gru",
            "Fixed GRU",
            "gru",
            fixed_candidate,
            "pre-registered default configuration",
        ),
        recurrent_model(
            "tuned_rnn",
            "Tuned RNN",
            "vanilla_rnn",
            recurrent["winners"]["vanilla_rnn"]["candidate"],
            "validation-only recurrent baseline screen",
        ),
        recurrent_model(
            "tuned_lstm",
            "Tuned LSTM",
            "lstm",
            recurrent["winners"]["lstm"]["candidate"],
            "validation-only recurrent baseline screen",
        ),
        {
            "model_id": "extra_trees",
            "display_name": "ExtraTrees",
            "family": "classical_machine_learning",
            "architecture": "ExtraTreesClassifier",
            "parameters": {
                "n_estimators": 500,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "class_weight": None,
                "n_jobs": -1,
            },
            "parameter_source": "frozen initial random-center validation screen",
        },
    ]
    split_seeds = [int(value) for value in ablation["final_split_seeds"]]
    training_seeds = [int(value) for value in ablation["final_training_seeds"]]
    wells = [str(value) for value in protocol["research_scope"]["wells"]]
    selected = ablation["selected_configuration"]
    expected_runs = len(models) * len(split_seeds) * len(training_seeds) * len(wells)
    plan = {
        "status": "FROZEN_BEFORE_ANY_FINAL_TEST_ASSIGNMENT_IS_OPENED",
        "protocol_id": protocol["protocol_id"],
        "research_task": protocol["research_scope"]["task"],
        "interpretation": ablation["interpretation"],
        "wells": wells,
        "models": models,
        "primary_method": "ga_gru",
        "split_seeds": split_seeds,
        "training_seeds": training_seeds,
        "features": selected["features"],
        "representation": selected["representation"],
        "input_channels": int(selected["input_channels"]),
        "imbalance_strategy": selected["imbalance_strategy"],
        "window_length": int(protocol["windows"]["length"]),
        "split_fractions": protocol["split"]["fractions"],
        "batch_size": 512,
        "maximum_selection_epochs": 60,
        "selection_patience": 8,
        "epoch_selection_rule": (
            "for each recurrent model/well/split/training-seed run, maximize validation "
            "macro-F1 with the frozen checkpoint tie-break implemented by the trainer"
        ),
        "refit_rule": ablation["final_refit_rule"],
        "preprocessing_rule": (
            "selection preprocessing is fitted on train centers; final preprocessing "
            "is refitted on train+validation centers; test centers are transform-only"
        ),
        "balance_randomness_rule": (
            "one fixed SMOTE-Tomek result per split/well is shared by every model and "
            "training seed to preserve paired comparisons"
        ),
        "test_use_rule": (
            "each frozen model run predicts its corresponding test-center set once; "
            "test metrics cannot change any model, epoch, feature, or balance choice"
        ),
        "reported_metrics": [
            "accuracy",
            "macro_f1",
            "balanced_accuracy",
            "weighted_f1",
            "per_class_precision_recall_f1",
        ],
        "statistical_plan": {
            "descriptive": (
                "mean and standard deviation over six per-well means, where each well "
                "mean averages its nine split/training-seed repeats"
            ),
            "paired_unit": (
                "well; average the nine repeats within each well before comparison"
            ),
            "test": "two-sided paired Wilcoxon signed-rank",
            "multiplicity": "Holm correction across all GA-versus-baseline tests",
            "confidence_interval": (
                "95% well-cluster bootstrap interval for the mean paired gain, "
                "10000 resamples, seed 81017"
            ),
            "primary_metrics": ["macro_f1", "accuracy"],
        },
        "expected_runs": expected_runs,
        "expected_recurrent_runs": 6 * len(split_seeds) * len(training_seeds) * len(wells),
        "expected_extra_trees_runs": len(split_seeds) * len(training_seeds) * len(wells),
        "test_assignment_files_opened_while_freezing_plan": False,
        "test_metrics_available_while_freezing_plan": False,
        "source_hashes": {
            "protocol": sha256_file(PROTOCOL_PATH),
            "optimizer_freeze": sha256_file(OPTIMIZER_FREEZE_PATH),
            "ablation_freeze": sha256_file(ABLATION_FREEZE_PATH),
            "recurrent_freeze": sha256_file(RECURRENT_FREEZE_PATH),
            "runner_code": sha256_file(RUNNER_PATH),
            "analysis_code": sha256_file(ANALYSIS_PATH),
        },
    }
    if PLAN_PATH.is_file():
        if json.loads(PLAN_PATH.read_text(encoding="utf-8")) != plan:
            raise RuntimeError("Existing final-evaluation plan conflicts")
    else:
        write_json_atomic(PLAN_PATH, plan)
    print("Final evaluation plan: FROZEN")
    print(f"Models: {len(models)}")
    print(f"Wells: {len(wells)}")
    print(f"Split seeds: {split_seeds}")
    print(f"Training seeds: {training_seeds}")
    print(f"Expected runs: {expected_runs}")
    print("Test assignment files opened: False")
    print(f"Plan: {PLAN_PATH}")


if __name__ == "__main__":
    main()
