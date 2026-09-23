"""Freeze fixed and legacy-parameter recurrent supplemental baselines."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = (
    PROJECT_DIR
    / "experiment_protocol_v5_random_center"
    / "random_center_protocol_v5.json"
)
PRIMARY_DIR = (
    PROJECT_DIR / "outputs" / "independent_wells_v5" / "final_evaluation"
)
PRIMARY_PLAN_PATH = PRIMARY_DIR / "final_evaluation_plan.json"
PRIMARY_SUMMARY_PATH = PRIMARY_DIR / "analysis" / "final_results_summary.json"
BASELINE_MANIFEST_PATH = (
    PROJECT_DIR
    / "outputs"
    / "independent_wells_v5"
    / "recurrent_baseline_screen"
    / "screen_manifest.json"
)
OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "independent_wells_v5"
    / "fixed_recurrent_supplement"
)
PLAN_PATH = OUTPUT_DIR / "supplement_plan.json"
RUNNER_PATH = PROJECT_DIR / "92_run_fixed_recurrent_supplement.py"
ANALYSIS_PATH = PROJECT_DIR / "93_analyze_fixed_recurrent_supplement.py"
PRIMARY_RUNNER_PATH = PROJECT_DIR / "80_run_final_repeated_evaluation.py"

LEGACY_RNN_NOTEBOOK = PROJECT_DIR / "private_references" / "RNN.ipynb"
LEGACY_LSTM_NOTEBOOK = PROJECT_DIR / "private_references" / "lstm.ipynb"
LEGACY_THESIS = PROJECT_DIR / "private_references" / "legacy_thesis.docx"


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


def fixed_candidate() -> dict[str, int | float]:
    return {
        "hidden_size": 96,
        "num_layers": 2,
        "learning_rate": 0.0007,
        "dropout": 0.2,
        "weight_decay": 0.0001,
    }


def legacy_candidate() -> dict[str, int | float]:
    return {
        "hidden_size": 12,
        "num_layers": 2,
        "learning_rate": 0.01,
        "dropout": 0.0,
        "weight_decay": 0.0,
    }


def main() -> None:
    required = (
        PROTOCOL_PATH,
        PRIMARY_PLAN_PATH,
        PRIMARY_SUMMARY_PATH,
        BASELINE_MANIFEST_PATH,
        RUNNER_PATH,
        ANALYSIS_PATH,
        PRIMARY_RUNNER_PATH,
        LEGACY_RNN_NOTEBOOK,
        LEGACY_LSTM_NOTEBOOK,
        LEGACY_THESIS,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Required source files are missing: {missing}")

    primary_plan = json.loads(PRIMARY_PLAN_PATH.read_text(encoding="utf-8"))
    baseline_manifest = json.loads(
        BASELINE_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    fixed_records = [
        record
        for record in baseline_manifest["candidates"]
        if record["source"] == "fixed"
    ]
    if len(fixed_records) != 1 or fixed_records[0]["candidate"] != fixed_candidate():
        raise RuntimeError("The pre-test fixed recurrent candidate changed")
    fixed_gru = next(
        model for model in primary_plan["models"] if model["model_id"] == "fixed_gru"
    )
    if fixed_gru["candidate"] != fixed_candidate():
        raise RuntimeError("The primary fixed-GRU candidate differs from the screen")

    models = [
        {
            "model_id": "fixed_rnn",
            "display_name": "Fixed RNN",
            "family": "recurrent_neural_network",
            "architecture": "vanilla_rnn",
            "candidate": fixed_candidate(),
            "candidate_source": (
                "pre-registered default configuration frozen in the recurrent "
                "baseline screen before primary test access"
            ),
            "reporting_role": "fixed-configuration architecture baseline",
        },
        {
            "model_id": "fixed_lstm",
            "display_name": "Fixed LSTM",
            "family": "recurrent_neural_network",
            "architecture": "lstm",
            "candidate": fixed_candidate(),
            "candidate_source": (
                "pre-registered default configuration frozen in the recurrent "
                "baseline screen before primary test access"
            ),
            "reporting_role": "fixed-configuration architecture baseline",
        },
        {
            "model_id": "legacy_parameter_rnn",
            "display_name": "Legacy-parameter RNN",
            "family": "recurrent_neural_network",
            "architecture": "vanilla_rnn",
            "candidate": legacy_candidate(),
            "candidate_source": (
                "hidden size, layer count, and learning rate transcribed from "
                "the original RNN notebook"
            ),
            "reporting_role": "legacy-hyperparameter sensitivity analysis",
        },
        {
            "model_id": "legacy_parameter_lstm",
            "display_name": "Legacy-parameter LSTM",
            "family": "recurrent_neural_network",
            "architecture": "lstm",
            "candidate": legacy_candidate(),
            "candidate_source": (
                "hidden size, layer count, and learning rate transcribed from "
                "the original LSTM notebook"
            ),
            "reporting_role": "legacy-hyperparameter sensitivity analysis",
        },
    ]
    plan = {
        "status": "FROZEN_BEFORE_SUPPLEMENTAL_MODEL_TEST_RUNS",
        "scope": "post-primary-analysis fixed recurrent baseline extension",
        "primary_test_results_were_already_available_when_frozen": True,
        "supplemental_configurations_selected_from_primary_test_metrics": False,
        "rationale": (
            "Add fixed-configuration and legacy-parameter RNN/LSTM baselines "
            "without changing the frozen primary GA-GRU analysis."
        ),
        "models": models,
        "wells": list(primary_plan["wells"]),
        "split_seeds": list(primary_plan["split_seeds"]),
        "training_seeds": list(primary_plan["training_seeds"]),
        "features": list(primary_plan["features"]),
        "representation": primary_plan["representation"],
        "input_channels": int(primary_plan["input_channels"]),
        "imbalance_strategy": primary_plan["imbalance_strategy"],
        "window_length": int(primary_plan["window_length"]),
        "split_fractions": dict(primary_plan["split_fractions"]),
        "batch_size": int(primary_plan["batch_size"]),
        "maximum_selection_epochs": int(
            primary_plan["maximum_selection_epochs"]
        ),
        "selection_patience": int(primary_plan["selection_patience"]),
        "epoch_selection_rule": primary_plan["epoch_selection_rule"],
        "refit_rule": primary_plan["refit_rule"],
        "preprocessing_rule": primary_plan["preprocessing_rule"],
        "balance_randomness_rule": primary_plan["balance_randomness_rule"],
        "test_use_rule": (
            "supplemental configurations were frozen from pre-existing defaults "
            "and legacy source code; each test set is scored once per run"
        ),
        "legacy_source_configuration": {
            "hidden_size": 12,
            "num_layers": 2,
            "learning_rate": 0.01,
            "dropout_default": 0.0,
            "weight_decay_default": 0.0,
            "reported_epochs": 3200,
            "reported_batch_size": 32,
            "source_code_batch_size_was_not_used": True,
            "source_code_effective_training": "full-batch update per epoch",
            "source_code_sequence_length": 1,
            "thesis_claimed_window_length": 4,
        },
        "legacy_adaptation": (
            "Only hidden size, layer count, learning rate, and default "
            "regularization are transferred. All models use the current common "
            "seven-curve, L=9, validation-selected epoch, SMOTE-Tomek protocol."
        ),
        "expected_runs": (
            len(models)
            * len(primary_plan["wells"])
            * len(primary_plan["split_seeds"])
            * len(primary_plan["training_seeds"])
        ),
        "reported_metrics": list(primary_plan["reported_metrics"]),
        "analysis_role": (
            "supplemental descriptive evidence; the original primary analysis "
            "and multiplicity-controlled tests remain unchanged"
        ),
        "source_hashes": {
            "protocol": sha256_file(PROTOCOL_PATH),
            "primary_plan": sha256_file(PRIMARY_PLAN_PATH),
            "primary_results_summary": sha256_file(PRIMARY_SUMMARY_PATH),
            "pre_test_recurrent_screen_manifest": sha256_file(
                BASELINE_MANIFEST_PATH
            ),
            "primary_runner_code": sha256_file(PRIMARY_RUNNER_PATH),
            "supplement_runner_code": sha256_file(RUNNER_PATH),
            "supplement_analysis_code": sha256_file(ANALYSIS_PATH),
            "legacy_rnn_notebook": sha256_file(LEGACY_RNN_NOTEBOOK),
            "legacy_lstm_notebook": sha256_file(LEGACY_LSTM_NOTEBOOK),
            "legacy_thesis": sha256_file(LEGACY_THESIS),
        },
    }
    if PLAN_PATH.is_file():
        existing = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        if existing != plan:
            raise RuntimeError("Existing supplemental plan conflicts with this freeze")
    else:
        write_json_atomic(PLAN_PATH, plan)
    print("Fixed recurrent supplement plan: FROZEN")
    print(f"Models: {', '.join(model['model_id'] for model in models)}")
    print(f"Expected runs: {plan['expected_runs']}")
    print(f"Plan: {PLAN_PATH}")


if __name__ == "__main__":
    main()
