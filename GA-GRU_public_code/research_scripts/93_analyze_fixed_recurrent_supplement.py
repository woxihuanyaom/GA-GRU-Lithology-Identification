"""Audit and summarize fixed recurrent supplemental baselines."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "independent_wells_v5"
    / "fixed_recurrent_supplement"
)
PLAN_PATH = OUTPUT_DIR / "supplement_plan.json"
COMPLETION_PATH = OUTPUT_DIR / "execution_completion.json"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"
PRIMARY_DIR = (
    PROJECT_DIR / "outputs" / "independent_wells_v5" / "final_evaluation"
)
PRIMARY_RUN_METRICS_PATH = PRIMARY_DIR / "analysis" / "run_metrics.csv"
PRIMARY_RUNS_DIR = PRIMARY_DIR / "runs"
PRIMARY_MODEL = "ga_gru"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 93_017


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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)
    write_hash(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)
    write_hash(path)


def close(observed: float, expected: float) -> bool:
    return math.isclose(observed, expected, rel_tol=0, abs_tol=1e-12)


def summarize_runs(run_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_names = ("accuracy", "macro_f1", "balanced_accuracy", "weighted_f1")
    per_well = (
        run_frame.groupby(["model_id", "display_name", "well_id"], as_index=False)
        .agg(
            **{metric: (metric, "mean") for metric in metric_names},
            accuracy_repeat_sd=("accuracy", "std"),
            macro_f1_repeat_sd=("macro_f1", "std"),
            mean_total_runtime_seconds=("total_runtime_seconds", "mean"),
            mean_selected_epoch=("selected_epoch", "mean"),
            trainable_parameters=("trainable_parameters", "first"),
            repeats=("accuracy", "size"),
        )
    )
    rows = []
    for (model_id, display_name), subset in per_well.groupby(
        ["model_id", "display_name"], sort=False
    ):
        runs = run_frame.loc[run_frame["model_id"].eq(model_id)]
        row: dict[str, Any] = {
            "model_id": model_id,
            "display_name": display_name,
            "wells": int(len(subset)),
            "runs": int(len(runs)),
            "pooled_repeated_accuracy": float(
                np.average(runs["accuracy"], weights=runs["test_samples"])
            ),
            "mean_total_runtime_seconds": float(
                runs["total_runtime_seconds"].mean()
            ),
            "mean_inference_runtime_seconds": float(
                runs["inference_runtime_seconds"].mean()
            ),
            "trainable_parameters": (
                None
                if runs["trainable_parameters"].dropna().empty
                else int(runs["trainable_parameters"].dropna().iloc[0])
            ),
        }
        for metric in metric_names:
            row[f"{metric}_mean"] = float(subset[metric].mean())
            row[f"{metric}_sd_across_wells"] = float(
                subset[metric].std(ddof=1)
            )
            row[f"{metric}_run_sd"] = float(runs[metric].std(ddof=1))
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(
        ["macro_f1_mean", "accuracy_mean"], ascending=False, kind="stable"
    )
    return per_well, summary


def supplemental_comparisons(
    combined_per_well: pd.DataFrame, supplemental_models: list[str]
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for metric in ("macro_f1", "accuracy"):
        pivot = combined_per_well.pivot(
            index="well_id", columns="model_id", values=metric
        )
        for comparator in supplemental_models:
            difference = (
                pivot[PRIMARY_MODEL].to_numpy(float)
                - pivot[comparator].to_numpy(float)
            )
            if np.allclose(difference, 0):
                statistic, p_value = 0.0, 1.0
            else:
                test = wilcoxon(
                    difference,
                    zero_method="wilcox",
                    correction=False,
                    alternative="two-sided",
                    method="auto",
                )
                statistic, p_value = float(test.statistic), float(test.pvalue)
            sampled = difference[
                rng.integers(
                    0,
                    len(difference),
                    size=(BOOTSTRAP_RESAMPLES, len(difference)),
                )
            ].mean(axis=1)
            rows.append(
                {
                    "metric": metric,
                    "comparison": f"ga_gru_minus_{comparator}",
                    "comparator": comparator,
                    "analysis_role": "post-hoc supplemental descriptive comparison",
                    "paired_unit": "well_mean_over_nine_repeats",
                    "paired_units": int(len(difference)),
                    "mean_gain": float(difference.mean()),
                    "median_gain": float(np.median(difference)),
                    "bootstrap_95_ci_low": float(np.quantile(sampled, 0.025)),
                    "bootstrap_95_ci_high": float(np.quantile(sampled, 0.975)),
                    "ga_higher_wells": int(np.sum(difference > 0)),
                    "ties": int(np.sum(np.isclose(difference, 0))),
                    "ga_lower_wells": int(np.sum(difference < 0)),
                    "wilcoxon_statistic": statistic,
                    "unadjusted_p_value": p_value,
                    "confirmatory_significance_claim_allowed": False,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    if not PLAN_PATH.is_file() or not COMPLETION_PATH.is_file():
        raise RuntimeError("Complete the frozen supplemental execution first")
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    completion = json.loads(COMPLETION_PATH.read_text(encoding="utf-8"))
    if sha256_file(Path(__file__)) != plan["source_hashes"][
        "supplement_analysis_code"
    ]:
        raise RuntimeError("Supplemental analysis code changed after plan freeze")
    if sha256_file(PLAN_PATH) != completion["plan_sha256"]:
        raise RuntimeError("Supplemental plan changed after execution")
    if completion["status"] != "COMPLETE_PENDING_ANALYSIS":
        raise RuntimeError("Supplemental execution is incomplete")
    if completion["test_metrics_used_for_model_or_configuration_selection"]:
        raise RuntimeError("Supplement reports test-based model selection")

    expected_models = [str(model["model_id"]) for model in plan["models"]]
    expected_wells = [str(value) for value in plan["wells"]]
    expected_splits = [int(value) for value in plan["split_seeds"]]
    expected_training_seeds = [int(value) for value in plan["training_seeds"]]
    expected_keys = {
        (model, well, split, seed)
        for model in expected_models
        for well in expected_wells
        for split in expected_splits
        for seed in expected_training_seeds
    }
    rows = []
    observed_keys = set()
    verified_primary_signatures: dict[tuple[str, int, int], tuple[tuple[int, int], ...]] = {}
    for relative in completion["result_files"]:
        result_path = PROJECT_DIR / str(relative)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        key = (
            str(result["model_id"]),
            str(result["well_id"]),
            int(result["split_seed"]),
            int(result["training_seed"]),
        )
        if key in observed_keys:
            raise RuntimeError(f"Duplicate supplemental result: {key}")
        observed_keys.add(key)
        if result["status"] != "COMPLETE":
            raise RuntimeError(f"Incomplete supplemental result: {key}")
        if result["test_metrics_used_for_model_or_configuration_selection"]:
            raise RuntimeError(f"Test feedback reported for {key}")
        predictions_path = PROJECT_DIR / str(result["predictions_file"])
        if sha256_file(predictions_path) != result["predictions_sha256"]:
            raise RuntimeError(f"Prediction hash mismatch: {predictions_path}")
        predictions = pd.read_csv(predictions_path, encoding="utf-8-sig")
        y_true = predictions["true_local_class_id"].to_numpy(np.int64)
        y_pred = predictions["predicted_local_class_id"].to_numpy(np.int64)
        labels = np.arange(len(result["global_classes"]), dtype=np.int64)
        recomputed = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(
                f1_score(
                    y_true,
                    y_pred,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
            ),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "weighted_f1": float(
                f1_score(
                    y_true,
                    y_pred,
                    labels=labels,
                    average="weighted",
                    zero_division=0,
                )
            ),
        }
        for metric, value in recomputed.items():
            if not close(value, float(result["metrics"][metric])):
                raise RuntimeError(f"Recomputed {metric} differs for {key}")

        signature_key = (key[1], key[2], key[3])
        signature = tuple(
            zip(
                predictions["center_row_id"].astype(int),
                predictions["true_global_class_id"].astype(int),
                strict=True,
            )
        )
        if signature_key not in verified_primary_signatures:
            primary_predictions_path = (
                PRIMARY_RUNS_DIR
                / f"split_{key[2]}"
                / key[1]
                / PRIMARY_MODEL
                / f"seed_{key[3]}"
                / "test_predictions.csv"
            )
            primary_predictions = pd.read_csv(
                primary_predictions_path, encoding="utf-8-sig"
            )
            verified_primary_signatures[signature_key] = tuple(
                zip(
                    primary_predictions["center_row_id"].astype(int),
                    primary_predictions["true_global_class_id"].astype(int),
                    strict=True,
                )
            )
        if signature != verified_primary_signatures[signature_key]:
            raise RuntimeError(f"Supplement used different test centers for {key}")

        rows.append(
            {
                "model_id": result["model_id"],
                "display_name": result["display_name"],
                "model_family": result["model_family"],
                "architecture": result["architecture"],
                "well_id": result["well_id"],
                "split_seed": int(result["split_seed"]),
                "training_seed": int(result["training_seed"]),
                "classes": len(result["global_classes"]),
                "test_samples": int(result["test_samples"]),
                **recomputed,
                "selected_epoch": result["selected_epoch"],
                "selection_runtime_seconds": float(
                    result["selection_runtime_seconds"]
                ),
                "refit_runtime_seconds": float(result["refit_runtime_seconds"]),
                "inference_runtime_seconds": float(
                    result["inference_runtime_seconds"]
                ),
                "total_runtime_seconds": float(
                    result["selection_runtime_seconds"]
                )
                + float(result["refit_runtime_seconds"])
                + float(result["inference_runtime_seconds"]),
                "trainable_parameters": result["trainable_parameters"],
                "tree_nodes": None,
            }
        )

    if observed_keys != expected_keys:
        raise RuntimeError("Supplemental result grid differs from the frozen plan")
    run_frame = pd.DataFrame(rows).sort_values(
        ["model_id", "well_id", "split_seed", "training_seed"], kind="stable"
    )
    per_well, supplemental_summary = summarize_runs(run_frame)

    primary_runs = pd.read_csv(PRIMARY_RUN_METRICS_PATH, encoding="utf-8-sig")
    combined_runs = pd.concat((primary_runs, run_frame), ignore_index=True)
    combined_per_well, combined_summary = summarize_runs(combined_runs)
    comparisons = supplemental_comparisons(combined_per_well, expected_models)

    run_path = ANALYSIS_DIR / "supplemental_run_metrics.csv"
    per_well_path = ANALYSIS_DIR / "supplemental_per_well_summary.csv"
    model_path = ANALYSIS_DIR / "supplemental_model_summary.csv"
    combined_path = ANALYSIS_DIR / "combined_model_summary.csv"
    comparison_path = ANALYSIS_DIR / "ga_supplemental_comparisons.csv"
    write_csv(run_path, run_frame)
    write_csv(per_well_path, per_well)
    write_csv(model_path, supplemental_summary)
    write_csv(combined_path, combined_summary)
    write_csv(comparison_path, comparisons)
    summary_path = ANALYSIS_DIR / "supplemental_results_summary.json"
    write_json(
        summary_path,
        {
            "status": "SUPPLEMENTAL_ANALYSIS_COMPLETE",
            "analysis_role": plan["analysis_role"],
            "primary_results_available_before_supplement": True,
            "test_metrics_used_for_model_or_configuration_selection": False,
            "supplemental_ranking_by_macro_f1": supplemental_summary[
                ["model_id", "macro_f1_mean", "accuracy_mean"]
            ].to_dict(orient="records"),
            "confirmatory_significance_claim_allowed": False,
        },
    )
    artifact_paths = (
        run_path,
        per_well_path,
        model_path,
        combined_path,
        comparison_path,
        summary_path,
    )
    audit_path = ANALYSIS_DIR / "supplemental_analysis_audit.json"
    write_json(
        audit_path,
        {
            "status": "PASS",
            "planned_runs": int(plan["expected_runs"]),
            "verified_runs": len(run_frame),
            "complete_factorial_grid": True,
            "prediction_hashes_verified": True,
            "all_metrics_independently_recomputed": True,
            "test_centers_match_primary_ga_runs": True,
            "test_metrics_used_for_model_or_configuration_selection": False,
            "post_primary_analysis_supplement_disclosed": True,
            "artifacts": {
                path.name: sha256_file(path) for path in artifact_paths
            },
        },
    )
    print("Fixed recurrent supplemental audit: PASS")
    print(supplemental_summary.to_string(index=False))
    print("\nGA-GRU supplemental comparisons")
    print(comparisons.to_string(index=False))
    print(f"Combined summary: {combined_path}")
    print(f"Audit: {audit_path}")


if __name__ == "__main__":
    main()
