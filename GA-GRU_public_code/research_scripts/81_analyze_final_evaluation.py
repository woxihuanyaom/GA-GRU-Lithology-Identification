"""Audit, summarize, and statistically analyze frozen final test results."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs" / "independent_wells_v5" / "final_evaluation"
PLAN_PATH = OUTPUT_DIR / "final_evaluation_plan.json"
COMPLETION_PATH = OUTPUT_DIR / "execution_completion.json"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"
AUDIT_PATH = ANALYSIS_DIR / "final_evaluation_audit.json"
PRIMARY_MODEL = "ga_gru"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 81_017


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


def holm_adjust(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    order = np.argsort(result["p_value"].to_numpy(float), kind="stable")
    adjusted = np.empty(len(result), dtype=float)
    running = 0.0
    count = len(result)
    for rank_index, original_index in enumerate(order):
        candidate = min(1.0, (count - rank_index) * result.iloc[original_index]["p_value"])
        running = max(running, candidate)
        adjusted[original_index] = running
    result["holm_adjusted_p"] = adjusted
    result["significant_at_0_05"] = result["holm_adjusted_p"] < 0.05
    return result


def paired_statistics(per_well: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for metric in ("macro_f1", "accuracy"):
        pivot = per_well.pivot(index="well_id", columns="model_id", values=metric)
        if PRIMARY_MODEL not in pivot:
            raise RuntimeError("Primary GA-GRU results are absent")
        for comparator in models:
            if comparator == PRIMARY_MODEL:
                continue
            difference = (
                pivot[PRIMARY_MODEL].to_numpy(float)
                - pivot[comparator].to_numpy(float)
            )
            if np.allclose(difference, 0):
                statistic = 0.0
                p_value = 1.0
            else:
                test = wilcoxon(
                    difference,
                    zero_method="wilcox",
                    correction=False,
                    alternative="two-sided",
                    method="auto",
                )
                statistic = float(test.statistic)
                p_value = float(test.pvalue)
            sampled = difference[
                rng.integers(0, len(difference), size=(BOOTSTRAP_RESAMPLES, len(difference)))
            ].mean(axis=1)
            rows.append(
                {
                    "metric": metric,
                    "comparison": f"ga_gru_minus_{comparator}",
                    "comparator": comparator,
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
                    "p_value": p_value,
                }
            )
    return holm_adjust(pd.DataFrame(rows))


def main() -> None:
    if not PLAN_PATH.is_file() or not COMPLETION_PATH.is_file():
        raise RuntimeError("Final execution must complete before analysis")
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    completion = json.loads(COMPLETION_PATH.read_text(encoding="utf-8"))
    if sha256_file(Path(__file__)) != plan["source_hashes"]["analysis_code"]:
        raise RuntimeError("Analysis code changed after the final plan was frozen")
    if sha256_file(PLAN_PATH) != completion["plan_sha256"]:
        raise RuntimeError("Final plan changed after execution")
    if completion["status"] != "COMPLETE_PENDING_ANALYSIS":
        raise RuntimeError("Final execution is incomplete")
    if completion["test_metrics_used_for_any_selection"] is not False:
        raise RuntimeError("Execution does not certify test-blind model choices")

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
    per_class_rows = []
    predictions_by_cell: dict[tuple[int, str], set[tuple[int, int]]] = {}
    observed_keys = set()
    primary_prediction_parts: dict[str, list[pd.DataFrame]] = {
        model: [] for model in expected_models
    }

    for relative in completion["result_files"]:
        result_path = PROJECT_DIR / str(relative)
        if not result_path.is_file():
            raise RuntimeError(f"Missing result file: {result_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        key = (
            str(result["model_id"]),
            str(result["well_id"]),
            int(result["split_seed"]),
            int(result["training_seed"]),
        )
        if key in observed_keys:
            raise RuntimeError(f"Duplicate final result: {key}")
        observed_keys.add(key)
        if result["status"] != "COMPLETE":
            raise RuntimeError(f"Incomplete final result: {key}")
        if result["test_metrics_used_for_any_selection"] is not False:
            raise RuntimeError(f"Test feedback was reported for {key}")

        predictions_path = PROJECT_DIR / str(result["predictions_file"])
        if sha256_file(predictions_path) != result["predictions_sha256"]:
            raise RuntimeError(f"Prediction hash mismatch: {predictions_path}")
        predictions = pd.read_csv(predictions_path, encoding="utf-8-sig")
        if len(predictions) != int(result["test_samples"]):
            raise RuntimeError(f"Prediction count mismatch: {key}")
        y_true = predictions["true_local_class_id"].to_numpy(np.int64)
        y_prediction = predictions["predicted_local_class_id"].to_numpy(np.int64)
        labels = np.arange(len(result["global_classes"]), dtype=np.int64)
        recomputed = {
            "accuracy": float(accuracy_score(y_true, y_prediction)),
            "macro_f1": float(
                f1_score(
                    y_true,
                    y_prediction,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_true, y_prediction)
            ),
            "weighted_f1": float(
                f1_score(
                    y_true,
                    y_prediction,
                    labels=labels,
                    average="weighted",
                    zero_division=0,
                )
            ),
        }
        for metric, value in recomputed.items():
            if not close(value, float(result["metrics"][metric])):
                raise RuntimeError(f"Recomputed {metric} differs for {key}")
        cell = (int(result["split_seed"]), str(result["well_id"]))
        target_signature = set(
            zip(
                predictions["center_row_id"].astype(int),
                predictions["true_global_class_id"].astype(int),
            )
        )
        if cell in predictions_by_cell and predictions_by_cell[cell] != target_signature:
            raise RuntimeError(f"Models used different test centers for {cell}")
        predictions_by_cell[cell] = target_signature

        row = {
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
            "selection_runtime_seconds": float(result["selection_runtime_seconds"]),
            "refit_runtime_seconds": float(result["refit_runtime_seconds"]),
            "inference_runtime_seconds": float(result["inference_runtime_seconds"]),
            "total_runtime_seconds": float(result["selection_runtime_seconds"])
            + float(result["refit_runtime_seconds"])
            + float(result["inference_runtime_seconds"]),
            "trainable_parameters": result["trainable_parameters"],
            "tree_nodes": result["tree_nodes"],
        }
        rows.append(row)
        for class_record in result["metrics"]["per_class"]:
            per_class_rows.append(
                {
                    "model_id": result["model_id"],
                    "well_id": result["well_id"],
                    "split_seed": int(result["split_seed"]),
                    "training_seed": int(result["training_seed"]),
                    **class_record,
                }
            )
        if (
            int(result["split_seed"]) == expected_splits[0]
            and int(result["training_seed"]) == expected_training_seeds[0]
        ):
            primary_prediction_parts[str(result["model_id"])].append(predictions)

    if observed_keys != expected_keys:
        missing = sorted(expected_keys.difference(observed_keys))
        extra = sorted(observed_keys.difference(expected_keys))
        raise RuntimeError(f"Final run grid differs; missing={missing}, extra={extra}")
    if len(rows) != int(plan["expected_runs"]):
        raise RuntimeError("Final metric row count differs from the plan")

    run_frame = pd.DataFrame(rows).sort_values(
        ["model_id", "well_id", "split_seed", "training_seed"], kind="stable"
    )
    per_class_frame = pd.DataFrame(per_class_rows)
    metric_names = ("accuracy", "macro_f1", "balanced_accuracy", "weighted_f1")
    per_well = (
        run_frame.groupby(["model_id", "display_name", "well_id"], as_index=False)
        .agg(
            **{
                metric: (metric, "mean")
                for metric in metric_names
            },
            accuracy_repeat_sd=("accuracy", "std"),
            macro_f1_repeat_sd=("macro_f1", "std"),
            mean_total_runtime_seconds=("total_runtime_seconds", "mean"),
            mean_selected_epoch=("selected_epoch", "mean"),
            trainable_parameters=("trainable_parameters", "first"),
            mean_tree_nodes=("tree_nodes", "mean"),
            repeats=("accuracy", "size"),
        )
    )

    summary_rows = []
    for (model_id, display_name), subset in per_well.groupby(
        ["model_id", "display_name"], sort=False
    ):
        runs = run_frame.loc[run_frame["model_id"].eq(model_id)]
        summary = {
            "model_id": model_id,
            "display_name": display_name,
            "wells": int(len(subset)),
            "runs": int(len(runs)),
            "pooled_repeated_accuracy": float(
                np.average(runs["accuracy"], weights=runs["test_samples"])
            ),
            "mean_total_runtime_seconds": float(runs["total_runtime_seconds"].mean()),
            "mean_inference_runtime_seconds": float(
                runs["inference_runtime_seconds"].mean()
            ),
            "trainable_parameters": (
                None
                if runs["trainable_parameters"].dropna().empty
                else int(runs["trainable_parameters"].dropna().iloc[0])
            ),
            "mean_tree_nodes": (
                None
                if runs["tree_nodes"].dropna().empty
                else float(runs["tree_nodes"].dropna().mean())
            ),
        }
        for metric in metric_names:
            summary[f"{metric}_mean"] = float(subset[metric].mean())
            summary[f"{metric}_sd_across_wells"] = float(subset[metric].std(ddof=1))
            summary[f"{metric}_run_sd"] = float(runs[metric].std(ddof=1))
        summary_rows.append(summary)
    model_summary = pd.DataFrame(summary_rows).sort_values(
        ["macro_f1_mean", "accuracy_mean"], ascending=False, kind="stable"
    )
    paired = paired_statistics(per_well, expected_models)

    per_class_summary = (
        per_class_frame.groupby(["model_id", "global_class_id"], as_index=False)
        .agg(
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean"),
            f1_sd=("f1", "std"),
            repeated_support=("support", "sum"),
            contributing_runs=("f1", "size"),
        )
        .sort_values(["model_id", "global_class_id"], kind="stable")
    )

    confusion_rows = []
    class_order = np.arange(10, dtype=np.int64)
    for model_id, parts in primary_prediction_parts.items():
        if len(parts) != len(expected_wells):
            raise RuntimeError(f"Primary confusion data are incomplete for {model_id}")
        predictions = pd.concat(parts, ignore_index=True)
        matrix = confusion_matrix(
            predictions["true_global_class_id"],
            predictions["predicted_global_class_id"],
            labels=class_order,
        )
        denominators = np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        normalized = matrix / denominators
        for true_class in class_order:
            for predicted_class in class_order:
                confusion_rows.append(
                    {
                        "model_id": model_id,
                        "split_seed": expected_splits[0],
                        "training_seed": expected_training_seeds[0],
                        "true_global_class_id": int(true_class),
                        "predicted_global_class_id": int(predicted_class),
                        "count": int(matrix[true_class, predicted_class]),
                        "row_normalized_fraction": float(
                            normalized[true_class, predicted_class]
                        ),
                    }
                )
    confusion_frame = pd.DataFrame(confusion_rows)

    run_path = ANALYSIS_DIR / "run_metrics.csv"
    per_well_path = ANALYSIS_DIR / "per_well_model_summary.csv"
    model_path = ANALYSIS_DIR / "model_summary.csv"
    paired_path = ANALYSIS_DIR / "ga_paired_statistics.csv"
    per_class_path = ANALYSIS_DIR / "per_class_summary.csv"
    confusion_path = ANALYSIS_DIR / "primary_run_confusion_matrices.csv"
    write_csv(run_path, run_frame)
    write_csv(per_well_path, per_well)
    write_csv(model_path, model_summary)
    write_csv(paired_path, paired)
    write_csv(per_class_path, per_class_summary)
    write_csv(confusion_path, confusion_frame)

    ga_summary = model_summary.loc[model_summary["model_id"].eq(PRIMARY_MODEL)].iloc[0]
    final_summary = {
        "status": "FINAL_TEST_ANALYSIS_COMPLETE",
        "primary_method": PRIMARY_MODEL,
        "primary_method_metrics": {
            metric: {
                "mean_over_six_well_means": float(ga_summary[f"{metric}_mean"]),
                "sd_across_six_well_means": float(
                    ga_summary[f"{metric}_sd_across_wells"]
                ),
            }
            for metric in metric_names
        },
        "primary_method_pooled_repeated_accuracy": float(
            ga_summary["pooled_repeated_accuracy"]
        ),
        "ranking_is_descriptive_not_a_new_selection_step": True,
        "test_metrics_used_to_change_pipeline": False,
        "model_ranking_by_macro_f1": model_summary[
            ["model_id", "macro_f1_mean", "accuracy_mean"]
        ].to_dict(orient="records"),
        "statistical_unit": "six well means; each averages nine repeats",
        "holm_significant_comparisons": paired.loc[
            paired["significant_at_0_05"]
        ]["comparison"].tolist(),
    }
    summary_json_path = ANALYSIS_DIR / "final_results_summary.json"
    write_json(summary_json_path, final_summary)

    artifact_paths = (
        run_path,
        per_well_path,
        model_path,
        paired_path,
        per_class_path,
        confusion_path,
        summary_json_path,
    )
    audit = {
        "status": "PASS",
        "planned_runs": int(plan["expected_runs"]),
        "verified_runs": len(run_frame),
        "complete_factorial_grid": True,
        "prediction_hashes_verified": True,
        "all_metrics_independently_recomputed": True,
        "identical_test_centers_across_models_within_each_split_well": True,
        "test_metrics_used_for_any_selection": False,
        "paired_unit_avoids_training_seed_pseudoreplication": True,
        "analysis_code_matches_pre_test_freeze": True,
        "artifacts": {path.name: sha256_file(path) for path in artifact_paths},
    }
    write_json(AUDIT_PATH, audit)
    print("Final evaluation audit: PASS")
    print(model_summary.to_string(index=False))
    print("\nGA paired comparisons")
    print(paired.to_string(index=False))
    print(f"Summary: {summary_json_path}")
    print(f"Audit: {AUDIT_PATH}")


if __name__ == "__main__":
    main()
