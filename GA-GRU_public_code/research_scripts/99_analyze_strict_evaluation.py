"""Audit and analyze the frozen strict complete-interval test results."""

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
    precision_recall_fscore_support,
)


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "independent_wells_v7_strict_interval"
    / "final_evaluation"
)
PLAN_PATH = OUTPUT_DIR / "strict_evaluation_plan.json"
COMPLETION_PATH = OUTPUT_DIR / "execution_completion.json"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"
AUDIT_PATH = ANALYSIS_DIR / "strict_evaluation_audit.json"
PRIMARY_MODEL = "ga_gru"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 82_017


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


def metric_bundle(
    predictions: pd.DataFrame, global_classes: tuple[int, ...]
) -> dict[str, float]:
    true_local = predictions["true_local_class_id"].to_numpy(np.int64)
    predicted_local = predictions["predicted_local_class_id"].to_numpy(np.int64)
    labels = np.arange(len(global_classes), dtype=np.int64)
    supported = np.unique(true_local)
    true_family = predictions["true_family_id"].to_numpy(np.int64)
    predicted_family = predictions["predicted_family_id"].to_numpy(np.int64)
    supported_families = np.unique(true_family)
    return {
        "accuracy": float(accuracy_score(true_local, predicted_local)),
        "supported_macro_f1": float(
            f1_score(
                true_local,
                predicted_local,
                labels=supported,
                average="macro",
                zero_division=0,
            )
        ),
        "fixed_local_macro_f1": float(
            f1_score(
                true_local,
                predicted_local,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(true_local, predicted_local)
        ),
        "weighted_f1": float(
            f1_score(
                true_local,
                predicted_local,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "family_accuracy": float(accuracy_score(true_family, predicted_family)),
        "family_macro_f1": float(
            f1_score(
                true_family,
                predicted_family,
                labels=supported_families,
                average="macro",
                zero_division=0,
            )
        ),
        "family_balanced_accuracy": float(
            balanced_accuracy_score(true_family, predicted_family)
        ),
    }


def pooled_global_bundle(predictions: pd.DataFrame) -> dict[str, float | int]:
    true_global = predictions["true_global_class_id"].to_numpy(np.int64)
    predicted_global = predictions["predicted_global_class_id"].to_numpy(np.int64)
    true_family = predictions["true_family_id"].to_numpy(np.int64)
    predicted_family = predictions["predicted_family_id"].to_numpy(np.int64)
    supported_classes = np.unique(true_global)
    supported_families = np.unique(true_family)
    return {
        "test_samples": int(len(predictions)),
        "supported_global_classes": int(len(supported_classes)),
        "ten_class_accuracy": float(accuracy_score(true_global, predicted_global)),
        "fixed_10_class_macro_f1": float(
            f1_score(
                true_global,
                predicted_global,
                labels=np.arange(10),
                average="macro",
                zero_division=0,
            )
        ),
        "supported_10_class_macro_f1": float(
            f1_score(
                true_global,
                predicted_global,
                labels=supported_classes,
                average="macro",
                zero_division=0,
            )
        ),
        "ten_class_balanced_accuracy": float(
            balanced_accuracy_score(true_global, predicted_global)
        ),
        "ten_class_weighted_f1": float(
            f1_score(
                true_global,
                predicted_global,
                labels=np.arange(10),
                average="weighted",
                zero_division=0,
            )
        ),
        "family_accuracy": float(accuracy_score(true_family, predicted_family)),
        "fixed_3_family_macro_f1": float(
            f1_score(
                true_family,
                predicted_family,
                labels=np.arange(3),
                average="macro",
                zero_division=0,
            )
        ),
        "supported_3_family_macro_f1": float(
            f1_score(
                true_family,
                predicted_family,
                labels=supported_families,
                average="macro",
                zero_division=0,
            )
        ),
        "family_balanced_accuracy": float(
            balanced_accuracy_score(true_family, predicted_family)
        ),
    }


def holm_adjust(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    order = np.argsort(result["p_value"].to_numpy(float), kind="stable")
    adjusted = np.empty(len(result), dtype=float)
    running = 0.0
    count = len(result)
    for rank_index, original_index in enumerate(order):
        candidate = min(
            1.0,
            (count - rank_index) * float(result.iloc[original_index]["p_value"]),
        )
        running = max(running, candidate)
        adjusted[original_index] = running
    result["holm_adjusted_p"] = adjusted
    result["significant_at_0_05"] = result["holm_adjusted_p"] < 0.05
    return result


def paired_statistics(per_well: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for metric in ("supported_macro_f1", "accuracy", "family_accuracy"):
        pivot = per_well.pivot(index="well_id", columns="model_id", values=metric)
        if PRIMARY_MODEL not in pivot:
            raise RuntimeError("Strict GA-GRU results are absent")
        for comparator in models:
            if comparator == PRIMARY_MODEL:
                continue
            difference = pivot[PRIMARY_MODEL].to_numpy(float) - pivot[
                comparator
            ].to_numpy(float)
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
        raise RuntimeError("Strict execution must complete before analysis")
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    completion = json.loads(COMPLETION_PATH.read_text(encoding="utf-8"))
    if sha256_file(Path(__file__)) != plan["source_hashes"]["analysis_code"]:
        raise RuntimeError("Strict analysis code changed after plan freeze")
    if sha256_file(PLAN_PATH) != completion["plan_sha256"]:
        raise RuntimeError("Strict plan changed after execution")
    if completion["status"] != "COMPLETE_PENDING_ANALYSIS":
        raise RuntimeError("Strict execution is incomplete")
    if completion["strict_test_metrics_used_for_any_selection"] is not False:
        raise RuntimeError("Strict execution does not certify test-blind choices")

    expected_models = [str(model["model_id"]) for model in plan["models"]]
    display_names = {
        str(model["model_id"]): str(model["display_name"]) for model in plan["models"]
    }
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

    rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []
    observed_keys: set[tuple[str, str, int, int]] = set()
    target_signatures: dict[tuple[int, str], set[tuple[int, int, str]]] = {}
    prediction_parts: list[pd.DataFrame] = []
    for relative in completion["result_files"]:
        result_path = PROJECT_DIR / str(relative)
        if not result_path.is_file():
            raise RuntimeError(f"Missing strict result: {result_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        key = (
            str(result["model_id"]),
            str(result["well_id"]),
            int(result["split_seed"]),
            int(result["training_seed"]),
        )
        if key in observed_keys:
            raise RuntimeError(f"Duplicate strict result: {key}")
        observed_keys.add(key)
        if result["status"] != "COMPLETE":
            raise RuntimeError(f"Incomplete strict result: {key}")
        if result["strict_test_metrics_used_for_any_selection"] is not False:
            raise RuntimeError(f"Strict test feedback was reported for {key}")

        predictions_path = PROJECT_DIR / str(result["predictions_file"])
        if sha256_file(predictions_path) != result["predictions_sha256"]:
            raise RuntimeError(f"Strict prediction hash mismatch: {predictions_path}")
        predictions = pd.read_csv(predictions_path, encoding="utf-8-sig")
        if len(predictions) != int(result["test_samples"]):
            raise RuntimeError(f"Strict prediction count mismatch: {key}")
        global_classes = tuple(int(value) for value in result["global_classes"])
        recomputed = metric_bundle(predictions, global_classes)
        for metric, value in recomputed.items():
            if not close(value, float(result["metrics"][metric])):
                raise RuntimeError(f"Recomputed strict {metric} differs for {key}")

        cell = (int(result["split_seed"]), str(result["well_id"]))
        signature = set(
            zip(
                predictions["center_row_id"].astype(int),
                predictions["true_global_class_id"].astype(int),
                predictions["lithology_interval_id"].astype(str),
            )
        )
        if cell in target_signatures and target_signatures[cell] != signature:
            raise RuntimeError(f"Models used different strict test targets for {cell}")
        target_signatures[cell] = signature

        row = {
            "model_id": result["model_id"],
            "display_name": result["display_name"],
            "model_family": result["model_family"],
            "architecture": result["architecture"],
            "well_id": result["well_id"],
            "split_seed": int(result["split_seed"]),
            "training_seed": int(result["training_seed"]),
            "modeled_classes": len(global_classes),
            "supported_test_classes": int(
                predictions["true_global_class_id"].nunique()
            ),
            "test_intervals": int(predictions["lithology_interval_id"].nunique()),
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

        true_global = predictions["true_global_class_id"].to_numpy(np.int64)
        predicted_global = predictions["predicted_global_class_id"].to_numpy(np.int64)
        precision, recall, class_f1, support = precision_recall_fscore_support(
            true_global,
            predicted_global,
            labels=np.arange(10),
            zero_division=0,
        )
        for class_id in range(10):
            per_class_rows.append(
                {
                    "model_id": result["model_id"],
                    "well_id": result["well_id"],
                    "split_seed": int(result["split_seed"]),
                    "training_seed": int(result["training_seed"]),
                    "global_class_id": class_id,
                    "precision": float(precision[class_id]),
                    "recall": float(recall[class_id]),
                    "f1": float(class_f1[class_id]),
                    "support": int(support[class_id]),
                }
            )
        prediction_parts.append(predictions)

    if observed_keys != expected_keys:
        missing = sorted(expected_keys.difference(observed_keys))
        extra = sorted(observed_keys.difference(expected_keys))
        raise RuntimeError(f"Strict run grid differs; missing={missing}, extra={extra}")
    if len(rows) != int(plan["expected_runs"]):
        raise RuntimeError("Strict metric row count differs from the plan")

    run_frame = pd.DataFrame(rows).sort_values(
        ["model_id", "well_id", "split_seed", "training_seed"], kind="stable"
    )
    prediction_frame = pd.concat(prediction_parts, ignore_index=True)
    per_class_frame = pd.DataFrame(per_class_rows)
    metric_names = (
        "accuracy",
        "supported_macro_f1",
        "fixed_local_macro_f1",
        "balanced_accuracy",
        "weighted_f1",
        "family_accuracy",
        "family_macro_f1",
        "family_balanced_accuracy",
    )
    per_well = run_frame.groupby(
        ["model_id", "display_name", "well_id"], as_index=False
    ).agg(
        **{metric: (metric, "mean") for metric in metric_names},
        accuracy_repeat_sd=("accuracy", "std"),
        supported_macro_f1_repeat_sd=("supported_macro_f1", "std"),
        mean_total_runtime_seconds=("total_runtime_seconds", "mean"),
        mean_selected_epoch=("selected_epoch", "mean"),
        trainable_parameters=("trainable_parameters", "first"),
        mean_tree_nodes=("tree_nodes", "mean"),
        repeats=("accuracy", "size"),
    )

    summary_rows = []
    for (model_id, display_name), subset in per_well.groupby(
        ["model_id", "display_name"], sort=False
    ):
        runs = run_frame.loc[run_frame["model_id"].eq(model_id)]
        summary: dict[str, object] = {
            "model_id": model_id,
            "display_name": display_name,
            "wells": int(len(subset)),
            "runs": int(len(runs)),
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
        ["supported_macro_f1_mean", "accuracy_mean"],
        ascending=False,
        kind="stable",
    )

    pooled_rows = []
    pooled_group_columns = ["model_id", "split_seed", "training_seed"]
    for key, group in prediction_frame.groupby(pooled_group_columns, sort=False):
        model_id, split_seed, training_seed = key
        pooled_rows.append(
            {
                "model_id": str(model_id),
                "display_name": display_names[str(model_id)],
                "split_seed": int(split_seed),
                "training_seed": int(training_seed),
                **pooled_global_bundle(group),
            }
        )
    pooled_repeat = pd.DataFrame(pooled_rows)
    pooled_metric_names = (
        "ten_class_accuracy",
        "fixed_10_class_macro_f1",
        "supported_10_class_macro_f1",
        "ten_class_balanced_accuracy",
        "ten_class_weighted_f1",
        "family_accuracy",
        "fixed_3_family_macro_f1",
        "supported_3_family_macro_f1",
        "family_balanced_accuracy",
    )
    pooled_summary_rows = []
    for (model_id, display_name), group in pooled_repeat.groupby(
        ["model_id", "display_name"], sort=False
    ):
        row: dict[str, object] = {
            "model_id": model_id,
            "display_name": display_name,
            "repeats": int(len(group)),
            "samples_per_repeat_mean": float(group["test_samples"].mean()),
        }
        for metric in pooled_metric_names:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        pooled_summary_rows.append(row)
    pooled_summary = pd.DataFrame(pooled_summary_rows).sort_values(
        ["supported_10_class_macro_f1_mean", "ten_class_accuracy_mean"],
        ascending=False,
        kind="stable",
    )

    paired = paired_statistics(per_well, expected_models)
    supported_per_class = per_class_frame.loc[per_class_frame["support"] > 0].copy()
    per_class_summary = (
        supported_per_class.groupby(["model_id", "global_class_id"], as_index=False)
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

    primary_mask = prediction_frame["split_seed"].eq(
        expected_splits[0]
    ) & prediction_frame["training_seed"].eq(expected_training_seeds[0])
    primary_predictions = prediction_frame.loc[primary_mask].copy()
    confusion_rows = []
    family_confusion_rows = []
    for model_id in expected_models:
        group = primary_predictions.loc[primary_predictions["model_id"].eq(model_id)]
        if group["well_id"].nunique() != len(expected_wells):
            raise RuntimeError(f"Strict primary confusion data incomplete: {model_id}")
        matrix = confusion_matrix(
            group["true_global_class_id"],
            group["predicted_global_class_id"],
            labels=np.arange(10),
        )
        denominators = np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        normalized = matrix / denominators
        for true_class in range(10):
            for predicted_class in range(10):
                confusion_rows.append(
                    {
                        "model_id": model_id,
                        "split_seed": expected_splits[0],
                        "training_seed": expected_training_seeds[0],
                        "true_global_class_id": true_class,
                        "predicted_global_class_id": predicted_class,
                        "count": int(matrix[true_class, predicted_class]),
                        "row_normalized_fraction": float(
                            normalized[true_class, predicted_class]
                        ),
                    }
                )
        family_matrix = confusion_matrix(
            group["true_family_id"],
            group["predicted_family_id"],
            labels=np.arange(3),
        )
        family_denominators = np.maximum(family_matrix.sum(axis=1, keepdims=True), 1)
        family_normalized = family_matrix / family_denominators
        for true_family in range(3):
            for predicted_family in range(3):
                family_confusion_rows.append(
                    {
                        "model_id": model_id,
                        "split_seed": expected_splits[0],
                        "training_seed": expected_training_seeds[0],
                        "true_family_id": true_family,
                        "predicted_family_id": predicted_family,
                        "count": int(family_matrix[true_family, predicted_family]),
                        "row_normalized_fraction": float(
                            family_normalized[true_family, predicted_family]
                        ),
                    }
                )

    artifacts = {
        "run_metrics.csv": run_frame,
        "per_well_model_summary.csv": per_well,
        "model_summary.csv": model_summary,
        "pooled_ten_class_and_family_repeats.csv": pooled_repeat,
        "pooled_ten_class_and_family_summary.csv": pooled_summary,
        "ga_paired_statistics.csv": paired,
        "per_class_summary.csv": per_class_summary,
        "primary_run_ten_class_confusion.csv": pd.DataFrame(confusion_rows),
        "primary_run_family_confusion.csv": pd.DataFrame(family_confusion_rows),
    }
    artifact_paths: list[Path] = []
    for name, frame in artifacts.items():
        path = ANALYSIS_DIR / name
        write_csv(path, frame)
        artifact_paths.append(path)

    ga_per_well = model_summary.loc[model_summary["model_id"].eq(PRIMARY_MODEL)].iloc[0]
    ga_pooled = pooled_summary.loc[pooled_summary["model_id"].eq(PRIMARY_MODEL)].iloc[0]
    final_summary = {
        "status": "STRICT_TEST_ANALYSIS_COMPLETE",
        "reporting_role": plan["reporting_role"],
        "primary_method": PRIMARY_MODEL,
        "ga_gru_per_well_metrics": {
            metric: {
                "mean_over_six_well_means": float(ga_per_well[f"{metric}_mean"]),
                "sd_across_six_well_means": float(
                    ga_per_well[f"{metric}_sd_across_wells"]
                ),
            }
            for metric in (
                "accuracy",
                "supported_macro_f1",
                "balanced_accuracy",
                "family_accuracy",
                "family_macro_f1",
            )
        },
        "ga_gru_pooled_global_metrics": {
            metric: {
                "mean_over_nine_repeats": float(ga_pooled[f"{metric}_mean"]),
                "sd_over_nine_repeats": float(ga_pooled[f"{metric}_sd"]),
            }
            for metric in pooled_metric_names
        },
        "model_ranking_by_per_well_supported_macro_f1": model_summary[
            ["model_id", "supported_macro_f1_mean", "accuracy_mean"]
        ].to_dict(orient="records"),
        "model_ranking_by_pooled_fixed_10_class_macro_f1": pooled_summary[
            ["model_id", "fixed_10_class_macro_f1_mean", "ten_class_accuracy_mean"]
        ].to_dict(orient="records"),
        "ranking_is_descriptive_not_a_new_selection_step": True,
        "strict_test_metrics_used_to_change_pipeline": False,
        "statistical_unit": "six well means; each averages nine repeats",
        "holm_significant_comparisons": paired.loc[paired["significant_at_0_05"]][
            "comparison"
        ].tolist(),
        "interpretation_limit": (
            "held-out complete intervals within already represented wells; not "
            "zero-shot transfer to an unseen well"
        ),
    }
    summary_path = ANALYSIS_DIR / "strict_results_summary.json"
    write_json(summary_path, final_summary)
    artifact_paths.append(summary_path)

    audit = {
        "status": "PASS",
        "planned_runs": int(plan["expected_runs"]),
        "verified_runs": int(len(run_frame)),
        "complete_factorial_grid": True,
        "prediction_hashes_verified": True,
        "all_metrics_independently_recomputed": True,
        "identical_test_centers_labels_and_intervals_across_models": True,
        "strict_test_metrics_used_for_any_selection": False,
        "paired_unit_avoids_training_seed_pseudoreplication": True,
        "analysis_code_matches_pre_test_freeze": True,
        "global_ten_class_and_three_family_metrics_reported": True,
        "artifacts": {path.name: sha256_file(path) for path in artifact_paths},
    }
    write_json(AUDIT_PATH, audit)
    print("Strict evaluation audit: PASS")
    print("\nPer-well supported-class summary")
    print(model_summary.to_string(index=False))
    print("\nPooled global 10-class and 3-family summary")
    print(pooled_summary.to_string(index=False))
    print("\nGA-GRU paired comparisons")
    print(paired.to_string(index=False))
    print(f"Summary: {summary_path}")
    print(f"Audit: {AUDIT_PATH}")


if __name__ == "__main__":
    main()
