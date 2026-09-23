"""Audit and freeze equal-budget GA, random-search, and TPE winners."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
ROOT = PROJECT_DIR / "outputs" / "independent_wells_v5"
OUTPUT_DIR = ROOT / "optimizer_comparison_audit"
METHOD_DIRS = {
    "genetic_algorithm": ROOT / "ga_search",
    "random_search": ROOT / "random_search",
    "tpe": ROOT / "tpe_search",
}

sys.path.insert(0, str(PROJECT_DIR))

from gagru.search import CandidateEvaluation, evaluation_rank  # noqa: E402


def read_log(path: Path) -> list[CandidateEvaluation]:
    evaluations = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            evaluations.append(CandidateEvaluation.from_dict(json.loads(line)))
        except Exception as exc:
            raise RuntimeError(f"Invalid log line {line_number}: {path}") from exc
    return evaluations


def main() -> None:
    method_rows = []
    candidate_rows = []
    winner_well_parts = []
    freezes: dict[str, object] = {}
    reference_constraints: dict[str, object] | None = None

    for method, directory in METHOD_DIRS.items():
        manifest_path = directory / "search_manifest.json"
        summary_path = directory / "search_summary.json"
        log_path = directory / "candidate_evaluations.jsonl"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        evaluations = read_log(log_path)
        budget = int(manifest["budget"])
        if len(evaluations) != budget or budget != 14:
            raise RuntimeError(f"{method} did not use the common 14-candidate budget")
        if [item.candidate_index for item in evaluations] != list(range(1, budget + 1)):
            raise RuntimeError(f"{method} candidate indices are not sequential")
        if len({item.candidate_key for item in evaluations}) != budget:
            raise RuntimeError(f"{method} contains duplicate candidates")
        if manifest["test_assignment_files_opened"] is not False:
            raise RuntimeError(f"{method} manifest does not lock test assignments")
        if summary["test_metrics_read_for_selection"] is not False:
            raise RuntimeError(f"{method} summary does not certify validation-only selection")
        if manifest["samples_shared_between_well_models"] is not False:
            raise RuntimeError(f"{method} permits sample sharing between well models")
        constraints = {
            "budget": budget,
            "split_seed": int(manifest["split_seed"]),
            "base_training_seed": int(manifest["base_training_seed"]),
            "batch_size": int(manifest["batch_size"]),
            "max_epochs": int(manifest["max_epochs"]),
            "patience": int(manifest["patience"]),
            "selection_metric": manifest["selection_metric"],
            "secondary_metric": manifest["secondary_metric"],
            "search_space": manifest["search_space"],
            "warm_start_candidates": manifest["warm_start_candidates"],
            "wells": manifest["wells"],
            "features": manifest["features"],
            "window_length": int(manifest["window_length"]),
        }
        if reference_constraints is None:
            reference_constraints = constraints
        elif constraints != reference_constraints:
            raise RuntimeError(f"{method} search constraints differ from other methods")

        for evaluation in evaluations:
            per_well = pd.DataFrame(evaluation.metrics["per_well"])
            recomputed = {
                "mean_macro_f1": float(per_well["macro_f1"].mean()),
                "mean_accuracy": float(per_well["accuracy"].mean()),
                "mean_balanced_accuracy": float(per_well["balanced_accuracy"].mean()),
                "pooled_accuracy": float(
                    np.average(
                        per_well["accuracy"], weights=per_well["validation_samples"]
                    )
                ),
            }
            recorded = {
                "mean_macro_f1": evaluation.selection_score,
                "mean_accuracy": evaluation.metrics["mean_per_well_accuracy"],
                "mean_balanced_accuracy": evaluation.balanced_accuracy,
                "pooled_accuracy": evaluation.metrics["pooled_accuracy"],
            }
            for name in recomputed:
                if not math.isclose(
                    recomputed[name], float(recorded[name]), abs_tol=1e-12
                ):
                    raise RuntimeError(
                        f"{method} candidate {evaluation.candidate_index} {name} differs"
                    )
            candidate_rows.append(
                {
                    "method": method,
                    "candidate_index": evaluation.candidate_index,
                    **evaluation.candidate.to_dict(),
                    **recomputed,
                    "charged_runtime_seconds": evaluation.runtime_seconds,
                    "evaluation_reused": bool(
                        evaluation.provenance.get("evaluation_reused", False)
                    ),
                }
            )

        winner = max(evaluations, key=evaluation_rank)
        if winner.candidate_index != int(summary["best_candidate_index"]):
            raise RuntimeError(f"{method} recomputed winner differs from summary")
        if winner.candidate.to_dict() != summary["best_candidate"]:
            raise RuntimeError(f"{method} winner parameters differ from summary")
        winner_wells = pd.DataFrame(winner.metrics["per_well"])
        winner_wells.insert(0, "method", method)
        winner_well_parts.append(winner_wells)
        charged_runtime = float(sum(item.runtime_seconds for item in evaluations))
        method_rows.append(
            {
                "method": method,
                "winner_candidate_index": winner.candidate_index,
                **winner.candidate.to_dict(),
                "validation_mean_per_well_macro_f1": winner.selection_score,
                "validation_mean_per_well_accuracy": winner.metrics[
                    "mean_per_well_accuracy"
                ],
                "validation_pooled_accuracy": winner.metrics["pooled_accuracy"],
                "validation_mean_per_well_balanced_accuracy": winner.balanced_accuracy,
                "winner_total_parameters_across_six_models": winner.trainable_parameters,
                "winner_parameters_mean_per_model": winner.trainable_parameters / 6,
                "charged_search_training_runtime_seconds": charged_runtime,
                "charged_search_training_runtime_minutes": charged_runtime / 60,
                "winner_was_warm_start": bool(
                    winner.provenance.get("warm_start", False)
                ),
            }
        )
        freezes[method] = {
            "candidate_index": winner.candidate_index,
            "candidate": winner.candidate.to_dict(),
            "validation_mean_per_well_macro_f1": winner.selection_score,
            "validation_mean_per_well_accuracy": winner.metrics[
                "mean_per_well_accuracy"
            ],
            "validation_pooled_accuracy": winner.metrics["pooled_accuracy"],
            "search_manifest": str(manifest_path),
            "search_summary": str(summary_path),
            "test_metrics_used_for_selection": False,
        }

    methods = pd.DataFrame(method_rows).sort_values(
        [
            "validation_mean_per_well_macro_f1",
            "validation_mean_per_well_balanced_accuracy",
        ],
        ascending=False,
        kind="stable",
    )
    winners = pd.concat(winner_well_parts, ignore_index=True)
    pivot_accuracy = winners.pivot(index="well_id", columns="method", values="accuracy")
    pivot_macro = winners.pivot(index="well_id", columns="method", values="macro_f1")
    paired_rows = []
    for comparator in ("random_search", "tpe"):
        for well_id in reference_constraints["wells"]:
            paired_rows.append(
                {
                    "comparison": f"genetic_algorithm_minus_{comparator}",
                    "well_id": well_id,
                    "accuracy_gain": float(
                        pivot_accuracy.loc[well_id, "genetic_algorithm"]
                        - pivot_accuracy.loc[well_id, comparator]
                    ),
                    "macro_f1_gain": float(
                        pivot_macro.loc[well_id, "genetic_algorithm"]
                        - pivot_macro.loc[well_id, comparator]
                    ),
                }
            )
    paired = pd.DataFrame(paired_rows)
    paired_summary = (
        paired.groupby("comparison", as_index=False)
        .agg(
            mean_accuracy_gain=("accuracy_gain", "mean"),
            wells_accuracy_higher=("accuracy_gain", lambda values: int((values > 0).sum())),
            mean_macro_f1_gain=("macro_f1_gain", "mean"),
            wells_macro_f1_higher=("macro_f1_gain", lambda values: int((values > 0).sum())),
        )
    )
    freeze = {
        "status": "FROZEN_AFTER_EQUAL_BUDGET_VALIDATION_SEARCHES",
        "common_constraints": reference_constraints,
        "methods": freezes,
        "primary_selection_metric": reference_constraints["selection_metric"],
        "test_assignment_files_opened": False,
        "test_metrics_used_for_selection": False,
        "final_evaluation_rule": (
            "Use all three pre-frozen data split seeds. For each method, well, split seed, "
            "and model seed, select an epoch on validation centers, refit from scratch on "
            "train plus validation centers for that epoch count, and evaluate test centers "
            "once. No parameter changes follow test access."
        ),
    }
    audit = {
        "status": "PASS",
        "methods": list(METHOD_DIRS),
        "equal_candidate_budget": True,
        "equal_search_space": True,
        "equal_data_split": True,
        "equal_training_seed_policy": True,
        "equal_warm_start_policy": True,
        "validation_metrics_recomputed": True,
        "test_assignment_files_opened": False,
        "test_metrics_used_for_selection": False,
        "validation_best_macro_f1_method": str(methods.iloc[0]["method"]),
        "validation_best_mean_accuracy_method": str(
            methods.sort_values(
                "validation_mean_per_well_accuracy", ascending=False
            ).iloc[0]["method"]
        ),
        "validation_best_pooled_accuracy_method": str(
            methods.sort_values("validation_pooled_accuracy", ascending=False).iloc[0][
                "method"
            ]
        ),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    methods.to_csv(
        OUTPUT_DIR / "validation_optimizer_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(candidate_rows).to_csv(
        OUTPUT_DIR / "all_candidates.csv", index=False, encoding="utf-8-sig"
    )
    winners.to_csv(
        OUTPUT_DIR / "winner_per_well_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    paired.to_csv(
        OUTPUT_DIR / "ga_paired_validation_gains.csv",
        index=False,
        encoding="utf-8-sig",
    )
    paired_summary.to_csv(
        OUTPUT_DIR / "ga_paired_validation_gain_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (OUTPUT_DIR / "optimizer_winner_freeze.json").write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Equal-budget optimizer audit: PASS")
    print(methods.to_string(index=False))
    print("\nGA paired validation gains")
    print(paired_summary.to_string(index=False))
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"Freeze: {OUTPUT_DIR / 'optimizer_winner_freeze.json'}")


if __name__ == "__main__":
    main()
