"""Audit and freeze equal-budget optimizers for the final plain GRU model."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
ROOT = PROJECT_DIR / "outputs" / "independent_wells_v5" / "plain_gru_searches"
OUTPUT_DIR = ROOT / "audit"
METHOD_DIRS = {method: ROOT / method for method in ("ga", "random", "tpe")}


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


def rank(record: dict[str, Any]) -> tuple[float, float, float, int, float]:
    return (
        float(record["mean_per_well_macro_f1"]),
        float(record["mean_per_well_balanced_accuracy"]),
        float(record["mean_per_well_accuracy"]),
        -int(record["total_parameters_across_six_models"]),
        -float(record["charged_runtime_seconds"]),
    )


def common_constraints(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": manifest["model"],
        "budget": int(manifest["budget"]),
        "split_seed": int(manifest["split_seed"]),
        "base_training_seed": int(manifest["base_training_seed"]),
        "batch_size": int(manifest["batch_size"]),
        "max_epochs": int(manifest["max_epochs"]),
        "patience": int(manifest["patience"]),
        "selection_metric": manifest["selection_metric"],
        "secondary_metric": manifest["secondary_metric"],
        "search_space": manifest["search_space"],
        "warm_start_candidates": manifest["warm_start_candidates"],
        "warm_start_candidates_count_toward_budget": manifest[
            "warm_start_candidates_count_toward_budget"
        ],
        "wells": manifest["wells"],
        "features": manifest["features"],
        "derived_channels": manifest["derived_channels"],
        "window_length": int(manifest["window_length"]),
    }


def main() -> None:
    method_rows = []
    candidate_rows = []
    winner_well_parts = []
    method_freezes: dict[str, object] = {}
    reference: dict[str, Any] | None = None

    for method, directory in METHOD_DIRS.items():
        manifest_path = directory / "search_manifest.json"
        summary_path = directory / "search_summary.json"
        log_path = directory / "candidate_evaluations.jsonl"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        records = read_jsonl(log_path)
        constraints = common_constraints(manifest)
        if reference is None:
            reference = constraints
        elif constraints != reference:
            raise RuntimeError(f"{method} does not share common search constraints")
        if constraints["budget"] != 14 or len(records) != 14:
            raise RuntimeError(f"{method} does not contain 14 candidate evaluations")
        if [int(record["candidate_index"]) for record in records] != list(range(1, 15)):
            raise RuntimeError(f"{method} candidate indices are not sequential")
        if len({record["candidate_key"] for record in records}) != 14:
            raise RuntimeError(f"{method} candidate list contains duplicates")
        if manifest["test_assignment_files_opened"] is not False:
            raise RuntimeError(f"{method} manifest does not certify locked test centers")
        if summary["test_metrics_used_for_selection"] is not False:
            raise RuntimeError(f"{method} used test metrics for selection")
        if manifest["samples_shared_between_well_models"] is not False:
            raise RuntimeError(f"{method} permits sample sharing between well models")

        for record in records:
            per_well = pd.DataFrame(record["per_well"])
            if set(per_well["well_id"].astype(str)) != set(reference["wells"]):
                raise RuntimeError(
                    f"{method} candidate {record['candidate_index']} lacks six wells"
                )
            recomputed = {
                "mean_per_well_macro_f1": float(per_well["macro_f1"].mean()),
                "mean_per_well_accuracy": float(per_well["accuracy"].mean()),
                "pooled_accuracy": float(
                    np.average(
                        per_well["accuracy"], weights=per_well["validation_samples"]
                    )
                ),
                "mean_per_well_balanced_accuracy": float(
                    per_well["balanced_accuracy"].mean()
                ),
                "total_parameters_across_six_models": int(
                    per_well["trainable_parameters"].sum()
                ),
                "charged_runtime_seconds": float(per_well["runtime_seconds"].sum()),
            }
            for name, value in recomputed.items():
                if name == "total_parameters_across_six_models":
                    if int(record[name]) != value:
                        raise RuntimeError(
                            f"{method} candidate {record['candidate_index']} {name} differs"
                        )
                elif not math.isclose(float(record[name]), value, abs_tol=1e-12):
                    raise RuntimeError(
                        f"{method} candidate {record['candidate_index']} {name} differs"
                    )
            candidate_rows.append(
                {
                    "method": method,
                    "candidate_index": int(record["candidate_index"]),
                    **record["candidate"],
                    **recomputed,
                    "evaluation_reused": bool(
                        record["provenance"].get("evaluation_reused", False)
                    ),
                }
            )

        winner = max(records, key=rank)
        if int(summary["best_candidate_index"]) != int(winner["candidate_index"]):
            raise RuntimeError(f"{method} recomputed winner index differs")
        if summary["best_candidate"] != winner["candidate"]:
            raise RuntimeError(f"{method} recomputed winner parameters differ")
        well_frame = pd.DataFrame(winner["per_well"])
        well_frame.insert(0, "method", method)
        winner_well_parts.append(well_frame)
        method_rows.append(
            {
                "method": method,
                "winner_candidate_index": int(winner["candidate_index"]),
                **winner["candidate"],
                "validation_mean_per_well_macro_f1": winner[
                    "mean_per_well_macro_f1"
                ],
                "validation_mean_per_well_accuracy": winner[
                    "mean_per_well_accuracy"
                ],
                "validation_pooled_accuracy": winner["pooled_accuracy"],
                "validation_mean_per_well_balanced_accuracy": winner[
                    "mean_per_well_balanced_accuracy"
                ],
                "winner_parameters_mean_per_model": winner[
                    "total_parameters_across_six_models"
                ]
                / 6,
                "charged_search_runtime_minutes": float(
                    sum(record["charged_runtime_seconds"] for record in records) / 60
                ),
                "winner_was_warm_start": bool(
                    winner["provenance"].get("warm_start", False)
                ),
            }
        )
        method_freezes[method] = {
            "candidate_index": int(winner["candidate_index"]),
            "candidate": winner["candidate"],
            "validation_mean_per_well_macro_f1": winner[
                "mean_per_well_macro_f1"
            ],
            "validation_mean_per_well_accuracy": winner[
                "mean_per_well_accuracy"
            ],
            "validation_pooled_accuracy": winner["pooled_accuracy"],
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
    accuracy = winners.pivot(index="well_id", columns="method", values="accuracy")
    macro = winners.pivot(index="well_id", columns="method", values="macro_f1")
    paired_rows = []
    for comparator in ("random", "tpe"):
        for well_id in reference["wells"]:
            paired_rows.append(
                {
                    "comparison": f"ga_minus_{comparator}",
                    "well_id": well_id,
                    "accuracy_gain": float(
                        accuracy.loc[well_id, "ga"]
                        - accuracy.loc[well_id, comparator]
                    ),
                    "macro_f1_gain": float(
                        macro.loc[well_id, "ga"] - macro.loc[well_id, comparator]
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
        "status": "FROZEN_AFTER_EQUAL_BUDGET_PLAIN_GRU_SEARCHES",
        "model": "unidirectional_many_to_one_GRU",
        "common_constraints": reference,
        "methods": method_freezes,
        "primary_selection_metric": reference["selection_metric"],
        "test_assignment_files_opened": False,
        "test_metrics_used_for_selection": False,
        "final_training_seeds": [17, 29, 43],
        "final_split_seeds": [20260917, 20260918, 20260919],
        "final_evaluation_rule": (
            "For each method, well, split seed, and training seed, choose an epoch using "
            "training and validation centers, refit from scratch on their union, then score "
            "the untouched test centers once."
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
        "all_metrics_recomputed": True,
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
    (OUTPUT_DIR / "plain_gru_optimizer_winner_freeze.json").write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Plain-GRU optimizer audit: PASS")
    print(methods.to_string(index=False))
    print("\nGA paired validation gains")
    print(paired_summary.to_string(index=False))
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"Freeze: {OUTPUT_DIR / 'plain_gru_optimizer_winner_freeze.json'}")


if __name__ == "__main__":
    main()
