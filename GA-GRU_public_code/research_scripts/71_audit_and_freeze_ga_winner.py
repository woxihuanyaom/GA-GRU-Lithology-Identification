"""Audit the v5 GA search and freeze its validation-selected winner."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = (
    PROJECT_DIR
    / "experiment_protocol_v5_random_center"
    / "random_center_protocol_v5.json"
)
SEARCH_DIR = PROJECT_DIR / "outputs" / "independent_wells_v5" / "ga_search"
MANIFEST_PATH = SEARCH_DIR / "search_manifest.json"
SUMMARY_PATH = SEARCH_DIR / "search_summary.json"
LOG_PATH = SEARCH_DIR / "candidate_evaluations.jsonl"
OUTPUT_DIR = SEARCH_DIR / "audit"

sys.path.insert(0, str(PROJECT_DIR))

from gagru.data import file_sha256  # noqa: E402
from gagru.search import CandidateEvaluation, evaluation_rank  # noqa: E402


def load_evaluations() -> list[CandidateEvaluation]:
    evaluations = []
    for line_number, line in enumerate(
        LOG_PATH.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            evaluations.append(CandidateEvaluation.from_dict(json.loads(line)))
        except Exception as exc:
            raise RuntimeError(f"Invalid GA log line {line_number}") from exc
    return evaluations


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    evaluations = load_evaluations()
    budget = int(manifest["budget"])
    if len(evaluations) != budget:
        raise RuntimeError("GA log length differs from frozen budget")
    if [item.candidate_index for item in evaluations] != list(range(1, budget + 1)):
        raise RuntimeError("GA candidate indices are not sequential")
    if len({item.candidate_key for item in evaluations}) != budget:
        raise RuntimeError("GA candidate log contains duplicates")
    if manifest["test_assignment_files_opened"] is not False:
        raise RuntimeError("GA manifest does not certify locked test assignments")
    if summary["test_metrics_read_for_selection"] is not False:
        raise RuntimeError("GA summary does not certify validation-only selection")
    if manifest["samples_shared_between_well_models"] is not False:
        raise RuntimeError("GA manifest permits sample sharing between well models")
    if file_sha256(PROTOCOL_PATH) != manifest["source_sha256"][PROTOCOL_PATH.name]:
        raise RuntimeError("Frozen v5 protocol changed after the GA manifest")
    for name in ("random_center.py", "residual_bigru.py"):
        if file_sha256(PROJECT_DIR / "gagru" / name) != manifest["source_sha256"][name]:
            raise RuntimeError(f"{name} changed after the GA manifest")

    leaderboard_rows = []
    per_well_rows = []
    for evaluation in evaluations:
        per_well = pd.DataFrame(evaluation.metrics["per_well"])
        wells = set(per_well["well_id"].astype(str))
        if wells != set(manifest["wells"]) or len(per_well) != len(wells):
            raise RuntimeError(
                f"Candidate {evaluation.candidate_index} lacks exactly one result per well"
            )
        recomputed_macro = float(per_well["macro_f1"].mean())
        recomputed_accuracy = float(per_well["accuracy"].mean())
        recomputed_balanced = float(per_well["balanced_accuracy"].mean())
        recomputed_pooled = float(
            np.average(per_well["accuracy"], weights=per_well["validation_samples"])
        )
        checks = {
            "selection_score": (recomputed_macro, evaluation.selection_score),
            "mean_accuracy": (
                recomputed_accuracy,
                evaluation.metrics["mean_per_well_accuracy"],
            ),
            "balanced_accuracy": (recomputed_balanced, evaluation.balanced_accuracy),
            "pooled_accuracy": (
                recomputed_pooled,
                evaluation.metrics["pooled_accuracy"],
            ),
        }
        for name, (recomputed, recorded) in checks.items():
            if not math.isclose(float(recomputed), float(recorded), abs_tol=1e-12):
                raise RuntimeError(
                    f"Candidate {evaluation.candidate_index} {name} did not recompute"
                )
        if "candidate_index" in per_well:
            if set(per_well["candidate_index"].astype(int)) != {
                evaluation.candidate_index
            }:
                raise RuntimeError("Per-well candidate index differs from the GA log")
        else:
            per_well.insert(0, "candidate_index", evaluation.candidate_index)
        per_well_rows.append(per_well)
        leaderboard_rows.append(
            {
                "candidate_index": evaluation.candidate_index,
                **evaluation.candidate.to_dict(),
                "mean_per_well_macro_f1": recomputed_macro,
                "mean_per_well_accuracy": recomputed_accuracy,
                "pooled_accuracy": recomputed_pooled,
                "mean_per_well_balanced_accuracy": recomputed_balanced,
                "runtime_seconds": evaluation.runtime_seconds,
                "generation": evaluation.provenance.get("generation"),
                "warm_start": evaluation.provenance.get("warm_start"),
            }
        )

    winner = max(evaluations, key=evaluation_rank)
    if winner.candidate_index != int(summary["best_candidate_index"]):
        raise RuntimeError("Recomputed GA winner differs from search summary")
    if winner.candidate.to_dict() != summary["best_candidate"]:
        raise RuntimeError("Winner parameters differ from search summary")
    fixed = evaluations[0]
    transferred = evaluations[1]
    winner_wells = pd.DataFrame(winner.metrics["per_well"]).set_index("well_id")
    fixed_wells = pd.DataFrame(fixed.metrics["per_well"]).set_index("well_id")
    transferred_wells = pd.DataFrame(transferred.metrics["per_well"]).set_index(
        "well_id"
    )
    gain_rows = []
    for well_id in manifest["wells"]:
        gain_rows.append(
            {
                "well_id": well_id,
                "winner_accuracy": float(winner_wells.loc[well_id, "accuracy"]),
                "fixed_accuracy": float(fixed_wells.loc[well_id, "accuracy"]),
                "winner_minus_fixed_accuracy": float(
                    winner_wells.loc[well_id, "accuracy"]
                    - fixed_wells.loc[well_id, "accuracy"]
                ),
                "transferred_accuracy": float(
                    transferred_wells.loc[well_id, "accuracy"]
                ),
                "winner_minus_transferred_accuracy": float(
                    winner_wells.loc[well_id, "accuracy"]
                    - transferred_wells.loc[well_id, "accuracy"]
                ),
                "winner_macro_f1": float(winner_wells.loc[well_id, "macro_f1"]),
                "fixed_macro_f1": float(fixed_wells.loc[well_id, "macro_f1"]),
                "winner_minus_fixed_macro_f1": float(
                    winner_wells.loc[well_id, "macro_f1"]
                    - fixed_wells.loc[well_id, "macro_f1"]
                ),
            }
        )
    gains = pd.DataFrame(gain_rows)

    winner_freeze = {
        "status": "FROZEN_AFTER_VALIDATION_ONLY_GA_SEARCH",
        "source_protocol": str(PROTOCOL_PATH),
        "source_protocol_sha256": file_sha256(PROTOCOL_PATH),
        "source_search_manifest": str(MANIFEST_PATH),
        "source_search_manifest_sha256": file_sha256(MANIFEST_PATH),
        "source_search_summary": str(SUMMARY_PATH),
        "source_search_summary_sha256": file_sha256(SUMMARY_PATH),
        "selection_used_test_metrics": False,
        "candidate_index": winner.candidate_index,
        "candidate": winner.candidate.to_dict(),
        "selection_metric": manifest["selection_metric"],
        "validation_mean_per_well_macro_f1": winner.selection_score,
        "validation_mean_per_well_accuracy": winner.metrics[
            "mean_per_well_accuracy"
        ],
        "validation_pooled_accuracy": winner.metrics["pooled_accuracy"],
        "winner_generation": winner.provenance["generation"],
        "winner_was_warm_start": winner.provenance["warm_start"],
        "well_specific_best_epochs_for_search_seed": {
            str(row["well_id"]): int(row["best_epoch"])
            for row in winner.metrics["per_well"]
        },
        "final_evaluation_rule": (
            "For each data split and model seed, select epochs using validation centers, "
            "refit from scratch on train plus validation centers for that epoch count, then "
            "evaluate the untouched test centers once."
        ),
        "model_family": "residual bidirectional GRU with attention and window branch",
        "features": manifest["features"],
        "window_length": manifest["window_length"],
        "independent_model_per_well": True,
    }
    audit = {
        "status": "PASS",
        "candidate_log_rows": len(evaluations),
        "unique_candidates": len({item.candidate_key for item in evaluations}),
        "all_metrics_recomputed": True,
        "winner_recomputed_from_frozen_ranking": True,
        "test_assignment_files_opened_by_search": False,
        "test_metrics_used_for_selection": False,
        "samples_shared_between_well_models": False,
        "winner_candidate_index": winner.candidate_index,
        "winner_improves_accuracy_over_fixed_in_wells": int(
            (gains["winner_minus_fixed_accuracy"] > 0).sum()
        ),
        "winner_improves_accuracy_over_transferred_in_wells": int(
            (gains["winner_minus_transferred_accuracy"] > 0).sum()
        ),
        "wells": len(protocol["research_scope"]["wells"]),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(leaderboard_rows).sort_values(
        ["mean_per_well_macro_f1", "mean_per_well_balanced_accuracy"],
        ascending=False,
    ).to_csv(OUTPUT_DIR / "candidate_leaderboard.csv", index=False, encoding="utf-8-sig")
    pd.concat(per_well_rows, ignore_index=True).to_csv(
        OUTPUT_DIR / "candidate_per_well_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    gains.to_csv(OUTPUT_DIR / "winner_paired_gains.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "ga_winner_freeze.json").write_text(
        json.dumps(winner_freeze, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("GA search audit: PASS")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(gains.to_string(index=False))
    print(f"Frozen winner: {OUTPUT_DIR / 'ga_winner_freeze.json'}")


if __name__ == "__main__":
    main()
