"""Audit and summarize repeated GA, random-search, and TPE searches."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = (
    PROJECT_DIR
    / "experiment_protocol_v8_optimizer_repeats"
    / "optimizer_repeat_protocol_v8.json"
)
ROOT = PROJECT_DIR / "outputs" / "optimizer_repeats_v1"
SEARCH_ROOT = ROOT / "searches"
OUTPUT_DIR = ROOT / "analysis"
METHODS = ("genetic_algorithm", "random_search", "tpe")
CHECKPOINTS = (6, 12, 18, 24)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text_atomic(path: Path, value: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding=encoding)
    os.replace(temporary, path)


def write_json_atomic(path: Path, value: object) -> None:
    write_text_atomic(
        path,
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


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
            raise RuntimeError(f"Invalid JSONL at {path}:{line_number}") from exc
    return records


def rank(record: dict[str, Any]) -> tuple[float, float, float, int, float]:
    return (
        float(record["mean_per_well_macro_f1"]),
        float(record["mean_per_well_balanced_accuracy"]),
        float(record["mean_per_well_accuracy"]),
        -int(record["total_parameters_across_six_models"]),
        -float(record["charged_training_runtime_seconds"]),
    )


def best_so_far(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    winner: dict[str, Any] | None = None
    charged = 0.0
    for record in records:
        charged += float(record["charged_training_runtime_seconds"])
        if winner is None or rank(record) > rank(winner):
            winner = record
        rows.append(
            {
                "candidate_index": int(record["candidate_index"]),
                "candidate_macro_f1": float(
                    record["mean_per_well_macro_f1"]
                ),
                "best_so_far_macro_f1": float(
                    winner["mean_per_well_macro_f1"]
                ),
                "best_so_far_candidate_index": int(winner["candidate_index"]),
                "cumulative_charged_runtime_seconds": charged,
            }
        )
    return rows


def descriptive(values: pd.Series, prefix: str) -> dict[str, float]:
    array = values.to_numpy(float)
    return {
        f"mean_{prefix}": float(np.mean(array)),
        f"sd_{prefix}": float(np.std(array, ddof=1)),
        f"median_{prefix}": float(np.median(array)),
        f"minimum_{prefix}": float(np.min(array)),
        f"maximum_{prefix}": float(np.max(array)),
    }


def render_report(
    method_summary: pd.DataFrame,
    paired: pd.DataFrame,
    checkpoint_summary: pd.DataFrame,
    overall_winners: dict[str, int],
) -> str:
    lines = [
        "# Repeated Optimizer Search Analysis",
        "",
        "Three independent searches per method; each search used the same frozen "
        "seven-curve, engineered-feature, training-only SMOTE-Tomek pipeline and "
        "24-candidate budget. No test file or test metric was read.",
        "",
        "## Best validation score distribution",
        "",
        "| Method | Macro-F1 mean ± SD | Range | Charged time, min | Wins |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in method_summary.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.mean_best_validation_macro_f1:.4f} ± "
            f"{row.sd_best_validation_macro_f1:.4f} | "
            f"{row.minimum_best_validation_macro_f1:.4f}-"
            f"{row.maximum_best_validation_macro_f1:.4f} | "
            f"{row.mean_charged_runtime_minutes:.1f} ± "
            f"{row.sd_charged_runtime_minutes:.1f} | "
            f"{overall_winners.get(row.method, 0)}/3 |"
        )
    lines.extend(
        [
            "",
            "## GA paired repeat differences",
            "",
            "| Comparator | Mean macro-F1 delta | Per-repeat GA wins/ties/losses | "
            "Mean charged-time ratio |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in paired.itertuples(index=False):
        lines.append(
            f"| {row.comparator} | {row.mean_macro_f1_delta:+.4f} | "
            f"{row.ga_wins}/{row.ties}/{row.ga_losses} | "
            f"{row.mean_ga_to_comparator_runtime_ratio:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Budget checkpoints",
            "",
            "| Method | Candidate | Mean best-so-far macro-F1 |",
            "|---|---:|---:|",
        ]
    )
    for row in checkpoint_summary.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.candidate_index} | "
            f"{row.mean_best_so_far_macro_f1:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "With only three independent searches, these results quantify descriptive "
            "stability and budget efficiency. They do not establish statistical "
            "superiority. If GA does not lead in score, variability, or convergence, "
            "describe it as a competitive automated tuner rather than a uniquely "
            "superior optimizer.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    if not PROTOCOL_PATH.is_file():
        raise RuntimeError("Repeated-search protocol is missing")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    budget = int(protocol["candidate_budget_per_method_per_repeat"])
    if budget != 24:
        raise RuntimeError("Analysis expects the frozen 24-candidate budget")

    search_rows = []
    trace_rows = []
    configuration_rows = []
    artifact_hashes = {}
    for repeat in protocol["repeats"]:
        repeat_id = int(repeat["repeat_id"])
        initial_keys = [
            json.dumps(value, sort_keys=True, separators=(",", ":"))
            for value in repeat["shared_initial_candidates"]
        ]
        for method in METHODS:
            directory = SEARCH_ROOT / f"repeat_{repeat_id}" / method
            manifest_path = directory / "search_manifest.json"
            log_path = directory / "candidate_evaluations.jsonl"
            summary_path = directory / "search_summary.json"
            for path in (manifest_path, log_path, summary_path):
                if not path.is_file():
                    raise RuntimeError(f"Missing completed search artifact: {path}")
                artifact_hashes[str(path.relative_to(PROJECT_DIR))] = sha256_file(path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            records = read_jsonl(log_path)
            if manifest["test_assignment_files_opened"] is not False:
                raise RuntimeError(f"Test assignment access recorded in {manifest_path}")
            if summary["test_metrics_read"] is not False:
                raise RuntimeError(f"Test metric access recorded in {summary_path}")
            if len(records) != budget:
                raise RuntimeError(f"Incomplete candidate log: {log_path}")
            if [int(value["candidate_index"]) for value in records] != list(
                range(1, budget + 1)
            ):
                raise RuntimeError(f"Nonsequential candidates: {log_path}")
            if len({value["candidate_key"] for value in records}) != budget:
                raise RuntimeError(f"Duplicate candidate in search: {log_path}")
            observed_initial = [value["candidate_key"] for value in records[:6]]
            if observed_initial != initial_keys:
                raise RuntimeError(f"Shared initial design differs: {log_path}")
            winner = max(records, key=rank)
            if int(winner["candidate_index"]) != int(summary["best_candidate_index"]):
                raise RuntimeError(f"Winner does not recompute: {summary_path}")
            charged_seconds = float(
                sum(value["charged_training_runtime_seconds"] for value in records)
            )
            if not np.isclose(
                charged_seconds,
                float(summary["charged_search_training_runtime_seconds"]),
                atol=1e-9,
            ):
                raise RuntimeError(f"Charged runtime does not recompute: {summary_path}")
            search_rows.append(
                {
                    "repeat_id": repeat_id,
                    "base_search_seed": int(repeat["base_search_seed"]),
                    "method": method,
                    "best_candidate_index": int(winner["candidate_index"]),
                    "best_validation_macro_f1": float(
                        winner["mean_per_well_macro_f1"]
                    ),
                    "best_validation_accuracy": float(
                        winner["mean_per_well_accuracy"]
                    ),
                    "best_validation_balanced_accuracy": float(
                        winner["mean_per_well_balanced_accuracy"]
                    ),
                    "charged_runtime_seconds": charged_seconds,
                    "charged_runtime_minutes": charged_seconds / 60.0,
                    "actual_runtime_seconds": float(
                        sum(
                            value["actual_training_runtime_seconds"]
                            for value in records
                        )
                    ),
                    "reused_candidate_evaluations": int(
                        sum(
                            bool(
                                value["provenance"].get(
                                    "evaluation_reused", False
                                )
                            )
                            for value in records
                        )
                    ),
                    **winner["candidate"],
                }
            )
            for row in best_so_far(records):
                trace_rows.append(
                    {
                        "repeat_id": repeat_id,
                        "method": method,
                        **row,
                    }
                )
            configuration_rows.append(
                {
                    "repeat_id": repeat_id,
                    "method": method,
                    "candidate_index": int(winner["candidate_index"]),
                    "candidate_key": winner["candidate_key"],
                    **winner["candidate"],
                }
            )

    searches = pd.DataFrame(search_rows).sort_values(
        ["repeat_id", "method"], kind="stable"
    )
    traces = pd.DataFrame(trace_rows).sort_values(
        ["repeat_id", "method", "candidate_index"], kind="stable"
    )
    configurations = pd.DataFrame(configuration_rows).sort_values(
        ["repeat_id", "method"], kind="stable"
    )

    winner_methods: dict[str, int] = {method: 0 for method in METHODS}
    repeat_winner_rows = []
    for repeat_id, part in searches.groupby("repeat_id", sort=True):
        best_score = float(part["best_validation_macro_f1"].max())
        winners = part.loc[
            np.isclose(part["best_validation_macro_f1"], best_score), "method"
        ].tolist()
        for method in winners:
            winner_methods[str(method)] += 1
        repeat_winner_rows.append(
            {
                "repeat_id": int(repeat_id),
                "best_validation_macro_f1": best_score,
                "winning_methods": ";".join(str(value) for value in winners),
                "tie": len(winners) > 1,
            }
        )

    method_rows = []
    for method, part in searches.groupby("method", sort=False):
        method_rows.append(
            {
                "method": method,
                "independent_searches": int(len(part)),
                **descriptive(
                    part["best_validation_macro_f1"],
                    "best_validation_macro_f1",
                ),
                **descriptive(
                    part["best_validation_accuracy"],
                    "best_validation_accuracy",
                ),
                **descriptive(
                    part["charged_runtime_minutes"],
                    "charged_runtime_minutes",
                ),
                "repeat_wins_including_ties": int(winner_methods[method]),
                "unique_winning_configurations": int(
                    part[
                        [
                            "hidden_size",
                            "num_layers",
                            "learning_rate",
                            "dropout",
                            "weight_decay",
                        ]
                    ].drop_duplicates().shape[0]
                ),
            }
        )
    method_summary = pd.DataFrame(method_rows).sort_values(
        "mean_best_validation_macro_f1", ascending=False, kind="stable"
    )

    paired_rows = []
    ga = searches.loc[searches["method"] == "genetic_algorithm"].set_index(
        "repeat_id"
    )
    for comparator in ("random_search", "tpe"):
        other = searches.loc[searches["method"] == comparator].set_index("repeat_id")
        if not ga.index.equals(other.index):
            raise RuntimeError(f"Unpaired repeats for {comparator}")
        delta = (
            ga["best_validation_macro_f1"]
            - other["best_validation_macro_f1"]
        )
        runtime_ratio = ga["charged_runtime_seconds"] / other[
            "charged_runtime_seconds"
        ]
        paired_rows.append(
            {
                "comparator": comparator,
                "paired_repeats": int(len(delta)),
                "mean_macro_f1_delta": float(delta.mean()),
                "median_macro_f1_delta": float(delta.median()),
                "minimum_macro_f1_delta": float(delta.min()),
                "maximum_macro_f1_delta": float(delta.max()),
                "ga_wins": int((delta > 0).sum()),
                "ties": int(np.isclose(delta, 0.0).sum()),
                "ga_losses": int((delta < 0).sum()),
                "mean_ga_to_comparator_runtime_ratio": float(
                    runtime_ratio.mean()
                ),
                "statistical_test": "not performed; n=3 is descriptive only",
            }
        )
    paired = pd.DataFrame(paired_rows)

    checkpoint_rows = []
    for (method, candidate_index), part in traces.loc[
        traces["candidate_index"].isin(CHECKPOINTS)
    ].groupby(["method", "candidate_index"], sort=False):
        checkpoint_rows.append(
            {
                "method": method,
                "candidate_index": int(candidate_index),
                "mean_best_so_far_macro_f1": float(
                    part["best_so_far_macro_f1"].mean()
                ),
                "sd_best_so_far_macro_f1": float(
                    part["best_so_far_macro_f1"].std(ddof=1)
                ),
                "mean_cumulative_charged_runtime_minutes": float(
                    part["cumulative_charged_runtime_seconds"].mean() / 60.0
                ),
            }
        )
    checkpoint_summary = pd.DataFrame(checkpoint_rows).sort_values(
        ["candidate_index", "method"], kind="stable"
    )

    same_winner_counts = []
    for method, part in configurations.groupby("method", sort=False):
        counts = part["candidate_key"].value_counts()
        same_winner_counts.append(
            {
                "method": method,
                "unique_winning_configurations": int(len(counts)),
                "maximum_repeats_same_configuration": int(counts.max()),
            }
        )
    configuration_stability = pd.DataFrame(same_winner_counts)

    primary_ranking = method_summary["method"].tolist()
    conclusion = {
        "status": "COMPLETE_DESCRIPTIVE_STABILITY_ANALYSIS",
        "test_assignment_files_opened": False,
        "test_metrics_read": False,
        "independent_searches_per_method": 3,
        "candidate_budget_per_search": budget,
        "validation_macro_f1_ranking": primary_ranking,
        "repeat_wins_including_ties": winner_methods,
        "ga_paired_comparisons": paired.to_dict(orient="records"),
        "claim_limit": (
            "Three independent searches quantify descriptive optimizer stability "
            "and budget efficiency; they are insufficient for a statistical "
            "superiority claim."
        ),
        "interpretation": (
            "GA may be described as advantageous only for dimensions supported by "
            "these results: best score, lower variation, earlier convergence, or "
            "comparable performance at acceptable cost. Otherwise describe it as "
            "competitive automated tuning."
        ),
        "artifact_sha256": artifact_hashes,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(OUTPUT_DIR / "search_level_results.csv", searches)
    write_csv_atomic(OUTPUT_DIR / "method_summary.csv", method_summary)
    write_csv_atomic(OUTPUT_DIR / "ga_paired_repeat_differences.csv", paired)
    write_csv_atomic(OUTPUT_DIR / "search_traces.csv", traces)
    write_csv_atomic(OUTPUT_DIR / "budget_checkpoint_summary.csv", checkpoint_summary)
    write_csv_atomic(OUTPUT_DIR / "winning_configurations.csv", configurations)
    write_csv_atomic(
        OUTPUT_DIR / "configuration_stability.csv", configuration_stability
    )
    write_csv_atomic(
        OUTPUT_DIR / "repeat_winners.csv", pd.DataFrame(repeat_winner_rows)
    )
    write_json_atomic(OUTPUT_DIR / "analysis_summary.json", conclusion)
    write_text_atomic(
        OUTPUT_DIR / "analysis_report.md",
        render_report(method_summary, paired, checkpoint_summary, winner_methods),
    )

    print("Repeated optimizer search audit: PASS")
    print(method_summary.to_string(index=False))
    print("\nGA paired repeat differences")
    print(paired.to_string(index=False))
    print(f"Report: {OUTPUT_DIR / 'analysis_report.md'}")


if __name__ == "__main__":
    main()
