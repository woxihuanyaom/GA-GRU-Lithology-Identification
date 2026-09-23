"""Analyze validation-only convergence and cost of GRU search methods."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_DIR = Path(__file__).resolve().parent
SEARCH_ROOT = (
    PROJECT_DIR / "outputs" / "independent_wells_v5" / "plain_gru_searches"
)
OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "independent_wells_v5"
    / "search_efficiency_analysis"
)
TRACE_PATH = OUTPUT_DIR / "best_so_far_trace.csv"
SUMMARY_PATH = OUTPUT_DIR / "search_efficiency_summary.csv"
FIGURE_PATH = OUTPUT_DIR / "search_convergence.png"
FIGURE_PDF_PATH = OUTPUT_DIR / "search_convergence.pdf"
AUDIT_PATH = OUTPUT_DIR / "analysis_audit.json"

METHODS = ("ga", "random", "tpe")
DISPLAY_NAMES = {
    "ga": "Genetic algorithm",
    "random": "Random search",
    "tpe": "TPE",
}
COLORS = {"ga": "#b6423c", "random": "#238b57", "tpe": "#3567a8"}
THRESHOLDS = (0.900, 0.905, 0.910)


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


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def read_json_lines(path: Path) -> list[dict[str, object]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise RuntimeError(f"Search log is empty: {path}")
    return records


def build_trace(
    method: str, records: list[dict[str, object]]
) -> pd.DataFrame:
    ordered = sorted(records, key=lambda value: int(value["candidate_index"]))
    indices = np.asarray(
        [int(value["candidate_index"]) for value in ordered], dtype=int
    )
    if not np.array_equal(indices, np.arange(1, len(ordered) + 1)):
        raise RuntimeError(f"Non-contiguous candidate indices for {method}")
    score = np.asarray(
        [float(value["mean_per_well_macro_f1"]) for value in ordered],
        dtype=float,
    )
    accuracy = np.asarray(
        [float(value["mean_per_well_accuracy"]) for value in ordered],
        dtype=float,
    )
    runtimes = np.asarray(
        [float(value["charged_runtime_seconds"]) for value in ordered],
        dtype=float,
    )
    best_score = np.maximum.accumulate(score)
    best_source_index = np.empty(len(ordered), dtype=int)
    winner = 0
    for index, value in enumerate(score):
        if value > score[winner]:
            winner = index
        best_source_index[index] = indices[winner]
    return pd.DataFrame(
        {
            "method": method,
            "display_name": DISPLAY_NAMES[method],
            "candidate_index": indices,
            "candidate_macro_f1": score,
            "candidate_accuracy": accuracy,
            "best_so_far_macro_f1": best_score,
            "best_source_candidate_index": best_source_index,
            "candidate_charged_runtime_seconds": runtimes,
            "cumulative_charged_runtime_minutes": np.cumsum(runtimes) / 60.0,
            "warm_start": [
                bool(value.get("provenance", {}).get("warm_start", False))
                for value in ordered
            ],
            "evaluation_reused": [
                bool(value.get("provenance", {}).get("evaluation_reused", False))
                for value in ordered
            ],
        }
    )


def first_threshold(trace: pd.DataFrame, threshold: float) -> tuple[object, object]:
    reached = trace[trace["best_so_far_macro_f1"] >= threshold]
    if reached.empty:
        return pd.NA, np.nan
    first = reached.iloc[0]
    return (
        int(first["candidate_index"]),
        float(first["cumulative_charged_runtime_minutes"]),
    )


def summarize_method(
    method: str, trace: pd.DataFrame, search_summary: dict[str, object]
) -> dict[str, object]:
    winner_index = int(search_summary["best_candidate_index"])
    winner_row = trace[trace["candidate_index"] == winner_index].iloc[0]
    row: dict[str, object] = {
        "method": method,
        "display_name": DISPLAY_NAMES[method],
        "candidate_budget": int(search_summary["budget"]),
        "shared_warm_start_candidates": int(trace["warm_start"].sum()),
        "best_candidate_index": winner_index,
        "best_validation_macro_f1": float(
            search_summary["best_validation_macro_f1"]
        ),
        "best_validation_mean_accuracy": float(
            search_summary["best_validation_mean_accuracy"]
        ),
        "best_validation_pooled_accuracy": float(
            search_summary["best_validation_pooled_accuracy"]
        ),
        "charged_search_runtime_minutes": float(
            search_summary["charged_search_runtime_seconds"]
        )
        / 60.0,
        "actual_wall_runtime_minutes": float(
            search_summary["actual_wall_runtime_seconds"]
        )
        / 60.0,
        "charged_minutes_when_winner_first_seen": float(
            winner_row["cumulative_charged_runtime_minutes"]
        ),
        "best_so_far_at_candidate_4": float(
            trace.loc[trace["candidate_index"] == 4, "best_so_far_macro_f1"].iloc[0]
        ),
        "best_so_far_at_candidate_8": float(
            trace.loc[trace["candidate_index"] == 8, "best_so_far_macro_f1"].iloc[0]
        ),
        "best_so_far_at_final_budget": float(
            trace["best_so_far_macro_f1"].iloc[-1]
        ),
        "mean_best_so_far_after_shared_warm_start": float(
            trace.loc[trace["candidate_index"] > 2, "best_so_far_macro_f1"].mean()
        ),
    }
    for threshold in THRESHOLDS:
        candidate_index, minutes = first_threshold(trace, threshold)
        suffix = str(threshold).replace("0.", "")
        row[f"candidate_index_to_macro_f1_{suffix}"] = candidate_index
        row[f"charged_minutes_to_macro_f1_{suffix}"] = minutes
    return row


def plot_trace(trace: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), constrained_layout=True)
    for method in METHODS:
        part = trace[trace["method"] == method]
        style = {
            "color": COLORS[method],
            "linewidth": 2.0,
            "marker": "o",
            "markersize": 3.5,
            "label": DISPLAY_NAMES[method],
        }
        axes[0].plot(
            part["candidate_index"], part["best_so_far_macro_f1"], **style
        )
        axes[1].plot(
            part["cumulative_charged_runtime_minutes"],
            part["best_so_far_macro_f1"],
            **style,
        )
    axes[0].axvspan(0.5, 2.5, color="#d9d9d9", alpha=0.45, linewidth=0)
    axes[0].text(
        1.5,
        0.755,
        "shared\nwarm start",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    axes[0].set_xlabel("Unique candidate evaluations")
    axes[1].set_xlabel("Cumulative charged GPU training time (min)")
    axes[0].set_ylabel("Best validation mean macro-F1")
    axes[1].set_ylabel("Best validation mean macro-F1")
    axes[0].set_title("Full search trajectory", fontsize=11)
    axes[1].set_title("High-score region (zoomed)", fontsize=11)
    axes[0].set_xlim(1, 14)
    axes[0].set_ylim(0.72, 0.92)
    axes[1].set_ylim(0.900, 0.914)
    for axis in axes:
        axis.grid(True, color="#e2e2e2", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, loc="lower right")
    figure.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    figure.savefig(FIGURE_PDF_PATH, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    traces = []
    summaries = []
    input_hashes: dict[str, str] = {}
    for method in METHODS:
        candidate_path = SEARCH_ROOT / method / "candidate_evaluations.jsonl"
        search_summary_path = SEARCH_ROOT / method / "search_summary.json"
        manifest_path = SEARCH_ROOT / method / "search_manifest.json"
        for path in (candidate_path, search_summary_path, manifest_path):
            input_hashes[str(path.relative_to(PROJECT_DIR))] = sha256_file(path)
        records = read_json_lines(candidate_path)
        search_summary = json.loads(
            search_summary_path.read_text(encoding="utf-8")
        )
        if search_summary["method"] != method:
            raise RuntimeError(f"Search method mismatch for {method}")
        trace = build_trace(method, records)
        if len(trace) != int(search_summary["budget"]):
            raise RuntimeError(f"Candidate budget mismatch for {method}")
        if not np.isclose(
            trace["best_so_far_macro_f1"].iloc[-1],
            float(search_summary["best_validation_macro_f1"]),
        ):
            raise RuntimeError(f"Winner score mismatch for {method}")
        traces.append(trace)
        summaries.append(summarize_method(method, trace, search_summary))

    combined_trace = pd.concat(traces, ignore_index=True)
    summary = pd.DataFrame(summaries).sort_values(
        "best_validation_macro_f1", ascending=False, kind="stable"
    )
    write_csv_atomic(TRACE_PATH, combined_trace)
    write_csv_atomic(SUMMARY_PATH, summary)
    plot_trace(combined_trace)
    audit = {
        "status": "PASS",
        "scope": "existing validation-only optimizer search trajectories",
        "new_model_training": False,
        "test_assignment_files_opened": False,
        "test_metrics_used": False,
        "methods": list(METHODS),
        "candidate_budget_per_method": int(summary["candidate_budget"].iloc[0]),
        "shared_warm_start_candidates": 2,
        "runtime_definition": (
            "charged runtime counts all candidate evaluations, including the two "
            "shared cached warm-start candidates, for a fair compute comparison"
        ),
        "input_hashes": input_hashes,
        "trace_sha256": sha256_file(TRACE_PATH),
        "summary_sha256": sha256_file(SUMMARY_PATH),
        "figure_sha256": sha256_file(FIGURE_PATH),
        "figure_pdf_sha256": sha256_file(FIGURE_PDF_PATH),
        "code_sha256": sha256_file(Path(__file__)),
    }
    write_json_atomic(AUDIT_PATH, audit)
    print("Search-efficiency analysis: PASS")
    print(summary.to_string(index=False))
    print(f"Figure: {FIGURE_PATH}")
    print(f"Audit: {AUDIT_PATH}")


if __name__ == "__main__":
    main()
