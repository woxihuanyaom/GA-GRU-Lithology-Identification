"""Screen independent per-well models on v5 train/validation centers only."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


PROJECT_DIR = Path(__file__).resolve().parent
PROTOCOL_DIR = PROJECT_DIR / "experiment_protocol_v5_random_center"
PROTOCOL_PATH = PROTOCOL_DIR / "random_center_protocol_v5.json"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "independent_wells_v5" / "initial_screen"

TRAINING_SEED = 1701
BATCH_SIZE = 512
MAX_EPOCHS = 80
PATIENCE = 12
TREES = 500

sys.path.insert(0, str(PROJECT_DIR))

from gagru.random_center import (  # noqa: E402
    build_random_center_windows,
    impute_from_training_windows,
    remap_labels,
)
from gagru.residual_bigru import (  # noqa: E402
    build_model,
    fit_with_validation,
    predict_logits,
    prepare_per_well_inputs,
)
from gagru.search import GRUSearchCandidate  # noqa: E402


CANDIDATES = {
    "fixed_residual_bigru": GRUSearchCandidate(
        hidden_size=96,
        num_layers=2,
        learning_rate=7e-4,
        dropout=0.2,
        weight_decay=1e-4,
    ),
    "transferred_ga_candidate_pilot": GRUSearchCandidate(
        hidden_size=128,
        num_layers=3,
        learning_rate=0.0029380455401443366,
        dropout=0.3,
        weight_decay=6.532858135218046e-05,
    ),
}


def metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    labels = np.arange(int(np.max(y_true)) + 1)
    return {
        "samples": int(len(y_true)),
        "correct": int((y_true == prediction).sum()),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "macro_f1": float(
            f1_score(y_true, prediction, labels=labels, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
    }


def read_development_assignments(
    well_id: str, split_seed: int
) -> pd.DataFrame:
    parts = []
    for split in ("train", "validation"):
        path = (
            PROTOCOL_DIR
            / "center_assignments"
            / f"seed_{split_seed}"
            / well_id
            / f"{well_id}_{split}_centers.csv"
        )
        parts.append(pd.read_csv(path, encoding="utf-8-sig"))
    return pd.concat(parts, ignore_index=True)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for model, group in results.groupby("model", sort=False):
        main = group.loc[group["well_id"].eq("贝16")].iloc[0]
        records.append(
            {
                "model": model,
                "mean_per_well_accuracy": float(group["accuracy"].mean()),
                "sd_per_well_accuracy": float(group["accuracy"].std(ddof=1)),
                "pooled_accuracy": float(group["correct"].sum() / group["samples"].sum()),
                "mean_per_well_macro_f1": float(group["macro_f1"].mean()),
                "sd_per_well_macro_f1": float(group["macro_f1"].std(ddof=1)),
                "main_well_accuracy": float(main["accuracy"]),
                "main_well_macro_f1": float(main["macro_f1"]),
                "runtime_seconds": float(group["runtime_seconds"].sum()),
            }
        )
    return pd.DataFrame(records).sort_values(
        ["mean_per_well_macro_f1", "pooled_accuracy"],
        ascending=[False, False],
        kind="stable",
    )


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    split_seed = int(protocol["split"]["primary_model_selection_split_seed"])
    wells = tuple(str(value) for value in protocol["research_scope"]["wells"])
    features = tuple(str(value) for value in protocol["features"]["extended_seven"])
    window_length = int(protocol["windows"]["length"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Formal v5 neural screen requires the project CUDA environment")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    history_dir = OUTPUT_DIR / "histories"
    history_dir.mkdir(exist_ok=True)

    rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    preprocessing_records: dict[str, object] = {}
    started = time.perf_counter()
    for well_id in wells:
        well_record = protocol["well_protocols"][well_id]
        classes = tuple(int(value) for value in well_record["included_classes"])
        frame = pd.read_csv(
            PROTOCOL_DIR / str(well_record["source_snapshot"]),
            encoding="utf-8-sig",
        )
        assignments = read_development_assignments(well_id, split_seed)
        windows = build_random_center_windows(
            frame,
            assignments,
            features,
            window_length,
            requested_splits=("train", "validation"),
        )
        windows, medians = impute_from_training_windows(windows)
        train = windows["train"]
        validation = windows["validation"]
        y_train, mapping = remap_labels(train, classes)
        y_validation, _ = remap_labels(validation, classes)
        X_train, X_validation = prepare_per_well_inputs(train, validation)
        preprocessing_records[well_id] = {
            "training_window_medians": medians,
            "global_to_local_labels": {str(key): value for key, value in mapping.items()},
            "train_centers": int(len(y_train)),
            "validation_centers": int(len(y_validation)),
        }

        tree_started = time.perf_counter()
        tree = ExtraTreesClassifier(
            n_estimators=TREES,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight=None,
            random_state=TRAINING_SEED,
            n_jobs=-1,
        )
        tree.fit(X_train.reshape(len(X_train), -1), y_train)
        tree_prediction = tree.predict(
            X_validation.reshape(len(X_validation), -1)
        ).astype(np.int64)
        tree_metrics = metrics(y_validation, tree_prediction)
        rows.append(
            {
                "model": "ExtraTrees_unweighted",
                "well_id": well_id,
                "classes": len(classes),
                **tree_metrics,
                "runtime_seconds": time.perf_counter() - tree_started,
                "best_epoch": np.nan,
                "epochs_completed": np.nan,
                "trainable_parameters": np.nan,
            }
        )
        inverse = np.asarray(classes, dtype=np.int64)
        predictions.append(
            pd.DataFrame(
                {
                    "model": "ExtraTrees_unweighted",
                    "well_id": well_id,
                    "depth": validation.depths,
                    "center_row_id": validation.center_ids,
                    "true_class_id": inverse[y_validation],
                    "predicted_class_id": inverse[tree_prediction],
                }
            )
        )

        for model_name, candidate in CANDIDATES.items():
            fit = fit_with_validation(
                candidate,
                X_train,
                y_train,
                X_validation,
                y_validation,
                validation.wells,
                device=device,
                seed=TRAINING_SEED,
                batch_size=BATCH_SIZE,
                max_epochs=MAX_EPOCHS,
                patience=PATIENCE,
            )
            model = build_model(
                candidate,
                X_train,
                device,
                output_size=len(classes),
            )
            model.load_state_dict(fit.state_dict)
            prediction = predict_logits(
                model, X_validation, batch_size=BATCH_SIZE, device=device
            ).argmax(axis=1).astype(np.int64)
            model_metrics = metrics(y_validation, prediction)
            rows.append(
                {
                    "model": model_name,
                    "well_id": well_id,
                    "classes": len(classes),
                    **model_metrics,
                    "runtime_seconds": fit.runtime_seconds,
                    "best_epoch": fit.best_epoch,
                    "epochs_completed": fit.epochs_completed,
                    "trainable_parameters": fit.trainable_parameters,
                }
            )
            predictions.append(
                pd.DataFrame(
                    {
                        "model": model_name,
                        "well_id": well_id,
                        "depth": validation.depths,
                        "center_row_id": validation.center_ids,
                        "true_class_id": inverse[y_validation],
                        "predicted_class_id": inverse[prediction],
                    }
                )
            )
            pd.DataFrame(fit.history).to_csv(
                history_dir / f"{well_id}_{model_name}.csv",
                index=False,
                encoding="utf-8-sig",
            )
            print(
                f"{well_id} {model_name}: accuracy={model_metrics['accuracy']:.4f}, "
                f"macro-F1={model_metrics['macro_f1']:.4f}, epoch={fit.best_epoch}",
                flush=True,
            )
            del model
            torch.cuda.empty_cache()
        print(
            f"{well_id} ExtraTrees: accuracy={tree_metrics['accuracy']:.4f}, "
            f"macro-F1={tree_metrics['macro_f1']:.4f}",
            flush=True,
        )
        pd.DataFrame(rows).to_csv(
            OUTPUT_DIR / "well_results.csv", index=False, encoding="utf-8-sig"
        )

    result_frame = pd.DataFrame(rows)
    summary = summarize(result_frame)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(predictions, ignore_index=True).to_csv(
        OUTPUT_DIR / "validation_predictions.csv", index=False, encoding="utf-8-sig"
    )
    audit = {
        "status": "COMPLETE",
        "scope": "training_and_validation_centers_only",
        "protocol": str(PROTOCOL_PATH),
        "split_seed": split_seed,
        "wells_trained_independently": True,
        "samples_shared_between_well_models": False,
        "test_assignment_files_opened": False,
        "test_metrics_computed": False,
        "features": list(features),
        "derived_channels": "three resistivity separations plus first differences",
        "window_length": window_length,
        "device": str(device),
        "training_seed": TRAINING_SEED,
        "candidates": {name: value.to_dict() for name, value in CANDIDATES.items()},
        "transferred_candidate_is_not_formal_v5_ga_result": True,
        "preprocessing": preprocessing_records,
        "runtime_seconds": time.perf_counter() - started,
    }
    (OUTPUT_DIR / "screen_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nIndependent random-center validation summary")
    print(summary.to_string(index=False))
    print("No test assignment file was loaded or scored.")
    print(f"Audit: {OUTPUT_DIR / 'screen_audit.json'}")


if __name__ == "__main__":
    main()
