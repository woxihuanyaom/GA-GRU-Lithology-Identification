from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from .errors import DataValidationError


@dataclass(frozen=True)
class ClassificationMetrics:
    selection_score: float
    per_well_supported_macro_f1: dict[str, float]
    pooled_fixed_macro_f1: float
    accuracy: float
    balanced_accuracy: float
    macro_precision: float
    macro_recall: float
    weighted_f1: float

    def to_dict(self) -> dict[str, object]:
        return {
            "selection_score": self.selection_score,
            "per_well_supported_macro_f1": self.per_well_supported_macro_f1,
            "pooled_fixed_macro_f1": self.pooled_fixed_macro_f1,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "weighted_f1": self.weighted_f1,
        }


def _validated_vectors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    well_ids: Iterable[str],
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    wells = np.asarray(tuple(well_ids), dtype=str)
    if true.ndim != 1 or pred.ndim != 1 or wells.ndim != 1:
        raise DataValidationError("Metric inputs must be one-dimensional")
    if len(true) == 0 or len(true) != len(pred) or len(true) != len(wells):
        raise DataValidationError("Metric inputs must be non-empty and have equal lengths")
    if true.min() < 0 or true.max() >= num_classes:
        raise DataValidationError("Ground-truth labels fall outside the fixed class range")
    if pred.min() < 0 or pred.max() >= num_classes:
        raise DataValidationError("Predicted labels fall outside the fixed class range")
    return true, pred, wells


def supported_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-F1 over only the classes with ground-truth support in one well."""

    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    labels = np.unique(true)
    if len(labels) == 0:
        raise DataValidationError("Cannot score an empty well")
    return float(
        f1_score(true, pred, labels=labels, average="macro", zero_division=0)
    )


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    well_ids: Iterable[str],
    *,
    num_classes: int = 10,
) -> ClassificationMetrics:
    true, pred, wells = _validated_vectors(y_true, y_pred, well_ids, num_classes)
    ordered_wells = tuple(dict.fromkeys(wells.tolist()))
    per_well = {
        well: supported_macro_f1(true[wells == well], pred[wells == well])
        for well in ordered_wells
    }
    fixed_labels = list(range(num_classes))
    return ClassificationMetrics(
        selection_score=float(np.mean(tuple(per_well.values()))),
        per_well_supported_macro_f1=per_well,
        pooled_fixed_macro_f1=float(
            f1_score(true, pred, labels=fixed_labels, average="macro", zero_division=0)
        ),
        accuracy=float(accuracy_score(true, pred)),
        balanced_accuracy=float(
            recall_score(
                true,
                pred,
                labels=np.unique(true),
                average="macro",
                zero_division=0,
            )
        ),
        macro_precision=float(
            precision_score(
                true, pred, labels=fixed_labels, average="macro", zero_division=0
            )
        ),
        macro_recall=float(
            recall_score(true, pred, labels=fixed_labels, average="macro", zero_division=0)
        ),
        weighted_f1=float(f1_score(true, pred, average="weighted", zero_division=0)),
    )
