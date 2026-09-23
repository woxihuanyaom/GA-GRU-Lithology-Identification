from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .errors import DataValidationError, ExternalTestLockedError


@dataclass(frozen=True)
class CurvePreprocessor:
    feature_names: tuple[str, ...]
    resistivity_indices: tuple[int, ...]
    means: np.ndarray
    scales: np.ndarray
    fitted_wells: tuple[str, ...]

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        feature_names: Sequence[str],
        resistivity_names: Iterable[str],
        *,
        training_wells: Iterable[str],
        locked_external_wells: Iterable[str] = (),
    ) -> "CurvePreprocessor":
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 3 or values.shape[0] == 0:
            raise DataValidationError("Training windows must be a non-empty three-dimensional array")

        features = tuple(feature_names)
        wells = tuple(training_wells)
        locked = set(locked_external_wells)
        overlap = set(wells).intersection(locked)
        if overlap:
            raise ExternalTestLockedError(
                f"A preprocessor cannot be fitted on locked external wells: {sorted(overlap)}"
            )

        resistivity_set = set(resistivity_names)
        missing = resistivity_set.difference(features)
        if missing:
            raise DataValidationError(f"Unknown resistivity features: {sorted(missing)}")
        indices = tuple(index for index, name in enumerate(features) if name in resistivity_set)
        transformed = cls._log_resistivities(values, indices)
        means = transformed.mean(axis=(0, 1))
        scales = transformed.std(axis=(0, 1), ddof=0)
        scales = np.where(scales > 0, scales, 1.0)
        return cls(
            feature_names=features,
            resistivity_indices=indices,
            means=means,
            scales=scales,
            fitted_wells=wells,
        )

    @staticmethod
    def _log_resistivities(X: np.ndarray, indices: Sequence[int]) -> np.ndarray:
        transformed = np.array(X, dtype=np.float64, copy=True)
        for index in indices:
            if np.any(transformed[..., index] <= 0):
                raise DataValidationError("Resistivity must be positive before log10 transformation")
            transformed[..., index] = np.log10(transformed[..., index])
        return transformed

    def transform(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 3 or values.shape[2] != len(self.feature_names):
            raise DataValidationError(
                f"Expected [samples, sequence, {len(self.feature_names)}], got {values.shape}"
            )
        transformed = self._log_resistivities(values, self.resistivity_indices)
        transformed = (transformed - self.means) / self.scales
        if not np.isfinite(transformed).all():
            raise DataValidationError("Preprocessing produced non-finite values")
        return transformed.astype(np.float32, copy=False)

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_names": list(self.feature_names),
            "resistivity_indices": list(self.resistivity_indices),
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "fitted_wells": list(self.fitted_wells),
        }


def compute_class_weights(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    labels = np.asarray(y, dtype=np.int64)
    if labels.ndim != 1 or labels.size == 0:
        raise DataValidationError("Labels must be a non-empty one-dimensional array")
    if labels.min() < 0 or labels.max() >= num_classes:
        raise DataValidationError(f"Labels must be in 0-{num_classes - 1}")
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise DataValidationError(f"Training labels do not cover classes: {missing}")
    weights = 1.0 / counts
    weights /= weights.mean()
    return weights.astype(np.float32)
