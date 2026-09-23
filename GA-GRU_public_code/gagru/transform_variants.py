"""Preprocessing variants used only by preregistered sensitivity analyses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .errors import DataValidationError, ExternalTestLockedError


@dataclass(frozen=True)
class RawScalePreprocessor:
    """Standardize raw curves without applying log10 to resistivities."""

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
    ) -> "RawScalePreprocessor":
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 3 or values.shape[0] == 0:
            raise DataValidationError("Training windows must be a non-empty three-dimensional array")
        features = tuple(feature_names)
        wells = tuple(training_wells)
        overlap = set(wells).intersection(locked_external_wells)
        if overlap:
            raise ExternalTestLockedError(
                f"A preprocessor cannot be fitted on locked external wells: {sorted(overlap)}"
            )
        resistivity_set = set(resistivity_names)
        missing = resistivity_set.difference(features)
        if missing:
            raise DataValidationError(f"Unknown resistivity features: {sorted(missing)}")
        indices = tuple(index for index, name in enumerate(features) if name in resistivity_set)
        means = values.mean(axis=(0, 1))
        scales = values.std(axis=(0, 1), ddof=0)
        scales = np.where(scales > 0, scales, 1.0)
        return cls(
            feature_names=features,
            resistivity_indices=indices,
            means=means,
            scales=scales,
            fitted_wells=wells,
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 3 or values.shape[2] != len(self.feature_names):
            raise DataValidationError(
                f"Expected [samples, sequence, {len(self.feature_names)}] input"
            )
        transformed = (values - self.means) / self.scales
        if not np.isfinite(transformed).all():
            raise DataValidationError("Raw-scale standardization produced non-finite values")
        return transformed.astype(np.float32)

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_names": list(self.feature_names),
            "resistivity_indices": list(self.resistivity_indices),
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "fitted_wells": list(self.fitted_wells),
            "transform_mode": "standardize_without_log10",
            "log10_applied": False,
        }
