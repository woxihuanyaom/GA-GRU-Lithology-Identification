"""Leakage-safe data foundation for the revised GA-GRU experiments."""

import os

# This must be set before PyTorch creates a CUDA context when deterministic
# algorithms are enabled.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from .data import DataRepository
from .folds import PreparedSplit, prepare_inner_fold, prepare_outer_fold
from .metrics import ClassificationMetrics, classification_metrics, supported_macro_f1
from .model import GRUClassifier, GRUModelConfig
from .pilot import candidate_budget, run_runtime_pilot, run_training_smoke_test
from .preprocessing import CurvePreprocessor, compute_class_weights
from .protocol import FoldSpec, FrozenProtocol, WellSpec, load_frozen_protocol
from .search import (
    ActiveSearchBudget,
    CandidateEvaluation,
    GRUSearchCandidate,
    GRUSearchSpace,
    load_active_gpu_budget,
    run_gru_search,
)
from .validation import validate_development_pipeline, write_validation_report
from .windows import WindowedDataset, build_windows
from .training import TrainingConfig, TrainingResult, train_gru

__all__ = [
    "CurvePreprocessor",
    "ClassificationMetrics",
    "CandidateEvaluation",
    "DataRepository",
    "FoldSpec",
    "FrozenProtocol",
    "GRUClassifier",
    "GRUModelConfig",
    "GRUSearchCandidate",
    "GRUSearchSpace",
    "PreparedSplit",
    "ActiveSearchBudget",
    "TrainingConfig",
    "TrainingResult",
    "WellSpec",
    "WindowedDataset",
    "build_windows",
    "candidate_budget",
    "classification_metrics",
    "compute_class_weights",
    "load_frozen_protocol",
    "load_active_gpu_budget",
    "prepare_inner_fold",
    "prepare_outer_fold",
    "run_runtime_pilot",
    "run_gru_search",
    "run_training_smoke_test",
    "supported_macro_f1",
    "train_gru",
    "validate_development_pipeline",
    "write_validation_report",
]
