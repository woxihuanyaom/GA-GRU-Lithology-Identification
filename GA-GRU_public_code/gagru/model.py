from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class GRUModelConfig:
    input_size: int = 5
    hidden_size: int = 64
    num_layers: int = 2
    output_size: int = 10
    dropout: float = 0.2

    def __post_init__(self) -> None:
        if self.input_size < 1 or self.hidden_size < 1 or self.num_layers < 1:
            raise ValueError("GRU dimensions must be positive")
        if self.output_size < 2:
            raise ValueError("output_size must be at least two")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def effective_dropout(self) -> float:
        return self.dropout if self.num_layers > 1 else 0.0

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["effective_dropout"] = self.effective_dropout
        return values


class GRUClassifier(nn.Module):
    """Unidirectional many-to-one GRU using the complete centered window."""

    def __init__(self, config: GRUModelConfig) -> None:
        super().__init__()
        self.config = config
        self.gru = nn.GRU(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.effective_dropout,
            bidirectional=False,
        )
        self.classifier = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim != 3 or X.shape[-1] != self.config.input_size:
            raise ValueError(
                f"Expected [batch, sequence, {self.config.input_size}], got {tuple(X.shape)}"
            )
        output, _ = self.gru(X)
        return self.classifier(output[:, -1, :])

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
