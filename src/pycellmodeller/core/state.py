"""State containers for simulation runtime data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(slots=True)
class SimulationState:
    """Tensor-backed simulation state."""

    positions: torch.Tensor
    velocities: torch.Tensor
    time: float = 0.0
    step_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.positions.ndim != 2:
            msg = "positions must be a 2D tensor"
            raise ValueError(msg)

        if self.velocities.shape != self.positions.shape:
            msg = "velocities must match positions shape"
            raise ValueError(msg)

        if self.positions.device != self.velocities.device:
            msg = "positions and velocities must be on the same device"
            raise ValueError(msg)

        if self.positions.dtype != self.velocities.dtype:
            msg = "positions and velocities must have the same dtype"
            raise ValueError(msg)

        if self.time < 0.0:
            msg = "time must be non-negative"
            raise ValueError(msg)

        if self.step_index < 0:
            msg = "step_index must be non-negative"
            raise ValueError(msg)
