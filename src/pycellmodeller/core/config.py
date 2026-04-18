"""Configuration models for simulations."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class SimulationConfig:
    """Runtime configuration for a simulation."""

    device: str | torch.device = "cpu"
    dt: float = 0.1
    seed: int = 0
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)

        if self.dt <= 0.0:
            msg = "dt must be strictly positive"
            raise ValueError(msg)

        if self.seed < 0:
            msg = "seed must be non-negative"
            raise ValueError(msg)

        if not self.dtype.is_floating_point:
            msg = "dtype must be a floating point torch dtype"
            raise TypeError(msg)

        if self.device.type == "cuda" and not torch.cuda.is_available():
            msg = "CUDA device requested but CUDA is not available"
            raise ValueError(msg)
