"""Unit tests for simulation configuration behavior."""

from __future__ import annotations

import torch

from pycellmodeller.core.config import SimulationConfig


def test_simulation_config_defaults() -> None:
    """Config should provide expected default bootstrap values."""
    config = SimulationConfig()

    assert config.device == torch.device("cpu")
    assert config.dt == 0.1
    assert config.seed == 0
    assert config.dtype is torch.float32


def test_simulation_config_explicit_overrides() -> None:
    """Explicit config overrides should be retained."""
    config = SimulationConfig(device="cpu", dt=0.05, seed=7, dtype=torch.float64)

    assert config.device == torch.device("cpu")
    assert config.dt == 0.05
    assert config.seed == 7
    assert config.dtype is torch.float64
