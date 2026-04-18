"""Integration tests for simulation initialization on CPU."""

from __future__ import annotations

import torch

from pycellmodeller.api.simulation import Simulation
from pycellmodeller.core.config import SimulationConfig


def test_initialize_example_creates_expected_tensor_layout() -> None:
    """Initialization should create 2x2 tensors with configured dtype/device."""
    config = SimulationConfig(device="cpu", dtype=torch.float64)
    simulation = Simulation(config)

    state = simulation.initialize_example()

    assert state.positions.shape == (2, 2)
    assert state.velocities.shape == (2, 2)
    assert state.positions.dtype is torch.float64
    assert state.velocities.dtype is torch.float64
    assert state.positions.device.type == "cpu"
    assert state.velocities.device.type == "cpu"
