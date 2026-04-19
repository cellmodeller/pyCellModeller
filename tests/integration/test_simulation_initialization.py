"""Integration tests for simulation initialization on CPU."""

from __future__ import annotations

import torch

from pycellmodeller.api.simulation import Simulation
from pycellmodeller.biophysics import TorchBacterium
from pycellmodeller.core.config import SimulationConfig
from pycellmodeller.programs.base import CellProgram


class NoOpProgram(CellProgram):
    pass


def test_initialize_creates_empty_state_with_configured_dtype_and_device() -> None:
    """Initialization should create empty tensors with configured dtype/device."""
    config = SimulationConfig(device="cpu", dtype=torch.float64)
    simulation = Simulation(config, TorchBacterium(), NoOpProgram())

    state = simulation.initialize()

    assert state.positions.shape == (0, 2)
    assert state.velocities.shape == (0, 2)
    assert state.positions.dtype is torch.float64
    assert state.velocities.dtype is torch.float64
    assert state.positions.device.type == "cpu"
    assert state.velocities.device.type == "cpu"
