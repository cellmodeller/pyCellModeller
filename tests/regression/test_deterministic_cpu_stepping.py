"""Regression tests for deterministic CPU stepping behavior."""

from __future__ import annotations

import torch

from pycellmodeller.api.simulation import Simulation
from pycellmodeller.biophysics import TorchBacterium
from pycellmodeller.core.config import SimulationConfig
from pycellmodeller.programs.base import CellProgram


class NoOpProgram(CellProgram):
    pass


def test_cpu_step_is_deterministic_with_fixed_seed_and_expected_values() -> None:
    """Fixed-seed CPU stepping should produce stable numeric outputs."""
    config = SimulationConfig(device="cpu", dt=0.1, seed=123, dtype=torch.float32)
    simulation = Simulation(config, TorchBacterium(growth_rate=0.0, division_length=10.0), NoOpProgram())
    initial_state = simulation.initialize()
    simulation.add_cell(position=torch.tensor([0.0, 0.0]), velocity=torch.tensor([0.25, 0.0]))
    simulation.add_cell(position=torch.tensor([1.0, 0.0]), velocity=torch.tensor([0.0, 0.25]))

    stepped_state = simulation.step()

    expected_positions = torch.tensor([[0.025, 0.0], [1.0, 0.025]], dtype=torch.float32)
    expected_velocities = torch.tensor([[0.25, 0.0], [0.0, 0.25]], dtype=torch.float32)

    assert initial_state.positions.device.type == "cpu"
    assert stepped_state.positions.device.type == "cpu"
    assert stepped_state.step_index == 1
    assert stepped_state.time == 0.1
    assert torch.allclose(stepped_state.positions[:2], expected_positions)
    assert torch.equal(stepped_state.velocities[:2], expected_velocities)
