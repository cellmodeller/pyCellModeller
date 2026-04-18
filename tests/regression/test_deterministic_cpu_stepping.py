"""Regression tests for deterministic CPU stepping behavior."""

from __future__ import annotations

import torch

from pycellmodeller.api.simulation import Simulation
from pycellmodeller.core.config import SimulationConfig


def test_cpu_step_is_deterministic_with_fixed_seed_and_expected_values() -> None:
    """Fixed-seed CPU stepping should produce stable numeric outputs."""
    config = SimulationConfig(device="cpu", dt=0.1, seed=123, dtype=torch.float32)
    simulation = Simulation(config)

    initial_state = simulation.initialize_example()
    stepped_state = simulation.step()

    expected_positions = torch.tensor([[0.025, 0.0], [1.0, 0.025]], dtype=torch.float32)
    expected_velocities = torch.tensor([[0.25, 0.0], [0.0, 0.25]], dtype=torch.float32)

    assert initial_state.positions.device.type == "cpu"
    assert stepped_state.positions.device.type == "cpu"
    assert stepped_state.step_index == 1
    assert stepped_state.time == 0.1
    assert torch.allclose(stepped_state.positions, expected_positions)
    assert torch.equal(stepped_state.velocities, expected_velocities)
