"""Integration tests for an end-to-end simulation using TorchBacterium."""

from __future__ import annotations

import math

import torch

from pycellmodeller.api.simulation import Simulation
from pycellmodeller.biophysics import TorchBacterium
from pycellmodeller.core.cells import CellCollection, CellView
from pycellmodeller.core.config import SimulationConfig
from pycellmodeller.programs.base import CellProgram


class SmallProgram(CellProgram):
    """Small deterministic program that drives predictable growth/division."""

    def __init__(self) -> None:
        super().__init__()
        self.division_count = 0
        self.seed_cell_count = 0

    def setup(self, sim: Simulation) -> None:
        super().setup(sim)
        sim.add_cell(position=torch.tensor([0.0, 0.0]), velocity=torch.tensor([0.1, 0.0]))
        self.seed_cell_count = sim.state.n_cells

    def initialize_cell(self, cell: CellView) -> None:
        cell.target_volume = 0.8
        cell.growth_rate = 1.0
        cell.color = torch.tensor([0.0, 1.0, 0.0])

    def update_cells(self, cells: CellCollection) -> None:
        for cell in cells:
            volume = math.pi * (cell.radius**2) * cell.length
            cell.divide = volume >= cell.target_volume

    def on_division(self, parent: CellView, daughter_a: CellView, daughter_b: CellView) -> None:
        _ = parent, daughter_a, daughter_b
        self.division_count += 1


def test_end_to_end_torch_bacterium_simulation_progression_and_tensor_integrity() -> None:
    torch.manual_seed(123)
    config = SimulationConfig(device="cpu", dt=0.1, seed=123, dtype=torch.float32)
    program = SmallProgram()
    sim = Simulation(
        config=config,
        biophysics=TorchBacterium(growth_rate=1.0, division_length=100.0, partition_noise_std=0.0),
        program=program,
    )

    sim.initialize()
    initial_n_cells = sim.state.n_cells
    final_state = sim.run(steps=5)

    assert program.seed_cell_count == 1
    assert final_state.step_index == 5
    assert final_state.time == 0.5
    assert final_state.n_cells > initial_n_cells
    assert program.division_count >= 1

    active = final_state.active_slice()
    assert torch.all(final_state.target_volume[active] > 0.0)
    assert final_state.positions.device == config.device
    assert final_state.velocities.device == config.device
    assert final_state.lengths.device == config.device
    assert torch.isfinite(final_state.positions[active]).all()
    assert torch.isfinite(final_state.velocities[active]).all()
    assert torch.isfinite(final_state.lengths[active]).all()
    assert torch.isfinite(final_state.target_volume[active]).all()
