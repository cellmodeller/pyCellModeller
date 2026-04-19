"""Integration tests for an end-to-end simulation using TorchBacterium."""

from __future__ import annotations

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
        sim.add_cell(position=torch.tensor([0.0, 0.0]), velocity=torch.tensor([0.1, 0.0]), direction=torch.tensor([1.0, 0.0]))
        self.seed_cell_count = sim.state.n_cells

    def initialize_cell(self, cell: CellView) -> None:
        cell.target_volume = 2.0
        cell.growth_rate = 1.0
        cell.color = torch.tensor([0.0, 1.0, 0.0])

    def update_cells(self, cells: CellCollection) -> None:
        _ = cells

    def on_division(self, parent: CellView, daughter_a: CellView, daughter_b: CellView) -> None:
        _ = parent, daughter_a, daughter_b
        self.division_count += 1


def test_end_to_end_torch_bacterium_simulation_progression_and_tensor_integrity() -> None:
    torch.manual_seed(123)
    config = SimulationConfig(device="cpu", dt=0.1, seed=123, dtype=torch.float32)
    program = SmallProgram()
    sim = Simulation(
        config=config,
        biophysics=TorchBacterium(growth_rate=1.0, division_length=2.0, partition_noise_std=0.02),
        program=program,
    )

    sim.initialize()
    initial_n_cells = sim.state.n_cells

    pre_division_volume = None
    for _ in range(20):
        state = sim.state
        active = state.active_slice()
        ready = state.lengths[active] >= 2.0
        if pre_division_volume is None and torch.any(ready):
            idx = int(torch.nonzero(ready, as_tuple=False).flatten()[0].item())
            pre_division_volume = float(sim._biophysics.compute_volume(state.lengths[idx:idx+1], state.radii[idx:idx+1])[0].item())
        sim.step()
        if sim.state.n_cells > initial_n_cells:
            break

    final_state = sim.state
    assert program.seed_cell_count == 1
    assert final_state.step_index > 0
    assert final_state.time > 0.0
    assert final_state.n_cells > initial_n_cells
    assert program.division_count >= 1

    active = final_state.active_slice()
    assert final_state.positions.device == config.device
    assert final_state.velocities.device == config.device
    assert final_state.lengths.device == config.device
    assert torch.isfinite(final_state.positions[active]).all()
    assert torch.isfinite(final_state.velocities[active]).all()
    assert torch.isfinite(final_state.lengths[active]).all()
    assert torch.isfinite(final_state.radii[active]).all()
    assert torch.isfinite(final_state.directions[active]).all()

    direction_norms = torch.linalg.vector_norm(final_state.directions[active], dim=1)
    assert torch.allclose(direction_norms, torch.ones_like(direction_norms), atol=1e-5, rtol=1e-5)

    unique_positions = torch.unique(final_state.positions[active], dim=0)
    assert unique_positions.shape[0] > 1

    if pre_division_volume is not None:
        parent_id = int(final_state.parent_ids[1].item())
        daughters = torch.where(final_state.parent_ids[active] == parent_id)[0]
        if daughters.numel() >= 1:
            first_idx = int(daughters[0].item())
            post_volume = float(
                (
                    sim._biophysics.compute_volume(final_state.lengths[0:1], final_state.radii[0:1])
                    + sim._biophysics.compute_volume(final_state.lengths[first_idx:first_idx+1], final_state.radii[first_idx:first_idx+1])
                )[0].item()
            )
            assert abs(post_volume - pre_division_volume) < 1e-5
