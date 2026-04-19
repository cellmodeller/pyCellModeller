"""Unit tests for config validation and simulation callback/division semantics."""

from __future__ import annotations

import torch
import pytest

from pycellmodeller.api.simulation import Simulation
from pycellmodeller.biophysics import TorchBacterium
from pycellmodeller.core.config import SimulationConfig
from pycellmodeller.core.state import SimulationState
from pycellmodeller.programs.base import CellProgram


class OrderingProgram(CellProgram):
    """Program that records hook invocation order for one forced division."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[str] = []

    def setup(self, sim: Simulation) -> None:
        super().setup(sim)
        self.events.append("setup")
        sim.add_cell()

    def initialize_cell(self, cell: object) -> None:
        self.events.append(f"initialize_cell:{getattr(cell, 'id', 'unknown')}")

    def update_cells(self, cells: object) -> None:
        self.events.append("update_cells")
        for cell in cells:
            cell.divide = True

    def on_division(self, parent: object, daughter_a: object, daughter_b: object) -> None:
        _ = parent, daughter_a
        self.events.append(f"on_division:{getattr(daughter_b, 'id', 'unknown')}")


def test_simulation_config_validation_errors() -> None:
    with pytest.raises(ValueError, match="dt must be strictly positive"):
        SimulationConfig(dt=0.0)

    with pytest.raises(ValueError, match="seed must be non-negative"):
        SimulationConfig(seed=-1)

    with pytest.raises(TypeError, match="dtype must be a floating point"):
        SimulationConfig(dtype=torch.int64)


def test_structured_state_initialization_defaults_and_custom_values() -> None:
    state = SimulationState.allocate(capacity=1, device=torch.device("cpu"), dtype=torch.float32)
    cell_id = state.add_cell(
        parent_id=-1,
        position=torch.tensor([1.0, -2.0]),
        velocity=torch.tensor([0.5, 0.25]),
        length=1.3,
        radius=0.6,
        target_volume=2.7,
        growth_rate=0.15,
        color=torch.tensor([0.1, 0.2, 0.3]),
        divide=True,
    )

    assert cell_id == 0
    assert state.n_cells == 1
    assert torch.allclose(state.positions[0], torch.tensor([1.0, -2.0]))
    assert torch.allclose(state.velocities[0], torch.tensor([0.5, 0.25]))
    assert float(state.lengths[0]) == pytest.approx(1.3)
    assert float(state.radii[0]) == pytest.approx(0.6)
    assert float(state.target_volume[0]) == pytest.approx(2.7)
    assert float(state.growth_rate[0]) == pytest.approx(0.15)
    assert torch.allclose(state.color[0], torch.tensor([0.1, 0.2, 0.3]))
    assert bool(state.divide[0]) is True


def test_division_split_correctness_without_noise() -> None:
    model = TorchBacterium(min_length=0.1, partition_noise_std=0.0)
    lengths = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    divide_mask = torch.tensor([False, True, True])

    daughter_a, daughter_b = model.divide(lengths, divide_mask)

    assert torch.allclose(daughter_a, torch.tensor([1.0, 1.0, 1.5]))
    assert torch.allclose(daughter_b, torch.tensor([1.0, 1.0, 1.5]))
    assert torch.allclose((daughter_a + daughter_b)[divide_mask], lengths[divide_mask])


def test_callback_invocation_order_for_single_division_step() -> None:
    config = SimulationConfig(device="cpu", dt=0.1, seed=13)
    biophysics = TorchBacterium(growth_rate=0.0, division_length=10.0, partition_noise_std=0.0)
    program = OrderingProgram()
    sim = Simulation(config=config, biophysics=biophysics, program=program)

    sim.initialize()
    sim.step()

    assert program.events[0] == "setup"
    assert program.events[1].startswith("initialize_cell:")
    assert "update_cells" in program.events
    update_index = program.events.index("update_cells")
    division_init_index = next(i for i, event in enumerate(program.events) if event.startswith("initialize_cell:") and i > 1)
    on_division_index = next(i for i, event in enumerate(program.events) if event.startswith("on_division:"))

    assert update_index < division_init_index < on_division_index
