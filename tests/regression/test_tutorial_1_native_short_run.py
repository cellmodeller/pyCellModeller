"""Regression test for Tutorial 1 native short-run behavior."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from pycellmodeller.api.simulation import Simulation
from pycellmodeller.biophysics import TorchBacterium
from pycellmodeller.core.config import SimulationConfig
from pycellmodeller.core.state import SimulationState


def _load_tutorial_program_class() -> type:
    tutorial_path = Path(__file__).resolve().parents[2] / "examples" / "tutorials" / "tutorial_1_native.py"
    spec = importlib.util.spec_from_file_location("tutorial_1_native", tutorial_path)
    if spec is None or spec.loader is None:
        msg = "Unable to load tutorial_1_native module"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Tutorial1Program


def _assert_state_integrity(state: SimulationState) -> None:
    active = state.active_slice()
    n_cells = state.n_cells

    assert n_cells > 0
    assert state.cell_ids[active].numel() == n_cells
    assert state.parent_ids[active].numel() == n_cells
    assert state.positions[active].shape == (n_cells, 2)
    assert state.velocities[active].shape == (n_cells, 2)
    assert state.lengths[active].shape == (n_cells,)
    assert torch.isfinite(state.positions[active]).all()
    assert torch.isfinite(state.velocities[active]).all()
    assert torch.isfinite(state.lengths[active]).all()
    assert torch.isfinite(state.target_volume[active]).all()


def test_tutorial_1_native_short_run_divides_and_preserves_state_integrity() -> None:
    torch.manual_seed(7)
    config = SimulationConfig(device="cpu", dt=0.2, seed=7)
    tutorial_program_cls = _load_tutorial_program_class()
    program = tutorial_program_cls()
    sim = Simulation(
        config=config,
        biophysics=TorchBacterium(growth_rate=1.0, division_length=1.2, partition_noise_std=0.0),
        program=program,
    )

    sim.initialize()
    starting_cells = sim.state.n_cells
    final_state = sim.run(steps=6)

    assert starting_cells == 1
    assert final_state.n_cells > 1
    assert program.division_count >= 1
    _assert_state_integrity(final_state)
