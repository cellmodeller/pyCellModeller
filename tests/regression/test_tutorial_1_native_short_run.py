"""Regression test for Tutorial 1 native short-run behavior."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _load_tutorial_module() -> object:
    tutorial_path = Path(__file__).resolve().parents[2] / "examples" / "tutorials" / "tutorial_1_native.py"
    spec = importlib.util.spec_from_file_location("tutorial_1_native", tutorial_path)
    if spec is None or spec.loader is None:
        msg = "Unable to load tutorial_1_native module"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tutorial_1_native_short_run_spreads_colony_and_dumps_frames(tmp_path: Path) -> None:
    torch.manual_seed(7)
    tutorial = _load_tutorial_module()

    simulation, summary = tutorial.run_tutorial(
        steps=12,
        dt=0.2,
        seed=7,
        frame_every=3,
        frame_dir=tmp_path / "frames",
        dump_frames=True,
    )
    state = simulation.state
    active = state.active_slice()

    assert summary["final_cell_count"] > 1
    assert state.n_cells > 1

    pos = state.positions[active]
    x_extent = float(torch.max(pos[:, 0]).item() - torch.min(pos[:, 0]).item())
    y_extent = float(torch.max(pos[:, 1]).item() - torch.min(pos[:, 1]).item())
    assert (x_extent > 0.0) or (y_extent > 0.0)

    unique_positions = torch.unique(pos, dim=0)
    assert unique_positions.shape[0] > 1

    frames = sorted((tmp_path / "frames").glob("frame_*.png"))
    assert len(frames) >= 1
