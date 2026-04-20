"""Unit tests for tensor-backed cell state and author-facing views."""

from __future__ import annotations

import pytest
import torch

from pycellmodeller.core.cells import CellCollection
from pycellmodeller.core.state import SimulationState


def test_add_cell_assigns_id_parent_and_biological_defaults() -> None:
    state = SimulationState.allocate(capacity=1, device=torch.device("cpu"), dtype=torch.float32)

    first = state.add_cell(parent_id=-1)
    second = state.add_cell(parent_id=first, growth_rate=0.12, target_volume=2.5, divide=True)

    assert first == 0
    assert second == 1
    assert state.n_cells == 2
    assert state.parent_ids[1].item() == 0
    assert torch.isclose(state.target_volume[1], torch.tensor(2.5))
    assert torch.isclose(state.growth_rate[1], torch.tensor(0.12))
    assert bool(state.divide[1].item()) is True
    assert torch.allclose(state.directions[0], torch.tensor([1.0, 0.0]))


def test_cell_collection_view_supports_read_and_write_access() -> None:
    state = SimulationState.allocate(capacity=2, device=torch.device("cpu"), dtype=torch.float32)
    state.add_cells(2)

    cells = CellCollection(state)
    cell = cells.by_id(1)
    cell.target_volume = 4.0
    cell.growth_rate = 0.25
    cell.color = torch.tensor([0.2, 0.5, 1.0])
    cell.divide = True
    cell.direction = torch.tensor([0.0, 2.0])

    assert cell.id == 1
    assert float(state.target_volume[1]) == 4.0
    assert float(state.growth_rate[1]) == 0.25
    assert torch.allclose(state.color[1], torch.tensor([0.2, 0.5, 1.0]))
    assert bool(state.divide[1]) is True
    assert torch.allclose(state.directions[1], torch.tensor([0.0, 1.0]))
    assert cell.angle == pytest.approx(float(torch.pi / 2), abs=1e-6)
