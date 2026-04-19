"""Author-facing proxy objects over tensor-backed cell state."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch

from pycellmodeller.core.state import SimulationState


@dataclass(slots=True)
class CellView:
    """Thin mutable view for one cell row inside ``SimulationState``."""

    state: SimulationState
    index: int

    @property
    def id(self) -> int:
        return int(self.state.cell_ids[self.index].item())

    @property
    def parent_id(self) -> int:
        return int(self.state.parent_ids[self.index].item())

    @parent_id.setter
    def parent_id(self, value: int) -> None:
        self.state.parent_ids[self.index] = value

    @property
    def position(self) -> torch.Tensor:
        return self.state.positions[self.index]

    @position.setter
    def position(self, value: torch.Tensor) -> None:
        self.state.positions[self.index] = value.to(
            device=self.state.positions.device,
            dtype=self.state.positions.dtype,
        )

    @property
    def velocity(self) -> torch.Tensor:
        return self.state.velocities[self.index]

    @velocity.setter
    def velocity(self, value: torch.Tensor) -> None:
        self.state.velocities[self.index] = value.to(
            device=self.state.velocities.device,
            dtype=self.state.velocities.dtype,
        )

    @property
    def length(self) -> float:
        return float(self.state.lengths[self.index].item())

    @length.setter
    def length(self, value: float) -> None:
        self.state.lengths[self.index] = value

    @property
    def radius(self) -> float:
        return float(self.state.radii[self.index].item())

    @radius.setter
    def radius(self, value: float) -> None:
        self.state.radii[self.index] = value

    @property
    def target_volume(self) -> float:
        return float(self.state.target_volume[self.index].item())

    @target_volume.setter
    def target_volume(self, value: float) -> None:
        self.state.target_volume[self.index] = value

    @property
    def growth_rate(self) -> float:
        return float(self.state.growth_rate[self.index].item())

    @growth_rate.setter
    def growth_rate(self, value: float) -> None:
        self.state.growth_rate[self.index] = value

    @property
    def color(self) -> torch.Tensor:
        return self.state.color[self.index]

    @color.setter
    def color(self, value: torch.Tensor) -> None:
        self.state.color[self.index] = value.to(device=self.state.color.device, dtype=self.state.color.dtype)

    @property
    def divide(self) -> bool:
        return bool(self.state.divide[self.index].item())

    @divide.setter
    def divide(self, value: bool) -> None:
        self.state.divide[self.index] = value


class CellCollection:
    """Indexed/iterable view over active cells."""

    def __init__(self, state: SimulationState) -> None:
        self._state = state

    def __len__(self) -> int:
        return self._state.n_cells

    def __iter__(self) -> Iterator[CellView]:
        for idx in range(self._state.n_cells):
            yield CellView(self._state, idx)

    def by_index(self, index: int) -> CellView:
        if not 0 <= index < self._state.n_cells:
            msg = "cell index out of range"
            raise IndexError(msg)
        return CellView(self._state, index)

    def by_id(self, cell_id: int) -> CellView:
        matches = torch.where(self._state.cell_ids[: self._state.n_cells] == cell_id)[0]
        if matches.numel() == 0:
            msg = f"unknown cell_id {cell_id}"
            raise KeyError(msg)
        return CellView(self._state, int(matches[0].item()))

    def add(self, **kwargs: object) -> CellView:
        """Create one cell and return a mutable ``CellView`` for callbacks."""
        cell_id = self._state.add_cell(**kwargs)
        return self.by_id(cell_id)
