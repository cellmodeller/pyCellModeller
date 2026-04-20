"""State containers for simulation runtime data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(slots=True)
class SimulationState:
    """Tensor-backed simulation state with structured per-cell fields."""

    positions: torch.Tensor
    velocities: torch.Tensor
    directions: torch.Tensor
    lengths: torch.Tensor
    radii: torch.Tensor
    cell_ids: torch.Tensor
    parent_ids: torch.Tensor
    target_volume: torch.Tensor
    growth_rate: torch.Tensor
    color: torch.Tensor
    divide: torch.Tensor
    n_cells: int
    next_cell_id: int
    time: float = 0.0
    step_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def allocate(
        cls,
        capacity: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        spatial_dim: int = 2,
        color_dim: int = 3,
        time: float = 0.0,
        step_index: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> SimulationState:
        """Allocate empty tensor state on the target device."""
        if capacity < 0:
            msg = "capacity must be non-negative"
            raise ValueError(msg)
        if spatial_dim != 2:
            msg = "v1 state supports only 2D spatial_dim=2"
            raise ValueError(msg)

        directions = torch.zeros((capacity, spatial_dim), dtype=dtype, device=device)
        if capacity > 0:
            directions[:, 0] = 1.0

        return cls(
            positions=torch.zeros((capacity, spatial_dim), dtype=dtype, device=device),
            velocities=torch.zeros((capacity, spatial_dim), dtype=dtype, device=device),
            directions=directions,
            lengths=torch.ones((capacity,), dtype=dtype, device=device),
            radii=torch.full((capacity,), 0.5, dtype=dtype, device=device),
            cell_ids=torch.full((capacity,), -1, dtype=torch.int64, device=device),
            parent_ids=torch.full((capacity,), -1, dtype=torch.int64, device=device),
            target_volume=torch.ones((capacity,), dtype=dtype, device=device),
            growth_rate=torch.zeros((capacity,), dtype=dtype, device=device),
            color=torch.ones((capacity, color_dim), dtype=dtype, device=device),
            divide=torch.zeros((capacity,), dtype=torch.bool, device=device),
            n_cells=0,
            next_cell_id=0,
            time=time,
            step_index=step_index,
            metadata={} if metadata is None else dict(metadata),
        )

    def __post_init__(self) -> None:
        if self.positions.ndim != 2:
            msg = "positions must be a 2D tensor"
            raise ValueError(msg)

        capacity, spatial_dim = self.positions.shape
        if spatial_dim != 2:
            msg = "positions must use 2D coordinates"
            raise ValueError(msg)

        if self.velocities.shape != (capacity, spatial_dim):
            msg = "velocities must match positions shape"
            raise ValueError(msg)

        if self.directions.shape != (capacity, spatial_dim):
            msg = "directions must match positions shape"
            raise ValueError(msg)

        if self.lengths.shape != (capacity,):
            msg = "lengths must have shape (capacity,)"
            raise ValueError(msg)

        if self.radii.shape != (capacity,):
            msg = "radii must have shape (capacity,)"
            raise ValueError(msg)

        if self.cell_ids.shape != (capacity,):
            msg = "cell_ids must have shape (capacity,)"
            raise ValueError(msg)

        if self.parent_ids.shape != (capacity,):
            msg = "parent_ids must have shape (capacity,)"
            raise ValueError(msg)

        if self.target_volume.shape != (capacity,):
            msg = "target_volume must have shape (capacity,)"
            raise ValueError(msg)

        if self.growth_rate.shape != (capacity,):
            msg = "growth_rate must have shape (capacity,)"
            raise ValueError(msg)

        if self.divide.shape != (capacity,):
            msg = "divide must have shape (capacity,)"
            raise ValueError(msg)

        if self.color.ndim != 2 or self.color.shape[0] != capacity:
            msg = "color must have shape (capacity, n_channels)"
            raise ValueError(msg)

        tensor_fields = (
            self.positions,
            self.velocities,
            self.directions,
            self.lengths,
            self.radii,
            self.target_volume,
            self.growth_rate,
            self.color,
        )
        integer_fields = (self.cell_ids, self.parent_ids)

        device = self.positions.device
        dtype = self.positions.dtype

        for tensor in tensor_fields:
            if tensor.device != device:
                msg = "all floating tensors must be on the same device"
                raise ValueError(msg)
            if tensor.dtype != dtype:
                msg = "all floating tensors must have the same dtype"
                raise ValueError(msg)

        for tensor in integer_fields:
            if tensor.device != device:
                msg = "all integer tensors must be on the same device"
                raise ValueError(msg)
            if tensor.dtype != torch.int64:
                msg = "cell_ids and parent_ids must be int64"
                raise TypeError(msg)

        if self.divide.device != device:
            msg = "divide must be on same device"
            raise ValueError(msg)

        if self.divide.dtype != torch.bool:
            msg = "divide must be a boolean tensor"
            raise TypeError(msg)

        if not 0 <= self.n_cells <= capacity:
            msg = "n_cells must be in [0, capacity]"
            raise ValueError(msg)

        if self.next_cell_id < 0:
            msg = "next_cell_id must be non-negative"
            raise ValueError(msg)

        if self.time < 0.0:
            msg = "time must be non-negative"
            raise ValueError(msg)

        if self.step_index < 0:
            msg = "step_index must be non-negative"
            raise ValueError(msg)

    @property
    def capacity(self) -> int:
        return int(self.positions.shape[0])

    @staticmethod
    def normalize_direction(direction: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """Normalize one 2D direction vector with safe fallback to +x."""
        if direction.shape != (2,):
            msg = "direction must have shape (2,)"
            raise ValueError(msg)
        out = direction.clone()
        norm = torch.linalg.vector_norm(out)
        if float(norm.item()) <= eps:
            out.zero_()
            out[0] = 1.0
            return out
        return out / norm

    def normalize_active_directions(self, eps: float = 1e-12) -> None:
        """Normalize all active directions in-place."""
        if self.n_cells == 0:
            return
        active = self.active_slice()
        dirs = self.directions[active]
        norms = torch.linalg.vector_norm(dirs, dim=1, keepdim=True)
        invalid = norms.squeeze(1) <= eps
        safe_norms = torch.where(invalid.unsqueeze(1), torch.ones_like(norms), norms)
        dirs = dirs / safe_norms
        if torch.any(invalid):
            dirs[invalid] = torch.tensor([1.0, 0.0], dtype=dirs.dtype, device=dirs.device)
        self.directions[active] = dirs

    def _ensure_capacity(self, required: int) -> None:
        if required <= self.capacity:
            return

        new_capacity = max(required, max(1, self.capacity * 2))
        self.positions = self._expand_2d(self.positions, new_capacity, fill_value=0.0)
        self.velocities = self._expand_2d(self.velocities, new_capacity, fill_value=0.0)
        self.directions = self._expand_2d(self.directions, new_capacity, fill_value=0.0)
        self.lengths = self._expand_1d(self.lengths, new_capacity, fill_value=1.0)
        self.radii = self._expand_1d(self.radii, new_capacity, fill_value=0.5)
        self.cell_ids = self._expand_1d(self.cell_ids, new_capacity, fill_value=-1)
        self.parent_ids = self._expand_1d(self.parent_ids, new_capacity, fill_value=-1)
        self.target_volume = self._expand_1d(self.target_volume, new_capacity, fill_value=1.0)
        self.growth_rate = self._expand_1d(self.growth_rate, new_capacity, fill_value=0.0)
        self.color = self._expand_2d(self.color, new_capacity, fill_value=1.0)
        self.divide = self._expand_1d(self.divide, new_capacity, fill_value=False)

        if self.n_cells < new_capacity:
            self.directions[self.n_cells :, 0] = 1.0

    def _expand_1d(self, tensor: torch.Tensor, new_capacity: int, fill_value: float | int | bool) -> torch.Tensor:
        out = torch.full((new_capacity,), fill_value=fill_value, dtype=tensor.dtype, device=tensor.device)
        out[: self.n_cells] = tensor[: self.n_cells]
        return out

    def _expand_2d(self, tensor: torch.Tensor, new_capacity: int, fill_value: float | int | bool) -> torch.Tensor:
        out = torch.full((new_capacity, tensor.shape[1]), fill_value=fill_value, dtype=tensor.dtype, device=tensor.device)
        out[: self.n_cells] = tensor[: self.n_cells]
        return out

    def add_cell(
        self,
        *,
        parent_id: int = -1,
        position: torch.Tensor | None = None,
        velocity: torch.Tensor | None = None,
        direction: torch.Tensor | None = None,
        length: float = 1.0,
        radius: float = 0.5,
        target_volume: float = 1.0,
        growth_rate: float = 0.0,
        color: torch.Tensor | None = None,
        divide: bool = False,
    ) -> int:
        """Insert one cell and return its assigned globally-unique cell ID."""
        self._ensure_capacity(self.n_cells + 1)

        idx = self.n_cells
        cell_id = self.next_cell_id
        self.n_cells += 1
        self.next_cell_id += 1

        spatial_dim = self.positions.shape[1]
        color_dim = self.color.shape[1]

        if position is None:
            self.positions[idx].zero_()
        else:
            pos = position.to(device=self.positions.device, dtype=self.positions.dtype)
            if pos.shape != (spatial_dim,):
                msg = f"position must have shape ({spatial_dim},)"
                raise ValueError(msg)
            self.positions[idx] = pos

        if velocity is None:
            self.velocities[idx].zero_()
        else:
            vel = velocity.to(device=self.velocities.device, dtype=self.velocities.dtype)
            if vel.shape != (spatial_dim,):
                msg = f"velocity must have shape ({spatial_dim},)"
                raise ValueError(msg)
            self.velocities[idx] = vel

        if direction is None:
            self.directions[idx] = torch.tensor([1.0, 0.0], dtype=self.directions.dtype, device=self.directions.device)
        else:
            direction_tensor = direction.to(device=self.directions.device, dtype=self.directions.dtype)
            if direction_tensor.shape != (spatial_dim,):
                msg = f"direction must have shape ({spatial_dim},)"
                raise ValueError(msg)
            self.directions[idx] = self.normalize_direction(direction_tensor)

        if color is None:
            self.color[idx].fill_(1.0)
        else:
            color_tensor = color.to(device=self.color.device, dtype=self.color.dtype)
            if color_tensor.shape != (color_dim,):
                msg = f"color must have shape ({color_dim},)"
                raise ValueError(msg)
            self.color[idx] = color_tensor

        self.cell_ids[idx] = cell_id
        self.parent_ids[idx] = parent_id
        self.lengths[idx] = length
        self.radii[idx] = radius
        self.target_volume[idx] = target_volume
        self.growth_rate[idx] = growth_rate
        self.divide[idx] = divide

        return cell_id

    def add_cells(self, count: int, *, parent_id: int = -1) -> torch.Tensor:
        """Insert ``count`` default-initialized cells and return assigned IDs."""
        if count < 0:
            msg = "count must be non-negative"
            raise ValueError(msg)

        assigned = [self.add_cell(parent_id=parent_id) for _ in range(count)]
        return torch.tensor(assigned, dtype=torch.int64, device=self.cell_ids.device)

    def active_slice(self) -> slice:
        """Slice that spans currently active cells."""
        return slice(0, self.n_cells)
