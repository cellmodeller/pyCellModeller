"""Torch-based simulation engine."""

from __future__ import annotations

import torch

from pycellmodeller.core.config import SimulationConfig
from pycellmodeller.core.state import SimulationState


class TorchEngine:
    """Deterministic, tensor-based stepping engine."""

    def __init__(self, config: SimulationConfig) -> None:
        self._config = config
        self._seed_torch()

    @property
    def config(self) -> SimulationConfig:
        """Return engine configuration."""
        return self._config

    def _seed_torch(self) -> None:
        torch.manual_seed(self._config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._config.seed)
        torch.use_deterministic_algorithms(True)

    def to_device_tensor(self, values: list[list[float]]) -> torch.Tensor:
        """Create a tensor on the configured device/dtype."""
        return torch.tensor(values, dtype=self._config.dtype, device=self._config.device)

    def make_state(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        *,
        time: float = 0.0,
        step_index: int = 0,
    ) -> SimulationState:
        """Move input tensors to configured target and allocate full state tensors."""
        positions_on_device = positions.to(device=self._config.device, dtype=self._config.dtype)
        velocities_on_device = velocities.to(device=self._config.device, dtype=self._config.dtype)

        n_cells = int(positions_on_device.shape[0])
        state = SimulationState.allocate(
            capacity=n_cells,
            device=self._config.device,
            dtype=self._config.dtype,
            spatial_dim=int(positions_on_device.shape[1]),
            time=time,
            step_index=step_index,
        )
        state.positions[:n_cells] = positions_on_device
        state.velocities[:n_cells] = velocities_on_device
        state.n_cells = n_cells
        state.cell_ids[:n_cells] = torch.arange(n_cells, dtype=torch.int64, device=self._config.device)
        state.next_cell_id = n_cells
        return state

    def step(self, state: SimulationState) -> SimulationState:
        """Advance state with a deterministic, fixed-velocity position update."""
        active = state.active_slice()
        state.positions[active] = state.positions[active] + (state.velocities[active] * self._config.dt)
        state.time += self._config.dt
        state.step_index += 1
        return state
