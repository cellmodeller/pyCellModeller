"""Simulation API entry points."""

from __future__ import annotations

import torch

from pycellmodeller.biophysics.torch_bacterium import TorchBacterium
from pycellmodeller.core.cells import CellCollection, CellView
from pycellmodeller.core.config import SimulationConfig
from pycellmodeller.core.state import SimulationState
from pycellmodeller.engine.torch_engine import TorchEngine
from pycellmodeller.programs.base import CellProgram


class Simulation:
    """High-level simulation façade over the torch engine."""

    def __init__(self, config: SimulationConfig, biophysics: TorchBacterium, program: CellProgram) -> None:
        self._config = config
        self._biophysics = biophysics
        self._program = program
        self._engine = TorchEngine(config=config)
        self._state: SimulationState | None = None
        self._rng: torch.Generator | None = None
        self._context: dict[str, object] = {}

    @property
    def config(self) -> SimulationConfig:
        """Return immutable-ish simulation configuration."""
        return self._config

    @property
    def state(self) -> SimulationState:
        """Return current simulation state."""
        if self._state is None:
            msg = "Simulation state is not initialized. Call initialize() first."
            raise RuntimeError(msg)
        return self._state

    @property
    def cells(self) -> CellCollection:
        """Author-friendly cell view over current state tensors."""
        return CellCollection(self.state)

    def initialize(self) -> SimulationState:
        """Initialize simulation runtime context and empty tensor-backed state."""
        self._rng = torch.Generator(device=self._config.device.type)
        self._rng.manual_seed(self._config.seed)
        self._context = {
            "device": self._config.device,
            "dtype": self._config.dtype,
            "seed": self._config.seed,
        }

        self._state = SimulationState.allocate(
            capacity=0,
            device=self._config.device,
            dtype=self._config.dtype,
            spatial_dim=2,
            time=0.0,
            step_index=0,
            metadata={"initialized": True},
        )
        self._program.setup(self)
        self.state.divide[: self.state.n_cells] = False
        self.state.normalize_active_directions()
        return self.state

    def add_cell(
        self,
        *,
        parent_id: int = -1,
        position: torch.Tensor | None = None,
        velocity: torch.Tensor | None = None,
        direction: torch.Tensor | None = None,
        length: float | None = None,
        radius: float = 0.5,
        target_volume: float = 1.0,
        growth_rate: float | None = None,
        color: torch.Tensor | None = None,
        divide: bool = False,
    ) -> CellView:
        """Insert one cell and initialize program-owned state."""
        state = self.state
        resolved_length = max(1.0, self._biophysics.min_length) if length is None else length
        resolved_growth_rate = self._biophysics.growth_rate if growth_rate is None else growth_rate
        cell = self.cells.add(
            parent_id=parent_id,
            position=position,
            velocity=velocity,
            direction=direction,
            length=resolved_length,
            radius=radius,
            target_volume=target_volume,
            growth_rate=resolved_growth_rate,
            color=color,
            divide=divide,
        )
        _ = state
        self._program.initialize_cell(cell)
        cell.direction = cell.direction
        return cell

    def step(self) -> SimulationState:
        """Advance the simulation by one fixed timestep."""
        state = self.state
        active = state.active_slice()

        if state.n_cells > 0:
            state.divide[active] = False
            state.normalize_active_directions()

        self._program.update_cells(self.cells)

        if state.n_cells > 0:
            state.lengths[active] = self._biophysics.grow(state.lengths[active], self._config.dt)
            current_volumes = self._biophysics.compute_volume(state.lengths[active], state.radii[active])
            divide_mask = state.divide[active] | self._biophysics.should_divide(state.lengths[active])
            divide_mask = divide_mask & self._biophysics.supports_longitudinal_division(
                state.lengths[active],
                state.radii[active],
            )

            daughter_a_lengths, daughter_b_lengths = self._biophysics.divide(
                state.lengths[active],
                state.radii[active],
                divide_mask,
                generator=self._rng,
            )

            divide_indices = torch.nonzero(divide_mask, as_tuple=False).flatten().tolist()
            for local_idx in divide_indices:
                idx = int(local_idx)
                parent = self.cells.by_index(idx)
                parent_id = parent.id

                parent_position = parent.position.clone()
                parent_velocity = parent.velocity.clone()
                parent_direction = parent.direction.clone()
                parent_color = parent.color.clone()
                parent_radius = parent.radius
                parent_target_volume = parent.target_volume
                parent_growth_rate = parent.growth_rate

                daughter_a_length = float(daughter_a_lengths[idx].item())
                daughter_b_length = float(daughter_b_lengths[idx].item())
                daughter_a_pos, daughter_b_pos = self._biophysics.place_daughters(
                    parent_position=parent_position,
                    parent_direction=parent_direction,
                    daughter_a_length=daughter_a_length,
                    daughter_b_length=daughter_b_length,
                    parent_radius=parent_radius,
                )

                parent.position = daughter_a_pos
                parent.length = daughter_a_length
                parent.divide = False

                daughter_b = self.add_cell(
                    parent_id=parent_id,
                    position=daughter_b_pos,
                    velocity=parent_velocity,
                    direction=parent_direction,
                    length=daughter_b_length,
                    radius=parent_radius,
                    target_volume=parent_target_volume,
                    growth_rate=parent_growth_rate,
                    color=parent_color,
                    divide=False,
                )
                daughter_a = self.cells.by_index(idx)
                self._program.on_division(parent, daughter_a, daughter_b)

            active = state.active_slice()
            state.positions[active] = self._biophysics.resolve_contacts(
                positions=state.positions[active],
                directions=state.directions[active],
                lengths=state.lengths[active],
                radii=state.radii[active],
            )
            state.normalize_active_directions()
            _ = current_volumes

        self._engine.step(state)
        return state

    def run(self, steps: int) -> SimulationState:
        """Run ``steps`` simulation steps and return the updated state."""
        if not isinstance(steps, int):
            msg = "steps must be an integer"
            raise TypeError(msg)
        if steps < 0:
            msg = "steps must be non-negative"
            raise ValueError(msg)

        for _ in range(steps):
            self.step()
        return self.state
