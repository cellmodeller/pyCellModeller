"""Simulation API entry points."""

from __future__ import annotations

import torch

from pycellmodeller.core.config import SimulationConfig
from pycellmodeller.core.cells import CellCollection
from pycellmodeller.core.state import SimulationState
from pycellmodeller.engine.torch_engine import TorchEngine
from pycellmodeller.programs.base import CellProgram


class Simulation:
    """High-level simulation façade over the torch engine."""

    def __init__(self, config: SimulationConfig, *, program: CellProgram | None = None) -> None:
        self._config = config
        self._engine = TorchEngine(config=config)
        self._state: SimulationState | None = None
        self._program = program
        if self._program is not None:
            self._program.setup(self)

    @property
    def config(self) -> SimulationConfig:
        """Return immutable-ish simulation configuration."""
        return self._config

    @property
    def state(self) -> SimulationState:
        """Return current simulation state."""
        if self._state is None:
            msg = "Simulation state is not initialized. Call initialize_example() first."
            raise RuntimeError(msg)
        return self._state

    @property
    def cells(self) -> CellCollection:
        """Author-friendly cell view over current state tensors."""
        return CellCollection(self.state)

    def initialize_example(self) -> SimulationState:
        """Initialize a tiny deterministic 2-cell system for development/testing."""
        positions = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=self._config.dtype)
        velocities = torch.tensor([[0.25, 0.0], [0.0, 0.25]], dtype=self._config.dtype)
        self._state = self._engine.make_state(positions=positions, velocities=velocities)

        if self._program is not None:
            for cell in self.cells:
                self._program.initialize_cell(cell)

        return self._state

    def step(self) -> SimulationState:
        """Advance the simulation by one fixed timestep."""
        self._state = self._engine.step(self.state)

        if self._program is not None:
            self._program.update_cells(self.cells)

        return self._state
