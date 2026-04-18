"""Simulation API entry points."""

from __future__ import annotations

import torch

from pycellmodeller.core.config import SimulationConfig
from pycellmodeller.core.state import SimulationState
from pycellmodeller.engine.torch_engine import TorchEngine


class Simulation:
    """High-level simulation façade over the torch engine."""

    def __init__(self, config: SimulationConfig) -> None:
        self._config = config
        self._engine = TorchEngine(config=config)
        self._state: SimulationState | None = None

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

    def initialize_example(self) -> SimulationState:
        """Initialize a tiny deterministic 2-cell system for development/testing."""
        positions = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=self._config.dtype)
        velocities = torch.tensor([[0.25, 0.0], [0.0, 0.25]], dtype=self._config.dtype)
        self._state = self._engine.make_state(positions=positions, velocities=velocities)
        return self._state

    def step(self) -> SimulationState:
        """Advance the simulation by one fixed timestep."""
        self._state = self._engine.step(self.state)
        return self._state
