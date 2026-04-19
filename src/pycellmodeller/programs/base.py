"""Base interfaces for per-cell biological programs."""

from __future__ import annotations


class CellProgram:
    """Base class for user-defined cell programs.

    Subclasses can override lifecycle hooks to customize simulation behavior.
    """

    sim: object | None

    def __init__(self) -> None:
        self.sim = None

    def setup(self, sim: object) -> None:
        """Initialize program state using the simulation object."""
        self.sim = sim

    def initialize_cell(self, cell: object) -> None:
        """Initialize per-cell state when a cell is created."""

    def update_cells(self, cells: object) -> None:
        """Update a collection of cells during a simulation step."""

    def on_division(self, parent: object, daughter_a: object, daughter_b: object) -> None:
        """Handle post-division bookkeeping for parent and daughters."""
