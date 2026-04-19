"""Tutorial 1: native Torch program with growth, division, and summary output."""

from __future__ import annotations

import math

import torch

from pycellmodeller import CellProgram, Simulation, SimulationConfig, TorchBacterium
from pycellmodeller.core.cells import CellCollection, CellView


class Tutorial1Program(CellProgram):
    """Simple tutorial program with randomized volume targets and green cells."""

    def __init__(self) -> None:
        super().__init__()
        self.division_count = 0
        self._rng: torch.Generator | None = None

    def setup(self, sim: Simulation) -> None:
        super().setup(sim)
        self._rng = torch.Generator(device=sim.config.device.type)
        self._rng.manual_seed(sim.config.seed + 101)
        sim.add_cell()

    def _sample_target_volume(self) -> float:
        if self._rng is None:
            msg = "program RNG is not initialized"
            raise RuntimeError(msg)
        return float(torch.empty((), device="cpu").uniform_(2.0, 4.0, generator=self._rng).item())

    def _apply_growth_policy(self, cell: CellView) -> None:
        cell.target_volume = self._sample_target_volume()
        cell.growth_rate = 1.0
        cell.color = torch.tensor([0.0, 1.0, 0.0])

    def initialize_cell(self, cell: CellView) -> None:
        self._apply_growth_policy(cell)

    def update_cells(self, cells: CellCollection) -> None:
        for cell in cells:
            volume = math.pi * (cell.radius ** 2) * cell.length
            cell.divide = volume > cell.target_volume

    def on_division(self, parent: CellView, daughter_a: CellView, daughter_b: CellView) -> None:
        _ = parent
        self.division_count += 1
        self._apply_growth_policy(daughter_a)
        self._apply_growth_policy(daughter_b)


def main() -> None:
    steps = 30
    config = SimulationConfig(device="cpu", dt=0.1, seed=7)
    biophysics = TorchBacterium(growth_rate=1.0, division_length=100.0)
    program = Tutorial1Program()

    simulation = Simulation(config=config, biophysics=biophysics, program=program)
    simulation.initialize()
    final_state = simulation.run(steps)

    summary = {
        "step_count": final_state.step_index,
        "simulation_time": round(final_state.time, 6),
        "final_cell_count": final_state.n_cells,
        "division_count": program.division_count,
    }
    print(summary)


if __name__ == "__main__":
    main()
