"""Run a tiny deterministic simulation on CPU and print a state summary."""

from __future__ import annotations

import torch

from pycellmodeller import CellProgram, Simulation, SimulationConfig, TorchBacterium


class NoOpProgram(CellProgram):
    """Minimal program used for deterministic stepping demos."""


def main() -> None:
    config = SimulationConfig(device="cpu", dt=0.1, seed=1)
    simulation = Simulation(
        config=config,
        biophysics=TorchBacterium(growth_rate=0.0, division_length=10.0),
        program=NoOpProgram(),
    )

    simulation.initialize()
    simulation.add_cell(position=torch.tensor([0.0, 0.0]), velocity=torch.tensor([0.25, 0.0]))
    simulation.add_cell(position=torch.tensor([1.0, 0.0]), velocity=torch.tensor([0.0, 0.25]))
    state = simulation.step()

    summary = {
        "step_index": state.step_index,
        "time": round(state.time, 6),
        "position_sum": [round(v, 6) for v in state.positions.sum(dim=0).tolist()],
        "velocity_sum": [round(v, 6) for v in state.velocities.sum(dim=0).tolist()],
    }
    print(summary)


if __name__ == "__main__":
    main()
