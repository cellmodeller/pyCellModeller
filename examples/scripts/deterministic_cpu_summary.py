"""Run a tiny deterministic simulation on CPU and print a state summary."""

from __future__ import annotations

from pycellmodeller import Simulation, SimulationConfig


def main() -> None:
    config = SimulationConfig(device="cpu", dt=0.1, seed=1)
    simulation = Simulation(config)

    simulation.initialize_example()
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
