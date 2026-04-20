"""Tutorial 1: native Torch colony growth with simple spatial mechanics and frame dumping."""

from __future__ import annotations

import argparse
import base64
import math
from pathlib import Path

import torch

from pycellmodeller import CellProgram, Simulation, SimulationConfig, TorchBacterium
from pycellmodeller.core.cells import CellCollection, CellView


def spherocylinder_volume(length: float, radius: float) -> float:
    return (math.pi * (radius**2) * length) + ((4.0 / 3.0) * math.pi * (radius**3))


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
        sim.add_cell(position=torch.tensor([0.0, 0.0]), direction=torch.tensor([1.0, 0.0]))

    def _sample_target_volume(self) -> float:
        if self._rng is None:
            msg = "program RNG is not initialized"
            raise RuntimeError(msg)
        return float(torch.empty((), device="cpu").uniform_(3.0, 4.5, generator=self._rng).item())

    def _apply_growth_policy(self, cell: CellView) -> None:
        cell.target_volume = self._sample_target_volume()
        cell.growth_rate = 1.0
        cell.color = torch.tensor([0.1, 0.8, 0.2])

    def initialize_cell(self, cell: CellView) -> None:
        self._apply_growth_policy(cell)

    def update_cells(self, cells: CellCollection) -> None:
        for cell in cells:
            volume = spherocylinder_volume(cell.length, cell.radius)
            cell.divide = volume > cell.target_volume

    def on_division(self, parent: CellView, daughter_a: CellView, daughter_b: CellView) -> None:
        _ = parent
        self.division_count += 1
        self._apply_growth_policy(daughter_a)
        self._apply_growth_policy(daughter_b)


def dump_frame(simulation: Simulation, frame_path: Path) -> None:
    """Render rods as line segments and save PNG frame."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional tutorial dependency
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        # 1x1 transparent PNG fallback so frame dumping still works without matplotlib.
        png_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAFgwJ/lRjA8QAAAABJRU5ErkJggg=="
        )
        frame_path.write_bytes(png_bytes)
        return

    state = simulation.state
    active = state.active_slice()
    positions = state.positions[active].cpu()
    directions = state.directions[active].cpu()
    lengths = state.lengths[active].cpu()
    radii = state.radii[active].cpu()

    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(state.n_cells):
        center = positions[i]
        direction = directions[i]
        half = 0.5 * lengths[i]
        p0 = center - (half * direction)
        p1 = center + (half * direction)
        linewidth = max(1.0, float(8.0 * radii[i].item()))
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="forestgreen", linewidth=linewidth, solid_capstyle="round")

    if state.n_cells > 0:
        xmin = float(torch.min(positions[:, 0]).item()) - 2.0
        xmax = float(torch.max(positions[:, 0]).item()) + 2.0
        ymin = float(torch.min(positions[:, 1]).item()) - 2.0
        ymax = float(torch.max(positions[:, 1]).item()) + 2.0
        span = max(xmax - xmin, ymax - ymin, 6.0)
        xmid = 0.5 * (xmin + xmax)
        ymid = 0.5 * (ymin + ymax)
        ax.set_xlim(xmid - 0.5 * span, xmid + 0.5 * span)
        ax.set_ylim(ymid - 0.5 * span, ymid + 0.5 * span)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"step={state.step_index} t={state.time:.2f} n={state.n_cells}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(False)

    frame_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(frame_path, dpi=120)
    plt.close(fig)


def run_tutorial(
    *,
    steps: int = 40,
    dt: float = 0.1,
    seed: int = 7,
    frame_every: int = 5,
    frame_dir: str | Path = "frames",
    dump_frames: bool = True,
) -> tuple[Simulation, dict[str, float | int]]:
    config = SimulationConfig(device="cpu", dt=dt, seed=seed)
    biophysics = TorchBacterium(
        growth_rate=1.0,
        division_length=100.0,
        partition_noise_std=0.04,
        contact_iterations=8,
        contact_gap=1e-3,
    )
    program = Tutorial1Program()

    simulation = Simulation(config=config, biophysics=biophysics, program=program)
    simulation.initialize()

    frame_dir_path = Path(frame_dir)
    if dump_frames:
        dump_frame(simulation, frame_dir_path / "frame_00000.png")

    for step in range(1, steps + 1):
        simulation.step()
        if dump_frames and (step % frame_every == 0):
            dump_frame(simulation, frame_dir_path / f"frame_{step:05d}.png")

    final_state = simulation.state
    summary = {
        "step_count": final_state.step_index,
        "simulation_time": round(final_state.time, 6),
        "final_cell_count": final_state.n_cells,
        "division_count": program.division_count,
    }
    return simulation, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run native tutorial 1 colony simulation.")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--frame-every", type=int, default=5)
    parser.add_argument("--frame-dir", type=Path, default=Path("frames"))
    parser.add_argument("--no-frames", action="store_true", help="Disable PNG frame dumping")
    args = parser.parse_args()

    _, summary = run_tutorial(
        steps=args.steps,
        dt=args.dt,
        seed=args.seed,
        frame_every=args.frame_every,
        frame_dir=args.frame_dir,
        dump_frames=not args.no_frames,
    )
    print(summary)


if __name__ == "__main__":
    main()
