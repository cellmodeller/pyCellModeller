# pyCellModeller

`pyCellModeller` is a Torch-native rewrite of CellModeller.

## Current status (bootstrap truth)

This repository is in an early bootstrap phase and currently implements only a small deterministic toy simulator:

- ✅ Public API exports: `Simulation`, `SimulationConfig`, `SimulationState`
- ✅ CPU-first deterministic stepping path (tested)
- ✅ Torch engine (`TorchEngine`) as the only execution backend in v1
- ✅ Native initialization + stepping flow via `Simulation(config, biophysics, program)`
- ✅ Native Torch bacterium supports oriented rods, growth/division, and first-pass contact relaxation
- ✅ Tutorial 1 native example can dump PNG frames of spatial colony growth
- 🚧 Mechanics / fields / biology / IO / CLI modules are scaffolded but not fully implemented yet

**Important:** legacy CellModeller compatibility is **not** a v1 goal. Historical PyOpenCL/OpenCL execution paths are intentionally not carried forward.

## Implemented API surface (today)

Top-level import:

```python
from pycellmodeller import Simulation, SimulationConfig, SimulationState
```

Core usage:

```python
from pycellmodeller import CellProgram, Simulation, SimulationConfig, TorchBacterium

class NoOpProgram(CellProgram):
    pass

sim = Simulation(
    config=SimulationConfig(device="cpu", dt=0.1, seed=0),
    biophysics=TorchBacterium(),
    program=NoOpProgram(),
)
state0 = sim.initialize()
state1 = sim.step()
```

### `SimulationConfig`

Current fields:

- `device: str | torch.device = "cpu"`
- `dt: float = 0.1` (must be > 0)
- `seed: int = 0` (must be >= 0)
- `dtype: torch.dtype = torch.float32` (must be floating)

### `SimulationState`

Current fields:

- `positions: torch.Tensor`
- `velocities: torch.Tensor`
- `directions: torch.Tensor` (shape `(capacity, 2)`, unit vectors)
- `time: float = 0.0`
- `step_index: int = 0`
- `metadata: dict[str, Any] = {}`

## Next milestone

Current milestone delivers the first spatial native colony simulation:

- oriented 2D rods (`position`, `direction`, `length`, `radius`)
- mass-conserving rod division with non-overlapping daughter placement
- first-pass capsule contact relaxation for colony spreading
- frame dumping from `examples/tutorials/tutorial_1_native.py` for video generation

## Current repository layout

```text
pyCellModeller/
  pyproject.toml
  README.md
  ARCHITECTURE.md
  PRODUCT.md
  AGENT.md
  ADR-001.md
  DEVELOPMENT_PLAN.md
  src/
    pycellmodeller/
      __init__.py
      api/
      core/
      engine/
      mechanics/
      fields/
      biology/
      io/
      cli/
  tests/
    unit/
    integration/
    regression/
  examples/
    scripts/
    notebooks/
  .github/
    workflows/
```

> Note: folders such as `viz/`, `docs/`, and `docker/` are not present yet and are intentionally omitted from the current structure.

## Scope for v1 bootstrap

- Deterministic CPU reference behavior first
- Torch tensor state and stepping loop first
- No legacy compatibility shims unless explicitly requested
- No multi-backend abstraction in v1

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
pytest
```
