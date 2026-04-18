# pyCellModeller

`pyCellModeller` is a Python-native, Torch-based rewrite of CellModeller.

## Current status (bootstrap truth)

This repository is in an early bootstrap phase and currently implements only a small deterministic toy simulator:

- ✅ Public API exports: `Simulation`, `SimulationConfig`, `SimulationState`
- ✅ CPU-first deterministic stepping path (tested)
- ✅ Torch engine (`TorchEngine`) as the only execution backend in v1
- ✅ Minimal example initialization (`initialize_example`) + single-step update (`step`)
- 🚧 Mechanics / fields / biology / IO / CLI modules are scaffolded but not implemented yet

**Important:** the historical PyOpenCL path is intentionally not carried forward. In v1, the project is Torch-only by design.

## Implemented API surface (today)

Top-level import:

```python
from pycellmodeller import Simulation, SimulationConfig, SimulationState
```

Core usage:

```python
from pycellmodeller.api import Simulation, SimulationConfig

sim = Simulation(SimulationConfig(device="cpu", dt=0.1, seed=0))
state0 = sim.initialize_example()
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
- `time: float = 0.0`
- `step_index: int = 0`
- `metadata: dict[str, Any] = {}`

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
- No PyOpenCL compatibility layer
- No multi-backend abstraction in v1

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
pytest
```
