# ARCHITECTURE

## System overview

`pyCellModeller` is currently a **CPU-first deterministic toy simulation engine** implemented with PyTorch tensors.

The historical PyOpenCL/OpenCL engine direction is intentionally replaced by a **Torch-only v1 path**. There is no compatibility layer for PyOpenCL in this codebase.

## Current implemented layers

### 1) Public API (`pycellmodeller.api`)
Implemented:
- `Simulation`
- `SimulationConfig`
- `SimulationState`

The API supports:
- construction with config
- deterministic example initialization (`initialize_example`)
- one-step advancement (`step`)

### 2) Core (`pycellmodeller.core`)
Implemented:
- `SimulationConfig` dataclass with validation
- `SimulationState` dataclass with tensor-shape/device/dtype validation
- `events.py` scaffold (not active in stepping yet)

### 3) Engine (`pycellmodeller.engine`)
Implemented:
- `TorchEngine`
- deterministic seeding via `torch.manual_seed`
- deterministic algorithm mode
- state construction on configured device/dtype
- fixed-velocity deterministic position update kernel

### 4) Scientific module namespaces
Directories exist but are currently placeholders:
- `pycellmodeller.mechanics`
- `pycellmodeller.fields`
- `pycellmodeller.biology`

### 5) Interfaces
Current placeholders:
- `pycellmodeller.io`
- `pycellmodeller.cli`

## Current runtime data model

### `SimulationConfig`
- `device`
- `dt`
- `seed`
- `dtype`

### `SimulationState`
- `positions`
- `velocities`
- `time`
- `step_index`
- `metadata`

No field grid, cell lineage schema, or event queue orchestration is wired into stepping yet.

## Current execution flow

1. User creates `SimulationConfig`
2. User creates `Simulation(config)`
3. `initialize_example()` produces a deterministic 2-cell tensor state
4. `step()` applies `positions + velocities * dt`
5. New `SimulationState` is returned with incremented time and step index

## Determinism and device model

- CPU determinism is the reference behavior for this phase.
- CUDA can be requested through `SimulationConfig(device=...)` where PyTorch supports it, but CPU correctness is the active baseline.
- No alternative backend abstraction is implemented.

## Current directory structure (implemented)

```text
src/pycellmodeller/
  __init__.py
  api/
    __init__.py
    simulation.py
  core/
    __init__.py
    config.py
    state.py
    events.py
  engine/
    __init__.py
    torch_engine.py
  mechanics/
    __init__.py
  fields/
    __init__.py
  biology/
    __init__.py
  io/
    __init__.py
  cli/
    __init__.py
```

`viz/` is not present in the current scaffold.
