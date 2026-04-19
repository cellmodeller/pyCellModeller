# ARCHITECTURE

## System overview

`pyCellModeller` is a **Torch-native, CPU-first deterministic simulation runtime**.

The historical PyOpenCL/OpenCL engine direction is intentionally retired. Public APIs should not include CL/OpenCL compatibility naming.

## Core runtime concepts

### `Simulation`

`Simulation` is the top-level runtime orchestrator. It owns lifecycle operations such as deterministic initialization and stepping, and coordinates configuration, engine execution, and state updates.

### `TorchBacterium`

`TorchBacterium` is the canonical per-cell runtime representation for Torch-native biology/mechanics workflows. It is conceptually backed by structured tensor state rather than legacy object/CL-compatible wrappers.

### `CellProgram`

`CellProgram` defines per-cell behavioral logic in the native pyCellModeller architecture. Programs operate over runtime cell state semantics and are intended to compose with Torch execution, not legacy tutorial-file compatibility layers.

### Structured tensor-backed cell state

Cell state is represented as structured, tensor-backed data (positions, velocities, and future extensible fields) validated for shape/device/dtype consistency. This is the foundation for deterministic stepping and future mechanics/biology extension.

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

## Determinism and device model

- CPU correctness is the reference behavior for this phase.
- CUDA can be requested through `SimulationConfig(device=...)` where PyTorch supports it, but deterministic CPU behavior is the baseline acceptance path.
- No alternative backend abstraction is implemented.
- CL/OpenCL compatibility naming is explicitly excluded from public APIs.

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
