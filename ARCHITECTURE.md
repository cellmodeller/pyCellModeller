# ARCHITECTURE

## System overview

`pyCellModeller` is a Python-native multicellular simulation package with a refactored numerical engine built on **PyTorch**.

It is intended to replace the legacy PyOpenCL-based execution model with:
- explicit Python modules
- tensorized simulation state
- CPU-first correctness
- optional CUDA acceleration
- modern testing and packaging

The package is a **simulation library first**. Web services, GUIs, and hosted orchestration belong outside the core package.

## Design principles

1. **Correctness before optimization**
   Scientific behavior must be validated on CPU before performance work.

2. **PyTorch is the engine**
   Use Torch tensors and Torch device semantics as the main execution substrate.

3. **Separate domain logic from tensor plumbing**
   Core simulation concepts should stay readable and testable.

4. **Vectorize where it matters**
   Use tensorized updates for performance-critical kernels, but do not sacrifice clarity unnecessarily.

5. **Keep the public API small and stable**
   Internal modules can evolve faster than the user-facing API.

6. **Infrastructure is not the product**
   Docker, notebooks, and future service integration must not dictate core simulation architecture.

## Major components

### `pycellmodeller.api`
Public entrypoints for:
- creating simulations
- configuring device and runtime behavior
- loading models
- stepping or running a simulation
- saving and loading outputs

### `pycellmodeller.core`
Backend-independent domain objects and contracts:
- simulation config
- cell state schema
- field state schema
- events
- snapshots
- stepping interfaces

This layer should not depend on CUDA-specific behavior.

### `pycellmodeller.engine`
Torch-specific implementation details:
- tensor allocation
- device movement
- dtype handling
- random seeds
- batched update utilities
- optimized kernels where appropriate

This is the direct replacement zone for the old PyOpenCL engine ideas.

### `pycellmodeller.mechanics`
Mechanical simulation logic:
- geometry representation
- neighbor interactions
- growth-induced shape change
- overlap or force resolution
- boundary conditions

### `pycellmodeller.fields`
Extracellular field logic:
- scalar or multi-channel field grids
- diffusion and decay
- secretion and uptake coupling

### `pycellmodeller.biology`
Biological state updates:
- intracellular rules
- regulatory state
- user-provided update functions
- type-specific behaviors

### `pycellmodeller.io`
Trajectory storage, snapshots, exports, and config IO.

### `pycellmodeller.viz`
Optional lightweight visualization adapters. Keep optional.

### `pycellmodeller.cli`
Command-line entrypoints for examples, experiments, and batch runs.

## Data model

### SimulationConfig
Defines:
- timestep
- total steps or stop conditions
- device
- dtype
- seed
- output settings
- mechanics parameters
- field parameters

### CellState
Torch-friendly columnar or structured state for all cells, for example:
- `cell_id`
- `parent_id`
- `cell_type`
- `alive`
- `age`
- `length`
- `radius`
- `position`
- `orientation`
- `velocity`
- `growth_rate`
- user-defined state tensors

### FieldState
Represents extracellular state:
- grid shape
- spacing
- channels
- concentration tensor
- boundary settings

### SimulationState
Single runtime state object:
- current time
- step index
- cell state
- field state
- diagnostics
- pending events

## Core execution flow

### One simulation step
1. validate runtime configuration
2. update growth state
3. compute mechanics interactions
4. update positions and orientations
5. apply intracellular logic
6. update extracellular fields
7. resolve division and lifecycle events
8. emit diagnostics and optionally persist snapshot

## Device model

### CPU
Mandatory reference path.
All new scientific logic must work on CPU.

### CUDA
Optional acceleration path through PyTorch.
Performance-critical modules should support CUDA once correctness is established.

### Other devices
Do not commit to non-CUDA accelerators in v1. They can be evaluated later.

## Container and deployment model

### Native local Python
Primary experience for researchers and developers.

### Docker Compose
Useful for:
- reproducible dev environments
- CI
- Linux/NVIDIA workflows

### Future services
A hosted or WebCM-like system should wrap this package rather than merge into it.

## Testing strategy

- unit tests for domain functions and utilities
- integration tests for stepping loops
- regression tests for reference example behavior
- CPU/CUDA parity checks where reasonable
- benchmark suite kept separate from correctness tests

## Risks and open questions

- direct tensorization of mechanics may require iterative redesign
- some historical CellModeller behaviors may not map cleanly to Torch-first execution
- balancing differentiability with event-heavy simulation semantics may require explicit boundaries
- GPU speedups may depend heavily on data layout and batching strategy
