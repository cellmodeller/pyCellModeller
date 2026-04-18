# pyCellModeller Development Plan

## Goal

Build `pyCellModeller` as a clean, Python-native reimplementation of CellModeller with a refactored simulation engine based on **PyTorch**, replacing the legacy **PyOpenCL/OpenCL** execution model.

This project should prioritize:
- local execution from Python and notebooks
- modern packaging and testability
- CPU and NVIDIA GPU support through PyTorch
- a clean simulation core separated from visualization or web concerns
- future extensibility for differentiable simulation and AI/ML workflows

## Strategic position

This is a **greenfield rewrite**, not a line-by-line port.

The project should preserve the scientific intent of CellModeller:
- multicellular simulation
- growth, mechanics, division, signaling
- Python-driven model definition

But it should modernize the engineering approach:
- remove PyOpenCL dependence
- use typed Python modules and clear package boundaries
- make the engine testable on CPU first
- use PyTorch tensors and kernels as the execution substrate
- keep service or browser integration out of the core package

## Recommended architecture

Use a layered architecture:

1. **Public API layer**
   - stable Python API for users
   - simulation/session construction
   - model registration
   - run/step/save/load operations

2. **Simulation core**
   - domain entities
   - stepping loop
   - event handling
   - configuration
   - deterministic reference logic

3. **Torch engine**
   - tensorized state representation
   - device placement
   - vectorized update kernels
   - optional compilation / optimization hooks
   - performance-sensitive mechanics and field kernels

4. **Scientific modules**
   - mechanics
   - intracellular programs
   - extracellular fields
   - division and lineage logic

5. **Interfaces**
   - CLI
   - notebooks
   - file IO
   - lightweight visualization adapters

## Key design decision

**PyTorch should be the primary and only accelerated engine in v1.**

Do not introduce a general multi-backend abstraction in the initial version.

Why:
- replacing PyOpenCL is already a large modernization step
- premature backend abstraction would slow delivery
- PyTorch gives:
  - CPU support
  - mature CUDA support
  - strong tensor ecosystem
  - autograd and differentiable opportunities
  - easy integration with AI/ML tooling

If multi-backend support is needed later, design it from demonstrated needs, not speculation.

## Development phases

### Phase 0 — repo bootstrap
- create repo docs
- scaffold package and test layout
- set up `pyproject.toml`
- add linting, formatting, typing, CI
- define public API boundaries

### Phase 1 — minimal CPU simulator
- implement config and state dataclasses
- create a Torch-backed state container
- implement a deterministic toy stepping loop
- ensure tests pass on CPU only

### Phase 2 — core simulation primitives
- cell state arrays
- growth and division
- lineage tracking
- simulation snapshots
- event queue or event resolution model

### Phase 3 — mechanics refactor
- port core cell mechanics concepts from CellModeller to Torch
- express geometry and interactions in tensor form
- keep a correctness-first CPU path
- benchmark and optimize hot kernels

### Phase 4 — fields and biology
- extracellular diffusion / decay
- secretion and uptake
- intracellular update hooks
- user-extensible model modules

### Phase 5 — GPU enablement
- ensure Linux/NVIDIA support through PyTorch CUDA
- add device abstraction at the Torch level
- benchmark representative workloads
- keep CPU parity tests mandatory

### Phase 6 — usability and packaging
- CLI runner
- example models
- notebook demos
- trajectory save/load
- reproducibility helpers

### Phase 7 — future-facing extensions
- optional `torch.compile` evaluation
- differentiable simulation experiments
- surrogate modeling interfaces
- future service or web integration in a separate layer

## First 10 milestones

1. Repo docs committed
2. `pyproject.toml` and package scaffold created
3. `pip install -e .[dev]` works
4. Torch-backed config and state objects implemented
5. Toy simulation runs on CPU
6. Unit and integration tests pass
7. Division and lineage tracking added
8. First tensorized mechanics module added
9. CUDA path validated on Linux/NVIDIA
10. One legacy CellModeller example reproduced approximately

## Acceptance criteria for v1 bootstrap

A successful first version should:
- install cleanly
- run locally with only Python + PyTorch
- execute a toy simulation deterministically on CPU
- have a clean module structure for future mechanics work
- make PyTorch the clear replacement for PyOpenCL
- include tests and docs from day one

## What not to do early

- do not recreate the historical repo structure blindly
- do not build a web platform in this repo
- do not over-generalize into a multi-backend abstraction
- do not optimize before CPU correctness is locked in
- do not couple scientific code to visualization code
