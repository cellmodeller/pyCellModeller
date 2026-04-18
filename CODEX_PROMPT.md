# Codex Prompt for Bootstrapping `pyCellModeller`

You are working in a brand-new repository: `cellmodeller/pyCellModeller`.

Treat this as a clean-slate project. Ignore any prior structure or placeholders that may exist in the repository. The repository should be set up from scratch around the requirements below.

## Project summary

Build `pyCellModeller`, a Python-native reimplementation of CellModeller with a refactored **PyTorch-based engine** that replaces the legacy **PyOpenCL/OpenCL** execution path.

This is a **simulation library first**, not a web platform.

## Product goals

The repository should establish a clean, modern foundation for:
- multicellular simulation in Python
- Torch-based numerical execution
- CPU correctness first
- optional CUDA execution through PyTorch
- future mechanics, fields, and biology extensions
- future AI/ML workflows built on simulation outputs

## Hard constraints

- Python 3.11
- package under `src/pycellmodeller`
- use `pyproject.toml`
- use `pytest`, `ruff`, and type hints
- use PyTorch as the simulation engine
- do not use PyOpenCL
- do not recreate the old repo layout blindly
- do not introduce multi-backend abstraction in v1
- do not build web, Unity, or browser code in this repo
- keep the project installable and runnable after each step

## First deliverables

Create or replace these core docs first:
- `README.md`
- `ARCHITECTURE.md`
- `PRODUCT.md`
- `AGENT.md`
- `ADR-001.md`
- `DEVELOPMENT_PLAN.md`

Then scaffold the package with an initial layout like:

```text
src/pycellmodeller/
  __init__.py
  api/__init__.py
  api/simulation.py
  core/__init__.py
  core/config.py
  core/state.py
  core/events.py
  engine/__init__.py
  engine/torch_engine.py
  mechanics/__init__.py
  fields/__init__.py
  biology/__init__.py
  io/__init__.py
  cli/__init__.py
tests/
  unit/
  integration/
  regression/
examples/
  scripts/
  notebooks/
```

## First implementation target

Deliver a minimal but real installable package that can:

1. import successfully
2. create a simulation config
3. initialize a tiny Torch-backed simulation state
4. step the simulation deterministically on CPU
5. expose a small public API
6. pass tests

A toy stepping rule is acceptable at first. Real mechanics can come later. The priority is clean architecture, tests, and a credible Torch-based engine foundation.

## Suggested public API

Aim for something like:

```python
from pycellmodeller.api.simulation import Simulation
from pycellmodeller.core.config import SimulationConfig

config = SimulationConfig(device="cpu", dt=0.1, seed=1)
sim = Simulation(config=config)
sim.initialize_example()
sim.step()
state = sim.state
```

## Implementation guidance

- Use Torch tensors for simulation state.
- Keep config/state structures typed and explicit.
- Separate pure domain logic from Torch plumbing where practical.
- Design for CPU correctness first.
- Keep CUDA as an optional device path, not a bootstrap requirement.
- Add tests with every non-trivial behavior change.
- Keep docs aligned with the codebase.

## Initial testing expectations

Add tests for:
- package import
- config creation
- simulation initialization
- deterministic stepping on CPU
- state shape and type expectations

## Definition of done for the bootstrap phase

The bootstrap phase is complete when:
- the repo has strong foundational docs
- the package installs
- the public API is present
- a tiny Torch-backed simulation runs on CPU
- tests pass
- the code clearly replaces PyOpenCL with a Torch-based architecture

## Execution style

Work in small, reviewable commits.

Do not overengineer. Build the minimum clean foundation first.

If you encounter a decision that changes API shape, state semantics, or engine boundaries, stop and capture it in a new ADR before continuing.
