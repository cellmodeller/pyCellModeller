# pyCellModeller

A Python-native reimplementation of CellModeller with a refactored **PyTorch-based engine** replacing the legacy **PyOpenCL/OpenCL** execution path.

## Why this exists

CellModeller demonstrated a powerful model: define multicellular simulations in Python while accelerating the core numerical work on parallel hardware. But the legacy engine architecture is difficult to modernize, package, and extend.

`pyCellModeller` starts fresh:

- Python-native package structure
- PyTorch as the tensor and execution engine
- CPU-first correctness
- CUDA support through PyTorch
- clean separation between simulation logic and infrastructure

## Project goals

The first goal is not feature parity with the historical codebase. The first goal is a **clean, modern, testable foundation** for multicellular simulation.

Planned v1 priorities:
- installable Python package
- stable public API
- Torch-backed simulation state
- deterministic CPU reference behavior
- NVIDIA GPU support through PyTorch CUDA
- modular mechanics, fields, and biology layers
- examples, tests, and reproducible workflows

## High-level architecture

The codebase is organized around five layers:

- **API** — user-facing simulation construction and execution
- **Core** — config, state, stepping contracts, events
- **Engine** — Torch-backed state operations and numerical kernels
- **Scientific modules** — mechanics, fields, biology
- **Interfaces** — CLI, notebooks, IO, visualization adapters

## Expected repo layout

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
      viz/
      cli/
  tests/
    unit/
    integration/
    regression/
  examples/
    scripts/
    notebooks/
  docs/
  docker/
    cpu/
    nvidia/
  .github/
    workflows/
```

## Recommended local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
pytest
```

## GPU workflow

PyTorch should provide:
- CPU execution by default
- CUDA execution on Linux/NVIDIA systems with the correct PyTorch install

Example validation step:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Container workflow

Use containers for reproducibility and CI, not as the only supported way to run the package.

Recommended approach:
- CPU development container by default
- optional NVIDIA-enabled development profile for Linux
- avoid making containerization a prerequisite for normal local Python use

## Quality bar

Run these on every meaningful change:

```bash
ruff check .
ruff format --check .
pytest
```

## How ChatGPT and Codex should use this repo

- ChatGPT owns planning, architecture, tradeoffs, and milestone sequencing.
- Codex owns implementation, tests, and scoped file changes.
- Large refactors should be guided by an ADR or written plan first.
- PyTorch engine code should remain isolated from pure domain logic where possible.

## Status

This repository should be treated as a clean-slate project. Ignore prior structure and build the foundation from scratch.
