# PRODUCT

## Problem

CellModeller is valuable scientifically, but the legacy PyOpenCL execution model makes modern local development, packaging, and extension harder than it should be.

Researchers need a modern implementation that:
- runs locally in Python
- is easier to test and maintain
- supports current tensor and ML ecosystems
- remains scientifically extensible

## Target users

Primary users:
- computational biology researchers
- synthetic biology researchers
- developers extending multicellular simulation models

Secondary users:
- ML researchers using simulation-generated data
- engineers integrating simulation into downstream tools
- educators creating notebook-based demonstrations

## Jobs to be done

Users need to:
- define a simulation in Python
- run it locally without legacy GPU tooling friction
- extend rules for growth, mechanics, and signaling
- generate trajectories and outputs for analysis
- use Torch-native tensors and workflows where helpful

## Goals

### Core goals
- replace the PyOpenCL engine with a PyTorch engine
- make the package easy to run locally
- establish a maintainable repo with tests and docs
- support CPU and CUDA execution through PyTorch
- create a stable foundation for future scientific extensions

### Strategic goals
- enable AI/ML workflows around simulation outputs
- make differentiable or learned components feasible later
- improve onboarding for new contributors
- support future service integration without polluting the core library

## Non-goals

For v1, do **not** attempt to:
- reproduce every historical CellModeller feature
- preserve the historical file layout
- build a hosted platform
- support every accelerator target
- over-abstract into a generic simulation framework

## Main workflows

### Workflow 1 — local scientist
- install package
- run a provided example
- inspect outputs
- tweak model parameters

### Workflow 2 — simulation developer
- implement or modify a model rule
- run tests on CPU
- validate on CUDA if relevant
- compare trajectories or summary outputs

### Workflow 3 — ML researcher
- run batches of simulations
- export trajectories or summary metrics
- feed outputs into Torch-based pipelines

## v1 scope

### In scope
- package scaffold
- public API
- Torch engine
- CPU reference execution
- CUDA enablement path
- minimal mechanics baseline
- growth and division logic
- tests, examples, docs, CI

### Out of scope
- full legacy parity
- browser-first UI
- service orchestration
- broad visualization tooling
- generalized backend support

## Success criteria

The first working version is successful if it can:
- install cleanly in a new Python environment
- run a toy multicellular simulation on CPU
- use PyTorch as the execution engine
- support a CUDA path for at least one representative example
- save and inspect outputs
- support contributor development with a clean repo structure

## Product constraints

- scientific behavior must remain understandable
- engineering complexity should be introduced only when justified
- CPU correctness remains mandatory
- public API churn should be minimized
- examples and docs are part of the product, not optional extras
