# PRODUCT

## Product statement

`pyCellModeller` is a clean-slate reimplementation of CellModeller focused on a Torch-based simulation core.

For bootstrap v1, the product is intentionally narrow: **a deterministic CPU-first toy engine with a minimal public API**.

## Explicit backend decision

The PyOpenCL execution path is intentionally retired for this rewrite.

- v1 backend strategy: **Torch-only**
- no PyOpenCL compatibility layer
- no multi-backend architecture in bootstrap scope

## Target users (current phase)

- Developers establishing the simulation core architecture
- Researchers validating deterministic stepping behavior
- Contributors building future mechanics/biology/fields modules on a tested foundation

## Current user value delivered

Users can currently:
- install the package scaffold
- import stable top-level API symbols
- initialize a deterministic toy state
- run deterministic CPU stepping
- run unit/integration/regression tests for the implemented path

## Current implemented API surface

- `pycellmodeller.Simulation`
- `pycellmodeller.SimulationConfig`
- `pycellmodeller.SimulationState`

Supported flow:
1. build config
2. create simulation
3. initialize example state
4. step simulation deterministically

## In-scope for current bootstrap

- CPU-first deterministic behavior as reference
- torch tensor state model
- minimal stepping kernel (`x <- x + v*dt`)
- documentation and tests that match actual implementation

## Not yet implemented (intentionally)

- production mechanics models
- extracellular field simulation
- biological rule engine
- persistence/export workflows
- rich CLI workflows
- visualization module (`viz` directory not present)
- repository `docs/` and `docker/` directories (not present)

## Success criteria for this stage

This stage is successful if:
- documented API matches what imports and runs
- deterministic CPU stepping remains stable under tests
- architecture stays clean for future mechanics/fields/biology additions
- docs clearly state that PyOpenCL was replaced by Torch-only v1
