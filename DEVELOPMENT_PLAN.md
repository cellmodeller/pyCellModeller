# pyCellModeller Development Plan

## Current baseline (truth source)

As of bootstrap, the implemented system is:

- a minimal public API (`Simulation`, `SimulationConfig`, `SimulationState`)
- a deterministic toy Torch engine (`TorchEngine`) with CPU-first tests
- scaffolded placeholders for mechanics/fields/biology/io/cli

The historical PyOpenCL path is intentionally replaced. v1 planning assumes **Torch-only execution**.

## Guiding constraints

1. Keep docs and plan synchronized with what exists in the repository.
2. Prioritize deterministic CPU behavior before any performance expansion.
3. Grow from the current toy engine incrementally.
4. Do not introduce multi-backend abstractions in v1 bootstrap.

## Implemented repository structure

```text
pyCellModeller/
  pyproject.toml
  README.md
  ARCHITECTURE.md
  PRODUCT.md
  AGENT.md
  ADR-001.md
  DEVELOPMENT_PLAN.md
  src/pycellmodeller/
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
  .github/workflows/
```

Not present yet: `viz/`, `docs/`, `docker/`.

## Near-term roadmap (from current state)

### Phase 1 — solidify toy engine (current)
- keep deterministic stepping stable
- expand config/state validation tests
- maintain truthful docs for bootstrap API

### Phase 2 — enrich state semantics
- introduce explicit cell identity and lineage tensors
- keep migration path from current `SimulationState`
- add regression fixtures for deterministic behavior

### Phase 3 — first mechanics increment
- implement a minimal mechanics rule in `mechanics/`
- call mechanics update from Torch engine step loop
- validate CPU determinism on representative fixtures

### Phase 4 — first biology and fields hooks
- add optional biology callback interface
- add minimal field state container and no-op update path
- keep default behavior backward-compatible

### Phase 5 — usability and IO
- add practical CLI entrypoint(s) beyond scaffold
- add minimal snapshot/export format
- extend example script and notebook flows

## API stewardship plan

Current public API to preserve while expanding:

- `Simulation`
- `SimulationConfig`
- `SimulationState`

When adding new public API:
- update README + architecture docs in the same change
- add/adjust tests covering imports and behavior
- avoid breaking bootstrap examples without deprecation notes

## Quality gates

Every milestone should keep:
- unit tests passing
- integration initialization tests passing
- regression deterministic stepping tests passing
- documentation aligned with implemented code

## Explicit non-goals during bootstrap

- no PyOpenCL reintroduction
- no early generalized backend layer
- no speculative directory additions not present in repo
- no platform/web orchestration work inside core package
