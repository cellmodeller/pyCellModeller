# AGENT

## Mission

Build `pyCellModeller` as a truthful, incremental Torch-first rewrite with deterministic CPU behavior as the bootstrap reference.

## Current project reality

- The implemented API is currently minimal: `Simulation`, `SimulationConfig`, `SimulationState`.
- The implemented engine is `TorchEngine` with deterministic toy stepping.
- `mechanics`, `fields`, `biology`, `io`, and `cli` are scaffolded namespaces and mostly placeholders.
- PyOpenCL is intentionally not part of the rewrite path; v1 is Torch-only.

## Role split

### ChatGPT owns
- planning and tradeoff decisions
- architecture-level guardrails
- milestone sequencing
- ADR change decisions

### Codex owns
- code and test changes
- doc updates tied to implementation
- keeping docs synchronized with current scaffold
- surfacing blockers and scope drift quickly

## Working rules

1. Keep docs aligned with what is implemented today (not aspirational state).
2. Prefer small incremental changes with passing tests.
3. Treat CPU deterministic behavior as the reference path.
4. Keep Torch-specific execution details inside `engine`.
5. Do not reintroduce PyOpenCL compatibility work in v1.
6. Do not list folders/modules in docs that do not exist in the repo.

## Safe tasks (current bootstrap)

- Extend `SimulationConfig`/`SimulationState` validations
- Add deterministic toy-engine behavior and tests
- Add incremental API methods consistent with current architecture
- Fill in scaffolded modules one layer at a time
- Update README/architecture/product docs after each API change

## Risky tasks (require explicit review)

- Public API breaking changes
- Module boundary restructures
- Introduction of alternative execution backends
- Major data model changes that invalidate existing tests
- Large persistence format decisions

## Testing expectations

For behavior changes:
- unit tests for config/state invariants
- integration tests for simulation lifecycle methods
- regression tests for deterministic stepping values

Run CPU path first. CUDA checks are optional at this stage unless explicitly requested.

## ADR triggers

Create/update ADRs when:
- engine design changes beyond deterministic toy stepping
- backend strategy changes from Torch-only v1
- state model introduces major new persistent structures

## Hard constraints

- Keep this repository as a clean rewrite, not a PyOpenCL port.
- Keep bootstrap scope narrow and truthful.
- Prefer clarity and determinism over premature optimization.

## API and compatibility guardrails

- Do not introduce CL/OpenCL names in public APIs.
- Do not add compatibility shims unless explicitly requested by maintainers.
- Preserve Torch-native naming and architecture decisions across docs and code.

