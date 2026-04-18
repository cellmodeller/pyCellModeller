# AGENT

## Mission

Build `pyCellModeller` as a clean-slate, PyTorch-powered reimplementation of CellModeller with modern engineering standards and clear module boundaries.

## Role split

### ChatGPT owns
- ambiguity reduction
- architecture and tradeoff analysis
- milestone planning
- deciding when new ADRs are required
- reviewing implementation direction against product goals

### Codex owns
- file creation and editing
- package scaffolding
- implementation of approved tasks
- tests
- refactors within approved boundaries
- progress reporting and blocker identification

## Working rules

1. Prefer small, incremental commits.
2. Keep the project installable and testable.
3. CPU correctness comes before CUDA optimization.
4. Do not reintroduce PyOpenCL concepts as implementation baggage.
5. Keep Torch-specific code in `engine` or clearly bounded scientific kernels.
6. Do not hide architecture changes inside “small” edits.
7. Update docs when public APIs or design boundaries change.
8. Make assumptions explicit in commit messages or task summaries.

## Safe tasks

A task is safe when it:
- affects one module or a small related set of files
- preserves documented architecture boundaries
- includes tests when behavior changes
- does not materially change the public API

Examples:
- add dataclasses for config/state
- create package skeleton
- add a CLI stub
- implement deterministic toy stepping
- add unit tests

## Risky tasks

A task is risky when it:
- changes the public API
- restructures package boundaries
- changes state layout semantics
- introduces major dependencies
- changes persistence formats
- changes mechanics assumptions
- expands scope beyond the current milestone

Risky tasks must be reviewed with ChatGPT before implementation proceeds.

## Handoff protocol

### ChatGPT to Codex
Provide:
- goal
- scope
- acceptance criteria
- files likely to change
- non-goals
- whether docs or ADRs must be updated

### Codex to ChatGPT
Report:
- files changed
- tests added or run
- assumptions made
- blockers
- architecture or scope concerns

## Testing expectations

For non-trivial behavior:
- add unit tests
- add integration tests for stepping logic
- add regression tests for example behaviors when appropriate
- validate CPU first, then CUDA if relevant

## Documentation update rules

Update docs when:
- public API changes
- module boundaries change
- engine behavior changes materially
- development workflow changes
- roadmap or scope changes substantially

At minimum, keep `README.md`, `ARCHITECTURE.md`, `PRODUCT.md`, and `AGENT.md` aligned.

## ADR triggers

Create or update an ADR when:
- engine design changes materially
- persistence formats are chosen or revised
- a new execution model is introduced
- the project broadens beyond Torch-first execution
- service integration becomes part of scope

## Hard constraints

- Treat this as a fresh repository.
- PyTorch is the engine in v1.
- NumPy helpers are acceptable for testing or conversion, but not as the main simulator engine.
- Web, GUI, or platform concerns must stay outside the core library unless explicitly planned.
