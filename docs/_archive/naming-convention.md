# Burst-Coast MPC Naming And Task-Parallel Layout

## Summary

Rename the repo/package to Burst-Coast MPC while keeping shared runtime files flat in the main package. Do not introduce a `core/` wrapper layer. Keep `cli.py`, `coordinator.py`, `data_generation.py`, `__main__.py`, and common utilities directly under the main package, and move task-specific physics, schemas, MPC formulations, messages, visualization, configs, and training code into parallel task packages.

## Risk And Structure Decisions

- Use `burst_coast_mpc` as the main Python package; use `burst-coast-mpc` for CLI, conda env, distribution, and repo folder.
- Use task packages under `src/`: `src/inverted_pendulum/` and `src/CR3BP/`, per your requested parallel layout.
- Avoid `src/inverted-pendulum/` because hyphens are invalid in Python imports.
- `src/CR3BP/` is legal but case-sensitive; all imports must consistently use `CR3BP`.
- Keep `docs/_archive/` unchanged.
- Keep config/artifact schemas task-owned where fields differ; do not force pendulum-shaped records onto CR3BP.

## Key Changes

- Packaging:
  - Rename distribution from `wake-sleep-mpc` to `burst-coast-mpc`.
  - Rename command from `wsmpc` to `burst-coast-mpc`.
  - Rename conda env from `wsmpc` to `burst-coast-mpc`.
  - Package discovery includes `burst_coast_mpc*`, `inverted_pendulum*`, and `CR3BP*`.
  - Delete generated `wake_sleep_mpc.egg-info/` and regenerate after editable install.
- Shared runtime:
  - Move current `src/wsmpc/cli.py`, `coordinator.py`, `data_generation.py`, `__main__.py`, and `__init__.py` into `src/burst_coast_mpc/`.
  - Keep `Coordinator` as the shared synchronous episode owner.
  - Make CLI task selection explicit with mutually exclusive launch flags:
    - `burst-coast-mpc --inverted-pendulum`
    - `burst-coast-mpc --inverted-pendulum generate-montecarlo-data`
    - `burst-coast-mpc --cr3bp`
    - `burst-coast-mpc --cr3bp generate-montecarlo-data`
  - The CLI selects the task module and passes its concrete config, environment, controller, artifact plotting, and data-generation classes into the shared coordinator path.
- Inverted pendulum task:
  - Move current pendulum-specific modules into `src/inverted_pendulum/`: environment, dynamics, messages, config schema, configs, MPC formulation, visualization, Monte Carlo action utilities, and offline-training scripts.
  - Preserve current behavior and current tests after import updates.
  - Keep active controller naming as the pendulum task’s internal formulation name unless the code itself needs a cleaner task-local rename.
- CR3BP task:
  - Add only an empty package scaffold when needed: `src/CR3BP/__init__.py` plus task-local subdirectories mirroring the pendulum layout.
  - Do not implement CR3BP equations, policies, configs, NN models, or training in this rename pass unless explicitly requested later.
  - CR3BP will share the coordinator pattern, but owns its own state/action messages, dynamics, MPC problem, hyperparameters, artifacts, and NN/training code.

## Test Plan

- Text sweep confirms old naming remains only in `docs/_archive/` or intentional historical prose:
  - `rg -n 'wake[-_ ]?sleep|WakeSleep|Wake Sleep|wake_sleep|wake-sleep|\\bwsmpc\\b|wake_sleep_mpc'`
- Import checks:
  - `python -m burst_coast_mpc --help`
  - `python -c 'import burst_coast_mpc, inverted_pendulum, CR3BP'`
- CLI checks:
  - `burst-coast-mpc --inverted-pendulum --help`
  - `burst-coast-mpc --inverted-pendulum run-episode --help`
  - `burst-coast-mpc --inverted-pendulum generate-montecarlo-data --help`
- Regression checks:
  - `conda run -n burst-coast-mpc ruff check .`
  - `conda run -n burst-coast-mpc python -m pytest`

## Assumptions

- The first implementation pass preserves inverted-pendulum behavior and only creates the structure needed for CR3BP.
- The main package remains `burst_coast_mpc`; task packages are siblings under `src/`.
- `CR3BP` is intentionally capitalized as a package directory and import name.
- No generic base classes or thin wrapper hierarchy should be added until CR3BP creates proven duplication.
