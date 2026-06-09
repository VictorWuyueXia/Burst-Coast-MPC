# WakeSleepMPC Style Cleanup Plan

## Summary

- Apply the repo style rule strictly: compact line-of-logic files, minimum realization, no fallback behavior, no unnecessary CLI/config surface, no `try/except`, no thin wrappers, and concise professional comments only where code blocks need orientation.
- Scope: active `src`, tests, configs, and tracked source archive cleanup. Leave `docs/_archive` and the user-modified `docs/prompt.md` untouched.
- Preserve the current `generate-mc-data --epochs` CLI option because the user selected that exception.

## Key Changes

- Remove dead tracked source archive: delete `src/wsmpc/mpc/_archive/` and any imports/tests that could imply it is active.
- Freeze runtime/system policy in code: remove `RuntimeConfig`, `runtime:` YAML blocks, `utils/resources.py`, `utils/parallel.py`, and runtime tests.
- Keep simulation and mpc parameters (steps, term weights, etc.) in config.yaml untouched.
- Check for unused config entries in data generation config, and remove unused ones
- Simplify CLI flow in `src/wsmpc/cli.py`: keep only `app`, `console`, `run_episode_command`, `generate_mc_data_command`, and `main`; inline `load_runtime_context`, remove nested observer callbacks where practical, and keep `--epochs`.
- Simplify `Coordinator.run_episode`: remove `try/except KeyboardInterrupt`; let interruption propagate loudly and delete the interrupted-summary test path.
- Split oversized files only where it improves comprehension: reduce `utils/config_schema.py` below 250 lines by removing runtime schema and dead fields; split `visualization/realtime.py` into compact diagnostics and pendulum animation modules.
- Remove unused or weak surface: delete `NodeStatus` if only tests use it, remove context-manager hooks from `ArtifactWriter`, and consolidate single-use artifact helpers only when that keeps files readable.

## Planned Structure Budget

- Classes/objects:
  - `Coordinator`: owns one normal MPC episode.
  - `MonteCarloDataGenerator`: owns sequential offline RL transition generation.
  - `CasadiMPCController` and `NaturalPeriodMPCController`: active controller boundary and natural-period solver.
  - `ArtifactWriter`: writes run artifacts, without context-manager methods.
  - `RealtimeEpisodePlot` and `PendulumAnimation`: split into separate visualization files.
- Data classes:
  - Keep `_ActivePlan`, `SplitCandidate`, `CandidateSolution`, `SelectedPlan`, `MonteCarloAction`, `EpisodeResult`.
  - Remove `RuntimeResourceReport`; remove `NodeStatus` unless a real runtime caller appears during implementation.
- Functions/methods:
  - Keep active public entrypoints: `load_config`, `load_data_generation_config`, `run_episode_command`, `generate_mc_data_command`, `main`.
  - Keep numerics: pendulum dynamics, RK4, natural-period features, candidate solve/build/split functions.
  - Remove wrappers/helpers: `load_runtime_context`, `ordered_process_map`, resource helpers, archive helpers, context-manager hooks.
- Independent variables/constants:
  - Keep `CONFIG_ROOT`, `STANDARD_PACKAGE`, `DATA_GENERATION_PACKAGE`, `IPOPT_OPTIONS`, `DIAGNOSTIC_PHASE_EPSILON`.
  - Add one frozen MPC worker constant near the controller, e.g. `MPC_WORKER_COUNT = 1`.
- Parameters:
  - Keep CLI parameters: `run-episode --config-package`, `--alias`, `--no-visual`, and `generate-mc-data --epochs`.
  - Remove runtime YAML parameters: `node-id`, `max-worker-threads`, `blas-threads`, `cpu-affinity`, `set-env`.
  - Keep experiment/data parameters that change actual experiment setup.

## Test Plan

- Update config tests to assert runtime blocks are absent and missing fields fail loudly.
- Update CLI tests for preserved `--epochs`, removed runtime mutation, and artifact outputs.
- Update coordinator tests to remove interrupted-summary behavior and verify normal episode/event-trigger behavior.
- Update artifact/message tests after `NodeStatus` and context-manager cleanup.
- Run `conda run -n wsmpc python -m pytest`.

