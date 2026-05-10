# Visualization, Animation, And Experiment Artifacts Plan

## Summary
Add a small, opt-in recording and visualization layer around the existing synchronous episode flow. Keep the simulation/control code mostly unchanged: `Coordinator` still owns the episode loop, `Environment` still emits `StepRecord`, and new helper modules handle artifact writing and realtime Matplotlib views.

Use **CSV + JSON** for v1 because it is human-readable, dependency-light, easy to inspect, and enough for the current dense pendulum records. Add Matplotlib as the only new runtime visualization dependency.

## Key Changes
- Add an `artifacts` config section with defaults:
  - `root-dir: artifacts/experiments`
  - `alias: null`
  - `enabled: true`
- Each run creates one dedicated directory:
  - With alias: `artifacts/experiments/{safe_alias}_{YYYYMMDDTHHMMSS}`
  - Without alias: `artifacts/experiments/{YYYYMMDDTHHMMSS}`
  - Use local timezone, sanitize alias to lowercase letters/numbers/`-`/`_`, and fail if the directory already exists.
- Store recreate-focused files in that directory:
  - `config.json`: fully resolved `RootConfig` using YAML aliases.
  - `metadata.json`: run directory, config package, CLI args, package version, Python version, platform, git commit if available, dirty git flag if available, start/end wall datetimes.
  - `steps.csv`: one row per `StepRecord`, stable column order, aliases such as `theta-rad`, `u-applied-nm`.
  - `summary.json`: final `ExperimentSummary`, including final observation.
  - `run.log`: structured log lines for the experiment after artifact setup.
  - `manifest.json`: artifact format version, file list, row counts, and completion status.
- Add `artifacts/` to `.gitignore`.

## Runtime Flow
- Extend `wsmpc run-episode` with:
  - `--artifact-root PATH`
  - `--run-alias TEXT`
  - `--no-artifacts`
  - `--visualize`
  - `--animate`
  - `--viz-update-every N`
- Keep artifacts enabled by default, because experiment data should be recorded unless explicitly disabled.
- Create a small artifact writer, likely under `src/wsmpc/utils/artifacts.py`, with:
  - `create_run_directory(...)`
  - `write_config(...)`
  - `open_step_writer(...)`
  - `write_step(record)`
  - `write_summary(summary)`
  - `finalize_manifest(...)`
- Modify `Coordinator.run_episode()` minimally to accept optional callbacks:
  - `on_episode_start(observation)`
  - `on_step(observation, record)`
  - `on_episode_finish(summary)`
- The CLI wires callbacks to the artifact writer and visualizers. Existing tests and direct `Coordinator` use continue to work without callbacks.

## Visualization
- Add a lightweight Matplotlib module, likely `src/wsmpc/visualization/realtime.py`.
- Realtime plots use one figure with grouped subplots sharing `t_sec`:
  - State: `theta_rad`, `omega_rad_s`
  - Observation/diagnostics: `energy_j`, `energy_error_j`, `constraint_margin`, `goal_flag`
  - Actions: `u_commanded_nm`, `u_applied_nm`
- Update plots every `viz-update-every` steps using in-memory lists from callback data.
- Realtime animation uses a second Matplotlib figure when `--animate` is enabled:
  - Draw pivot at origin.
  - Draw pendulum rod with length equal to `environment.pendulum.length_m`.
  - Draw bob size proportional to `sqrt(mass_kg)` so mass changes are visually noticeable without huge scaling.
  - Use angle convention from current dynamics: `theta_rad = 0` is downward, `theta_rad = pi` is upright.
  - Bob position: `x = length_m * sin(theta_rad)`, `y = -length_m * cos(theta_rad)`.
  - Draw torque marker as a curved arrow or signed text/color marker near the pivot, scaled against `torque_limit_nm`.
  - Keep axes equal and auto-sized from `length_m`.

## Public Interfaces
- New config model:
  - `ArtifactConfig` with `enabled`, `root_dir`, and `alias`.
  - Add `artifacts: ArtifactConfig` to `RootConfig`.
- New CLI options override config values for artifact root, alias, artifact disabling, and visualization.
- New artifact directory format version: `1`.
- New dependency:
  - Add `matplotlib>=3.8` to project dependencies, imported lazily by visualization code so non-visual runs stay simple.

## Test Plan
- Add artifact tests using `tmp_path`:
  - Run a short episode with artifact writer enabled.
  - Assert directory naming includes alias and datetime-like suffix.
  - Assert `config.json`, `metadata.json`, `steps.csv`, `summary.json`, and `manifest.json` exist.
  - Assert `steps.csv` row count equals `summary.records_emitted`.
  - Assert CSV headers use stable aliases.
- Add CLI tests:
  - `run-episode --artifact-root tmp --run-alias smoke` creates artifacts.
  - `run-episode --no-artifacts` still runs and does not create a run directory.
- Add callback tests:
  - `Coordinator.run_episode()` invokes step callback once per emitted record.
  - Existing coordinator/environment behavior remains unchanged when callbacks are omitted.
- Add visualization smoke tests with Matplotlib `Agg` backend:
  - Realtime plot accepts one or more records and updates without error.
  - Pendulum animation computes bob position from `theta_rad`, `mass_kg`, and `length_m`.
- Run full `pytest`.

## Assumptions
- CSV + JSON is the v1 standard; Parquet/Zarr can be added later behind a new artifact format version.
- Artifacts are recorded by default; realtime visualization and animation are opt-in CLI flags.
- The first version is for local synchronous experiments, not remote dashboards or notebook replay.
- Git metadata is best-effort: if git is unavailable, metadata records `null` values instead of failing the experiment.
