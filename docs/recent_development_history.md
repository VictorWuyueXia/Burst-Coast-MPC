# Recent development history and agent handoff

Recorded on 2026-10-08. This is a handoff for the current working directory, not a claim that all work is committed. Numeric experiment seeds such as `20261022` are identifiers, not execution dates.

## Read these first

1. [Repository instructions](../AGENTS.md): coding discipline, experiment reporting, bilingual documents and shared-GPU rules.
2. [Current controller analysis](13-single-action-controller/controller_analysis.md): physical rationale, measured outcomes and remaining failures.
3. [Controller formulation and experiment specification](13-single-action-controller/controller_formulation_and_experiment_specification.md): exact equations, parameter roles, sampling rules and evidence definitions.
4. [Experiment artifact index](../artifacts/rotary_pendulum/experiment-results/README.md): figures, brief interpretations, numerical records and original runs.
5. The Git section below before changing branches, committing or pushing.

## What happened

| Stage | Operation and conclusion |
| --- | --- |
| Original analytical controller | Work started from commit `8f3a4c5`, whose rewritten equivalent is `d77c09b`. Later commits mainly archived experiments. The long-term project concerns burst control and sleep, but the current analytical controller has no scheduled sleep behavior. |
| Torque, arm range and goal studies | Compared original, half and double usable torque; arm ranges including ±180 degrees; and a goal requiring only pendulum angle within 15 degrees of upright. Angular-velocity conditions were removed from the goal. Double torque and ±180 degrees at 100-millisecond decisions became the selected default. |
| Timing comparison | A smaller integration-step experiment and a separate 40-millisecond decision experiment were run. These are distinct changes. The current default retains a 20-millisecond physics integration step and 100-millisecond decisions. The goal uses five consecutive 20-millisecond samples. |
| Downward-start diagnosis | Monte Carlo runs exposed limitations of the former appended braking prediction, terminal arm-speed restriction and energy weighting. Earlier proposals and results remain historical evidence; they are not the current controller specification. |
| Angle convention | Pendulum position now uses the full interval from 0 inclusive to 360 degrees exclusive: downward at 0, upright at 180. Simulation state uses the equivalent radians. Arm angle and both angular velocities remain signed. A signed upright error may be computed separately; it is not pendulum position. |
| Single-action controller | Removed the appended 300-millisecond slowing prediction, hard predicted arm-speed cap, adjustable arm-energy weighting and inward arm-limit margin. Each tested torque is held constant for exactly the next 100-millisecond prediction. |
| Artifact organization | Moved 789 files and eight original-run directories into descriptive experiment folders. Moved reproduction scripts into `scripts/experiments/`. Recent documentation folders now retain plans, formulations and substantive analyses. |
| Local history cleanup | Removed 86 relocated artifact paths from every commit in local `dev`. Only four commit hashes changed. Publication was pending approval at handoff time; see the subsequent authorization below. |

## Current controller and evidence

The usable torque is ±0.01836 newton metres and the arm soft limit is exactly ±180 degrees. Physical actuator capacity is a separate ±0.0204 newton metres. The controller predicts five 20-millisecond integration steps under one constant torque; it does not append another action. Capture requires pendulum position between 165 and 195 degrees for five consecutive samples, with no angular-velocity requirement. Episodes end at capture or after 20 seconds, so these experiments do not establish sustained balancing.

Physical energy logs include the arm body's kinetic energy, the pendulum body's full kinetic energy including arm-carried motion and coupling, and gravitational potential energy. No tuning coefficient rescales these physical energies. The controller requests motor work toward upright-rest total energy and favors pendulum elevation. As potential energy approaches total energy, less kinetic energy remains; this explains the intended slowing mechanism without imposing an immediate arm stop. It is not a convergence proof.

The current tuning parameters are work-request gain `0.004`, work-error penalty `0.02` and arm-limit penalty `10000`. These are dimensionless controller-design choices, not physical constants. See the formulation for their definitions. “Tested torque” means one torque value evaluated for the next 100 milliseconds; “parameter setting” means a combination of controller parameters.

The experiment collection contains 23,668 episode evaluations across 29 case runs, including repeated comparisons on 6,400 distinct random initial states. Do not count repeated parameter comparisons as independent initial states. For the selected setting, the two final validation seeds produced:

- 4,096 captures from 4,096 random starts, including all 1,024 downward starts.
- No downward-start arm-limit crossings; six moving-start episodes crossed the soft limit.
- Downward median peak absolute arm speed of about 11.41 radians per second and median absolute speed at capture of about 0.46 radians per second; median capture time about 11.1 seconds.
- A separate 2-millisecond plant-integration run captured all 2,052 cases on one validation seed, with zero downward crossings and two moving-start crossings.

Across the larger parameter comparison and both final validation seeds, the selected setting had seven first-crossing events. Three had no tested torque capable of avoiding a crossing during the next action; four had bounded alternatives but the finite soft penalty selected a crossing. These seven events cover a different sample set from the six final-validation crossings. The maximum observed arm angle was 189.03 degrees.

Implementation entry points: [energy calculation and work request](../src/rotary_pendulum/heuristic/energy.py), [torque selection](../src/rotary_pendulum/heuristic/decoder.py), [default controller parameters](../src/rotary_pendulum/configs/heuristic.json), [mission definition](../src/rotary_pendulum/configs/mission.yaml), [evaluation](../src/rotary_pendulum/heuristic/evaluation.py), and [controller tests](../tests/test_energy_heuristic.py).

## Where results and reproduction code now live

Start at `artifacts/rotary_pendulum/experiment-results/`. Each experiment separates:

- `interpretations/`: short English and Chinese result explanations.
- `figures/`: readable plots with expanded parameter names in collected filenames.
- `records/`: configuration files, numerical tables and audit evidence; `run-records/` groups collected evidence by original run identifier.
- `raw-runs/`: original trajectories, logs and source snapshots where available.

The current study is [single-constant-torque-action-controller](../artifacts/rotary_pendulum/experiment-results/single-constant-torque-action-controller/). Its [analysis script](../scripts/experiments/single-constant-torque-action-controller/analyze_study.py) reads the relocated original runs and regenerates summaries and figures. The [relocation manifest](../artifacts/rotary_pendulum/experiment-results/records/file_relocation_manifest.json) records original paths, new paths and pre-move checksums.

All files under this artifact root remain Git-ignored by existing rules. They are available in this workspace, not guaranteed to exist in a fresh clone. Preserve or transfer them separately when moving machines. Chinese `-CN` companions are also ignored. Historical run identifiers and paths inside original records were retained for provenance. Older reproduction workflows require their archived controller source, not today's `src/`.

The old `docs/13-single-action-controller/interpretation_summary.md` is now `controller_analysis.md`, because it contains substantive analysis. A new brief interpretation lives with the artifacts. The affected organization pass covered recent `docs/12-energy-transfer/` and `docs/13-single-action-controller/`; it did not reorganize every older documentation chapter.

## Git state: important before continuing

Local branch: `dev`, at `a7b9cc1c9f7d846298cb5a41ec3f1398e089cdff`. There is substantial uncommitted controller, test and documentation work. Do not discard it or treat HEAD alone as the current implementation.

| Original commit | Rewritten local commit |
| --- | --- |
| `8f3a4c58b53fdee7fe6919ed46252b9f4c2518de` | `d77c09b410b7ea46b9b3afd102ad7c959f84b4e3` |
| `96159eeba9073336c416679a8f949229b616be6e` | `5b6c883bdf581568a40253416045c473de7ddbbc` |
| `016d6176afb3a370657fbf7479d7ad4418ee4760` | `43f17921f17c3f8cee2d3aabef76b5236078a241` |
| `dc2f15ee7dae3a59b390b54b1374c40bb90c2d16` | `a7b9cc1c9f7d846298cb5a41ec3f1398e089cdff` |

The 86 removed paths comprised 3 images, 47 JSON files, 21 CSV files, 8 NumPy archives, 2 YAML files, 2 text files and 3 brief Markdown interpretations. Source scripts and substantive documents were preserved in history. Their old paths still show 11 working-tree deletions because their replacements were moved into new script/document folders; those moves have not yet been committed.

The remote was not pushed. `origin/dev` still records `dc2f15ee7dae3a59b390b54b1374c40bb90c2d16`; that is a local tracking observation, not a newly queried server value. Old history therefore remains reachable through that reference and recovery data. The operation removed paths from local `dev` commit trees; it did not purge old objects from every reference, reflog, backup or GitHub storage.

At handoff time, publication using `--force-with-lease` was awaiting approval. The user subsequently explicitly requested staging all changes, committing with message `heuristic action 1024/1024`, and pushing to origin. This authorizes publication of the rewritten `dev` branch. The live remote tip was verified as `dc2f15ee7dae3a59b390b54b1374c40bb90c2d16`; use that exact lease and refuse to overwrite unexpected remote changes. Do not merge the old remote branch into the rewritten branch, because that would reintroduce the removed history.

Recovery materials are in `/tmp/controller-artifact-history-cleanup/`: `original-history.bundle`, `original-index`, working and staged patches, `removed-paths.txt`, `commit-mapping.json`, and working-file checksums. This temporary directory is outside the repository and may be cleaned by the host. It intentionally preserves the original history for recovery.

## Validation already completed

- Controller implementation stage: 118 repository tests were reported passing. A later test-local float64 setup correction passed its four targeted tests. These were earlier checks, not a full suite rerun during the file organization or Git rewrite.
- Artifact organization: all 789 moved files present; numerical and image checksums unchanged; local links in the reorganized reports resolved; affected documentation folders contained only Markdown. Unrelated broken links in older chapters were left untouched.
- Reproduction after relocation: the current study analysis ran successfully and reproduced 23,668 evaluations across 29 case runs and the reported arm-speed statistics. Python syntax, lint and whitespace checks passed for the affected scripts/changes.
- History rewrite: compared all 75 local `dev` commit trees with their originals, allowing only the 86 selected path removals. Only four commit hashes changed. Verified all 1,104 checked working files unchanged, unrelated Git status identical, no staged changes, and no selected path remaining in local `dev` history. This handoff is an additional working-tree change after that check.

## Suggested next steps and completion checks

1. **Finish repository housekeeping if requested.** Review the current diff, preserve the user's edits, and commit coherent source/document moves when authorized. Publication of the rewritten history was subsequently authorized with the requested commit; verify the resulting remote state before further work. Completion means no accidental artifact additions and an explicitly approved, verified remote update if one is requested.
2. **Investigate remaining moving-start boundary crossings.** Read the [first-crossing records](../artifacts/rotary_pendulum/experiment-results/single-constant-torque-action-controller/records/boundary_events.json) and representative trajectories. A steeper overshoot penalty beginning at exactly 180 degrees is a proposed experiment, not a demonstrated fix. Keep the single 100-millisecond prediction and physical energy definitions. Compare paired states, then separate validation seeds; report capture rate, crossing rate and integration sensitivity, including regressions.
3. **Clarify the next control objective before extending the method.** Capture, sustained upright balancing, reinforcement-learning initialization and eventual burst/sleep scheduling are different tasks. No new RL prior training or sleep mechanism was established by this work. Choose and define the next objective rather than inferring it from the project's long-term goal.

Use precise, defined language and descriptive file names. For GPU experiments, first inspect shared-server occupancy and follow `AGENTS.md`; independent runs should use available parallel resources and print progress. File organization and Git cleanup require no GPU run.
