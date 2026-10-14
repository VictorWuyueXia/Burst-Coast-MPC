# Recent development history and agent handoff

**2026-10-13 weak-motor audit:** [Torque and trajectory study](18-weak-motor-study/interpretation_summary.md)
corrects the interpretation of the 2.36 s result: its torque cap was doubled to 18.36 mN m.
Two additional eight-GPU waves compared 16 runs and 15,924,736 finite transitions.
At 4.59/2.295 mN m, fresh downward trajectories need about 10/20 cycles before upright;
all reach the upright band, but only 73.8%/29.3% hold for one second. Almost all cross
the soft arm boundary. Half torque achieves 100% angle success in both seeds, but arm
behavior is seed-dependent. One weak-motor exploration failure reproduces at 20 ms
and stays finite at finer integration. Figures and records are under
`artifacts/rotary_pendulum/weak-motor-20261013/`. All workers exited and GPUs were checked idle;
125 tests, Ruff and seven-module strict mypy passed. No hard arm constraint or integrator
fix has been implemented. Explicit per-run caps now replace implicit teacher/student coupling.

**2026-10-13 sustained-goal update:** [Heuristic demonstration training](17-heuristic-demonstrations/interpretation_summary.md)
changes the JAX goal to strictly less than 5 degrees for one second, with a 40-second default
and 30-second comparisons. Three eight-GPU waves compared 24 learner runs. The selected
TD3 policy passes 2,048/2,048 unseen starts; 28 cases cross the soft arm bound. The actual
heuristic generated 3,801 qualifying trajectories, with 3,048 training episodes retained
as demonstrations. Actor/critic initialization plus persistent demonstration replay works
without a continued imitation loss. A separate failure replay identifies coarse integration
under uncontrolled high-speed exploration. Artifacts: `artifacts/rotary_pendulum/heuristic-demonstrations-20261013/`.
All jobs finished; 123 repository tests passed. The selected checkpoint and strict-goal
comparison are linked in the report. The historical sections below describe earlier goals;
the old 15-degree/100-millisecond result is not the current success definition.

**2026-10-13 update:** [TD3 training diagnosis](16-td3-training-diagnosis/interpretation_summary.md)
completed 32 runs and 56,623,104 transitions on all eight idle GPUs, then audited
16 retained checkpoints on fresh starts. Angle capture succeeds; strict capture
does not show learned recovery. Artifacts are under
`artifacts/rotary_pendulum/td3-diagnosis-20261013/`. All jobs finished and GPUs were
released. User authorization explicitly covered training, tuning and all eight
GPUs for this campaign, superseding the older conditional hold and four-GPU cap.

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

The old single-action interpretation is now `docs/10-heuristic-controller/13-single-action-controller/controller_analysis.md`, because it contains substantive analysis. A new brief interpretation lives with the artifacts. The affected organization pass covered `docs/10-heuristic-controller/12-energy-transfer/` and `docs/10-heuristic-controller/13-single-action-controller/`; the later stage-wide pass also grouped all RL training under `docs/9-RL/`.

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

## Subsequent publication and RL preparation

Commit `831f37e` (`heuristic action 1024/1024`) was pushed to `origin/dev` using the verified force-with-lease; it published the earlier history rewrite and the working changes. The pending-publication wording above records the earlier sequence, not an outstanding approval.

The next task produced an [RL prior assessment and development plan](14-reinforcement-learning-prior/assessment_and_next_step_plan.md) and eight [portable animation episodes](../artifacts/rotary_pendulum/experiment-results/single-constant-torque-action-controller/portable-animation-records/interpretation_summary.md). They include typical captures, the slowest downward capture from one validation seed, exact downward rest, and two types of arm-limit crossing. The user requested force-adding this portable subset under ignored artifact paths; the complete archive remains ignored. No RL training or animation rendering was performed. The new bundle and documents are staged for a later commit, not yet published by this task.

## Signed pendulum coordinates supersede the positive full-turn convention

The user subsequently requested **[−180°, 180°)** pendulum position: 0 downward, both ±180 upright, exact +180 stored as −180. This supersedes the earlier 0–360 representation described above. See the [representation and validation specification](15-signed-pendulum-angle/representation_and_validation.md). NumPy, JAX and CasADi complete steps, resets, RL state reconstruction, current logs and plots now use the signed convention; plots split the upright goal band across both edges. Physical parameters, controller weights and the circular angle-only goal are unchanged.

The portable bundle had been committed as `1cc16ce` before this task. Its eight animation records were converted and their manifest updated to schema 2. All non-angle NumPy trace fields are bitwise unchanged; CSV geometry matches the original to numerical precision. Original run archives and source snapshots retain their historical coordinates, explicitly as provenance. Regenerated current-study summaries still contain 23,668 evaluations across 29 case runs. The three representative validation figures now use signed axes.

Validation: all 118 repository tests passed on CPU (two empty-slice warnings in the existing offline-diagnostics test); changed Python passed lint. Conversion checks verified both wrap directions and all eight portable cases. No new Monte Carlo campaign or RL training was run. These changes are working-tree edits, not a new commit or push.

## 2026-10-13: ±135° arm travel and continued balancing

The [sustained-balance report](19-sustained-balance/interpretation_summary.md) supersedes the earlier ±180° JAX range: JAX and legacy MPC now use ±135°. JAX heuristic/RL margin penalties begin at 108°; violations remain soft and recorded. Existing 20 ms physics observations now use ten 2 ms RK4 substeps to repair a reproduced coarse-integration failure; decisions remain 100 ms. The default goal remains strictly less than 5° for 1 second, while these experiments train ten-second holds with 120-second deadlines and continue physical feedback after capture.

The original heuristic does not reliably sustain balance. An explicitly separate teacher with local LQR and predictive braking holds through 600 seconds at 2.295 mN m on 512 fresh random starts, with no arm crossings. Two eight-GPU TD3 rounds compared 16 variants using qualified extended teacher records, short prefixes, balanced sampling, imitation strength and arm penalties. At 4.59 mN m, short-prefix TD3 succeeds through 120 seconds but fails later as clock inputs leave their training range. Capping only the observed clock at 60 seconds yields 600-second sustained balance in all 512 random starts, with one arm crossing during the full run; none of 128 downward starts crosses. The cap was selected on diagnostic starts, so this is a diagnostic result rather than a new untouched final test. It is saved as an explicit experiment setting, not silently enabled in the default actor.

The successful RL still reverses roughly ±2.95 mN m every 100 ms and oscillates within 0.664° at the end, whereas the augmented teacher settles with nearly zero torque. More quiet demonstration frames did not consistently improve TD3; the 2.295 mN m learner still fails despite a successful teacher. Next work should address stationary actor inputs, capture/recovery coverage and quiet torque-efficient balancing, not only timeout or penalty size. All 128 tests passed; changed runtime modules passed strict type checking and lint. Eight GPUs were released. All new changes remain uncommitted.

## 2026-10-13: stationary continuous-balancing TD3

The [stationary balancing report](20-stationary-balancing/interpretation_summary.md) completes the next formulation experiment. Explicit v3 uses five physical features in actor and critic, a discounted continuing objective, squared torque cost, no time-dependent reward, no success termination, and correct bootstrap through sampling timeouts. The actor commands torque directly, with no teacher switch or predictive filter. The augmented teacher is unchanged and remains the baseline.

New targeted starts cover capture, outward arm braking, upright recovery and downward swing-up. Qualified demonstrations retain complete compliant, settled trajectories and balance five transition categories with episode-disjoint splits. Three rounds trained 20 variants (20,971,520 online transitions). The final weakest-cap round found no further clear gain from stronger imitation, more pretraining, arm penalties or another seed. Selection preceded the independent 600-second nominal and disturbed evaluations on a new seed.

Both selected actors finish all 512 independent episodes upright in each condition and recover after both physical disturbances. Nominal full-duration compliance is 510/512 at 4.59 mN m and 494/512 at 2.295 mN m; disturbed compliance is 507/512 and 473/512. Nominal final-ten-second torque RMS is 0.0693 and 0.00412 mN m, with maximum upright error 0.0736° and 0.0159°. This resolves sustained low-effort balance for these tested conditions; soft arm constraints still permit crossings. Recovery can require a new swing-up and take more than two minutes. No indefinite/global robustness claim is made.

Artifacts are under `artifacts/rotary_pendulum/stationary-balancing-20261013`, with human plots/interpretations separated from machine records. Selected checkpoints are `wave-2/quarter_bc1000` at 524,288 transitions and `wave-2/eighth_wide_bc1000` at 786,432, with full settings and hashes. All 131 tests passed (two existing warnings), five changed runtime modules passed strict type checks, and lint/whitespace checks passed. All GPUs were released; no dependency, commit or push was added. The next mathematical issue is full-duration constraint handling and efficient equilibrium, rather than another hidden control heuristic.
