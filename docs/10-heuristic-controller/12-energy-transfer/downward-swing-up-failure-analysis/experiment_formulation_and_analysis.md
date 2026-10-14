# Downward Monte Carlo study: reproducibility and data contract

> Archive note: the method and commands below describe the controller at the time of the experiment. Reproduction requires its original source snapshot; the current `src/` has since changed. Collected figures, records and logs are in [experiment artifacts](../../../artifacts/rotary_pendulum/experiment-results/downward-swing-up-failure-analysis/).

Read [the interpretation report](../../../artifacts/rotary_pendulum/experiment-results/downward-swing-up-failure-analysis/interpretations/interpretation_summary.md) for conclusions. This specification defines the numerical records and the experiment mechanics. The study starts from the energy-work controller introduced at commit `8f3a4c58b53fdee7fe6919ed46252b9f4c2518de`, with the requested defaults promoted after repository HEAD `dc2f15ee7dae3a59b390b54b1374c40bb90c2d16`. Source hashes and snapshots, rather than a commit alone, identify the executed uncommitted source.

## Scope and implementation budget

No classes, data classes or helper wrappers were added. Four scripts each contain one task-oriented `main`: `prepare_runs.py` samples paired states, snapshots sources and schedules GPUs; `run_rollouts.py` executes and audits trajectories; `analyze_runs.py` classifies outcomes and computes paired comparisons; `plot_results.py` renders the retained tables and selected traces. Each script stays between 40 and 300 lines. The principal data objects are campaign/case dictionaries, initial-state arrays, trace dictionaries, episode/summary tables and plot selections. Existing decoder/environment/artifact functions were edited directly.

Milestones and success flags:

1. Default promotion: explicit torque, arm-angle and angle-only capture contracts; all 115 repository tests passed. The default CLI completed 260 lanes on CPU, without a `--config` argument.
2. Monte Carlo reference: four seeds, 8,192 distinct randomized states, including 6,144 downward states. All traces finite, torque caps respected, capture and excursion flags independently reconstructed.
3. Diagnosis and intervention: identical initial arrays for every paired case; 24 isolated variations; three candidates confirmed on three separate seeds; four fine-plant cases; 41,984 independent held-action replays. All runs completed and provenance is retained.

No dependencies were added. `pyproject.toml` now packages the default heuristic JSON in addition to YAML configurations. `AGENTS.md` and the previous study artifacts were not changed by this task.

## State, physics and control definitions

Each state row is `[theta, alpha, omega, nu]`: `theta` is unwrapped arm angle in rad; `alpha` is unwrapped pendulum angle in rad with zero downward and pi upright; `omega` is arm angular speed in rad/s; `nu` is pendulum angular speed in rad/s. These are simulated physical states. Angles are wrapped only for measuring distance from upright, not for arm-bound checking.

The following model constants are computed from `physics.yaml` by `derive_model`, not fitted in this campaign:

| Symbol | Meaning | Value |
|---|---|---:|
| `I_a` | Arm rotational inertia | 0.00022879166666666672 kg m² |
| `I_b` | Arm-axis base inertia including the carried pendulum | 0.00040219166666666673 kg m² |
| `I_p` | Pendulum hinge inertia | 0.00013312800000000002 kg m² |
| `c` | Inertial coupling coefficient | 0.00013158 kg m² |
| `G` | Gravitational torque coefficient | 0.01518588 N m |

Define `K_a` as arm-body kinetic energy, `K_p` as full pendulum-body kinetic energy including carried motion and coupling, `V` as gravitational potential relative to downward, and `E_star` as the gravitational energy at upright. All energies are computed in joules:

$$
K_a=\tfrac12 I_a\omega^2,\qquad
K_p=\tfrac12(I_b-I_a+I_p\sin^2\alpha)\omega^2+c\cos\alpha\,\omega\nu+\tfrac12 I_p\nu^2,
$$

$$
V=G(1-\cos\alpha),\qquad E_\mathrm{star}=2G=0.03037176\ \mathrm{J}.
$$

`K_p` includes motion carried by the arm; it must not be interpreted as pendulum hinge kinetic energy alone. The physical total is `K_a + K_p + V`. Both damping coefficients are zero.

Define `k` as the tunable dimensionless work gain, `w_a` as the tunable dimensionless arm-energy weight, `W_max` as the fixed per-decision requested-work cap, and `W_req` as the upper policy's requested motor work. `clip` restricts its first argument to the stated interval. The default parameter array `[work_gain, kinetic_weight, work_weight]` is `[0.04, 1, 1]`; `kinetic_weight` is `w_a`. `work_weight` is a separate dimensionless decoder work-error penalty coefficient.

$$
W_\mathrm{req}=\operatorname{clip}\left(k(E_\mathrm{star}-V-K_p-w_aK_a),-W_\mathrm{max},W_\mathrm{max}\right),
\qquad W_\mathrm{max}=0.04E_\mathrm{star}=0.0012148704\ \mathrm{J}.
$$

The policy therefore requests energy injection when its weighted energy is below the upright requirement. Lowering `w_a` discounts temporary arm storage in this request; it does not alter the physical equations. The cap stays fixed in all cases, including the work-gain sweep.

Define `u` as the selected constant motor torque, `theta_start` and `theta_end` as measured arm angles at the start and end of the actual active action, and `W` as delivered work. A final action may last less than 100 ms because capture ends the episode:

$$
W=u(\theta_\mathrm{end}-\theta_\mathrm{start}).
$$

The usable torque cap is 0.01836 N m, 90% of the unchanged physical actuator limit 0.0204 N m. The original cap was 45%, hence “double torque.” RK4 physics samples are 20 ms; five form a 100 ms decision. `recovery_steps=20` means a 0.4 s prediction including the first 0.1 s pulse. Later preview torques are clipped arm brakes, refreshed every 0.1 s. Only the first pulse is applied before replanning.

Default candidate search uses 33 grid torques and 32 bracket-root candidates, with ten bisections per bracket. A candidate is “recoverable” only if finite, its predicted peak absolute arm angle is at most pi minus 0.05 rad, and its predicted terminal absolute arm speed is at most 0.15 rad/s. The speed interventions change only that last scalar. Actual arm excursions use pi, without the 0.05 rad preview margin, and remain nonterminal. Exact work means absolute work error at most 0.000003037176 J. Exact recoverable candidates have priority over other recoverable candidates; if recovery cannot be found, the finite-candidate branch minimizes predicted boundary violations and outward motion. Modes are 0 exact work, 1 recoverable work override, 2 unresolved recovery, 3 numerical failure. No mode-3 trace occurred.

Capture measures the wrapped distance of `alpha` from pi. It requires at most pi/12 rad for five consecutive active 20 ms samples. This is a nominal 100 ms discrete hold, not a continuous-time proof or sustained stabilization test. There are no arm-position or angular-speed capture conditions in the JAX environment. Episodes freeze on capture or time out at 1,000 physics samples. Legacy NumPy goal code still uses position and velocity fields in the shared mission configuration; the legacy TD3 torque prior is a separate controller.

## Sampling, stages and pairing

NumPy `default_rng(seed)` samples each coordinate independently and uniformly within the following bounds. Group order is the order in `counts`; moving-angle signs are independent equiprobable ±1. The same saved float64 initial arrays are supplied to every case. Per-seed lane ranges refer to the full 2,053-lane array.

| Label | Count per seed / lanes | Lower state row | Upper state row |
|---|---|---|---|
| downward | 1,024 / 0–1023 | `[-0.2,-0.2,-0.5,-0.5]` | `[0.2,0.2,0.5,0.5]` |
| rest | 256 / 1024–1279 | `[-0.2,-0.02,-0.05,-0.05]` | `[0.2,0.02,0.05,0.05]` |
| wide_down | 256 / 1280–1535 | `[-2.8,-0.35,-0.5,-0.5]` | `[2.8,0.35,0.5,0.5]` |
| moving | 256 / 1536–1791 | `[-0.5,0.4,-2,-6]` | `[0.5,2.6,2,6]`, then signed alpha |
| near | 128 / 1792–1919 | `[-0.25,pi-0.25,-1,-1]` | `[0.25,pi+0.25,1,1]` |
| tight | 128 / 1920–2047 | `[-0.08,pi-0.12,-0.15,-0.3]` | `[0.08,pi+0.12,0.15,0.3]` |

The five deterministic probes, lanes 2048–2052, are `[0,0,0,0]`, `[0,pi,0,0]`, `[2.8,0,0,0]`, `[-2.8,0,0,0]`, and `[0,0,0.5,0]`. They repeat across seeds and are excluded from all randomized rates and inferential claims.

| Configuration | Seeds | Jobs | Randomized evaluations |
|---|---|---:|---:|
| baseline.json | 20261020–20261023 | 4 | 8,192 |
| pilot.json | 20261020 | 8 | 16,384 |
| threshold_sweep.json | 20261020 | 4 | 8,192 |
| refinement.json | 20261020 | 8 | 16,384 |
| kinetic_refinement.json | 20261020 | 4 | 8,192 |
| confirmation.json | 20261021–20261023 | 9 | 18,432 |
| fine_plant.json | 20261020 | 4 | 8,192 |

There are 41 jobs, 83,968 randomized evaluations and 205 probe evaluations. The extra CPU default-command check is excluded from those totals. Seed identifiers are arbitrary RNG integers, not execution dates. “Confirmation” excludes the exploration seed, but is not a preregistered statistical trial. Wilson intervals in `summary.csv` describe specified random reset groups; probe intervals are not used. Paired comparisons match seed and lane and assert exact equality of all four initial coordinates.

## Trace and table schema

Full NPZ traces have 200 decisions and 2,053 lanes. `physics_x` has shape `(200,2053,5,4)`; `physics_active` has shape `(200,2053,5)` and must be used to exclude frozen samples. `start_x` and `x` are `(200,2053,4)`. `energy_j` contains `[K_a,K_p,V]` at each decision endpoint. Scalar fields have shape `(200,2053)`. Arrays are finite; continuous state/control data use float64.

- `torque_nm`, `work_j`, `requested_work_j`, `predicted_work_j`: applied torque, actual work, upper request and decoder prediction. `elapsed_s` is actual active duration, `time_s` is elapsed simulated time. Applied torque is logged as zero after freezing.
- `success`, `timeout`, `arm_violation`, `goal_count`: cumulative capture/deadline/soft-bound status and consecutive goal samples.
- `mode`, `root_count`, `recoverable_count`, `exact_count`, `bounded_count`: selected branch, matched bracket roots, fully recoverable candidates, exact recoverable candidates, and finite candidates satisfying the predicted angle constraint. Grid and root entries can coincide; counts are candidate entries, not distinct physical paths.
- `grid_work_min_j`, `grid_work_max_j`: extrema over the sampled grid, not a proof of the continuous work range. `unreachable_request_fraction` in the episode table means outside these sampled extrema only.
- `predicted_peak_arm_rad`, `predicted_terminal_arm_speed`, `selected_energy_cost`: selected preview quantities. The energy cost is half the sum of squared normalized deviations of `[K_a,K_p,V]` from `[0,0,E_star]`. `energy_preference_gap` is selected cost minus the minimum cost over recoverable candidates, or over finite candidates when none recover. It diagnoses selection tradeoffs, not achieved closed-loop cost.
- `work_mismatch_j` is actual minus requested work for active decisions. `energy_balance_j` is physical total-energy change minus actual work; cumulative sums audit numerical drift.

`episodes.csv` has 84,173 rows, one per evaluated lane. It records case, stage, seed, lane, stratum, initial coordinates, outcome, work totals and decoder fractions. Fractions divide by active decisions, giving each episode equal weight when medians are reported. `no_root_fraction` counts zero matched bracket roots; `speed_reject_fraction` counts bounded candidates existing while no fully recoverable candidate exists. `negative_despite_request_fraction` counts positive work request with negative delivered work. `exact_energy_gap_fraction` counts mode 0 with normalized energy-cost gap above 0.01. `saturated_fraction` uses at least 99.9% of the torque cap.

For work cancellation, define `W_plus` as the sum of positive actual action work, `W_minus` as the positive magnitude of summed negative action work, and `C` as the fraction of total absolute work that cancels. It is computed as:

$$
C=\frac{2\min(W_\mathrm{plus},W_\mathrm{minus})}{W_\mathrm{plus}+W_\mathrm{minus}}.
$$

If no work is exchanged, `C` is defined as zero. This is a motor-work accounting statistic, not dissipative loss or actuator efficiency. The plant has no damping. Gross work remains meaningful for active samples even if an episode captures early.

Failure labels are mutually exclusive: `captured`; `brief_visit` if uncaptured but at least one active sample is in the angle band; `low_lift` if uncaptured, never in the band and peak potential fraction below 0.5; otherwise `partial_lift`. Potential fraction is `(1-cos(alpha))/2`. Longest consecutive goal samples are recomputed from flattened physics traces.

`tail_*` energy fields average decisions in 15–20 s; recurrence fields compare the final 50 decision states with lags 5–30 decisions using coordinate scales `[pi,pi,5,10]`, then select the minimum RMS distance. These are descriptive diagnostics only for timeouts. Successful episodes have frozen tails, making their tail energy/recurrence unsuitable as evidence of maintained balance or periodic motion. Classification itself uses active physics samples, not decision endpoints or recurrence thresholds.

`summary.csv` aggregates cases over all available seeds. `paired.csv` compares each case with the default on matching starts, counting rescued and lost captures separately. To reproduce confirmation-only results, filter `episodes.csv` by seeds 20261021–20261023, then by case and stratum. Figure selection takes the median peak potential within each available ordinary-downward failure class of the first seed containing that class. The paired example takes a median baseline low-lift case rescued by the speed-only intervention. `plot_selection.json` preserves exact selection identities and metrics.

## Audits, artifact layout and reproduction

Each run samples 1,024 evenly spaced complete 100 ms actions and replays them through independent NumPy dynamics with fifty 2 ms RK4 steps. `integration_audit.json` records sampled flattened decision/lane indices, maximum state-component differences, work differences and energy-balance errors. These are local held-action checks, not full trajectory equality. Fine-plant cases additionally use ten 2 ms JAX RK4 substeps inside each environment step, while leaving the decoder model and the 20 ms sampling clock unchanged. Their exact source changes are in `experiment.patch`.

Readable PNGs are in `../../human-readables/downward_study/`. Machine `records/` contains CSV tables, selected NPZ traces, plot selection, retained initial arrays, logs, source hashes, patches and the reference source snapshot. Full NPZ rollouts and per-case complete source snapshots remain under `artifacts/rotary_pendulum/experiment-results/downward-swing-up-failure-analysis/raw-runs/<stage>/`; this large working archive is gitignored. The compact retained source snapshot and patches allow reconstruction without depending on later runtime edits. Historical torque/range scripts targeted earlier source constants; use their saved historical snapshots to reproduce those old studies.

Run from the repository root with the existing `.venv`. The GPU launcher checks all device occupancies before every process: it stops launching if more than four GPUs are occupied or a selected GPU is occupied, and uses only the explicitly provided GPUs 0–3. It prints per-job heartbeats every five wall seconds; workers print each simulated second. Use a new output directory.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python \
  scripts/experiments/downward-swing-up-failure-analysis/prepare_runs.py \
  --config artifacts/rotary_pendulum/experiment-results/downward-swing-up-failure-analysis/records/baseline.json \
  --output artifacts/rotary_pendulum/experiment-results/downward-swing-up-failure-analysis/raw-runs/reproduced-baseline
```

Other configurations use the same command. The launcher snapshots the current source; exact historical replay requires the retained reference source, worker and case patches. Analyze by passing the baseline stage first, followed by the desired paired stages:

```bash
.venv/bin/python scripts/experiments/downward-swing-up-failure-analysis/analyze_runs.py \
  --runs artifacts/rotary_pendulum/experiment-results/downward-swing-up-failure-analysis/raw-runs/baseline \
         artifacts/rotary_pendulum/experiment-results/downward-swing-up-failure-analysis/raw-runs/pilot \
         artifacts/rotary_pendulum/experiment-results/downward-swing-up-failure-analysis/raw-runs/threshold-sweep \
         artifacts/rotary_pendulum/experiment-results/downward-swing-up-failure-analysis/raw-runs/refinement \
         artifacts/rotary_pendulum/experiment-results/downward-swing-up-failure-analysis/raw-runs/kinetic-refinement \
         artifacts/rotary_pendulum/experiment-results/downward-swing-up-failure-analysis/raw-runs/confirmation \
         artifacts/rotary_pendulum/experiment-results/downward-swing-up-failure-analysis/raw-runs/fine-plant \
  --output artifacts/rotary_pendulum/experiment-results/downward-swing-up-failure-analysis/records/records
MPLCONFIGDIR=/tmp/bcmpc-mpl .venv/bin/python \
  scripts/experiments/downward-swing-up-failure-analysis/plot_results.py \
  --records artifacts/rotary_pendulum/experiment-results/downward-swing-up-failure-analysis/records/records \
  --output artifacts/rotary_pendulum/experiment-results/downward-swing-up-failure-analysis/figures
```

Localized `-CN.md` documents are present locally and excluded by the existing `*-CN.*` gitignore rule. The main report distinguishes measured improvements from untested structural suggestions; no claim of long-term balance, hardware robustness, exhaustive parameter optimality or RL-prior effectiveness follows from this campaign.
