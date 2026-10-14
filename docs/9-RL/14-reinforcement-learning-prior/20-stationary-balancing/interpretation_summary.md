# Stationary continuous balancing: trained and independently tested

The stationary formulation works at both tested torque caps. Selected TD3 actors maintained upright balance through ten-minute tests and recovered from two explicit disturbances, with much less holding torque than the previous clock-dependent actor. Full-duration arm compliance remains imperfect, particularly at the weakest cap. The unchanged augmented teacher remains the feasible baseline.

## What was actually trained

Actor and critic receive only arm angle, sine/cosine of pendulum angle, and both angular velocities: five physical features, with no clock or hold memory. The actor directly commands bounded torque. It contains no LQR switch, predictive braking filter, clock cap or hidden action smoothing.

The task is now a discounted continuing objective. Success does not end training or freeze the plant. The reward penalizes departure from upright, energy error, squared normalized torque and arm travel; it contains no time-dependent cost. Sampling resets at 120 seconds, but value targets use the real next state before reset and continue bootstrapping. This is an explicit reformulation of the earlier finite-horizon capture task.

The teacher is unchanged: the original energy-work heuristic plus local LQR balancing and predictive braking, using a one-second prediction at 4.59 mN m and two seconds at 2.295 mN m. Teacher filtering is not applied to the neural actors.

We generated 2,048 teacher episodes per cap, equally split among downward swing-up, moving capture, outward arm braking and perturbed-upright recovery. We retained 1,902 and 2,029 complete trajectories, respectively, that never crossed the arm range and were balanced at the end. Whole episodes were split between training and validation. Replay explicitly balances swing, capture, braking, recovery and quiet-state categories. Repeated equilibrium frames do not dominate by duration.

## Independent ten-minute results

Each row uses 512 new random starts:128 from each of the four groups. Model selection used a separate 256-start validation set. No parameter changes were made from these final test results. “Compliant” means no ±135° arm crossing anywhere in the 600-second run **and** a final uninterrupted hold of at least 10 seconds within 5° of upright. Every controller in every row was balanced at the end of all 512 episodes.

| Controller | Torque cap (mN m) | Test | Full-duration compliant | Final 10 s torque RMS (mN m) | Mean absolute motor work (J) |
|---|---:|---|---:|---:|---:|
| Stationary TD3 | 4.59 | Nominal | **99.6%** (510/512) | 0.0693 | 0.1195 |
| Augmented teacher | 4.59 | Nominal | 92.4% (473/512) | 0.000416 | 0.1088 |
| Stationary TD3 | 2.295 | Nominal | **96.5%** (494/512) | 0.00412 | 0.0666 |
| Augmented teacher | 2.295 | Nominal | 99.2% (508/512) | 0.000430 | 0.0635 |
| Stationary TD3 | 4.59 | Two disturbances | **99.0%** (507/512) | 0.0726 | 0.4243 |
| Augmented teacher | 4.59 | Two disturbances | 70.5% (361/512) | 0.000415 | 0.4456 |
| Stationary TD3 | 2.295 | Two disturbances | **92.4%** (473/512) | 0.00411 | 0.1972 |
| Augmented teacher | 2.295 | Two disturbances | 96.3% (493/512) | 0.000438 | 0.1816 |

Nominal TD3 episodes stayed continuously upright for at least the final 489.42 seconds at 4.59 mN m and 485.58 seconds at 2.295 mN m. Maximum upright errors over the final ten seconds, across all 512 episodes, were 0.0736° and 0.0159°. Small arm offsets are allowed by this angle-only task; the learned actor need not settle exactly at arm zero.

Torque RMS measures motor command magnitude, not electrical consumption. Absolute motor work integrates the absolute value of torque times arm angular speed over the entire run. Disturbance-injected energy is external and is not counted as motor work. The teacher remains substantially quieter once settled. At 4.59 mN m, the new actor's nominal final torque RMS is about 43 times lower than the previous actor's roughly 2.95 mN m, although the earlier cohort and formulation differ, so this is descriptive rather than a paired ablation.

## Disturbances and recovery

At 120 and 300 seconds, each trajectory receives a signed physical state displacement:0.05 rad arm angle,0.12 rad pendulum angle,0.2 rad/s arm speed and 0.35 rad/s pendulum speed. Adjacent episode indices receive opposite signs. These are instantaneous physical disturbances, not changes to the actor's inputs or torque authority.

Every actor and teacher episode completed a new uninterrupted ten-second hold after each disturbance, before the next disturbance or end of the test. For TD3, median completion delays were 22.47/23.69 seconds at 4.59 mN m and 27.54/25.64 seconds at 2.295 mN m. These delays include the ten-second confirmation period. Worst delays reached 114.32 and 170.92 seconds, respectively: some disturbances cause a fall and a fresh swing-up. “Recovered eventually” therefore does not imply rapid local rejection or continuous uprightness throughout the disturbance.

At the first disturbance,511/512 quarter-cap and 509/512 eighth-cap actor episodes had already held upright for the preceding ten seconds; all were settled before the second disturbance. Recovery records preserve this distinction.

## What the three training rounds established

We trained 20 variants over three rounds, each with 1,048,576 online transitions, for 20,971,520 total transitions. Both selected checkpoints include online TD3 updates; neither is merely a relabeled teacher or behavior-cloning-only model.

- **Round 1:** stationary three-layer 256-unit models learned at 4.59 mN m. Raising squared-torque cost to 2 produced 100% compliant 120-second validation with 0.0755 mN m final RMS. A two-layer 128-unit actor failed, and the initial 2.295 mN m models all failed. More torque penalty was not automatically better.
- **Round 2:** 100,000 pretraining updates and three 512-unit layers, with imitation weight 1000, raised weakest-cap compliant validation to 97.3% with 0.0155 mN m RMS. The selected 4.59 mN m actor used three 256-unit layers, 20,000 pretraining updates and imitation 1000, reaching 100% compliant validation with 0.0696 mN m RMS. Both use squared-torque cost 2 and discount 0.995 per 100 ms action.
- **Round 3:** at 2.295 mN m, imitation 10000, 200,000 pretraining steps, arm cost 2 and another seed did not improve the selected validation score. Some pretrained models were quieter but crossed more often; constraint compliance takes priority. Further ordinary tuning showed no clear gain in this tested neighborhood.

Selected checkpoints are [4.59 mN m](../../artifacts/rotary_pendulum/stationary-balancing-20261013/wave-2/quarter_bc1000/machine-scannables/best.msgpack), after 524,288 online transitions, and [2.295 mN m](../../artifacts/rotary_pendulum/stationary-balancing-20261013/wave-2/eighth_wide_bc1000/machine-scannables/best.msgpack), after 786,432. Full configurations and checkpoint hashes are in [selected_models.json](../../artifacts/rotary_pendulum/stationary-balancing-20261013/machine-scannables/selected_models.json). Models require their matching five-feature architecture and torque cap.

## Interpretation and remaining formulation issues

Removing time dependence eliminates the previous out-of-range-clock failure by construction. Continuing reward/bootstrapping explicitly values staying balanced after capture. Targeted recovery/braking examples and stronger imitation provide physical-state coverage that repeated quiet frames alone did not. Sufficient pretraining and capacity were important at the weakest motor, where normalized corrective actions change more sharply with state. These observations support the combined formulation; they do not isolate the contribution of every simultaneous change.

The remaining arm crossings are consistent with the objective: a finite soft cost permits trade-offs, and the actor has no runtime constraint filter. More penalty and pretraining did not eliminate them. To study a hard bound, the next mathematical change should explicitly model admissible states/actions or constrained return, rather than quietly attaching a safety controller and attributing its success to RL.

The teacher is feasible for long holds but is not a globally safe oracle: it also crosses during difficult starts and perturbations. Its two caps use different prediction lengths, so teacher differences cannot be attributed solely to motor strength. The actor's residual torque and arm offsets show that successful angle holding is not the same as minimum-effort equilibrium or readiness for a burst/coast schedule. Ten minutes and this fixed perturbation family do not prove indefinite stability or general robustness.

## Plots, artifacts and verification

Start with the [aggregate comparison](../../artifacts/rotary_pendulum/stationary-balancing-20261013/human-readables/heldout_comparison.png), [nominal episodes](../../artifacts/rotary_pendulum/stationary-balancing-20261013/human-readables/nominal_episodes.png), [final quiet-balance detail](../../artifacts/rotary_pendulum/stationary-balancing-20261013/human-readables/nominal_last10s.png), and [disturbance/recovery detail](../../artifacts/rotary_pendulum/stationary-balancing-20261013/human-readables/disturbance_recovery.png). [Full disturbed episodes](../../artifacts/rotary_pendulum/stationary-balancing-20261013/human-readables/disturbed_episodes.png) show both kicks. Episode plots use the same fixed downward episode 0, not a favorable trajectory selected afterward. Green is the ±5° goal, red is the ±135° arm range, and purple marks disturbances; upright axes are zoomed to ±10°, and arm scales can differ to reveal excursions.

The full archive is `artifacts/rotary_pendulum/stationary-balancing-20261013` from the repository root. `teacher-data` retains all generation successes and failures; `datasets` contains categorized replay and manifests; `wave-1/2/3` contain pretraining, online logs and checkpoints; `teacher-heldout` and `actor-heldout` contain live physical records and per-episode CSVs. Top-level `machine-scannables` contains training/test comparisons, per-disturbance recovery records, selected-model hashes, reward reconstruction checks and `make_report.py` for reproducing these figures. Each launched wave includes source snapshots and environment records. The mathematical protocol and stage plan are under `docs/9-RL/14-reinforcement-learning-prior/20-stationary-balancing` from the repository root.

All 131 repository tests passed, with two existing empty-array warnings. Five changed runtime modules passed strict type checking; changed code passed lint and whitespace checks. Across 2,048 sampled demonstration transitions, maximum reward discrepancy was below 2.4e-7 and next-feature discrepancy below 1e-6; train/validation episode IDs are disjoint. All jobs completed and all eight GPUs were released. Changes are uncommitted; no new dependency was introduced.
