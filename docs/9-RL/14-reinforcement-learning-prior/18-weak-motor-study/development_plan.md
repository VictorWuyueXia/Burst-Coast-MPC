# Torque usage and weak-motor learning

Keep compact task-oriented logic, no new classes, thin wrappers, fallback behaviors or dependencies; Python files should remain 40–300 lines. Reuse the actual dynamics, teacher decoder, TD3 learner and immutable demonstration mechanism.

## Stages and success flags

1. Inspect existing held-out physical records. Plot representative downward episodes (median duration, extremes, and an arm-crossing example) with applied torque, saturation, pendulum/arm motion, physical energy and motor work. Pass: torque bounds checked numerically, energy/work balance audited, episode-selection indices and full metrics saved.
2. Make the motor limit explicit in teacher/student task settings. Compare 18.36, 9.18, 4.59 and 2.295 mN m; keep mass, damping, timing, reward coefficients and the five-degree/one-second/40-second goal fixed. Pass: new torque-limit tests, matched normalization, no reuse of physically invalid strong-motor transitions as weak-motor demonstrations.
3. Generate lower-torque demonstrations and evaluate the existing policy under reduced authority. Train paired weak-motor TD3 trials, inspect results, then adjust training if justified. Pass: actual lower-cap trajectories and qualified successful transitions, recorded finite training, closed-loop successes and number of pump cycles. Distinguish transfer, retraining, and teacher feasibility.
4. Test retained candidates on new starts, plot gradual energy buildup if learned, and report limits. Pass: independently recounted holds, bound checks, quantitative saturation/work/cycle records, bilingual summaries, code checks and released GPUs. Do not claim a weak-motor solution from clipping an existing plot or from short near-upright starts alone.

## Structure and parameter budget

No new classes/data classes/wrappers. Reuse `EnvState`, `Actor`, `Critic`, `learner`, `replay`, `settings`, teacher traces and dataset manifests.

- `records.py`: one `main` function; objects `records`, `metrics`, `selection`, `figure` organize physics traces, per-episode torque/energy statistics, deterministic representative selection and plots. Independent quantities: cap, positive/negative motor work, saturation duration, turning points, complete oscillations before first upright encounter, capture time. All thresholds and units declared in the report.
- Existing runtime functions `residual_action`, `filter_torque`, `transition`, `collect`, TD3 `update`, `evaluate`, `write_validation` and workflow `train`: propagate one explicit `torque_limit_nm` through applied torque, replay/critic normalization and plots. Retain current constants as legacy API defaults only where already called without settings; full training settings explicitly declare the motor limit.
- Existing teacher `decode`, `evaluate`, validation `main`, and data preparation `main`: use the declared motor cap for candidate torques and replay reconstruction. No new teacher policy.
- `evaluate.py`: one `main` function for restoring/checking the policy at stated caps, independent teacher/policy numerical comparisons and fresh-start audits; reuse the existing rollout/plot writers. At most one substantive rollout closure if teacher-policy traces must share a kernel.
- Reuse the experiment launcher for independent GPU runs and source snapshots. Move the existing substantive trajectory-plotting block to one `plot_trajectories` function in one module only if needed to keep edited files under 300 lines.
- At most three new semantic tests: cap enforcement/normalization, demonstrator/student action parity, and energy/work/cycle interpretation. Adapt existing affected assertions rather than duplicating tests.

New tunable parameter: `torque_limit_nm` only. Experiment knobs remain existing teacher penalties, network widths, pretraining/update counts, exploration and demonstration mixing. Seeds, rollout populations and budgets are recorded in JSON. The cap ladder is an empirical experiment design, not a theoretical minimum motor torque. Historical stronger-policy artifacts retain their original cap and source snapshots.

## Completion status

All four stages were executed. The reporting-only `machine-scannables/make_comparison.py` artifact uses top-level records/metrics/selection/figure objects and no additional functions or classes. Two eight-GPU training waves and fresh paired evaluation are complete. Slow pumping is demonstrated, but reliable weak-motor capture with bounded arm travel is not solved. One numerical failure is preserved and independently replayed; simulator and hard-bound fixes are explicitly future work.
