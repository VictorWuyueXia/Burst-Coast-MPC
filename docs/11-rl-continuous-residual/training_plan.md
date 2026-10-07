# Conditional authority for residual TD3 training

## Quick reading

The user has authorized autonomous training and tuning **once the analytical
heuristic is reasonably acceptable**. No additional training permission is
needed after that condition is met. This supersedes the earlier unconditional
“no training yet” instruction with a conditional authorization; it does not
declare the current heuristic acceptable.

The current assessment is **not ready to start training**. Across the existing
22 trials at 20/40/60 s, no downward reset reaches 90% of target energy and no
downward, moving or near-upright reset achieves capture. Tight successes also
occur with zero torque. The best energy-pumping configuration settles around
60% of target, and the interval-work diagnostic exposes a systematic limitation
of the instantaneous energy inverse. These findings are in the
[implementation/result summary](interpretation_summary.md).

A useful residual prior need not solve upright balance by itself. The missing
prerequisite is credible swing-up assistance under the actual 100 ms held
action, rather than persistent low-energy oscillation. Correcting the estimate
of work over that interval is the recommended next development step. This
assessment adds no user-approved numerical success threshold.

## Machine-scannable authority and scope

| Field | Value |
| --- | --- |
| Authority | User message: “if the heuristic turns out to be reasonably acceptable, you can start the training with tuning authority we have discussed before” |
| Authorization | Conditional autonomous TD3 training and iteration |
| Current condition assessment | Not met, based on saved heuristic-only validation |
| Training executed under this authorization | None |
| Permitted physical GPUs | 0, 1, 2, 3 only |
| Policy / physical torque limits | 0.00918 / 0.0204 N m, unchanged |
| Decision / physics intervals | 100 / 20 ms, unchanged |
| Deadlines to compare | 20, 40, 60 seconds |
| Tunable training scope | Training schemes, budgets, optimizer settings, exploration, network sizes, activations and losses, within the agreed task |
| Task contract | Five accepted dense reward terms; nonterminal arm excursions; strict 100 ms hold ends an episode early |
| Required iteration evidence | Training logs, fixed-reset validation, trajectory plots inspected before the next iteration |
| Required handoff | Best reproducible setup and checkpoint, or a failure analysis supported by logs and plots |

## Stages and decision flags

1. **Heuristic reassessment.** Validate a corrected analytical heuristic on the
   existing paired reset fixtures and inspect complete and zoomed trajectories.
   Record whether it supplies reproducible swing-up assistance, how often the
   arm filter intervenes, and whether its held-action energy predictions agree
   with actual motion. Passing does not require a finished balancing controller;
   it requires an explicitly justified assessment that the prior is useful.
   Current flag: **not passed**.
2. **Bounded TD3 pilot, conditional on stage 1.** Compare learned behavior with
   the same heuristic-only resets at 20/40/60 s. Check finite losses and targets,
   action saturation, arm-filter intervention, episode termination, and actual
   capture/recovery beyond favorable initial states. Save and inspect plots.
   Current flag: **not started**.
3. **Autonomous tuning.** Iterate configurations only with a stated hypothesis
   drawn from preceding logs and plots. Use paired seeds and repeat promising
   changes across seeds. Stop expanding a parameter sweep when its evidence
   indicates a structural problem. Current flag: **not started**.
4. **Handoff.** Separate human reports/figures from machine-readable configs,
   histories, checkpoints and numerical trajectories. Include an interpretation
   summary, failure cases, comparison to the heuristic and next-step suggestions.
   Current flag: **heuristic-only handoff already available; training handoff pending**.

No new runtime classes, functions, wrappers or parameters are introduced by
this authorization record. Any subsequent implementation change must first
update the structure budget in [development_plan.md](development_plan.md),
retaining compact task-oriented files, parallel independent computations and
the existing coding-style discipline.
