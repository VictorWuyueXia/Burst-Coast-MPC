# Phase 9: Bang-Off-Bang Q Prior

## Quick reading

The discrete action-value learner is now implemented on the validated JAX rotary environment. It chooses negative full torque, zero torque, or positive full torque. No training has been run yet. The next run must establish upright capture before expanding to general swing-up, and its numerical values must be checked before they become a planning prior.

The reward combines progress toward the pendulum's upright energy and complete-state capture with powered duration, a weak elapsed-time cost, and success/failure outcomes. Energy is guidance: reaching the right energy while rotating through upright does not count as success. Switching penalties and continuous actions are deferred.

Read [formulation_plan.md](formulation_plan.md) for the task, reward, learning target, and later controller interpretation. Read [NN-design.md](NN-design.md) for the seven-input/three-output baseline, dedicated workflow, and explicit removal of the old rotary PPO code/configuration. Read [development_plan.md](development_plan.md) for implementation budgets, initial settings, validation gates, and artifacts. Read [training-plan.md](training-plan.md) for bounded autonomous iteration across training schemes, configurations, model sizes, activations, and losses after training is explicitly started. Each document starts with an intuitive explanation and then gives the implementation contract.

The important experimental comparison is ordinary exploration versus energy-guided exploration. Both networks learn complete action values from random initialization. The heuristic only proposes experience-collection actions; it is not a residual-value baseline or a restriction on learned actions.

Success in this phase means repeatable capture and swing-up, useful action-value estimates, and a demonstrated benefit in a small planning comparison. If full-torque pulses at 10 Hz prevent accurate capture, document that limitation after controlled tuning. Do not quietly change the action set, goal, or decision rate to obtain a passing result.

## Machine-scannable document contract

| Key | Value |
|---|---|
| `phase_id` | `9` |
| `contract_id` | `rotary-q-prior-v1` |
| `document_status` | Frozen specification with implementation complete; training results pending |
| `formulation_authority` | `formulation_plan.md` |
| `implementation_authority` | `development_plan.md` |
| `network_and_retirement_authority` | `NN-design.md` |
| `autonomous_training_authority` | `training-plan.md`, activated by a later instruction to implement/train/run/continue phase 9 |
| `workflow_entry` | `bringup.rotary_q_workflow.train(experiment)` |
| `replacement_policy` | Remove rotary PPO executable/configuration components; implement Q learning from scratch; preserve shared non-PPO dependencies and historical evidence |
| `current_scope` | Three-action Q learning, heuristic exploration comparison, value audit, fixed-horizon deployment test |
| `later_scope` | Richer/continuous torque, switching costs and switching-time optimization, joint sampled-MPC training |
| `language` | English only, as explicitly requested; no Chinese companion files |
| `existing_environment` | Reuse `jax_environment.py` and `jax_dynamics.py` |
| `implementation_status` | PPO runtime/configuration removed; functional JAX task, Double DQN, evaluation, artifact, and workflow modules added |
| `validation_status` | CPU semantic and full repository tests pass; no GPU pilot or training was run |
| `results_in_this_directory` | No learned result; this directory contains the frozen design and implementation status |
| `runtime_artifact_root` | `artifacts/rotary_pendulum/q-prior/` |

### How to interpret specification changes

The task semantics, return definition, observation order, and artifact meanings are fixed. A change to them requires a revised contract identifier before collecting replacement results. Initial optimizer, exploration, and reward coefficients are engineering starting points, not measured optimum settings. The development plan lists the permitted tuning dimensions and how to record revisions.

The numerical acceptance thresholds are proof-of-concept criteria selected for this plan. They are neither predictions of achievable performance nor guarantees about physical hardware. Failure of a gate must remain visible in the report.

### Background and evidence

- [Validated environment summary](../8-jax-rotary-environment/interpretation_summary.md): the simulator is ready for learning experiments.
- [Previous RL discussion](../8-jax-rotary-environment/human-readables/future_rl.md): background decisions superseded by this phase's explicit contract where they differ.
- [Earlier PPO findings](../6.5-rotary-pendulum-PPO-preliminary-results_interpretation_summary.md): energy acquisition alone failed to deliver capture; those experiments used different clocks and are not a matched benchmark.

### Judgment

The smallest useful next result is a calibrated discrete Q prior with verified capture behavior. More GPU throughput does not resolve an unsuitable objective or insufficient action resolution. Inspect those failure modes before increasing model size or adding controller complexity.
