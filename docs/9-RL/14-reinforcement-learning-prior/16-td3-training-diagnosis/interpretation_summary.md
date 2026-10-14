# Completed TD3 training diagnosis

The authorized eight-GPU campaign completed 32 training runs and 56,623,104
transitions, followed by fresh-seed audits of 16 checkpoints. All jobs completed.

Eight retained angle-trained policies each captured all 1,024 held-out starts.
The two deeper direct policies had no observed arm crossings and averaged about
2.35 seconds from downward starts. Strict-trained policies did not learn recovery;
their few successes ended at the initial 0.10-second hold. A fixed local linear
controller captured all 256 tight starts under the strict definition.

Read the [full interpretation and next experiment](../../artifacts/rotary_pendulum/td3-diagnosis-20261013/interpretation_summary.md)
and [held-out comparison](../../artifacts/rotary_pendulum/td3-diagnosis-20261013/human-readables/heldout_results.png).
The [protocol](../../../../artifacts/rotary_pendulum/reference-data/9-RL/14-reinforcement-learning-prior/16-td3-training-diagnosis/machine-scannables/protocol.md) defines the exact task and learner;
[development plan](development_plan.md) records structure and experimental decisions.
The JSON plans in `machine-scannables` reproduce each wave into a new output path.

The evidence supports two interacting learning issues: extremely scarce successful
strict-task experience, and slow value propagation. More updates substantially
improved value calibration but did not discover capture. Neither model capacity,
filter-induced gradient blockage nor noise magnitude alone explains the failure.
The next controlled test is local-controller training labels and retention of
successful replay, followed by checking whether RL preserves learned recovery.
This intervention remains proposed, not demonstrated.

Relevant code is `src/rotary_pendulum/RL/jax_td3.py`, the existing residual modules,
`src/bringup/rotary_residual_workflow.py`, and the three scripts under
`scripts/experiments/td3-diagnosis/`. Runtime modules retain the planned compact
structure. Verification passed: 120 repository tests, 13 focused final tests,
Ruff, and strict mypy with `--no-sqlite-cache` for this Python build.
