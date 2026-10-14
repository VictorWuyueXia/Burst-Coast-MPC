# Sustained upright training now works

The JAX task now requires **upright error strictly below 5 degrees for one consecutive second**, with a **40-second default deadline** and explicit 30-second comparisons. The control interval remains 100 milliseconds.

The selected TD3 policy passed **2,048/2,048 unseen starts** across two independent validation sets. Its mean downward-start capture time was **2.36 seconds**. There were **28 arm-bound crossings (1.37%)**, including one among 512 downward starts. Success here is angle-only; it does not require a centered, motionless arm or indefinite balancing.

We generated 3,801 qualifying heuristic trajectories from 4,096 random starts. The 40-second learning split contains **3,048 episodes / 248,888 transitions**; 753 complete episodes are withheld. After behavior cloning and critic initialization, retaining successful demonstrations in half of every TD3 minibatch solved the task with both tested seeds. Continued imitation loss was not required. Actor-only initialization followed by ordinary TD3 lost its captures; actor-plus-critic initialization without retained data was seed-sensitive.

Read the [full experiment interpretation](../../../../artifacts/rotary_pendulum/heuristic-demonstrations-20261013/interpretation_summary.md), [training comparison](../../../../artifacts/rotary_pendulum/heuristic-demonstrations-20261013/human-readables/training_comparison.png), and [held-out comparison](../../../../artifacts/rotary_pendulum/heuristic-demonstrations-20261013/human-readables/heldout_comparison.png). The [protocol](../../../../artifacts/rotary_pendulum/reference-data/9-RL/14-reinforcement-learning-prior/17-heuristic-demonstrations/machine-scannables/protocol.md) defines the data and all controls; the [plan](development_plan.md) records milestones and structure.

Useful artifacts:

- [Selected checkpoint](../../artifacts/rotary_pendulum/heuristic-demonstrations-20261013/wave-3/replay-only-s1/machine-scannables/best.msgpack) and [matching settings](../../artifacts/rotary_pendulum/heuristic-demonstrations-20261013/wave-3/replay-only-s1/machine-scannables/settings.json).
- [Demonstration dataset manifest](../../artifacts/rotary_pendulum/heuristic-demonstrations-20261013/datasets/machine-scannables/manifest.json).
- [Machine-readable results](../../artifacts/rotary_pendulum/heuristic-demonstrations-20261013/machine-scannables/heldout.csv).

Three eight-GPU waves compared 24 training schemes/seeds, followed by eight held-out audits. Three scratch runs stopped on nonfinite simulation values; the completed comparisons retained 21,105,664 finite online transitions. A separate reproduction showed the offending 100-millisecond action diverges with 20-millisecond integration even in float64, while finer integration stays finite. Successful data fixes the tested learning problem; numerical validity under uncontrolled exploration remains a separate simulator issue.

Verification: **123 repository tests passed**; targeted Ruff and strict mypy checks passed. No new dependency. Runtime artifacts and Chinese companion documents remain gitignored. Historical 15-degree/100-millisecond results must be interpreted using their source snapshots.
