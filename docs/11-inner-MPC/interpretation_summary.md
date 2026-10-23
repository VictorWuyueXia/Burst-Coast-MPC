# Inner MPC with a frozen terminal value

**Official baseline, 2026-10-23: [MPC+RL](human-readables/mpc_rl_baseline.md).** The 16-generation CEM controller with exact LQR stage weights and the frozen RL terminal value is now the development baseline. RL remains enabled after capture; no local handoff is used. Earlier study descriptions below are historical.

**Latest requested test: [LQR-cost CEM + frozen RL value](../../artifacts/rotary_pendulum/inner-mpc/20261022-lqr-cost-confirm-summary/interpretation_summary.md).** All 32 complete 40-second sessions sustain upright after capture; mean total/longest hold is 37.764 s. The existing hybrid remains retained unchanged. See the [exact cost formulation](human-readables/lqr_stage_cost.md) and [record contract](machine-scannables/lqr_stage_cost_contract.md). This is a separate experiment, with residual local motion and a nearly active arm boundary documented explicitly.

**Retained hybrid: [CEM swing-up → existing LQR](../../artifacts/rotary_pendulum/inner-mpc/20261022-hybrid-confirm-summary/interpretation_summary.md), full 40-second sessions.** Mean longest continuous hold increases from 8.663 to 35.520 s on 32 matched starts; all 32 hybrid sessions remain upright after handoff. Use 0.5 s measured stabilization; weak angle/speed additions are zero. [LQR derivation and switching rules](human-readables/lqr_handoff.md), [execution contract](machine-scannables/lqr_handoff_contract.md). Earlier studies below are historical.

Previous experiment: **CEM only, full 40-second sessions**, with the added weak angle costs removed. [Results](../../artifacts/rotary_pendulum/inner-mpc/20261020-speed-confirm-summary/interpretation_summary.md): weak speed strength 0.1 is best by the two mean hold-time metrics, but the gain is small and mixed; zero remains the default. See the [speed formulation](human-readables/weak_speed.md). Solver comparisons and five-second results below are historical.

The inner controller now optimizes squared torque effort, squared normalized arm angle and the existing frozen TD3 terminal value.
The arm-centering term was added explicitly at the user's request; the hard 135° arm boundary remains unchanged.
Timing decisions are still fixed: actuation is enabled across the prediction horizon and only the first 100 ms command is executed before replanning.
The outer burst/coast scheduler is not implemented here.

## Current implementation

The historical GPU comparison used JAX-native SQP and compiled batched CEM.
Both optimize three initial sequences across many independent measured states on-device.
The SQP implementation uses damped BFGS, ADMM quadratic subproblems and batched trial steps.
It is a new SQP implementation, not a claim that SciPy SLSQP itself now runs on CUDA.
The old SciPy path is retained as historical reference code.
CEM keeps population generation, feasible-only elites and distribution updates on-device.
Both retain best-so-far feasible candidates and independently replay them on a 1 ms grid.
Neither modifies the hard boundary or substitutes another controller after rejection.

## Best observed result

The completed GPU campaign contains 18,450 attempts, with 18,443 accepted plans and seven rejected plans.
It includes paired horizon/weight sweeps, numerical tuning, batch scaling, and five-second repeated replanning from 32 new states.
All five targeted GPU tests passed; no CPU solver benchmark was run.

The best observed five-second controller uses CEM with a 1 s horizon, centering rate 10 cost units/s,
1,024 samples per initial plan, 32 elites, 16 generations and normalized-torque initial standard deviation 0.5.
It completed all 32 episodes; 11/32 remained inside the 5° upright band throughout the final second.
Mean final-second arm RMS was 9.96°, compared with 49.20° for zero centering at the same CEM budget.
No downward-start episode achieved that final-second hold, so this is not a solved swing-up/balancing benchmark.

The local solver's single-state timing probe is 1.154 s; CEM's is 0.362 s at its tested scaling setting.
GPU SQP batches 256 simulations in 2.181 s, corresponding to 8.52 ms amortized throughput, not feedback latency.
Neither implementation demonstrates a 100 ms command deadline.
A lower optimized terminal-guided score did not reliably predict better closed-loop behavior.

The centering-weight extension to 30/100 was interrupted when other jobs occupied more than four GPUs.
The 30-weight run's partial records remain available; weight 100 was refused before CUDA initialization.
Weight 10 is best observed, not an identified optimum or weight plateau.
All our GPU jobs are stopped.

## Read the package

- [Formulation](human-readables/formulation.md): revised objective, constraints, frozen actor/critic convention and outer-loop interface.
- [GPU experiment guide](human-readables/gpu_experiment_guide.md): current execution workflow, validation milestones and timing interpretation.
- [GPU algorithm contract](machine-scannables/gpu_contract.md): exact numerical method, parameter roles, feasible-incumbent semantics and structure budget.
- [Original solver-selection discussion](human-readables/solver_selection.md): historical SciPy/CEM choice and why CasADi export was deferred.
- [Implementation contract](machine-scannables/implementation_contract.md): checkpoint, physics, feature map and original RL-cost diagnostic.
- [Results and interpretation](../../artifacts/rotary_pendulum/inner-mpc/interpretation_summary.md): complete tables, artifacts, failures and best-so-far recommendation.

## What remains scientifically unresolved

The frozen value includes upright and energy running costs that the primary MPC segment still omits.
The new centering term does not make that learned continuation fully consistent with the primary objective.
Critic calibration, longer task trajectories, and the declared reward-consistent diagnostic remain useful next comparisons.
Do not silently add costs, boundary reserves or fallback controls to obtain successful-looking episodes.
The recorded finite-grid checks are not a proof of continuous-time safety, recursive feasibility or stability.
CR3BP remains a scaffold and needs its own dynamics and learned value before the same experiment can be transferred.


The 2026-10-20 extension adds [weak local angle tracking and longer paired evaluation](human-readables/weak_tracking.md). The historical five-second scores remain short-window diagnostics.

Completed results: [120-second paired study](../../artifacts/rotary_pendulum/inner-mpc/20261020-tracking-confirm-summary/interpretation_summary.md). All 128 confirmation episodes completed, with no consistent benefit from the weak local term.
