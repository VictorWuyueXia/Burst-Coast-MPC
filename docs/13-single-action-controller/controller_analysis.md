# One constant-torque action, predicted for 100 ms

The default analytical controller now predicts **only the next 100 ms at one constant torque**. It has no predicted braking continuation, terminal arm-speed constraint, arm-energy weight, or 177.1° inner boundary. The usable torque remains ±0.01836 N m; the arm soft limit remains ±180°.

The method regulates physical total energy and favors conversion into pendulum potential energy. It is a finite search over torques using the nonlinear physical model. Its penalty strengths are control-design choices, explicitly separate from measured energy.

## What “candidate” means

In action selection, a candidate is **one torque value being tested**, such as +0.005 N m held for the next 100 ms. It is not a sequence of different torques. The implementation tests 33 evenly spaced torques and up to 32 additional values obtained by solving for requested motor work. I use “tested torque” here.

In the earlier parameter-study discussion, “candidate” instead meant a tested controller parameter setting. That different use was ambiguous; this report calls it a “parameter setting.”

## Why the arm can slow near upright

Let **E** be physical total mechanical energy, **Kₐ** the arm body's kinetic energy, **Kₚ** the pendulum body's full kinetic energy, and **V** the pendulum's gravitational potential energy, all in joules. These are computed from state, configured masses and link geometry. The pendulum term includes translation caused by the moving arm, rotation, and their coupling.

$$
E=K_a+K_p+V.
$$

Let **E★** be the computed potential energy at upright rest: **0.03037176 J** for this model. The motor-work request raises total energy when it is below this target and removes energy when it is above. Among the tested torques, the controller favors a higher pendulum position while penalizing work-request error and crossing the real arm boundary.

$$
K_a+K_p=E-V.
$$

As total energy approaches E★ and potential energy approaches E★, the kinetic energy available to both bodies approaches zero. This is the intended slowing mechanism. There is no instruction to stop the arm after each action, and there is no arm-speed threshold in torque selection. The 100 ms prediction and tuned penalties do not prove convergence; the measurements below test the behavior.

The familiar factor ½ in kinetic energy comes from mechanics. It remains in the physical calculation. The previous 0.5 **arm-energy preference coefficient** was a separate control weighting, not a physical-energy correction; that coefficient is removed from the current controller altogether.

## Tested behavior

Parameter development used seed 20261020, then a larger comparison used seed 20261021. The selected setting was fixed before the final validation seeds **20261022 and 20261023**. Each final seed supplied 512 starts in each of four classes. Four fixed checks per seed are counted separately.

| Initial-state class | Captures / random starts | Captures without an arm-limit crossing | Episodes crossing ±180° |
|---|---:|---:|---:|
| Downward, near rest | 1,024 / 1,024 | 1,024 | 0 |
| Moving, broader angles and velocities | 1,024 / 1,024 | 1,018 | 6 |
| Near upright | 1,024 / 1,024 | 1,024 | 0 |
| Tight upright | 1,024 / 1,024 | 1,024 | 0 |

All eight fixed checks also captured, including exact downward rest. A capture requires the pendulum to stay in **165–195° for five consecutive 20 ms samples**, before a 20 s deadline. It has no angular-velocity condition. Simulation stops at capture, so this is not evidence of sustained balancing.

Across the 1,024 downward validation starts, median peak absolute arm speed was **11.41 rad/s**, and median absolute arm speed at capture was **0.46 rad/s**. Median capture time was about **11.1 s**. Nonzero final speed is compatible with the requested angle-only goal; the arm is not claimed to have stopped completely.

![Validation outcomes and observed arm slowdown](../../artifacts/rotary_pendulum/experiment-results/single-constant-torque-action-controller/figures/validation.png)

The study contains **23,668 episode evaluations**, including 23,552 randomized evaluations on 6,400 distinct random initial states and 116 fixed-check evaluations. Repeated parameter comparisons are not independent new samples. There were 29 case runs and 7,424 independently integrated 100 ms action checks. The [complete tables](../../artifacts/rotary_pendulum/experiment-results/single-constant-torque-action-controller/records/summary.csv) include unsuccessful and boundary-crossing settings.

## What tuning showed

The first 100 ms-only design captured all 260 development cases, but 82 crossed the arm limit. Raising the boundary penalty from 100 to 1,000,000 barely changed this: it cannot undo outward motion already accumulated before the boundary enters the short prediction.

Reducing the work-request gain and increasing the work-error penalty modestly gave 260/260 captures with zero crossings on the small development set. The larger comparison revealed rare moving-state crossings. Nearby settings changed their frequency without eliminating them consistently; this is a local tuning result, not a claim of optimal gains.

The default now uses work-request gain **0.004**, work-error penalty **0.02**, and arm-limit penalty **10,000**. These dimensionless values govern action selection; they do not rescale the logged physical energies. The arm penalty is zero through exactly ±180° and positive beyond it. The [machine specification](controller_formulation_and_experiment_specification.md) defines the formulas, normalizations and parameter roles.

## Remaining boundary failures

Across the larger comparison and both final validation seeds, the selected setting had seven crossings, all from moving starts. Three first crossings had no tested torque capable of keeping the next 100 ms within ±180°. In four, bounded options existed, but the finite soft penalty allowed a crossing in exchange for the other objective terms. The largest observed arm angle was **189.03°**.

These are two distinct limitations: limited anticipation of accumulated motion, and an intentional finite penalty rather than guaranteed containment. If stronger containment is wanted next, an overshoot penalty with a steeper slope at 180° can be tested without moving the boundary or adding a braking sequence. Its benefit has not been established here. [First-crossing records](../../artifacts/rotary_pendulum/experiment-results/single-constant-torque-action-controller/records/boundary_events.json) preserve the evidence.

## Numerical and reporting checks

A separate run used a **2 ms plant integration step**, keeping controller prediction at 20 ms, decisions at 100 ms, and goal sampling at 20 ms. On the same 2,052 initial states from seed 20261022, all captures remained; all 512 downward starts stayed within the arm limit. Moving-state crossings changed from three to two. Individual paths can change even when these aggregate outcomes agree.

For the selected controller, the largest sampled motor-work difference between 20 ms prediction and independent 2 ms replay was **0.729 microjoules**. The 2 ms replay's worst mechanical-energy/work imbalance was below **3 nanojoules**. All stored capture and crossing flags were independently recomputed. A separate 260-case CPU check of the default command captured every case without an arm-limit crossing. All **118 repository tests passed**, including body-energy verification from center-of-mass motion and proof by replay that predictions contain exactly five constant-torque steps.

Current analytical-controller plots show physical energies in **mJ**, with all components using the same unit conversion. The NumPy simulation's generic energy logs now report both bodies' mechanical energy. Older MPC/RL formulas that deliberately use hinge-relative swing energy are identified as such; that quantity excludes arm-carried motion and is not labeled full pendulum-body energy. Historical experiments retain their original source snapshots and results.

Start with the [representative validation trajectories](../../artifacts/rotary_pendulum/experiment-results/single-constant-torque-action-controller/figures/validation_seed_20261022_trajectories.png). Pendulum position uses 0–360°; arm position and angular velocities remain signed. Lines break at a pendulum wrap. Representative sessions are selected by median minimum energy-distance diagnostic within each initial-state class, not by successful outcome. Exact selections, complete episode tables, source hashes and audit records are under [experiment records](../../artifacts/rotary_pendulum/experiment-results/single-constant-torque-action-controller/records).
