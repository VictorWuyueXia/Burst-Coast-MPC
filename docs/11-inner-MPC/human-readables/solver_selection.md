> Current revision: arm-centering cost and GPU-native SQP/CEM are implemented. Use the [GPU experiment guide](gpu_experiment_guide.md). The initial SciPy/CPU comparison below is historical.

# Choosing the optimization method

## Recommended first experiment

**Use constrained gradient optimization with JAX derivatives, initially through SciPy SLSQP and direct single shooting.** Retain the existing JAX dynamics and frozen Flax actor/critics. Compare a GPU cross-entropy method (CEM) on exactly the same planned controls, objective and feasibility checks. Add sampling-seeded local refinement only if the comparison shows a useful gain.

This recommendation follows from the inspected implementation: four continuous states, one continuous torque per decision, differentiable nominal mechanics, smooth SiLU networks and an explicit hard arm boundary. It is not a claim that SLSQP is fastest or globally optimal. The first experiments must measure that.

Gradient optimization and GPU computing are separate choices. JAX can batch both values and derivatives. Conversely, a sequential constrained solver can spend much of its time on host-side linear algebra even if its model evaluations use a GPU.

## Why derivatives are available

The current `jax_dynamics.rk4_step` uses a fixed ten-iteration integration loop. Fifty microsteps produce one held 100 ms command. Static loops support automatic differentiation; JAX documents the distinction between static and dynamic loops in [`fori_loop`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.fori_loop.html). A fixed-length [`scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html) expresses the prediction horizon.

Use the physical rollout, without success latches, sampling deadlines, environment resets or teacher filters. The model's inertia determinant is positive for the configured mechanics. Actor tanh and hidden SiLU activations are smooth. Pendulum wrapping creates coordinate seams, but the downstream dynamics and network features depend periodically on its angle; test the composed prediction near those seams.

The minimum of two terminal critics has a kink where they cross. There is no need to average critics, select one, add a smoothing temperature or approximate that kink. Introduce one scalar epigraph variable and two inequalities to represent the same terminal objective exactly. The [contract](../machine-scannables/implementation_contract.md) gives the equations. This is an algebraic solver reformulation, not another physical constraint.

The optional RL-cost comparison has additional absolute-value and piecewise terms. Report its nonsmooth points rather than promising the same smooth-solver behavior as the effort-only primary objective. Do not silently smooth its reward.

Frozen networks still need input derivatives. In particular, the terminal actor's derivative contributes through the critic's action input. Stopping that derivative would optimize a different local model. The existing observation routine explicitly casts to float32: simply enabling JAX x64 does not make its outputs double precision. The contract specifies precision-parity checks before using a float64 planning graph.

## Comparison of candidate methods

| Method | Useful property here | Limitation here | Proposed role |
|---|---|---|---|
| JAX derivatives + SLSQP single shooting | Small control vector; direct inequality constraints; existing dependencies | Local solution; dense host solver; sensitivity grows with horizon | First reference at short horizons |
| JAX batched CEM | Reuses exact rollout; parallel candidates explore different torque histories | Narrow feasible regions can exhaust samples; no stationarity certificate | Matched sampling comparison |
| CEM seeds + constrained local refinement | Exploration followed by precise feasible improvement | Added compute and settings; gains must exceed overhead | Conditional extension |
| CasADi + IPOPT multiple shooting | Sparse dynamic constraints and state-node initialization | Neural integration/export work; more decision variables | Escalation if single shooting is the bottleneck |
| Plain Adam, projected gradient or L-BFGS-B | Simple action-bound handling | Clipping torques does not enforce state-path constraints | Not the primary constrained solver |
| iLQR/DDP or a custom augmented-Lagrangian solver | Can exploit trajectory structure | Requires constraint machinery and second-order handling | Defer for this proof of concept |

Single shooting optimizes only the torques and computes all states by rollout. With N planned torques and the epigraph variable, there are N+1 decision variables. At N=20, this is 21 variables, although checking both arm sides at every 2 ms endpoint produces 2,000 arm inequalities. Forward-mode derivatives are a reasonable first choice for this tall constraint Jacobian; benchmark against reverse mode where the outputs are few. JAX describes the two modes in its [autodiff documentation](https://docs.jax.dev/en/latest/jacobian-vector-products.html).

SLSQP handles explicit bounds and nonlinear constraints and accepts supplied derivatives. Its stopping tolerance combines several numerical tests; independently check final feasibility. The SciPy [`SLSQP` reference](https://docs.scipy.org/doc/scipy-1.17.0/reference/optimize.minimize-slsqp.html) also makes clear that its `workers` option concerns numerical differentiation. It does not automatically parallelize independent optimization solves or turn SLSQP into a GPU solver.

## Initial guesses and sampling without hidden control rules

At the first solve, generate the frozen actor's torque sequence by rolling it forward. At later solves, shift the previous plan and append the actor's command at its predicted end. These are initialization choices, not restrictions on the optimized sequence. A shifted plan is not guaranteed feasible after a new measurement.

For the local-solver comparison, also use explicitly recorded positive and negative constant-command initializations, with normalized magnitude 0.5. These investigate the two swing directions and break exact downward symmetry without adding a runtime kick rule. Re-evaluate every seed under the same constraints; do not assume an actor-generated seed is feasible.

CEM samples bounded torque vectors, ranks actually feasible rollouts, and refits its distribution to feasible elites. Retain the best evaluated feasible sequence. An elite mean can be infeasible under nonlinear dynamics, so it must be rolled out and checked before selection. If a generation has too few feasible elites, record that sampling failure; do not turn arm violations into an arbitrary soft score to keep the run alive.

A zero sequence may be included as a declared candidate, but never as an automatic safe fallback. If no accepted solution exists, terminate that controller rollout with an explicit solver-failure record. Numerical search failure is not proof that the mathematical feasible set is empty.

Sampling and local optimization can complement one another; [Bharadhwaj, Xie and Shkurti](https://arxiv.org/abs/2004.08763) study such a combination. Their results motivate a comparison, not a speed or feasibility prediction for this pendulum. Standard MPPI would also require a sampling-temperature and constraint treatment; it offers no demonstrated advantage here that warrants a second sampling implementation initially.

## Where GPU parallelism helps

Parallelize independent trajectories, seed evaluations, critic checks and eventually fixed timing candidates. Keep the time recursion within each trajectory explicit. Batched rollout does not make the physical future independent of the present.

Start CPU JAX + SLSQP as the reference. Compare its measured latency with GPU model evaluations only after compilation is separated and host/device transfers are included. For large sampling batches and critic-return audits, GPU execution is a natural implementation candidate. Prefer one device per independent batch or experiment before introducing cross-device reductions inside a small solve.

Record cold compilation separately from warm solve time, synchronize results before stopping the timer, and include constraint verification in executable-command latency. These precautions follow JAX's [benchmarking guidance](https://docs.jax.dev/en/latest/benchmarking.html). A high rollout throughput does not establish low single-decision latency.

Before every future GPU launch, inspect occupancy outside the sandbox. Use idle devices among 6–7 by default, avoid devices occupied by other users, and stop if more than four GPUs are in use. Independent benchmark jobs may run in parallel on available devices. The implementation smoke tests now include GPU runs; see the experiment guide for their scope.

## Should we still use CasADi?

**Keep it available, but do not make a CasADi rewrite a prerequisite for the first Q-terminal MPC.** The older rotary solver already provides a useful historical reference, but it has a different objective, torque cap and control discretization. Its current NLP supplies torque bounds only; arm behavior is penalized in the objective. It is not the hard-boundary learned-terminal formulation in this package.

Both paths use RK4 mechanics. With the current physics file, the older solver assigns a torque variable per 2 ms step; the learned task holds each torque for fifty such steps. Reusing the old solver unchanged would therefore change the decision space, not merely the numerical backend.

If we use CasADi later, there are two concrete choices:

- Export the fixed Dense/SiLU actor and critics as CasADi expressions, retaining the actor-in-terminal-value derivative, normalization and weights; verify values and derivatives against Flax.
- Supply a properly differentiable external terminal evaluator. An ordinary Python call to a JAX function is not automatically a CasADi symbolic graph. Its callback interface needs explicit derivative support, and any Hessian approximation must be declared.

CasADi supports shooting transcriptions and derivative interfaces; multiple shooting adds state nodes and continuity equations, producing a larger but sparse problem. See its [optimal-control and callback documentation](https://web.casadi.org/docs/). IPOPT is a local nonlinear-programming solver with smoothness assumptions, described in its [documentation](https://coin-or.github.io/Ipopt/). Merely selecting these packages does not make the solve GPU-batched or guarantee global optimality.

The three-layer 512-unit critics make symbolic duplication a material engineering choice. A numerical MLP is already evaluated efficiently by JAX, and it is needed only at the terminal state. Measure the cost of integration and constraint derivatives before blaming the neural terminal evaluation.

Revisit sparse multiple shooting if increasing the horizon produces unstable sensitivities, repeated local-solver failures, or excessive dense-Jacobian cost after reasonable initialization and iteration tuning. It is also a candidate for later CR3BP trajectory problems, but no current CR3BP implementation or timing evidence supports selecting its solver now.

## Decision rule after the experiment

Select the simplest method that gives verified feasible plans, useful closed-loop behavior and acceptable measured latency on the declared test distribution. Compare methods at equal wall-clock budgets and report hardware use, rather than equating one sample with one derivative evaluation.

If all optimizers agree on a poor plan, inspect the objective and critic before increasing solver complexity. If costs decrease while realized return worsens, investigate critic exploitation. If only long horizons fail and derivative conditioning degrades, change transcription. If feasible elites disappear, sampling budget alone may not repair the geometry. These are different experimental outcomes and should remain distinguishable.
