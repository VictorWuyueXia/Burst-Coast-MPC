# Holding-region theorem: interpretation

**Plan-memory switching audit, 2026-10-25:** the [weighted queue study](human-readables/plan_memory_switching.md) proves the retained-shift construction conditionally and tests the missing replacement implication. Among 669 saved replacements, 345 lower the MPC objective while increasing the queue certificate relative to warm; 57 reachable transitions leave its proposed five-degree sublevel. No complete-controller holding region has been proved. Controller logic remains unchanged.

**Actual-network result, 2026-10-25:** the [new study](human-readables/actual_mpc_rl_holding.md) replaces the assumptions. A local optimizer branch is numerically stable, but its proposed five-degree holding ellipsoid is disproved by 73 saved CEM exits. A 14-state retained-plan branch is also analyzed. The complete switching-controller proof remains open; neither earlier illustrative ellipsoid is an established MPC+RL holding region.

**Evidence update, 2026-10-25:** commit `1ebd1fa` now supplies the actual frozen networks and complete confirmation records. CPU decoding and original-reference parity checks passed. The observed final-ten-second upright-error range is approximately -1.948 to +1.937 degrees; the active terminal Hessian at upright is indefinite. The earlier missing-data statements below describe the pre-transfer review. Current audit results are in `artifacts/rotary_pendulum/inner-mpc/mpc-rl-local-analysis/interpretation_summary.md`.

**Construction update, 2026-10-25:** the LQR-reference ellipsoids below are retained as a conditional reference calculation, not the adopted MPC+RL region. Use the [MPC-derived linearization and assumption study](human-readables/mpc_rl_linearization.md) for the current direction. No numerical region has yet been certified for the actual CEM controller.

Date: 2026-10-24. Baseline: commit `cdce8f93f98d6c978e6a723d4aea11f4524c59ad`, unchanged MPC+RL.

**Completed:** a concrete positive-definite quadratic metric and a rigorous 0.970 contraction certificate for a nominal linear reference; a proof that a bounded actual MPC successor deviation gives an invariant ellipsoid, bounded error, and uniform Lyapunov stability of a smaller ellipsoid. The controller is not switched to LQR.

**Not completed:** verification that the actual MPC+RL algorithm always returns an accepted action on the proposed domain and satisfies the required successor-deviation bound. The frozen checkpoint is absent locally. Consequently, the numerical ellipsoids below are explicit candidates with test thresholds, not certified regions for the implemented controller.

The proposed outer ellipsoid has quadratic-metric radius 0.2. A uniform actual-successor deviation at most 0.003 in the same metric would certify a stable inner ellipsoid of radius 0.1. In that inner ellipsoid, rigorous coordinate bounds imply arm error below 3.785 degrees, pendulum upright error below 0.607 degrees, arm speed at most 0.1330138 rad/s, and pendulum speed at most 0.1369292 rad/s, at command instants. These are bounds on an ellipsoid, not independent box limits that can all be saturated simultaneously.

Read the [formulation and proof](human-readables/holding_theorem.md) first, then the [verification specification](machine-scannables/certificate_specification.md). The latter identifies exactly which inequalities connect the theorem to the unchanged controller. Continuous-time near-upright bounds additionally require the inter-command tube check in the theorem.

The reference calculation uses exact rational arithmetic and a rigorous matrix-exponential remainder, not only floating-point eigenvalues. Its reproducer and machine certificate are under `artifacts/rotary_pendulum/inner-mpc/holding-region-certificate-20261024/machine-scannables/`. No GPU computation, model training, controller tuning, or controller-source modification was performed.
