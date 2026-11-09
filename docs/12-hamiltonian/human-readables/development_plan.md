# Development plan for the Hamiltonian controller

Add the orbital phase guided energy burst controller as a separately selectable experiment on `Hamiltonian`. The controller and the existing policies will issue torque commands to the same nonlinear rotary-pendulum dynamics. The official MPC+RL implementation, the existing energy-work heuristic, and the production CLI stay on their current entry points. The new experiment runner selects the policy for a trial. [Handover](../../12-Phase_energy.md) · [Official MPC+RL baseline](../../11-inner-MPC/human-readables/mpc_rl_baseline.md)

## Simulator and control loop

Use the shared JAX dynamics and use the NumPy implementation as a float64 reference. The measured state is the unwrapped arm angle θ, the pendulum angle α wrapped to [−π, π), the arm speed ω, and the pendulum speed ν. Angles are in radians and speeds in radians per second. Zero pendulum angle is down. Both representations of π describe upright. The shared environment normally advances in 20 ms blocks with 2 ms RK4 substeps. The matched experiment uses the official 4.59 mN·m command cap, ±135° arm bound, 1 ms prediction samples and 0.5 ms execution replay. The nominal configured plant has zero damping. A separate nonzero-damping sensitivity run tests drift of the conservative coordinates. [JAX dynamics](../../../src/rotary_pendulum/environment/jax_dynamics.py) · [Physical configuration](../../../src/rotary_pendulum/configs/physics.yaml)

Let H be total mechanical energy in joules, computed from the measured state, and let G be the gravitational torque coefficient from the simulator model. The upright-rest energy target is H★. At a permitted pulse opportunity, the signed request ΔHᵍ is a designer-chosen fraction `k_energy` of the current energy error.

$$
H^\star=2G,\qquad \Delta H_g=k_{\mathrm{energy}}(H^\star-H),\qquad 0<k_{\mathrm{energy}}<1.
$$

The sign allows the pulse to add or remove energy. Compute the arm's cyclic momentum and the regular conservative orbit from each measured state. The phase advances uniformly along that ideal orbit. Use it to identify windows around π/2 and 3π/2, and measure how close the windows come to the actual maxima of absolute arm speed. Motor power is torque times arm speed, so the maxima can vary with the orbit. Recompute phase from the state after torque or damping changes the orbit. [Routh reduction for the Furuta pendulum](https://doi.org/10.1007/s00707-022-03146-1)

At each 20 ms supervisory update, roll out passive motion and check predicted arm extrema over the half-period and full-period intervals. Screen finite constant-torque pulses followed by coast, and select an improving pulse from those that meet the arm bound with margin. Use a bounded prediction clock and give braking priority when coast threatens the arm within the half-period. Brake commands and local feedback undergo their own rollout checks. Log and terminate an episode with an explicit status if the prescribed candidates cannot meet the screen. Log numerical failures separately and surface them as errors.

Use a screened one-tick launch at exact downward rest, where orbital phase is undefined. Other physically excluded states follow the declared coast, brake, and capture rules. In particular, the first design can stall near the separatrix without entering the capture region. Report that outcome as a limitation of the formulation. After full-state capture, use the existing discrete linear-quadratic regulator (LQR) design with its 100 ms command period, verified on the current simulator. Apply a wider release region for hysteresis. The 20 ms supervisor can interrupt a local command for braking. [Existing LQR design](../../11-inner-MPC/human-readables/lqr_handoff.md)

## Implementation milestones

| Milestone | Work | Validation and success flag |
| --- | --- | --- |
| Plant and orbital coordinates | Reuse physical constants and energy encoding. Compute cyclic momentum, classify connected regular orbits, and integrate elapsed orbital time. | `phase_verified` when energy and momentum balances, state reconstruction, cycle times, signed rotation and turning-point continuity pass. Check critical levels explicitly. |
| Pulse and coast | Batch bounded torque-duration candidates and use a per-window latch. Execute the selected torque against the same nonlinear plant. | `pulse_verified` when regular-orbit cases show signed energy changes in both directions, one launch per window, no scheduled energy pulse beyond its duration or window, and stable outcomes under refinement. |
| Braking and capture | Add predictive arm screening, brake latching, the screened launch, and the LQR capture handoff. | `supervisor_verified` when replay detects an interior arm excursion missed by endpoint checks, a pulse that should be rejected, a successful brake interrupt, an explicitly infeasible candidate set, and a correct capture/release sequence. Test near-separatrix and exact-rest starts independently. |
| Matched evaluation | Tune on a development set and freeze the design. Run 40 s paired sessions and independent finer-step replays. | `comparison_complete` when all starts and failure outcomes are accounted for, including cases that terminate early. Decide whether gating helps by comparing capture, feasibility, arm excursions, powered time, and work with the ungated variant. |

Use the frozen MPC+RL starts and append named stress cases for exact rest, both rotation directions, off-center librations, near-separatrix motion, and outward motion near either arm bound. Pair the starts and torque caps across policies. Preserve MPC+RL's 100 ms action cadence. The ungated pulse ablation uses the new controller's own cadence, candidate grid, and supervisor so the comparison isolates the timing rule. Continue through capture to measure retention. Report historical native-cap heuristic results separately from the new matched-cap replay.

Record first full-state capture and time held, peak arm angle and crossing events, infeasible decisions and terminal status, requested and delivered energy per pulse, signed and absolute motor work, powered duration, switch-ons, brake entries, and decision time. Report swing-up and local stabilization work separately. Distinguish phase gating of actuation from controller sleep. This experiment still checks the state at each 20 ms update.

## Implementation discipline

Keep the implementation in compact, task-oriented files with a continuous line of logic. Use concise, professional one-line comments for key blocks. Keep files near 50–250 lines and functions near 20–100 lines through grouping by responsibility. Batch candidates and starts with array operations. Use the minimum set of experiment parameters and fixed runtime values. Avoid thin wrappers, fallback values or behavior, and `try`/`with`/`except` control paths. Use the named structure budget in the contract. Keep concise reports and figures under the run's `human-readables/` and traces, settings, and source provenance under `machine-scannables/`. [Implementation contract](../machine-scannables/controller_contract.md)
