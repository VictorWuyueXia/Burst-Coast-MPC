# Opportunity scheduling, task priorities, and boundary braking

The equal-weight two-energy loss in the previous comparison was an illustrative diagnostic, not an established task objective. Its optimizer should be called “best tested for that loss and horizon.” It does not establish the best swing-up decision. The user's proposed opportunity scheduler is a more appropriate next formulation to investigate: select the desired physical change, exploit favorable configuration, and wait only while boundary recovery remains feasible.

This is a formulation note, not an implementation or a new experiment. No controller, gains, classes, functions, or runtime parameters were added. The existing development stage is retained.

## Why equal weighting is insufficient

Normalization makes energy errors comparable in units; it does not justify valuing them equally. Total and pendulum energies are coupled rather than independent tasks. Energy targets also do not uniquely specify upright rest. The one-step objective has no charge for torque, switching, or loss of future opportunities, so it can recommend strong actuation even when waiting would serve the eventual task better. Its 100 ms horizon adds another preference not determined by the sensitivity map.

The completed comparison already demonstrates this limitation: changing the relative swing weight from 0.25 to 4 reverses the preferred sign at 23 of 160 recorded decision starts. Optimizing a different loss and then judging the original controller by that loss would be circular.

Three alternatives have distinct roles:

- **Pareto screening:** reject actions that worsen every relevant task compared with another action; display unresolved tradeoffs. This avoids arbitrary weighting but cannot choose uniquely among competing actions.
- **Priority and tolerance rules:** put boundary keeping first; ask whether energy is deficient, excessive, or acceptable; allow bounded temporary deterioration of a secondary task. Bands must come from an explicit task tolerance or a stated experimental choice. This is the simplest match to the proposed scheduler.
- **Task-level evaluation:** compare complete closed-loop capture time, success, effort, switching and boundary excursions. Longer prediction or a validated value function can approximate this objective, but does not eliminate the need to define it.

A sensible first hypothesis is to bring total energy into a target band, then keep it in that band while arranging pendulum motion toward capture. It is a hypothesis, not a universal priority law: coupling may require temporary energy departure. Avoid requiring both energy errors to decrease at every instant. Near capture, inspect the actual upright angle and velocities, not energy alone. Energy shaping commonly targets an energy-level orbit rather than the upright equilibrium itself; the distinction is illustrated in [Underactuated Robotics: Lyapunov Analysis, swing-up example](https://underactuated.mit.edu/lyapunov.html).

## Which pendulum energy?

The earlier discussion used two different definitions. They must not share a symbol in the controller specification.

| Energy coordinate | Meaning | Consequence for the map |
|---|---|---|
| Full-body pendulum energy | All kinetic and potential energy of the moving pendulum body, including motion carried by the arm | Its direct input sensitivity has the same sign as total-energy sensitivity for this model; only same-sign quadrants occur away from zeros |
| Relative swing energy | Hinge-relative kinetic energy plus gravitational potential | Its sensitivity can oppose total-energy sensitivity; this is the coordinate used in our plotted maps |

Full-body energy can still describe physical allocation between arm and pendulum. It simply cannot supply the proposed opposite-sign *direct-input* opportunities. Natural redistribution can still increase full-body pendulum energy while total energy falls. The full rates, not just input sensitivities, determine that outcome.

If the intended coordinate is relative swing energy, the quadrant idea is directly applicable. If the intended coordinate is full-body energy, keep that physical interpretation but include natural transfer explicitly or change the second control coordinate. Do not silently substitute the relative definition.

## How to use the quadrants

First decide the desired direction for each active task from its energy error and tolerance. Then inspect the two sensitivities and select a torque sign. The same raw quadrant can be helpful or harmful depending on whether each energy needs to rise or fall. If an energy is already acceptable, preserving its band is more meaningful than forcing its instantaneous error to decrease.

When both tasks agree on a torque sign and have appreciable authority, that sign is an opportunity. When they disagree, coast may allow useful natural exchange or movement toward a better configuration. However, coast is not necessarily progress: at exact downward rest it never changes the configuration, and on some trajectories the desired quadrant is not reachable by coasting. Signed state and natural dynamics are needed to distinguish these situations even if the sensitivity map repeats.

Do not identify negative torque with braking or positive torque with pumping. Mechanical power also depends on arm velocity. Nor should all unfavorable energy quadrants trigger braking: braking is a separate boundary/overspeed decision and can intentionally sacrifice energy progress.

At finite action duration, check the next segment rather than freezing the current sensitivities. The recorded A and E examples showed why the best 100 ms direction can oppose the instantaneous rule. A minimal scheduler can use the map to suggest a sign, then test a few explicit act/coast segments under the existing dynamics; it need not solve a long MPC problem or construct a phase variable.

## When to brake

Brake according to **remaining recoverable travel**, not a fixed fraction of arm angle and not an energy quadrant alone. The needed state includes arm position, velocity, pendulum configuration and pendulum velocity. At the same arm angle, outward motion may require immediate braking, while inward motion may not.

For a proof-of-concept, explicitly define a bounded braking action or backup policy. Predict two continuations from the current state: brake now, and wait until the next allowed decision and then brake. Include the action/observation delay, coupled dynamics and the whole trajectory to the first outward turning point. If immediate braking fits within the boundary with reserve, but waiting then braking does not, reject waiting and begin braking now. Check any proposed pumping burst in the same way before accepting it.

After the first stop, the pendulum can accelerate the arm outward again. Continue checking the backup trajectory, or require a terminal state from which a specified keeping policy remains feasible. A first-stop check alone is a useful diagnostic, not a perpetual boundary guarantee. If the chosen braking policy fails even when started now, report loss of recoverability under that policy; do not label the state globally unrecoverable without checking other controls.

Opposing outward arm velocity with maximum torque gives the strongest instantaneous motor contribution to deceleration because the arm input gain is positive. Natural coupling may nevertheless dominate, and this choice need not minimize the eventual peak angle over all possible torque sequences. A tested backup establishes a sufficient recovery option, not the exact viability boundary.

## Recommended decision order

1. Exclude actions and waits that consume the recovery reserve under the explicit boundary predictor.
2. Determine desired energy directions and which task currently has priority; preserve already acceptable energy bands where feasible.
3. Use the opportunity map to suggest torque sign, then check the natural and input-driven changes over the intended segment.
4. Coast when it is feasible and either natural evolution helps or a useful opportunity is predicted before the braking deadline.
5. If neither acting nor waiting achieves the requested simultaneous progress, state the conflict and invoke the declared priority. Do not hide it in an unexplained weight or assume a favorable quadrant must arrive.

This changes the scientific question from “which arbitrary scalar score is smallest?” to “does scheduling admissible energy changes by configuration improve the actual task?” Capture, effort, boundary behavior and stagnation must answer that in a later matched experiment.

## Mathematical reference

All formulas here are derived from the repository's nominal lossless rotary model. State x consists of arm angle theta, pendulum angle alpha from downward, and their velocities omega and nu; angles use rad and speeds rad/s. Torque tau is measured in N m. I_p is hinge inertia, I_b base inertia, I_c coupling inertia, and J_a arm-only inertia, computed from configured masses and geometry (kg m²). The computed gravitational torque coefficient g_p has units N m. Define inertia entries A and B and determinant D:

$$
A=I_b+I_p\sin^2\alpha,\quad B=I_c\cos\alpha,\quad D=AI_p-B^2>0.
$$

E is total mechanical energy, E_p full pendulum-body energy, and H relative swing energy, all in J:

$$
E=\tfrac12 A\omega^2+B\omega\nu+\tfrac12I_p\nu^2+g_p(1-\cos\alpha),
\quad E_p=E-\tfrac12J_a\omega^2,
\quad H=\tfrac12I_p\nu^2+g_p(1-\cos\alpha).
$$

The sensitivities s_E, s_p and s_H are the partial derivatives of their instantaneous rates with respect to torque, evaluated at fixed state (rad/s):

$$
s_E=\omega,\qquad s_p=\omega\left(1-\frac{J_aI_p}{D}\right),\qquad
s_H=-\frac{I_p B\nu}{D}.
$$

The bracket in s_p ranges from 0.159292 to 0.572607 for the configured model. Thus the total/full-body sensitivity pair cannot occupy opposite-sign quadrants. This does not constrain the signs of their complete rates in the same way.

For either chosen second energy z (explicitly E_p or H), let z_star be its desired value and E_star the desired total energy, computed as 2g_p for upright rest. Errors e_E=E−E_star and e_z=z−z_star specify desired correction. Define c_E=e_E s_E and c_z=e_z s_z. Their products with torque are the input contributions to the derivatives of half-squared errors. A torque sign is simultaneously helpful through its direct input contribution if

$$
c_E\tau<0,\qquad c_z\tau<0.
$$

Use non-strict inequalities to allow zero authority or holding a task unchanged. These conditions say nothing about the torque-independent terms. Define f_E and f_z as the energy rates computed at zero torque; f_E=0 in the lossless model, whereas f_z generally need not vanish. The full local descent conditions outside acceptable bands are

$$
e_E(f_E+s_E\tau)\leq0,\qquad e_z(f_z+s_z\tau)\leq0,\qquad
-\tau_{\max}\leq\tau\leq\tau_{\max}.
$$

Tau_max is the configured actuator limit for the experiment. Each inequality restricts a scalar torque interval. Their intersection can be empty, can contain coast, or can permit a range of actions. This gives a simple feasibility calculation before choosing a priority or action magnitude, without inventing an equal-weight objective. Enforcing it at every instant is only one candidate rule and may block useful temporary sacrifices; finite-segment tests and task priorities remain necessary.

For boundary intuition only, consider one side sigma=+1 or −1 and configured arm limit theta_max (rad). Define remaining distance d_sigma=theta_max−sigma theta, outward speed v_sigma=max(sigma omega,0), reserve delta (rad), and a_br>0 (rad/s²). If a_br is a valid lower bound on outward deceleration throughout braking, the ideal no-delay constant-bound stopping requirement is

$$
d_\sigma\geq\frac{v_\sigma^2}{2a_{\mathrm{br}}}+\delta.
$$

Delta is an explicit numerical/model-error reserve, not chosen or tuned in this note. A_br must be justified along the trajectory; torque limit divided by a nominal inertia alone is not such a guarantee for this coupled plant. Delay and changing coupling should instead be included in the predictive test proposed above. For fixed sigma and maximum opposing torque, the instantaneous outward acceleration is

$$
\sigma\dot\omega=\sigma f_\omega(x)-\frac{I_p}{D}\tau_{\max},
$$

where f_omega is zero-torque arm acceleration computed from the model. Its first term can offset the nominal braking contribution. Also, in this lossless model the arm generalized momentum p=A omega+B nu satisfies p_dot=tau. Coast preserves p and may permit substantial arm drift even when the pendulum cycle repeats.

## Validation milestones for a future experiment

This note does not implement a scheduler. A later experiment should first fix the pendulum-energy definition, task bands, priority and backup policy. Verify the scalar interval algebra and nominal prediction against direct simulation. Then measure latest safe waiting times using finer integration, including post-stop reacceleration. Finally compare the original heuristic with the explicitly specified scheduler on matched starts, logging capture, boundary crossings, effort, coast time and stagnation. Failure to reach useful quadrants or to retain recovery reserve should be an experimental finding, not repaired by undisclosed logic.
