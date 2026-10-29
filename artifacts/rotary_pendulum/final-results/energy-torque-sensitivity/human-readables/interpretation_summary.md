# Actuation opportunities and repeating motion

**The passive pendulum motion is periodic in the tested cases. The proposed sensitivity angle is not a usable phase: the map retraces itself and discards the sign of pendulum displacement.** The original heuristic produces recurring swings with a changing energy envelope, rather than a demonstrated periodic controlled orbit.

## Read the figures

- [Controlled physical states](controlled_physical_sessions.png): angles, velocities, applied torque, total and relative swing energies, and both sensitivities. Columns are the same three cases throughout.
- [Controlled opportunity maps](controlled_energy_sensitivity_planes.png): sensitivity pairs colored by total energy, error-weighted task pairs, and the proposed map angle.
- [Task competition and energy rates](energy_task_preferences_and_rates.png): whether the tasks favor the same torque sign, and how natural exchange compares with torque-driven swing-energy change.
- [Torque-free physical states](coast_physical_sessions.png): twelve seconds of new, explicitly torque-off continuations from recorded states. The arm position drifts even though pendulum motion repeats.
- [Torque-free maps and phase comparison](coast_energy_sensitivity_planes.png): retraced sensitivity curves above closed physical pendulum orbits. In the bottom row, the blue map angle reverses, while the orange signed-state angle advances; angular branch cuts are left as gaps.

All figures also have PDF and SVG versions. These are deterministic examples, not population estimates or uncertainty bands.

## Which records?

The archive contains a 2 ms integration validation of the original 100 ms energy/work heuristic: work gain 0.004, work weight 0.02, and arm penalty weight 10,000. Selection used the recorded capture duration, before inspecting the new maps. The upper median is used for each even-sized 512-episode stratum.

| Case | Archive lane | Selection | Duration | Complete measured swing intervals |
|---|---:|---|---:|---:|
| Typical downward | 302 | Median downward duration | 11.04 s | 13 |
| Slow downward | 325 | Longest downward duration | 16.02 s | 20 |
| Typical moving start | 766 | Median moving-start duration | 8.92 s | 10 |

All three were successful and arm-compliant **under their historical settings**: 15° upright tolerance for 100 ms and ±180° soft arm range. Their 15°/100 ms capture and ±180° arm range differ from the RL/MPC evaluation protocols. These are energy-heuristic records, not neural-actor or augmented-teacher trajectories.

Only active samples were retained, at 20 ms spacing, plus the initial state. Frozen post-capture padding was excluded. Replaying every retained observation interval through the nominal 2 ms model matched within 2.14 × 10⁻¹⁴ across the state coordinates.

## Controlled motion: recurring swings, changing cycles

Successive positive-velocity downward crossings define complete swing intervals. Their durations vary by 13–15% relative to their means, and their return velocities differ. This rejects an interpretation of these entire captures as one stationary periodic orbit, although local repeated motifs are evident.

| Case | Interval range | Interval variation | Longest recorded zero-torque interval | Conflicting task signs |
|---|---:|---:|---:|---:|
| Typical downward | 0.594–1.030 s | 15.0% | 0.20 s | 45.6% |
| Slow downward | 0.575–1.117 s | 14.7% | 0.20 s | 46.9% |
| Typical moving start | 0.687–1.029 s | 13.2% | 0.10 s | 45.4% |

“Variation” is the standard deviation divided by the mean interval. Conflict means the two energy-error-weighted sensitivities have opposite signs. It describes their input-dependent preferences; it does not imply that both complete energy rates cannot improve. The rate-decomposition plot shows why natural exchange must remain in the reasoning.

The recorded coast segments are shorter than a full swing. They cannot establish passive periodicity on their own; the following continuations are a separate numerical experiment.

## Torque-free motion: periodic shape, drifting arm

Three releases use the first recorded states in the slow capture above 20%, 60%, and 95% of upright total energy. No state was adjusted to create a favorable orbit. Torque was then set to zero for twelve seconds in the unchanged lossless plant.

| Source time | Total energy / upright target | Measured pendulum period | Complete intervals | Arm drift per cycle | Map-angle reverse fraction |
|---|---:|---:|---:|---:|---:|
| 0.80 s | 0.205 | 0.55864 s | 20 | +19.98° | 49.2% |
| 2.78 s | 0.600 | 0.69043 s | 16 | −59.28° | 49.6% |
| 14.88 s | 0.950 | 0.97706 s | 11 | +59.76° | 50.0% |

The return periods vary by less than 3 × 10⁻⁸ relative to their means. Normalized velocity-return errors stay below 9.1 × 10⁻⁵; interpolation of sampled crossings contributes to this residual. These are strong numerical evidence of periodic motion in pendulum angle and both angular velocities. The signed arm position does not return. Such motion is a **relative periodic orbit**: its internal shape repeats while the cyclic arm coordinate shifts. It is not an attracting limit cycle, and the unrestricted coast eventually leaves the permitted arm range.

For these lossless tests, total energy and arm angular momentum are conserved. Their relative energy drift is below 1.3 × 10⁻⁸; step halving from 1 ms to 0.5 ms changes angles by at most 1.95 × 10⁻⁶ rad and velocities by at most 1.40 × 10⁻⁵ rad/s. The initial 2 ms coast resolution did not meet the chosen convergence tolerance and was refined; the original records were not altered.

## Why the proposed phase fails

Let alpha be the measured pendulum angle from downward, omega the arm velocity, and nu the pendulum velocity. Total mechanical energy is E; H is relative swing energy, excluding the arm-carried motion. Both targets are the computed upright-at-rest energy, 0.03037176 J. The proposed input sensitivities and error-weighted preferences are

$$
s_E=\omega,\qquad s_H=I_p\nu(M^{-1})_{21},\qquad
c_E=(E-E^\star)s_E,\qquad c_H=(H-H^\star)s_H.
$$

Here I_p is the computed pendulum hinge inertia and M is the model's joint inertia matrix. Sensitivities measure instantaneous power change per unit shaft torque, with units rad/s. Opposite signs of c_E and c_H indicate competing torque preferences. Plotting both c terms divided by the same target energy preserves their signs and relative magnitudes.

The map angle phi is computed from these two sensitivities:

$$
\phi=\operatorname{atan2}(s_H,s_E).
$$

**Reflecting alpha to −alpha while keeping both velocities fixed leaves this pair unchanged.** Each tested passive libration visits these two different pendulum positions with the same velocities. The map therefore retraces an arc instead of uniquely identifying progress around the physical orbit. Even adding the energy level does not fix this symmetry: the energies also agree at the reflected states. The angle reverses for 49–50% of valid passive increments and 35–41% of controlled increments. Changing the near-origin exclusion from 0.1% to 5% of RMS map radius does not rescue it.

A simple comparison uses the signed pendulum state instead. Define Omega_n as the computed small-motion natural frequency, 12.98387 rad/s, and psi as the geometric angle

$$
\psi=\operatorname{atan2}(-\nu/\Omega_n,\alpha).
$$

After unwrapping, psi advances strictly in all three tested passive librations. It is a useful candidate coordinate for these orbits, **not a constant-speed clock or a validated global phase for rotations, separatrices, and balance**. The full derivation and phase diagnostics are in the machine-oriented methods note.

Relative swing energy can exceed total energy in these coordinates because the total kinetic energy contains a signed coupling cross term. That is not an energy-conservation violation. The full-pendulum-body sensitivity shares omega's sign throughout the nominal model; its positive multiplier ranges from 0.1593 to 0.5726.

## Consequence for the next heuristic experiment

Keep the sensitivity and error-weighted terms as instantaneous task-opportunity indicators. For phase, retain signed pendulum position and direction, and condition the orbit on total energy and arm momentum. Momentum is especially important: the almost closed opportunity map hides tens of degrees of arm drift per swing. A natural next comparison is phase-timed torque decisions that also regulate momentum, using matched initial states and the same torque limits. No such controller has been implemented or claimed better here.

The distinction between a repeating projection and an orbit-based phase is consistent with the transverse-coordinate treatment in [Underactuated Robotics: Algorithms for Limit Cycles](https://underactuated.mit.edu/limit_cycles.html). The symmetry and numerical conclusions above are derived from this repository's model and records.

[Detailed methods / 详细方法](../machine-scannables/methods.md)
