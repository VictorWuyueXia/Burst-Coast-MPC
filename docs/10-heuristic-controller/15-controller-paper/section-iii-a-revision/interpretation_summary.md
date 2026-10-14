# Interpreting the energy formulation and feasibility derivation

The revised formulation is in [main.tex](../latex-project/main.tex). Section III.A presents mechanics, energy rates, phase dependence, and finite-duration work. Section III.B introduces the work request and candidate endpoint energy set, then derives physical feasibility before the torque-response curve. Existing equation labels and terminology are retained. The writing follows Nature writing → control academic writing → Nature polishing → academic humanizer for IEEE control readers.

The argument is that underactuation and configuration-dependent coupling make swing-up depend on energy transfer across oscillations. Energy coordinates organize the desired change, while the physical state determines how a bounded torque realizes it. The revision does not claim an exact autonomous reduced-order model, faster optimization, or optimal control.

1. **Why start from energy?** Angular error can increase during a useful swing that accumulates energy. The physical state is sufficient to describe and control this motion, but angular error alone does not express the available energy. The energy representation makes the energy deficit and its distribution explicit. It guides work selection while retaining physical-state prediction for torque selection.

2. **Where does `eq:dynamics` come from?** Apply the forced Euler–Lagrange equation to the kinetic energy minus potential energy. Only the arm receives motor torque. The resulting two equations jointly determine arm and pendulum accelerations. The inertia matrix collects their acceleration coefficients; its off-diagonal entries describe inertial coupling. The revised display uses its inverse explicitly to give the governing first-order state equation. Numerical integration of that equation produces the trajectories used by the controller; differentiating the energy functions along the same equation gives their rates.

3. **What is the energy state?** It is the three-component vector

$$
e=H(x)=(K_a,K_p,V)^{\mathsf T}.
$$

Here the observed physical state is the vector $x$ of arm angle, pendulum angle, and their signed angular velocities. The computed map $H$ returns arm kinetic energy $K_a$, full pendulum-body kinetic energy $K_p$, and gravitational potential energy $V$. Pendulum-body energy includes motion of its hinge, not only rotation relative to the arm. These are physical energies computed from the state and mechanical parameters, not adjustable penalties.

4. **What governs motion in energy space?** Differentiating the energy state gives

$$
\dot e=DH(x)f(x,u).
$$

The dot denotes a time derivative. The vector field $f$ is the physical-state rate computed by the mechanical governing equation, and $u$ is motor torque. The Jacobian $DH$ is the matrix of derivatives of the three energy functions with respect to the four physical-state coordinates. It converts physical-state rates into energy rates. The equation still depends on the physical state: opposite pendulum-angle signs can produce the same energies but different directions of energy transfer.

5. **What happens at rest?** The revised sentence is: “At rest, the instantaneous energy rates are zero, but applied torque can initiate motion and change the energies over time.” Energy initially has zero slope because both angular velocities are zero. Nonzero acceleration can then create velocity, after which motor torque performs work.

6. **What does completing the square mean?** It is an exact algebraic rearrangement: expand $K_p=K-K_a$, then add and subtract $C^2\omega^2/(2J)$ to complete the square in pendulum angular velocity. Substituting $\omega^2=2K_a/I_a$ gives

$$
K_p=\frac{J}{2}\left(\nu+\frac{C}{J}\omega\right)^2
+\frac{A-I_a-C^2/J}{I_a}K_a.
$$

Here $\omega$ and $\nu$ are the signed arm and pendulum angular velocities. The fixed inertias $I_a$ and $J$ refer to arm rotation about the motor shaft and pendulum rotation about its hinge. The computed coefficients $A$ and $C$ are the configuration-dependent inertia associated with arm rotation and its coupling to pendulum rotation, respectively. All are derived from the rod geometry and masses defined in the manuscript. The first term is nonnegative, so the second term is a lower bound on $K_p$ for a prescribed $K_a$. With unrestricted pendulum velocity, the square can attain any nonnegative value, which supplies the sufficiency argument as well as necessity. The manuscript now shows this rearrangement rather than merely naming it.

7. **How are the inertias evaluated from potential energy?** The relation

$$
V=G(1-\cos\alpha)
$$

uses the pendulum angle $\alpha$, measured from downward, and the positive gravitational energy scale $G$ computed from pendulum mass, gravity, and center-of-mass distance. It determines $\cos\alpha=1-V/G$ and hence $\sin^2\alpha=1-(1-V/G)^2$. Substitute these values into the existing definitions of $A$ and $C$; no choice between the two angle signs is needed. “The coefficient” meant precisely the multiplier $(A-I_a-C^2/J)/I_a$ of $K_a$ in the kinetic-energy bound. The revised prose calls it that multiplier and gives the substitutions explicitly.

8. **Why might the requested work be impossible?** Motor work is torque integrated over arm displacement. For a fixed starting state and positive actuation duration $\Delta$, a torque bounded in magnitude by the prescribed limit $U$ cannot deliver an arbitrary energy change. A requested signed work value is attainable only when some constant torque in the interval $[-U,U]$ produces it through the governing dynamics. This is a reachability question, distinct from whether an energy vector is physically possible.

9. **Why check the entire trajectory?** An arm trajectory can cross its position bound and return before the action ends. Checking only its final position would miss that violation. The full state trajectory is therefore needed to establish continuous-time constraint satisfaction. The implemented finite penalty encourages bounded motion but provides no such guarantee.

For notation, retain “physical state” for the four mechanical coordinates, “energy state” for their three-energy image, “work” for signed motor energy transfer, and “state-transition map” for the integrated physical dynamics. The vector field gives an instantaneous rate; the state-transition map gives a state after an action.

“Assigned independently” meant choosing arbitrary numerical values for the three energies. The manuscript replaces that wording with a concrete restriction: moving the arm necessarily moves the pendulum body, so positive arm kinetic energy cannot coexist with zero pendulum kinetic energy. This explains why the candidate energy set needs physical bounds.

The subtraction and square completion are shown directly in the equations: substitute the arm kinetic energy, collect its contribution into the inertia difference, add and subtract the same coupling-dependent term, and form the square. The sufficiency statement now explicitly concerns the existence of a physical state with the prescribed energies when velocities are unrestricted. It does not certify that the state can be reached by a bounded torque over a given duration.

Feasibility appears after the work-budget surface is defined in III.B. Its purpose is to characterize the energy states that can belong to that surface, before the physical prediction selects reachable outcomes. The state-transition and work equations remain in III.A. The clipping interval is described as prescribed rather than guaranteed feasible. The prose uses “control update”; the official conference title in the bibliography remains unchanged.

The complete Method section has 1,531 prose tokens before this relocation and 1,537 afterward under a count that treats each inline mathematical expression as one token. Its thirteen equation groups and all existing labels are preserved. Scope comparison confirms that only the requested relocation, local explanations, and terminology changes were made; prompts.md is unchanged.

The isolated Method section compiles with IEEEtran. Full-project compilation remains unverified because the native compiler cannot load the external ieeeconf.cls. The expanded subtraction and square-completion identities passed 2,048 numerical checks with maximum absolute residual below 1.2e-15. Earlier velocity-reconstruction checks remain recorded in [machine-scannables/verification.json](machine-scannables/verification.json). These are algebra checks, not controller performance tests or reachability guarantees.
