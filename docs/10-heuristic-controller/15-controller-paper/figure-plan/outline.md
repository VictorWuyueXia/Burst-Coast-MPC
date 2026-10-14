# Three-figure plan

This plan connects the mechanism, the energy request, and torque realization in the current manuscript. It specifies figure content only; no figures, plotting scripts, controller changes, or manuscript edits are included. The figures explain the present single-action controller, whose act–coast extension remains future work.

## Shared visual argument

- **Figure 1:** define the physical configuration and the positive directions used by the equations.
- **Figure 2:** show that a work request fixes total energy while mechanical coupling restricts its possible distribution.
- **Figure 3:** show how the observed physical state and a bounded held torque determine endpoint energies and work-matching candidates.
- Use the manuscript's notation throughout. A physical state is the observed vector of arm angle, pendulum angle, arm angular velocity, and pendulum angular velocity. The energy state is computed from that physical state.
- Use white backgrounds, restrained linework, direct labels, and consistent symbols. Use charcoal for the current state, pale gray for the requested plane, muted blue for its physically realizable region, and a dark blue curve. Use orange open circles for work matches and an orange star for the selected action; distinguish them by shape as well as color.
- Figures 2 and 3 use the same energy-axis ordering and viewing direction. Normalize all three axes by the upright-rest energy, with explicit dimensionless axis labels. Figure 3 may use a declared local zoom.
- These are explanatory model figures, not population-performance plots. No error bars or statistical tests are needed. Numerical examples must be identified as selected illustrations and traced to their recorded state.

## Figure 1 — Rotary-pendulum configuration and notation

**Message:** motor torque rotates the horizontal arm; the unactuated pendulum swings in the vertical plane perpendicular to that arm.

- **Composition:** one dominant oblique 3D mechanism schematic, sized for one manuscript column. Use a generic nonsingular pose with both angular arcs visible; describe it as an illustrative configuration rather than a measured pose.
- **Frame:** show the motor-axis origin and a small fixed Cartesian triad, with the vertical axis pointing upward. Use a faint horizontal reference line for zero arm angle.
- **Bodies:** draw the horizontal arm and the full pendulum rod. Label arm length and mass as $r$ and $m_a$, and pendulum length and mass as $l$ and $m_p$. These are configured physical model inputs; use their ratio for the geometry.
- **Joint:** show a small passive hinge at the arm tip, with its axis parallel to the arm. A faint vertical patch through the hinge identifies the pendulum swing plane. It must be perpendicular to the arm, not contain the arm.
- **Arm motion:** label the signed arm angle $\theta$ from the fixed horizontal reference and the angular velocity $\omega=\dot\theta$. Positive motor torque $u$ points toward increasing arm angle. A static drawing displays orientation modulo one turn; its caption must state that the controller retains unwrapped arm angle.
- **Pendulum motion:** label the signed pendulum angle $\alpha$ from the downward vertical and the angular velocity $\nu=\dot\alpha$. Orient the positive arc consistently with the model's positive kinetic coupling term. A short downward reference is enough; do not add a second angle convention.
- **Mass and gravity:** mark the pendulum center of mass with a small point, and label its hinge distance $c=l/2$, derived from the uniform-rod assumption. Show a downward gravity arrow $g$, the configured gravitational acceleration. Keep the rod visible so the center marker is not mistaken for a point-mass pendulum.
- **Reference pose:** show a thin dashed upright direction from the hinge, labeled $\alpha=\pm\pi$; downward is $\alpha=0$. The signed stored angle lies in $[-\pi,\pi)$.
- **Caption allocation:** define all angle, velocity, length, mass, and torque labels; state the passive hinge and uniform-rod assumptions. Keep inertia formulas and the governing differential equations in the manuscript.
- **Placement:** Problem Formulation, immediately after the geometry and angle definitions.
- **Success flag:** the picture reconstructs the same Cartesian geometry and angle signs as the existing model, and every label remains readable at column width.

## Figure 2 — Requested energy plane and physically realizable region

**Message:** the requested work fixes a plane, but only a mechanically compatible subset of that plane represents physical states.

- **Composition:** one 3D energy-space view, initially allocated one column. Let the realizable region occupy most of the useful area; omit a surrounding filled 3D energy volume because it would obscure the plane.
- **Axes:** use $K_a/E_\star$, $K_p/E_\star$, and $V/E_\star$. Here $K_a$ is arm-body kinetic energy, $K_p$ is the full pendulum-body kinetic energy, and $V$ is gravitational potential energy measured from downward. The derived upright-rest energy $E_\star=m_pgl$ supplies a common scale.
- **Current state:** show a filled point $e_k=H(x_k)$, computed by the energy encoder $H$ from the observed physical state $x_k$ at update index $k$. Its total energy is $E_k=K_{a,k}+K_{p,k}+V_k$.
- **Request:** show a pale plane labeled with the following equation. The endpoint coordinates $z_a,z_p,z_V$ denote the same three physical energies, and $w_k$ is the signed work requested by the implemented clipped proportional policy.

$$
z_a+z_p+z_V=E_k+w_k.
$$

- **Realizable subset:** fill and outline $\mathcal S(e_k,w_k)$, the physically realizable part of the requested plane. Leave the rest of the displayed plane pale and label it as excluded by physical energy compatibility.
- **Boundary:** compute the boundary from the manuscript's square-completion inequality, not from nonnegativity alone. The pendulum must carry some kinetic energy when the arm moves; this restriction generally gives a curved edge within the plane. Retain the downward and upright potential bounds. The exact parameterization is in the companion mathematical contract.
- **Goal:** show a separate diamond at $e_\star=(0,0,E_\star)$, the upright-rest energy state. Do not put it on the requested plane unless that plane's total energy equals the target energy.
- **Interpretation:** one short arrow within the plane may indicate “larger potential, less kinetic energy.” Identify it as an energy-distribution direction, not a predicted trajectory.
- **Plane displacement:** explain that the current point has total energy $E_k$ while the request uses $E_k+w_k$. The proposed recorded example gives a small separation; retain the true scale and use a numeric callout or a declared inset if needed, without enlarging the work request.
- **Caption boundary:** physical realizability means that some physical state has these energies, with unrestricted velocities. It does not assert reachability from the current state during the next action or satisfaction of the arm envelope along a trajectory.
- **Placement:** Method, immediately after the physical energy feasibility and realization equations.
- **Success flag:** every point in the shaded region satisfies both the plane equality and the mechanical inequality; the current point and the upright-rest point are placed independently of that region.

## Figure 3 — Torque response and work-matching solutions

**Message:** varying one bounded held torque produces a curve of endpoint energies; its intersections with the requested plane identify work-matching torque candidates.

- **Composition:** a dominant 3D curve–plane panel plus a smaller 2D work-error panel, together spanning the manuscript text width. The second panel provides the unambiguous torque values behind the 3D intersections.
- **Shared example:** use the same recorded physical state and work request as Figure 2. Fix the action duration $\Delta=0.1$ s and usable torque limit $U=0.01836$ N m, both taken from the current controller.
- **Response curve:** overlay the endpoint-energy curve on the same plane and realizable region. Here $F_\Delta(x_k,u)$ is the physical state reached by holding torque $u$ for duration $\Delta$ from $x_k$.

$$
\mathcal C_{x_k}=\{H(F_\Delta(x_k,u)):u\in[-U,U]\}.
$$

- **Torque labels:** mark the endpoints corresponding to $-U$ and $+U$, and the zero-torque endpoint. Add at most one direction arrow labeled “increasing torque.” It is not a time arrow. The zero-torque endpoint is generally different from the current state because unforced motion continues during the action.
- **Solutions:** mark work-matching endpoints with open circles and label their torques $u_1,u_2$ when two distinct matches exist. Each circle denotes an endpoint and its associated torque trial, not a point along one executed trajectory.
- **Supporting panel:** plot signed work error against torque, with a horizontal zero line. The delivered shaft work $W_\Delta$ is computed as held torque times predicted unwrapped arm displacement.

$$
W_\Delta(x_k,u)-w_k=0.
$$

- **Implementation cues:** show the 33 tested grid torques as small subdued points and the refined work matches as open circles. Use the same markers in both panels. Numerical predictions use five 20 ms fourth-order Runge–Kutta steps; their energy balance is approximate.
- **Selected action:** show the implemented choice $u^\star$ with a star in both panels. It minimizes the common score over grid and refined candidates. Its endpoint need not lie on the requested plane because the score also values potential energy and penalizes arm excursion.
- **Proposed recorded example:** the existing diagnostic at 2.0 s has reported work matches near −1.921 and +3.327 mN m, and a logged selected torque of +5.7375 mN m. This example explains both multiple candidate solutions and selection away from them. Recheck the current source record and predictor before rendering numerical annotations.
- **Nonintersection cases:** state in the caption that a request can have no matching bounded held torque. Do not invent an extra curve or modify the recorded request to manufacture zero, one, and two solutions. A tangency or torque-bound endpoint can also supply a solution without a transverse crossing.
- **Arm envelope:** retain all torque-admissible curve segments. The current arm constraint is a soft score term, so deleting violating segments would misrepresent the implementation. The proposed example keeps the explanatory emphasis on work and potential energy.
- **Placement:** Method, after the candidate-ranking equation, so the caption can explain both intersections and the selected action.
- **Success flag:** each marked candidate has a recorded work residual; the selected star reproduces the decoder choice; energy-balance error is checked separately; no arrow suggests a time trajectory along the torque sweep.

## Production milestones and scope budget

- **Milestone 1 — scientific plan:** check notation, mechanism geometry, energy-set boundary, torque interpretation, and source availability against the manuscript and implementation. Success: a complete three-figure element list with explicit evidence boundaries. This milestone is the present deliverable.
- **Milestone 2 — numerical preparation:** recover the recorded state, construct the analytic realizable region, and evaluate independent torque trials in one batch. Success: checked geometry, plane/set residuals, work residuals, and agreement with the logged selected endpoint. No new controller tuning or performance campaign is needed.
- **Milestone 3 — visual review:** render only after the plan is accepted. Success: inspect each panel at its final physical size, with legible symbols, clear occlusion order, consistent energy axes, and no overlap. Target 8–9 pt annotation text; no rendered glyph below 5 pt.
- **Milestone 4 — manuscript integration:** export vector PDF with editable text, add self-contained captions and references, and inspect the complete compiled paper. Success: the full project compiles and all three figures remain legible at their placed sizes. Journal-specific production requirements remain to be checked against the chosen venue.
- **Deliverables later:** readable figures and a short interpretation in the human-facing location; numerical tables, model inputs, state provenance, residuals, and reproduction code in `machine-scannables`. Drawing logic stays outside the writing-focused manuscript source.
- **Executable structure budget now:** zero new classes, data classes, functions, methods, wrappers, helpers, CLI options, runtime parameters, or plotting scripts. The mathematical objects above are figure definitions, not new controller components.
- **Implementation discipline later:** reuse existing geometry, energy encoding, dynamics, and decoder operations; keep a compact continuous workflow with concise comments, batched independent trials, and explicit failures. Do not add fallback behavior, thin wrappers, configuration layers, or runtime tuning controls for figure production. A concrete code budget belongs to the subsequent rendering plan.
