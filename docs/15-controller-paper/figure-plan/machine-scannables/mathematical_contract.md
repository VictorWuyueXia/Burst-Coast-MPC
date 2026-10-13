# Mathematical and source contract

## Authority and scope

The current manuscript and current implementation are authoritative. Archived numerical diagnostics support selecting an illustration but require source verification before reuse as a new publication figure. This file specifies deterministic geometry and numerical checks; no plotting implementation is introduced.

## Mechanism geometry

Use a fixed right-handed Cartesian frame. Let the model inputs be arm mass $m_a$, arm length $r$, pendulum mass $m_p$, pendulum length $l$, and gravity magnitude $g$. Let $\theta$ be unwrapped arm angle, $\alpha\in[-\pi,\pi)$ pendulum angle from downward, $\omega=\dot\theta$ arm angular velocity, and $\nu=\dot\alpha$ pendulum angular velocity. Angles are in radians and angular velocities in radians per second.

The radial, tangential, and upward unit vectors are computed from arm angle:

$$
\boldsymbol e_r=(\cos\theta,\sin\theta,0)^{\mathsf T},\qquad
\boldsymbol e_t=(-\sin\theta,\cos\theta,0)^{\mathsf T},\qquad
\boldsymbol e_z=(0,0,1)^{\mathsf T}.
$$

With origin $O$, hinge $P$, pendulum center of mass $Q$, and pendulum tip $T$, and derived center distance $c=l/2$, use

$$
O=0,\qquad P=r\boldsymbol e_r,\qquad
Q=P+c(\sin\alpha\,\boldsymbol e_t-\cos\alpha\,\boldsymbol e_z),\qquad
T=P+l(\sin\alpha\,\boldsymbol e_t-\cos\alpha\,\boldsymbol e_z).
$$

Thus the passive hinge axis is parallel to the arm, and its swing plane is spanned by the tangential and vertical directions. Increasing pendulum angle from downward tilts the rod toward the positive tangent. This convention produces the positive coupling term in the manuscript. Positive shaft torque $u$ acts about the upward axis toward increasing arm angle.

Geometry checks: arm length equals $r$; pendulum length equals $l$; center distance equals $c$; the pendulum direction is perpendicular to the arm; downward and upright conventions match the model. Use the existing `mechanism_points` implementation rather than recreating a separate physical convention.

## Exact physical energy region

The derived inertial quantities are arm inertia $I_a$, pendulum transverse center inertia $I_c$, pendulum hinge inertia $J$, coupling coefficient $b$, and gravitational energy scale $G$:

$$
I_a=\frac{m_ar^2}{3},\qquad I_c=\frac{m_pl^2}{12},\qquad
J=I_c+m_pc^2,\qquad b=m_prc,\qquad G=m_pgc,\qquad E_\star=2G.
$$

Inertias have units kg m²; $G$ and the target energy $E_\star$ have units J. For potential energy $v$, a scalar integration coordinate in joules, define the arm-rotation inertia $A(v)$ and coupling inertia $C(v)$ through the known elevation:

$$
A(v)=I_a+m_pr^2+J\left[1-(1-v/G)^2\right],\qquad
C(v)=b(1-v/G).
$$

For this contract only, let $\eta(v)$ denote the dimensionless lower-bound multiplier. This is derived from the model, not tuned:

$$
\eta(v)=\frac{A(v)-I_a-C(v)^2/J}{I_a}.
$$

The physical energy image $\mathcal E$ consists of triples $z=(z_a,z_p,z_V)^{\mathsf T}$ for arm kinetic, pendulum kinetic, and potential energies:

$$
\mathcal E=\{z:z_a\ge0,\quad 0\le z_V\le E_\star,\quad
z_p\ge\eta(z_V)z_a\}.
$$

The inequality follows from a completed square in pendulum angular velocity. It is sufficient as well as necessary when velocities are unrestricted. Absolute arm position does not enter this energy image.

At update $k$, calculate the physical state $x_k$, encoded energy state $e_k=H(x_k)$, total energy $E_k$, and work request $w_k$ using the existing controller. For the parameterization only, define requested total energy $B=E_k+w_k$. For $B\ge0$, the requested realizable region is exactly

$$
z(a,v)=(a,B-v-a,v)^{\mathsf T},\qquad
0\le v\le\min(B,E_\star),\qquad
0\le a\le\frac{B-v}{1+\eta(v)}.
$$

Here $a$ is the arm-energy coordinate in joules. The two independent coordinates are $a$ and $v$; pendulum energy is computed by the plane equality. There are no fitted geometry parameters. For $B<0$ the region is empty, and for $B=0$ it is the origin. The proposed example has positive $B$.

The boundaries are the zero-arm-energy edge, the coupling-limited edge, the downward edge, and, when $B>E_\star$, the upright-potential edge. When $0<B\le E_\star$, the upper end reduces to one point. Do not assume the region is a triangle, ellipse, or convex patch. Crop the displayed plane for clarity, but label its physical subset independently of that crop.

For boundary verification, reconstruct a physical state. At a given $v$, choose any real angle with $\cos\alpha=1-v/G$, and let

$$
\omega=\pm\sqrt{2z_a/I_a},\qquad
\nu=-\frac{C(v)}J\omega\ \pm\sqrt{\frac2J\left[z_p-\eta(v)z_a\right]}.
$$

Encoding the reconstructed state must recover the original triple within floating-point tolerance. The zero square term gives equality on the coupling-limited edge. The arm angle can be chosen within the envelope for this existence check; this does not establish trajectory containment from the observed state.

## Torque curve and numerical work

The scalar independent variable for Figure 3 is held torque $u\in[-U,U]$, in N m. The physical state $x_k$ and action duration $\Delta$ are fixed. The exact flow $F_\Delta$ integrates the physical differential equations for that duration, and the exact response curve comprises $H(F_\Delta(x_k,u))$.

For exact lossless dynamics, define delivered motor work $W_\Delta$ by

$$
W_\Delta(x_k,u)=u[\theta(F_\Delta(x_k,u))-\theta(x_k)].
$$

Its equality to the request is equivalent to a curve–plane intersection. In the figure calculations, use the numerical flow $\widehat F_\Delta$ and its shaft-displacement work $\widehat W_\Delta$. Record two distinct residuals, both in joules:

$$
r_W(u)=\widehat W_\Delta(x_k,u)-w_k,
$$

$$
r_E(u)=\boldsymbol 1^{\mathsf T}H(\widehat F_\Delta(x_k,u))-E_k-\widehat W_\Delta(x_k,u),
$$

where $\boldsymbol 1=(1,1,1)^{\mathsf T}$ sums energy components. The numerical plane residual equals $r_W+r_E$, not just $r_W$. Numerical roots should therefore be labeled approximate work matches. Do not force endpoint coordinates onto the plane to hide the integration residual.

The implementation uses 33 uniform torque samples and ten bisection iterations for sign-bracketed intervals. It scores all valid grid and refined candidates. The plotting mesh may be denser to depict the response curve but must not silently replace the controller's tested candidate set. Finite sign sampling can miss tangencies or additional roots. Count distinct torque candidates rather than interpreting the logged bracket count as a proof of the exact root count; different torques can also map to one energy point.

The selected torque $u^\star$ minimizes the implemented sum of normalized potential-deficit, work-error, and sampled arm-excursion penalties. The weights are controller inputs, not physical energies. The arm envelope is soft; the curve includes all torques within the torque bound.

## Proposed example and live inspection

- Candidate source: the existing downward median-capture replay, decision index 20, at elapsed time 2.0 s. This is an explanatory choice, not a representative-performance assertion.
- Current model inputs: $m_a=0.095$ kg, $r=0.085$ m, $m_p=0.024$ kg, $l=0.129$ m, and $g=9.81$ m/s²; both modeled damping coefficients are zero.
- Current usable torque bound: $U=0.01836$ N m, equal to 90% of the physical actuator bound. Current action duration: $\Delta=0.1$ s, using five 0.02 s prediction steps. The separate generic simulation time step does not override this decoder contract.
- Work-request gain $\kappa=0.004$ and work bound $W_{\max}=0.04E_\star$ define the clipped proportional request. The work and arm penalty weights are $\lambda_W=0.02$ and $\lambda_\theta=10000$. The arm envelope is $\theta_{\max}=\pi$ rad.
- The archived diagnostic contains a historical positive-angle representation. Its equivalent signed state is approximately $(0.59777457,-1.37206342,0.16749673,-6.99861723)$, with angles in rad and velocities in rad/s. This conversion changes no geometry or energy.
- Direct arithmetic from the archived state and current physical constants gives $E_\star=0.03037176$ J, $e_k/E_\star\approx(0.0001056703,0.1064845299,0.4012863336)$, and $B/E_\star\approx0.5098450277$.
- The recorded request is approximately $5.9786623\times10^{-5}$ J; its normalized plane displacement in total energy is approximately 0.00196849. The target point $(0,0,1)$ is not on this requested plane.
- The archived audit reports two refined torque values, approximately −0.0019212781 and +0.0033265173 N m, and selected torque +0.0057375 N m. These values were read from existing artifacts, not recomputed by running the controller in this planning task.
- Inspection of the existing 257-row endpoint table found two adjacent sign changes of the energy-plane residual and positive physical-compatibility slack throughout, with minimum slack approximately 0.00121177 J. These are table checks, not a fresh predictor validation or proof that no other roots exist.
- The archived audit reports maximum energy-balance error approximately $5.14\times10^{-8}$ J. Recompute this quantity during figure preparation; it is not newly verified by this planning task.

## Source map

| Figure object | Current source | Relevant content |
| --- | --- | --- |
| Mechanical labels and conventions | `latex-project/main.tex`, lines 67–88, 123–167 | Geometry, state, inertias, energy partition |
| Mechanism points | `src/rotary_pendulum/environment/dynamics.py`, lines 159–182 | Radial/tangential geometry and pendulum direction |
| Physical constants | `src/rotary_pendulum/configs/physics.yaml` | Primitive model inputs and physical torque capacity |
| Energy point and request | `src/rotary_pendulum/heuristic/energy.py`, lines 14–41 | Full body-energy encoder and clipped work policy |
| Plane and physical region | `latex-project/main.tex`, lines 223–267 | Request plane, square completion, constructive realization |
| Prediction and curve | `latex-project/main.tex`, lines 269–297 | Held-input flow, work, endpoint curve, approximate candidates |
| Decoder and selected torque | `src/rotary_pendulum/heuristic/decoder.py`, lines 13–100 | Torque bounds, grid, bisection, scoring, selected diagnostics |
| Predictor step | `src/rotary_pendulum/environment/jax_dynamics.py`, lines 17, 50–61 | Fixed 20 ms RK4 prediction and signed wrapping |
| Controller weights | `src/rotary_pendulum/configs/heuristic.json` | Selected request and penalty settings |
| Existing diagnostic | `machine-scannables/figure_audit.json` and `machine-scannables/figure-data/decoder_curve.csv` under the paper folder | Archived state, torque candidates, selected action, endpoint samples |
| Existing preparation | `scripts/prepare_controller_paper.py`, lines 97–186 | Recorded-state extraction and batched diagnostic construction |

Manuscript and archive paths above are relative to the paper folder; `src/` and `scripts/` paths are relative to the repository root. Line numbers refer to the source inspected for this plan and may shift after editing.

## Rendering acceptance checks

1. Recover the current source replay and confirm the state, request, duration, physical constants, and selected torque against the diagnostic. Historical angle wrapping and archived checksums must be handled explicitly rather than silently assumed current.
2. Check plane membership and the exact physical-energy inequality for the constructed region; reconstruct boundary/interior states and re-encode them.
3. Compute shaft-work error and energy-balance error separately; report their tolerances and values in machine-readable data. Reproduce the selected endpoint and decoder choice with current code before using publication annotations.
4. Inspect the full bounded-torque sweep, zero torque, root labels, and selected star. Preserve actual endpoint coordinates and do not smooth away folds or root structure.
5. Render at the actual manuscript column/text widths, keep mathematical text editable, and inspect all labels, overlaps, depth ordering, and matching panel scales. Numerical verification does not substitute for visual inspection.

No code object, helper, runtime parameter, or software dependency is added by this contract. Source verification and numerical preparation remain separate from rendering and manuscript integration.
