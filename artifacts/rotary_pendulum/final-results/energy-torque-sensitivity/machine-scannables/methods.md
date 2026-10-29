# Energy/torque sensitivity: mathematical methods

Controlled observations and explicit zero-torque continuations use the stored lossless model. No new controller is implied by these diagnostic quantities.

## Model, coordinates, and physical constants

The physical state x consists of arm angle theta, pendulum angle alpha from downward, arm velocity omega, and pendulum velocity nu. Angles are radians and velocities rad/s. Arm angle remains signed and unbounded in the plant; pendulum angle is represented in [−pi, pi). Torque tau is the input shaft torque in N m. The archive normalizes alpha differently; its values are converted without changing the physical configuration.

Define I_p as hinge inertia, I_b as base inertia, I_c as coupling inertia, J_a as arm-only inertia, and g_p as gravitational torque coefficient. All are computed from configured link masses, lengths, and gravitational acceleration:

| Constant | Value | Unit |
|---|---:|---|
| I_p | 0.000133128 | kg m² |
| I_b | 0.000402191666666667 | kg m² |
| I_c | 0.00013158 | kg m² |
| J_a | 0.000228791666666667 | kg m² |
| g_p | 0.01518588 | N m |

Both viscous damping coefficients are zero in the actual configuration. Define A, B, and D as the angle-dependent matrix entry, coupling entry, and determinant, respectively; V is downward-referenced gravitational potential energy:

$$
A=I_b+I_p\sin^2\alpha,\qquad B=I_c\cos\alpha,\qquad
M=\begin{bmatrix}A&B\\B&I_p\end{bmatrix},\qquad D=AI_p-B^2>0,
\qquad V=g_p(1-\cos\alpha).
$$

A and B have inertia units; D has squared-inertia units; V is in joules. Total mechanical energy E and relative hinge-swing energy H are

$$
E=\tfrac12 A\omega^2+B\omega\nu+\tfrac12I_p\nu^2+V,
\qquad H=\tfrac12I_p\nu^2+V.
$$

The upright-at-rest targets E_star and H_star are both 2g_p = 0.03037176 J, a diagnostic target consistent with that equilibrium. H is not the full body's mechanical energy; it need not be less than E because E contains the signed cross term B omega nu.

## Input sensitivities, task preferences, and natural exchange

Define s_E and s_H as partial derivatives of the instantaneous energy rates with respect to torque, at fixed state. Define f_H as the full H rate evaluated at zero torque, computed through the existing state derivative. Then, for the actual lossless model,

$$
s_E=\omega,\qquad s_H=-\frac{I_p B\nu}{D},\qquad
\dot E=s_E\tau,\qquad
f_H=I_p\nu\ddot\alpha\big|_{\tau=0}+g_p\sin\alpha\,\nu,
\qquad \dot H=f_H+s_H\tau.
$$

Dots denote derivatives with respect to physical time. The natural term f_H and the input contribution s_H tau have units W. This decomposition evaluates local derivatives; it does not simulate an actual 1 N m action. With nonzero configured damping, the script includes the corresponding negative quadratic dissipation terms in the total energy drift, but the experiment uses zero damping.

For energy errors e_E = E − E_star and e_H = H − H_star, define c_E = e_E s_E and c_H = e_H s_H. Each c has units J rad/s; multiplying by torque gives the input contribution to the derivative of half the squared energy error. Their product being negative defines task conflict. All tabulated conflict and sign fractions are sample fractions, including the one initial sample; coast time fractions instead weight interval durations. Figures divide c_E and c_H by the same target energy to express their axes in rad/s.

Full pendulum body energy E_p = E − J_a omega²/2 has sensitivity s_p:

$$
s_p=\omega\left(1-\frac{J_aI_p}{D}\right).
$$

The bracket is positive for every alpha in this nominal model, ranging from 0.159292 to 0.572607; this explains why s_p cannot disagree in sign with s_E. `s_body_rad_s` retains this quantity in the data exports.

## Conserved momentum and exact reduction during coast

Let p be the arm's generalized angular momentum, in N m s. Because the Lagrangian does not depend on arm position, its time derivative is torque in this lossless plant:

$$
p=A\omega+B\nu,\qquad \dot p=\tau.
$$

Hence both p and E are constant during torque-off motion. Eliminating arm velocity gives

$$
\omega=\frac{p-B\nu}{A},\qquad
E=\tfrac12\frac{D}{A}\nu^2+U_{\mathrm{eff}}(\alpha;p),\qquad
U_{\mathrm{eff}}(\alpha;p)=V+\frac{p^2}{2A}.
$$

U_eff is a computed effective potential at the fixed measured momentum; D/A is a positive effective inertia. A bounded libration between two regular turning points therefore repeats in the reduced state (alpha, omega, nu), but theta can drift. The three releases are librations around downward. This is a family of conservative orbits, not an attractor. The study does not establish periodicity for every state, particularly equilibria and separatrices.

For a possible future orbit-conditioned phase calculation, let alpha_minus and alpha_plus be consecutive turning angles satisfying E = U_eff. Define v(alpha) as the positive speed magnitude determined by the energy equation, and T as the resulting full period:

$$
v(\alpha)=\sqrt{\frac{2A(\alpha)}{D(\alpha)}[E-U_{\mathrm{eff}}(\alpha;p)]},
\qquad T=2\int_{\alpha_-}^{\alpha_+}\frac{d\xi}{v(\xi)}.
$$

Xi is a dummy integration angle. Accumulated travel time along this orbit, with the sign of nu selecting the branch, can be rescaled by 2pi/T into a constant-speed phase. This quadrature is a proposed next method, not an implemented controller or numerical result in this study.

## Non-injectivity of the proposed map

The map angle phi is atan2(s_H, s_E). Its radius r is the Euclidean norm of that pair. A and D are even functions of alpha, as are B and V. At fixed velocities, reflecting alpha changes neither E, H, p nor either sensitivity. On each tested symmetric libration, both reflected positions occur with the same velocities. Thus even an energy-colored map is non-injective and retraces its curve.

The momentum relation makes the near-line collapse transparent:

$$
s_H=\frac{I_p}{D}(A s_E-p).
$$

At p = 0 the two sensitivities share sign and vanish together at turning points. For small swings A and D are approximately constant, so the entire map approaches a straight line. At nonzero p it remains an alpha-dependent affine relation; the reflection ambiguity persists. The measured smaller/larger singular-value ratios are 0.034, 0.051, and 0.050 for the passive maps.

For comparison, Omega_n is the computed small-amplitude natural frequency and psi is a signed-state geometric angle:

$$
\Omega_n=\sqrt{\frac{I_b g_p}{I_bI_p-I_c^2}}=12.9838739882\ \mathrm{rad/s},
\qquad \psi=\operatorname{atan2}(-\nu/\Omega_n,\alpha).
$$

Psi was unwrapped for the increment test and displayed with branch-cut gaps. No nonpositive increments occur in any of the three passive records. It need not advance at constant rate and is not a global phase definition for rotation or equilibrium. The controlled median-downward record has one nonpositive increment; no claim of universal validity is made.

## Return section and phase diagnostics

Section crossings satisfy alpha = 0 modulo 2pi with positive nu. The script detects sin(alpha) changing from negative to nonnegative, requires cos(alpha) positive, and linearly interpolates sin(alpha), time, and state fields between the two neighboring samples. This avoids counting crossings of the upright branch cut as downward returns.

The serialized section columns are theta, omega, nu, E, p in that order. Periods are successive section-time differences. Period CV uses population standard deviation (ddof=0) divided by mean period. The reduced return error is the largest Euclidean difference in (omega, nu) between consecutive crossings divided by the trajectory RMS speed-vector norm. Arm drift is the mean consecutive theta difference. Returning velocities plus the fixed section angle establish reduced recurrence; theta drift establishes the absence of full-state return.

For the phase test, retain adjacent samples only when both radii exceed a chosen fraction of the trajectory RMS radius. Compute the principal signed angle increment between the sensitivity pairs; discard increments below 10⁻⁸ rad in magnitude. Reverse fraction is the smaller of the positive/negative increment fractions, so it does not depend on arbitrary orientation. Zero indicates one-way progression; approximately one half indicates backtracking. Angle jumps between samples very near the origin are resolution-sensitive, especially in the 20 ms archive. The independent reflection proof and dense 1 ms passive results do not depend on those jumps. Threshold sweeps and retained angular travel are in `metrics.json`.

## Data and provenance

`controlled_typical_downward`, `controlled_slow_downward`, and `controlled_typical_moving` denote lanes 302, 325, and 766. Their retained counts, including the initial state, are 553, 802, and 447. Of 1,000 stored physics slots per lane, respectively 448, 199, and 554 frozen slots are excluded. None of the active samples is dropped from numerical analysis. `coast_low_energy`, `coast_intermediate_energy`, and `coast_near_upright_energy` each have 12,001 samples at 1 ms spacing. Initial source times are 0.80, 2.78, and 14.88 s; states and the full archive SHA-256 appear in `provenance.json`.

Each case has a CSV and NPZ with identical named columns. Suffixes `_rad`, `_rad_s`, `_nm`, `_nms`, `_j`, and `_w` identify radians, radians/second, torque, momentum, energy, and power. The `c_*` columns contain unnormalized error-weighted sensitivities. `phi_rad` is the principal map angle; `psi_rad` is the unwrapped signed-state angle. `time_s` is source-relative time for controlled cases and elapsed time since release for passive cases. Torque at row i>0 is the torque applied over the interval ending at row i; at a switch the displayed rate is therefore the left-limit rate. The initial row uses the first interval's torque. All state-derived sensitivities are independent of this endpoint torque convention.

Physical plots show every active sample. Pendulum-angle/phase branch cuts are broken visually rather than drawing spurious vertical lines. Map angles within 1% of RMS radius are also hidden in the display only. The passive map-angle comparison shows the first three seconds for readability; the other passive panels use all twelve seconds. Dense colored map segments are rasterized inside otherwise editable PDF/SVG figures; no data subsampling occurs.

## Reproduction and figure QA

From the repository root:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/experiments/actuation-opportunities/analyze.py
MPLCONFIGDIR=/tmp/burst-coast-mpl PYTHONPATH=/home/yl948/.codex/skills/nature-figure/scripts .venv/bin/python scripts/experiments/actuation-opportunities/plot.py
.venv/bin/ruff check scripts/experiments/actuation-opportunities
```

The plotting workflow uses the existing Python/Matplotlib repository environment; the figure skill supplies the imported alignment auditor. Its PDF collision auditor additionally uses PyMuPDF 1.26.7, installed only into `/tmp/burst-coast-figure-qa`; this does not change the project's runtime dependencies. `requirements-qa.txt` records that audit dependency. Audit scripts and commands come from the nature-figure skill; JSON reports are in `qa/`.

The figure contract is a quantitative grid testing recurrence versus phase ambiguity. Physical traces establish the motion; maps expose information loss; the state-based comparison tests whether retaining the missing sign helps; rate decomposition prevents an input sensitivity being mistaken for the full derivative. Outputs are readable 15-inch report figures, not journal submission layouts. Static preflight warnings about no TIFF, 300 rather than 600 dpi, and non-journal width are intentional for this screen-oriented report. PNG, PDF, and SVG are delivered. Five final figures require alignment, font-floor, collision, and visual inspection checks.

The relevant external conceptual reference is [Underactuated Robotics: Algorithms for Limit Cycles](https://underactuated.mit.edu/limit_cycles.html), especially orbit-based transverse coordinates. All quantitative results and the reflection/reduction derivations above come from the local physical model and archived records.
