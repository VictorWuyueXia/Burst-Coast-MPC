# Analytical contract and validation record

> Implementation update (2026-10-07): the derivation and earlier study record below are retained. Earlier arm-veto and unchanged-runtime statements are superseded: shared damping is now zero, JAX capture omits arm centering, and ±90° receives a soft penalty. See the [implementation contract](implementation_contract.md) and [current formulation](../formulation_plan.md).

This file preserves the general damped derivation and original damped measurements. The current study uses zero damping, removes arm centering, and applies zero torque on rejection; see the [lossless contract](lossless_contract.md). The older decoder branch-selection rules below are historical, superseded by that revision.

## Scope and notation

All identities refer to `src/rotary_pendulum/environment/dynamics.py` and its JAX equivalent, evaluated with the current nominal `physics.yaml`. The mechanics assume the repository's uniform slender links and viscous hinge damping. Angles are in radians and radians are dimensionless in power calculations. No claims about unmodeled rotor inertia, drive electronics, compliance or noise are implied.

Independent coordinates q = (θ, α) are arm azimuth and pendulum angle from downward. Their measured/simulated rates v = (ω, ν) satisfy ω = dθ/dt and ν = dα/dt. The physical state x concatenates q and v; t is time in seconds. Input u is shaft torque in N m. Dots denote time derivatives.

Primitive configured constants are arm mass mₐ = 0.095 kg, arm length r = 0.085 m, pendulum mass mₚ = 0.024 kg, pendulum length l = 0.129 m, gravity g = 9.81 m/s², rotary damping bᵣ = 0.001 N m s and pendulum damping bₚ = 0.00005 N m s. The following are computed, not tuned:

$$
l_c=l/2,\quad J_a=m_a r^2/3,\quad J_c=m_p l^2/12,
\quad J=J_c+m_p l_c^2,
$$
$$
J_0=J_a+m_p r^2,\qquad B=m_p r l_c,\qquad G=m_pgl_c.
$$

Here l꜀ is COM distance, J꜀ is pendulum COM inertia, J is pendulum hinge inertia, J₀ is base inertia, B is coupling inertia and G is the gravity coefficient. All inertias are in kg m²; G is in joules, equivalently N m for angular differentiation.

## Lagrangian and existing ODE

Define configuration-dependent inertia entries A and C, their determinant Dₘ, and the 2 × 2 mass matrix M:

$$
A=J_0+J\sin^2\alpha,\qquad C=B\cos\alpha,\qquad
D_M=AJ-C^2,\qquad M=\begin{pmatrix}A&C\\C&J\end{pmatrix}.
$$

Dₘ is positive for the nominal physical constants. Kinetic energy T, potential V, Lagrangian L, and Rayleigh dissipation function R are defined by

$$
T=\tfrac12A\omega^2+C\omega\nu+\tfrac12J\nu^2,
\quad V=G(1-\cos\alpha),\quad L=T-V,
\quad R=\tfrac12b_r\omega^2+\tfrac12b_p\nu^2.
$$

T, V and L have units J; R has units W. The generalized applied force vector Q = (u, 0)ᵀ acts only on the arm. The forced Euler–Lagrange equation is

$$
\frac{d}{dt}\frac{\partial L}{\partial v}-\frac{\partial L}{\partial q}
=Q-\frac{\partial R}{\partial v}.
$$

Expanding gives exactly the implemented two acceleration equations:

$$
A\dot\omega+C\dot\nu
=u-b_r\omega-2J\sin\alpha\cos\alpha\,\omega\nu+B\sin\alpha\,\nu^2,
$$
$$
C\dot\omega+J\dot\nu
=-b_p\nu+J\sin\alpha\cos\alpha\,\omega^2-G\sin\alpha.
$$

Using L itself as a state derivative would be dimensionally incorrect. L generates these equations in configuration and velocity; changing to energy outputs requires differentiating those outputs along this ODE. It does not make them independent generalized coordinates.

An equivalent optional energy-based formalism retains generalized momentum p = Mv and uses the Hamiltonian H, with ∇ denoting a vector gradient:

$$
H(q,p)=\tfrac12p^TM(q)^{-1}p+V(q),\quad
\dot q=\nabla_pH,\quad
\dot p=-\nabla_qH+Q-\operatorname{diag}(b_r,b_p)\nabla_pH.
$$

p has units kg m²/s; H equals total mechanical energy. This representation retains four independent state coordinates. It is not needed to rewrite the current code.

## Physical-body partition and positivity

Define arm kinetic energy Kₐ, pendulum full kinetic energy Kₚ and carried inertia Aₚ:

$$
K_a=\tfrac12J_a\omega^2,\qquad A_p=A-J_a=m_pr^2+J\sin^2\alpha,
$$
$$
K_p=\tfrac12A_p\omega^2+C\omega\nu+\tfrac12J\nu^2,
\qquad H=K_a+K_p+V.
$$

The determinant of Kₚ's quadratic-form matrix is bounded below by a positive physical quantity:

$$
A_pJ-C^2=m_pr^2J-B^2+(J^2+B^2)\sin^2\alpha
\ge m_pr^2J_c>0.
$$

Thus Kₚ ≥ 0, and Kₚ = 0 only when both velocities vanish. The cross term can have either sign but is not a separate independently stored energy. The energy target e★ = (0, 0, 2G)ᵀ, where e = (Kₐ, Kₚ, V)ᵀ, leaves arm angle free. This partition differs from the old pendulum-relative diagnostic Eᵣ = ½Jν² + V.

## Signed internal power

Define input power Pᵢ = uω, arm dissipation Dₐ = bᵣω², pendulum dissipation Dₚ = bₚν², and signed arm-to-pendulum power P. All are computed in watts, and only the dissipation terms are necessarily nonnegative:

$$
P=(u-b_r\omega-J_a\dot\omega)\omega,
\quad \dot K_a=P_i-D_a-P,
\quad \dot K_p=P-D_p-\dot V,
\quad \dot V=G\sin\alpha\,\nu.
$$

The remaining shaft torque after arm inertia and rotary damping is the torque exerted on the pendulum through its moving support. At ω = 0 this support transfers no instantaneous power, although acceleration and subsequent finite-time transfer can be nonzero. Pendulum damping is relative-hinge loss under this model, so it appears in the pendulum pool.

Direct differentiation provides an independent check of the second pool:

$$
\dot K_p=A_p\omega\dot\omega+J\sin\alpha\cos\alpha\,\nu\omega^2
+C(\dot\omega\nu+\omega\dot\nu)-B\sin\alpha\,\omega\nu^2+J\nu\dot\nu.
$$

Summing the three balances eliminates internal exchange:

$$
\dot H=u\omega-b_r\omega^2-b_p\nu^2.
$$

For a step from time t₀ to t₁, define Δ as t₁−t₀, ΔKₐ/ΔKₚ/ΔV as endpoint differences, W as signed input work, Wₚ as integrated transfer, and Lₐ/Lₚ as integrated damping losses, all in J:

$$
W=\int_{t_0}^{t_1}P_i\,dt,\quad W_p=\int_{t_0}^{t_1}P\,dt,
\quad L_a=\int_{t_0}^{t_1}D_a\,dt,\quad L_p=\int_{t_0}^{t_1}D_p\,dt,
$$
$$
\Delta K_a=W-L_a-W_p,\qquad
\Delta K_p=W_p-L_p-\Delta V,\qquad
\Delta H=W-L_a-L_p.
$$

These are the analytical distribution law integrated along the physical trajectory. Input work alone does not fix Wₚ or ΔV. Wₚ may be negative or greater than W because initial stored energy can be released. A distribution constrained to nonnegative fractions summing to one cannot represent these balances.

The old relative energy's exact derivative is also useful to distinguish existing control from this proposal:

$$
\dot E_r=-C\nu\dot\omega+J\sin\alpha\cos\alpha\,\omega^2\nu-b_p\nu^2.
$$

Its instantaneous torque sensitivity is −JCν/Dₘ. This is the quantity already used by `heuristic_torque`; the current heuristic does account for coupling. At ν = 0 or cos α = 0 that first-order sensitivity vanishes, without ruling out finite-time control authority.

## Closure and inversion failures

An exact energy-only ODE would require one function fₑ with de/dt = fₑ(e,u). The recorded pair x₊ = (0,0.7,1,2) and x₋ = (0,−0.7,1,2), both at u = 0.002 N m, have the same e but unequal de/dt. Therefore such an fₑ does not exist globally. Even replacing the physical-body split with relative-coordinate energies would retain angle/sign ambiguity. All energy encoders also omit θ in this rotationally symmetric model, whereas the task's arm centering/bounds depend on θ.

For constant u over Δ, exact shaft work obeys

$$
W=u[\theta(t_0+\Delta)-\theta(t_0)].
$$

The startup symmetry (x,u) → (−x,−u) gives equal work with opposite motions from x = 0. Locally, define A₀ = A evaluated at downward rest and D₀ = A₀J−B². As Δ tends to zero, the leading work is

$$
W=\tfrac12(J/D_0)u^2\Delta^2+O(\Delta^3).
$$

O(Δ³) denotes higher-order terms bounded by a constant times Δ³ near zero duration. The leading work is quadratic in torque and independent of its sign. The linear power inversion u = requested power / ω is undefined at rest. A work interface needs a directional branch, a waveform/duration convention and a reachable-work calculation, with no invented division floor that changes its physical meaning.

For the same initial aliased pair, a zero-torque rollout delivers exactly zero work in both cases but produces different three-energy vectors after 100 ms. `coast_alias.csv` records both. This directly tests the proposed work action when zero requested work is defined as coasting. The maximum endpoint energy-component difference exceeds 0.001 J. This is a counterexample to deterministic closure, not a general numerical bound on approximation error.

## Confirmed three-state/work-action approximation

The user requires exactly e = (Kₐ,Kₚ,V) as high-level state and requested work w as action, and has confirmed that the separate decoder reads measured angles and signed velocities. Its input is the full physical state x and requested work w; the high-level policy receives only e as its state. Work means signed mechanical shaft work per fixed 100 ms decision in this proposed contract; electrical energy is not modeled. Zero work commands exact zero torque. The policy torque cap remains 0.00918 N m within the physical 0.0204 N m limit. The original proposal logged an infeasible inverse explicitly; the current revision additionally applies zero torque on rejection, records actual zero work and re-encodes the endpoint, as specified in the lossless contract.

Full-state access permits deterministic branch selection by the decoder once its rule is fixed; it does not imply identical encoded endpoints for all physical states sharing e and w. Validate work realization and conditional energy spread separately. Achieved-work/feasibility logs are diagnostics, not extra high-level state inputs.

Define E(x) as the energy encoder, FΔ(x,u) as a physical rollout of duration Δ = 0.1 s at constant torque, and D(x,w) as the proposed phase-aware work realization controller. D is selected among bounded solutions of

$$
|u[F_\Delta(x,u)_\theta-\theta]-w|\le\epsilon_W,
\qquad \epsilon_W=10^{-4}E_\star,\quad E_\star=2G.
$$

εW is a proposed numerical tolerance in J, not a plant parameter. Among feasible roots, minimize predicted peak absolute arm angle, then absolute torque, then use positive-first sign tie-breaking. This does not guarantee arm-limit compliance. The inverse solver must validate returned work and explicitly record unresolved/unreachable requests. Keeping the decoder fixed is necessary for reproducible energy-transition data.

Let c = 1−V/G be the cosine implied by potential. For an admissible energy vector, representative pendulum angles lie at α = ±arccos(c) modulo 2π. Arm speeds have branches ω = ±sqrt(2Kₐ/Jₐ). Given α and ω, pendulum velocity branches follow the quadratic kinetic-energy constraint:

$$
\nu=\frac{-C\omega\ \pm\ \sqrt{C^2\omega^2+2JK_p-JA_p\omega^2}}{J}.
$$

Here C = Bc and Aₚ = mₚr² + J(1−c²) are computed from V. A branch is real only for nonnegative discriminant. At zero discriminant or zero speeds/angles, duplicate branches must be merged. The admissible set obeys

$$
K_a\ge0,\quad 0\le V\le2G,\quad
K_p\ge\frac{A_p-C^2/J}{J_a}K_a.
$$

Thus arbitrary triples of positive energies are not all realizable. The encoder image is constrained, and averages of valid triples need not be valid because this set depends nonlinearly on V. Arm position θ is free in this energy encoding and must be sampled from a declared interval or empirical distribution; it influences the decoder and task bounds even though it does not enter the nominal ODE coefficients.

For fixed e and w, enumerate physical branches and arm positions, apply the same D, and propagate the existing ODE. The resulting energy endpoints define a set ΨD(e,w). A declared probability measure μD(x|e) over hidden states induces an empirical kernel pD(e′|e,w). Neither the branch distribution nor its stationarity follows from the Lagrangian. Policies and histories can change it; a multi-step Markov rollout using this kernel is an additional approximation requiring validation. Report the fraction of infeasible hidden states alongside the conditional distribution of feasible endpoints.

A future energy-only NN should take (e,w), predict a conditional energy distribution and feasibility, and be trained/validated under the frozen decoder and declared sampling measure. It cannot select the actual hidden phase from these inputs. A deterministic point estimate is justified only after small conditional spread has been measured in the relevant domain. Conservation and admissibility are constraints, not substitutes for phase information.

The initial pilot budget and unvalidated screening thresholds are specified in the parent formulation. No decoder, kernel fit or NN is implemented in this delivery; mathematical counterexamples and the nominal power identities are independently executable now.

## Reproducible checks

Run from repository root:

```bash
PYTHONPATH=src .venv/bin/python docs/12-energy-transfer/machine-scannables/validate_energy.py
.venv/bin/ruff check docs/12-energy-transfer/machine-scannables/validate_energy.py
```

The existing environment already supplies NumPy, CasADi and OmegaConf. The diagnostic uses CPU float64, no training, no GPU, no additional dependencies, and no changes to runtime physics. It maps CasADi differentiation over 512 states and vectorizes physical computations; time integration alone remains sequential.

Seed 20261007 selects independent uniform states over θ ∈ [−1.5,1.5] rad, α ∈ [−π,π] rad, ω ∈ [−3,3] rad/s and ν ∈ [−8,8] rad/s, with u ∈ [−0.00918,0.00918] N m. These are diagnostic ranges, not new reset or safety contracts. The directional finite difference displaces x along its ODE for ±10⁻⁶ s. Lagrangian derivatives are generated independently with CasADi; the implemented ODE remains the comparison reference.

| Check | Observed maximum error | Proposed pass bound |
| --- | ---: | ---: |
| Euler–Lagrange versus ODE acceleration | 4.2633 × 10⁻¹⁴ rad/s² | 10⁻¹⁰ rad/s² |
| Sum of body energies versus existing total | 6.9389 × 10⁻¹⁸ J | 10⁻¹⁴ J |
| Total power balance | 3.1225 × 10⁻¹⁷ W | 10⁻¹² W |
| Pendulum-pool power balance | 2.4720 × 10⁻¹⁷ W | 10⁻¹² W |
| Analytic versus directional energy derivatives | 5.5989 × 10⁻¹² W | 10⁻⁷ W |
| Startup integrated balance, 1 ms RK4 | 7.5091 × 10⁻¹⁵ J | 10⁻¹⁰ J |

Startup integration holds ±0.003 N m for 0.1 s and integrates work/dissipation at the same four RK4 stages as state. Its absolute balance residual decreases from 5.6742 × 10⁻¹⁰ J at 20 ms to 1.1684 × 10⁻¹³ J at 2 ms and 7.5091 × 10⁻¹⁵ J at 1 ms. This validates this small probe, not global 20 ms accuracy at high velocities.

## Machine files and field contract

| File | Contents |
| --- | --- |
| `validate_energy.py` | Two-function reproduction workflow; progress and assertions fail visibly |
| `identity_checks.csv` | Metric name, maximum absolute error, unit, strict upper pass bound |
| `phase_alias.csv` | Two equal-energy/different-phase states, common torque, three energies/rates and four powers |
| `coast_alias.csv` | Equal initial three-energy vectors, exactly zero work, and different physical/energy endpoints after 100 ms |
| `startup_work.csv` | Six rows: two torque signs × three integration steps; endpoint state/energies, input work, losses and balance residual |

CSV headers carry units. `pendulum_kinetic_J` always means full body Kₚ; `transfer_power_W` is positive arm-to-pendulum; `input_work_J` is signed mechanical shaft work; `damping_loss_J` sums the two nonnegative losses; `balance_residual_J` equals endpoint total energy minus initial total energy minus input work plus damping loss. Initial total energy is zero in the startup probe. These files are machine records; the parent formulation and interpretation summary are the human entry points.

No plot or controller success claim is derived from these identity checks. Existing runtime tests were not rerun because runtime files did not change; the independent checks and focused script lint cover this delivery.
