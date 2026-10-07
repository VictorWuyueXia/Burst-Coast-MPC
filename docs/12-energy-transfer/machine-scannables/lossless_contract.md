# Lossless surface, decoder and cost contract

> Implementation update (2026-10-07): the derivation and earlier study record below are retained. Earlier arm-veto and unchanged-runtime statements are superseded: shared damping is now zero, JAX capture omits arm centering, and ±90° receives a soft penalty. See the [implementation contract](implementation_contract.md) and [current formulation](../formulation_plan.md).

## Revision scope and state

This contract supersedes the earlier arm-centering objective and unresolved-action handling. The three-energy planner commands work; the full-state decoder chooses torque; an unavailable or constraint-rejected selected command applies zero torque. Both physical damping coefficients are explicitly zero in this study. The original damped validation files remain historical and are not overwritten.

Let x = (θ,α,ω,ν) be measured physical state: arm angle θ and pendulum angle α from downward, in rad; signed arm/pendulum angular velocities ω and ν, in rad/s. Let u be motor shaft torque in N m. Time derivatives use seconds. Constants Jₐ, J, J₀ and B are respectively arm-link, pendulum-hinge, base and coupling inertias in kg m², and G is the gravitational energy coefficient in J. They are computed from the nominal configured geometry as defined in the general analytical contract.

Define state-dependent inertia coefficients A and C, determinant Dₘ, full pendulum carried inertia Aₚ, and the energy encoder E:

$$
A=J_0+J\sin^2\alpha,\quad C=B\cos\alpha,\quad D_M=AJ-C^2,\quad A_p=A-J_a,
$$
$$
K_a=\tfrac12J_a\omega^2,\quad
K_p=\tfrac12A_p\omega^2+C\omega\nu+\tfrac12J\nu^2,\quad V=G(1-\cos\alpha),
$$
$$
e=E(x)=(K_a,K_p,V)^T,\quad H=\mathbf1^Te,\quad
E_\star=2G,\quad e_\star=(0,0,E_\star)^T,\quad\mathbf1=(1,1,1)^T.
$$

Kₐ, Kₚ, V, H and E★ have units J. The target omits arm centering. The underlying Lagrangian is L = Kₐ + Kₚ − V; its Euler–Lagrange equations are the existing ODE with bᵣ = bₚ = 0 N m s, where bᵣ and bₚ are the arm and pendulum viscous damping coefficients. No empirical energy-allocation fit is needed.

## Admissible region and total-work plane

Define ℰ as the image of the physical energy encoder. For a candidate energy triple, set c = 1−V/G, the implied cosine of α. Then C = Bc and Aₚ = J₀−Jₐ + J(1−c²). Completing the square in Kₚ gives

$$
K_p=\frac J2\left(\nu+\frac C J\omega\right)^2
+\frac12\left(A_p-\frac{C^2}{J}\right)\omega^2.
$$

For this nominal model the resulting energy-space conditions are necessary and sufficient when physical angles/velocities have no additional bounds:

$$
\mathcal E=\left\{(K_a,K_p,V):K_a\ge0,\quad0\le V\le2G,\quad
K_p\ge\frac{A_p-C^2/J}{J_a}K_a\right\}.
$$

Arm-position bounds do not change this encoder image because θ does not enter E; they do restrict physical trajectories and decoder feasibility. Additional speed bounds, if introduced later, would shrink the region.

For a held action lasting Δ = 0.1 s, let w denote requested work and W actual delivered shaft work, in J. The exact lossless balance and candidate requested-work surface are

$$
\dot H=u\omega,\qquad H'=H+W,\qquad
\mathcal S(e,w)=\{z\in\mathcal E:\mathbf1^Tz=\mathbf1^Te+w\}.
$$

H′ is endpoint total energy and z is a candidate next energy. Equality with w holds only if the decoder actually realizes w. S is generally a two-dimensional plane portion, which can degenerate or be empty. For zero applied torque W = 0, the energy point stays on S(e,0) but generally moves within it. Neither the full plane nor all physically admissible points on it are necessarily reachable from the current phase in one decision.

## Instantaneous actuation and finite-step curve

Let f(x,u) be the physical ODE, in angle/speed derivative units, and let ∂E/∂x denote the encoder Jacobian. Define a(x) as the unforced energy drift in W and χ as the derived dimensionless input-power fraction:

$$
a(x)=\frac{\partial E}{\partial x}f(x,0),\qquad
\chi=\frac{J_aJ}{D_M},\qquad
\dot e=a(x)+u\omega(\chi,1-\chi,0)^T.
$$

Proof: the arm acceleration sensitivity to u is J/Dₘ, hence the arm kinetic-rate sensitivity is JₐωJ/Dₘ. Total energy-rate sensitivity is ω, while potential-rate sensitivity at fixed state is zero. Subtraction gives the pendulum-body term. Positive definiteness of the pendulum kinetic form implies A−C²/J > Jₐ, so 0 < χ < 1. In the lossless model the three drift components sum to zero.

Thus possible instantaneous energy derivatives at fixed x form a line segment as bounded u varies; when ω = 0 the input line collapses to a point. This is an instantaneous rank loss, not a proof that finite-duration torque cannot affect energy. The initial rest-to-motion work is second-order in duration.

Define FΔ(x,u) as the physical endpoint under constant u and policy cap U = 0.00918 N m. The exact scalar work map and endpoint curve are

$$
\mathcal W(x,u)=u[F_\Delta(x,u)_\theta-\theta],\qquad
\mathcal C_x=\{E(F_\Delta(x,u)):u\in[-U,U]\}.
$$

The θ subscript selects arm angle. Cₓ is a parametrized curve, not a full two-dimensional control surface. Fixed requested work intersects Cₓ with S(e,w); multiple, absent or degenerate roots are possible. A more flexible within-step torque waveform would introduce more control degrees of freedom, but is not part of this held-torque proposal.

## A second coast invariant and waiting limits

Define arm generalized momentum pθ in kg m²/s by differentiating the Lagrangian with respect to ω:

$$
p_\theta=\frac{\partial L}{\partial\omega}=A\omega+C\nu,\qquad
\dot p_\theta=u.
$$

The result follows because L has no explicit θ dependence and damping is zero. During zero torque both H and pθ are conserved. Different phase states sharing the same e may have different pθ, hence different accessible coast trajectories. At e★ both velocities vanish, so pθ = 0. If coast begins with nonzero pθ, it cannot reach exact e★ merely by waiting. Even H = E★ and pθ = 0 are not sufficient to guarantee finite-time arrival or stable capture.

Work and angular impulse are distinct: work integrates uω, while momentum change integrates u. A decoder must account for both physical effects even though the high-level action is work. A policy that requests zero forever once total energy matches E★ can therefore fail to redistribute the energy. Near-target positive and negative work steps may still be needed.

## Squared objective and physical scaling

Define dimensionless terminal cost Φ₂ with equal component weights:

$$
\Phi_2(e)=\frac{K_a^2+K_p^2+(V-E_\star)^2}{2E_\star^2}.
$$

This is not (H−E★)²: the latter is constant on each total-energy plane and cannot guide redistribution. On the target-total plane, substitute E★−V = Kₐ+Kₚ:

$$
\Phi_2=\frac{K_a^2+K_p^2+(K_a+K_p)^2}{2E_\star^2}.
$$

It vanishes only when both kinetic terms vanish. Define β = α−π locally near upright. At zero speeds, using the cosine expansion,

$$
E_\star-V=\tfrac12G\beta^2+O(\beta^4),\qquad
\Phi_2=\frac{\beta^4}{32}+O(\beta^6).
$$

O denotes the order of the omitted small-angle terms. More generally, kinetic energies and the height deficit are quadratic in small physical deviations, so squaring them yields fourth-order leading cost. Positive definiteness in energy space does not establish a nonzero quadratic physical-state Hessian, local stability, or well-conditioned gradient control. The earlier dimensionless linear score Φ₁ = (Kₐ+Kₚ+E★−V)/E★ remains a comparison option; no blend parameter is added.

## Decoder interface and actual transition

D receives the physical state x and work request w; high-level state remains e. Proposed work tolerance is εW = 10⁻⁴E★ J. Resolve roots of |𝒲(x,u)−w| ≤ εW within |u| ≤ U and select minimum endpoint Φ₂, with smaller |u| then positive-first ties. Check the selected root's path against |θ| ≤ Θ, where Θ = π/2 rad. There is no centering penalty or centering tie-break.

For a violating selected root or absent acceptable root, set applied torque to zero. For w = 0 also apply zero directly. This explicit controller branch supersedes the original unresolved-request contract. Record w, selected/applied torque, actual W, rejection reason and excursion. Advance the physical model under the applied torque, then encode its actual endpoint. On rejection the correct work surface is S(e,0), regardless of requested w. A root solver's numerical failure must be recorded separately from evidence that no physical root exists.

The arm check is sampled in an implementation; refine between samples when validating near a crossing. A pure coast may also violate the bound. In this model x = (1.55,0,1,0) and u = 0 gives θ(t) = 1.55+t, α = 0, ω = 1 and ν = 0. At 0.1 s the arm is 1.65 rad. The requested rule still coasts and flags the violation; it is not a hard-constraint guarantee.

For this study the physical success checker drops |θ| ≤ 0.08 rad and instead requires |θ| ≤ π/2 along with upright error ≤0.08 rad, arm speed magnitude ≤0.15 rad/s, pendulum speed magnitude ≤0.20 rad/s, for five consecutive 20 ms samples. This revised checker is a planned study contract; the existing runtime checker has not been edited.

## Evidence, artifacts and reproduction

`validate_lossless.py` has one new `main()` and reuses the original `energy_ledger`. It creates a physical-config copy with both damping values exactly zero, then runs independent probes in vectorized arrays. It does not edit shared physics, fit a model or implement the decoder.

| Machine artifact | Meaning |
| --- | --- |
| `lossless_coast.csv` | A 1 s uncontrolled coast at 20/2/1 ms integration; states, three energies, total and generalized arm momentum |
| `lossless_torque_curve.csv` | 257 bounded torques held 100 ms from the same state; endpoint energy, actual work, plane residual and squared cost |
| `lossless_cost.csv` | 31 local upright angle errors, linear height score and squared component score at zero velocities |
| `lossless_checks.json` | Explicit damping values, probes, refinement errors, power-allocation residual, cost slope and boundary counterexample |

The separate human figure `../human-readables/lossless_energy_geometry.png` displays energy exchange, the constant-total plane and local cost scaling. Its plane uses H₀, the initial total energy, as normalization; H₀ is not E★ for this probe. The plane triangle deliberately displays conservation only, not the complete admissibility or reachability constraints.

Observed 1 ms coast drift is 1.31 × 10⁻¹² J and momentum drift 3.44 × 10⁻¹³ kg m²/s, both below the diagnostic 10⁻⁹ bounds and decreasing with refinement. Individual energy change exceeds 0.001 J. The finite-step work-plane residual is below 2.34 × 10⁻¹² J, versus a 10⁻⁹ J pass bound. Instantaneous allocation agrees within the 10⁻¹² W bound. The local log–log squared-cost slope is 3.99980, within 0.01 of four. All checks passed; none is a control-performance result.

Reproduce from repository root:

```bash
MPLCONFIGDIR=/tmp/bcmpc-mpl PYTHONPATH=src .venv/bin/python docs/12-energy-transfer/machine-scannables/validate_lossless.py
.venv/bin/ruff check docs/12-energy-transfer/machine-scannables/validate_lossless.py
```

The new zero-damping study uses fine CPU integration as a reference; current JAX defaults still load the shared damped configuration. Future prediction and plant instances must both use the lossless parameters, and numerical drift must not be mislabeled as physical damping.
