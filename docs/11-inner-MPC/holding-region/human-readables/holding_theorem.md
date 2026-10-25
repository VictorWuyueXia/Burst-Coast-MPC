# A concrete Lyapunov holding region for MPC+RL

**Construction update, 2026-10-25:** this remains a valid conditional reference theorem. Its LQR-derived metric and illustrative radii are not the adopted MPC+RL certificate. The [current construction](mpc_rl_linearization.md) derives the reference map from the finite-horizon MPC objective, includes assumed neural terminal features, and treats the finite-CEM policy separately.

Date: 2026-10-24. Source baseline: `cdce8f93f98d6c978e6a723d4aea11f4524c59ad`. This document supplies a concrete certificate construction and its proof. Its application to the unchanged controller remains conditional on the explicitly identified acceptance and residual checks; those checks have not been completed without the frozen checkpoint.

## 1. The property to prove

The appropriate target is stability of a small region around upright rest. A controller with persistent oscillations need not preserve the zero-error state, so Lyapunov stability of that equilibrium cannot simply be asserted.

Let physical state and upright error be

$$
x=(\theta,\alpha,\omega,\nu)^{\mathsf T},\qquad
\beta=\operatorname{atan2}(-\sin\alpha,-\cos\alpha),\qquad
e=(\theta,\beta,\omega,\nu)^{\mathsf T}.
$$

Arm and pendulum angles are in radians; their respective speeds are in rad/s. The pendulum angle is measured from downward, so upright corresponds to zero wrapped error. The target arm angle and both target speeds are zero. Fixed coordinate units are used throughout the matrix inequalities.

Commands have duration 0.1 s. Let the exact nominal nonlinear flow over one command, expressed in error coordinates, be the map below. The normalized torque and the actual MPC action are defined by

$$
\Delta=0.1\ {\rm s},\qquad
a=u/u_{\max},\qquad u_{\max}=0.00459\ {\rm N\,m},\qquad
e_{k+1}=g_{\Delta}(e_k,a_k),\qquad a_k=\mu(e_k,h_k).
$$

The torque limit is the documented baseline value; its match to the original run must be checked. The integer index counts commands. The controller map is the actual CEM MPC with the frozen RL terminal value and final numerical audit. Its memory contains the previous plan, random-key/index information, fixed batch layout, and any other context needed to determine execution. The map is undefined when no accepted action is returned; a stopped solver is not an allowed trajectory in the theorem.

For a theorem about the implemented discrete simulator, replace the exact flow by its executed 0.5 ms RK4 command map throughout. The proof is identical, but such a result certifies that discrete simulator rather than the exact ODE. Prediction and seed integrations remain part of the actual controller map in either version.

## 2. A specific metric with a verified linear contraction

Linearize the exact nominal flow at upright rest. Denote its state and normalized-input Jacobians by the matrices below. A fixed reference feedback and reference successor matrix are

$$
A=\partial_e g_{\Delta}(0,0),\quad B=\partial_a g_{\Delta}(0,0),\qquad
K=(-0.075714367,\ 15.360350473,\ -0.121826284,\ 1.221156714),\qquad
A_K=A-BK.
$$

The feedback row is a finite-decimal reference derived from the existing quadratic design. It is used only for analysis. The actual commanded action remains the MPC output, not the reference feedback.

Choose the following symmetric metric matrix, with its displayed finite decimals treated as exact:

$$
P=\begin{bmatrix}
2.898281&-14.621739&0.936788&-1.365264\\
-14.621739&645.880505&-21.397460&51.139393\\
0.936788&-21.397460&1.458784&-1.988443\\
-1.365264&51.139393&-1.988443&4.698294
\end{bmatrix}.
$$

Its entries account for the stated angle and speed units. Define the quadratic function and its associated norm by

$$
L(e)=e^{\mathsf T}Pe,\qquad \|e\|_P=\sqrt{L(e)}.
$$

Exact rational calculations with a rigorously bounded matrix-exponential remainder establish

$$
P\succ0,\qquad A_K^{\mathsf T}PA_K\prec q^2P,\qquad q=0.970.
$$

Thus the reference successor contracts this norm by at least the stated factor. The reference proof and reproducible calculation are specified in the [machine certificate](../machine-scannables/certificate_specification.md). This result alone does not establish anything about the actions selected by MPC.

## 3. The explicit connection to the actual MPC+RL action

Define the actual one-command successor deviation from the contracting reference by

$$
w(e,h)=g_{\Delta}(e,\mu(e,h))-A_Ke.
$$

This is an evaluable vector with four physical error coordinates. It includes nonlinear dynamics and every effect of the RL terminal cost, finite CEM search, warm start, and selected first action. It is not an external disturbance added to the controller.

For a proposed outer norm radius, choose a nonempty permitted memory set. Define

$$
S>0,\qquad \mathcal D_S=\{e:\|e\|_P\le S\},\qquad
d(S)=\sup_{e\in\mathcal D_S,\ h\in\mathcal H}\|w(e,h)\|_P.
$$

The memory set must be specified before verification and include all histories to which the result is applied. The supremum is a uniform bound to certify, not the maximum of sampled trajectories. Because a controller failure makes the successor undefined, acceptance must be established separately.

The two actual-controller obligations are:

1. Every state and memory in the declared product domain returns an accepted normalized action; its actual memory update remains in the permitted memory set. Inherited swing-up memory must belong to that set before this theorem can be composed with swing-up.
2. The successor deviation has a verified bound satisfying

$$
d(S)\le(1-q)\tau,\qquad 0\le\tau<S.
$$

The smaller radius is a certificate specification, not a controller parameter. It determines the target region

$$
\Omega_{\tau}=\{e:e^{\mathsf T}Pe\le\tau^2\}.
$$

These obligations cannot be replaced by a claim that MPC behaves like LQR. Their purpose is to check precisely how far its actual successor can depart from a proven contracting reference.

## 4. Theorem and proof

**Theorem.** Under the two obligations above, the outer domain and inner region are invariant at command instants. All initial states in the outer domain, with permitted memories, approach the inner region exponentially in distance. The inner region is uniformly Lyapunov stable in physical error, uniformly over permitted controller memories. The conclusion concerns a set, not asymptotic stability of upright rest.

**Proof.** The reference inequality and the triangle inequality give, for every accepted transition in the domain,

$$
\|e_{k+1}\|_P\le\|A_Ke_k\|_P+\|w(e_k,h_k)\|_P
\le q\|e_k\|_P+d(S).
$$

At the outer boundary the right side is at most the outer radius, since

$$
qS+d(S)\le qS+(1-q)\tau\le S.
$$

The memory update also remains permitted by the first obligation. Induction therefore makes the bound applicable at every later command. The same argument at the smaller radius gives its invariance.

Define the nonnegative residual floor and the positive-part operator by

$$
b=\frac{d(S)}{1-q},\qquad [s]_+=\max(s,0).
$$

Iteration proves an explicit full-state error estimate,

$$
\|e_k\|_P\le q^k\|e_0\|_P+(1-q^k)b,\qquad
\limsup_{k\to\infty}\|e_k\|_P\le b\le\tau.
$$

To prove Lyapunov stability of the region itself, use its weighted distance and a set Lyapunov function,

$$
\operatorname{dist}_P(e,\Omega_{\tau})=
\inf_{v\in\Omega_{\tau}}\|e-v\|_P=[\|e\|_P-\tau]_+,
\qquad V_{\tau}(e)=[\|e\|_P-\tau]_+^2.
$$

The distance identity follows by transforming the ellipsoid into a Euclidean ball. The variable inside the infimum is a candidate point of that ellipsoid. Since the deviation bound is no larger than the contraction margin at the target radius,

$$
[\|e_{k+1}\|_P-\tau]_+\le q[\|e_k\|_P-\tau]_+,
\qquad V_{\tau}(e_{k+1})\le q^2V_{\tau}(e_k).
$$

Let the smallest and largest eigenvalues of the positive-definite matrix be denoted by the minimum and maximum subscripts. Norm equivalence gives

$$
\operatorname{dist}_2(e_k,\Omega_{\tau})
\le\sqrt{\frac{\lambda_{\max}(P)}{\lambda_{\min}(P)}}\,
q^k\operatorname{dist}_2(e_0,\Omega_{\tau}).
$$

The subscript two denotes ordinary Euclidean distance in the fixed physical coordinate convention. Let the displayed eigenvalue factor be called the norm-equivalence factor. For a Euclidean distance tolerance, a concrete initial-distance choice is

$$
c_P=\sqrt{\lambda_{\max}(P)/\lambda_{\min}(P)},\qquad
\epsilon>0,\qquad
\delta=\min\left\{\frac{\epsilon}{c_P},\frac{S-\tau}{2\sqrt{\lambda_{\max}(P)}}\right\}>0.
$$

Initial distance below this value places the physical state inside the outer domain and keeps its distance to the target below the tolerance at all later commands. The choice is independent of decision index and permitted memory. This proves uniform Lyapunov stability and exponential attraction relative to the certified outer domain. The argument applies from any initial command instant. ∎

If the target radius strictly exceeds the residual floor, entry is finite. For an initial norm larger than the target, an upper bound on the entry decision is

$$
K_{\tau}\le
\left\lceil\frac{\log((\tau-b)/(\|e_0\|_P-b))}{\log q}\right\rceil.
$$

The ceiling rounds upward to an integer. Starts already in the target have entry decision zero. At equality between the target and floor, the theorem guarantees attraction but not finite-time entry.

## 5. Concrete candidate regions and their physical meaning

Take the proposed outer and inner norm radii to be 0.2 and 0.1. These are analysis choices, not measured regions of attraction. The exact algebraic sets are

$$
\mathcal D_{0.2}=\{e:e^{\mathsf T}Pe\le0.04\},\qquad
\Omega_{0.1}=\{e:e^{\mathsf T}Pe\le0.01\}.
$$

For these sets the actual-controller condition becomes the concrete verification target

$$
\boxed{d(0.2)\le0.003.}
$$

The number is the proven linear contraction margin times the proposed inner radius. It is not an observed MPC residual. If it and accepted continuation are certified, the theorem proves the claimed holding result. An outer-domain invariance margin alone would only require a residual at most 0.006; the tighter condition produces the smaller attracting region.

For each state coordinate, numbered in the declared error-state order, the ellipsoid implies

$$
|e_i|\le\tau\sqrt{(P^{-1})_{ii}},\qquad i=1,2,3,4.
$$

Exact cofactor calculations certify the upper factors 0.660434, 0.105915, 1.330138, and 1.369292, respectively. Multiplying these by the radii gives:

| Coordinate | Outer domain | Inner holding region |
|---|---:|---:|
| Arm error magnitude | at most 0.1320868 rad; below 7.569° | at most 0.0660434 rad; below 3.785° |
| Pendulum upright error magnitude | at most 0.0211830 rad; below 1.214° | at most 0.0105915 rad; below 0.607° |
| Arm speed magnitude | at most 0.2660276 rad/s | at most 0.1330138 rad/s |
| Pendulum speed magnitude | at most 0.2738584 rad/s | at most 0.1369292 rad/s |

These are coordinate bounds implied by the ellipsoids. The enclosing boxes are not certified invariant sets themselves.

## 6. Holding between commands and what is still unproved

Let the nominal error trajectory over a held command be denoted by the flow with a variable time subscript. The continuous holding tube generated from the inner region is

$$
\mathcal C_{\tau}=
\{g_t(e,\mu(e,h)):e\in\Omega_{\tau},\ h\in\mathcal H,\ 0\le t\le\Delta\}.
$$

Once a command state is in the invariant inner region, every subsequent inter-command trajectory lies in this tube. This follows interval by interval from endpoint invariance. It is an invariant tube for the sampled-data execution when command phase and held action are retained. It need not equal the endpoint ellipsoid.

For continuous-time angle and speed claims, bound each coordinate of this tube using validated integration under the actual selected commands. In particular, to claim uninterrupted five-degree holding, its pendulum-error supremum must not exceed five degrees. Small endpoint errors alone do not prove this. A theorem for RK4 node values instead needs to state that discrete temporal scope explicitly.

The verified reference inequality and the set-stability proof are complete. A nonempty certified region for the actual MPC+RL controller is not yet established: its checkpoint is missing, its permitted memory domain is not yet certified, and neither universal accepted continuation nor the 0.003 successor bound has been checked. The precise next mathematical calculation is that bound over the specified domain, not a new proof of LQR stability or a modification of the controller.
