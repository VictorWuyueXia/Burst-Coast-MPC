# Holding certificate: reference proof and implementation obligations

Date: 2026-10-24. Commit: `cdce8f93f98d6c978e6a723d4aea11f4524c59ad`. Read the [theorem](../human-readables/holding_theorem.md) for definitions and conclusions.

## Structure budget and milestones

This work adds three English Markdown documents and three Chinese companions. One straight-line, standard-library CPU script verifies the reference matrices and writes one JSON certificate under `artifacts/`. It adds zero controller classes, functions, methods, dataclasses, wrappers, helper functions, CLI options, tunable runtime parameters, or fallback behaviors. Preserve compact continuous logic and the repository's minimum-realization discipline.

The calculation's objects are the primitive physical-value dictionary, the augmented linear generator and Taylor matrices, fixed metric/gain/contraction data, matrix positivity witnesses, coordinate cofactors, and output record. Loop indices only enumerate fixed matrix coordinates and Taylor terms. Mathematical certificate choices are the outer radius, target radius, and actual-successor bound; they are not new controller settings.

| Milestone | Check | Success flag |
|---|---|---|
| Construct reference | Exact nominal sampled linearization; positive metric; bounded exponential remainder | Positive exact principal minors for the metric and contraction witness |
| Prove set stability | Invariance, residual-floor estimate, distance-to-set contraction, memory closure | Complete theorem with the actual-controller obligations stated explicitly |
| Check geometry | Exact inverse diagonal bounds and proposed radius margins | Outward coordinate factors certified by rational cofactor comparisons |
| Apply to MPC+RL | Accepted continuation, permitted histories, uniform successor deviation, inter-command tube | Not completed; requires frozen checkpoint and rigorous regional analysis |

## Exact nominal reference model

Use zero damping and the documented 0.00459 N m torque cap. Let arm and pendulum masses and lengths be the four primitive quantities below; the last quantity is gravity:

$$
m_a=0.095\ {\rm kg},\quad l_a=0.085\ {\rm m},\quad
m_p=0.024\ {\rm kg},\quad l_p=0.129\ {\rm m},\quad g=9.81\ {\rm m/s^2}.
$$

The base, pendulum, coupling, and gravity coefficients, and the vertical inertia determinant, are derived rather than fitted:

$$
A_0=m_al_a^2/3+m_pl_a^2,\quad J_p=m_pl_p^2/3,\quad
C_0=m_pl_al_p/2,\quad G_0=m_pgl_p/2,\quad D_0=A_0J_p-C_0^2>0.
$$

The inertia coefficients have kg m squared units, the gravity coefficient has N m units, and the determinant has squared inertia units. In the error-state order of the theorem, the continuous state and normalized-input matrices are

$$
A_c=\begin{bmatrix}0&0&1&0\\0&0&0&1\\0&C_0G_0/D_0&0&0\\0&A_0G_0/D_0&0&0\end{bmatrix},\qquad
B_c=u_{\max}(0,0,J_p/D_0,C_0/D_0)^{\mathsf T}.
$$

Their exact zero-order-hold matrices are the blocks of

$$
T=\Delta\begin{bmatrix}A_c&B_c\\0&0\end{bmatrix},\qquad
\exp(T)=\begin{bmatrix}A&B\\0&1\end{bmatrix},\qquad \Delta=0.1\ {\rm s}.
$$

The lower block consists of a zero row and one scalar. The matrix with the single-letter name is dimensionless and five by five. All its entries are rational for the declared primitives.

## Rigorous verification of the contraction

The fixed metric and reference feedback are exactly the finite decimals printed in the theorem. They were selected by rounding a Riccati construction with the existing stage weights, but their validity below does not rely on that floating-point construction.

Define a 100th-order Taylor polynomial and the induced maximum-row-sum norm of the dimensionless generator:

$$
E_{100}=\sum_{j=0}^{100}\frac{T^j}{j!},\qquad M=\|T\|_{\infty}.
$$

The index enumerates Taylor terms. Since the generator norm is below 102, a geometric bound on successive tail terms gives

$$
\|\exp(T)-E_{100}\|_{\infty}
\le\eta=\frac{M^{101}}{101!}\frac{1}{1-M/102}
<1.435\times10^{-32}.
$$

All calculations through this bound use exact fractions. Let the reference successor approximation be obtained from the upper blocks of the Taylor polynomial. Denote it by the same reference-matrix symbol with a hat. Its spectral-norm error is bounded by

$$
\|A_K-\widehat A_K\|_2\le\zeta=4\eta(1+\max_j|K_j|).
$$

The factor four bounds the spectral norm of a four-by-four matrix by four times its largest entry magnitude. Define two convenient spectral-norm upper bounds and a congruence-error upper bound by

$$
p_{\rm ub}=\|P\|_{\infty},\qquad
c_{\rm ub}=4\max_{i,j}|(\widehat A_K)_{ij}|,\qquad
\gamma=p_{\rm ub}(2c_{\rm ub}\zeta+\zeta^2)<1.171\times10^{-25}.
$$

These are analytic bounds, not controller parameters. They imply

$$
q^2P-A_K^{\mathsf T}PA_K\succeq
q^2P-\widehat A_K^{\mathsf T}P\widehat A_K-\gamma I,\qquad q=97/100.
$$

The identity matrix has dimension four. Exact rational leading-principal-minor tests show that the metric and the right-hand matrix are positive definite. Their approximate minor magnitudes, included only for human inspection, are

| Matrix | First | Second | Third | Fourth |
|---|---:|---:|---:|---:|
| Metric | 2.898281 | 1658.147945 | 1111.273611 | 592.693187 |
| Lower contraction witness | 0.01457082 | 3.43993151 | 0.07672091 | 0.02970906 |

The positivity decisions themselves use exact fractions. Therefore the finite-decimal metric is positive definite and the exact nominal linear reference contracts its norm by a factor strictly below 0.970.

## Actual-controller residual and acceptance verification

Retain the actual frozen-critic CEM policy. Define the nonlinear linearization remainder, using the exact nonlinear command map and exact sampled matrices, by

$$
n(e,a)=g_{\Delta}(e,a)-Ae-Ba.
$$

Then the theorem's actual successor deviation decomposes exactly as

$$
w(e,h)=B[\mu(e,h)+Ke]+n(e,\mu(e,h)).
$$

This identifies two concrete sources: the difference between the selected MPC action and reference action, and nonlinear flow error. It also gives a conservative verification route by bounding the two norms separately. A direct joint bound can be tighter. No positivity or Bellman-consistency assumption on the RL critic is needed if the actual successor bound is verified.

For the proposed norm radius 0.2, specify the permitted memory set, prove its closure under actual memory updates, and establish accepted actions throughout the product domain. Do not infer acceptance from the survival of a nominal incumbent: final audit may discard a replacement. Then rigorously maximize the weighted successor deviation over that domain. A bound at most 0.003 certifies the stated inner norm radius 0.1.

The checkpoint, original runtime settings, and accepted-plan archives are still absent locally. A sampled maximum is only a lower bound on the supremum and cannot certify it. Include inherited swing-up plans and the declared random-key/batch scope. If only an invariant augmented subset can be verified, state that subset explicitly rather than claiming the full product domain.

Continuous-time holding additionally requires bounds on the flow tube in the theorem. For a discrete-execution theorem, recompute the successor deviation using the actual refined map; its difference from the exact linear reference is already included in that deviation. For an exact-ODE theorem, numerical approximations used to bound the residual and tube need rigorous error enclosures.

## Reproduction and files

Repository-relative reproducer: `artifacts/rotary_pendulum/inner-mpc/holding-region-certificate-20261024/machine-scannables/verify_reference.py`. Run it with Python's standard library. It writes `linear_reference_certificate.json` beside itself. The machine record preserves exact physical primitives, fixed matrices, remainder bounds, positivity decisions, and outward coordinate factors. The enormous rational principal minors are reproduced by the script; their JSON magnitudes are approximate, while their saved positivity flags are decided exactly.

The inverse metric diagonal is computed by exact cofactors. Each printed outward factor in the theorem is certified by squaring it and comparing to the exact corresponding inverse diagonal. The 0.003 nonlinear-controller threshold is explicitly unverified in the output record.

The distinction between state-only and warm-start-dependent guarantees follows the setup of suboptimal MPC; see Rawlings, Mayne, and Diehl, [Model Predictive Control, second edition, fifth printing](https://sites.engineering.ucsb.edu/~jbraw/mpc/MPC-book-2nd-edition-5th-printing.pdf), Section 2.7. The holding theorem here is proved directly and does not assume the learned terminal function meets a textbook terminal-decrease condition.
