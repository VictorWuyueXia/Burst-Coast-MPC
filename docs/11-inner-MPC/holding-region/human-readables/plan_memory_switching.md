# Plan-memory certificate for the unchanged MPC+RL controller

**Result.** The plan-memory construction gives a genuine retained-plan Lyapunov argument, and a weighted version can accommodate arbitrary switching in principle. It does **not** certify the existing complete controller: 57 saved transitions leave its proposed five-degree sublevel, all at CEM replacements. Lower MPC objective is insufficient to preserve this certificate. This rejects this candidate set, not the possibility of another holding region.

The analysis uses commit **1ebd1fa3f25224edce50e77567f3d25767500d4b**, its frozen TD3 arrays, and its 32 saved 40-second confirmation sessions. No controller source, acceptance condition, training, or full experiment was changed or rerun. Calculations use the actual nonlinear dynamics and decoded actor and critics.

## State, plan, and the role of each component

At a command boundary, define the upright-error coordinates and selected plan as

$$
e=(\theta,\beta,\omega,\nu),\qquad
\beta=\operatorname{atan2}(-\sin\alpha,-\cos\alpha),\qquad U=(a_0,\ldots,a_9).
$$

Here \(\theta\) is arm angle, \(\alpha\) is pendulum angle measured from downward, \(\omega,\nu\) are their angular velocities, and \(a_i\in[-1,1]\) is normalized torque. Actual torque is \(0.00459a_i\) N m; each command lasts 0.1 s. The continuous augmented state after selection is \(q=(e,U)\). The random key and decision counter also affect selection; a deterministic regional guarantee must cover their possible values.

Let \(f_h(e,a)\) be one command integrated with RK4 step \(h\). The implemented maps differ: prediction uses \(h=0.001\) s, execution \(h=0.0005\) s, and actor-tail construction \(h=0.002\) s. Write \(f=f_{0.001}\). The stored plan predicts

$$
\xi_0=e,\qquad \xi_{i+1}=f(\xi_i,a_i),\quad i=0,\ldots,9.
$$

The frozen actor \(\pi\) appends the warm-plan tail. With the observation map \(o(e)\) from the repository, define \(\pi(e)\) to mean the actor evaluated on \(o(e)\). The two frozen critics determine

$$
F(e)=\max_{j=1,2}\{-Q_j(o(e),\pi(e))\},\qquad
J(e,U)=\sum_{i=0}^{9}\left(\xi_i^{\mathsf T}W\xi_i+a_i^2\right)+F(\xi_{10}),
$$

$$
W=\operatorname{diag}\left((135\pi/180)^{-2},\,0.15^{-2},\,0.25,\,0.25\right).
$$

These weights and networks come from the saved experiment. CEM proposes replacements and ranks feasible incumbents by \(J\); its final audit checks arm limits and integration agreement. It does not constrain the queue certificate below.

## A stronger queue certificate

The actual actor has a numerically computed resting reference

$$
e_\pi=(-0.1673914425202332,0,0,0),
$$

corresponding to arm angle \(-9.590823^\circ\). Define \(L(v)=(v-e_\pi)^{\mathsf T}P(v-e_\pi)\). The positive-definite matrix \(P\) is computed from the actual actor linearization using \(P-A_\pi^{\mathsf T}PA_\pi=W\). It is an analysis metric and supplies no replacement controller:

$$
P\simeq
\begin{pmatrix}
4.64868746&-131.25515672&3.83812293&-10.55851033\\
-131.25515672&8282.43330728&-215.69144780&653.40493528\\
3.83812293&-215.69144780&6.25819303&-17.28520529\\
-10.55851033&653.40493528&-17.28520529&52.18530815
\end{pmatrix}.
$$

Full precision is stored in the machine artifacts. Suppose the actual nonlinear actor satisfies, on a specified terminal-state domain,

$$
L(f(v,\pi(v)))\le\gamma L(v),\qquad 0<\gamma<1.
$$

This regional inequality needs verification; a stable Jacobian alone does not establish it. For diagnostic calculations we use \(\gamma=0.998834661326013\), halfway between the computed linear squared contraction factor \(0.997669322652026\) and one. Define

$$
S(q)=\max_{0\le i\le10}\gamma^{-i/2}\sqrt{L(\xi_i)},\qquad
\Omega_R=\{(e,U): |a_i|\le1,\ S(e,U)\le R\}.
$$

Later predicted states receive slightly greater weight. This encodes progress within the queue instead of restarting an eleven-step progress clock after each switch.

**Retained-plan theorem, common-map case.** Execute \(a_0\), shift \(a_1,\ldots,a_9\), and append \(\pi(\xi_{10})\), using the same map \(f\) throughout. If the actor inequality holds at the terminal state, then

$$
S(q^+_{\rm shift})\le\sqrt{\gamma}\,S(q).
$$

**Proof.** New predictions are \(\xi_1,\ldots,\xi_{10},f(\xi_{10},\pi(\xi_{10}))\). For the first ten entries, changing index \(i+1\) to \(i\) multiplies the weighted radius by \(\sqrt\gamma\). For the appended entry,

$$
\gamma^{-10/2}\sqrt{L(f(\xi_{10},\pi(\xi_{10})))}
\le\sqrt\gamma\,\gamma^{-10/2}\sqrt{L(\xi_{10})}
\le\sqrt\gamma\,S(q).
$$

Taking the maximum proves the inequality. Thus \(\Omega_R\) is invariant under this retained update and its radius contracts geometrically, provided the hypotheses hold throughout it. Locally, \(S\) vanishes only at \((e_\pi,0)\): the command map's nonzero input derivative makes successive equal resting predictions require zero commands. This gives local augmented-equilibrium Lyapunov stability under the regional hypotheses.

Arbitrary switching would preserve this proof, without a dwell-time restriction, if each replacement obeyed

$$
S(e^+,U_{\rm CEM})\le S(e^+,U_{\rm warm}).
$$

The controller does not implement this condition, and the records contradict it.

## What the actual replacements do

The audit reconstructs 3,200 selections at decisions 300–399 and preceding queues at decisions 299–398. Each warm competitor is regenerated from the actual successor and original 2 ms tail rule. Nominal stage states agree with the archive within \(5.17\times10^{-9}\), and objectives within \(2.97\times10^{-7}\).

| Saved-record check | Result |
|---|---:|
| Selections retaining the preceding plan's first nine shifted commands | 2,531 |
| Replacements changing those commands | 669 |
| Replacements increasing \(S\) relative to the warm competitor | 345 |
| Those increases also accompanied by lower \(J\) | 345 |
| Increases in \(S\) relative to the preceding selected queue | 342 |
| Exits from the proposed five-degree queue sublevel | 57 |
| Such exits occurring at replacements | 57 |

Counts use numerical comparison tolerances recorded in the JSON. Five retained-nine cases have reconstructed tail discrepancies above \(10^{-8}\); they are not labeled exact full-plan retention.

At command instants, Cauchy–Schwarz gives

$$
|\beta|\le\sqrt{(P^{-1})_{22}}\,S(q),\qquad
R_5=\frac{5\pi/180}{\sqrt{(P^{-1})_{22}}}=0.8732357502790972.
$$

The second coordinate is pendulum error. Thus \(R_5\) is the largest radius whose ellipsoid projection guarantees five degrees. At decision 325, episode 10:

| Quantity | Value |
|---|---:|
| Preceding selected queue radius | 0.1029276591 |
| Warm queue radius at the actual successor | 0.1004051108 |
| Newly selected queue radius | 5.0193137548 |
| Warm objective | 3.1977420235 |
| Selected objective | 3.1938071121 |

A reachable augmented state inside \(\Omega_{R_5}\) therefore transitions outside it while CEM lowers its objective. The increase is much larger than reconstruction error. Exact rational comparisons also verify the strict sublevel exit on the stored forecast numbers; these are not an interval proof of integration or neural rounding.

Across this archive, ideal common-map shifts satisfy weighted contraction to approximately \(9\times10^{-16}\). The largest difference between actual warm and ideal-shift radii is \(2.18\times10^{-4}\). These sampled checks distinguish the replacement failure from map mismatch; they are not unsampled regional bounds.

This metric strongly penalizes coupled angle–velocity directions. A large radius does not imply that the physical pendulum exceeds five degrees. The final-ten-second physical upright error remains approximately \([-1.948^\circ,1.937^\circ]\). Those observations and the rejected set are compatible.

## Why objective improvement cannot supply the missing implication

At the resting reference with a zero queue, all predictions equal \(e_\pi\) and \(S=0\). The critic branches are separated there, so the active branch is differentiable. Perturbing the last command gives

$$
\left.\frac{\partial J(e_\pi,U)}{\partial a_9}\right|_{U=0}
=\nabla F(e_\pi)^{\mathsf T}\partial_a f(e_\pi,0)
\simeq0.84185347\ne0.
$$

The stage derivative is zero: \(a_9\) affects only its quadratic effort and the terminal state. Arbitrarily small negative last-command perturbations therefore lower \(J\) while making \(S>0\). One feasible offline comparison is

$$
U=(0,\ldots,0,-0.001):
\quad J:3.2121083746\longrightarrow3.2112884022,\qquad
S:0\longrightarrow0.0200146895.
$$

Nominal and refined arm nodes pass; maximum integration disagreement is below \(8\times10^{-13}\). This is a plan comparison, not a regenerated CEM draw. Together with actual selected replacements, it shows why critic-guided improvement cannot replace queue-Lyapunov decrease. An unrestricted cost-improvement argument also cannot establish stability of the actor's single resting point.

## Complete-controller bounded-error theorem and its failed numerical test

A practical-region theorem can allow replacement increases. Put \(s=\sqrt\gamma\). Compare the ideal common-map shifted queue \(q_I\), implemented warm queue \(q_H\), and actual CEM successor \(q^+\). The warm queue uses actual execution and the 2 ms tail map. Define

$$
b_\pi(q)=[S(q_I)-sS(q)]_+,\quad
b_h(q)=|S(q_H)-S(q_I)|,\quad
b_C(q,\kappa)=[S(q^+)-S(q_H)]_+.
$$

Here \([x]_+=\max(x,0)\); \(\kappa\) is the key and counter. These errors separate actor nonlinearity, integration mismatch, and replacement selection. A proof needs independently certified **uniform upper bounds** over the augmented domain and all admitted keys.

Suppose these upper bounds sum to \(b\), every decision returns an accepted plan, and the bounds hold on an outer sublevel \(\Omega_{\bar R}\). Then

$$
S(q^+)\le sS(q)+b,\qquad R_*=\frac{b}{1-s}.
$$

If \(R_*\le\bar R\), then \(\Omega_{R_*}\) is invariant, with

$$
S(q_k)\le s^kS(q_0)+(1-s^k)R_*,
\qquad [S(q_k)-R_*]_+\le s^k[S(q_0)-R_*]_+.
$$

**Proof.** Add the three comparison bounds and iterate the scalar recurrence. Subtract \(R_*=sR_*+b\) to obtain the excess-radius contraction. On a compact admissible augmented domain, continuity of \(S\) gives distance-to-sublevel Lyapunov stability and convergence from this excess bound. This establishes bounded-error set stability, rather than convergence to zero physical error.

For the chosen metric and factor, any constant-residual bound covering all audited transitions must satisfy

$$
b\ge\max_k[S(q_{k+1})-sS(q_k)]_+=4.916446086,\qquad
R_*\ge8435.34.
$$

This sample-derived **necessary** radius is far above \(R_5\). It is neither a physical-error prediction nor a certified bound. Restricting the domain to \(\Omega_{R_5}\) does not restore invariance, because decision 325 starts inside and leaves it.

## What remains necessary

The complete-controller five-degree region is not established. Raising \(R\) to the largest archived radius would only enclose a finite record and would not prove invariance.

The next certificate must allow cost-beneficial queue excursions actually made by CEM. A replacement-aware storage function, possibly combining predicted stage cost and terminal value with a queue penalty, needs verification against feasible cost sublevels. Adding an acceptance filter would change the controller; none was added.

Without new experiments, the required calculations are: certify nonlinear actor and critic bounds on a stated terminal domain; bound CEM replacements on a reachable joint state–plan region, including critic switching and the final audit; prove at least one incumbent remains accepted; and bound the dense inter-command tube. A five-degree claim requires that tube bound, in addition to command-boundary invariance. Deterministic guarantees must quantify over keys. Expected decrease would require a separate probabilistic theorem and would not establish sample-path invariance.

The saved networks and source supply inputs for these verification problems; finite records alone cannot resolve universal inequalities. Reproduction and exact definitions are in [the machine contract](../machine-scannables/plan_memory_contract.md). Numerical outputs are in the existing actual-holding-study artifact folder under the plan_memory prefix.
