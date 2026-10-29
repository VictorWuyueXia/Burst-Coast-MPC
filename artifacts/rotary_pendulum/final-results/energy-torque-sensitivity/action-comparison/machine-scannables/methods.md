# Recorded actions versus one-step energy predictions

Independent choices are lane 325; 100 ms constant-action duration; 257 uniform torques plus actual torque; torque bounds ±90% of physical cap = ±0.01836 N m; steps 1 and 0.5 ms; weights 0.25, 1, 4; six declared sample-selection rules. The 129-point check is the every-other-point subset of the same 257-point grid. No free controller gains are introduced. Original prediction uses its historical five 20 ms RK4 steps; original plant replay uses fifty 2 ms steps.

## Definitions and score

State x comprises arm angle theta, pendulum angle alpha measured from downward, arm velocity omega, pendulum velocity nu. Angles are rad and speeds rad/s. Let I_b, I_p, I_c and g_p be the base inertia, hinge inertia, coupling inertia and gravitational torque coefficient derived from stored masses/lengths/gravity. The inertial coefficients have units kg m² and g_p has units N m. Define total energy E, relative swing energy H and potential V, all in J:

$$
V=g_p(1-\cos\alpha),\qquad
H=\tfrac12 I_p\nu^2+V,\qquad
E=\tfrac12(I_b+I_p\sin^2\alpha)\omega^2+I_c\cos\alpha\,\omega\nu+H.
$$

The common upright-rest target E_star = 2g_p = 0.03037176 J. For candidate torque tau (N m), evaluate E and H after 0.1 s. The dimensionless objective L_lambda is half the sum of squared normalized total-energy error and lambda times squared normalized swing-energy error. Lambda is a diagnostic priority, not learned or tuned. Exact equation and interpretation are in the human report.

Define D as the inertia determinant, and s_E/s_H as the instantaneous power derivatives per unit torque, in rad/s:

$$
D=(I_b+I_p\sin^2\alpha)I_p-I_c^2\cos^2\alpha,
\qquad s_E=\omega,\qquad s_H=-I_pI_c\nu\cos\alpha/D.
$$

For errors e_E=E−E_star and e_H=H−E_star, define c_E=e_E s_E and c_H=e_H s_H. The input-dependent derivative of the current equal-weight loss is (c_E+c_H)tau/E_star². Thus the instantaneous bounded-action rule uses negative saturation when that sum is positive, positive saturation when negative, and zero if exactly zero. It optimizes this local derivative only; natural drift is independent of torque at the same state but influences finite-duration predictions. `instant` stores this rule.

## Historical objective and reproduction

Let V_end be predicted potential after 100 ms, W be predicted motor work, W_req the requested work, and P the maximum absolute arm angle at the initial state and five historical prediction endpoints. The archived score J_old is

$$
W=\tau(\theta_{\mathrm{end}}-\theta_0),\qquad
W_{\mathrm{req}}=\operatorname{clip}(0.004(E^\star-E),-0.04E^\star,0.04E^\star),
$$
$$
J_{\mathrm{old}}=\left(\frac{E^\star-V_{\mathrm{end}}}{E^\star}\right)^2
+0.02\left(\frac{W-W_{\mathrm{req}}}{0.04E^\star}\right)^2
+10000\left(\frac{\max(P-\pi,0)}{\pi}\right)^2.
$$

Theta_0 is initial arm angle, theta_end the predicted endpoint, pi is the historical arm bound in rad, and clip limits its argument to the two stated endpoints. Constants 0.004, 0.04, 0.02 and 10000 are the archived gain, work cap fraction and objective weights. The source decoder searches a 33-point torque grid plus work-root candidates. We do not claim to reconstruct that entire candidate search: we reproduce the saved action's predictions and cost terms, then compare that action with the new recommendation under the same old score. Every tested old-predictor intermediate endpoint lies within the historical arm bound, so its arm term is zero.

The archived coarse predictor and the refined counterfactual predictor are different numerical schemes, explicitly preserved. Hence the diagnostic comparison changes both objective and prediction accuracy. The original-objective comparison evaluates both actions with the same historical coarse scheme and shows the actual action preferred at all six samples; it supports the objective explanation without attributing all global disagreement to it.

## Sampling, signs and outputs

Use `start_x` and the corresponding `torque_nm` of complete 100 ms decisions, never the preceding interval's terminal torque. All 160 complete actions are analyzed. The final 20 ms action is omitted from score comparisons but remains in the gray physical record. Angles are converted to [−pi,pi); no active sample is removed from the sensitivity curve. Each of 160 initial states has 258 candidate actions (257 grid plus exact actual), and all independent pairs are integrated simultaneously. The sequential integration axis is time.

Six illustrative categories are chosen deterministically, with repeated indices excluded: first time ≥0.8 s; maximum loss gap among states with c_E c_H<0; maximum gap where instantaneous and finite-duration signs disagree; latest exactly-zero actual action (threshold 10⁻¹⁰ N m); smallest |s_E|/|s_H|; first time ≥15 s. All recorded swing sensitivities are asserted nonzero before this ratio is evaluated. Sort selected states by time and label A–F. These categories deliberately include extreme disagreements, not unbiased population samples.

Opposite-sign counts use the product of two torques <−10⁻¹² (N m)² to exclude numerical zero. “Loss gap” is actual-action loss minus best-tested loss; because the actual action is included among candidates it cannot be negative beyond numerical noise. The main best index is lambda=1. The two other weights are diagnostic priorities; no claim that one is correct for all tasks is made. There is no effort cost, so even tiny gains may recommend a nonzero action. No continuous-optimality or closed-loop superiority claim is made.

`comparison.npz` contains all starts, actions, energy endpoints, three-weight scores, historical scores, both sensitivities, error-weighted terms and natural swing drift. `decisions.csv` summarizes all decision starts, torques, sensitivities and equal-weight loss gaps. `samples.csv` and `summary.json` provide the six selected examples, changes in both energies, original and new objective values, weight sensitivity, validation residuals and selection rules. `run.log` records progress. Previous provenance includes the source archive SHA-256 and complete physical constants.

## Reproduction and visual QA

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/experiments/actuation-opportunities/decisions.py
MPLCONFIGDIR=/tmp/burst-coast-mpl PYTHONPATH=/home/yl948/.codex/skills/nature-figure/scripts .venv/bin/python scripts/experiments/actuation-opportunities/plot_decisions.py
.venv/bin/ruff check scripts/experiments/actuation-opportunities
```

Use the existing Python plotting environment and previously recorded optional PyMuPDF audit dependency. No dependency change is required. Three quantitative report figures provide complementary evidence: decision map for trajectory location and actions, sampled loss curves for action ranking, physical traces for context. PNGs are 300 dpi; editable PDF/SVG are included. Intentional static warnings concern screen-report dimensions, absence of TIFF, and 300 instead of 600 dpi. Final rendered alignment, font-floor and collision audit JSON files reside in `qa/`; inspect all final images as well. Formula and numerical analysis do not require an external literature claim.
