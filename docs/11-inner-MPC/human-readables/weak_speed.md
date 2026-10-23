# Weak speed penalties: 40-second CEM experiment

This replaces the added weak angle-tracking terms. The broad arm-centering cost (rate 10) and torque effort (rate 2) remain. The frozen TD3 terminal value, one-second prediction horizon, 100 ms replanning period, physical model and hard 135° arm boundary are unchanged. SQP is not run in this experiment.

Every session runs for 40 seconds, irrespective of performance. The main measurements are total upright time and longest continuous upright time, both using a ±5° pendulum-angle band. Arm angle and angular speed are not additional conditions on these scores. Thus a slow passage through the band contributes some time; a long uninterrupted hold is the stronger balancing measure. We also report sensitivity to ±2° and ±10°.

## A bounded local speed preference

Let alpha be the simulated pendulum angle in radians, zero down and pi upright; omega and nu are simulated arm and pendulum speeds in rad/s. The target speeds are zero. Set sigma = 5 pi/180 radians as a chosen angular localization width, and reuse the original RL upright-speed widths s_omega = 0.3 rad/s and s_nu = 0.4 rad/s. These are local balancing scales, not the much larger network-observation normalizers. Epsilon is the tunable speed-cost strength in objective units/s.

$$
g(\alpha)=\exp[-(1+\cos\alpha)/\sigma^2],\qquad
p(v;s)=1-\exp[-v^2/(2s^2)],
$$

$$
\ell_{\mathrm{speed}}(\alpha,\omega,\nu)
=\frac{\epsilon}{2}g(\alpha)
\left[p(\omega;s_\omega)+p(\nu;s_\nu)\right].
$$

Each speed penalty is approximately quadratic near zero speed and saturates at one. Their average keeps the entire added rate between zero and epsilon. The angle only gates the speed penalty; there is no added angle-error penalty. At zero speeds the addition is zero at every angle. Far from upright the gate approaches zero. This formulation can encourage avoiding the upright region when speeds are high, because leaving the region reduces the gated penalty; this is an explicit property to examine, not a hidden capture guarantee.

Let k index the ten planned 100 ms control intervals, j index the fifty 2 ms prediction nodes in each interval, and gamma = 0.995 be the inherited per-control-interval discount. The trajectory states are computed by the existing GPU rollout. Add the following to the effort, broad arm-centering and frozen twin-Q terminal objective:

$$
J_{\mathrm{speed}}=0.002\sum_{k=0}^{9}\gamma^k
\sum_{j=1}^{50}\ell_{\mathrm{speed}}(\alpha_{k,j},\omega_{k,j},\nu_{k,j}).
$$

Its maximum one-second contribution is 0.977797 times epsilon. We screen epsilon = 0, 0.01 and 0.1, then inspect results before confirmation. Near upright, if both speeds equal one tenth of their respective widths, the added rate is about 0.00499 times epsilon. The frozen critic is not retrained.

## Implementation and validation

The existing `evaluate` function computes the two speed contributions and archives them separately. Its only added settings are `speed_cost_per_s` and `speed_width_rad`; existing `upright_widths` supplies speed normalization. The old angle-study implementation remains in its immutable experiment source archives. New preparation uses speed strength zero by default.

`speed_run.py` contains one compiled `advance` function and a straight-line execution/archive loop. It batches independent sessions on GPU, keeps the existing CEM budget (three seeds, 1024 samples per seed, 32 elites, 16 generations, initial standard deviation 0.5), and prints progress every simulated second. It checks GPU occupancy before initialization and refuses occupied devices. Numerical infeasibility remains an explicit execution error, never a controller substitution.

Screening uses the first eight archived starts; confirmation uses all 32, with eight starts each from downward, moving swing, near-upright and outward-arm groups. Both last 40 seconds. Starts and CEM random seeds are paired within each round. These are reused benchmark starts, not an independent held-out sample; changing batch size changes random draws between rounds.

Milestone 1: GPU cost reconstruction, boundedness, zero-speed and zero-strength checks, and CEM acceptance pass. Milestone 2: every run has exactly 400 accepted decisions per session and 40,000 finite executed 1 ms states within the arm boundary, with passing rollout-refinement checks. Milestone 3: `speed_report.py` produces per-session scores, paired differences, tolerance sensitivity, full-session plots and a written conclusion. No new classes, wrappers or controller fallback are introduced.

For measurement, let b_i equal one if the absolute wrapped upright angle error at the end of executed millisecond i is at most the chosen tolerance, and zero otherwise. The 40,000 post-integration samples exclude the initial state, preventing an extra millisecond of credit. Total hold time is 0.001 times the sum of b_i; maximum continuous hold time is 0.001 times the longest consecutive run of ones. Both are discrete-grid measurements, with no gap bridging or early termination.

Results and reproduction records belong under `artifacts/rotary_pendulum/inner-mpc/20261020-speed-*`. Human-readable summaries/figures are separate from detailed machine-scannable data and source snapshots.

Screening completed with no mean improvement at 0.01 or 0.1. Confirmation therefore retains those values, adds the intermediate 0.03 and compares all three against zero on all 32 starts, all for 40 seconds. This is parameter refinement within the same bounded formulation, not a change of controller or solver.

Completed [results](../../../artifacts/rotary_pendulum/inner-mpc/20261020-speed-confirm-summary/interpretation_summary.md): 0.1 gives small mean improvements (+0.422 s total, +0.264 s longest hold) in the 32-start comparison, but regresses in screening and does not consistently improve individual starts. Keep the default strength at zero; 0.1 is the best tested mean-score experimental candidate, not an established optimum. All 152 sessions completed.
