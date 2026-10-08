# Model, controller, experiment and record specification

## State and primitive physical parameters

State order is `[theta, alpha, omega, nu]`: arm angle θ in unwrapped radians, pendulum position α in [0, 2π) radians, arm angular velocity ω and pendulum angular velocity ν in signed rad/s. Downward is α = 0; upright is α = π. Human pendulum plots use [0°, 360°).

The configured model parameters are arm mass mₐ = 0.095 kg, arm length r = 0.085 m, pendulum mass mₚ = 0.024 kg, pendulum length l = 0.129 m, and gravity g = 9.81 m/s². These are defined simulation inputs in `src/rotary_pendulum/configs/physics.yaml`, not fitted values from these experiments. Both damping coefficients are zero. Links are modeled as uniform rods.

Define the computed pendulum center-of-mass distance c, arm inertia Iₐ, pendulum center-of-mass inertia I꜀ and pendulum hinge inertia J:

$$
c=l/2,\qquad I_a=m_a r^2/3,\qquad I_c=m_p l^2/12,\qquad J=I_c+m_p c^2.
$$

Distances are in metres; inertias are in kg m². The code derives these from the configured masses and lengths.

## Physical energy and work

Kₐ and Kₚ below are the physical kinetic energies of the arm and pendulum bodies. V is gravitational potential, with zero at downward. E is their total, all in joules.

$$
K_a=\tfrac12 I_a\omega^2,
$$

$$
K_p=\tfrac12(m_p r^2+J\sin^2\alpha)\omega^2
+m_p r c\cos\alpha\,\omega\nu+\tfrac12J\nu^2,
$$

$$
V=m_pgc(1-\cos\alpha),\qquad E=K_a+K_p+V.
$$

The coupling term may be negative; the full pendulum kinetic energy is nonnegative. An independent test reconstructs it from center-of-mass translation plus rotation about the center. Specifically, its squared center velocity is the sum of the squares of `−c sin(alpha) omega`, `r omega + c cos(alpha) nu`, and `c sin(alpha) nu`; its squared perpendicular rotation rate is `nu² + omega² sin²(alpha)`. This verifies body-energy allocation rather than merely checking an algebraically duplicated total.

The physical factors ½ are retained. There is no adjustable coefficient multiplying either body's energy. Multiplying all joule-valued components by 1,000 in a figure is a unit conversion to mJ.

Define the upright-rest target E★, computed from the same gravity and geometry:

$$
E_\star=2m_pgc=0.03037176\ \mathrm{J}.
$$

Let u be constant motor torque in N m during one action, and let Δθ be arm angular displacement during that action. W is motor work in joules. The lossless governing model satisfies:

$$
W=u\Delta\theta,\qquad \dot E=u\omega.
$$

`energy_components` independently checks total kinetic, potential and total energy. `encode` returns `[K_a, K_p, V]`. The NumPy observation fields `kinetic_energy_j` and `energy_j` now mean total kinetic and total mechanical energy respectively. NumPy artifact format version 3 records that definition. Older MPC/RL hinge-relative swing energy remains a named control quantity, not full body energy; historical archives are unchanged.

## Work request and exactly one action prediction

Define k = `work_gain`, a dimensionless feedback gain chosen by the parameter study. Define Wₘₐₓ = 0.04 E★, a fixed design bound on the *requested* work, not a physical actuator bound. The requested motor work Wᵣ is:

$$
W_r=\operatorname{clip}\bigl(k(E_\star-E),-W_{\max},W_{\max}\bigr).
$$

The physical actuator ceiling is ±0.0204 N m. The controller uses ±0.01836 N m. The numerical action set starts with 33 evenly spaced torque values over this usable interval. Adjacent values whose predicted work brackets Wᵣ receive ten bisection iterations. Up to 32 resulting torque values supplement the grid. Invalid brackets are masked, not replaced by an alternative controller.

Every such prediction starts at the current state and holds its torque constant for five RK4 steps of 0.02 s each. Total prediction length is always 0.1 s. Root refinement repeats this same prediction; it does not extend simulated time. No future controller or braking sequence is appended. Selected torque is held for the next 100 ms, unless the episode terminates sooner at a sampled capture.

For one tested torque u, let V₊(u) be predicted potential at 100 ms, W₊(u) its predicted motor work, and Θₘₐₓ(u) the maximum absolute arm angle across the initial state and five predicted samples. Define dimensionless work and arm penalties λw and λθ. The scored quantity C is a control objective, not physical energy:

$$
C(u)=\left(\frac{E_\star-V_+(u)}{E_\star}\right)^2
+\lambda_w\left(\frac{W_+(u)-W_r}{W_{\max}}\right)^2
+\lambda_\theta\left(\frac{\max(\Theta_{\max}(u)-\pi,0)}{\pi}\right)^2.
$$

The first term favors elevation. The second regulates motor work relative to the unweighted total-energy target. The third is zero throughout ±180°, including the boundary, and positive outside. There is no 0.05 rad margin, candidate rejection by arm speed, or immediate arm-kinetic-energy penalty. Finite penalties permit violations; they do not promise containment. A tie uses smaller absolute torque, then positive torque to resolve exact symmetry. Nonfinite inputs produce an explicit numerical failure rather than fallback torque.

The selected runtime parameters are **k = 0.004, λw = 0.02, λθ = 10,000**. They were tuned, not derived physical constants. Wₘₐₓ, the grid size, ten bisections and a work-match diagnostic tolerance of 0.0001 E★ are also numerical/design choices. The tolerance only labels work matching; it does not constrain speed or exclude other scored torques.

The slowing rationale is the identity Kₐ + Kₚ = E − V. If E and V both tend to E★, both bodies' kinetic energy tends to zero. Finite-step action ranking is not a proof that this limit is reached. The requested ±15° capture band permits residual kinetic energy and does not test sustained stabilization.

## Implementation structure and verification milestones

Existing functions remain the implementation: `energy.encode` computes body energies; `energy.policy` requests motor work; `decoder.decode` chooses torque using its local `predict`, `advance` and `bisect` calculations; `evaluation.evaluate` executes and logs; `artifacts.write_artifacts` writes tables and physical-unit figures; `validate_rotary_heuristic.main` runs independent settings in parallel. No controller classes, wrappers or configurable prediction horizons were added. The only controller tuning parameters are the three quantities above.

The analysis file `analyze_study.py` is one direct workflow, with no new functions/classes or CLI. Its objects are loaded trajectory arrays, episode/summary rows, numerical audit rows, first-crossing rows, selected-lane records and Matplotlib figures. It recomputes labels and plots rather than fitting a model.

Milestone success flags:

1. Body energies equal the rigid-body COM calculation; their sum equals mechanical energy, and its derivative equals motor power: passed.
2. Predicted endpoint energy, arm speed, peak arm angle and work match exactly five constant-torque steps; speeds above the old cap are accepted; no inward angle penalty: passed.
3. Paired tuning, separate validation seeds, full-angle logs, sampled goal reconstruction and soft-boundary flags: passed. Zero crossings is not a success requirement for this soft constraint; observed crossings are reported.
4. Independent NumPy 2 ms action audits, isolated 2 ms closed-loop plant run, repository tests, plotting inspection and bilingual reports: passed.

## Reset distributions and sample accounting

Each parameter comparison reuses identical initial arrays. Uniform draws are in the following ranges before wrapping the pendulum position. All angle bounds here are radians and all velocities rad/s.

| Group | Arm angle | Pendulum angle | Arm velocity | Pendulum velocity |
|---|---|---|---|---|
| downward | ±0.20 | ±0.20 about downward | ±0.5 | ±0.5 |
| moving | ±0.50 | random sign times [0.40, 2.60] | ±2 | ±6 |
| near | ±0.25 | π ± 0.25 | ±1 | ±1 |
| tight | ±0.08 | π ± 0.12 | ±0.15 | ±0.30 |

Fixed states `[theta, alpha, omega, nu]` are `[0,0,0,0]`, `[0,π,0,0]`, `[1.45,0.4,2,0]` and `[-1.45,-0.4,-2,0]`, with the last pendulum angle wrapped into [0,2π). They are not extra random samples or tests of arbitrary wide-arm initial states.

Five development stages each test four settings on the same 256 random states plus four fixed states, seed 20261020. `confirmation.json` compares four settings on 2,048 random states plus four fixed states, seed 20261021; this stage still informed selection. `validation_20261022.json` and `validation_20261023.json` each compare the two selected neighboring settings on another 2,048 random states plus four fixed states. Those two seeds are final validation, not gain selection. The fine plant repeats the selected setting on seed 20261022. This gives 23,668 evaluations on 6,400 distinct random states plus repeated fixed checks. A separate 260-case default-command check is outside this comparison count.

A capture is five consecutive post-integration samples in 165–195°; elapsed samples are 20 ms apart. No speed or arm-position condition enters capture. Failed samples reset the count. Episodes stop at capture or 20 s. A crossing flag records any active sample at or beyond ±180° and does not terminate the episode. Equality at the boundary is flagged even though the soft overshoot penalty is zero there.

## Records and schemas

All current state arrays use `[theta_rad, alpha_rad, omega_rad_s, nu_rad_s]`. Let D = decision slots (200), N = episode count, and S = physics samples per action (5). Terminal padding is retained and must be masked with `physics_active` or `elapsed_s`.

| Field | Shape | Meaning |
|---|---|---|
| `start_x`, `x` | D × N × 4 | Before/after applied action; angles rad, velocities rad/s |
| `physics_x` | D × N × S × 4 | Every 20 ms sampled state |
| `physics_active` | D × N × S | True only for executed samples |
| `pendulum_angle_deg` | D × N × S | Actual pendulum position in [0,360) |
| `energy_j`, `predicted_energy_j` | D × N × 3 | Actual end / predicted full-100 ms `[K_a,K_p,V]` in J |
| `torque_nm` | D × N | Applied torque; zero in completed padding |
| `requested_work_j`, `predicted_work_j`, `work_j` | D × N | Requested, full-action predicted, actually delivered motor work |
| `energy_balance_j` | D × N | Actual mechanical energy change minus delivered work |
| `predicted_terminal_arm_speed` | D × N | Diagnostic signed rad/s at 100 ms; never an admissibility cap |
| `predicted_peak_arm_rad` | D × N | Maximum absolute predicted arm position |
| `root_count`, `bounded_count` | D × N | Resolved work matches / tested torques staying within sampled ±π |
| `selected_potential_cost`, `selected_work_cost`, `selected_arm_limit_cost` | D × N | Separate dimensionless summands of C for the chosen torque |
| `mode` | D × N | 0 work matched; 1 work differed; 3 numerical failure; no recovery mode |

The aggregate `episodes.csv` retains every evaluation. `summary.csv` reports each stage/setting/group, counts, median duration, median peak/final absolute arm speed, 95th percentile final absolute speed, maximum absolute arm angle and median final physical total energy. `clean_captures` excludes episodes with any crossing. `boundary_events.json` records the first crossing for the selected setting in the larger comparison and final validation. `representation_checks.json` verifies goals, crossings, angle range and complete-action prediction/execution agreement. Fine-plant prediction/execution differences are expected because only the plant's integration changed.

`integration_audits.json` contains 256 complete held-action checks per case, hence 7,424 checks. It replays each from its recorded start with NumPy at 2 ms. Angular differences use the shortest circular difference for pendulum position. `records/` retains per-run summaries, settings and source-hash provenance. Selected representative and crossing trajectories are in `representatives.npz`; the `lanes` array maps their compact axis back to original lane numbers.

Representative figures use each group's median minimum squared energy-distance diagnostic; the legacy diagnostic is half the sum of squared deviations of `[K_a,K_p,V]/E★` from `[0,0,1]`. It is a dimensionless sorting statistic, not physical energy or the new control objective. Multiplying that statistic by any positive constant leaves the selected lane unchanged. Energy curves themselves use all active 20 ms states and unweighted body energies in mJ. The arm-speed boxplot retains outliers.

## Reproduction and source scope

Use the virtual environment specified by `pyproject.toml`; no dependencies were added. Before every GPU invocation, inspect all GPUs with `nvidia-smi`. Stop if more than four are occupied; use only idle devices among 0–3. One visible GPU is needed per setting in these configurations.

For example, after confirming GPUs 0–3 are idle:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 JAX_PLATFORMS=cuda JAX_ENABLE_X64=true \
  XLA_PYTHON_CLIENT_PREALLOCATE=false MPLCONFIGDIR=/tmp/bcmpc-mpl \
  .venv/bin/python scripts/validate_rotary_heuristic.py \
  --config artifacts/rotary_pendulum/experiment-results/single-constant-torque-action-controller/records/confirmation.json \
  --output artifacts/rotary_pendulum/experiment-results/single-constant-torque-action-controller/raw-runs/confirmation
```

Output directories must be new. Repeat for each saved stage configuration, using two idle GPUs for each final-validation configuration. Run `fine_plant.json` with one GPU in an isolated source copy after applying `fine_plant.patch`; leave the current runtime untouched. That patch integrates only the plant with ten 2 ms steps per existing 20 ms environment sample. The decoder continues using a single 20 ms prediction step and the same 100 ms action horizon.

After all stage runs exist, rebuild compact records and human plots on CPU:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' MPLCONFIGDIR=/tmp/bcmpc-mpl \
  .venv/bin/python scripts/experiments/single-constant-torque-action-controller/analyze_study.py
```

Complete raw traces, resolved initial states and source snapshots are under the gitignored `artifacts/rotary_pendulum/experiment-results/single-constant-torque-action-controller/raw-runs/`. The pre-change analytical modules are preserved in its `before/` directory. Historical 400 ms recovery and energy-weight studies are evidence about an older controller and require their saved sources, not the new API.
