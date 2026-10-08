# Implemented energy-work controller contract

> Historical soft-penalty decoder contract. The encoder and upper policy remain applicable; decoder modes, coast rules and campaign schema below are archived behavior. Use the [recovery contract](decoder_recovery_contract.md) for the executed revision and [current results](../../../artifacts/rotary_pendulum/experiment-results/physical-energy-transfer-controller/interpretations/interpretation_summary.md) for performance.

## State, action and physical configuration

The current implementation is `src/rotary_pendulum/heuristic/`; the exact structure budget is in `../development_plan.md`. Both shared physical damping coefficients are zero. The existing JAX integrator advances 20 ms with RK4; one held action comprises five samples, or Δ = 0.1 s. No new dependency or learned model is used.

The physical state x contains measured/simulated arm angle θ and downward-referenced pendulum angle α, in rad, followed by their signed velocities ω and ν, in rad/s. The policy input e contains three energies in J. Constants Jₐ, J₀, J and B are configured/derived inertias in kg m², and G is the gravity coefficient in J; their definitions and values are in `../formulation_plan.md`.

$$
x=(\theta,\alpha,\omega,\nu),\qquad
e=(K_a,K_p,V),\qquad E_\star=2G=0.03037176\ \mathrm J.
$$
$$
K_a=\tfrac12J_a\omega^2,\quad
K_p=\tfrac12(J_0-J_a+J\sin^2\alpha)\omega^2+B\cos\alpha\,\omega\nu+\tfrac12J\nu^2,
\quad V=G(1-\cos\alpha).
$$

`encode(x)` preserves all leading batch axes and produces a final axis of length three. It includes the pendulum's carried motion and cross term. `policy(e, work_gain, kinetic_weight)` has no access to physical state, history or arm position. Its output w is signed commanded shaft work, not electrical energy or torque.

## Implemented policy

Let γ be the dimensionless `work_gain`, κ the dimensionless `kinetic_weight` applied only to arm kinetic energy, and wₘₐₓ the fixed work limit. The current equation is

$$
w=\operatorname{clip}\left(\gamma(E_\star-V-K_p-\kappa K_a),-w_{\max},w_{\max}\right),
\qquad w_{\max}=0.04E_\star.
$$

κ = 1 gives total-energy regulation. The selected soft-penalty baseline uses γ = 0.04, κ = 1; its decoder uses arm penalty 0.1 and zero coast lookahead. The unpenalized comparison uses γ = 0.02, κ = 1 and arm penalty zero. These differ in two settings, so their comparison does not isolate the arm penalty causally.

Waves 1–2 used γ[E★−V−κ(Kₐ+Kₚ)] before clipping; wave 3 changed to the current formula to test arm storage. Each campaign retains the executed source and hashes. The selected κ = 1 settings agree across both versions.

## Work inversion and branch choice

`decode(x,w,arm_weight,coast_steps)` receives the full physical state. Let FΔ(x,u) denote the predicted physical endpoint after constant torque u in N m. Work W is computed from arm displacement:

$$
W(x,u)=u[F_\Delta(x,u)_\theta-\theta],\quad
|u|\le U=0.00918\ \mathrm{N\,m},\quad
|W-w|\le\epsilon_W=10^{-4}E_\star=3.037176\times10^{-6}\ \mathrm J.
$$

The cap U is 45% of the physical 0.0204 N m limit. Sample 33 uniformly spaced torques; keep finite adjacent work-error sign brackets and bisect them ten times in parallel. This is a finite-resolution search: tangent roots, multiple roots within a bracket, or very narrow branches can be missed. `root_count` counts accepted brackets, not guaranteed distinct mathematical roots.

Let c = `coast_steps` be a nonnegative integer, eᶜ the energy after the held action followed by c zero-torque prediction samples, θₚ the largest absolute arm angle over the current state and that sampled path, λ = `arm_weight` the dimensionless penalty, and Θ = π/2 rad. Score each accepted branch by

$$
S(u)=\frac{(e^c_1)^2+(e^c_2)^2+(e^c_3-E_\star)^2}{2E_\star^2}
+\lambda\left[\max\left(\frac{\theta_p}{\Theta}-1,0\right)\right]^2.
$$

Subscripts 1–3 select arm kinetic, pendulum kinetic and potential energy. Choose the minimum finite score; exact ties prefer smaller torque magnitude, then positive torque. The arm penalty vanishes inside the bound. It never vetoes a torque or modifies the state. The executed action is only the held interval; prediction coast is not committed sleep.

| Diagnostic mode | Meaning and executed action |
| --- | --- |
| 0 | A resolved work root with finite score is applied |
| 1 | Exactly zero requested work: apply zero torque |
| 2 | No accepted work root resolved on the numerical grid: apply zero torque |
| 3 | Work-tolerant candidates exist but their prediction/score is invalid: apply zero torque |
| −1 | Comparison controller, without work inversion |

Mode 2 is a numerical unresolved status, not proof of physical impossibility; `unreachable_fraction` is the CSV field for it. A failed numerical bracket can also lead to mode 2, so mode 3 is not an exhaustive detector of all numerical failures. Actual work after an early terminal capture uses the shorter executed displacement. Root-tolerance auditing includes only complete 100 ms actions in mode 0.

## Exact structural counterexample

For any positive κ define Ω, a constant arm speed in rad/s, by

$$
\Omega^2=\frac{2E_\star}{\kappa J_a+J_0-J_a}.
$$

Take α = 0, ν = 0 and ω = Ω, at any initial θ. Then V = 0, Kₐ = JₐΩ²/2 and Kₚ = (J₀−Jₐ)Ω²/2. The policy requests w = 0; the decoder applies u = 0. In the exact lossless ODE both accelerations vanish, so the pendulum remains downward while θ increases at Ω. For κ = 1, Ω ≈ 12.29 rad/s and total energy equals E★ despite being far from upright.

Neither γ, the arm penalty, the squared endpoint cost nor longer lookahead can remove this exact zero-work/coast equilibrium: branch scoring is bypassed by the zero-work rule. Observed trajectories near this family explain stopping parameter-only tuning. This counterexample invalidates this particular policy/decoder combination as a general swing-up law; it does not prove that every three-energy policy is impossible.

## Evaluation and artifact schema

`evaluate` accepts existing `EnvState`, three gains, coast length, chunk decisions and mode. It records every physics sample and preserves the environment's early capture/timeout masks. JAX capture requires wrapped upright error at most 0.08 rad and speed magnitudes at most 0.15/0.20 rad/s for five consecutive 20 ms samples. Arm angle does not gate capture; excursions are nonterminal. `in_bound_capture` checks the terminal arm angle only, not that the entire trajectory remained inside. This is first 100 ms capture, not proven sustained balance.

Campaigns run a 20 s deadline, report each simulated second, and batch independent trial settings across devices. Four matched random reset strata are downward, moving, near upright and tighter upright, reused from `validation_resets`; four deterministic probes are reported separately. Pilots: seed 20261007, 16 starts per stratum, 24 settings in three eight-GPU waves. Confirmation: seeds 20261008–20261010, 64 starts per stratum, two energy settings and two baselines. Probe captures never enter the main rates.

`previous` reuses the earlier full-state heuristic with gains 0.05 and 2.0 plus its existing arm filter and 20-step prediction. This controller still contains its old centering behavior; it is a reference, not a controlled one-factor ablation. `zero` always commands zero torque. Both use the same lossless plant, resets and JAX capture predicate as the new controller.

| Record | Content / interpretation |
| --- | --- |
| Root campaign/provenance/source snapshot | Explicit JSON, JAX/device/damping metadata, source copies and SHA256 |
| `initial_states.npz` | Exact confirmation/wave-3 reset state fields and labels; earlier waves are reproduced from saved source and seed |
| Trial `trajectories.npz` | Decision-major arrays; physical states, five samples, active masks, energies, torque, requested/actual work, work-root diagnostics and terminal flags |
| Trial `episodes.csv`, `summary.csv` | Per-lane values and stratum means; boolean means are rates, numerical means are not maxima |
| `integration_audit.json` | Up to 256 evenly spaced complete actions independently replayed with NumPy 2 ms RK4; selected flattened decision/lane indices included |
| `plot_selection.json` | Median minimum-energy-cost lane per random stratum; no selection by success |
| `confirmation_episodes.csv`, `confirmation_summary.csv` | Combined seed records and explicit aggregate counts/maxima, with source paths in `confirmation_sources.json` |

NPZ arrays have no object payload. Energy balance residual is actual endpoint total-energy change minus actual shaft work, in J. It measures numerical integration error when physical damping is zero. `approach` checks a loose 0.35 rad / 2 rad/s / 2 rad/s region at any active sample; it can count a favorable initial state and is not new swing-up. Full closed-loop fine-step confirmation and long unforced holds have not been run.

## Reproduction

From repository root, using the existing `.venv` and host GPU access:

```bash
CUDA_VISIBLE_DEVICES=0,1 JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false MPLCONFIGDIR=/tmp/bcmpc-mpl PYTHONPATH=src .venv/bin/python scripts/validate_rotary_heuristic.py --config artifacts/rotary_pendulum/experiment-results/physical-energy-transfer-controller/records/campaign_confirm_20261008.json --output /tmp/energy-work-confirm
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/bcmpc-mpl .venv/bin/pytest -q tests/test_energy_heuristic.py tests/test_jax_rotary_environment.py
```

Use a new output directory for each run. One visible device is required per trial; a one-trial JSON also runs on CPU. Wave-1/2 non-unit kinetic weights require their archived source formula. Existing environment dependencies already contain JAX and plotting support; no package requirements changed.
