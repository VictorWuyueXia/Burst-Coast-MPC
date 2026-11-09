# Orbital phase guided energy bursts

Implement the handover as a separate controller for the existing nonlinear Furuta simulator. The controller estimates where the pendulum is on its natural orbit, tests short torque pulses near two prescribed phases, and commands zero torque between interventions. It tests the predicted arm-angle excursion before it applies a pulse. A separate brake can interrupt a pulse, and the existing local linear-quadratic regulator design takes over after the full state settles near upright.

The main comparison is against the official MPC+RL controller on the same starts, at the same 4.59 mN·m torque cap and ±135° arm bound. An ungated version of the new pulse policy isolates the effect of phase timing. A gated version without screening isolates the effect of prediction and braking. The existing energy-work heuristic provides a second reference.

The first decision is whether phase gating earns its place. Compare full-state capture and arm-bound outcomes before interpreting reductions in powered time or work. Report starts that fail to reach the local capture region, including trajectories that stall near the separatrix, and distinguish a rejected candidate set from an observed boundary crossing. Finite rollout checks provide numerical evidence on the tested trajectories. The branch should proceed to a period atlas only if the controller results justify it and timing records identify quadrature as a material cost.

From the repository root, run the CPU comparisons at the nominal trial settings with the project environment. The output directory must be new. Use `evaluate(starts, variant, tuning, output)` in the runner for a single variant or to replay settings from a saved `design.json`.

```powershell
$env:JAX_PLATFORMS = 'cpu'
$env:JAX_ENABLE_X64 = 'true'
$env:PYTHONPATH = "$PWD/src"
.venv/Scripts/python.exe scripts/experiments/hamiltonian/run.py artifacts/rotary_pendulum/hamiltonian/new-cpu-run
```

- [Development plan](human-readables/development_plan.md)
- [Implementation and validation contract](machine-scannables/controller_contract.md)
- [Original handover](../12-Phase_energy.md)
