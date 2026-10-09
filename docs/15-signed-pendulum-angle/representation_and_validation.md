# Signed pendulum position: representation and validation

Pendulum position is measured from downward and stored in **[−180°, 180°)**, or **[−π, π)** radians. Downward is 0; upright can be labeled −180° or +180°. Stored values use −180° for the exact boundary so one physical orientation has one canonical value. Arm angle remains unwrapped and angular velocities remain signed. The pendulum's direction is carried by angular velocity, not by its position sign alone.

## Conversion and goal

Let a be an input pendulum position in radians, measured or loaded from a record. Let α be its canonical stored representation, computed by positive-remainder modulo. The constants π and 2π are the half-turn and full-turn angles; there is no tuning parameter:

$$
\alpha=((a+\pi)\bmod 2\pi)-\pi.
$$

For example, old 350° becomes −10°, old 190° becomes −170°, and old 180° becomes −180°. The conversion preserves physical orientation, velocities, motor torque and mechanical energy. Apply wrapping to completed integration steps, not to the intermediate Runge–Kutta slope states. Geometry and dynamics use periodic sine and cosine.

Let β be the signed angular error from upright, computed from α, and ε be the retained user-specified angle tolerance of 15 degrees, converted to π/12 radians:

$$
\beta=\operatorname{atan2}(\sin(\alpha-\pi),\cos(\alpha-\pi)),\qquad |\beta|\leq\varepsilon.
$$

This is the existing circular goal, now displayed as two bands: [−180°, −165°] and [165°, 180°). Capture still requires five consecutive 20-millisecond samples; it adds no speed condition. The controller still predicts one constant-torque 100-millisecond action. Torque capacity, arm soft limits, energy definitions and controller weights are unchanged.

## Implementation scope and structure budget

Modify existing reset and complete-step calculations in NumPy, JAX and CasADi; normalize the independent tight-upright RL reset and observation-to-state reconstruction as well. Existing evaluator, writer and plotting functions retain their responsibilities. Update the existing export and analysis workflows to convert archived inputs before writing current-format records. No new class, wrapper, helper function, runtime parameter or configuration switch is needed. Local arrays are the same state, trace, angle and plot arrays already used by those workflows.

All new state arrays and pendulum degree fields use the signed convention. NumPy simulation artifact format is version 4; portable animation records use schema version 2 and explicitly record coordinate conversion. Plot lines break at ±180° to avoid a false line across the full axis. Upright goal shading appears at both edges. RL observations may continue to use sine and cosine, which remain continuous across this coordinate boundary.

## Existing records

The [portable animation bundle](../../artifacts/rotary_pendulum/experiment-results/single-constant-torque-action-controller/portable-animation-records/interpretation_summary.md) is regenerated in the signed convention, including initial states, per-step CSV, per-decision NumPy arrays and its checksummed manifest. Existing source snapshots and original run archives retain their historical coordinates and provenance. They are not current-format logs. Current readers normalize those sources when regenerating derived records and figures.

The [current controller study analysis](../../scripts/experiments/single-constant-torque-action-controller/analyze_study.py) regenerates the selected validation figures with signed axes and two goal bands. Coordinate conversion is not a new simulation experiment. The earlier capture statistics remain measurements of the archived runs, not a claim that a new full Monte Carlo run was performed.

## Validation milestones

1. **Numerics:** crossing downward in both directions and crossing upright in both directions agree between NumPy, compiled JAX and CasADi. Repeated full turns remain canonical. Equivalent angles preserve physical energy, controller decisions and phase coordinates.
2. **Goal and reset:** both representations of upright capture; downward does not. Random resets remain in the specified physical distributions, now in signed coordinates. Goal sampling remains unchanged.
3. **Artifacts:** new arrays, settings and figures report signed position; live plot limits cover both upright edges. Exported coordinates reproduce the original mechanism geometry and all non-angle trace fields remain unchanged. Manifest checksums identify the converted bundle.
4. **Regression:** run the repository tests on CPU, lint changed Python, regenerate current study figures, and inspect a representative plot. No GPU or controller tuning is required for a representation change.

The development history records execution results once these checks finish. Historical documents describing the old 0–360° format should be read as historical; this document and the current controller specification define the active convention.
