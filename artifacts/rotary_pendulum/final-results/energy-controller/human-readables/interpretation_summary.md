# Constant-torque energy controller

This selected heuristic chooses one constant torque for the next 100 milliseconds. Its usable torque is ±0.01836 newton metres, with a ±180-degree soft arm limit. It predicts potential-energy gain while penalizing requested-work mismatch and arm-boundary violation. The fixed work gain is 0.004, work-error penalty is 0.02, and arm-limit penalty is 10,000. These dimensionless design weights do not rescale physical energy.

Start with [population performance](population_performance.png), then the representative sessions for [seed 20261022](validation_seed_20261022_sessions.png) and [seed 20261023](validation_seed_20261023_sessions.png). Each plot's four sessions are retained numerically with its original lane selections. These figure selections differ from the duration-based animation examples.

The two validation seeds contain 4,096 random starts in total. All captured; all 1,024 downward starts captured without crossing the arm limit. Six moving-start episodes crossed the soft limit. Eight deterministic probes, recorded separately, also captured. Do not count deterministic probes as independent random samples.

Capture requires pendulum position within 15 degrees of upright for five consecutive 20-millisecond samples before a 20-second deadline. Angular velocity is unrestricted. Recording stops at capture: these figures demonstrate capture, not sustained balance or sleep.

The [fine-plant validation](fine_plant_validation_sessions.png) integrates the plant at 2 milliseconds while leaving prediction and goal sampling unchanged. It preserves all captures on the same 2,052 starts from seed 20261022. Moving-start arm crossings change from three to two; individual trajectories can differ.

Each validation directory under [machine-scannables](../machine-scannables/) contains complete episode metrics, group summaries, exact selected settings, initial states, source provenance, numerical replay audits, plot selections, and the four plotted trajectories. No unselected parameter candidates are included.

For animations, read the [playback guide](animation_guide.md). The [animation records](../machine-scannables/animation/) contain typical and slow captures, exact downward rest, and two arm-boundary examples. Their CSV files contain active observations and geometry; their NPZ files additionally retain padded decision arrays. Apply `physics_active` before analyzing those padded samples.

Physical energy components are arm-body kinetic energy, full pendulum-body kinetic energy, and gravitational potential, all in joules. Their sum is total mechanical energy. This full-body partition differs from the hinge-relative pendulum energy used in the separate [torque-sensitivity study](../../energy-torque-sensitivity/human-readables/interpretation_summary.md).

The soft penalty does not guarantee containment. Boundary examples preserve both failure mechanisms: no bounded tested action available, and a crossing selected despite a bounded tested option. The collection changes neither action selection nor success criteria.
