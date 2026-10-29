# Selected RL policy: performance and reuse

The selected stationary TD3 actor applies torque directly, with a limit of ±0.00459 newton metres and a 100-millisecond action interval. Its twin critics are retained for the learned MPC terminal value. The inference package needs no teacher, simulator, or GPU.

Start with [nominal sessions](nominal_sessions.png) and [disturbed sessions](disturbed_sessions.png). The corresponding [nominal](nominal_final_ten_seconds.png) and [disturbed](disturbed_final_ten_seconds.png) tail figures show the final ten seconds. The plotted episodes were selected in the original evaluation; packaging did not reselect favorable examples.

Each condition contains 512 independently evaluated starts and lasts 600 seconds. All sessions ended balanced. The nominal evaluation had 510/512 arm-compliant final-balance successes; the disturbed evaluation had 507/512. Nominal final-ten-second torque RMS was approximately 0.0693 millinewton metres. The prescribed arm envelope is ±135 degrees. Full episode tables retain violations rather than dropping them.

The disturbed evaluation applies state displacements at 120 and 300 seconds. Signed magnitudes are 0.05 radians in arm angle, 0.12 radians in pendulum angle, 0.2 radians/second in arm speed, and 0.35 radians/second in pendulum speed. Recovery can include a fall and renewed swing-up. These observations do not establish indefinite stability or robustness to arbitrary disturbances.

Use [population_performance.csv](../machine-scannables/population_performance.csv) for aggregate and per-stratum results. Each condition directory contains all 512 episode metrics, exact settings, original plot selections, and the three plotted sessions in a compact NPZ archive. The archive's session axis follows the order in `plot_selection.json`; those values are original archive lane numbers.

The [inference guide](../README.md) defines the public model API. The [operating contract](../machine-scannables/operating_contract.md) defines observations, reward, critic interpretation, selection evidence, and physical assumptions. Critic accuracy is not established merely by successful actor trajectories. The [manifest](../machine-scannables/manifest.json) preserves exact model architecture, physics, and weight identity.

These are actor-only 600-second results. The separate MPC bundle contains 40-second MPC results; do not merge their durations or claims. Earlier weaker-torque policies, teacher comparisons, optimizer states, and training histories are excluded.
