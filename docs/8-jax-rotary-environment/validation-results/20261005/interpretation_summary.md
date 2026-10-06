# JAX Rotary Environment: Measured Validation Summary

## Verdict

The show-of-concept environment is ready for the next reward-and-learning phase. All planned numerical, episode-contract, transformation, capacity, and single-GPU throughput gates passed. The legacy rotary plant and MPC path remain unchanged, and the complete repository suite passes with 88 tests.

## What the measurements show

- CPU float64 matches the NumPy derivative within 5.7e-14 and the identical 20 ms RK4 map within 1.8e-15 across 12,288 state/action cases.
- Against the 2 ms NumPy reference over a 100 ms held action, the worst CPU float64 error is 0.000096 rad for angles and 0.0027 rad/s for angular rates. GPU float32 produces essentially the same bounds. The planned limits are 0.02 rad and 0.5 rad/s.
- No between-sample arm-limit miss was observed in the 12,288-case, 100 ms search. This is strong benchmark evidence, not a continuous-time safety certificate.
- All 30,000 reset samples stayed in support and inside the arm boundary. None happened to begin inside the complete four-coordinate goal.
- The 300 random-initialization rollouts exercised terminal behavior: 270 random-torque episodes reached the arm limit, and 30 zero-torque episodes timed out. No untrained rollout succeeded, as expected.
- At batch 8,192, one L40S completed about 721,488 full 200-decision rollouts per second versus about 2,804 on CPU, a 257-times throughput ratio. Batch 65,536 completed without running out of memory.
- Sustained telemetry recorded a 97% utilization peak on GPU 0 and zero utilization on GPUs 1–7. JAX's allocator reported about 110 MiB peak use, while the driver showed about 34.5 GiB reserved through default preallocation.

## How to read the artifacts

Start with [CPU accuracy, resets, and rollouts](cpu/human-readables/accuracy_resets_rollouts.png), [GPU resource usage](gpu/human-readables/resource_usage.png), and the [CPU/GPU throughput comparison](human-readables/backend_throughput.png). Exact measurements, seeds, complete rollout tensors, resource samples, metadata, and console logs are under each backend's `machine-scannables` directory. The fixed seed is 20261005.

The reset distributions remain provisional experiment-design ranges. Their numerical support is implemented correctly, but later learning results should decide whether they provide the right task coverage. The GPU benchmark uses one card deliberately; multi-GPU sharding is deferred until a training or prediction workload demonstrates that it is useful.
