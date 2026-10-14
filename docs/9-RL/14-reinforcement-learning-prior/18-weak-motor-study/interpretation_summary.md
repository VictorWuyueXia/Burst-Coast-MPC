# What the torque and episode records show

**The 2.36-second result used a stronger motor than intended. RL does learn gradual energy pumping when retrained with less torque, but weak-motor capture and bounded arm motion remain unresolved.**

Start with [energy pumping at four motor limits](../../artifacts/rotary_pendulum/weak-motor-20261013/human-readables/energy_pumping_comparison.png). Each column uses the same downward initial state. Read top to bottom: pendulum angle, applied shaft torque, physical energy, and arm angle. Each column has its own time scale. Red dashed lines show torque/arm limits; green marks first upright arrival. The blue energy curve grows over repeated swings as torque decreases. The example takes 1, 3, 9 and 19 complete downward-crossing cycles respectively. This is a representative example selected by the strong policy's median duration, not a success-rate estimate.

The [complete episodes](../../artifacts/rotary_pendulum/weak-motor-20261013/human-readables/motor_limit_comparison.png) show what happens afterward: the weakest example reaches upright, loses it, rotates several times, and eventually captures. Unwrapped angles beyond 180 degrees represent physical revolutions; all odd multiples of 180 degrees are upright. [Original fast/median/slow records](../../artifacts/rotary_pendulum/weak-motor-20261013/human-readables/representative_episodes_1.png) and [additional original examples](../../artifacts/rotary_pendulum/weak-motor-20261013/human-readables/representative_episodes_2.png) include an upright-error zoom and motor-work accounting. Detailed new episode galleries are under `record-audits/<policy>/human-readables/` and include timeouts.

## Was the previous torque excessive?

The selected previous actor stays inside its declared ±18.36 mN m cap. However, that cap was increased from the earlier TD3 limit of ±9.18 mN m to match the teacher. This change made the task easier and was inappropriate evidence for the intended weak-motor benchmark. Torque limits are now explicit per experiment and checked against demonstration metadata.

In the original 256 downward records, mean first upright arrival is **1.377 s**; mean completed capture is **2.359 s**, including the required one-second hold. About **49.4% of pre-upright time** is at least 95% of the cap. Mean positive motor work is 51.18 mJ and removed work is 20.82 mJ; upright potential is 30.37 mJ. Thus the policy actively injects and then removes energy. It uses about three substantial swing reversals, not a single direct lift. The torque penalty is small: full saturation costs 0.02 reward units per second, while the explicit time cost alone is at least 1.05 per second. Fast, aggressive control is consistent with that objective.

The mean per-episode maximum mechanical-energy/work discrepancy is only 0.0224 mJ in these original records. Independent 2 ms replay of 256 complete actions has maximum pendulum-angle difference about 0.061 degrees. This supports real motor-driven energy input in the successful trajectories; it is a local integration check, not a full fine-step closed-loop certification.

## What happened after lowering torque and retraining?

Two eight-GPU waves compared 16 runs, with 15,924,736 finite online transitions. New heuristic demonstrations were generated at each lower cap and split by entire episode. Goal: strictly less than five degrees for one consecutive second, within 40 seconds. No mass, damping, action interval or success-goal relaxation was used.

Fresh paired evaluation, 256 downward starts per policy:

| Motor cap | First upright, mean | Complete pump cycles, mean | One-second capture | Capture without any arm crossing |
|---|---:|---:|---:|---:|
| 18.36 mN m, previous actor | 1.38 s | 1.00 | 100% | 100% |
| 9.18 mN m, retrained seed 0 | 2.61 s | 3.02 | 100% | 59.4% |
| 4.59 mN m, retrained with imitation | 7.45 s | 9.84 | 73.8% | 0.8% |
| 2.295 mN m, retrained with imitation | 14.21 s | 20.45 | 29.3% | 0% |

**Every downward episode in this table reaches the upright band at least once, including the weakest motor. The dominant remaining failure is capture/holding after energy buildup.** First arrival alone is not task success. Successful quarter/eighth-torque episodes complete the one-second hold after averages of 18.77/24.48 s. Mean pre-upright saturation rises to 90%/94%: a weaker motor can still use almost all its available torque while pumping gradually.

Both half-torque training seeds pass all 1,024 fresh random starts. Their downward capture averages are 3.60 and 4.73 s, but arm crossings differ sharply: 40.6% versus 100%. Checkpoint selection optimizes angle success then return, so it can retain a policy with poor arm behavior. The ±180-degree boundary is currently a soft penalty, not a failure condition.

Without continued imitation, TD3 collapses at the weaker limits. Stronger imitation weight 100 helps: quarter-torque downward success improves from its matched supervised initialization's 58.6% to 73.8%; eighth torque improves from 16.8% to 29.3%. Longer supervised training alone reaches 30.5% at eighth torque, so this does not establish a robust advantage for TD3 there. Failed transfer of the unchanged strong actor—2.3%, 0%, 0% downward success at lower caps—should not be confused with inability to learn a new policy.

One weak-motor run also fails numerically after 196,352 attempted transitions: the arm has already reached approximately −2,623 rad with −149 rad/s velocity. Replaying that action fails at 20 ms and remains finite at 2, 1 and 0.5 ms. Weak torque does not prevent runaway speed when damping is zero, travel is unlimited and an episode lasts 40 seconds. Long weak-motor trajectories also accumulate larger energy discrepancies, about 1.1–1.4 mJ on average, so their quoted rates remain results of the current 20 ms simulator.

## What is necessary next?

1. Keep the chosen cap fixed and explicit. **4.59 mN m is a useful next diagnostic setting:** roughly ten cycles before upright, with enough remaining time to expose capture problems. This is an empirical choice, not a minimum feasible torque.
2. Make arm travel a real task constraint if bounded motion is intended. Regenerate demonstrations under it; almost all current low-torque demonstrations/policies cross the soft boundary. Increasing only the penalty has not established a bounded solution.
3. Use finer internal integration while preserving the intended 100 ms control interval and one-second hold semantics, and check full closed-loop outcomes. The observed numerical failure requires a simulator fix or a physically defined termination region, not more optimizer tuning.
4. Improve demonstrations of the transition from pumping to capture: enter upright with controllable angular speed and sustain balance at the same torque limit. Then test a capture curriculum or faster feedback separately. Current data prove energy can be accumulated; they do not teach a consistently reliable bounded capture maneuver.

## Artifacts and verification

- Human figures: `human-readables/`; detailed galleries: `record-audits/`.
- All episode statistics: `record-audits/*/machine-scannables/episodes.csv`; aggregate table: [torque_comparison.csv](../../artifacts/rotary_pendulum/weak-motor-20261013/machine-scannables/torque_comparison.csv).
- Sixteen-run table: [training_runs.csv](../../artifacts/rotary_pendulum/weak-motor-20261013/machine-scannables/training_runs.csv); settings, histories, checkpoints and source snapshots: `wave-1/` and `wave-2/`.
- Fresh complete physical records and actor provenance: `heldout/`; unchanged-actor transfer: `transfer/`.
- Teacher comparisons/data: `teacher-sweep/`, `teacher-data/`; qualified demonstration arrays and manifests: `datasets/`.
- Numerical failure: [failure_replay.json](../../artifacts/rotary_pendulum/weak-motor-20261013/machine-scannables/failure_replay.json). Plot recreation: `machine-scannables/make_comparison.py`, run from the repository with its virtual environment and `JAX_PLATFORMS=cpu`.

All recorded torque bounds and capture holds were independently checked for the five audited policies. The repository suite passes 125 tests (two existing empty-array warnings); changed runtime files pass Ruff and strict mypy. All eight GPUs were released and checked idle. These experiments establish slow pumping in the current simulator, not a completed weak-motor benchmark with hard arm limits.
