# ±135° arm travel and sustained upright balance

The arm range is now ±135°. Sustained balance is feasible with the weak motors in this simulator. The original heuristic does not reliably balance, but adding local balancing feedback and predictive braking succeeds. TD3 also sustains small upright oscillations for ten minutes at 4.59 mN m after correcting an inference-time clock problem. The 2.295 mN m TD3 runs still fail, although the augmented teacher succeeds at that cap.

## Read the records

- [Paired 300-second episodes](../../artifacts/rotary_pendulum/sustained-balance-20261013/human-readables/paired_episodes.png): the same downward initial state, four controllers, all with a 4.59 mN m cap. Read upright error, its ±10° zoom, arm travel and applied torque from top to bottom. Green means the ±5° target; red marks ±135° arm travel. Arm axes intentionally differ to expose crossings. The original heuristic and the augmented teacher both cross in this particular episode; the latter subsequently balances.
- [First 120 seconds](../../artifacts/rotary_pendulum/sustained-balance-20261013/human-readables/paired_first120s.png) shows swing-up and acquisition more clearly.
- [Ten-minute rollouts](../../artifacts/rotary_pendulum/sustained-balance-20261013/human-readables/ten_minute_balance.png) and [last-ten-second detail](../../artifacts/rotary_pendulum/sustained-balance-20261013/human-readables/ten_minute_balance_last10s.png) show continuing physical feedback, not frozen successful states. These use paired downward episode zero, chosen without searching for a favorable result.
- [Time-budget comparison](../../artifacts/rotary_pendulum/sustained-balance-20261013/human-readables/time_budget.png) separates getting a one-second hold eventually from achieving a ten-second hold.

## What changed about the arm constraint

Both JAX and the legacy MPC now use ±135° arm travel. The JAX heuristic and RL boundary penalties begin increasing at ±108°, leaving 27° to slow down. RL also pays a smooth centering cost throughout the range. Crossing remains physically possible and is recorded; the simulator does not clamp the arm angle.

New checkpoints are ranked first by ten-second success **without any arm crossing**, then angle-only success, then return. Teacher replay excludes episodes that crossed before completing their ten-second hold. Demonstration rewards are recomputed for each tested arm penalty, so the critic never receives inconsistent old cost labels.

Simply increasing the penalty was insufficient. On 64 downward starts at 4.59 mN m, original heuristic weights 10 and 100 still eventually crossed in every episode. A one-second braking prediction combined with local balancing reduced crossing to 15.6%; increasing the angle cost alone did not. Weak torque makes arm velocity and available stopping distance important before the boundary is reached.

To promote compliance, retain the early margin, compliant demonstrations and constraint-aware checkpoint selection, and include braking/recovery states in training. For a strict no-crossing requirement, use a predictive constraint filter that rejects unsafe actions while the state remains recoverable. Our tested filter is best effort and can still cross; a finite reward penalty cannot provide a hard guarantee. Merely ending an episode on crossing would need a correctly valued failure cost, otherwise a policy can escape accumulating negative rewards by failing early.

## Long holds: original heuristic, augmented teacher and RL

The original heuristic repeatedly swings through upright. At 4.59 mN m, extending the same 64 downward rollouts from 40 to 300 seconds raised one-second success from 39.1% to 100%, but ten-second success only reached 17.2%. Every episode crossed the arm range. More time provides retries; it does not repair unstable capture.

The augmented teacher uses the original swing-up policy, a locally computed linear-quadratic balancing controller near upright, and predictive braking for arm travel. It is a separate experimental controller, not an undisclosed replacement of the original heuristic. At 2.295 mN m with a two-second braking prediction, all 512 fresh random starts stayed upright throughout at least the final 500.66 seconds of a 600-second rollout, with no arm crossings. The maximum upright error over the final ten seconds was 0.00265°.

The table uses 128 fresh downward starts, excludes four deterministic probes, and checks arm crossings over the **entire stated duration**, including after the first successful hold. “Final upright” is the fraction of physical samples within 5° over the final ten seconds, averaged across episodes.

| Controller | Cap (mN m) | Duration | Ever held 30 s | Never crossed and held ≥10 s | Final upright |
|---|---:|---:|---:|---:|---:|
| Original heuristic | 4.59 | 300 s | 0.8% | 0% | 9.8% |
| Augmented teacher, 1 s braking prediction | 4.59 | 300 s | 100% | 85.9% | 100% |
| TD3, short-prefix demonstrations, original clock | 4.59 | 120 s | 100% | 99.2% | 100% |
| Same TD3, original clock | 4.59 | 300 s | 100% | 0% | 6.1% |
| Same TD3, clock input capped at 60 s | 4.59 | 600 s | 100% | 100% | 100% |
| Augmented teacher, 2 s braking prediction | 2.295 | 600 s | 100% | 100% | 100% |

Across all four initial-state groups (512 random starts), the clock-capped 4.59 mN m TD3 had one arm crossing, hence 99.8% bounded success. Every episode remained upright for at least the final 497.98 seconds; maximum final-ten-second error was 0.664°. Ten minutes of nominal simulation is evidence of sustained operation, not a proof of indefinite or disturbance-robust stability.

## What the demonstrations and TD3 experiments established

We generated 2,048 random teacher starts per setting plus probes. Qualified complete, arm-compliant ten-second demonstrations numbered 2,048 at 9.18 mN m, 1,841 at 4.59 mN m, and 2,035 at 2.295 mN m. Whole episodes were split between training and validation.

Two eight-GPU rounds trained 16 TD3 variants, each for 1,048,576 online transitions after 20,000 supervised updates. Comparisons used short prefixes through the first one-second hold, all records through ten-second completion, and a balanced subset retaining fewer repetitive settled frames. Short-prefix records are nonterminal for the ten-second task and still use return targets from the real full teacher continuation.

At 4.59 mN m, short-prefix replay reached 100% bounded ten-second success on training validation. Adding every later balancing frame produced no ten-second success in either arm-cost adjustment, and neither subsampling those frames nor reducing imitation strength fixed it. Increasing the short-prefix arm cost from 0.2 to 2 or 20 also failed to improve the best result. At 9.18 mN m, one extended-data seed learned the target, but another violated the arm constraint; more data was not uniformly beneficial. Both tested 2.295 mN m variants failed to learn ten-second holds. These are bounded experiments, not evidence that demonstration learning cannot work.

The long continuation revealed a specific additional problem: the actor observes elapsed and remaining time. After its 120-second training deadline, those inputs leave their training range. Holding only the observed clock at 30, 60 or 120 seconds kept the successful 4.59 mN m policy balanced through 600 physical seconds. No state, torque or goal counter was frozen. This paired intervention directly supports clock dependence as the cause of its late failure. At 9.18 mN m the extended-data policy succeeds with a 60-second clock cap but fails with a 30-second cap, showing that a clock-dependent policy cannot be corrected with an arbitrary universal constant.

The clock-cap candidates were compared on the same diagnostic starts; the selected 60-second result is not a separate, untouched final test. It remains an explicit experiment setting, not a silent default policy change. A stationary actor for continuous balancing, with time retained only where the finite-horizon task needs it, is the cleaner next training design.

## Why smaller torque caps hurt, beyond the deadline

1. **Capture requires braking, not just reaching the top.** The original heuristic arrives upright with mean pendulum speeds of about 6.93, 5.10, 3.82 and 2.74 rad/s as caps decrease from 18.36 to 2.295 mN m. Arrival speed falls, but control authority falls too. An illustrative local stopping estimate for 2 rad/s grows from 1.72° to 13.75°, exceeding the ±5° capture band at the weakest cap. Gravity and coupling change during motion, so this is an explanation of the need for earlier braking, not a rigorous feasibility bound.
2. **Swing-up, arm braking and local balancing require different feedback.** A single short-horizon height-seeking decoder can keep injecting energy or acquire upright with too much speed. The augmented teacher shows the same weak plant can capture and settle when these phases are handled properly.
3. **RL learns a distribution-dependent controller.** Extra quiet frames alone did not teach robust acquisition/recovery. The policy depends on clock and hold-memory features, and its state feedback changes across those inputs. The weakest-cap teacher succeeds where TD3 fails, isolating the remaining gap to learning/control design rather than demonstrated physical impossibility.
4. **Successful RL holding is still inefficient.** The clock-capped 4.59 mN m actor oscillates with roughly 0.66° peak error and about 2.95 mN m mean absolute torque in the final ten seconds, switching torque sign every 100 ms. This respects the cap but uses about 64% of it continuously. The augmented teacher settles with almost zero torque. A better RL target should include quiet balancing and disturbance recovery, not only angle occupancy; the current result is not yet suitable evidence for burst/coast efficiency.

We also repaired a reproduced numerical failure: each 20 ms observation interval now uses ten 2 ms RK4 substeps. The actuator still updates only every 100 ms. This removes the demonstrated integration blow-up without increasing motor authority; old coarse-step results are not a controlled comparison to this campaign.

## Artifacts, validation and next step

The full archive is `artifacts/rotary_pendulum/sustained-balance-20261013` from the repository root. `teacher-1/2/3` contain controller sweeps, `teacher-data` contains generation records, `datasets` contains cap/cost/scheme-matched replay and manifests, `wave-1/2` contain checkpoints and training histories, and `heldout` plus `clock-diagnostic` contain live trajectories. The top-level `machine-scannables` folder contains comparison CSVs, continuation checks, reward-parity checks, capture-authority estimates, policy probes and the figure reproduction script. Each launched wave includes settings, source snapshots, hashes and exit codes.

All 128 repository tests passed, changed runtime modules passed strict type checking, and changed code passed lint and whitespace checks. Sampled demonstration transitions matched actual reward/state/terminal reconstruction; maximum reward discrepancy was below 6e-8. All experiment processes finished and released the eight GPUs. Chinese companions are ignored by Git as requested.

Next, train a stationary continuous-balancing actor using targeted capture, braking and perturbed-upright recovery demonstrations, while measuring full-duration arm compliance and torque effort. Keep the successful augmented teacher as a feasible baseline. Neither a longer timeout nor a larger arm penalty alone addresses the observed remaining failures.
