# CEM–LQR execution and record contract

The complete design equations, state definitions, weight units and gain construction are in [lqr_handoff.md](../human-readables/lqr_handoff.md). The run uses the archived 4.59 mN m gain, a frozen TD3 terminal value, centering rate 10, effort rate 2 and zero added weak speed/angle costs. The unmodified saved gain is checked against current dynamics before execution. The historical pole comparison has absolute tolerance 1e−5, and every current closed-loop pole must independently have magnitude below one. The initially stricter 1e−10 comparison rejected the preflight before any simulation; the largest discrepancy was 2.738e−6, not an unstable pole.

## State and ordering

Physical state order is arm angle, pendulum angle, arm speed, pendulum speed, with angles in radians, speeds in rad/s and pendulum zero downward. The carried arrays are current physical states; previous ten-command CEM plans; a fixed PRNG key; previous local-mode flags; and consecutive qualifying-interval counts. All independent sessions share a fixed batch dimension but have independent physical states.

At decision index k, time is 0.1 k seconds. Compute wrapped upright error from the current measured state. Let q be the previous consecutive qualifying-interval count, m the previous mode flag, and D the chosen dwell divided by 0.1 s (an integer). Let r mean current absolute upright error <0.25 rad and absolute pendulum speed <2 rad/s. The current local flag is:

$$
m_{\mathrm{new}}=(m\lor(q\ge D))\land r.
$$

Dwell zero is an explicit experiment selector for plain CEM, not immediate LQR entry. Its local flag is always false. Otherwise D is five or ten in screening. Entry requires preceding complete intervals, never future samples. Handoff means new local mode and old CEM mode; release means old local mode and new CEM mode.

For CEM-mode lanes, the selected plan must pass the existing one-second nominal/refined feasibility audit. The first normalized control is executed for 100 ms. For LQR-mode lanes, calculate clipped negative state-error times the archived gain and execute at the same control rate. No LQR switch occurs within that interval. Reintegrate each chosen action on nominal 2 ms and refined 1 ms grids. Use the refined endpoint as the next measured state; archive every refined sample. Recorded `accepted` requires CEM-plan acceptance for CEM lanes, finite executed states, the strict 135° arm boundary and refinement error within [0.001 rad, 0.001 rad, 0.01 rad/s, 0.01 rad/s]. All actions must lie within normalized magnitude one. No torque filter or other controller is silently substituted.

After execution, increment q only if every refined sample has absolute upright error ≤5°, absolute arm speed ≤0.3 rad/s and absolute pendulum speed ≤0.4 rad/s; otherwise reset q to zero. Update the carried mode to the mode just executed. A release after local control restarts CEM from a fresh actor rollout; normal CEM operation shifts its preceding plan. The other two CEM seeds remain constant ±0.5 normalized torque. Whole-batch CEM is skipped only when every current lane is local.

## Run and output layout

`hybrid_run.py screen DWELL` uses eight starts; `confirm DWELL` uses 32. All have 400 decisions and 40 seconds of simulated duration. The source initial-state groups are downward, moving swing, near-upright and outward arm, coded 0–3; there are two/eight examples per group in screen/confirmation. The seed and original starts are shared with the previous CEM study. No SQP, CPU simulation benchmark or network training is performed.

Each `chunk-NNNN.npz` contains ten consecutive decisions. Per-lane arrays have leading dimensions (decision, session). `executed` adds dimensions (100 refined samples, four physical states). `action` is the actual normalized action, not an unused CEM candidate for LQR lanes. `local`, `handoff`, `release`, `quiet` and `solver_called` identify exact switching behavior. `solver_called` repeats the same whole-batch boolean across lanes; it does not imply that every lane executed CEM. `refinement_error` has four state components. `solve_s` has one synchronized batch duration per decision, excluding archival I/O.

`design.json` archives the full plant settings, gain, poles, thresholds, source hashes, device and network manifest. `starts.npz` preserves initial states and group labels. `completed.json` exists only after 400 recorded decisions. Source copies include the teacher settings. Runs are not overwritten. A technical inability to produce physically accepted finite execution is an explicit error with saved diagnostics, not an undisclosed fallback.

## Report and success flags

`hybrid_report.py` independently reconstructs all 400 switching decisions from the executed records, checking mode, handoff/release predicates, qualifying counts, whole-batch solve flags and actual LQR control values. It verifies completion, source hashes, finite states, torque/arm bounds and integration refinement. It does not infer correct switching merely from a good final score.

Hold-time scores use all 40,000 post-integration samples; the initial state earns no extra millisecond. Total hold time is sample count within ±5° times 0.001 s. Maximum continuous hold time is the longest uninterrupted accepted sample run times 0.001 s, without bridging gaps. Mean/median longest holds summarize these per-session maxima. `first_handoff_s` is the start time of the first LQR interval, not its ending timestamp. `lqr_time_s` counts actual local intervals. Null first handoff means none occurred. Final ten-second hold is a secondary descriptive score and never ends execution.

The machine-readable summary and per-session CSV are separate from human figures and interpretation. The screening choice is made after examining completed 0.5-second-dwell results; its independent 32-start confirmation may run while the slower one-second screen completes. Both screens still finish all 40 seconds. This reused-start comparison is not an independently sampled robustness certification or a proof of a global region of attraction.
