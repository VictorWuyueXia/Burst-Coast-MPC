# Phase 10 Interpretation

Current code implements [version 3 dense reward](dense_reward_proposal.md): five direct terms, no shaping potentials, nonterminal arm excursions, and early termination after 100 ms of successful holding. The first dense-reward campaign completed 25 trials and 236,978,176 transitions on GPUs 0–3. No greedy policy passed the local-capture gates. SiLU Q endpoints modestly improved a 0.3 s planner in both paired seeds, but remain diagnostic. Read the [campaign analysis and validation plots](../../artifacts/rotary_pendulum/q-prior-v3/campaign/human-readables/interpretation_summary.md) and [experiment lineage](../../artifacts/rotary_pendulum/q-prior-v3/campaign/human-readables/experiment_lineage.md). Implementation structure and verification are in [dense_reward_development.md](dense_reward_development.md). The following action-resolution discussion is historical version-2 context.

The version-1 learner failed because maximum torque held for 100 ms was too coarse for capture and drove repeated pendulum rotations. The new contract keeps the physical clip unchanged but gives the learner a 45% pump pair and a 2% fine pair.

Use normalized `±0.45`, equal to `±0.00918 N m`, for energy pumping and removal. In the fixed downward-start diagnostic, 45% reached mean peak energy ratio `0.919` without arm violations and kept mean peak pendulum magnitude to `1.83 rad`. The 60% candidate brought back multi-rotation behavior, reaching mean `10.54 rad` and 95th percentile `24.88 rad`.

Use normalized `±0.02`, equal to `±0.000408 N m` at the current physical limit. A `±0.1` action was measured and rejected for upright hold because one decision creates roughly `0.68 rad/s` arm speed and `0.80 rad/s` pendulum speed from rest.

The revised success range is:

- arm angle: `±0.08 rad`;
- wrapped pendulum upright error: `±0.08 rad`;
- arm speed: `±0.15 rad/s`;
- pendulum speed: `±0.20 rad/s`;
- dwell: five consecutive 20 ms samples, still `100 ms` total.

“Tight” is the local reset/evaluation box `±[0.08,0.12,0.15,0.30]` around upright in `[theta,beta,omega,nu]`. It is deliberately larger than the success box. “Wave” is an experiment round of up to four independent trials run in parallel, one trial per physical GPU 0–3.

The fine- and pump-resolution plots and their CSV/JSON inputs are under `artifacts/rotary_pendulum/action-resolution-study/`. Phase-9 validation plots copied into the handoff show falling value loss alongside almost no held-out success; this means value fitting improved while task completion did not.

This revision has been implemented and trained, but no checkpoint has qualified as a successful prior. Version-1 checkpoints are structurally incompatible because they contain three outputs rather than five. Version-2 checkpoints use different reward and terminal semantics.
