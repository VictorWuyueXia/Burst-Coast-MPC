# Introduction literature and claim ledger

Verified on 2026-10-29. This is the detailed drafting record for the [introduction plan](../intro_formulation_paragraph_plan.md), not manuscript prose. Use primary sources for technical claims. Suggested citation keys are drafting identifiers; no bibliography entries have been inserted.

## Applications

| Key and primary source | Evidence location | Supported use and boundary |
|---|---|---|
| app-space: [Ariane 6 User's Manual for Multi-Launch Service, July 2021](https://www.arianespace.com/app/uploads/sites/4/2024/10/MLSs-users-manual-ed0.0.pdf) | Launcher description, Vinci restart capability; search “4 ignitions.” | The specified upper stage has a finite ignition capability. Use qualitatively to motivate intervention count; omit the numerical limit in the introduction and do not generalize to all spacecraft engines. |
| app-coast: [Ariane 6 upper-stage development paper, EUCASS 2023](https://www.eucass.eu/doi/EUCASS2023-873.pdf) | Abstract describing main-engine ignitions with long coasting phases. | Supports the burn/coast pattern. This is launch-vehicle evidence, not a demonstrated controller application in this project. |
| app-start: [ISRO vacuum ignition trial](https://www.isro.gov.in/LVM3_CE20_vacuum_ignition_trial_cryogenic_engine_successful.html) | Description of the multi-element igniter and turbopump startup alternatives. | Supports hardware-specific startup requirements. It does not by itself prove a consumable charge is expended at every restart. |
| app-sail: [NASA solar-sail technical discussion](https://www.nasa.gov/podcasts/small-steps-giant-leaps/small-steps-giant-leaps-episode-74-solar-sails/) | Technical explanation that force depends on sail orientation and that misalignment creates attitude torque. | Supports orientation-dependent environmental forcing. Sparse reorientation is our illustrative design motivation, not a reported universal operating practice or zero-force coast. |

**Example selection:** Keep propulsive maneuvering and a brief qualified solar-sail example. The sailboat analogy remains a substitute, not a third application. Our zero-command model can include natural environmental forces, but mapping a held sail configuration into that model requires an explicit state and actuator definition. No application model is added by this plan.

## Five method families

### bang-time — Minimum-time bang–bang control

**Primary source:** D. Liberzon, *Calculus of Variations and Optimal Control Theory: A Concise Introduction*, Princeton University Press, 2012; [author's text, Section 4.4.1](https://liberzon.csl.illinois.edu/teaching/cvoc/node85.html).

**Credit:** The bounded double-integrator example derives a minimum-time input using extreme admissible values and a state-dependent switching curve.

**Comparison:** Minimum-time design answers rapid transfer. Its objective alone does not charge enabled time, restarting, or feedback decisions. Do not assert that all nonlinear minimum-time problems have nonsingular bang–bang solutions or that all bang–bang control chatters. The introductory wording needs no double-integrator equations.

### hands-off — Maximum hands-off control

**Primary source:** M. Nagahara, D. E. Quevedo, and D. Nesic, [*Maximum Hands-Off Control: A Paradigm of Control Effort Minimization*](https://arxiv.org/html/1408.3025v2), author manuscript, revised 2015. Relevant parts: finite-horizon support objective and normality qualification; Section VI-B, Algorithm 1, and the practical-stability analysis.

**Credit:** Minimizes nonzero-input duration, establishes conditional L1 equivalence, and supplies a self-triggered feedback construction.

**Comparison:** Support duration does not itself penalize the number or placement of on/off transitions. The cited next-update rule uses a minimum-time/sparsity construction, rather than our comparison of timing-conditioned inner solves with enabled/restart charges. Do not claim that sparse control lacks feedback timing or stability analysis. Later minimum-switch variants exist, so this is a contrast with the cited objective, not a field-wide novelty claim.

### nonuniform-mpc — The verified Forbes reference

**Primary source:** F. Ahmed, L. Sobiesiak, and J. R. Forbes, [*Model Predictive Control of a Tandem-Rotor Helicopter With a Nonuniformly Spaced Prediction Horizon*](https://arxiv.org/html/2206.06857v1), IEEE Control Systems Letters, 2022. Relevant part: Section III-F; also listed on the [coauthor's publication page](https://jrforbes.github.io/publications/).

**Credit:** Extends prediction span without increasing optimization-variable count by using fine near-term and coarse later prediction intervals for fast and slow dynamics.

**Comparison:** A numerical prediction grid is different from an optimized physical shut-off and next feedback time. Do not label this paper an online dynamic-grid optimizer without evidence, or suggest that coarse prediction steps force zero actuator input. Use its verified title rather than inventing a “dynamic time resolution MPC” title.

### event-mpc — Event-triggered predictive control

**Primary source:** D. Lehmann, E. Henriksson, and K. H. Johansson, [*Event-Triggered Model Predictive Control of Discrete-Time Linear Systems Subject to Disturbances*](https://people.kth.se/~kallej/papers/event-triggered_ecc13.pdf), European Control Conference, 2013, pp. 1156–1161; DOI 10.23919/ECC.2013.6669580. Relevant parts: Section III, equation (7), execution architecture, and conditional stationary-error bounds.

**Credit:** Reduces feedback communication and optimization by triggering on prediction error or horizon expiry, and relates stationary deviations to disturbance and threshold conditions.

**Comparison:** The actuator executes the planned sequence between events; fewer optimizations do not imply a zero-input coast. The selected trigger monitors state prediction error, whereas our next replan is scheduled at the current decision. Do not attribute this information pattern to every event-triggered or self-triggered method, or claim that our nominal formulation is more robust.

### tdmpc2 — Recent reinforcement learning comparison

**Primary source:** N. Hansen, H. Su, and X. Wang, [*TD-MPC2: Scalable, Robust World Models for Continuous Control*](https://proceedings.iclr.cc/paper_files/paper/2024/hash/cf73d57b6dcda32b293df7c2d5341f49-Abstract-Conference.html), ICLR 2024. Read [the proceedings paper](https://proceedings.iclr.cc/paper_files/paper/2024/file/cf73d57b6dcda32b293df7c2d5341f49-Paper-Conference.pdf), Sections 2–3, together with [the accessible author manuscript](https://arxiv.org/html/2310.16828v2). Its preprint first appeared in 2023; the conference publication meets the requested 2024-or-later date.

**Credit:** Combines learned latent dynamics, terminal-value estimation, and online trajectory optimization, with broad continuous-control evaluation.

**Comparison:** The published action-sequence objective does not explicitly select our two event times or certify path constraints during coast. Its learned dynamics differ from our known physical model; it supports the continuation-learning comparison and does not establish that our controller is TD-MPC2 or superior to it.

**Citation-selection evidence:** The [MLsanity paper record](https://mlsanity.com/paper/2310.16828/) reports 540 citing papers in its own map, with data through 2026-10-26, checked 2026-10-29. This is a dated corpus-specific count, not an independently audited global citation count. OpenAlex returned only eight for its arXiv record and Semantic Scholar API access was rate-limited, so counts are not interchangeable. Use TD-MPC2 as an established recent candidate; keep citation counts out of manuscript prose.

## Novelty and evidence checks

Recent work already connects learning and event-triggered MPC. [*Event-Triggered Model Predictive Control With Deep Reinforcement Learning for Autonomous Driving*](https://ieeexplore.ieee.org/document/10306336/) appears in a January 2024 issue but was first published online in November 2023. Its publisher abstract describes a learned triggering policy; no detailed limitation is inferred from that abstract. It is a guard against claiming that RL-based feedback scheduling is absent, rather than an additional required introductory digression.

The defensible distinction is the precise event structure, the two outer timing variables, conditional bounded-input optimization, declared resource charges, and the scope of any eventual analysis. This reference set is a focused introduction, not an exhaustive novelty search. Neither finite prediction checks nor a learned continuation estimate supplies a continuous-time certificate by itself.

## Drafting handoff

Attach each numerical citation to a method-led clause. Introduce contributions before scope limits. Check normality, information assumptions, and the difference between prediction discretization and physical execution. Carry only qualitative application and method statements into the introduction; keep equations in Formulation/Methods and measured comparisons in Results. The current paper formulates timing selection but reports only the continuously enabled MPC+RL baseline.
