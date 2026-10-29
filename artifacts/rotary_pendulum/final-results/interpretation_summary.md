# Selected rotary-pendulum results

These four bundles contain the selected controllers, population performance, representative sessions, and the energy/torque sensitivity study. Directory names describe their scientific role rather than development dates. No new simulations or controller changes were made during packaging.

| Bundle | Start here | Scope |
|---|---|---|
| RL policy | [Reading guide](rl-policy/human-readables/interpretation_summary.md) | Selected actor and twin critics; nominal and disturbed 600-second evaluation. |
| RL+MPC | [Reading guide](rl-mpc/interpretation_summary.md) | Final nonlinear MPC with quadratic stage cost and learned terminal value; 32 complete 40-second sessions. |
| Energy controller | [Reading guide](energy-controller/human-readables/interpretation_summary.md) | Final 100-millisecond constant-torque heuristic; two validation seeds and a fine-integration check. |
| Energy/torque sensitivity | [Reading guide](energy-torque-sensitivity/human-readables/interpretation_summary.md) | Total-energy versus relative swing-energy sensitivity planes; controlled sessions, coast continuations, and action comparisons. |

Human-readable figures and guides are separate from numerical tables and arrays. The RL inference code is at its bundle root so that the model remains portable. The MPC session records include exact executed trajectories and the original source snapshots.

The controller evaluations use different torque limits, arm ranges, durations, and success criteria. Do not interpret them as a matched controller ranking. Energy-controller captures terminate their recordings; RL and MPC evaluations continue through balancing.

The sensitivity plane identifies local input opportunities. The pendulum quantity is relative hinge-swing energy, not full pendulum-body energy. Both energy errors and natural exchange matter when deciding between energy injection, redistribution, and coast. No quadrant controller was implemented by this study.

For numerical reuse, read the [data contract](machine-scannables/data_contract.md). The [file manifest](machine-scannables/file_manifest.json) records bundle-relative paths, sizes, and SHA-256 checksums. [Packaging provenance](machine-scannables/packaging_provenance.json) records source paths and selected archive lanes. Historical paths in provenance and source snapshots identify original experiments; they are not runtime dependencies.

The collection script requires the source workstation's local archives. Downstream users need only the selected payloads to inspect metrics, redraw trajectories, or run policy inference. Unselected training histories, tuning campaigns, full RL rollout archives, and figure-audit scratch files remain local.
