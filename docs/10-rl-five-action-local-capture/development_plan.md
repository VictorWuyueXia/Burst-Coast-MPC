# Phase 10 Development: Five-Action Local Capture

## Human quick reading

Update the existing compact JAX Q module in place. Replace the exposed physical-limit actions with the measured `±45%` pump pair, add the measured `±2%` fine pair, widen the configured upright goal, resize the Q output to five, and make collection, value audit, logging, and three-step planning enumerate all five actions. Keep the physical clip and phase-9 learning algorithm unchanged.

No training belongs to this implementation change. Finish semantic tests, rerun the environment suite, and retain the action-resolution evidence. A later training launch must start fresh under contract `rotary-q-prior-v2`.

Use the simplest direct implementation. Do not add adapters, compatibility fallbacks, a new configuration layer, thin wrappers, or exception handling. Keep task logic in the existing task-oriented files, and keep substantive files between 40 and 300 lines.

## Machine-scannable development contract

### D10.0 Structure budget

No new production class, dataclass, wrapper, or helper function is authorized.

| Kind | Symbol / parameter | Role |
|---|---|---|
| Existing class, modified | `QNetwork` | Emit five Q-values from the existing two-hidden-layer MLP |
| Existing function, modified | `collect` | Explore, guide, replay, and count five actions |
| Existing function, modified | `evaluate` | Audit five first actions, enumerate 125 three-step sequences, and retain tight/near validation trajectories |
| Existing function, modified | `write_artifacts` | Declare version-2 metadata, plot validation histories, and plot first successes/failures from tight/near rollouts |
| Existing function, modified | `train` | Record tight/near success and violation rates at every scheduled validation |
| Existing function, modified | environment `step` behavior via mission config | Use the revised four goal tolerances; dwell logic is unchanged |
| New module constant | `ACTION_COUNT=5` in `jax_q.py` | Single Python-sized Q output and branch count |
| New module constant | `PUMP_TORQUE_FRACTION=0.45` in `jax_task.py` | Derive both energy actions without changing the physical clip |
| New module constant | `FINE_TORQUE_FRACTION=0.02` in `jax_task.py` | Derive both perturbation actions from the physical limit |
| Existing module constant, modified | `ACTION_TORQUES_NM` | Ordered five-torque device array |
| New diagnostic script | `diagnose_rotary_action_resolution.py::main` | Recreate local fine-action CSV/JSON evidence and plot |
| New diagnostic script | `diagnose_rotary_pump_resolution.py::main` | Recreate global pump-action CSV evidence and plot |
| Mission parameters, modified | `theta=0.08`, `beta=0.08`, `omega=0.15`, `nu=0.20` | Revised sampled success box |

Each diagnostic script has one workflow function and fixed study constants. They are reproducibility code, not runtime dependencies of the learner.

### D10.1 Production edits

1. `jax_q.py`: define `ACTION_COUNT`; use it as the final dense width. Do not change hidden widths, activations, initialization, loss, target update, or optimizer logic.
2. `jax_task.py`: expose negative/positive 45% pump torques and append negative/positive 2% fine torques after off. Draw random actions from `[0,5)`, evaluate five guided candidates, and use a five-bin action count.
3. `jax_task.py`: export separate counts for off, negative pump, positive pump, negative fine, and positive fine. Compute reversal from torque signs rather than unequal action indices.
4. `jax_evaluation.py`: repeat five value-audit branches, force indices `0..4`, allocate five predicted values, and construct the 125 length-three sequences. Compute reversal from torque signs.
5. `jax_artifacts.py`: write `rotary-q-prior-v2`, the five names and torques, and five-value output meaning. Display action indices with their names in trajectory plots.
6. `mission.yaml`: apply the four F10.2 tolerances. Keep `hold-steps=5`.
7. Documentation: update the live phase-8 environment interface values and point to the version-2 rationale. Leave frozen phase-9 formulation and results unchanged.
8. Validation evidence: every new training trial writes tight/near success and arm-violation histories plus a tight/near state-action trajectory figure. The figure uses the first five successes and first five nonsuccesses by fixed validation index; it does not cherry-pick visually favorable episodes.

### D10.2 Test edits

| Check | Exact success flag |
|---|---|
| Network shape | Output shapes end in `5`; parameter count is `18,181`; six parameter leaves remain |
| Action values | `ACTION_TORQUES_NM / u_max == [0,-0.45,0.45,-0.02,0.02]` within float tolerance |
| Transition semantics | Reward telescoping test covers indices `0..4` |
| Collection | Five named action counts sum to every collected transition |
| Goal contract | Loaded tolerances equal `[0.08,0.08,0.15,0.20]`; hold count remains `5` |
| Value audit | Predicted and realized first-action axes have length `5` |
| Planner | All four deployment modes execute and terminal masking remains finite |
| Compatibility | Metadata names version 2 and the five-action order |
| Regression | Complete repository test suite passes |

Tests should verify semantics with existing task tests. Do not add tests that merely repeat constants without exercising an interface, except the action order and mission values because they are public contracts.

### D10.3 Milestones

| Milestone | Work | Validation | Success flag |
|---|---|---|---|
| M10.1 Evidence | Run fixed fine- and pump-resolution sweeps | Inspect both CSVs and plots | `±0.45` pump, `±0.02` fine, and revised box supported by recorded values |
| M10.2 Task contract | Update mission, action array, Q head, collection, and evaluation | Focused environment/Q tests | Every five-action semantic check passes |
| M10.3 Artifact contract | Update metadata and readable trajectory action labels | Artifact round-trip test and manual metadata inspection | No version-1 label or three-action shape in a new artifact |
| M10.4 Regression | Run full suite | Pytest and static checks used by repository | No new failure |
| M10.5 Training readiness | Review diff and resolved contract | Clean source status except intended changes | Fresh version-2 training can start without loading version-1 state |

### D10.4 Next training phase

Start with the strongest phase-9 exploration setup, but initialize a fresh five-output network and replay. Run a short semantic/throughput pilot, then one eight-GPU wave comparing ordinary and guided exploration with common seeds. Evaluate local capture before global curriculum expansion. Tune action/reward or architecture only in later controlled waves and preserve every resolved experiment in artifacts.

Do not call a checkpoint a prior until it passes the existing local-capture gate on held-out tight and near-upright sets. If the new action set still plateaus, analyze state trajectories and longer local reachability before changing network size.
