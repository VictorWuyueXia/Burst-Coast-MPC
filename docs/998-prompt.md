
- how to run monte carlo in parallel?



# remaining plan
1. Monte Carlo data generation
2. Q-network design
3. offline critic training
4. compute-time model fitting from solve logs
5. online simulated training while preserving MPC as independent controller
You are now past step 3: you have a good offline-trained critic candidate.
Next steps:
- Freeze the candidate snapshot
Use artifacts/model-snapshots/structured-critic_20260614T215834 as the current candidate. Keep its config.json, normalization.json, lambda.json, checkpoint, and diagnostics together.

- Update the report
Change the last paragraph from “current active step is Monte Carlo data generation” to “current active step is validating and deploying the offline critic.”

- Build critic loading/inference
Add code that loads:
-- critic_state_dict.pt or best checkpoint
-- normalization.json
-- lambda.json
-- model structure assumptions
Then expose a function that evaluates Q(s, bbar, hbar) over the action grid.

- Add grid action selection
For each decision state, evaluate the critic over the (bbar, hbar) grid and select:
`argmin_a Q(s, a)`
MPC still solves torque; the critic only chooses burst/horizon.

- Fit or decide compute-time handling
The report says compute-time model fitting is next. Your critic currently uses logged solve-time-s; deployment needs either:
a fitted solve-time predictor, or
a simple measured/constant compute-time assumption for first rollout tests.

- Run closed-loop simulation A/B tests
Compare:
existing fixed/sampled burst-coast MPC
critic-selected burst/horizon MPC
maybe full-horizon or short-horizon baselines
Track success rate, return cost, time to goal, actuation effort, solve time, and number of replans.

- Inspect failures
For bad episodes, log selected (bbar, hbar), predicted Q, realized return, and state region. This tells you whether the critic is choosing badly or MPC itself is failing.

- Only then consider online training
Do not jump to online RL yet. First prove the offline critic improves closed-loop simulation against baselines.







# Retrain time fitting
Rerun CMP time fitting on the larger dataset.

The current frozen model I regenerated was fit from the local available Monte Carlo rows only: `278` RL rows under `artifacts/experiments`. That was enough to fix the negative-prediction bug and validate the logic, but it is not the best final compute-time artifact if your other PC has a larger, more representative dataset.

Use the updated script there:

```bash
python scripts/inverted_pendulum/cmp-time-fitting/fit_cmp_time_model.py
```

It now fits a nonnegative model over realized MPC dimensions:

```text
solve_time_s = c_H * H + c_B * B + c_BH * B * H
```

with nonnegative coefficients, so it should not produce negative deployment-grid solve-time predictions.

After fitting, copy the selected output folder contents into the frozen time-model directory used by config:

```text
artifacts/inverted_pendulum_frozen/linear_20260615T191818/
```

or create a new frozen folder and update:

```yaml
rl:
  time-model-artifact-dir: artifacts/inverted_pendulum_frozen/<new-time-model-dir>
```

Then run:

```bash
python -m pytest tests/test_config_loading.py
```

That will verify the frozen model stays positive over the deployment action grid.



# Before online training
Read the current [800-report.md](/Users/vic/Documents/GitRepo/Burst-Coast-MPC/Burst-Coast-MPC/docs/800-report.md) report and [5.6-NN-design.md](/Users/vic/Documents/GitRepo/Burst-Coast-MPC/Burst-Coast-MPC/docs/5.6-NN-design.md) [5.61-offline-training-dev-plan.md](/Users/vic/Documents/GitRepo/Burst-Coast-MPC/Burst-Coast-MPC/docs/5.61-offline-training-dev-plan.md) for this stage of dev plan. Right now we have a time fitting model and an offline trained model. We need to dev a handle in [coordinator.py](/Users/vic/Documents/GitRepo/Burst-Coast-MPC/Burst-Coast-MPC/src/burst_coast_mpc/coordinator.py) for the online RL training.

- See and get what the model artifact looks like.
- The program now will have four modes: 1. simulate->mpc-only which runs our current simulation path; 2. intelligent, which is the deployment mode for RL+MPC with out learning; 3. train, which is the online training mode for RL policy 4. montecarlo, as current. Change CLI file and coordinator file accordingly.
- use pytorchlightning as specified in environment file
- each mode is its own file for bring up the simulation, rename the coordinator file to epoch_coordinator that better reflect its row of coordinating progress within an epoch.
- Keep the RL and model inference and online learning related file in src/inverted_pendulum/RL