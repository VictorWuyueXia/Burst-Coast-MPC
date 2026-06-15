- Besure to include professional and concise one-liner comments for each key code blocks, but do not include any "stage 1" or dev progress or dev versin information in the code base. All info should be provided as if the dev has finished and being presented to another developer. 

- Code with ultimate aim of best human comprehension and human maintenance ease, with best belance between atomization of logic blocks and fluent chain of logic for human to read. The empirical best practice for our own code is to generally keep every code file 50-250lines, with each function 20-100 lines. Atomize the workflow steps, logics, and nodes, but do not over-layer them. Write everything in minimum realization. 

- Be more aggressive in removing unecessary codes, be more prudent in adding in extra logics.

- Prefer a streamlined continuous un-broken line-of-thoughts or line-of-logic isntead of catagorizing code by type or function.

- Be conservative of using CLI or config parameters, do not include anything other than specifically asked or anything you are not confident that is needed. Avoid parameterizing system or runtime related values to config, they should be hardcoded and frozen permanently. Only keep minimum config parameters that are for different experiment setups.

- Code like an academic scholar expert in control theory, dynamics, autonmous robot systems, less like a software developer. 

- Remove all fallback value or fallback behaviour logics. We prefer sacrificing robustness for 100% code-intent alignment, if something goes wrong let the error emerge laudly.

- Be productive, hard working as if you are trying to impress a mentor/boss in a busy work day. Realize logic in the most efficient and elegant way for computational efficiency. Take advantages of Python's C++ backbone to replace most loops with batch matrices.


- remove any pause related logics from the code
- unwrap @src/wsmpc/utils/loaders.py into @src/wsmpc/utils/config_schema.py or @src/wsmpc/cli.py , integrate small helper functions into main logic (i.e. _package_file() register_resovlers())
- remove unused roolout() function, and scan entire code base for similar unused dead code like this and remove them


1. continuous or discrete for casadi?
2. zero or constant coast input for casadi?
3.

+ clean up config files


+ move main run_episode logics into coordinator, which should only contain two methods: init() and run_episode(), event-trigger()
+ add a parameter "event-trigger=ture" in config#coordinator and logic in run_episode that if ture, trigger replanning based on a rule defined by event-trigger(). For this version, the event-trigger is when state is sufficiently close to theta=0



# Coding Style Descipline
- Adhere strictly to our coding style descipline, realizing goals with simplest possbile method, write your logic in compact streamlined line-of-logic files, avoid short wrapper/helper functions, avoid unescesary CLI/configs, avoid fallback values or behaviors, avoid try/with/except. Include this rule in your plan.

- Eplicitly write all your planned class/objects, functions/methods, independent variables, data classes, wrappers/helpers, and parameters in you plan. Each with their specific role and task. You should have a maximum of a handful of each, devided my task oriented workflow or task independent standard operations, and minimize the presence of wrappers/helpers, and parameters. You will not be allowed to exceed your planned structure budget.







[5.5-RL-dev-plan.md](docs/5.5-RL-dev-plan.md) reflects my current stage of dev plan. The overall goal of this stage is to devlop the actor-critic RL closely coupled with our existing burst-coast MPC controller.

Now make plan for the first dev step: Generate Monte Carlo data for offline training. Run the standalone MPC simulation under sampled or scripted $(\bar B,\bar H)$ policies. Log full trajectories, including:$(s_k, a_k, G_k, s_{k+1}, d_k,  u_k, t_{\mathrm{solve},k})$ At episode end, compute Monte Carlo returns:$G_k = \sum_{i=k}^{T} \gamma^{i-k}c_i$

This creates the initial offline dataset:$\mathcal D_{\mathrm{MC}} = \left\{(s_k, a_k, G_k)\right\}$

I want to make changes with minimum touch on current logics. 
- we should make a dedicated config file "data-generation" for this step, including config parameters to make a uniform montecarlo generation of B and H
- We should add small quick functions in utils to generate monte carlo actions.
- we should add time counting and re-plann step logs, in parallel to the current detailed steps.csv log. The new logs are dedicated for RL training, which adhere to the RL expected information.
- we should not be using parallel computing for this step, as each time we only carry out one set of B and H.

+ do not change root dir name or git related naming yet, or you will loose your work-dir mid-run. Give step-by-step instruction for me to do it manually.



# Coding Style Descipline

Our coding style rules:
- Adhere strictly to our coding style descipline, realizing goals with simplest possbile method, write your logic in compact streamlined line-of-logic files, avoid short wrapper/helper functions, avoid unescesary CLI/configs, avoid fallback values or behaviors, avoid try/with/except. Include this rule in your plan.

- Explicitly write all your planned class/objects, functions/methods, independent variables, data classes, wrappers/helpers, and parameters in your plan. Each with their specific role and task. You should have a maximum of a handful of each, divided by task oriented workflow or task independent standard operations, and minimize the presence of wrappers/helpers, and parameters. You will not be allowed to exceed your planned structure budget.

- We want to strictly adhere to this rule. We hate thin wrapper functions with too little logic or mega functions with too much logic, files that are too long (>300lines) or too short (<40lines), over abstractions, stand alone parameters/variables/functions/methods that are only called once by others, or poorly organized code logic (in file or class that is not close enough to what the code chunk actually does).


Sweep the entire program repo and hunt for these vialation of coding style desciplines. And plan appropriate fixes.


- how to run monte carlo with more data in fast/low-energy and high-return cases?




Suggestions, in order:
Run longer before changing the model
Try max-epochs: 4000 or 5000. The curve is still descending smoothly, so 2000 epochs is undertrained, not overtrained.

Keep this as a promising but not final critic
It is good enough for offline ranking experiments/sanity checks, but I would not trust it as a final controller policy selector yet. There are still 38 / 204 validation rows with absolute error over 300.

Add more data next
Region issue: low_energy_fast has only 25 validation rows and the worst MSE: 125,120. I’d still target 300 total Monte Carlo runs next, with special attention to fast/low-energy and high-return cases.

Watch the next training curve
If 4000-5000 epochs flattens near current validation MSE, data is the bottleneck. If it keeps dropping, optimization was the bottleneck.

- how to run monte carlo in parallel?




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




Read the current [800-report.md](/Users/vic/Documents/GitRepo/Burst-Coast-MPC/Burst-Coast-MPC/docs/800-report.md) report and [5.6-NN-design.md](/Users/vic/Documents/GitRepo/Burst-Coast-MPC/Burst-Coast-MPC/docs/5.6-NN-design.md) [5.61-offline-training-dev-plan.md](/Users/vic/Documents/GitRepo/Burst-Coast-MPC/Burst-Coast-MPC/docs/5.61-offline-training-dev-plan.md) for this stage of dev plan. Right now we have a time fitting model and an offline trained model. We need to dev a handle in [coordinator.py](/Users/vic/Documents/GitRepo/Burst-Coast-MPC/Burst-Coast-MPC/src/burst_coast_mpc/coordinator.py) for the online RL training.

- See and get what the model artifact looks like.
- The program now will have four modes: 1. simulate->mpc-only which runs our current simulation path; 2. intelligent, which is the deployment mode for RL+MPC with out learning; 3. train, which is the online training mode for RL policy 4. montecarlo, as current. Change CLI file and coordinator file accordingly.
- use pytorchlightning as specified in environment file
- each mode is its own file for bring up the simulation, rename the coordinator file to epoch_coordinator that better reflect its row of coordinating progress within an epoch.