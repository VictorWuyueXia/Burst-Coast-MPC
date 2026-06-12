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



Our coding style rules:
# Coding Style Descipline
- Adhere strictly to our coding style descipline, realizing goals with simplest possbile method, write your logic in compact streamlined line-of-logic files, avoid short wrapper/helper functions, avoid unescesary CLI/configs, avoid fallback values or behaviors, avoid try/with/except. Include this rule in your plan.

- Eplicitly write all your planned class/objects, functions/methods, independent variables, data classes, wrappers/helpers, and parameters in you plan. Each with their specific role and task. You should have a maximum of a handful of each, devided my task oriented workflow or task independent standard operations, and minimize the presence of wrappers/helpers, and parameters. You will not be allowed to exceed your planned structure budget.

- We want to strictly adhere to this rule. We hate thin wrapper functions with too less logic or mega functions with too much logic, files that are too long (>300lines) or too short (<40lines), over abstractions, stand alone parameters/variables/functions/methods that are only called once by others, or poorly organized code logics (in file or class that is not close enough to what the code chunk actually does).


Sweep the entire program repo and hunt for these vialation of coding style desciplines. And plan appropriate fixes.