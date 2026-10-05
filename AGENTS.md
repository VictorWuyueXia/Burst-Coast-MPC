# Project Goal

Our overall grand mission goal here is to develope a easy-effort minimum-realization theory-oriented benchmark simulation for an MPC to solve for a burst of control commands for a short segment of time followed by a longer segment of sleep idle. The prediction horizon and the terminal value will eventually be determined by a neural networkd trained by reinforcement learning style.

At start of a conversation, briefly scan through the code base and the docs to have a grand picture of our goal and current stage.

# Our coding style rules

- Adhere strictly to our coding style descipline, realizing goals with simplest possbile method, write your logic in compact streamlined line-of-logic files, avoid short wrapper/helper functions, avoid unescesary CLI/configs, avoid fallback values or behaviors, avoid try/with/except. Include this rule in your plan.

- Eplicitly write all your planned class/objects, functions/methods, independent variables, data classes, wrappers/helpers, and parameters in you plan. Each with their specific role and task. You should have a maximum of a handful of each, devided my task oriented workflow or task independent standard operations, and minimize the presence of wrappers/helpers, and parameters. You will not be allowed to exceed your planned structure budget.

- We want to strictly adhere to this rule. We hate thin wrapper functions with too less logic or mega functions with too much logic, files that are too long (>300lines) or too short (<40lines), over abstractions, stand alone parameters/variables/functions/methods that are only called once by others, or poorly organized code logics (in file or class that is not close enough to what the code chunk actually does). The designated file length should be achieved with proper code organization and logic grouping, not by removing blank-line seperations or comments.

# dev rules
- Always prefer parallel computing for parallel tasks. For some tasks that seem to need for loops or sequence logics at first glance, think if you can parallel compute it by pre-printing parameters or preparing resources in advance for each iteraition.

- If you encounter long-wait-no-update script runtimes, stop the run, add update messeages in the scripts, and rerun

- When delievering artifacts, seperate the scope between human-reading oriented reports and visuals, and machine-reading (code scripts or other agents) oriented data. Do not mix them together: Keep the human-reading artifacts straight foward, easy-to-understand, shallow organized and easily accessible, keep labels, entries, and legends fully defined and explained but short. Add a interpretation_summary.md written by you to briefly and quickly explain how to interpret the artifact and your judement/conclusion; Make machine-reading oriented data detailed and organized in subfolders, ready to be analyzed or used to recreate results. For example, when saving runtime/training histories or confusion matrices, keep a csv in machine-scannables and a png in human-readables.

- When training models or making controllers with tunable parameters, proactively train or test out the agent in several iterations. Between iterations, scan through the artifacts and identify any potential improvements that could be achieced by tuning the parameters before the next iteration. Do so until you believe tuning parameters alone cannot significantly improve anymore, and if the agent still fails at this stage, infer a reasong why and suggest a next step fix.

# doc rules
- When writing docs, if your answer include mathematical expressions/formulas/variables, fully define them when they emerge, explain how to get the values (defined, measured, computed, heuristic, tunable, etc.), and give an intuitive interpretations for each of them. Output in .md files with "$$" latex math syntax. Split into human readable parts which is more illustrative and intuitive with major governing formulas, and machine schanning parts with as much rigorous details as you can write. Also make a 中文 version of the same doc file with same name and suffix -CN, and gitignore any files with suffix -CN.

- When formulating a plan, explicitly identify stages and milestones of the dev process, give specific validation procedures for each milestone and their success flags. And always rememver our projects are meant to be show-of-concept, not to deliver a clinic/industry-ready product. So never over engineering the method, but proactively keep the mathematicall elegance (streamline, efficiency, naturally working, etc.) and strigency of our methods.

# venv
We are on a ubuntu remote server where conda or sudo is not available, we need to build or see the already built .venv acoording to the encironment or requirement files. But keep and keep updating the encironment or requirement files so we have them available when switching to other machines.

# How to use GPU
We are on a 8-GPU servers now which means we want to use the maximum parallel computing ability of them with our code to lighten the time cost of our computation. But you cannot access GPU with your sandbox terminal directly, and sudo is not available on this remote server. So the way is to submit the training/run command through the terminal tool with sandbox_permissions="require_escalated". After my approval, the command executed outside the sandbox on the same Ubuntu host, where CUDA and the GPUs were visible.
For example, the earlier eight-GPU run used:
```
.venv/bin/python scripts/run_full_capability.py \
  --config configs/controller_smoke.yaml \
  --compute 8gpu \
  --output-dir /tmp/laser_ablation_smoke_20260825 \
  --seed 20260825
```
The tool invocation explicitly requested approval with the reason that JAX needed access to the server GPUs. I then monitored the returned process session incrementally.