# Coding Style Descipline

Our coding style rules:
- Adhere strictly to our coding style descipline, realizing goals with simplest possbile method, write your logic in compact streamlined line-of-logic files, avoid short wrapper/helper functions, avoid unescesary CLI/configs, avoid fallback values or behaviors, avoid try/with/except. Include this rule in your plan.

- Eplicitly write all your planned class/objects, functions/methods, independent variables, data classes, wrappers/helpers, and parameters in you plan. Each with their specific role and task. You should have a maximum of a handful of each, devided my task oriented workflow or task independent standard operations, and minimize the presence of wrappers/helpers, and parameters. You will not be allowed to exceed your planned structure budget.

- We want to strictly adhere to this rule. We hate thin wrapper functions with too less logic or mega functions with too much logic, files that are too long (>300lines) or too short (<40lines), over abstractions, stand alone parameters/variables/functions/methods that are only called once by others, or poorly organized code logics (in file or class that is not close enough to what the code chunk actually does). The designated file length should be achieved with proper code organization and logic grouping, not by removing blank-line seperations or comments.

- Always prefer parallel computing for parallel tasks. For some tasks that seem to need for loops or sequence logics at first glance, think if you can parallel compute it by pre-printing parameters or preparing resources in advance for each iteraition.

- If you encounter long-wait-no-update script runtimes, stop the run, add update messeages in the scripts, and rerun

- When delievering artifacts, seperate the scope between human-reading oriented reports and visuals, and machine-reading (code scripts or other agents) oriented data. Do not mix them together: Keep the human-reading artifacts straight foward, easy-to-understand, shallow organized and easily accessible, keep labels, entries, and legends fully defined and explained but short. Add a interpretation_summary.md written by you to briefly and quickly explain how to interpret the artifact and your judement/conclusion; Make machine-reading oriented data detailed and organized in subfolders, ready to be analyzed or used to recreate results. For example, when saving runtime/training histories or confusion matrices, keep a csv in machine-scannables and a png in human-readables.