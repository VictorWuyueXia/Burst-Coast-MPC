


- remove any pause related logics from the code
- unwrap @src/wsmpc/utils/loaders.py into @src/wsmpc/utils/config_schema.py or @src/wsmpc/cli.py , integrate small helper functions into main logic (i.e. _package_file() register_resovlers())
- remove unused roolout() function, and scan entire code base for similar unused dead code like this and remove them


1. continuous or discrete for casadi?
2. zero or constant coast input for casadi?
3.



Adhere strictly to our coding style descipline, realizing goals with simplest possbile method, write your logic in compact streamlined line-of-logic files, avoid short wrapper/helper functions, avoid unescesary CLI/configs, avoid fallback values or behaviors, avoid try/with/except. Include this rule in your plan.


make a new version of gates for both locations, but keep the original code by commenting them out.

- The new gate is formulated as g_gaussian=exp(-(K/E*)^2/sigma^2), where K is the kinetic energy, E* is the desired total energy, and sigma is the tunnable distribution factor that should be added to weights.py

- The new first and second terms of the cost functions are now w_e*e_E^2+w_L*e_E^2*g

Also simplify our current version such that the controller plans for a whole natural period horizon, but truncate

- clean up config files
- move main run_episode logics into coordinator, which should only contain two methods: init() and run_episode(), event-trigger()
- add a parameter "event-trigger=ture" in config#coordinator and logic in run_episode that if ture, trigger replanning based on a rule defined by event-trigger(). For this version, the event-trigger is when state is sufficiently close to theta=0