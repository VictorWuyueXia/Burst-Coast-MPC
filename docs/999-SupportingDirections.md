Two actor-critic formulations:
1. learn state value V, and let dynamic programming to compile prediction horizon -- coast horizon for each state is when dynamics evolve the system with monotone decreasing state value (later formulate this as neural CLBF)
2. directly train state-action Q value, with action for each state being the prediction horizon (which is also replanning timing)



RL critic Q values ->
1. replanning schedules by DP
2. control Lyapunov barrier function
3. terminal cost, cost smoothing, warm starts
4. Dynamic DeltaT

To support the project:
1. Reduced-order dynamics representations by energy and phase
based on energy-like, phase
2. Distributed RL+MPC as non convex solver