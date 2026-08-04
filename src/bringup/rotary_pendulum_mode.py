"""Physics-simulation bringup for the rotary-pendulum scenario."""

from __future__ import annotations

import time

import numpy as np
from rich.console import Console

from rotary_pendulum.environment.dynamics import derive_model
from rotary_pendulum.environment.environment import RotaryPendulumEnvironment
from rotary_pendulum.utils.config_schema import load_config
from rotary_pendulum.utils.monte_carlo import sample_action_plan
from rotary_pendulum.visualization.realtime import RealtimeRotaryPendulumPlot


def run_rotary_pendulum_mode(*, console: Console) -> None:
    """Run one realtime physics episode under sampled open-loop torque plans."""

    # Assemble the scenario directly because MPC and RL ownership paths are intentionally empty.
    config = load_config()
    model = derive_model(config.rotary_pendulum)
    environment = RotaryPendulumEnvironment(config)
    rng = np.random.default_rng(config.experiment.seed)
    monitor = RealtimeRotaryPendulumPlot(config, model)
    observation = environment.reset()
    monitor.start(observation)

    # Sample one constant signed torque sequence at each replan and execute it open loop.
    replan_index = 0
    while environment.t_index < config.experiment.max_steps and not observation.goal_reached:
        sampling_started_at = time.perf_counter()
        plan = sample_action_plan(
            rng,
            config,
            model,
            replan_index,
        )
        solve_time_s = time.perf_counter() - sampling_started_at
        console.print(
            f"replan={replan_index} t={environment.t_sec:.3f}s "
            f"H={plan.horizon_steps} h={plan.hbar:.3f} b={plan.bbar:.3f} "
            f"tau={plan.torques_nm[0]:+.5f}Nm solve={solve_time_s:.6f}s"
        )

        for plan_step, commanded_torque_nm in enumerate(plan.torques_nm):
            if environment.t_index >= config.experiment.max_steps or observation.goal_reached:
                break
            observation, record = environment.step(
                float(commanded_torque_nm),
                plan,
                solve_time_s if plan_step == 0 else 0.0,
                replan_flag=plan_step == 0,
            )
            monitor.add_step(observation, record)
        replan_index += 1

    # Leave the completed trajectory visible and report the minimal episode outcome.
    monitor.finish()
    status = "upright_goal_reached" if observation.goal_reached else "max_steps_reached"
    console.print(
        {
            "run_id": config.experiment.run_id,
            "scenario": "rotary_pendulum",
            "mode": "physics_simulation",
            "status": status,
            "steps": environment.t_index,
            "replans": replan_index,
            "simulated_time_s": environment.t_sec,
            "natural_period_s": model.natural_period_s,
            "final_state": environment.state.tolist(),
        }
    )
