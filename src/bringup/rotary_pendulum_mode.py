"""Physics, monitoring, MPC, and artifact bringup for the rotary pendulum."""

from __future__ import annotations

from typing import Any

import numpy as np
from rich.console import Console

from rotary_pendulum.environment.environment import RotaryPendulumEnvironment
from rotary_pendulum.mpc import RotaryMPCController
from rotary_pendulum.utils.artifacts import RotaryArtifactWriter
from rotary_pendulum.utils.config_schema import (
    EPISODE_CONFIG_PATHS,
    VISUAL_CONFIG_PATH,
    RuntimeMode,
    load_episode_config,
    load_visualization_config,
)
from rotary_pendulum.utils.messages import ActionPlan, EpisodeSummary, StepRecord
from rotary_pendulum.visualization.artifact_plots import create_artifact_figures
from rotary_pendulum.visualization.realtime import RealtimeRotaryPendulumPlot


def run_rotary_pendulum_mode(
    *,
    alias: str | None,
    no_visual: bool,
    console: Console,
) -> EpisodeSummary:
    """Run one complete nonlinear burst-coast MPC episode and save its evidence."""

    # Compose only the runtime domains required by the selected interactive policy.
    runtime_mode: RuntimeMode = "headless" if no_visual else "realtime"
    config = load_episode_config(EPISODE_CONFIG_PATHS, runtime_mode=runtime_mode)
    visualization = None if no_visual else load_visualization_config()
    artifact_alias = alias if alias is not None else config.artifacts.alias
    config.artifacts.alias = artifact_alias
    writer = RotaryArtifactWriter(
        config,
        alias=artifact_alias,
        cli_args={
            "scenario": "rotary_pendulum",
            "mode": "mpc-only",
            "alias": alias,
            "no_visual": no_visual,
        },
        config_sources=(
            EPISODE_CONFIG_PATHS
            if visualization is None
            else (*EPISODE_CONFIG_PATHS, VISUAL_CONFIG_PATH)
        ),
        visualization=visualization,
    )
    if no_visual:
        import matplotlib

        matplotlib.use("Agg")

    # Bind one shared physical model to the plant, predictor, controller, and optional monitor.
    environment = RotaryPendulumEnvironment(config)
    console.print("Compiling rotary-pendulum MPC candidates...")
    writer.log("controller_compile_started")
    controller = RotaryMPCController(config, config.mpc)
    writer.log("controller_compile_finished")
    monitor: RealtimeRotaryPendulumPlot | None = None
    if visualization is not None:
        monitor = RealtimeRotaryPendulumPlot(
            config,
            visualization,
            environment.model,
        )
    observation = environment.reset()
    if monitor is not None:
        monitor.start(observation)

    # Solve only after the prior burst and exact zero-coast sequence has fully executed.
    records: list[StepRecord] = []
    plans: list[ActionPlan] = []
    previous_applied_torque_nm = 0.0
    replan_index = 0
    while environment.t_index < config.experiment.max_steps and not observation.goal_reached:
        console.print(f"Solving replan {replan_index} at t={environment.t_sec:.3f}s...")
        plan = controller.solve_plan(
            observation,
            previous_applied_torque_nm,
            replan_index,
        )
        if plan.torques_nm.shape != (plan.horizon_steps,):
            raise ValueError("The selected MPC torque sequence must contain exactly H samples")
        if np.any(plan.torques_nm[plan.burst_steps :] != 0.0):
            raise ValueError("The selected MPC coast suffix must contain exact zero torque")
        plans.append(plan)
        event = (
            f"replan={replan_index} t={environment.t_sec:.3f}s H={plan.horizon_steps} "
            f"B={plan.burst_steps} C={plan.coast_steps} h={plan.hbar:.6f} "
            f"b={plan.bbar:.6f} objective={plan.objective_value:.6f} "
            f"solve={plan.solve_time_s:.6f}s"
        )
        console.print(event)
        writer.log(event)

        # Feed every planned action through one dynamics step and one monitor sample.
        for plan_step, commanded_torque_nm in enumerate(plan.torques_nm):
            if environment.t_index >= config.experiment.max_steps or observation.goal_reached:
                break
            observation, record = environment.step(
                float(commanded_torque_nm),
                plan,
                replan_flag=plan_step == 0,
            )
            records.append(record)
            previous_applied_torque_nm = record.u_applied_nm
            if monitor is not None:
                monitor.add_step(observation, record)
        replan_index += 1

    # Finalize visual state and persist both machine-readable and interpretive artifacts.
    if monitor is not None:
        monitor.finish()
    status = "upright_goal_reached" if observation.goal_reached else "max_steps_reached"
    theta_rad, alpha_rad, omega_rad_s, nu_rad_s = environment.state
    summary = EpisodeSummary(
        status=status,
        steps=environment.t_index,
        replans=replan_index,
        simulated_time_s=environment.t_sec,
        natural_period_s=environment.model.natural_period_s,
        final_state=(
            float(theta_rad),
            float(alpha_rad),
            float(omega_rad_s),
            float(nu_rad_s),
        ),
        final_energy_error_j=observation.energy_error_j,
        final_normalized_energy_error=observation.normalized_energy_error,
        goal_reached=observation.goal_reached,
    )
    figures = create_artifact_figures(records, config.rotary_pendulum, environment.model)
    writer.log(f"episode_finished status={status} steps={summary.steps} replans={summary.replans}")
    writer.finalize(records, plans, summary, figures)
    from matplotlib import pyplot as plt

    for figure in figures.values():
        plt.close(figure)
    output: dict[str, Any] = {
        "run_id": config.experiment.run_id,
        "scenario": "rotary_pendulum",
        "mode": "mpc-only",
        **summary.__dict__,
        "artifact_dir": str(writer.run_dir),
    }
    console.print(output)
    return summary
