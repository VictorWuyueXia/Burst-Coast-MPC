"""Greedy, lookahead, and value-calibration evaluation for the discrete Q prior."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax import Array

from rotary_pendulum.environment.jax_dynamics import MODEL, PHYSICS_DT_S
from rotary_pendulum.environment.jax_environment import ARM_LIMIT_RAD, EnvState
from rotary_pendulum.RL.jax_q import ACTION_COUNT, QNetwork
from rotary_pendulum.RL.jax_task import ACTION_TORQUES_NM, TARGET_ENERGY_J, observe, transition


def evaluate(
    learner: Mapping[str, Any],
    initial_states: EnvState | Mapping[str, EnvState],
    mode: str,
    experiment: Mapping[str, Any],
) -> tuple[dict[str, Array], dict[str, Array]]:
    """Evaluate one frozen checkpoint under a declared controller or value audit."""

    suite_modes = {
        "validation_suite": ("greedy",),
        "final_suite": ("greedy", "lookahead_zero", "lookahead_q"),
    }
    if mode in suite_modes:
        strata = cast(Mapping[str, EnvState], initial_states)
        combined = jax.tree.map(
            lambda *parts: jnp.concatenate(parts, axis=0),
            *(strata[name] for name in ("downward", "moving", "near")),
        )
        suite_metrics: dict[str, Array] = {}
        suite_trajectories: dict[str, Array] = {}
        for controller in suite_modes[mode]:
            compiled = jax.jit(
                lambda q, s, selected=controller: evaluate(q, s, selected, experiment)
            )
            metrics, trajectories = compiled(learner, combined)
            suite_metrics.update(
                {f"{controller}_overall_{key}": value for key, value in metrics.items()}
            )
            for name, states in strata.items():
                stratum_metrics, stratum_trajectories = compiled(learner, states)
                suite_metrics.update(
                    {f"{controller}_{name}_{key}": value for key, value in stratum_metrics.items()}
                )
                if controller == "greedy" and name in ("tight", "near"):
                    suite_trajectories.update(
                        {f"{name}_{key}": value for key, value in stratum_trajectories.items()}
                    )
            if controller == "greedy":
                suite_metrics.update(metrics)
                suite_trajectories.update(trajectories)
        if mode == "final_suite":
            audit_metrics, audit_trajectories = jax.jit(
                lambda q, s: evaluate(q, s, "value_audit", experiment)
            )(learner, combined)
            suite_metrics.update({f"audit_{key}": value for key, value in audit_metrics.items()})
            suite_trajectories.update(
                {f"audit_{key}": value for key, value in audit_trajectories.items()}
            )
        return suite_metrics, suite_trajectories

    initial_states = cast(EnvState, initial_states)
    valid_modes = {
        "greedy",
        "lookahead_zero",
        "lookahead_q",
        "value_audit",
    }
    if mode not in valid_modes:
        raise ValueError(f"Unsupported Q evaluation mode: {mode}")
    network = QNetwork(tuple(experiment["hidden_widths"]), experiment["activation_name"])
    audit = mode == "value_audit"
    if audit:
        observation = observe(initial_states)
        predicted_train = cast(Array, network.apply(learner["params"], observation))
        initial_done = initial_states.success | initial_states.timeout
        predicted_base = jnp.where(
            initial_done[:, None],
            0.0,
            predicted_train,
        )
        env_state = jax.tree.map(
            lambda value: jnp.repeat(value[:, None, ...], ACTION_COUNT, axis=1), initial_states
        )
        forced_action = jnp.broadcast_to(
            jnp.arange(ACTION_COUNT, dtype=jnp.int32), env_state.physics_steps.shape
        )
    else:
        predicted_base = jnp.zeros((initial_states.x.shape[0], ACTION_COUNT), dtype=jnp.float32)
        env_state = initial_states
        forced_action = jnp.zeros(initial_states.physics_steps.shape, dtype=jnp.int32)
    batch_shape = env_state.physics_steps.shape
    _, initial_alpha, _, initial_nu = jnp.moveaxis(env_state.x, -1, 0)
    initial_energy_ratio = (
        0.5 * MODEL.pendulum_inertia_kg_m2 * initial_nu**2
        + MODEL.gravity_torque_nm * (1.0 - jnp.cos(initial_alpha))
    ).astype(jnp.float32) / TARGET_ENERGY_J
    initial_totals = jnp.zeros(batch_shape + (11,), dtype=jnp.float32)
    initial_totals = initial_totals.at[..., 7].set(initial_energy_ratio)
    initial_totals = initial_totals.at[..., 8].set(jnp.abs(env_state.x[..., 0]).astype(jnp.float32))

    def advance(
        carry: tuple[EnvState, Array, Array], _unused: None
    ) -> tuple[tuple[EnvState, Array, Array], tuple[Array, ...]]:
        state, totals, decision = carry
        active = ~(state.success | state.timeout)
        observation = observe(state)
        q_values = cast(Array, network.apply(learner["params"], observation))
        greedy_action = jnp.argmax(q_values, axis=-1).astype(jnp.int32)
        if audit:
            action = jnp.where(decision == 0, forced_action, greedy_action)
        elif mode.startswith("lookahead"):
            action_indices = jnp.arange(ACTION_COUNT, dtype=jnp.int32)
            sequences = jnp.stack(
                jnp.meshgrid(action_indices, action_indices, action_indices, indexing="ij"),
                axis=-1,
            ).reshape((ACTION_COUNT**3, 3))
            candidate_state = jax.tree.map(
                lambda value: jnp.repeat(value[:, None, ...], ACTION_COUNT**3, axis=1), state
            )
            candidate_score = jnp.zeros((state.x.shape[0], ACTION_COUNT**3), dtype=jnp.float32)
            for depth in range(3):
                candidate_action = jnp.broadcast_to(
                    sequences[None, :, depth], candidate_score.shape
                )
                candidate_transition = transition(candidate_state, candidate_action, experiment)
                candidate_state = candidate_transition[0]
                candidate_base = candidate_transition[2]
                candidate_score += candidate_base
            endpoint_observation = observe(candidate_state)
            endpoint_done = candidate_state.success | candidate_state.timeout
            if mode == "lookahead_q":
                endpoint_train = cast(Array, network.apply(learner["params"], endpoint_observation))
                endpoint_base = jnp.max(endpoint_train, axis=-1)
                candidate_score += jnp.where(endpoint_done, 0.0, endpoint_base)
            selected_sequence = jnp.argmax(candidate_score, axis=-1)
            action = sequences[selected_sequence, 0]
        else:
            action = greedy_action

        result = transition(state, action, experiment)
        next_state, base_reward, train_reward, done = result[0], result[2], result[3], result[5]
        elapsed = PHYSICS_DT_S * (next_state.physics_steps - state.physics_steps).astype(
            jnp.float32
        )
        previous_action = totals[..., 6].astype(jnp.int32)
        off_to_on = active & (previous_action == 0) & (action != 0)
        reversal = active & (ACTION_TORQUES_NM[previous_action] * ACTION_TORQUES_NM[action] < 0.0)
        theta, alpha, _, nu = jnp.moveaxis(next_state.x, -1, 0)
        energy_ratio = (
            0.5 * MODEL.pendulum_inertia_kg_m2 * nu**2
            + MODEL.gravity_torque_nm * (1.0 - jnp.cos(alpha))
        ).astype(jnp.float32) / TARGET_ENERGY_J
        totals = totals.at[..., 0].add(base_reward)
        totals = totals.at[..., 1].add(train_reward)
        totals = totals.at[..., 2].add(elapsed * (action != 0))
        totals = totals.at[..., 3].add(elapsed)
        totals = totals.at[..., 4].add(off_to_on)
        totals = totals.at[..., 5].add(reversal)
        totals = totals.at[..., 6].set(action)
        totals = totals.at[..., 7].max(energy_ratio)
        totals = totals.at[..., 8].max(jnp.abs(theta).astype(jnp.float32))
        totals = totals.at[..., 9].add(elapsed * jnp.abs(ACTION_TORQUES_NM[action]))
        totals = totals.at[..., 10].add(elapsed * (jnp.abs(theta) >= ARM_LIMIT_RAD))
        return (next_state, totals, decision + 1), (
            next_state.x,
            action,
            base_reward,
            train_reward,
            done,
            active,
            result[4],
            PHYSICS_DT_S * next_state.physics_steps,
            next_state.goal_count,
        )

    (final_state, totals, _), history = jax.lax.scan(
        advance,
        (env_state, initial_totals, jnp.array(0, dtype=jnp.int32)),
        None,
        length=200,
    )
    (
        state_history,
        actions,
        base_rewards,
        train_rewards,
        done_history,
        valid_history,
        reward_components,
        time_s,
        goal_count,
    ) = history
    if audit:
        realized = totals[..., 0]
        error = predicted_base - realized
        selected = jnp.argmax(predicted_base, axis=-1)
        selected_realized = jnp.take_along_axis(realized, selected[:, None], axis=-1)[:, 0]
        regret = jnp.max(realized, axis=-1) - selected_realized
        metrics = {
            "rmse": jnp.sqrt(jnp.mean(error**2)),
            "signed_bias": jnp.mean(error),
            "overprediction_fraction": jnp.mean(error > 2.0),
            "mean_action_regret": jnp.mean(regret),
            "p90_action_regret": jnp.quantile(regret, 0.90),
        }
        trajectories = {
            "initial_state": initial_states.x,
            "predicted_base": predicted_base,
            "realized_base": realized,
            "selected_action": selected,
            "action_regret": regret,
            "state": state_history,
            "action": actions,
            "base_reward": base_rewards,
            "train_reward": train_rewards,
            "done": done_history,
            "valid": valid_history,
            "reward_components": reward_components,
            "time_s": time_s,
            "goal_count": goal_count,
        }
        return metrics, trajectories

    success = final_state.success
    violation = final_state.arm_violation
    timeout = final_state.timeout
    success_count = jnp.sum(success)
    proportions = jnp.stack((jnp.mean(success), jnp.mean(violation), jnp.mean(timeout)))
    episode_count = jnp.asarray(success.size, dtype=jnp.float32)
    wilson_denominator = 1.0 + 1.959964**2 / episode_count
    wilson_centers = (proportions + 1.959964**2 / (2.0 * episode_count)) / wilson_denominator
    wilson_half_widths = (
        1.959964
        * jnp.sqrt(
            proportions * (1.0 - proportions) / episode_count
            + 1.959964**2 / (4.0 * episode_count**2)
        )
        / wilson_denominator
    )
    metrics = {
        "episodes": jnp.asarray(success.size),
        "successes": success_count,
        "success_rate": proportions[0],
        "success_rate_lower_95": wilson_centers[0] - wilson_half_widths[0],
        "success_rate_upper_95": wilson_centers[0] + wilson_half_widths[0],
        "arm_violation_rate": proportions[1],
        "arm_violation_rate_lower_95": wilson_centers[1] - wilson_half_widths[1],
        "arm_violation_rate_upper_95": wilson_centers[1] + wilson_half_widths[1],
        "timeout_rate": proportions[2],
        "timeout_rate_lower_95": wilson_centers[2] - wilson_half_widths[2],
        "timeout_rate_upper_95": wilson_centers[2] + wilson_half_widths[2],
        "mean_base_return": jnp.mean(totals[..., 0]),
        "mean_train_return": jnp.mean(totals[..., 1]),
        "mean_powered_s": jnp.mean(totals[..., 2]),
        "success_powered_s": jnp.where(
            success_count > 0,
            jnp.sum(totals[..., 2] * success) / success_count,
            jnp.nan,
        ),
        "mean_completion_s": jnp.mean(totals[..., 3]),
        "success_completion_s": jnp.where(
            success_count > 0,
            jnp.sum(totals[..., 3] * success) / success_count,
            jnp.nan,
        ),
        "mean_off_to_on": jnp.mean(totals[..., 4]),
        "mean_reversals": jnp.mean(totals[..., 5]),
        "mean_peak_energy_ratio": jnp.mean(totals[..., 7]),
        "maximum_arm_angle_rad": jnp.max(totals[..., 8]),
        "mean_absolute_torque_impulse_nm_s": jnp.mean(totals[..., 9]),
        "mean_decision_sampled_arm_excursion_s": jnp.mean(totals[..., 10]),
    }
    trajectories = {
        "initial_state": initial_states.x,
        "state": state_history,
        "action": actions,
        "base_reward": base_rewards,
        "train_reward": train_rewards,
        "done": done_history,
        "valid": valid_history,
        "reward_components": reward_components,
        "time_s": time_s,
        "goal_count": goal_count,
        "success": success,
        "arm_violation": violation,
        "timeout": timeout,
        "episode_totals": totals,
    }
    return metrics, trajectories
