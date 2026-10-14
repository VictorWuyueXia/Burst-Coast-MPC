"""Continue real heuristic or neural feedback after capture and record sustained balance."""

import json
import sys
from pathlib import Path

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import solve_discrete_are

from rotary_pendulum.environment.jax_dynamics import rk4_step
from rotary_pendulum.environment.jax_environment import ARM_LIMIT_RAD, EnvState, step
from rotary_pendulum.heuristic.decoder import decode
from rotary_pendulum.heuristic.energy import encode, policy
from rotary_pendulum.RL.jax_residual_control import filter_torque, residual_action
from rotary_pendulum.RL.jax_residual_evaluation import validation_resets
from rotary_pendulum.RL.jax_residual_task import observe
from rotary_pendulum.RL.jax_td3 import Actor, initialize


def rollout(state: EnvState, settings: dict, actor: dict | None) -> tuple[EnvState, dict]:
    """Run five seconds of feedback, including every post-capture physics sample."""

    def advance(current: EnvState, unused: None) -> tuple[EnvState, dict]:
        if "perturbation_times_s" in settings:
            kick = jnp.any(
                current.physics_steps[:, None]
                == 50 * jnp.asarray(settings["perturbation_times_s"]),
                axis=1,
            )
            sign = jnp.where(jnp.arange(current.x.shape[0]) % 2 == 0, 1.0, -1.0)
            perturbed = current.x + sign[:, None] * jnp.asarray(settings["perturbation_delta"])
            perturbed = perturbed.at[:, 1].set((perturbed[:, 1] + jnp.pi) % (2 * jnp.pi) - jnp.pi)
            current = current._replace(
                x=jnp.where(kick[:, None], perturbed, current.x),
                goal_count=jnp.where(kick, 0, current.goal_count),
            )
        if settings["controller"] == "policy":
            observed = current
            if "clock_clip_s" in settings:
                observed = current._replace(
                    physics_steps=jnp.minimum(current.physics_steps, settings["clock_clip_s"] * 50)
                )
            observation = observe(
                observed,
                jnp.full(current.x.shape[0], settings["training_deadline_s"] * 50),
                round(settings["hold_duration_s"] * 50),
                stationary=settings["contract_id"] == "rotary-stationary-td3-v3",
            )
            # Historical policies retain their physical observation normalization.
            ratio = ARM_LIMIT_RAD / settings["source_arm_limit_rad"]
            observation = observation.at[:, 0].multiply(ratio).at[:, 3].multiply(ratio)
            command = Actor(tuple(settings["hidden_widths"])).apply(actor, observation)
            torque = residual_action(observation, command, jnp.zeros_like(command), settings)[0]
        else:
            requested = policy(encode(current.x), settings["work_gain"])
            torque, _ = decode(
                current.x,
                requested,
                settings["work_weight"],
                settings["arm_limit_weight"],
                settings["torque_limit_nm"],
            )
            if settings["controller"].startswith("hybrid"):
                beta = jnp.arctan2(-jnp.sin(current.x[:, 1]), -jnp.cos(current.x[:, 1]))
                local = (abs(beta) < 0.25) & (abs(current.x[:, 3]) < 2)
                error = current.x.at[:, 1].set(beta)
                balance = settings["torque_limit_nm"] * jnp.clip(
                    -error @ jnp.asarray(settings["lqr_gain"]), -1, 1
                )
                torque = jnp.where(local, balance, torque)

        if settings["controller"] == "hybrid_filtered":
            torque, _ = filter_torque(
                current.x, torque, settings["filter_steps"], settings["torque_limit_nm"]
            )

        def physics(carry: EnvState, unused: None) -> tuple[EnvState, jax.Array]:
            following = step(
                carry,
                torque,
                physics_steps=1,
                max_physics_steps=settings["deadline_s"] * 50,
                goal_hold_steps=round(settings["hold_duration_s"] * 50),
                stop_on_success=False,
            )
            return following, following.x

        following, samples = jax.lax.scan(physics, current, None, length=5)
        return following, dict(
            start_x=current.x,
            physics_x=jnp.moveaxis(samples, 0, 1),
            torque_nm=torque,
            time_s=following.physics_steps * 0.02,
            goal_count=following.goal_count,
            arm_violation=following.arm_violation,
        )

    return jax.lax.scan(advance, state, None, length=50)


def main() -> None:
    """Save finite long-horizon trajectories with explicit policy/controller provenance."""
    settings = json.loads(Path(sys.argv[1]).read_text())
    if settings["controller"] not in ("heuristic", "hybrid", "hybrid_filtered", "policy"):
        raise ValueError("Declare heuristic, hybrid or policy controller")
    if settings["deadline_s"] % 5:
        raise ValueError("Use complete five-second progress chunks")
    root = Path(settings["run_dir"]) / "machine-scannables"
    root.mkdir(parents=True, exist_ok=False)
    equilibrium = jnp.array([0.0, -jnp.pi, 0.0, 0.0])
    discrete = lambda x, u: jax.lax.fori_loop(  # noqa: E731
        0, 5, lambda _, y: rk4_step(y, u * settings["torque_limit_nm"]), x
    )
    a, b = jax.jacfwd(discrete, argnums=(0, 1))(equilibrium, jnp.array(0.0))
    a, b = np.asarray(a, float), np.asarray(b, float)[:, None]
    value = solve_discrete_are(
        a, b, np.diag([1 / float(ARM_LIMIT_RAD) ** 2, 1 / 0.15**2, 0.25, 0.25]), np.ones((1, 1))
    )
    gain = np.linalg.solve(1 + b.T @ value @ b, b.T @ value @ a)[0]
    settings.update(
        lqr_gain=gain.tolist(),
        arm_limit_rad=float(ARM_LIMIT_RAD),
        integration_dt_s=0.002,
        lqr_closed_loop_eigenvalues_abs=abs(np.linalg.eigvals(a - b @ gain[None])).tolist(),
    )
    (root / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    initial, _, labels = validation_resets(
        settings["validation_seed"], settings["episodes_per_stratum"], [settings["deadline_s"]]
    )
    if "reset_suite" in settings:
        from rotary_pendulum.RL.jax_stationary_task import targeted_resets

        if settings["reset_suite"] != "targeted":
            raise ValueError("Unknown reset suite")
        initial, groups = targeted_resets(
            jax.random.PRNGKey(settings["validation_seed"]), 4 * settings["episodes_per_stratum"]
        )
        labels = np.asarray(["downward", "capture", "braking", "recovery"])[np.asarray(groups)]
    actor = None
    if settings["controller"] == "policy":
        learner = flax.serialization.from_bytes(
            initialize(jax.random.PRNGKey(settings["seed"]), settings),
            Path(settings["checkpoint_path"]).read_bytes(),
        )
        actor = learner["actor"]
    execute = jax.jit(lambda state: rollout(state, settings, actor))
    state, chunks = initial, []
    print(
        f"START {settings['controller']} cap={settings['torque_limit_nm']} "
        f"lanes={len(labels)} horizon={settings['deadline_s']}s; compiling",
        flush=True,
    )
    for chunk in range(settings["deadline_s"] // 5):
        state, trace = execute(state)
        host = jax.device_get(trace)
        if not all(np.isfinite(value).all() for value in host.values()):
            np.savez_compressed(root / "numerical_failure.npz", **host)
            raise FloatingPointError(f"Nonfinite trajectory at {(chunk + 1) * 5}s")
        chunks.append(host)
        if chunk % 4 == 0 or (chunk + 1) * 5 == settings["deadline_s"]:
            print(
                f"t={(chunk + 1) * 5}s ever_hold={int(state.success.sum())}/{len(labels)} "
                f"current_hold_max={float(state.goal_count.max()) * 0.02:.2f}s "
                f"arm_crossings={int(state.arm_violation.sum())}",
                flush=True,
            )
    traces = {name: np.concatenate([chunk[name] for chunk in chunks]) for name in chunks[0]}
    np.savez_compressed(root / "trajectories.npz", **traces, labels=np.asarray(labels))
    print(f"DONE: {root}", flush=True)


if __name__ == "__main__":
    main()
