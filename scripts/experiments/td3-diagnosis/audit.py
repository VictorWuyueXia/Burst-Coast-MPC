"""Evaluate a saved TD3 checkpoint on fresh resets and inspect its action/value behavior."""

import csv
import io
import json
import sys
from pathlib import Path

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import solve_discrete_are

from rotary_pendulum.environment.jax_dynamics import rk4_step
from rotary_pendulum.RL.jax_residual_control import POLICY_TORQUE_NM, residual_action
from rotary_pendulum.RL.jax_residual_evaluation import evaluate, validation_resets, write_validation
from rotary_pendulum.RL.jax_residual_task import observe
from rotary_pendulum.RL.jax_td3 import Actor, Critic, initialize


def main() -> None:
    """Audit both capture definitions, zero control, critic calibration and action gradients."""
    settings = json.loads(Path(sys.argv[1]).read_text())
    root = Path(settings["run_dir"])
    machine = root / "machine-scannables"
    machine.mkdir(parents=True, exist_ok=False)
    learner = flax.serialization.from_bytes(
        initialize(jax.random.PRNGKey(settings["seed"]), settings),
        Path(settings["checkpoint_path"]).read_bytes(),
    )
    initial, deadlines, labels = validation_resets(
        settings["validation_seed"], settings["episodes_per_stratum"], [settings["deadline_s"]]
    )
    observation = observe(initial, deadlines)
    actor = Actor(tuple(settings["hidden_widths"]))
    critic = Critic(tuple(settings["hidden_widths"]))
    command = actor.apply(learner["actor"], observation)
    torque = residual_action(observation, command, jnp.zeros_like(command), settings)[0]
    predicted = np.asarray(
        critic.apply(learner["critics"][0], observation, torque / POLICY_TORQUE_NM)
    )
    reports = []
    equilibrium = jnp.array([0.0, -jnp.pi, 0.0, 0.0])
    discrete = lambda x, u: jax.lax.fori_loop(  # noqa: E731
        0, 5, lambda _, state: rk4_step(state, u * POLICY_TORQUE_NM), x
    )
    a, b = jax.jacfwd(discrete, argnums=(0, 1))(equilibrium, jnp.array(0.0))
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)[:, None]
    penalty = np.diag(1 / np.array([0.08, 0.08, 0.15, 0.20]) ** 2)
    value = solve_discrete_are(a, b, penalty, np.ones((1, 1)))
    gain = np.linalg.solve(1 + b.T @ value @ b, b.T @ value @ a)[0]
    np.savez_compressed(
        machine / "local_linear_controller.npz",
        A=a,
        B=b,
        state_penalty=penalty,
        gain=gain,
        closed_loop_eigenvalues=np.linalg.eigvals(a - b @ gain[None, :]),
    )
    for goal in ("angle", "strict"):
        for mode in ("policy", "zero"):
            print(f"audit goal={goal} mode={mode}: compiling/evaluating", flush=True)
            task = settings | {"goal_mode": goal}
            traces = jax.device_get(
                jax.jit(
                    lambda task_=task, mode_=mode: evaluate(
                        initial,
                        deadlines,
                        task_,
                        mode_,
                        learner["actor"],
                    )
                )()
            )
            if not (np.isfinite(traces["physics_x"]).all() and np.isfinite(traces["reward"]).all()):
                raise FloatingPointError(f"Nonfinite checkpoint audit: {goal}/{mode}")
            rows = write_validation(root / f"{goal}-{mode}", traces, labels, task)
            for label in ("downward", "moving", "near", "tight"):
                subset = [row for row in rows if row["stratum"] == label]
                reports.append(
                    {
                        "goal": goal,
                        "controller": mode,
                        "stratum": label,
                        **{
                            name: float(np.mean([row[name] for row in subset]))
                            for name in ("success", "return", "duration_s", "arm_violation")
                        },
                    }
                )
            if goal == settings["goal_mode"] and mode == "policy":
                realized = np.asarray(traces["reward"]).sum(axis=0)
                np.savez_compressed(
                    machine / "value_calibration.npz",
                    predicted=predicted,
                    realized=realized,
                    labels=np.asarray(labels),
                    observation=np.asarray(observation),
                )
    print("audit local linear controller: tight starts only", flush=True)
    count = settings["episodes_per_stratum"]
    tight = jax.tree.map(lambda value: value[3 * count : 4 * count], initial)
    local = jax.device_get(
        jax.jit(
            lambda: evaluate(
                tight,
                deadlines[3 * count : 4 * count],
                settings | {"goal_mode": "strict"},
                "linear",
                gain,
            )
        )()
    )
    if not np.isfinite(local["physics_x"]).all():
        raise FloatingPointError("Local linear benchmark left its valid numerical domain")
    np.savez_compressed(machine / "local_linear_traces.npz", **local)
    reports.append(
        {
            "goal": "strict",
            "controller": "linear",
            "stratum": "tight",
            "success": float(local["success"].mean()),
            "return": float(local["reward"].sum(axis=0).mean()),
            "duration_s": float(local["duration_s"].mean()),
            "arm_violation": float(local["arm_violation"].mean()),
        }
    )
    source = Path(settings["checkpoint_path"]).parent
    replay = np.load(source / "replay_sample.npz")
    samples = jnp.asarray(replay["observation"])
    residual = actor.apply(learner["actor"], samples)
    sensitivity = jax.jit(
        jax.vmap(
            jax.grad(
                lambda obs, action: (
                    residual_action(obs, action, jnp.array(0), settings)[0] / POLICY_TORQUE_NM
                ),
                argnums=1,
            )
        )
    )(samples, residual)
    diagnostics = {
        "replay_action_gradient_nonzero": float(jnp.mean(jnp.abs(sensitivity) > 1e-6)),
        "actor_output_saturated": float(jnp.mean(jnp.abs(residual) > 0.99)),
        "replay_terminal_fraction": float(np.mean(replay["done"])),
        "replay_success_fraction": float(np.mean(replay["next_observation"][:, 7] >= 1)),
        "mean_predicted_return": float(predicted[:-4].mean()),
        "mean_realized_return": float(realized[:-4].mean()),
        "value_rmse": float(np.sqrt(np.mean((predicted[:-4] - realized[:-4]) ** 2))),
    }
    (machine / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    (machine / "settings.json").write_text(json.dumps(settings, indent=2))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(reports[0]))
    writer.writeheader()
    writer.writerows(reports)
    (machine / "outcomes.csv").write_text(buffer.getvalue())
    print(json.dumps({"diagnostics": diagnostics, "outcomes": reports}, indent=2), flush=True)


if __name__ == "__main__":
    main()
