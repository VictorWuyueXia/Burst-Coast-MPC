"""Reconstruct stationary discounted teacher targets and balanced physical recovery replay."""

import json
import sys
from pathlib import Path

import numpy as np

from rotary_pendulum.environment.jax_dynamics import MODEL
from rotary_pendulum.environment.jax_environment import ARM_LIMIT_RAD
from rotary_pendulum.RL.jax_residual_control import ARM_SPEED_SCALE, PENDULUM_SPEED_SCALE


def main() -> None:
    """Qualify whole live episodes, then balance five declared transition categories."""
    config = json.loads(Path(sys.argv[1]).read_text())
    source = Path(config["source"]) / "machine-scannables"
    settings = config["settings"]
    metadata = json.loads((source / "settings.json").read_text())
    assert metadata["torque_limit_nm"] == settings["torque_limit_nm"]
    assert metadata["arm_limit_rad"] == float(ARM_LIMIT_RAD)
    assert metadata["integration_dt_s"] == 0.002
    archive = np.load(source / "trajectories.npz")
    records = {key: archive[key] for key in archive.files}
    archive.close()
    states, starts = records["physics_x"], records["start_x"]
    steps, lanes = starts.shape[:2]
    selected = (~records["arm_violation"][-1]) & (records["goal_count"][-1] >= 500)
    assert selected.any(), "No fully compliant settled teacher trajectories"
    observations = []
    for state in [starts, states[:, :, -1]]:
        observations.append(
            np.stack(
                (
                    state[..., 0] / float(ARM_LIMIT_RAD),
                    np.sin(state[..., 1]),
                    np.cos(state[..., 1]),
                    state[..., 2] / float(ARM_SPEED_SCALE),
                    state[..., 3] / float(PENDULUM_SPEED_SCALE),
                ),
                axis=-1,
            ).astype(np.float32)
        )
    beta = np.arctan2(-np.sin(states[..., 1]), -np.cos(states[..., 1]))
    upright = np.exp(
        -0.5
        * np.sum(
            (
                np.stack((beta, states[..., 3], states[..., 2]), axis=-1)
                / np.asarray(settings["upright_widths"])
            )
            ** 2,
            axis=-1,
        )
    )
    energy = 0.5 * MODEL.pendulum_inertia_kg_m2 * states[..., 3] ** 2 + MODEL.gravity_torque_nm * (
        1 - np.cos(states[..., 1])
    )
    arm = (states[..., 0] / float(ARM_LIMIT_RAD)) ** 2 + (
        np.maximum(abs(states[..., 0]) - 0.8 * float(ARM_LIMIT_RAD), 0)
        / (0.2 * float(ARM_LIMIT_RAD))
    ) ** 2
    base = (
        -settings["upright_reward_per_s"] * (1 - upright)
        - settings["energy_cost_per_s"] * abs(energy / (2 * MODEL.gravity_torque_nm) - 1)
        - settings["arm_cost_per_s"] * arm
    )
    action = records["torque_nm"] / settings["torque_limit_nm"]
    assert abs(action).max() <= 1 + 1e-7
    start_beta = np.arctan2(-np.sin(starts[..., 1]), -np.cos(starts[..., 1]))
    # Categories describe physical decisions, with recovery origin retained for its transient.
    category = np.zeros((steps, lanes), np.int32)
    category[abs(start_beta) < 0.35] = 1
    category[(records["labels"][None, :] == "recovery") & (np.arange(steps)[:, None] < 50)] = 3
    category[
        (abs(starts[..., 0]) > 0.55 * float(ARM_LIMIT_RAD)) & (starts[..., 0] * starts[..., 2] > 0)
    ] = 2
    quiet = (abs(start_beta) < 0.03) & (abs(starts[..., 2]) < 0.15) & (abs(starts[..., 3]) < 0.15)
    category[quiet] = 4
    episode_id = np.broadcast_to(np.arange(lanes), (steps, lanes))
    splits = {}
    for split in ["train", "validation"]:
        enabled = selected & (
            (np.arange(lanes) % 5 == 0) if split == "validation" else (np.arange(lanes) % 5 != 0)
        )
        rng = np.random.default_rng(config["sampling_seed"])
        selection, counts = [], {}
        for group, name in enumerate(["swing", "capture", "braking", "recovery", "quiet"]):
            pool = np.flatnonzero((category == group) & enabled[None, :])
            assert len(pool), f"No qualified {split} {name} samples"
            amount = 50000 if split == "train" else 5000
            chosen = rng.choice(pool, size=amount, replace=len(pool) < amount)
            selection.append(chosen)
            counts[name] = dict(available=len(pool), sampled=amount, unique=len(np.unique(chosen)))
        splits[split] = (np.concatenate(selection), counts)
    for torque_cost in config["torque_costs"]:
        machine = Path(config["output_dir"]) / f"torque_{torque_cost:g}" / "machine-scannables"
        machine.mkdir(parents=True, exist_ok=False)
        reward = (
            (0.02 * (base - torque_cost * action[..., None] ** 2)).sum(axis=2).astype(np.float32)
        )
        returns = np.empty_like(reward)
        tail = reward[-100:].mean(axis=0) / (1 - settings["discount"])
        for index in range(steps - 1, -1, -1):
            tail = reward[index] + settings["discount"] * tail
            returns[index] = tail
        data = dict(
            observation=observations[0],
            next_observation=observations[1],
            action=action,
            reward=reward,
            done=np.zeros_like(reward, bool),
            **{"return": returns},
            episode_id=episode_id,
            category=category,
        )
        for split, (indices, _) in splits.items():
            np.savez_compressed(
                machine / f"{split}.npz",
                **{
                    key: value.reshape((-1, *value.shape[2:]))[indices]
                    for key, value in data.items()
                },
            )
        manifest = dict(
            source=str(source.parent),
            contract=settings
            | dict(
                torque_cost_per_s=torque_cost,
                arm_limit_rad=float(ARM_LIMIT_RAD),
                integration_dt_s=0.002,
            ),
            qualified_by_stratum={
                label: int((selected & (records["labels"] == label)).sum())
                for label in ["downward", "capture", "braking", "recovery"]
            },
            rejected_episodes=int((~selected).sum()),
            sampling={split: counts for split, (_, counts) in splits.items()},
            return_boundary=(
                "Discounted observed 120s trajectory plus constant final10s "
                "mean-reward tail; approximate continuing teacher return, not terminal."
            ),
            maximum_qualified_abs_tail_estimate=float(
                abs(reward[-100:, selected].mean(axis=0) / (1 - settings["discount"])).max()
            ),
        )
        (machine / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Prepared {machine}; qualified={manifest['qualified_by_stratum']}", flush=True)


if __name__ == "__main__":
    main()
