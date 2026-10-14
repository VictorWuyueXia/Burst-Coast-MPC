"""Build matched short-prefix and extended-balance replay from real live teacher records."""

import json
import sys
from pathlib import Path

import numpy as np

from rotary_pendulum.environment.jax_dynamics import MODEL
from rotary_pendulum.environment.jax_environment import ARM_LIMIT_RAD
from rotary_pendulum.RL.jax_residual_control import ARM_SPEED_SCALE, PENDULUM_SPEED_SCALE


def main() -> None:
    """Compute rewards/returns at each requested arm weight, keeping whole-episode splits."""
    config = json.loads(Path(sys.argv[1]).read_text())
    source, output = Path(config["source"]), Path(config["output_dir"])
    settings = config["settings"]
    metadata = json.loads((source / "machine-scannables/settings.json").read_text())
    assert metadata["torque_limit_nm"] == settings["torque_limit_nm"]
    assert metadata["arm_limit_rad"] == float(ARM_LIMIT_RAD)
    assert metadata["integration_dt_s"] == 0.002
    archive = np.load(source / "machine-scannables/trajectories.npz")
    records = {key: archive[key] for key in archive.files}
    archive.close()
    decisions, lanes = records["torque_nm"].shape
    assert decisions == settings["deadline_s"] * 10
    states = records["physics_x"]
    beta = np.arctan2(-np.sin(states[..., 1]), -np.cos(states[..., 1]))
    inside = (abs(beta) < np.deg2rad(5)).transpose(0, 2, 1).reshape(-1, lanes)
    hold = np.zeros(lanes, int)
    first = np.full((2, lanes), decisions * 5 + 1, int)
    for index, membership in enumerate(inside):
        hold = np.where(membership, hold + 1, 0)
        for k, threshold in enumerate([50, round(settings["hold_duration_s"] * 50)]):
            first[k] = np.where(
                (hold >= threshold) & (first[k] > decisions * 5), index + 1, first[k]
            )
    physical_index = np.arange(1, decisions * 5 + 1).reshape(decisions, 1, 5)
    active = physical_index <= first[1][None, :, None]
    crossed = ((abs(states[..., 0]) >= float(ARM_LIMIT_RAD)) & active).any(axis=(0, 2))
    selected = (first[1] <= decisions * 5) & ~crossed & (records["labels"] != "probe")
    assert selected.any(), "No bounded sustained episodes available for demonstrations"
    goal_before = np.concatenate((np.zeros((1, lanes), int), records["goal_count"][:-1]))
    final_index = np.clip(first[1][None, :] - np.arange(decisions)[:, None] * 5 - 1, 0, 4)
    following = np.take_along_axis(states, final_index[..., None, None], axis=2)[..., 0, :]
    elapsed = np.arange(decisions)[:, None] * 5 + np.zeros((1, lanes), int)
    next_elapsed = np.minimum(elapsed + 5, first[1][None, :])
    observation, next_observation = [], []
    for state, steps, count in [
        (records["start_x"], elapsed, goal_before),
        (
            following,
            next_elapsed,
            np.where(
                next_elapsed >= first[1][None, :],
                settings["hold_duration_s"] * 50,
                records["goal_count"],
            ),
        ),
    ]:
        observation_value = np.stack(
            (
                state[..., 0] / float(ARM_LIMIT_RAD),
                np.sin(state[..., 1]),
                np.cos(state[..., 1]),
                state[..., 2] / float(ARM_SPEED_SCALE),
                state[..., 3] / float(PENDULUM_SPEED_SCALE),
                steps / 3000,
                (settings["deadline_s"] * 50 - steps) / 3000,
                np.minimum(count / (settings["hold_duration_s"] * 50), 1),
            ),
            axis=-1,
        ).astype(np.float32)
        if not observation:
            observation.append(observation_value)
        else:
            next_observation.append(observation_value)
    energy = 0.5 * MODEL.pendulum_inertia_kg_m2 * states[..., 3] ** 2 + MODEL.gravity_torque_nm * (
        1 - np.cos(states[..., 1])
    )
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
    torque = records["torque_nm"] / settings["torque_limit_nm"]
    assert abs(torque).max() <= 1 + 1e-7
    base_rates = (
        -settings["energy_cost_per_s"] * abs(energy / (2 * MODEL.gravity_torque_nm) - 1)
        - settings["torque_cost_per_s"] * abs(torque[..., None])
        - settings["time_cost_per_s"] * (1 + physical_index / (settings["deadline_s"] * 50))
        + settings["upright_reward_per_s"] * upright
    )
    arm_rates = (states[..., 0] / float(ARM_LIMIT_RAD)) ** 2 + (
        np.maximum(abs(states[..., 0]) - 0.8 * float(ARM_LIMIT_RAD), 0)
        / (0.2 * float(ARM_LIMIT_RAD))
    ) ** 2
    for weight in config["arm_weights"]:
        reward = (
            (np.where(active, 0.02 * (base_rates - weight * arm_rates), 0))
            .sum(axis=2)
            .astype(np.float32)
        )
        returns = np.cumsum(reward[::-1], axis=0)[::-1].copy()
        done = next_elapsed >= first[1][None, :]
        data = dict(
            observation=observation[0],
            next_observation=next_observation[0],
            action=torque,
            reward=reward,
            done=done,
            **{"return": returns},
            episode_id=np.broadcast_to(np.arange(lanes), (decisions, lanes)),
        )
        for scheme in config["schemes"]:
            machine = output / f"arm_{weight:g}_{scheme}" / "machine-scannables"
            machine.mkdir(parents=True, exist_ok=False)
            stop = first[0] if scheme == "short" else first[1]
            enabled = (elapsed < stop[None, :]) & selected[None, :]
            if scheme == "balanced":
                enabled &= (elapsed < first[0][None, :]) | (elapsed % 50 == 0) | done
            summaries = {}
            for split in ["train", "validation"]:
                lane_mask = (
                    (np.arange(lanes) % 5 == 0)
                    if split == "validation"
                    else (np.arange(lanes) % 5 != 0)
                )
                mask = enabled & lane_mask[None, :]
                assert mask.any()
                np.savez_compressed(
                    machine / f"{split}-{settings['deadline_s']}s.npz",
                    **{key: value[mask] for key, value in data.items()},
                )
                summaries[split] = dict(
                    transitions=int(mask.sum()),
                    episodes=int((selected & lane_mask).sum()),
                    terminal_transitions=int(done[mask].sum()),
                )
            goal = dict(
                angle_degrees=5,
                torque_limit_nm=settings["torque_limit_nm"],
                arm_limit_rad=float(ARM_LIMIT_RAD),
                integration_dt_s=0.002,
                hold_duration_s=settings["hold_duration_s"],
                arm_cost_per_s=weight,
            )
            manifest = dict(
                source=str(source),
                scheme=scheme,
                goal=goal,
                settings=settings | dict(arm_cost_per_s=weight),
                summary=summaries,
                qualified_by_stratum={
                    label: int((selected & (records["labels"] == label)).sum())
                    for label in ["downward", "moving", "near", "tight"]
                },
                short_prefix_semantics=(
                    "Prefix ends after first 1 s hold; nonterminal for the 10 s task. "
                    "Return targets use the actual extended teacher continuation."
                ),
            )
            (machine / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            np.savez_compressed(
                machine / "episode_starts.npz",
                x=records["start_x"][0],
                selected=selected,
                split_validation=np.arange(lanes) % 5 == 0,
            )
            print(
                f"Prepared cap={settings['torque_limit_nm']} arm={weight} {scheme}: {summaries}",
                flush=True,
            )


if __name__ == "__main__":
    main()
