"""Qualify successful teacher episodes and rebuild matched TD3 transitions without leakage."""

import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from rotary_pendulum.environment.jax_environment import GOAL, EnvState
from rotary_pendulum.RL.jax_residual_task import observe, transition


def main() -> None:
    """Audit holds, recompute task rewards, and split by entire initial-state episodes."""
    manifest = json.loads(Path(sys.argv[1]).read_text())
    source, root = Path(manifest["teacher_dir"]), Path(manifest["output_dir"])
    settings = manifest["settings"]
    assert manifest["goal"]["torque_limit_nm"] == settings["torque_limit_nm"]
    teacher_settings = json.loads((source / "machine-scannables/settings.json").read_text())
    assert teacher_settings["torque_limit_nm"] == settings["torque_limit_nm"]
    root.mkdir(parents=True, exist_ok=False)
    archive = np.load(source / "machine-scannables/trajectories.npz", allow_pickle=False)
    traces = {name: archive[name] for name in archive.files if name != "allow_pickle"}
    archive.close()
    rows = list(csv.DictReader((source / "machine-scannables/episodes.csv").open()))
    labels = np.asarray([row["stratum"] for row in rows])
    decisions, lanes = traces["torque_nm"].shape
    active = traces["physics_active"]
    beta = np.arctan2(
        np.sin(traces["physics_x"][..., 1] - np.pi),
        np.cos(traces["physics_x"][..., 1] - np.pi),
    )
    inside = (abs(beta) < GOAL.beta_tolerance_rad) & active
    hold, maximum = np.zeros(lanes, int), np.zeros(lanes, int)
    for membership, enabled in zip(
        inside.transpose(0, 2, 1).reshape(-1, lanes),
        active.transpose(0, 2, 1).reshape(-1, lanes),
        strict=True,
    ):
        hold = np.where(enabled, np.where(membership, hold + 1, 0), hold)
        maximum = np.maximum(maximum, hold)
    np.testing.assert_array_equal(maximum >= GOAL.hold_steps, traces["success"][-1])
    selected = traces["success"][-1] & (labels != "probe")
    # Split before flattening; no frames from one trajectory may enter both splits.
    validation = np.arange(lanes) % 5 == 0
    print(f"Qualified {selected.sum()}/{sum(labels != 'probe')} complete episodes", flush=True)
    for row, length, split in zip(rows, maximum, validation, strict=True):
        row["longest_hold_s"] = float(length * 0.02)
        row["split"] = "validation" if split else "train"
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (root / "episodes.csv").write_text(buffer.getvalue())
    np.savez_compressed(root / "episode_starts.npz", x=traces["start_x"][0], labels=labels)
    zero = np.zeros((1, lanes), np.int32)
    prior_hold = np.concatenate((zero, traces["goal_count"][:-1]))
    prior_success = np.concatenate((zero.astype(bool), traces["success"][:-1]))
    prior_timeout = np.concatenate((zero.astype(bool), traces["timeout"][:-1]))
    prior_arm = np.concatenate((zero.astype(bool), traces["arm_violation"][:-1]))
    summaries = {}
    for deadline_s in (30, 40):
        deadline = jnp.full((10, lanes), deadline_s * 50, jnp.int32)
        rebuild = jax.jit(
            lambda state, torque, horizon=deadline: transition(state, torque, horizon, settings)
        )
        chunks = []
        for index in range(0, deadline_s * 10, 10):
            section = slice(index, index + 10)
            state = EnvState(
                jnp.asarray(traces["start_x"][section]),
                jnp.asarray(
                    np.rint((traces["time_s"][section] - traces["elapsed_s"][section]) * 50),
                    jnp.int32,
                ),
                jnp.asarray(prior_hold[section]),
                jnp.asarray(prior_success[section]),
                jnp.asarray(prior_arm[section]),
                jnp.asarray(prior_timeout[section]),
            )
            following, reward, _, _, next_observation = rebuild(
                state, jnp.asarray(traces["torque_nm"][section])
            )
            np.testing.assert_array_equal(following.success, traces["success"][section])
            chunks.append(
                jax.device_get(
                    {
                        "observation": observe(state, deadline),
                        "next_observation": next_observation,
                        "action": traces["torque_nm"][section] / settings["torque_limit_nm"],
                        "reward": reward,
                        "done": following.success | following.timeout,
                    }
                )
            )
            if index % 100 == 0:
                print(f"Rebuilt {deadline_s}s: {index + 10} decisions", flush=True)
        dataset = {name: np.concatenate([chunk[name] for chunk in chunks]) for name in chunks[0]}
        dataset["return"] = np.cumsum(dataset["reward"][::-1], axis=0)[::-1].copy()
        qualified = selected & (traces["time_s"][-1] <= deadline_s + 1e-5)
        for split, included in (("train", ~validation), ("validation", validation)):
            mask = (traces["elapsed_s"][: deadline_s * 10] > 0) & (qualified & included)[None, :]
            values = {name: value[mask] for name, value in dataset.items()}
            values["episode_id"] = np.broadcast_to(np.arange(lanes), mask.shape)[mask]
            if not mask.any() or not all(np.isfinite(value).all() for value in values.values()):
                raise ValueError(f"Invalid demonstration split: {deadline_s}/{split}")
            np.savez_compressed(root / f"{split}-{deadline_s}s.npz", **values)
            summaries[f"{split}-{deadline_s}s"] = {
                "episodes": int(np.sum(qualified & included)),
                "transitions": int(mask.sum()),
                "terminal_transitions": int(values["done"].sum()),
                "max_normalized_action": float(abs(values["action"]).max()),
            }
            assert summaries[f"{split}-{deadline_s}s"]["episodes"] == int(values["done"].sum())
    manifest["summary"] = summaries
    manifest["source_sha256"] = hashlib.sha256(
        (source / "machine-scannables/trajectories.npz").read_bytes()
    ).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
