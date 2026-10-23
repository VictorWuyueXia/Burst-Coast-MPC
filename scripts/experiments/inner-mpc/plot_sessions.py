"""Plot executed saved MPC sessions; no new simulations or solver calls."""

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import LogNorm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("artifacts/rotary_pendulum/inner-mpc")
output = root / "20261020-session-plots"
human, machine = output / "human-readables", output / "machine-scannables"
human.mkdir(parents=True, exist_ok=True)
machine.mkdir(exist_ok=True)
controllers = [
    dict(
        run=f"20261019-gpu-closed-{solver}",
        configuration=c,
        label=f"{solver.upper()}, w={weight:g}",
        weight=weight,
    )
    for solver in ("cem", "sqp")
    for c, weight in enumerate((0, 0.1, 1, 10))
] + [dict(run="20261019-gpu-confirm-cem", configuration=0, label="CEM, w=1, spread=0.8", weight=1)]
reference = json.loads(
    (root / "20261019-gpu-comparison/machine-scannables/episodes.json").read_text()
)
paths, torques, settings, hashes = [], [], [], {}
initial = None
for controller in controllers:
    folder = root / controller["run"] / "machine-scannables"
    design = json.loads((folder / "design.json").read_text())
    config = design["configurations"][controller["configuration"]]
    settings.append(config)
    assert design["decisions"] == 50 and config["horizon"] == 10
    archive = np.load(folder / "starts.npz", allow_pickle=False)
    starts, groups = archive["states"], archive["groups"]
    archive.close()
    if initial is None:
        initial = starts
    np.testing.assert_array_equal(initial, starts)
    states, actions = [starts[:, None]], []
    step_paths = sorted((folder / f"config-{controller['configuration']:02d}").glob("step-*"))
    assert len(step_paths) == 50
    for step in step_paths:
        archive = np.load(step / "input.npz", allow_pickle=False)
        np.testing.assert_allclose(archive["states"], states[-1][:, -1], atol=1e-12)
        assert archive["alive"].all()
        archive.close()
        archive = np.load(step / "plans.npz", allow_pickle=False)
        index = np.arange(32), archive["selected"]
        assert archive["accepted"][index].all() and archive["physical_pass"][index].all()
        # Execute only the first decision, never concatenate the future prediction horizons.
        states.append(archive["refined"][index][:, 0])
        actions.append(archive["plans"][index][:, 0] * config["plant"]["torque_limit_nm"])
        archive.close()
    paths.append(np.concatenate(states, axis=1))
    torques.append(np.stack(actions, axis=1))
    for path in (folder / "design.json", folder / "starts.npz"):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"Extracted and verified {controller['label']}: 32 sessions x 50 actions", flush=True)

states, torque = np.stack(paths), np.stack(torques)
time = np.arange(5001) * 0.001
error = np.rad2deg(np.abs(np.arctan2(-np.sin(states[..., 1]), -np.cos(states[..., 1]))))
arm = np.rad2deg(states[..., 0])
final_rms = np.sqrt(np.mean(error[..., -1000:] ** 2, axis=-1))
hold = np.all(error[..., -1000:] < 5, axis=-1)
assert states.shape == (9, 32, 5001, 4) and np.isfinite(states).all()
assert np.all(np.abs(arm) < 135) and np.all(np.abs(torque) <= 0.00459)
for k, controller in enumerate(controllers):
    expected = sorted(
        [
            r
            for r in reference
            if r["run"] == controller["run"] and r["configuration"] == controller["configuration"]
        ],
        key=lambda r: r["episode"],
    )
    np.testing.assert_allclose(
        final_rms[k], np.rad2deg([r["final_second_upright_rms_rad"] for r in expected]), atol=1e-10
    )
    np.testing.assert_array_equal(hold[k], [r["final_second_in_5deg"] for r in expected])

# Explicitly distinguish best-case selection from representative and paired comparisons.
best = np.argmin(final_rms, axis=-1)
matched = []
for group in range(4):
    ids = np.flatnonzero(groups == group)
    order = ids[np.argsort(final_rms[3, ids], kind="stable")]
    matched.append(int(order[(len(order) - 1) // 2]))
selection = dict(
    controllers=controllers,
    settings=settings,
    source_sha256=hashes,
    best_episodes=best.tolist(),
    best_rule="Minimum final-second upright-error RMS within each controller's 32 sessions",
    matched_episodes=matched,
    matched_rule="Lower median final-second upright RMS within each start group under CEM w=10",
    groups=["Downward", "Moving swing-up", "Near upright", "Outward arm"],
    final_second_hold=hold.sum(axis=-1).tolist(),
    definition="Upright distance: 0 deg upright, 180 deg downward; w is centering cost rate",
    note=(
        "Five seconds of executed 1 ms replay; no new simulation; "
        "samples are not independent episodes"
    ),
)
(machine / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
np.savez_compressed(
    machine / "executed_sessions.npz",
    time_s=time,
    states=states,
    torque_nm=torque,
    action_edges_s=np.arange(51) * 0.1,
    groups=groups,
    upright_distance_deg=error,
    final_rms_deg=final_rms,
    hold=hold,
)
(machine / "plot_sessions.py").write_bytes(Path(__file__).read_bytes())
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
colors = ["#0072B2", "#D55E00", "#6B7280", "#CC79A7"]

# Each controller gets its own best episode; do not interpret these as paired performance.
figure, axes = plt.subplots(3, 4, figsize=(16, 8), sharex=True, constrained_layout=True)
for column, controller_id in enumerate((3, 7, 0, 4)):
    episode = int(best[controller_id])
    axes[0, column].set_title(
        f"{controllers[controller_id]['label']} — best episode {episode}\n"
        f"{selection['groups'][int(groups[episode])]} | "
        f"final RMS {final_rms[controller_id, episode]:.2f}°"
    )
    axes[0, column].plot(time, error[controller_id, episode], color=colors[column], lw=1.5)
    axes[1, column].plot(time, arm[controller_id, episode], color=colors[column], lw=1.5)
    axes[2, column].stairs(
        torque[controller_id, episode] * 1000,
        np.arange(51) * 0.1,
        baseline=None,
        color=colors[column],
        lw=1.3,
    )
    axes[0, column].axhspan(0, 5, color="#009E73", alpha=0.18)
    axes[0, column].axhline(5, color="#009E73", ls=":", lw=1)
    axes[0, column].set(ylim=(0, 180))
    axes[1, column].axhline(135, color="black", ls="--", lw=0.8)
    axes[1, column].axhline(-135, color="black", ls="--", lw=0.8)
    axes[1, column].set(ylim=(-145, 145), yticks=[-135, 0, 135])
    axes[2, column].set(ylim=(-4.9, 4.9), xlabel="Executed time (s)", yticks=[-4.59, 0, 4.59])
    for axis in axes[:, column]:
        axis.axvspan(4, 5, color="#E6AB02", alpha=0.08)
        axis.grid(alpha=0.2)
        axis.set_xlim(0, 5)
for row, label in enumerate(
    ("Distance from upright (°)", "Arm angle (°)", "Applied torque (mN m)")
):
    axes[row, 0].set_ylabel(label)
figure.suptitle(
    "Best individual sessions — independently selected, different initial states\n"
    "Green band: within 5° of upright; gold region: final-second scoring window",
    fontsize=13,
)
for extension in ("png", "pdf"):
    figure.savefig(human / f"best_sessions.{extension}", dpi=170)
plt.close(figure)

figure, axes = plt.subplots(3, 4, figsize=(16, 8), sharex=True, sharey="row", layout="constrained")
for column, episode in enumerate(matched):
    axes[0, column].set_title(f"{selection['groups'][column]} — episode {episode}")
    for color, controller_id in zip(colors, (3, 7, 0, 8), strict=True):
        label = controllers[controller_id]["label"]
        axes[0, column].plot(time, error[controller_id, episode], color=color, label=label, lw=1.3)
        axes[1, column].plot(time, arm[controller_id, episode], color=color, lw=1.3)
        axes[2, column].stairs(
            torque[controller_id, episode] * 1000,
            np.arange(51) * 0.1,
            baseline=None,
            color=color,
            lw=1.1,
        )
    axes[0, column].axhspan(0, 5, color="#009E73", alpha=0.15)
    axes[0, column].set(ylim=(0, 180), yticks=[0, 45, 90, 135, 180])
    axes[1, column].axhline(135, color="black", ls="--", lw=0.8)
    axes[1, column].axhline(-135, color="black", ls="--", lw=0.8)
    axes[1, column].set(ylim=(-145, 145), yticks=[-135, 0, 135])
    axes[2, column].set(ylim=(-4.9, 4.9), yticks=[-4.59, 0, 4.59], xlabel="Executed time (s)")
    for axis in axes[:, column]:
        axis.axvspan(4, 5, color="#E6AB02", alpha=0.08)
        axis.grid(alpha=0.2)
        axis.set_xlim(0, 5)
for row, label in enumerate(
    ("Distance from upright (°)", "Arm angle (°)", "Applied torque (mN m)")
):
    axes[row, 0].set_ylabel(label)
figure.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=4)
figure.suptitle(
    "Matched sessions — identical starting state within each column\n"
    "Chosen near each group's median under CEM w=10; w = arm-centering cost rate",
    fontsize=14,
)
for extension in ("png", "pdf"):
    figure.savefig(human / f"matched_sessions.{extension}", dpi=170)
plt.close(figure)

order = np.concatenate([np.flatnonzero(groups == group) for group in range(4)])
figure, axis = plt.subplots(figsize=(15, 5.2), layout="constrained")
mesh = axis.imshow(
    final_rms[:, order],
    aspect="auto",
    norm=LogNorm(vmin=0.1, vmax=180),
    cmap="viridis_r",
    interpolation="nearest",
)
row, column = np.nonzero(hold[:, order])
axis.scatter(
    column,
    row,
    facecolors="none",
    edgecolors="white",
    s=48,
    lw=1.3,
    label="White circle: every final-second sample within 5°",
)
axis.set(
    xticks=np.arange(32),
    xticklabels=order,
    yticks=np.arange(9),
    yticklabels=[c["label"] for c in controllers],
    xlabel="Episode ID",
)
for boundary in (7.5, 15.5, 23.5):
    axis.axvline(boundary, color="white", lw=1.5)
for group, label in enumerate(selection["groups"]):
    axis.text(group * 8 + 3.5, -0.8, label, ha="center", va="bottom", fontsize=10)
bar = figure.colorbar(mesh, ax=axis, ticks=[0.1, 0.5, 1, 5, 20, 60, 180], fraction=0.025)
bar.ax.set_yticklabels(["0.1", "0.5", "1", "5", "20", "60", "180"])
bar.set_label("Final-second upright RMS (degrees; log color scale)")
figure.suptitle(
    "All 32 sessions, all nine controller settings — lower error is better\n"
    "w = centering cost rate; CEM spread = 0.5 except where labeled",
    fontsize=13,
)
figure.legend(*axis.get_legend_handles_labels(), loc="outside lower center")
for extension in ("png", "pdf"):
    figure.savefig(human / f"all_sessions.{extension}", dpi=170)
plt.close(figure)
print(
    f"Verified 288 trajectories; best episodes {best.tolist()}; matched episodes {matched}",
    flush=True,
)
print(f"Saved three PNG/PDF figures to {human}", flush=True)
