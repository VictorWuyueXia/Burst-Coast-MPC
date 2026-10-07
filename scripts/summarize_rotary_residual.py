"""Reproduce the heuristic-only cross-wave figures and finite-hold energy audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

from rotary_pendulum.environment.jax_dynamics import MODEL, state_derivative
from rotary_pendulum.RL.jax_residual_control import POLICY_TORQUE_NM, TARGET_ENERGY_J
from rotary_pendulum.RL.jax_residual_evaluation import write_validation

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    human = args.root / "handoff/human-readables"
    machine = args.root / "handoff/machine-scannables"
    human.mkdir(parents=True, exist_ok=True)
    machine.mkdir(parents=True, exist_ok=True)
    summary = []
    for source in sorted(args.root.glob("heuristic-wave*/machine-scannables/summary.csv")):
        handle = source.open()
        summary.extend(dict(row, wave=source.parent.parent.name) for row in csv.DictReader(handle))
        handle.close()
    handle = (machine / "summary.csv").open("w", newline="")
    writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)
    handle.close()
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    comparison = [
        ("heuristic-wave1/zero", "Zero torque"),
        ("heuristic-wave1/t1-b05", "Filtered: τ=1 s, b₀=0.5/s"),
        ("heuristic-wave2/t01-b05", "Filtered: τ=0.1 s, b₀=0.5/s"),
        ("heuristic-wave3/t005-b2", "Filtered: τ=0.05 s, b₀=2/s"),
    ]
    for relative, label in comparison:
        archive = np.load(args.root / relative / "machine-scannables/trajectories.npz")
        lane = (archive["labels"] == "downward") & (archive["deadline_s"] == 60)
        x = archive["physics_x"][:, lane]
        energy = (
            0.5 * MODEL.pendulum_inertia_kg_m2 * x[..., 3] ** 2
            + MODEL.gravity_torque_nm * (1 - np.cos(x[..., 1]))
        ) / TARGET_ENERGY_J
        axes[0, 0].plot(
            archive["time_s"][:, lane][:, 0] + 0.1, energy.mean(axis=(1, 2)), label=label
        )
        archive.close()
    axes[0, 0].axhline(1, color="black", ls=":", label="Upright target")
    axes[0, 0].set(
        xlabel="Time (s)",
        ylabel="Mean E/E* over 64 downward resets",
        title="Energy rises but does not reach swing-up",
    )
    axes[0, 0].legend(fontsize=8)
    strata = ("downward", "moving", "near", "tight")
    for offset, trial, label in (
        (-0.15, "t01-b05", "With arm filter"),
        (0.15, "unfiltered-t01-b05", "Without arm filter"),
    ):
        rates = [
            100
            * float(
                next(
                    row["arm_violation"]
                    for row in summary
                    if row["trial"] == trial
                    and row["deadline_s"] == "60"
                    and row["stratum"] == stratum
                )
            )
            for stratum in strata
        ]
        axes[0, 1].bar(np.arange(4) + offset, rates, width=0.3, label=label)
    axes[0, 1].set(
        xticks=np.arange(4),
        xticklabels=strata,
        ylabel="Episodes with arm excursion (%)",
        title="Filter controls arm drift; identical τ=0.1 s, b₀=0.5/s",
    )
    axes[0, 1].legend(fontsize=8)
    filtered = [
        row
        for row in summary
        if row["trial"].startswith("t")
        and row["deadline_s"] == "60"
        and row["stratum"] == "downward"
    ]
    grid = np.zeros((9, 2))
    times = sorted([0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8])
    for row in filtered:
        config = json.loads(
            (
                args.root / row["wave"] / row["trial"] / "machine-scannables/settings.json"
            ).read_text()
        )
        grid[
            times.index(config["energy_time_s"]), [0.5, 2].index(config["sensitivity_floor_per_s"])
        ] = float(row["energy_error_mean"])
    display = axes[1, 0].imshow(grid, vmin=0, vmax=1, cmap="viridis_r", aspect="auto")
    for row in range(9):
        for column in range(2):
            axes[1, 0].text(column, row, f"{grid[row, column]:.3f}", ha="center", va="center")
    axes[1, 0].set(
        xticks=[0, 1],
        xticklabels=["0.5", "2"],
        yticks=np.arange(9),
        yticklabels=times,
        xlabel="Sensitivity floor b₀ (1/s)",
        ylabel="Energy time τ (s)",
        title="Mean |E/E* − 1|: downward, 60 s; smaller is better",
    )
    figure.colorbar(display, ax=axes[1, 0])
    for index, deadline in enumerate((20, 40, 60)):
        rates = [
            100
            * float(
                next(
                    row["success"]
                    for row in summary
                    if row["trial"] == "t01-b05"
                    and int(row["deadline_s"]) == deadline
                    and row["stratum"] == stratum
                )
            )
            for stratum in strata
        ]
        axes[1, 1].bar(np.arange(4) + (index - 1) * 0.25, rates, width=0.25, label=f"{deadline} s")
    axes[1, 1].set(
        xticks=np.arange(4),
        xticklabels=strata,
        ylabel="Strict hold success (%)",
        title="All settings: 0/64 except 9/64 favorable tight starts",
    )
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].text(
        0.02,
        0.95,
        "All tight successes occur by 0.12 s; zero torque also gets 9/64.",
        transform=axes[1, 1].transAxes,
        fontsize=8,
        va="top",
    )
    figure.tight_layout()
    figure.savefig(human / "campaign_overview.png", dpi=160)
    plt.close(figure)

    # A sampled-control diagnostic: exact instantaneous power versus the actual
    # average power over the held 100 ms interval on a sustained oscillation.
    source = args.root / "heuristic-wave1/t1-b05/machine-scannables"
    archive = np.load(source / "trajectories.npz")
    selection = json.loads((source / "plot_selection.json").read_text())
    lane = selection["downward"][0]
    x, torque = archive["x"][:, lane], archive["applied_nm"][:, lane]
    energy = 0.5 * MODEL.pendulum_inertia_kg_m2 * x[:, 3] ** 2 + MODEL.gravity_torque_nm * (
        1 - np.cos(x[:, 1])
    )
    final = archive["physics_x"][:, lane, -1]
    final_energy = 0.5 * MODEL.pendulum_inertia_kg_m2 * final[
        :, 3
    ] ** 2 + MODEL.gravity_torque_nm * (1 - np.cos(final[:, 1]))
    instantaneous = x[:, 3] * (
        MODEL.pendulum_inertia_kg_m2 * np.asarray(state_derivative(x, torque))[:, 3]
        + MODEL.gravity_torque_nm * np.sin(x[:, 1])
    )
    actual = (final_energy - energy) / archive["elapsed_s"][:, lane]
    time = archive["time_s"][:, lane]
    interval = (time >= 10) & (time < 20)
    diagnostics = {
        "source": str(source),
        "lane": lane,
        "window_s": [10, 20],
        "instantaneous_power_mean_w": float(instantaneous[interval].mean()),
        "held_interval_power_mean_w": float(actual[interval].mean()),
        "positive_instantaneous_negative_interval_fraction": float(
            np.mean((instantaneous[interval] > 0) & (actual[interval] < 0))
        ),
        "saturated_action_fraction": float(
            np.mean(np.abs(torque[interval]) >= 0.999 * POLICY_TORQUE_NM)
        ),
    }
    (machine / "energy_transfer_diagnostic.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n"
    )
    np.savez_compressed(
        machine / "energy_transfer_diagnostic.npz",
        time_s=time,
        energy_j=energy,
        instantaneous_w=instantaneous,
        held_interval_w=actual,
        torque_nm=torque,
        state=x,
    )
    figure, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(time, x[:, 1], label="Pendulum α (from downward)")
    axes[0].set_ylabel("Angle (rad)")
    axes[1].plot(time, energy / TARGET_ENERGY_J, label="E/E*")
    axes[1].axhline(1, color="black", ls=":")
    axes[1].set_ylabel("Normalized energy")
    axes[2].step(time, 1000 * torque, where="post", label="Applied torque")
    axes[2].set_ylabel("Torque (mN m)")
    axes[3].plot(time, instantaneous, "o-", label="Instantaneous power when action is selected")
    axes[3].plot(time, actual, "o-", label="Actual mean power over the following 100 ms")
    axes[3].axhline(0, color="black", ls=":")
    axes[3].set(xlabel="Time (s)", ylabel="Pendulum power (W)", xlim=(10, 12))
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle(f"Persistent oscillation: downward lane {lane}, τ=1 s, b₀=0.5/s")
    figure.tight_layout()
    figure.savefig(human / "energy_transfer_detail.png", dpi=160)
    plt.close(figure)
    archive.close()

    # Re-render selected complete trajectories with a first-five-second zoom.
    for relative, label in (
        ("heuristic-wave2/t01-b05", "reference"),
        ("heuristic-wave3/t005-b2", "lowest-energy-error"),
    ):
        print(f"Rendering handoff plots: {relative}", flush=True)
        source = args.root / relative / "machine-scannables"
        archive = np.load(source / "trajectories.npz")
        config = json.loads((source / "settings.json").read_text())
        rows = write_validation(
            args.root / "handoff" / label,
            {key: archive[key] for key in archive.files if key != "labels"},
            archive["labels"].tolist(),
            config,
        )
        archive.close()
        print(f"Saved {len(rows)} episodes and plots; no training", flush=True)
    print(json.dumps(diagnostics, indent=2), flush=True)


if __name__ == "__main__":
    main()
