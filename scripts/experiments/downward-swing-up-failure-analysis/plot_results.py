"""Plot Monte Carlo outcomes, observed failure mechanisms and representative trajectories."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    records, output = arguments.records, arguments.output
    output.mkdir(parents=True, exist_ok=True)
    handle = (records / "summary.csv").open()
    summary = list(csv.DictReader(handle))
    handle.close()
    handle = (records / "episodes.csv").open()
    episodes = list(csv.DictReader(handle))
    handle.close()
    baseline = [row for row in summary if row["case"] == "baseline" and row["stratum"] != "probe"]
    names = [
        "Downward",
        "Almost at rest",
        "Wide arm / down",
        "Moving",
        "Near upright",
        "Tight upright",
    ]
    colors = {
        "captured": "#27866c",
        "low_lift": "#cf543f",
        "partial_lift": "#e5a64e",
        "brief_visit": "#775c9e",
    }
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    rates = np.array([float(row["capture_pct"]) for row in baseline])
    error = np.array(
        [
            [float(row["wilson95_low_pct"]) for row in baseline],
            [float(row["wilson95_high_pct"]) for row in baseline],
        ]
    )
    axes[0].barh(names, rates, color="#376b91", xerr=np.abs(error - rates), capsize=3)
    for index, row in enumerate(baseline):
        axes[0].text(
            error[1, index] + 2,
            index,
            f"{row['captures']}/{row['episodes']}",
            va="center",
            fontsize=9,
        )
    axes[0].set_xlim(0, 115)
    axes[0].set_xlabel("Captured episodes [%]; whiskers: 95% Wilson intervals")
    axes[0].set_title("Default controller: 8,192 random starts")
    axes[0].invert_yaxis()
    bottom = np.zeros(3)
    for key, label in (
        ("captures", "Captured"),
        ("low_lift", "Never reached horizontal"),
        ("partial_lift", "Above horizontal, never goal band"),
        ("brief_visit", "Goal band reached, hold too short"),
    ):
        values = np.array([100 * int(row[key]) / int(row["episodes"]) for row in baseline[:3]])
        axes[1].bar(
            names[:3],
            values,
            bottom=bottom,
            label=label,
            color=colors["captured" if key == "captures" else key],
        )
        bottom += values
    axes[1].set_ylabel("Episodes [%]")
    axes[1].set_title("Downward outcomes at the 20 s deadline")
    axes[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.10))
    figure.savefig(output / "outcomes.png", dpi=160)
    plt.close(figure)

    failures = [
        row
        for row in episodes
        if row["case"] == "baseline"
        and row["stratum"] == "downward"
        and row["failure"] == "low_lift"
    ]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    keys = [
        "exact_fraction",
        "override_fraction",
        "unresolved_fraction",
        "speed_reject_fraction",
        "no_bounded_fraction",
        "no_root_fraction",
    ]
    labels = [
        "Exact work",
        "Work overridden",
        "Recovery unresolved",
        "Only speed test blocks recovery",
        "No arm-bounded candidate",
        "No work root",
    ]
    values = [100 * np.median([float(row[key]) for row in failures]) for key in keys]
    axes[0].barh(labels, values, color="#376b91")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Median share of active decisions [%]")
    axes[0].set_title("Why the decoder changes the requested work")
    axes[1].hist(
        [100 * float(row["cancelled_work_fraction"]) for row in failures], bins=25, color="#cf543f"
    )
    axes[1].set_xlabel("Cancelled share of total absolute motor work [%]")
    axes[1].set_ylabel("Episodes")
    axes[1].set_title("Work repeatedly added and removed")
    for key, label in (
        ("tail_total_energy", "Total energy"),
        ("tail_potential", "Potential energy"),
    ):
        axes[2].hist([float(row[key]) for row in failures], bins=25, alpha=0.65, label=label)
    axes[2].axvline(1, color="black", linestyle="--", label="Upright energy")
    axes[2].set_xlabel("Mean energy during 15–20 s / upright energy")
    axes[2].set_title("Energy stalls below the swing-up requirement")
    axes[2].legend(fontsize=8)
    figure.suptitle(f"Ordinary downward low-lift failures: {len(failures):,} episodes")
    figure.savefig(output / "failure_mechanisms.png", dpi=160)
    plt.close(figure)

    selections = json.loads((records / "plot_selection.json").read_text())
    for selection in selections:
        category = selection["category"]
        data = dict(np.load(records / f"representative_{category}.npz"))
        active = data["elapsed_s"] > 0
        samples = data["physics_x"][data["physics_active"]]
        times = 0.02 * np.arange(1, len(samples) + 1)
        decision_time = data["time_s"][active]
        angle_deg = np.rad2deg((samples[:, 1] + np.pi) % (2 * np.pi) - np.pi)
        angle_deg[np.abs(np.diff(angle_deg, prepend=angle_deg[0])) > 180] = np.nan
        figure, axes = plt.subplots(3, 2, figsize=(13, 10), layout="constrained")
        axes[0, 0].plot(times, angle_deg, color="#376b91")
        for edge in (-180, 180):
            axes[0, 0].axhspan(
                max(-180, edge - 15),
                min(180, edge + 15),
                color="#27866c",
                alpha=0.2,
                label="Goal near ±180°",
            )
        axes[0, 0].set_ylabel("Pendulum angle [deg; 0 down, ±180 up]")
        axes[0, 0].set_ylim(-180, 180)
        axes[0, 0].set_yticks([-180, -90, 0, 90, 180])
        energy = data["energy_j"][active] / 0.03037176
        for index, label in enumerate(("Arm kinetic", "Pendulum body kinetic", "Potential")):
            axes[0, 1].plot(decision_time, energy[:, index], label=label)
        axes[0, 1].plot(decision_time, energy.sum(axis=-1), "k--", label="Total")
        axes[0, 1].axhline(1, color="gray", linewidth=0.8)
        axes[0, 1].set_ylabel("Energy / upright energy (0.03037 J)")
        axes[1, 0].plot(times, np.rad2deg(samples[:, 0]), color="#376b91")
        axes[1, 0].axhline(180, color="#cf543f", linestyle="--", label="Soft arm limit")
        axes[1, 0].axhline(-180, color="#cf543f", linestyle="--")
        axes[1, 0].set_ylabel("Arm angle [degrees]")
        axes[1, 1].stairs(
            1000 * data["torque_nm"][active],
            np.r_[0, decision_time],
            color="#376b91",
            label="Applied torque",
        )
        axes[1, 1].set_ylabel("Motor torque [mN m]")
        positive = np.cumsum(np.maximum(data["work_j"][active], 0)) * 1000
        removed = -np.cumsum(np.minimum(data["work_j"][active], 0)) * 1000
        axes[2, 0].plot(decision_time, positive, label="Cumulative added work")
        axes[2, 0].plot(decision_time, removed, label="Cumulative removed work")
        axes[2, 0].plot(decision_time, positive - removed, "k--", label="Net work")
        axes[2, 0].set_ylabel("Motor work [mJ]")
        axes[2, 1].step(decision_time, data["mode"][active], where="pre", color="#775c9e")
        axes[2, 1].set_yticks([0, 1, 2], ["Exact work", "Work override", "Recovery unresolved"])
        axes[2, 1].set_ylabel("Decoder choice")
        for axis in axes.flat:
            axis.set_xlabel("Time [s]")
            axis.grid(alpha=0.2)
        for axis in axes.flat[:-1]:
            axis.legend(fontsize=8)
        figure.suptitle(
            f"{category.replace('_', ' ').title()} | seed {selection['seed']}, "
            f"lane {selection['lane']} | captured: {selection['success']}"
        )
        figure.savefig(output / f"representative_{category}.png", dpi=150)
        plt.close(figure)

    cases = [
        name
        for name in ("baseline", "speed05", "speed05_kinetic05", "speed05_kinetic075")
        if any(row["case"] == name for row in summary)
    ]
    if len(cases) > 1:
        figure, axes = plt.subplots(1, 2, figsize=(15, 6), layout="constrained")
        positions = np.arange(len(cases))
        case_labels = [
            {
                "baseline": "Default",
                "speed05": "Recovery 0.5",
                "speed05_kinetic05": "Recovery 0.5 + weight 0.5",
                "speed05_kinetic075": "Recovery 0.5 + weight 0.75",
            }[case]
            for case in cases
        ]
        for offset, label, title in (
            (-0.25, "downward", "Ordinary downward"),
            (0, "rest", "Almost at rest"),
            (0.25, "wide_down", "Wide arm / down"),
        ):
            selected = [
                next(row for row in summary if row["case"] == case and row["stratum"] == label)
                for case in cases
            ]
            axes[0].barh(
                positions + offset,
                [float(row["capture_pct"]) for row in selected],
                height=0.25,
                label=title,
            )
        axes[0].set_yticks(positions, case_labels)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Captured episodes [%]; sample sizes in report")
        axes[0].set_title("Confirmed candidates: 8,192 random starts each")
        axes[0].legend(fontsize=8)
        excursions = [
            sum(
                int(row["excursions"])
                for row in summary
                if row["case"] == case and row["stratum"] != "probe"
            )
            for case in cases
        ]
        counts = [
            sum(
                int(row["episodes"])
                for row in summary
                if row["case"] == case and row["stratum"] != "probe"
            )
            for case in cases
        ]
        axes[1].barh(case_labels, 100 * np.asarray(excursions) / counts, color="#cf543f")
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Episodes with any arm excursion beyond ±180° [%]")
        axes[1].set_title("Arm containment under each intervention")
        figure.suptitle(
            "Recovery = terminal arm-speed limit [rad/s]; "
            "weight = arm kinetic-energy coefficient in the work request",
            fontsize=11,
        )
        figure.savefig(output / "interventions.png", dpi=160)
        plt.close(figure)
    print(f"Saved {len(selections) + 2 + (len(cases) > 1)} figures in {output}", flush=True)


if __name__ == "__main__":
    main()
