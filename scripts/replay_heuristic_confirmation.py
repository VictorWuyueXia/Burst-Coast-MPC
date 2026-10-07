"""Animate the four committed historical sessions without simulation or GPU dependencies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def main() -> None:
    directory = (
        Path(__file__).resolve().parents[1]
        / "docs/12-energy-transfer"
        / "machine-scannables/heuristic_confirmation_replay"
    )
    manifest = json.loads((directory / "manifest.json").read_text())
    figure = plt.figure(figsize=(12, 9), layout="constrained")
    sessions, artists, axes = [], [], []
    extent = 1.12 * sum(manifest["geometry_m"].values())
    for index, session in enumerate(manifest["sessions"]):
        path = directory / f"{session['stratum']}.csv"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sha256"][path.name]
        data = np.genfromtxt(path, delimiter=",", names=True)
        sessions.append(data)
        axis = figure.add_subplot(2, 2, index + 1, projection="3d")
        axis.set(
            xlim=(-extent, extent),
            ylim=(-extent, extent),
            zlim=(-extent, extent),
            xlabel="x (m)",
            ylabel="y (m)",
            zlabel="z (m)",
        )
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=24, azim=38)
        (arm,) = axis.plot([], [], [], "o-", lw=4, color="tab:red", label="arm")
        (pendulum,) = axis.plot([], [], [], "o-", lw=3, color="tab:blue", label="pendulum")
        axis.legend(loc="upper left")
        axes.append(axis)
        artists.append((arm, pendulum))

    def update(frame: int) -> None:
        for session, data, axis, (arm, pendulum) in zip(
            manifest["sessions"], sessions, axes, artists, strict=True
        ):
            row = data[min(frame, len(data) - 1)]
            arm.set_data_3d(*[[row[f"origin_{a}_m"], row[f"pivot_{a}_m"]] for a in "xyz"])
            pendulum.set_data_3d(*[[row[f"pivot_{a}_m"], row[f"tip_{a}_m"]] for a in "xyz"])
            ended = " (ended)" if frame >= len(data) - 1 else ""
            axis.set_title(
                f"{session['stratum']} · lane {session['source_lane']}{ended}\n"
                f"t={row['time_s']:.2f} s, arm={np.degrees(row['theta_rad']):+.1f}°"
            )

    update(0)
    animation = FuncAnimation(
        figure,
        update,
        frames=max(map(len, sessions)),
        interval=1000 * manifest["physics_dt_s"],
        repeat=False,
        blit=False,
        cache_frame_data=False,
    )
    plt.show()
    # Retain the timer throughout the blocking GUI event loop.
    del animation


if __name__ == "__main__":
    main()
