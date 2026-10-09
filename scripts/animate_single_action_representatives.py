"""Render the committed single-action controller replays as 50 fps MP4 videos."""

# The frame updater runs to completion inside each session loop iteration.
# ruff: noqa: B023

import hashlib
import json
from pathlib import Path

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    bundle = root / (
        "artifacts/rotary_pendulum/experiment-results/"
        "single-constant-torque-action-controller/portable-animation-records"
    )
    records = bundle / "records"
    videos = bundle / "human-readables"
    videos.mkdir(exist_ok=True)
    manifest = json.loads((records / "manifest.json").read_text())
    sample_interval = manifest["physics_sample_interval_s"]
    assert sample_interval == 0.02 and len(manifest["sessions"]) == 8
    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

    # Project recorded 3D points onto a fixed orthographic view with equal spatial scale.
    azimuth, elevation = np.deg2rad((38.0, 24.0))
    view = np.array(
        [
            [-np.sin(azimuth), np.cos(azimuth), 0.0],
            [
                -np.sin(elevation) * np.cos(azimuth),
                -np.sin(elevation) * np.sin(azimuth),
                np.cos(elevation),
            ],
        ]
    )
    labels = {
        "downward_median_capture_time": "Typical downward swing-up",
        "downward_slowest_capture": "Slowest downward capture",
        "moving_median_capture_time_without_arm_crossing": "Typical moving start",
        "near_upright_median_capture_time": "Near-upright start",
        "tight_upright_median_capture_time": "Tight-upright start",
        "exact_downward_rest": "Exact downward rest",
        "arm_crossing_despite_bounded_tested_torques": "Arm crossing: bounded options existed",
        "arm_crossing_without_bounded_tested_torque": "Arm crossing: no bounded tested option",
    }

    for session in manifest["sessions"]:
        name = session["name"]
        path = records / f"{name}.csv"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sha256"][path.name]
        data = np.genfromtxt(path, delimiter=",", names=True)
        time = data["time_s"]
        arm = np.rad2deg(data["arm_angle_rad"])
        pendulum = np.rad2deg(data["pendulum_angle_rad"])
        torque = data["preceding_interval_torque_nm"]
        crossed = np.maximum.accumulate(np.abs(arm) >= 180.0)
        wraps = np.flatnonzero(np.abs(np.diff(pendulum)) > 180.0) + 1
        pendulum_time = np.insert(time, wraps, np.nan)
        pendulum_values = np.insert(pendulum, wraps, np.nan)
        assert len(time) == session["samples_including_initial"]
        np.testing.assert_allclose(time, np.arange(len(time)) * sample_interval, atol=1e-12)
        np.testing.assert_allclose(time[-1], session["duration_s"], atol=1e-12)
        assert crossed[-1] == (session["episode"]["arm_violation"] == "True")
        assert np.isfinite(np.column_stack((time, arm, pendulum, torque))).all()

        # Keep the physical geometry in the replay rather than reconstructing the state.
        points = np.stack(
            [
                np.column_stack([data[f"{point}_{axis}_m"] for axis in "xyz"])
                for point in ("origin", "pivot", "tip")
            ],
            axis=1,
        )
        projected = points @ view.T
        arm_length = np.linalg.norm(points[0, 1] - points[0, 0])
        pendulum_length = np.linalg.norm(points[0, 2] - points[0, 1])
        reach = 1.15 * (arm_length + pendulum_length)
        circle = np.linspace(0.0, 2.0 * np.pi, 200)
        sweep = (
            np.column_stack(
                (arm_length * np.cos(circle), arm_length * np.sin(circle), np.zeros_like(circle))
            )
            @ view.T
        )
        assert np.allclose(np.linalg.norm(points[:, 1] - points[:, 0], axis=1), arm_length)
        assert np.allclose(np.linalg.norm(points[:, 2] - points[:, 1], axis=1), pendulum_length)

        figure = plt.figure(figsize=(10.8, 6.0), facecolor="#f7f9fc")
        layout = figure.add_gridspec(3, 2, width_ratios=(1.4, 1.0), hspace=0.45)
        mechanism = figure.add_subplot(layout[:, 0])
        pendulum_axis = figure.add_subplot(layout[0, 1])
        arm_axis = figure.add_subplot(layout[1, 1], sharex=pendulum_axis)
        torque_axis = figure.add_subplot(layout[2, 1], sharex=pendulum_axis)
        figure.suptitle(labels[name], fontsize=16, fontweight="bold", y=0.97)
        figure.text(
            0.5,
            0.012,
            f"Validation seed {session['seed']} · source lane {session['source_lane']} · "
            "recorded simulation; stops at first capture",
            ha="center",
            fontsize=9,
            color="#475569",
        )

        # Draw the arm sweep and the recorded mechanism in a stable camera view.
        mechanism.plot(sweep[:, 0], sweep[:, 1], ":", color="#94a3b8", lw=1.5)
        mechanism.plot(0, 0, "o", color="#111827", ms=7)
        (arm_line,) = mechanism.plot([], [], "o-", color="#dc2626", lw=5, ms=6, label="arm")
        (pendulum_line,) = mechanism.plot(
            [], [], "o-", color="#2563eb", lw=4, ms=6, label="pendulum"
        )
        (upright_line,) = mechanism.plot([], [], "--", color="#16a34a", lw=1.5)
        mechanism.set(
            xlim=(-reach, reach),
            ylim=(-reach, reach),
            xlabel="projected horizontal (m)",
            ylabel="projected vertical (m)",
        )
        mechanism.set_aspect("equal")
        mechanism.grid(alpha=0.2)
        mechanism.legend(loc="upper left", frameon=False)
        status = mechanism.text(
            0.02,
            0.03,
            "",
            transform=mechanism.transAxes,
            fontsize=10,
            va="bottom",
            color="#111827",
            bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.94},
        )

        # Show the goal band, arm soft limits, and torque used over each preceding interval.
        pendulum_axis.axhspan(165, 195, color="#bbf7d0", alpha=0.8, label="goal: 165–195°")
        arm_axis.axhline(180, color="#ef4444", ls="--", lw=1)
        arm_axis.axhline(-180, color="#ef4444", ls="--", lw=1)
        (pendulum_trace,) = pendulum_axis.plot([], [], color="#2563eb", lw=1.8)
        (arm_trace,) = arm_axis.plot([], [], color="#dc2626", lw=1.8)
        (torque_trace,) = torque_axis.plot([], [], color="#7c3aed", lw=1.8, drawstyle="steps-pre")
        pendulum_axis.set(ylabel="pendulum (°)", ylim=(0, 360))
        arm_axis.set(ylabel="arm (°)", ylim=(-205, 205))
        torque_axis.set(
            ylabel="last torque (N m)", xlabel="recorded time (s)", ylim=(-0.021, 0.021)
        )
        torque_axis.set_xlim(0, max(time[-1], sample_interval))
        for axis in (pendulum_axis, arm_axis, torque_axis):
            axis.grid(alpha=0.2)
        pendulum_axis.legend(loc="upper right", fontsize=8, frameon=False)

        def update(frame: int) -> None:
            # Advance all artists to the same recorded sample without interpolation.
            origin, pivot, tip = projected[frame]
            arm_line.set_data([origin[0], pivot[0]], [origin[1], pivot[1]])
            pendulum_line.set_data([pivot[0], tip[0]], [pivot[1], tip[1]])
            upright = (points[frame, 1] + [0.0, 0.0, pendulum_length]) @ view.T
            upright_line.set_data([pivot[0], upright[0]], [pivot[1], upright[1]])
            trace_end = frame + 1 + np.searchsorted(wraps, frame, side="right")
            pendulum_trace.set_data(pendulum_time[:trace_end], pendulum_values[:trace_end])
            arm_trace.set_data(time[: frame + 1], arm[: frame + 1])
            torque_trace.set_data(time[: frame + 1], torque[: frame + 1])
            limit = "ARM LIMIT CROSSED" if crossed[frame] else "arm within ±180°"
            ending = "  |  CAPTURE: RECORDING ENDED" if frame == len(time) - 1 else ""
            applied = "none" if frame == 0 else f"{torque[frame]:+.5f} N m"
            status.set_text(
                f"t = {time[frame]:.2f} s   |   pendulum = {pendulum[frame]:.1f}°\n"
                f"arm = {arm[frame]:+.1f}°   |   preceding torque = {applied}\n"
                f"{limit}{ending}"
            )
            status.set_color("#b91c1c" if crossed[frame] else "#111827")

        output = videos / f"{name}.mp4"
        print(f"Rendering {name}: {len(time)} frames at 50 fps", flush=True)
        writer = FFMpegWriter(
            fps=50, codec="libx264", bitrate=2000, extra_args=["-pix_fmt", "yuv420p"]
        )
        writer.setup(figure, str(output), dpi=100)
        for frame in range(len(time)):
            update(frame)
            writer.grab_frame()
            if (frame + 1) % 200 == 0 or frame + 1 == len(time):
                print(f"  {frame + 1}/{len(time)} frames", flush=True)
        writer.finish()
        plt.close(figure)
        print(f"Saved {output.relative_to(root)}", flush=True)


if __name__ == "__main__":
    main()
