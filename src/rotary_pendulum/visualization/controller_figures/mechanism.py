"""Two projected spatial views of the rotary pendulum's signed rotations."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon

from rotary_pendulum.environment.dynamics import mechanism_points
from rotary_pendulum.visualization.controller_figures.data import PHYSICAL


def draw_configuration(colors: dict) -> plt.Figure:
    """Separate horizontal arm rotation from the pendulum's vertical-plane rotation."""
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.35))
    figure.subplots_adjust(left=0.025, right=0.98, bottom=0.18, top=0.90, wspace=0.16)
    vertical = np.array([0, 0, 1])
    for panel, ax in enumerate(axes):
        azimuth, elevation = np.deg2rad([(-50, 38), (55, 26)][panel])
        camera = np.array([[-np.sin(azimuth), np.cos(azimuth), 0],
                           [-np.sin(elevation)*np.cos(azimuth),
                            -np.sin(elevation)*np.sin(azimuth), np.cos(elevation)]])
        ax.set_aspect("equal")
        ax.set_axis_off()
        if panel == 0:
            # Two arm poses share the same fixed horizontal reference and motor axis.
            plane = np.array([[-0.35, -0.25, 0], [1.2, -0.25, 0],
                              [1.2, 1.2, 0], [-0.35, 1.2, 0]])
            ax.add_patch(Polygon(plane @ camera.T, facecolor=colors["sage"],
                                 edgecolor=colors["sage"], alpha=0.18, linewidth=0.7))
            reference = np.array([[0, 0, 0], [1.17, 0, 0]]) @ camera.T
            ax.plot(*reference.T, color=colors["sage"], linewidth=0.8)
            shaft = np.array([[0, 0, -0.30], [0, 0, 0.55]]) @ camera.T
            ax.plot(*shaft.T, color=colors["ink"], linewidth=1.1)
            for index, (angle, radius) in enumerate(zip(np.deg2rad([25, 100]),
                                                       [0.58, 0.85], strict=True)):
                points = (mechanism_points(np.array([angle, 0, 0, 0]), PHYSICAL)
                          / PHYSICAL.arm_length_m)
                arm = points[:2] @ camera.T
                color = colors["ink"]
                ax.plot(*arm.T, color=color, alpha=(0.4, 1)[index], linewidth=3.5,
                        solid_capstyle="round")
                ax.scatter(*arm[-1], s=15, color=color, zorder=4)
                angles = np.linspace(0, angle, 70)
                arc = (radius*np.column_stack((np.cos(angles), np.sin(angles), np.zeros(70)))
                       @ camera.T)
                ax.plot(*arc.T, color=color, linewidth=0.9)
                ax.annotate("", arc[-1], arc[-5], arrowprops={"arrowstyle": "->", "color": color,
                            "linewidth": 0.9, "shrinkA": 0, "shrinkB": 0})
                label = arc[len(arc)//2]+np.array([(0.09, -0.07), (0.10, 0.06)][index])
                ax.text(*label, f"$\\theta_{index+1}$", ha="left", va="center")
            # Positive torque and arm velocity circulate about the vertical motor axis.
            angles = np.linspace(-1.0, 2.9, 80)
            arc = (0.21*np.column_stack((np.cos(angles), np.sin(angles), np.zeros(80)))
                   + 0.43*vertical) @ camera.T
            ax.plot(*arc.T, color=colors["accent"], linewidth=1.1)
            ax.annotate("", arc[-1], arc[-5], arrowprops={"arrowstyle": "->",
                        "color": colors["accent"], "linewidth": 1.1, "shrinkB": 0})
            ax.text(*(arc[-1]+[-0.07, 0.06]), r"$u,\ \omega$", ha="right", va="bottom")
            ax.set(xlim=(-0.65, 1.90), ylim=(-0.75, 1.15))
        else:
            # The arm-attached swing plane translates and turns with the arm angle.
            tangent = np.array([0, 1, 0])
            plane = np.array([[0, -1.05, 0.20], [0, 1.28, 0.20],
                              [0, 1.28, -1.32], [0, -1.05, -1.32]])
            ax.add_patch(Polygon(plane @ camera.T, facecolor=colors["sage"],
                                 edgecolor=colors["sage"], alpha=0.18, linewidth=0.7))
            arm = np.array([[-1, 0, 0], [0, 0, 0]]) @ camera.T
            ax.plot(*arm.T, color=colors["ink"], linewidth=3.5, solid_capstyle="round")
            shaft = np.array([[-1, 0, -0.18], [-1, 0, 0.25]]) @ camera.T
            ax.plot(*shaft.T, color=colors["ink"], linewidth=1.1)
            corner = np.array([[-0.18, 0, 0], [-0.18, 0.18, 0], [0, 0.18, 0]]) @ camera.T
            ax.plot(*corner.T, color=colors["ink"], linewidth=0.8)
            reference = np.array([[0, 0, 0.13], [0, 0, -1.20]]) @ camera.T
            ax.plot(*reference.T, color=colors["sage"], linewidth=0.8)
            for index, angle in enumerate(np.deg2rad([-40, 55])):
                points = (mechanism_points(np.array([0, angle, 0, 0]), PHYSICAL)
                          / PHYSICAL.arm_length_m)
                rod = (points[[1, 3]]-points[1]) @ camera.T
                color = colors["blue"]
                ax.plot(*rod.T, color=color, alpha=(0.4, 1)[index], linewidth=3.0,
                        solid_capstyle="round")
                angles = np.linspace(0, angle, 70)
                arc = (0.52*(np.sin(angles)[:, None]*tangent-np.cos(angles)[:, None]*vertical)
                       @ camera.T)
                ax.plot(*arc.T, color=color, linewidth=0.9)
                ax.annotate("", arc[-1], arc[-5], arrowprops={"arrowstyle": "->", "color": color,
                            "linewidth": 0.9, "shrinkA": 0, "shrinkB": 0})
                ax.text(*(arc[35]+[(0.025, 0.07)[index], -0.13]),
                        f"$\\alpha_{index+1}$", ha="center")
            # A separate positive-velocity arrow and gravity vector define the directions.
            angles = np.linspace(0.85, 1.75, 50)
            arc = (0.32*(np.sin(angles)[:, None]*tangent-np.cos(angles)[:, None]*vertical)
                   @ camera.T)
            ax.plot(*arc.T, color=colors["blue"], linewidth=1.0)
            ax.annotate("", arc[-1], arc[-5],
                        arrowprops={"arrowstyle": "->", "color": colors["blue"]})
            ax.text(*(arc[-1]+[0.13, -0.08]), r"$\nu$", ha="left")
            gravity = np.array([[0, 1.48, -0.20], [0, 1.48, -0.75]]) @ camera.T
            ax.annotate("", gravity[1], gravity[0],
                        arrowprops={"arrowstyle": "->", "color": colors["ink"]})
            ax.text(*(gravity.mean(0)+[0.08, 0]), r"$g$", va="center")
            ax.set(xlim=(-1.08, 1.25), ylim=(-1.72, 0.76))
    figure.text(0.035, 0.96, "a  Arm rotation", va="top")
    figure.text(0.535, 0.96, "b  Pendulum rotation", va="top")
    figure.legend(handles=[Line2D([], [], color=colors["ink"], lw=3, label="Arm"),
                  Line2D([], [], color=colors["blue"], lw=3, label="Pendulum"),
                  Patch(facecolor=colors["sage"], alpha=0.18,
                        label="Swing plane in b (rotates with arm)")],
                  loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=3,
                  columnspacing=1.5, handlelength=1.8)
    return figure
