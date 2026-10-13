"""Physical energy compatibility and the held-torque endpoint curve."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.proj3d import proj_transform

from rotary_pendulum.visualization.controller_figures.data import LIMIT


def draw_energy_plane(data: dict, colors: dict) -> plt.Figure:
    """Emphasize the requested plane's physical subset relative to upright rest."""
    figure = plt.figure(figsize=(3.4, 3.65))
    ax = figure.add_axes([0.14, 0.22, 0.76, 0.72], projection="3d", computed_zorder=False)
    ax.set_proj_type("persp", focal_length=1.2)
    ax.view_init(elev=28, azim=35)
    ax.set_box_aspect((1, 1, 1))
    ax.set(xlim=(0, 1.05), ylim=(0, 1.05), zlim=(0, 1.05))

    # Equal energy scales and quiet coordinate walls expose the plane orientation.
    for axis, label in zip((ax.xaxis, ax.yaxis, ax.zaxis),
            (r"$K_a/E_\star$", r"$K_p/E_\star$", r"$V/E_\star$"), strict=True):
        axis.set_pane_color((0.965, 0.974, 0.971, 1))
        axis._axinfo["grid"].update(color=(0.62, 0.68, 0.66, 0.30),
                                   linewidth=0.6, linestyle="-")
        axis.line.set_color(colors["sage"])
        axis.set_rotate_label(False)
        axis.label.set_text(label)
        axis.labelpad = 2
        axis.set_ticks([0, 0.5, 1] if axis is ax.zaxis else [0, 0.5])
        axis.set_major_formatter("{x:.1f}")
    ax.tick_params(pad=0, length=2, colors=colors["ink"])
    ax.grid(True)

    # The curved blue edge separates realizable states from the remaining requested plane.
    ax.add_collection3d(Poly3DCollection([data["plane"]], facecolor=colors["sage"],
                        alpha=0.13, edgecolor=colors["sage"], linewidth=0.8, zorder=1))
    ax.add_collection3d(Poly3DCollection([data["region"]], facecolor=colors["blue"],
                        alpha=0.24, edgecolor="none", zorder=2))
    outline = np.vstack((data["plane"], data["plane"][0]))
    ax.plot(*outline.T, color=colors["sage"], linewidth=0.8, zorder=3)
    ax.plot(*data["boundary"].T, color=colors["blue"], linewidth=1.7, zorder=3)
    edge = np.array([[0, data["budget"], 0], [0, data["budget"]-min(data["budget"], 1),
                                             min(data["budget"], 1)]])
    ax.plot(*edge.T, color=colors["blue"], linewidth=1.0, zorder=3)

    # The upright target is the only state marker, keeping attention on plane structure.
    ax.scatter(0, 0, 1, s=23, marker="D", facecolor="white", edgecolor=colors["ink"],
               linewidth=0.9, depthshade=False, zorder=5)
    ax.annotate(r"$e_\star$", proj_transform(0, 0, 1, ax.get_proj())[:2],
                xytext=(8, 5), textcoords="offset points", zorder=6)
    figure.legend(handles=[Patch(facecolor=colors["sage"], alpha=0.25,
                                  label="Requested energy plane"),
                  Patch(facecolor=colors["blue"], alpha=0.35, label="Realizable energy states")],
                  loc="lower center", bbox_to_anchor=(0.5, 0.015), handlelength=1.5)
    return figure


def draw_torque_response(data: dict, colors: dict) -> plt.Figure:
    """Pair the 3D endpoint geometry with the scalar work-matching equation."""
    figure = plt.figure(figsize=(7.0, 3.3))
    ax = figure.add_axes([0.025, 0.08, 0.525, 0.83], projection="3d", computed_zorder=False)
    ax.set_proj_type("persp", focal_length=1.2)
    ax.view_init(elev=28, azim=-145)
    ax.set_box_aspect((1.1, 1.0, 0.9))
    curve = data["curve"]["energy"]
    low, high = data["low"], data["high"]
    extent = np.ptp(curve, axis=0)
    # The local coordinate room reveals depth without adding trajectory-like guides.
    for axis, label in zip((ax.xaxis, ax.yaxis, ax.zaxis),
            (r"$K_a/E_\star$", r"$K_p/E_\star$", r"$V/E_\star$"), strict=True):
        axis.set_pane_color((0.965, 0.974, 0.971, 1))
        axis._axinfo["grid"].update(color=(0.62, 0.68, 0.66, 0.30), linewidth=0.6, linestyle="-")
        axis.line.set_color(colors["sage"])
        axis.set_rotate_label(False)
        axis.label.set_text(label)
        axis.labelpad = 2
        axis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
        axis.set_major_formatter("{x:.2f}")
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]))
    ax.tick_params(pad=0, length=2)
    ax.grid(True)

    # A single transparent polygon avoids mesh artifacts on the flat request surface.
    ax.add_collection3d(Poly3DCollection([data["local_region"]], facecolor=colors["blue"],
                        alpha=0.24, edgecolor="none", zorder=1))
    ax.plot(*curve.T, color=colors["blue"], linewidth=1.7, zorder=4)
    root_count = len(data["roots"])
    roots = data["tested"]["energy"][:root_count]
    chosen = data["tested"]["energy"][data["chosen"]]
    ax.scatter(*roots.T, s=29, facecolor="white", edgecolor=colors["accent"],
               linewidth=1.3, depthshade=False, zorder=6)
    ax.scatter(*chosen, marker="*", s=70, color=colors["accent"], depthshade=False, zorder=7)
    for index, point in enumerate(roots):
        projected = proj_transform(*point, ax.get_proj())[:2]
        ax.annotate(f"$u_{index+1}$", projected, xytext=((-9, -14), (13, -6))[index],
                    textcoords="offset points", ha="center", zorder=8)
    ax.text(*(curve[0]+[0, 0, 0.06*extent[2]]), r"$-U$", zorder=8)
    ax.text(*(curve[-1]+[0, 0, 0.06*extent[2]]), r"$+U$", zorder=8)
    ax.annotate(r"$u^\star$", proj_transform(*chosen, ax.get_proj())[:2], xytext=(9, 6),
                textcoords="offset points", zorder=8)

    # The zero line defines work matching; circles and the star retain their meanings.
    residual = figure.add_axes([0.66, 0.20, 0.31, 0.63])
    residual.axhline(0, color=colors["sage"], linewidth=0.9)
    residual.plot(data["torques"]*1000, (data["curve"]["work"]-data["request"])*1000,
                  color=colors["blue"], linewidth=1.5)
    residual.scatter(data["roots"]*1000, (data["tested"]["work"][:root_count]-data["request"])*1000,
                     s=29, facecolor="white", edgecolor=colors["accent"], linewidth=1.3, zorder=5)
    residual.scatter(data["selected"]*1000,
                     (data["tested"]["work"][data["chosen"]]-data["request"])*1000,
                     marker="*", s=70, color=colors["accent"], zorder=6)
    residual.annotate(r"$u^\star$", (data["selected"]*1000,
                     (data["tested"]["work"][data["chosen"]]-data["request"])*1000),
                     xytext=(8, 5), textcoords="offset points")
    for index, torque in enumerate(data["roots"]):
        residual.annotate(f"$u_{index+1}$", (torque*1000, 0), xytext=(0, 9),
                          textcoords="offset points", ha="center")
    residual.set(xlabel="Torque (mN m)", ylabel="Work error (mJ)", xlim=(-LIMIT*1000, LIMIT*1000))
    residual.set_xticks([-15, 0, 15])
    figure.text(0.035, 0.97, "a  Endpoint energies (local view)", va="top")
    figure.text(0.66, 0.97, "b  Work matching", va="top")
    return figure
