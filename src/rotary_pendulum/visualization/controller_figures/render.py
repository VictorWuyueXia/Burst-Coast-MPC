"""Reproduce the three controller figures from the preserved validation records."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from rotary_pendulum.visualization.controller_figures.data import (
    TARGET,
    prepare_example,
    select_examples,
)
from rotary_pendulum.visualization.controller_figures.energy_figures import (
    draw_energy_plane,
    draw_torque_response,
)
from rotary_pendulum.visualization.controller_figures.mechanism import draw_configuration


def render() -> None:
    """Prepare verified numerical records and export three consistently styled figures."""
    root = Path(__file__).resolve().parents[4]
    records = root / ("artifacts/rotary_pendulum/to-show/"
                      "single-constant-torque-action-controller/portable-animation-records/records")
    output = root / "docs/15-controller-paper/figures"
    machine, human = output / "machine-scannables", output / "human-readables"
    machine.mkdir(parents=True, exist_ok=True)
    human.mkdir(parents=True, exist_ok=True)

    # One font, one base size, and three quiet colors apply to every panel and export.
    font_manager.findfont("DejaVu Sans", fallback_to_default=False)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9, "mathtext.fontset": "dejavusans",
        "axes.labelsize": 9, "axes.titlesize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "figure.titlesize": 9, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.7, "lines.linewidth": 1.2,
        "text.color": "#35434B", "axes.labelcolor": "#35434B", "xtick.color": "#35434B",
        "ytick.color": "#35434B", "axes.edgecolor": "#9DADA8", "legend.frameon": False,
        "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.facecolor": "white",
    })
    colors = {"ink": "#35434B", "blue": "#4D7C91", "sage": "#9DADA8", "accent": "#C27B45"}
    examples = select_examples(records, machine)
    data = prepare_example(examples[0], machine)
    plane_data = {key: data[key] for key in ("case", "seed", "lane", "decision", "time_s",
                  "sha256", "state", "request", "budget", "boundary", "region", "plane")}
    (machine / "plane_example.json").write_text(json.dumps({
        key: value.tolist() if hasattr(value, "tolist") else value
        for key, value in plane_data.items()
    }, indent=2)+"\n")
    print(f"Energy-plane case: {plane_data['case']} at {plane_data['time_s']:.1f} s.", flush=True)
    figures = {"01-rotary-pendulum": draw_configuration(colors),
               "02-energy-plane": draw_energy_plane(plane_data, colors),
               "03-torque-solutions": draw_torque_response(data, colors)}

    # Preserve fixed physical dimensions across vector exports and raster previews.
    exports = {}
    for name, figure in figures.items():
        figure.savefig(human / f"{name}.pdf")
        figure.savefig(human / f"{name}.svg")
        figure.savefig(human / f"{name}.png", dpi=300)
        exports[name] = {"width_in": float(figure.get_figwidth()),
                         "height_in": float(figure.get_figheight())}
        plt.close(figure)
        print(f"Exported {name}: PDF, SVG, PNG.", flush=True)
    (machine / "export_audit.json").write_text(json.dumps({
        "font": "DejaVu Sans", "base_font_pt": 9, "colors": colors, "exports": exports,
        "line_semantics": "Solid physical curves and boundaries; no decorative dotted lines.",
        "background_semantics": "Horizontal arm plane; vertical pendulum plane; "
                                "pale coordinate walls and solid grids in energy space.",
        "mechanism_semantics": "Charcoal arm; blue pendulum; faded alternate poses; "
                               "right-angle mark identifies the arm-normal swing plane.",
        "energy_views": {"axis_bound": 1.05, "current_state_shown": False,
                         "request_over_target": plane_data["request"]/TARGET,
                         "plane_total_over_target": float(plane_data["budget"]),
                         "selection": "Shared Figure 3 record; broad plane with visible curved "
                                      "compatibility boundary and separated upright target."},
        "view_degrees": {"arm": {"elevation": 38, "azimuth": -50},
                         "pendulum": {"elevation": 26, "azimuth": 55},
                         "energy_plane": {"elevation": 28, "azimuth": 35},
                         "torque_curve": {"elevation": 28, "azimuth": -145}},
        "marker_semantics": "Open circle: work match; "
                            "open diamond: upright energy target; star: selected torque.",
        "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in Path(__file__).parent.glob("*.py")},
    }, indent=2)+"\n")


if __name__ == "__main__":
    render()
