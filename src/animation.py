"""Render GIF animations of Babel Machine runs.

The animations are built from the existing :class:`src.babel_model.BabelModel`;
no model behavior is duplicated here. Everything shown is an abstract thought
experiment: ``llm_system_id`` labels are not real vendors or models, no external
API is called, and nothing here is an empirical claim.

``tower_progress`` is monotonic (it never decreases), so "collapse" is *not*
drawn as destruction of the tower. The stall scenario instead shows a loss of
coordinated construction capacity: few newly constructed levels, a low
construction rate, and a high share of failed interactions.

Run ``python -m src.animation --help`` for the command-line interface.
"""

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image

from .babel_model import BabelModel

# Tower progress units per stylized level per agent (display only).
LEVEL_UNIT_PER_AGENT = 5.0
MAX_FRAMES = 500
# Okabe-Ito colorblind-safe palette (language/platform fill colors).
LANGUAGE_COLORS = [
    "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442",
]
# Distinct marker shapes encode the abstract LLM system (redundant with outline).
SYSTEM_MARKERS = ["o", "s", "^", "D", "v", "P"]
SYSTEM_OUTLINES = ["#000000", "#555555", "#000000", "#555555", "#000000", "#555555"]
NOTE = (
    "Abstract simulation: system labels are not real vendors or models; "
    "tower progress never decreases."
)


@dataclass(frozen=True)
class Scenario:
    """Named preset: a title, a description, and BabelModel keyword arguments."""

    key: str
    title: str
    filename: str
    description: str
    model_kwargs: dict = field(default_factory=dict)


SCENARIOS = {
    "baseline": Scenario(
        key="baseline",
        title="Construction (baseline): one abstract system, favorable communication",
        filename="tower_construction_baseline.gif",
        description="One abstract system, one platform, low linguistic distance.",
        model_kwargs={
            "num_platforms": 1,
            "linguistic_distance": 0.1,
            "interoperability": 0.5,
            "platform_drift": 0.05,
            "network_density": 0.2,
        },
    ),
    "stall": Scenario(
        key="stall",
        title="Coordination stall: three abstract systems, low cross-compatibility",
        filename="tower_coordination_stall_low_compatibility.gif",
        description=(
            "Three abstract systems, three platforms, high linguistic distance, "
            "cross-system compatibility 0.05."
        ),
        model_kwargs={
            "num_platforms": 3,
            "linguistic_distance": 0.9,
            "interoperability": 0.0,
            "platform_drift": 0.0,
            "network_density": 0.2,
            "llm_systems": ["system_a", "system_b", "system_c"],
            "llm_cross_compatibility": 0.05,
        },
    ),
}


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def build_model(scenario, num_agents=30, seed=42):
    """Create the :class:`BabelModel` for a scenario preset."""
    if scenario not in SCENARIOS:
        raise ValueError(
            f"unknown scenario {scenario!r}; choose from {sorted(SCENARIOS)}"
        )
    _positive_int(num_agents, "num_agents")
    kwargs = dict(SCENARIOS[scenario].model_kwargs)
    kwargs["num_platforms"] = min(kwargs["num_platforms"], int(num_agents))
    if len(kwargs.get("llm_systems", ())) > num_agents:
        kwargs["llm_systems"] = kwargs["llm_systems"][: int(num_agents)]
    return BabelModel(num_agents=num_agents, num_steps=0, random_seed=seed, **kwargs)


def collect_frames(model, num_steps):
    """Advance ``model`` and return one state snapshot per frame (step 0 first).

    Each snapshot holds the agents' current languages, the step's per-edge
    outcomes, and step-level counts taken directly from the model's history.
    """
    _positive_int(num_steps, "num_steps")
    if num_steps + 1 > MAX_FRAMES:
        raise ValueError(f"num_steps must be at most {MAX_FRAMES - 1}")
    frames = [
        {
            "step": 0,
            "progress": 0.0,
            "gain": 0.0,
            "successes": 0,
            "failures": 0,
            "cooperations": 0,
            "languages": [a.language_id for a in model.agent_list],
            "edges": {},
        }
    ]
    positions = {a.unique_id: i for i, a in enumerate(model.agent_list)}
    previous = 0.0
    for _ in range(num_steps):
        model.step()
        record = model.history[-1]
        edges = {}
        for position, agent in enumerate(model.agent_list):
            for entry in reversed(agent.communication_history):
                if entry["step"] != model.current_step:
                    break
                key = tuple(sorted((position, positions[entry["partner_id"]])))
                edges[key] = (entry["success"], entry["cooperated"])
        frames.append(
            {
                "step": model.current_step,
                "progress": record["tower_progress"],
                "gain": record["tower_progress"] - previous,
                "successes": record["successful_communications"],
                "failures": record["failed_communications"],
                "cooperations": record["cooperation_events"],
                "languages": [a.language_id for a in model.agent_list],
                "edges": edges,
            }
        )
        previous = record["tower_progress"]
    return frames


def _draw_network(axis, model, layout, frame, systems):
    for left, right in model.network.edges:
        left, right = sorted((left, right))
        success, cooperated = frame["edges"].get((left, right), (None, False))
        if success is None:
            style = dict(color="#BBBBBB", linestyle="-", linewidth=0.6)
        elif success:
            style = dict(
                color="#009E73", linestyle="-", linewidth=2.4 if cooperated else 1.4
            )
        else:
            style = dict(color="#D55E00", linestyle=(0, (2, 2)), linewidth=1.4)
        (x1, y1), (x2, y2) = layout[left], layout[right]
        axis.plot([x1, x2], [y1, y2], zorder=1, **style)
    for number, system in enumerate(systems):
        members = [i for i, a in enumerate(model.agent_list) if a.llm_system_id == system]
        axis.scatter(
            [layout[i][0] for i in members],
            [layout[i][1] for i in members],
            c=[LANGUAGE_COLORS[frame["languages"][i] % len(LANGUAGE_COLORS)] for i in members],
            marker=SYSTEM_MARKERS[number % len(SYSTEM_MARKERS)],
            s=170,
            edgecolors=SYSTEM_OUTLINES[number % len(SYSTEM_OUTLINES)],
            linewidths=2.0,
            zorder=2,
        )
    axis.set_axis_off()
    axis.set_title("Agent network", fontsize=10)


def _draw_tower(axis, frame, max_levels, level_unit):
    axis.set_xlim(0, 1)
    axis.set_ylim(0, max_levels + 1)
    axis.set_xticks([])
    axis.set_ylabel("Constructed levels (progress / %g)" % level_unit, fontsize=8)
    axis.set_yticks(range(0, max_levels + 1, max(1, max_levels // 5)))
    levels = frame["progress"] / level_unit
    full = int(math.floor(levels))
    for level in range(full):
        width = 0.7 - 0.25 * level / max(1, max_levels)
        axis.add_patch(
            Rectangle((0.5 - width / 2, level), width, 1, facecolor="#C9A66B",
                      edgecolor="#4B3A1E", linewidth=1.0)
        )
    partial = levels - full
    if partial > 0:
        width = 0.7 - 0.25 * full / max(1, max_levels)
        axis.add_patch(
            Rectangle((0.5 - width / 2, full), width, partial, facecolor="#E6D3AA",
                      edgecolor="#4B3A1E", linewidth=1.0, hatch="//")
        )
    axis.set_title("Tower (cumulative)", fontsize=10)


def render_frame(model, layout, frames, index, scenario, max_levels, max_gain, max_edges):
    """Render frame ``index`` and return it as a Pillow RGB image."""
    level_unit = LEVEL_UNIT_PER_AGENT * model.num_agents
    frame = frames[index]
    figure = Figure(figsize=(12, 6.4), dpi=72)
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(2, 3, width_ratios=[2.2, 1, 1.5], top=0.82, bottom=0.2,
                               left=0.04, right=0.98, hspace=0.55, wspace=0.25)
    systems = list(model.llm_systems)
    _draw_network(figure.add_subplot(grid[:, 0]), model, layout, frame, systems)
    _draw_tower(figure.add_subplot(grid[:, 1]), frame, max_levels, level_unit)

    steps = [f["step"] for f in frames[1: index + 1]]
    rate = figure.add_subplot(grid[0, 2])
    rate.plot(steps, [f["gain"] for f in frames[1: index + 1]], color="#0072B2", marker="o", markersize=3)
    rate.set(xlim=(0, len(frames) - 1), ylim=(0, max_gain * 1.1 or 1))
    rate.set_title("Construction rate (progress gained per step)", fontsize=9)
    activity = figure.add_subplot(grid[1, 2])
    activity.bar(steps, [f["successes"] for f in frames[1: index + 1]], color="#009E73", label="successful")
    activity.bar(steps, [-f["failures"] for f in frames[1: index + 1]], color="#D55E00", hatch="//", label="failed")
    activity.set(xlim=(0, len(frames) - 1), ylim=(-max_edges, max_edges))
    activity.axhline(0, color="black", linewidth=0.6)
    activity.set_xlabel("Step", fontsize=8)
    activity.set_title("Interactions per step (+ successful / - failed)", fontsize=9)
    activity.legend(loc="upper left", fontsize=7)

    levels = int(frame["progress"] // level_unit)
    previous = int(frames[index - 1]["progress"] // level_unit) if index else 0
    total = frame["successes"] + frame["failures"]
    fail_share = frame["failures"] / total if total else 0.0
    figure.suptitle(SCENARIOS[scenario].title, fontsize=12, y=0.97)
    figure.text(
        0.5, 0.88,
        f"Step {frame['step']}/{len(frames) - 1} | tower progress {frame['progress']:.1f} "
        f"(levels completed: {levels}, newly completed: +{levels - previous}) | successful {frame['successes']}, "
        f"failed {frame['failures']} ({fail_share:.0%}), cooperative {frame['cooperations']}",
        ha="center", fontsize=9,
    )
    handles = [
        Line2D([], [], marker=SYSTEM_MARKERS[n % len(SYSTEM_MARKERS)], linestyle="", markerfacecolor="white",
               markeredgecolor=SYSTEM_OUTLINES[n % len(SYSTEM_OUTLINES)], markeredgewidth=2, label=f"{name} (shape)")
        for n, name in enumerate(systems)
    ]
    languages = sorted({language for f in frames for language in f["languages"]})
    handles += [
        Line2D([], [], marker="o", linestyle="", color=LANGUAGE_COLORS[l % len(LANGUAGE_COLORS)], label=f"language {l} (fill)")
        for l in languages
    ]
    handles += [
        Line2D([], [], color="#009E73", linewidth=2.4, label="success + cooperation"),
        Line2D([], [], color="#009E73", linewidth=1.4, label="success"),
        Line2D([], [], color="#D55E00", linestyle=(0, (2, 2)), label="failed (dashed)"),
    ]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.03), ncol=min(6, len(handles)), fontsize=7, frameon=False)
    figure.text(0.5, 0.008, NOTE, ha="center", fontsize=7, style="italic")
    figure.canvas.draw()
    return Image.fromarray(np.asarray(figure.canvas.buffer_rgba())[:, :, :3])


def render_animation(scenario, output_path, num_agents=30, num_steps=60, seed=42, frame_interval=150):
    """Run a preset and write its GIF to ``output_path``; return the path and frames."""
    _positive_int(frame_interval, "frame_interval")
    model = build_model(scenario, num_agents, seed)
    frames = collect_frames(model, num_steps)
    layout = nx.spring_layout(model.network, seed=seed)
    max_levels = max(1, int(math.ceil(frames[-1]["progress"] / (LEVEL_UNIT_PER_AGENT * model.num_agents))))
    max_gain = max(f["gain"] for f in frames)
    max_edges = max(1, model.network.number_of_edges())
    images = [
        render_frame(model, layout, frames, i, scenario, max_levels, max_gain, max_edges)
        for i in range(len(frames))
    ]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output_path, save_all=True, append_images=images[1:],
        duration=int(frame_interval), loop=0,
    )
    return output_path, frames


def build_parser():
    """Return the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Render GIF animations of abstract Babel Machine scenarios."
    )
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="baseline",
                        help="preset to render (default: baseline)")
    parser.add_argument("--all", action="store_true", help="render every preset")
    parser.add_argument("--output-dir", default="results/figures",
                        help="directory for default-named GIFs")
    parser.add_argument("--output", help="explicit GIF path (single scenario only)")
    parser.add_argument("--num-agents", type=int, default=30)
    parser.add_argument("--num-steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame-interval", type=int, default=150,
                        help="milliseconds per frame")
    return parser


def main(argv=None):
    """Command-line entry point; returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.all and args.output:
        parser.error("--output cannot be combined with --all; use --output-dir")
    if args.output and not args.output.lower().endswith(".gif"):
        parser.error("--output must end in .gif")
    keys = sorted(SCENARIOS) if args.all else [args.scenario]
    for key in keys:
        path = args.output or Path(args.output_dir) / SCENARIOS[key].filename
        try:
            written, _ = render_animation(
                key, path, args.num_agents, args.num_steps, args.seed, args.frame_interval
            )
        except (ValueError, OSError) as error:
            parser.error(str(error))
        print(f"Wrote {written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
