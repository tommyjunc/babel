"""Run reproducible parameter sweeps for The Babel Machine."""

import argparse
from itertools import product
from pathlib import Path

import pandas as pd

from .babel_model import BabelModel


DEFAULT_VALUES = {
    "linguistic_divergence": [0.0, 0.25, 0.5, 0.75, 1.0],
    "platform_proliferation": [1, 2, 5, 10, 20],
    "interoperability": [0.0, 0.25, 0.5, 0.75, 1.0],
    "drift": [0.0, 0.01, 0.05, 0.1, 0.25],
    "llm_system_mix": [1, 2, 3],
    "llm_compatibility": [0.0, 0.25, 0.5, 0.75, 1.0],
}
LLM_SYSTEM_LABELS = ("system_a", "system_b", "system_c", "system_d")
LLM_MIX_CROSS_COMPATIBILITY = 0.5
LLM_EXTRA_AGGREGATES = {
    "cross_system_success_rate_mean": ("cross_system_success_rate", "mean"),
    "cross_system_success_rate_std": ("cross_system_success_rate", "std"),
    "number_of_llm_systems_mean": ("number_of_llm_systems", "mean"),
}
EXPERIMENTS = (
    "linguistic_divergence",
    "platform_proliferation",
    "interoperability",
    "drift",
    "distance_interoperability",
    "llm_system_mix",
    "llm_compatibility",
)


def _model_config(config):
    """Translate sweep-only keys (``num_llm_systems``) into model arguments."""
    model_config = dict(config)
    count = model_config.pop("num_llm_systems", None)
    if count is not None:
        if not 1 <= count <= len(LLM_SYSTEM_LABELS):
            raise ValueError(
                f"num_llm_systems must be from 1 through {len(LLM_SYSTEM_LABELS)}"
            )
        model_config["llm_systems"] = list(LLM_SYSTEM_LABELS[:count])
    return model_config


def _run_points(
    experiment,
    parameter_names,
    points,
    seeds,
    base_config,
    output_dir,
    extra_aggregates=None,
):
    """Run each parameter point with each seed and save raw and aggregate CSVs."""
    rows = []
    for point, seed in product(points, seeds):
        config = {
            **base_config,
            **point,
            "random_seed": seed,
        }
        model = BabelModel(**_model_config(config))
        model.run()
        rows.append(
            {
                "experiment": experiment,
                **config,
                **model.summary(),
            }
        )

    raw = pd.DataFrame(rows)
    grouping = ["experiment", *parameter_names]
    summary = (
        raw.groupby(grouping, as_index=False)
        .agg(
            tower_progress_mean=("tower_progress", "mean"),
            tower_progress_std=("tower_progress", "std"),
            communication_success_rate_mean=("communication_success_rate", "mean"),
            communication_success_rate_std=("communication_success_rate", "std"),
            cooperation_events_mean=("cooperation_events", "mean"),
            **(extra_aggregates or {}),
            number_of_runs=("random_seed", "count"),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{experiment}_runs.csv"
    summary_path = output_dir / f"{experiment}_summary.csv"
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"{experiment}: {raw_path}, {summary_path}")
    return raw, summary


def run_experiments(
    experiments=None,
    seeds=(0, 1, 2, 3, 4),
    output_dir="results",
    num_agents=40,
    num_steps=50,
    values=None,
):
    """Run selected one-factor sweeps, the two-factor grid, and LLM-system sweeps."""
    selected = tuple(experiments or EXPERIMENTS)
    unknown = set(selected) - set(EXPERIMENTS)
    if unknown:
        raise ValueError(f"unknown experiment(s): {', '.join(sorted(unknown))}")
    sweep_values = {**DEFAULT_VALUES, **(values or {})}
    fixed = {"num_agents": num_agents, "num_steps": num_steps}
    output_dir = Path(output_dir)

    specifications = {
        "linguistic_divergence": (
            ["linguistic_distance"],
            [{"linguistic_distance": value} for value in sweep_values["linguistic_divergence"]],
            {"num_platforms": 2},
        ),
        "platform_proliferation": (
            ["num_platforms"],
            [{"num_platforms": value} for value in sweep_values["platform_proliferation"]],
            {},
        ),
        "interoperability": (
            ["interoperability"],
            [{"interoperability": value} for value in sweep_values["interoperability"]],
            {},
        ),
        "drift": (
            ["platform_drift"],
            [{"platform_drift": value} for value in sweep_values["drift"]],
            {},
        ),
        "distance_interoperability": (
            ["linguistic_distance", "interoperability"],
            [
                {"linguistic_distance": distance, "interoperability": interoperability}
                for distance, interoperability in product(
                    sweep_values["linguistic_divergence"],
                    sweep_values["interoperability"],
                )
            ],
            {"num_platforms": 2},
        ),
        "llm_system_mix": (
            ["num_llm_systems"],
            [{"num_llm_systems": value} for value in sweep_values["llm_system_mix"]],
            {"llm_cross_compatibility": LLM_MIX_CROSS_COMPATIBILITY},
        ),
        "llm_compatibility": (
            ["llm_cross_compatibility"],
            [
                {"llm_cross_compatibility": value}
                for value in sweep_values["llm_compatibility"]
            ],
            {"num_llm_systems": 2},
        ),
    }
    outputs = {}
    for experiment in selected:
        parameters, points, overrides = specifications[experiment]
        outputs[experiment] = _run_points(
            experiment,
            parameters,
            points,
            seeds,
            {**fixed, **overrides},
            output_dir,
            LLM_EXTRA_AGGREGATES if experiment.startswith("llm_") else None,
        )
    return outputs


def _parse_values(value, conversion):
    try:
        return [conversion(part.strip()) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated {conversion.__name__} values"
        ) from error


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("all", *EXPERIMENTS), default="all")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--num-agents", type=int, default=40)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--linguistic-distance-values", default="0,0.25,0.5,0.75,1"
    )
    parser.add_argument("--platform-values", default="1,2,5,10,20")
    parser.add_argument("--interoperability-values", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--drift-values", default="0,0.01,0.05,0.1,0.25")
    parser.add_argument(
        "--llm-system-counts",
        default="1,2,3",
        help="Numbers of LLM systems for the homogeneous-vs-mixed experiment",
    )
    parser.add_argument(
        "--llm-compatibility-values", default="0,0.25,0.5,0.75,1"
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    selected = EXPERIMENTS if args.experiment == "all" else (args.experiment,)
    values = {
        "linguistic_divergence": _parse_values(
            args.linguistic_distance_values, float
        ),
        "platform_proliferation": _parse_values(args.platform_values, int),
        "interoperability": _parse_values(args.interoperability_values, float),
        "drift": _parse_values(args.drift_values, float),
        "llm_system_mix": _parse_values(args.llm_system_counts, int),
        "llm_compatibility": _parse_values(args.llm_compatibility_values, float),
    }
    seeds = _parse_values(args.seeds, int)
    run_experiments(
        selected,
        seeds,
        args.output_dir,
        args.num_agents,
        args.num_steps,
        values,
    )


if __name__ == "__main__":
    main()
