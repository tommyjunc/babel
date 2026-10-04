"""Create publication-friendly figures from baseline and sweep CSV files."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _read_summary(results_dir, experiment):
    path = results_dir / f"{experiment}_summary.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run `python -m src.experiments` first"
        )
    return pd.read_csv(path)


def _save_errorbar(summary, x, y, xlabel, ylabel, path):
    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    axis.errorbar(
        summary[x],
        summary[f"{y}_mean"],
        yerr=summary[f"{y}_std"],
        marker="o",
        capsize=3,
        color="#315b7d",
    )
    axis.set(xlabel=xlabel, ylabel=ylabel)
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def create_figures(results_dir="results"):
    """Generate the five core figures, plus LLM-system figures when available."""
    results_dir = Path(results_dir)
    figure_dir = results_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")

    baseline_path = results_dir / "baseline.csv"
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"{baseline_path} not found; run `python -m src.babel_model` first"
        )
    baseline = pd.read_csv(baseline_path)
    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    sns.lineplot(data=baseline, x="step", y="tower_progress", ax=axis, color="#315b7d")
    axis.set(xlabel="Simulation step", ylabel="Tower progress")
    figure.tight_layout()
    figure.savefig(figure_dir / "tower_progress_over_time.png", dpi=300)
    plt.close(figure)

    distance = _read_summary(results_dir, "linguistic_divergence")
    _save_errorbar(
        distance,
        "linguistic_distance",
        "communication_success_rate",
        "Linguistic-distance penalty",
        "Communication success rate",
        figure_dir / "communication_success_by_linguistic_distance.png",
    )

    platforms = _read_summary(results_dir, "platform_proliferation")
    _save_errorbar(
        platforms,
        "num_platforms",
        "tower_progress",
        "Number of platforms",
        "Tower progress",
        figure_dir / "tower_progress_by_platform_diversity.png",
    )

    interoperability = _read_summary(results_dir, "interoperability")
    _save_errorbar(
        interoperability,
        "interoperability",
        "tower_progress",
        "Interoperability",
        "Tower progress",
        figure_dir / "tower_progress_by_interoperability.png",
    )

    grid = _read_summary(results_dir, "distance_interoperability")
    heatmap = grid.pivot(
        index="linguistic_distance",
        columns="interoperability",
        values="tower_progress_mean",
    )
    figure, axis = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        heatmap,
        annot=True,
        fmt=".1f",
        cmap="viridis",
        cbar_kws={"label": "Mean tower progress"},
        ax=axis,
    )
    axis.set(xlabel="Interoperability", ylabel="Linguistic-distance penalty")
    figure.tight_layout()
    figure.savefig(
        figure_dir / "tower_progress_distance_interoperability_heatmap.png",
        dpi=300,
    )
    plt.close(figure)

    llm_figures = (
        (
            "llm_system_mix",
            "num_llm_systems",
            "tower_progress",
            "Number of LLM systems (cross-system compatibility fixed)",
            "Tower progress",
            "tower_progress_by_llm_system_mix.png",
        ),
        (
            "llm_compatibility",
            "llm_cross_compatibility",
            "tower_progress",
            "Cross-system LLM compatibility",
            "Tower progress",
            "tower_progress_by_llm_compatibility.png",
        ),
        (
            "llm_compatibility",
            "llm_cross_compatibility",
            "cross_system_success_rate",
            "Cross-system LLM compatibility",
            "Cross-system communication success rate",
            "cross_system_success_by_llm_compatibility.png",
        ),
    )
    for experiment, x, y, xlabel, ylabel, filename in llm_figures:
        if not (results_dir / f"{experiment}_summary.csv").exists():
            print(f"Skipping {filename}: {experiment}_summary.csv not found")
            continue
        _save_errorbar(
            _read_summary(results_dir, experiment),
            x,
            y,
            xlabel,
            ylabel,
            figure_dir / filename,
        )
    print(f"Figures saved to {figure_dir}")
    return figure_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args(argv)
    create_figures(args.results_dir)


if __name__ == "__main__":
    main()
