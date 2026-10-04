# The Babel Machine

The Babel Machine is a small, reproducible Python/Mesa agent-based model for a political-theory research project on linguistic coordination, AI-mediated communication, and collective action. Agents share one project—building a tower—but communicate through conventions associated with different platforms. Version 0.1 uses transparent probabilistic rules, not an LLM or realistic language processing.

## Research question

What happens to collective coordination when agents pursuing a common project communicate through increasingly differentiated linguistic and AI-platform environments? The model also makes it possible to ask whether interoperability or convention drift changes that outcome.

## Theoretical motivation

Collective action depends in part on whether participants can communicate well enough to cooperate. A shared platform may provide common conventions, while platform proliferation may make communication more difficult. Interoperability could offset that friction. The model isolates these mechanisms so their assumptions and consequences can be inspected; it does not predict real-world effects.

## Model assumptions and agent rules

- Agents are connected by a fixed NetworkX social graph and interact only with its edges. By default the graph is Watts–Strogatz small-world.
- Each agent has a categorical integer `language_id`, an integer `platform_id`, a skill, a cooperation tendency, communication history, cumulative contribution, and successful/failed communication counts.
- An agent's platform convention is its platform ID. At the beginning of each step, with probability `platform_drift`, an agent's language becomes that convention.
- Every graph edge is attempted once per step. Communication succeeds with probability `0.9` for matching languages and `max(0, 0.9 - linguistic_distance * (1 - interoperability))` for different languages. Thus interoperability of 1 removes the divergence penalty.
- After successful communication, the pair cooperates with probability equal to the mean of their cooperation tendencies. A successful cooperative pair adds the mean of its skill levels to tower progress; half of that amount is credited to each agent.
- The model records both per-step communication/cooperation counts and cumulative counts. `average_linguistic_distance` is the share of graph edges connecting agents with different language IDs (0–1), not a measure of natural-language distance.
- No external API, scraping, machine learning, or human-like reasoning is used. Given the same configuration and seed, a run is reproducible.

## Parameters and defaults

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `num_agents` | 50 | Number of agents (at least 2). |
| `num_steps` | 100 | Number of interaction rounds. |
| `num_platforms` | 2 | Number of represented platforms (1 through `num_agents`); each receives at least one agent. |
| `linguistic_distance` | 0.5 | Strength of the cross-language communication penalty, from 0 to 1. |
| `platform_drift` | 0.01 | Per-agent, per-step probability of adopting that agent's platform convention, from 0 to 1. |
| `interoperability` | 0.0 | Cross-platform translation/interoperation, from 0 (none) to 1 (removes the divergence penalty). |
| `network_density` | 0.15 | For `small_world`, normalized control mapped to an even Watts–Strogatz neighborhood size from 2 to the largest valid even size. For `erdos_renyi`, this is the edge probability. |
| `cooperation_tendency` | 0.7 | Each agent's probability of cooperating after successful communication, from 0 to 1. |
| `random_seed` | 42 | Seed for model, Python/Mesa, NumPy, and NetworkX randomness. |
| `topology` | `small_world` | Network topology: `small_world` or `erdos_renyi`. |

## Installation

From the repository root, install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the baseline

```bash
python -m src.babel_model --output results/baseline.csv
```

The command prints JSON summary statistics and writes the per-step data to the specified CSV. All model parameters can be set as command-line options; for example:

```bash
python -m src.babel_model --num-agents 50 --num-steps 100 --num-platforms 2 \
  --linguistic-distance 0.5 --platform-drift 0.01 --interoperability 0 \
  --network-density 0.15 --cooperation-tendency 0.7 --random-seed 42
```

## Run experiments

Run the four requested one-factor sweeps and a distance-by-interoperability grid used for the heatmap:

```bash
python -m src.experiments --output-dir results
```

Defaults use five values per sweep and seeds `0,1,2,3,4`. Values, seeds, agent count, and run length are configurable:

```bash
python -m src.experiments --experiment linguistic_divergence \
  --linguistic-distance-values 0,0.25,0.5,0.75,1 --seeds 0,1,2,3,4
```

Each sweep writes a raw run-level CSV and an aggregated CSV (mean, standard deviation, and number of runs) to `results/`. The divergence experiment holds the platform count at two. Platform proliferation varies the platform count. The grid sweep crosses linguistic distance and interoperability. Aggregates are descriptive, not tests of statistical significance.

## Reproduce figures

First run the baseline and experiment commands above, then:

```bash
python -m src.visualization --results-dir results
```

The workflow creates five PNG figures in `results/figures/`: tower progress over time, communication success by linguistic distance, tower progress by platform diversity, tower progress by interoperability, and a tower-progress heatmap across linguistic distance × interoperability. Error bars, where present, show run-to-run standard deviation.

## Notebook

Open `notebooks/01_baseline.ipynb` from the repository root to run a baseline, inspect its data and summary, and plot the tower trajectory. It imports the same modules used by the command-line workflow.

## Outputs

`results/` contains reproducible experiment run-level and summary CSV files, the optional baseline CSV, and generated figures. Generated files are ignored by Git; `results/.gitkeep` preserves the directory. Regenerate outputs with the commands above.

## Known limitations

The language representation is categorical and the cross-language penalty is deliberately simple. The network is static, every edge is attempted each round, skills are fixed, and cooperation is a single probability. The platform convention is not learned except through the explicit drift rule. This is a thought experiment for exploring assumptions, not an empirically calibrated forecast.

## Interpretation

This model does **not** claim that Python agents are equivalent to humans or LLMs. Its agents are simple computational abstractions, and its results describe only the consequences of the rules specified here.
