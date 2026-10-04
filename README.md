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

## LLM-system layer (optional abstraction)

Each agent also has a categorical `llm_system_id`, an **abstract label** (default `system_a`, `system_b`, …) for the AI system that mediates its communication. These labels are illustrative placeholders; they do not refer to, or make claims about, any real vendor or model. **This mode does not call any external LLM API and does not establish empirical performance comparisons**: it only adds one more categorical source of communication friction.

- Communication success is `linguistic probability × llm_compatibility[system_i][system_j]`, where the linguistic probability is the rule described above.
- By default there is one system and compatibility is 1, so no penalty applies and results for the original columns are identical to the model without this layer. Differentiation is opt-in.
- The compatibility matrix is a probability matrix that must be complete (every pair of systems) and **symmetric** (no directional compatibility); each value must lie in [0, 1]. Without an explicit matrix, the diagonal is 1 and every off-diagonal value is `llm_cross_compatibility`.
- Agents are assigned to systems by `llm_system_allocation` (counts summing to `num_agents`) or, by default, an even split. The (seeded) assignment shuffle uses its own random stream, so other random draws are unchanged by adding systems.

| Parameter | CLI option | Default | Meaning |
| --- | --- | ---: | --- |
| `llm_systems` | `--llm-systems a,b` | `system_a` | Unique, non-empty system labels. |
| `llm_system_allocation` | `--llm-allocation a:30,b:20` | even split | Agents per system; must list every system and sum to `num_agents`. |
| `llm_cross_compatibility` | `--llm-cross-compatibility 0.5` | 1 (no penalty) | Compatibility between different systems, 0–1. |
| `llm_compatibility_matrix` | `--llm-compatibility-matrix '[[1,0.4],[0.4,1]]'` or `@matrix.json` | none | Full symmetric matrix as nested object (`{"a": {"b": 0.4, ...}}`) or rows in `llm_systems` order. Exclusive with `llm_cross_compatibility`. |

```bash
python -m src.babel_model --llm-systems system_a,system_b \
  --llm-allocation system_a:30,system_b:20 --llm-cross-compatibility 0.5
```

Per-agent communication histories record `llm_system_id` and `partner_llm_system_id`. Four columns are appended to the per-step CSV: `number_of_llm_systems`, `cross_system_successful_communications`, `cross_system_failed_communications`, and `cross_system_success_rate` (0 when no cross-system communication occurred). The summary JSON adds the same final metrics plus the serialized `llm_systems`, `llm_system_allocation`, `llm_cross_compatibility` and `llm_compatibility_matrix`.

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

Run the four one-factor sweeps, a distance-by-interoperability grid used for the heatmap, and two LLM-system experiments:

```bash
python -m src.experiments --output-dir results
```

Defaults use five values per sweep and seeds `0,1,2,3,4`. Values, seeds, agent count, and run length are configurable:

```bash
python -m src.experiments --experiment linguistic_divergence \
  --linguistic-distance-values 0,0.25,0.5,0.75,1 --seeds 0,1,2,3,4
```

Each sweep writes a raw run-level CSV and an aggregated CSV (mean, standard deviation, and number of runs) to `results/`. The divergence experiment holds the platform count at two. Platform proliferation varies the platform count. The grid sweep crosses linguistic distance and interoperability. Aggregates are descriptive, not tests of statistical significance.

The LLM-system experiments are `llm_system_mix` (homogeneous vs mixed populations of 1, 2 or 3 abstract systems at an even split, cross-system compatibility fixed at 0.5; `--llm-system-counts 1,2,3`) and `llm_compatibility` (two systems, cross-system compatibility swept; `--llm-compatibility-values 0,0.25,0.5,0.75,1`). They write `llm_system_mix_*.csv` and `llm_compatibility_*.csv`; their summaries add mean/std of `cross_system_success_rate` and the mean number of represented systems. The mean cross-system rate counts runs with no cross-system communication (e.g. a single system) as 0.

## Reproduce figures

First run the baseline and experiment commands above, then:

```bash
python -m src.visualization --results-dir results
```

The workflow creates five PNG figures in `results/figures/`: tower progress over time, communication success by linguistic distance, tower progress by platform diversity, tower progress by interoperability, and a tower-progress heatmap across linguistic distance × interoperability. Error bars, where present, show run-to-run standard deviation. If the LLM-system experiment summaries exist, three more figures are written: `tower_progress_by_llm_system_mix.png`, `tower_progress_by_llm_compatibility.png` and `cross_system_success_by_llm_compatibility.png`; otherwise they are skipped.

## Animations

Render GIFs of two abstract scenarios from the repository root:

```bash
python -m src.animation --scenario baseline --output-dir results/figures
python -m src.animation --scenario stall --output-dir results/figures
python -m src.animation --all   # both presets with default file names
```

Defaults write `tower_construction_baseline.gif` and `tower_coordination_stall_low_compatibility.gif` to `results/figures/`. Options: `--num-agents`, `--num-steps`, `--seed`, `--frame-interval` (ms), `--output` (single explicit `.gif` path) and `--output-dir`. Fill colour shows language, marker shape shows the abstract LLM system, and dashed edges are failed interactions. Tower progress is monotonic, so the stall scenario shows a loss of coordinated construction capacity (low construction rate, many failures), never a shrinking tower. The scenarios are abstract and say nothing about real LLM platforms.

## Quarto report

`reports/babel_analysis.qmd` is a publication-oriented report (research question, mechanism, interpretation guide, limitations). It reads existing `results/*.csv` and figures when present and renders without them. Rendering requires a [Quarto](https://quarto.org) installation and a Jupyter kernel for Python (`python -m pip install jupyter`); it is not needed for the rest of the project.

```bash
quarto render reports/babel_analysis.qmd
```

## Notebook

Open `notebooks/01_baseline.ipynb` from the repository root to run a baseline, inspect its data and summary, and plot the tower trajectory. It imports the same modules used by the command-line workflow.

## Outputs

`results/` contains reproducible experiment run-level and summary CSV files, the optional baseline CSV, and generated figures. Generated files are ignored by Git; `results/.gitkeep` preserves the directory. Regenerate outputs with the commands above.

## Known limitations

The language representation is categorical and the cross-language penalty is deliberately simple. The network is static, every edge is attempted each round, skills are fixed, and cooperation is a single probability. The platform convention is not learned except through the explicit drift rule. This is a thought experiment for exploring assumptions, not an empirically calibrated forecast.

The LLM-system layer is a single symmetric probability multiplier on communication; it does not model capability, translation quality, learning, or any real system. Results from it are consequences of the stated rules, not empirical comparisons.

## Interpretation

This model does **not** claim that Python agents are equivalent to humans or LLMs. Its agents are simple computational abstractions, and its results describe only the consequences of the rules specified here.
