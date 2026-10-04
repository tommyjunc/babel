"""Reproducible Mesa model and command-line baseline runner."""

import argparse
import json
from itertools import combinations
from random import Random
from pathlib import Path

import networkx as nx
import pandas as pd
from mesa import Model

from .babel_agent import DEFAULT_LLM_SYSTEM_ID, BabelAgent


DATA_COLUMNS = [
    "step",
    "tower_progress",
    "successful_communications",
    "failed_communications",
    "communication_success_rate",
    "cooperation_events",
    "number_of_languages",
    "number_of_platforms",
    "average_linguistic_distance",
    "cumulative_successful_communications",
    "cumulative_failed_communications",
    "cumulative_cooperation_events",
    "number_of_llm_systems",
    "cross_system_successful_communications",
    "cross_system_failed_communications",
    "cross_system_success_rate",
]


def _is_probability(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and 0 <= value <= 1
    )


def resolve_llm_settings(
    num_agents,
    llm_systems=None,
    llm_system_allocation=None,
    llm_cross_compatibility=None,
    llm_compatibility_matrix=None,
):
    """Validate LLM-system settings and return ``(systems, allocation, matrix)``.

    System names are abstract labels. ``llm_system_allocation`` maps every system
    to an agent count summing to ``num_agents`` (default: an even split).
    Compatibility is a symmetric matrix of probabilities, given either as a
    nested mapping or as a list of rows in system order; by default the diagonal
    is 1 and every cross-system pair uses ``llm_cross_compatibility`` (default 1,
    i.e. no penalty).
    """
    if llm_systems is None:
        llm_systems = (DEFAULT_LLM_SYSTEM_ID,)
    if isinstance(llm_systems, str) or not isinstance(llm_systems, (list, tuple)):
        raise ValueError("llm_systems must be a list of system names")
    systems = tuple(llm_systems)
    if not systems:
        raise ValueError("llm_systems must contain at least one system")
    for name in systems:
        if not isinstance(name, str) or not name or name != name.strip():
            raise ValueError(
                "llm_systems names must be non-empty strings without "
                "surrounding whitespace"
            )
    if len(set(systems)) != len(systems):
        raise ValueError("llm_systems names must be unique")

    if llm_system_allocation is None:
        base, extra = divmod(num_agents, len(systems))
        allocation = {
            name: base + (index < extra) for index, name in enumerate(systems)
        }
    else:
        if not isinstance(llm_system_allocation, dict):
            raise ValueError("llm_system_allocation must be a mapping of counts")
        if set(llm_system_allocation) != set(systems):
            raise ValueError(
                "llm_system_allocation must list exactly the systems in llm_systems"
            )
        for name, count in llm_system_allocation.items():
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(
                    "llm_system_allocation counts must be non-negative integers"
                )
        if sum(llm_system_allocation.values()) != num_agents:
            raise ValueError("llm_system_allocation counts must sum to num_agents")
        allocation = {name: llm_system_allocation[name] for name in systems}

    if llm_compatibility_matrix is not None and llm_cross_compatibility is not None:
        raise ValueError(
            "specify only one of llm_cross_compatibility and llm_compatibility_matrix"
        )
    if llm_compatibility_matrix is None:
        cross = 1.0 if llm_cross_compatibility is None else llm_cross_compatibility
        if not _is_probability(cross):
            raise ValueError("llm_cross_compatibility must be between 0 and 1")
        matrix = {
            row: {col: 1.0 if row == col else float(cross) for col in systems}
            for row in systems
        }
        return systems, allocation, matrix

    if isinstance(llm_compatibility_matrix, (list, tuple)):
        rows = list(llm_compatibility_matrix)
        if len(rows) != len(systems) or any(
            not isinstance(row, (list, tuple)) or len(row) != len(systems)
            for row in rows
        ):
            raise ValueError(
                "llm_compatibility_matrix must be a square matrix matching llm_systems"
            )
        raw = {
            row_name: dict(zip(systems, row)) for row_name, row in zip(systems, rows)
        }
    elif isinstance(llm_compatibility_matrix, dict):
        raw = llm_compatibility_matrix
        if set(raw) != set(systems) or any(
            not isinstance(row, dict) or set(row) != set(systems)
            for row in raw.values()
        ):
            raise ValueError(
                "llm_compatibility_matrix must define every pair of llm_systems"
            )
    else:
        raise ValueError("llm_compatibility_matrix must be a mapping or list of rows")
    matrix = {}
    for row in systems:
        matrix[row] = {}
        for col in systems:
            value = raw[row][col]
            if not _is_probability(value):
                raise ValueError(
                    "llm_compatibility_matrix values must be between 0 and 1"
                )
            matrix[row][col] = float(value)
    for left, right in combinations(systems, 2):
        if matrix[left][right] != matrix[right][left]:
            raise ValueError("llm_compatibility_matrix must be symmetric")
    return systems, allocation, matrix


class BabelModel(Model):
    """A shared tower-building project mediated by simple language rules."""

    def __init__(
        self,
        num_agents=50,
        num_steps=100,
        num_platforms=2,
        linguistic_distance=0.5,
        platform_drift=0.01,
        interoperability=0.0,
        network_density=0.15,
        cooperation_tendency=0.7,
        random_seed=42,
        topology="small_world",
        llm_systems=None,
        llm_system_allocation=None,
        llm_cross_compatibility=None,
        llm_compatibility_matrix=None,
    ):
        self._validate_parameters(
            num_agents,
            num_steps,
            num_platforms,
            linguistic_distance,
            platform_drift,
            interoperability,
            network_density,
            cooperation_tendency,
            topology,
        )
        (
            self.llm_systems,
            self.llm_system_allocation,
            self.llm_compatibility_matrix,
        ) = resolve_llm_settings(
            num_agents,
            llm_systems,
            llm_system_allocation,
            llm_cross_compatibility,
            llm_compatibility_matrix,
        )
        self.llm_cross_compatibility = (
            None
            if llm_compatibility_matrix is not None
            else float(1.0 if llm_cross_compatibility is None else llm_cross_compatibility)
        )
        super().__init__(rng=random_seed)
        self.num_agents = int(num_agents)
        self.num_steps = int(num_steps)
        self.num_platforms = int(num_platforms)
        self.linguistic_distance = float(linguistic_distance)
        self.platform_drift = float(platform_drift)
        self.interoperability = float(interoperability)
        self.network_density = float(network_density)
        self.cooperation_tendency = float(cooperation_tendency)
        self.random_seed = int(random_seed)
        self.topology = topology
        self.tower_progress = 0.0
        self.current_step = 0
        self.history = []
        self.cumulative_successful_communications = 0
        self.cumulative_failed_communications = 0
        self.cumulative_cooperation_events = 0
        self.cumulative_cross_system_successes = 0
        self.cumulative_cross_system_failures = 0

        self.network = self._create_network()
        platforms = [index % self.num_platforms for index in range(self.num_agents)]
        self.random.shuffle(platforms)
        llm_system_ids = [
            name for name, count in self.llm_system_allocation.items()
            for _ in range(count)
        ]
        if len(set(llm_system_ids)) > 1:
            # Separate stream so LLM assignment leaves all other draws unchanged
            Random(f"llm-systems-{self.random_seed}").shuffle(llm_system_ids)
        self.agent_list = []
        for platform_id, llm_system_id in zip(platforms, llm_system_ids):
            agent = BabelAgent(
                model=self,
                language_id=self.random.randrange(self.num_platforms),
                platform_id=platform_id,
                skill_level=self.random.uniform(0.5, 1.5),
                cooperation_tendency=self.cooperation_tendency,
                llm_system_id=llm_system_id,
            )
            self.agent_list.append(agent)
        self.agents_by_id = {agent.unique_id: agent for agent in self.agent_list}

    @staticmethod
    def _validate_parameters(
        num_agents,
        num_steps,
        num_platforms,
        linguistic_distance,
        platform_drift,
        interoperability,
        network_density,
        cooperation_tendency,
        topology,
    ):
        if isinstance(num_agents, bool) or int(num_agents) != num_agents or num_agents < 2:
            raise ValueError("num_agents must be an integer of at least 2")
        if isinstance(num_steps, bool) or int(num_steps) != num_steps or num_steps < 0:
            raise ValueError("num_steps must be a non-negative integer")
        if (
            isinstance(num_platforms, bool)
            or int(num_platforms) != num_platforms
            or not 1 <= num_platforms <= num_agents
        ):
            raise ValueError("num_platforms must be an integer from 1 through num_agents")
        for name, value in (
            ("linguistic_distance", linguistic_distance),
            ("platform_drift", platform_drift),
            ("interoperability", interoperability),
            ("network_density", network_density),
            ("cooperation_tendency", cooperation_tendency),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if topology not in {"small_world", "erdos_renyi"}:
            raise ValueError("topology must be 'small_world' or 'erdos_renyi'")

    def _create_network(self):
        """Create a seeded Watts–Strogatz or Erdos–Renyi graph."""
        if self.topology == "erdos_renyi":
            return nx.erdos_renyi_graph(
                self.num_agents, self.network_density, seed=self.random_seed
            )
        if self.num_agents == 2:
            return nx.complete_graph(2)

        largest_even_neighborhood = self.num_agents - 1
        if largest_even_neighborhood % 2:
            largest_even_neighborhood -= 1
        target = 2 + self.network_density * (largest_even_neighborhood - 2)
        neighborhood = max(
            2, min(largest_even_neighborhood, 2 * round(target / 2))
        )
        return nx.watts_strogatz_graph(
            self.num_agents,
            neighborhood,
            0.1,
            seed=self.random_seed,
        )

    def communication_probability(
        self, language_a, language_b, llm_system_a=None, llm_system_b=None
    ):
        """Return the success probability for a pair of agents.

        The linguistic probability is multiplied by the LLM-system compatibility
        of the pair; the factor is 1 unless both system IDs are given and the
        configured compatibility is below 1.
        """
        if language_a == language_b:
            probability = 0.9
        else:
            probability = max(
                0.0,
                0.9 - self.linguistic_distance * (1.0 - self.interoperability),
            )
        if llm_system_a is None or llm_system_b is None:
            return probability
        return probability * self.llm_compatibility_matrix[llm_system_a][llm_system_b]

    def _record_cooperation(self, agent_a, agent_b):
        """Record one successful cooperative contribution to the tower."""
        contribution = (agent_a.skill_level + agent_b.skill_level) / 2
        self.tower_progress += contribution
        agent_a.accumulated_contribution += contribution / 2
        agent_b.accumulated_contribution += contribution / 2
        return contribution

    def step(self):
        """Run one round of drift, neighbor communication, and cooperation."""
        self.current_step += 1
        for agent in self.agent_list:
            if self.random.random() < self.platform_drift:
                agent.language_id = agent.platform_language

        edges = sorted(tuple(sorted(edge)) for edge in self.network.edges)
        self.random.shuffle(edges)
        step_successes = 0
        step_failures = 0
        step_cooperations = 0
        step_cross_successes = 0
        step_cross_failures = 0

        for agent_id_a, agent_id_b in edges:
            agent_a = self.agent_list[agent_id_a]
            agent_b = self.agent_list[agent_id_b]
            succeeded = (
                self.random.random()
                < self.communication_probability(
                    agent_a.language_id,
                    agent_b.language_id,
                    agent_a.llm_system_id,
                    agent_b.llm_system_id,
                )
            )
            cross_system = agent_a.llm_system_id != agent_b.llm_system_id
            cooperated = False
            if succeeded:
                step_successes += 1
                step_cross_successes += cross_system
                agent_a.successful_communications += 1
                agent_b.successful_communications += 1
                cooperation_probability = (
                    agent_a.cooperation_tendency + agent_b.cooperation_tendency
                ) / 2
                cooperated = self.random.random() < cooperation_probability
                if cooperated:
                    self._record_cooperation(agent_a, agent_b)
                    step_cooperations += 1
            else:
                step_failures += 1
                step_cross_failures += cross_system
                agent_a.failed_communications += 1
                agent_b.failed_communications += 1

            for agent, partner in ((agent_a, agent_b), (agent_b, agent_a)):
                agent.communication_history.append(
                    {
                        "step": self.current_step,
                        "partner_id": partner.unique_id,
                        "llm_system_id": agent.llm_system_id,
                        "partner_llm_system_id": partner.llm_system_id,
                        "success": succeeded,
                        "cooperated": cooperated,
                    }
                )

        self.cumulative_successful_communications += step_successes
        self.cumulative_failed_communications += step_failures
        self.cumulative_cooperation_events += step_cooperations
        self.cumulative_cross_system_successes += step_cross_successes
        self.cumulative_cross_system_failures += step_cross_failures
        step_cross_total = step_cross_successes + step_cross_failures
        possible_edges = self.network.number_of_edges()
        mismatched_edges = sum(
            self.agent_list[left].language_id != self.agent_list[right].language_id
            for left, right in edges
        )
        total_communications = step_successes + step_failures
        self.history.append(
            {
                "step": self.current_step,
                "tower_progress": self.tower_progress,
                "successful_communications": step_successes,
                "failed_communications": step_failures,
                "communication_success_rate": (
                    step_successes / total_communications
                    if total_communications
                    else 0.0
                ),
                "cooperation_events": step_cooperations,
                "number_of_languages": len(
                    {agent.language_id for agent in self.agent_list}
                ),
                "number_of_platforms": len(
                    {agent.platform_id for agent in self.agent_list}
                ),
                "average_linguistic_distance": (
                    mismatched_edges / possible_edges if possible_edges else 0.0
                ),
                "cumulative_successful_communications": (
                    self.cumulative_successful_communications
                ),
                "cumulative_failed_communications": (
                    self.cumulative_failed_communications
                ),
                "cumulative_cooperation_events": self.cumulative_cooperation_events,
                "number_of_llm_systems": len(
                    {agent.llm_system_id for agent in self.agent_list}
                ),
                "cross_system_successful_communications": step_cross_successes,
                "cross_system_failed_communications": step_cross_failures,
                "cross_system_success_rate": (
                    step_cross_successes / step_cross_total
                    if step_cross_total
                    else 0.0
                ),
            }
        )

    def run(self):
        """Run the configured number of steps and return tidy per-step data."""
        while self.current_step < self.num_steps:
            self.step()
        return pd.DataFrame(self.history, columns=DATA_COLUMNS)

    def summary(self):
        """Return end-of-run summary statistics and the run configuration."""
        total_communications = (
            self.cumulative_successful_communications
            + self.cumulative_failed_communications
        )
        cross_total = (
            self.cumulative_cross_system_successes
            + self.cumulative_cross_system_failures
        )
        return {
            "num_agents": self.num_agents,
            "num_steps": self.num_steps,
            "num_platforms": self.num_platforms,
            "linguistic_distance": self.linguistic_distance,
            "platform_drift": self.platform_drift,
            "interoperability": self.interoperability,
            "network_density": self.network_density,
            "cooperation_tendency": self.cooperation_tendency,
            "random_seed": self.random_seed,
            "topology": self.topology,
            "llm_systems": json.dumps(list(self.llm_systems)),
            "llm_system_allocation": json.dumps(self.llm_system_allocation),
            "llm_cross_compatibility": self.llm_cross_compatibility,
            "llm_compatibility_matrix": json.dumps(self.llm_compatibility_matrix),
            "tower_progress": self.tower_progress,
            "successful_communications": self.cumulative_successful_communications,
            "failed_communications": self.cumulative_failed_communications,
            "communication_success_rate": (
                self.cumulative_successful_communications / total_communications
                if total_communications
                else 0.0
            ),
            "cooperation_events": self.cumulative_cooperation_events,
            "number_of_languages": len(
                {agent.language_id for agent in self.agent_list}
            ),
            "number_of_platforms_present": len(
                {agent.platform_id for agent in self.agent_list}
            ),
            "number_of_llm_systems": len(
                {agent.llm_system_id for agent in self.agent_list}
            ),
            "cross_system_successful_communications": (
                self.cumulative_cross_system_successes
            ),
            "cross_system_failed_communications": (
                self.cumulative_cross_system_failures
            ),
            "cross_system_success_rate": (
                self.cumulative_cross_system_successes / cross_total
                if cross_total
                else 0.0
            ),
        }


def _parse_llm_systems(value):
    return [part.strip() for part in value.split(",")]


def _parse_llm_allocation(value):
    allocation = {}
    try:
        for part in value.split(","):
            name, count = part.rsplit(":", 1)
            name = name.strip()
            if name in allocation:
                raise ValueError(name)
            allocation[name] = int(count)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected unique comma-separated name:count pairs, e.g. a:30,b:20"
        ) from error
    return allocation


def _parse_llm_matrix(value):
    """Parse a JSON matrix given inline or as ``@path/to/file.json``."""
    try:
        if value.startswith("@"):
            value = Path(value[1:]).read_text()
        return json.loads(value)
    except (OSError, json.JSONDecodeError) as error:
        raise argparse.ArgumentTypeError(
            f"could not read a JSON compatibility matrix: {error}"
        ) from error


def build_parser():
    """Build the baseline model command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-agents", type=int, default=50)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--num-platforms", type=int, default=2)
    parser.add_argument("--linguistic-distance", type=float, default=0.5)
    parser.add_argument("--platform-drift", type=float, default=0.01)
    parser.add_argument("--interoperability", type=float, default=0.0)
    parser.add_argument("--network-density", type=float, default=0.15)
    parser.add_argument("--cooperation-tendency", type=float, default=0.7)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--topology", choices=("small_world", "erdos_renyi"), default="small_world"
    )
    parser.add_argument(
        "--llm-systems",
        type=_parse_llm_systems,
        default=None,
        help="Comma-separated abstract LLM-system labels, e.g. system_a,system_b",
    )
    parser.add_argument(
        "--llm-allocation",
        dest="llm_system_allocation",
        type=_parse_llm_allocation,
        default=None,
        help="Agents per system as name:count pairs (default: even split)",
    )
    parser.add_argument(
        "--llm-cross-compatibility",
        type=float,
        default=None,
        help="Compatibility (0-1) between different systems (default: 1, no penalty)",
    )
    parser.add_argument(
        "--llm-compatibility-matrix",
        type=_parse_llm_matrix,
        default=None,
        help="Symmetric JSON matrix (nested object or rows in system order) "
        "or @file.json; exclusive with --llm-cross-compatibility",
    )
    parser.add_argument(
        "--output", default="results/baseline.csv", help="Per-step CSV output path"
    )
    return parser


def main(argv=None):
    """Run a model and print summary JSON."""
    args = vars(build_parser().parse_args(argv))
    output_path = Path(args.pop("output"))
    model = BabelModel(**args)
    data = model.run()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    print(json.dumps(model.summary(), indent=2, sort_keys=True))
    print(f"Per-step results saved to {output_path}")


if __name__ == "__main__":
    main()
