import pandas as pd
import pytest

from src.babel_agent import BabelAgent
from src.babel_model import BabelModel, DATA_COLUMNS


def test_model_initialization():
    model = BabelModel(num_agents=8, num_steps=3, num_platforms=2, random_seed=7)

    assert len(model.agent_list) == 8
    assert model.network.number_of_nodes() == 8
    assert nx_is_connected_to_agents(model)
    assert len({agent.unique_id for agent in model.agent_list}) == 8
    assert {agent.platform_id for agent in model.agent_list} == {0, 1}


def nx_is_connected_to_agents(model):
    return all(0 <= node < len(model.agent_list) for node in model.network.nodes)


def test_agent_initialization():
    model = BabelModel(num_agents=4, num_steps=0, num_platforms=2, random_seed=3)
    agent = model.agent_list[0]

    assert isinstance(agent, BabelAgent)
    assert isinstance(agent.language_id, int)
    assert isinstance(agent.platform_id, int)
    assert agent.skill_level > 0
    assert agent.cooperation_tendency == model.cooperation_tendency
    assert agent.communication_history == []
    assert agent.accumulated_contribution == 0
    assert agent.successful_communications == 0
    assert agent.failed_communications == 0


def test_communication_probability():
    model = BabelModel(
        num_agents=4, num_steps=0, linguistic_distance=0.4, interoperability=0
    )

    assert model.communication_probability(1, 1) == 0.9
    assert model.communication_probability(0, 1) == pytest.approx(0.5)


def test_interoperability_raises_cross_language_probability():
    without_interoperability = BabelModel(
        num_agents=4,
        num_steps=0,
        linguistic_distance=0.6,
        interoperability=0,
    )
    with_interoperability = BabelModel(
        num_agents=4,
        num_steps=0,
        linguistic_distance=0.6,
        interoperability=1,
    )

    assert (
        with_interoperability.communication_probability(0, 1)
        >= without_interoperability.communication_probability(0, 1)
    )
    assert with_interoperability.communication_probability(0, 1) == 0.9


def test_cooperation_increases_tower_progress():
    model = BabelModel(num_agents=2, num_steps=0, num_platforms=1, random_seed=2)
    first, second = model.agent_list
    first.skill_level = 1.0
    second.skill_level = 2.0

    contribution = model._record_cooperation(first, second)

    assert contribution == 1.5
    assert model.tower_progress == 1.5
    assert first.accumulated_contribution == 0.75
    assert second.accumulated_contribution == 0.75


def test_run_collects_step_and_cumulative_results():
    model = BabelModel(num_agents=8, num_steps=3, random_seed=11)
    data = model.run()

    assert list(data.columns) == DATA_COLUMNS
    assert len(data) == 3
    assert data["step"].tolist() == [1, 2, 3]
    assert data["tower_progress"].is_monotonic_increasing
    assert data["cumulative_successful_communications"].is_monotonic_increasing


def test_fixed_seed_reproduces_data_and_summary():
    config = {
        "num_agents": 10,
        "num_steps": 5,
        "num_platforms": 3,
        "linguistic_distance": 0.6,
        "platform_drift": 0.2,
        "interoperability": 0.4,
        "random_seed": 19,
    }
    first = BabelModel(**config)
    second = BabelModel(**config)

    pd.testing.assert_frame_equal(first.run(), second.run())
    assert first.summary() == second.summary()


@pytest.mark.parametrize(
    "parameters",
    [
        {"num_agents": 1},
        {"num_platforms": 0},
        {"num_platforms": 3, "num_agents": 2},
        {"interoperability": 1.1},
        {"topology": "unsupported"},
    ],
)
def test_invalid_parameters_are_rejected(parameters):
    with pytest.raises(ValueError):
        BabelModel(**parameters)
