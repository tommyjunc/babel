import json

import pandas as pd
import pytest

from src import experiments, visualization
from src.babel_model import BabelModel, DATA_COLUMNS, build_parser, main

LEGACY_COLUMNS = DATA_COLUMNS[:12]


def mixed_model(**overrides):
    config = {
        "num_agents": 10,
        "num_steps": 5,
        "llm_systems": ["system_a", "system_b"],
        "llm_cross_compatibility": 0.4,
        "random_seed": 5,
    }
    return BabelModel(**{**config, **overrides})


def test_default_has_one_system_and_no_penalty():
    model = BabelModel(num_agents=6, num_steps=0, linguistic_distance=0.4)

    assert model.llm_systems == ("system_a",)
    assert {agent.llm_system_id for agent in model.agent_list} == {"system_a"}
    assert model.communication_probability(0, 1, "system_a", "system_a") == (
        model.communication_probability(0, 1)
    )


def test_default_and_explicit_full_compatibility_match_legacy_behavior():
    config = {"num_agents": 12, "num_steps": 8, "platform_drift": 0.1, "random_seed": 3}
    default = BabelModel(**config).run()
    explicit = BabelModel(
        **config, llm_systems=["x", "y"], llm_cross_compatibility=1.0
    ).run()

    pd.testing.assert_frame_equal(default[LEGACY_COLUMNS], explicit[LEGACY_COLUMNS])
    assert default["cross_system_success_rate"].eq(0).all()


def test_even_default_allocation_and_agent_ids():
    model = mixed_model(num_agents=7, llm_systems=["a", "b", "c"])

    assert model.llm_system_allocation == {"a": 3, "b": 2, "c": 2}
    counts = pd.Series([agent.llm_system_id for agent in model.agent_list]).value_counts()
    assert counts.to_dict() == {"a": 3, "b": 2, "c": 2}


def test_explicit_allocation_and_zero_count_systems():
    model = mixed_model(llm_system_allocation={"system_a": 10, "system_b": 0})

    assert {agent.llm_system_id for agent in model.agent_list} == {"system_a"}
    assert model.run()["number_of_llm_systems"].eq(1).all()


@pytest.mark.parametrize(
    "overrides",
    [
        {"llm_systems": []},
        {"llm_systems": "system_a"},
        {"llm_systems": ["a", "a"]},
        {"llm_systems": ["a", ""]},
        {"llm_systems": ["a", " b"]},
        {"llm_systems": ["a", 1]},
        {"llm_system_allocation": {"system_a": 5, "system_b": 4}},
        {"llm_system_allocation": {"system_a": 10}},
        {"llm_system_allocation": {"system_a": 11, "system_b": -1}},
        {"llm_system_allocation": {"system_a": 5.0, "system_b": 5}},
        {"llm_system_allocation": {"system_a": 5, "system_b": 5, "other": 0}},
        {"llm_cross_compatibility": 1.2},
        {"llm_cross_compatibility": -0.1},
        {"llm_cross_compatibility": float("nan")},
        {"llm_cross_compatibility": True},
        {"llm_compatibility_matrix": [[1, 0.5], [0.5, 1]]},
        {
            "llm_cross_compatibility": None,
            "llm_compatibility_matrix": [[1, 0.5]],
        },
        {
            "llm_cross_compatibility": None,
            "llm_compatibility_matrix": [[1, 0.5], [0.4, 1]],
        },
        {
            "llm_cross_compatibility": None,
            "llm_compatibility_matrix": [[1, 1.5], [1.5, 1]],
        },
        {
            "llm_cross_compatibility": None,
            "llm_compatibility_matrix": {"system_a": {"system_a": 1}},
        },
        {
            "llm_cross_compatibility": None,
            "llm_compatibility_matrix": {
                "system_a": {"system_a": 1, "system_b": 0.5},
                "system_b": {"system_a": 0.5},
            },
        },
        {"llm_cross_compatibility": None, "llm_compatibility_matrix": "1"},
    ],
)
def test_invalid_llm_settings_are_rejected(overrides):
    with pytest.raises(ValueError):
        mixed_model(**overrides)


def test_matrix_as_rows_or_mapping_is_equivalent():
    rows = mixed_model(
        llm_cross_compatibility=None, llm_compatibility_matrix=[[0.9, 0.2], [0.2, 0.8]]
    )
    mapping = mixed_model(
        llm_cross_compatibility=None,
        llm_compatibility_matrix={
            "system_a": {"system_a": 0.9, "system_b": 0.2},
            "system_b": {"system_a": 0.2, "system_b": 0.8},
        },
    )

    assert rows.llm_compatibility_matrix == mapping.llm_compatibility_matrix
    assert rows.llm_cross_compatibility is None


def test_compatibility_scales_communication_probability():
    model = mixed_model(linguistic_distance=0.4)

    assert model.communication_probability(0, 0, "system_a", "system_a") == 0.9
    assert model.communication_probability(0, 0, "system_a", "system_b") == (
        pytest.approx(0.9 * 0.4)
    )
    assert model.communication_probability(0, 1, "system_b", "system_a") == (
        pytest.approx(0.5 * 0.4)
    )


def test_zero_compatibility_blocks_cross_system_success_only():
    model = mixed_model(num_steps=10, llm_cross_compatibility=0.0)
    data = model.run()

    assert data["cross_system_successful_communications"].sum() == 0
    assert data["cross_system_failed_communications"].sum() > 0
    assert model.summary()["cross_system_success_rate"] == 0
    assert data["number_of_llm_systems"].eq(2).all()


def test_compatibility_lowers_success_relative_to_full_compatibility():
    kwargs = {"num_agents": 20, "num_steps": 20, "random_seed": 1}
    full = mixed_model(llm_cross_compatibility=1.0, **kwargs).summary()
    low = mixed_model(llm_cross_compatibility=0.1, **kwargs)
    low.run()
    full_model = mixed_model(llm_cross_compatibility=1.0, **kwargs)
    full_model.run()

    assert low.summary()["cross_system_success_rate"] < (
        full_model.summary()["cross_system_success_rate"]
    )
    assert full["llm_cross_compatibility"] == 1.0


def test_histories_and_summary_record_systems():
    model = mixed_model()
    model.run()
    agent = model.agent_list[0]
    entry = agent.communication_history[0]
    partner = model.agents_by_id[entry["partner_id"]]
    summary = model.summary()

    assert entry["llm_system_id"] == agent.llm_system_id
    assert entry["partner_llm_system_id"] == partner.llm_system_id
    assert json.loads(summary["llm_systems"]) == ["system_a", "system_b"]
    assert json.loads(summary["llm_system_allocation"]) == {
        "system_a": 5,
        "system_b": 5,
    }
    assert summary["number_of_llm_systems"] == 2
    total = summary["cross_system_successful_communications"] + (
        summary["cross_system_failed_communications"]
    )
    assert total > 0


def test_fixed_seed_reproduces_mixed_runs():
    first = mixed_model()
    second = mixed_model()

    pd.testing.assert_frame_equal(first.run(), second.run())
    assert first.summary() == second.summary()
    assert [a.llm_system_id for a in first.agent_list] == [
        a.llm_system_id for a in second.agent_list
    ]


def test_cli_options_round_trip(tmp_path, capsys):
    output = tmp_path / "run.csv"
    main(
        [
            "--num-agents", "8", "--num-steps", "3", "--output", str(output),
            "--llm-systems", "system_a,system_b",
            "--llm-allocation", "system_a:5,system_b:3",
            "--llm-cross-compatibility", "0.25",
        ]
    )
    summary = json.loads(capsys.readouterr().out.split("\nPer-step")[0])

    assert json.loads(summary["llm_system_allocation"]) == {
        "system_a": 5,
        "system_b": 3,
    }
    assert summary["llm_cross_compatibility"] == 0.25
    assert list(pd.read_csv(output).columns) == DATA_COLUMNS


def test_cli_matrix_inline_and_file(tmp_path):
    matrix_file = tmp_path / "m.json"
    matrix_file.write_text("[[1, 0.5], [0.5, 1]]")
    parser = build_parser()
    inline = parser.parse_args(
        ["--llm-systems", "a,b", "--llm-compatibility-matrix", "[[1,0.2],[0.2,1]]"]
    )
    from_file = parser.parse_args(
        ["--llm-compatibility-matrix", f"@{matrix_file}"]
    )

    assert inline.llm_compatibility_matrix == [[1, 0.2], [0.2, 1]]
    assert from_file.llm_compatibility_matrix == [[1, 0.5], [0.5, 1]]
    for bad in (["--llm-compatibility-matrix", "{oops"],
                ["--llm-allocation", "a=1"],
                ["--llm-allocation", "a:1,a:2"]):
        with pytest.raises(SystemExit):
            parser.parse_args(bad)


def test_llm_experiments_write_outputs_and_figures(tmp_path):
    outputs = experiments.run_experiments(
        ["llm_system_mix", "llm_compatibility"],
        seeds=(0, 1),
        output_dir=tmp_path,
        num_agents=12,
        num_steps=4,
        values={"llm_system_mix": [1, 2], "llm_compatibility": [0.0, 1.0]},
    )

    mix_raw, mix_summary = outputs["llm_system_mix"]
    assert sorted(mix_raw["number_of_llm_systems"].unique()) == [1, 2]
    assert mix_summary["number_of_runs"].tolist() == [2, 2]
    assert mix_summary.loc[0, "cross_system_success_rate_mean"] == 0
    compat_raw, compat_summary = outputs["llm_compatibility"]
    assert compat_summary["llm_cross_compatibility"].tolist() == [0.0, 1.0]
    assert compat_summary.loc[0, "cross_system_success_rate_mean"] == 0
    assert compat_summary.loc[1, "cross_system_success_rate_mean"] > 0
    for name in ("llm_system_mix", "llm_compatibility"):
        assert (tmp_path / f"{name}_runs.csv").exists()
        assert (tmp_path / f"{name}_summary.csv").exists()

    pd.testing.assert_frame_equal(
        compat_raw,
        experiments.run_experiments(
            ["llm_compatibility"],
            seeds=(0, 1),
            output_dir=tmp_path / "again",
            num_agents=12,
            num_steps=4,
            values={"llm_compatibility": [0.0, 1.0]},
        )["llm_compatibility"][0],
    )

    BabelModel(num_agents=4, num_steps=2).run().to_csv(
        tmp_path / "baseline.csv", index=False
    )
    for name in (
        "linguistic_divergence",
        "platform_proliferation",
        "interoperability",
        "distance_interoperability",
    ):
        experiments.run_experiments(
            [name], seeds=(0,), output_dir=tmp_path, num_agents=6, num_steps=2,
            values={
                "linguistic_divergence": [0.5],
                "platform_proliferation": [2],
                "interoperability": [0.5],
            },
        )
    experiments.run_experiments(
        ["drift"], seeds=(0,), output_dir=tmp_path, num_agents=6, num_steps=2,
        values={"drift": [0.1]},
    )
    figures = visualization.create_figures(tmp_path)
    for name in (
        "tower_progress_by_llm_system_mix.png",
        "tower_progress_by_llm_compatibility.png",
        "cross_system_success_by_llm_compatibility.png",
        "tower_progress_over_time.png",
    ):
        assert (figures / name).exists()


def test_invalid_llm_experiment_count_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        experiments.run_experiments(
            ["llm_system_mix"], seeds=(0,), output_dir=tmp_path,
            values={"llm_system_mix": [9]},
        )
