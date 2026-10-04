import pytest

from src import animation


def test_presets_configuration():
    baseline = animation.build_model("baseline", num_agents=10, seed=1)
    stall = animation.build_model("stall", num_agents=10, seed=1)

    assert baseline.llm_systems == ("system_a",)
    assert len(stall.llm_systems) > 1
    assert stall.llm_cross_compatibility < 0.5
    assert animation.SCENARIOS["baseline"].filename == "tower_construction_baseline.gif"
    assert animation.SCENARIOS["stall"].filename == (
        "tower_coordination_stall_low_compatibility.gif"
    )


def test_invalid_inputs():
    with pytest.raises(ValueError):
        animation.build_model("nope")
    with pytest.raises(ValueError):
        animation.build_model("baseline", num_agents=0)
    with pytest.raises(ValueError):
        animation.collect_frames(animation.build_model("baseline", 6), 0)


def test_frames_are_monotonic_and_deterministic():
    first = animation.collect_frames(animation.build_model("stall", 10, 3), 5)
    second = animation.collect_frames(animation.build_model("stall", 10, 3), 5)
    progress = [frame["progress"] for frame in first]

    assert progress == [frame["progress"] for frame in second]
    assert progress == sorted(progress)
    assert len(first) == 6


@pytest.mark.parametrize("scenario", ["baseline", "stall"])
def test_gif_generated_and_deterministic(tmp_path, scenario):
    paths = [tmp_path / f"{n}.gif" for n in "ab"]
    for path in paths:
        animation.render_animation(scenario, path, num_agents=6, num_steps=2, seed=1)

    assert paths[0].stat().st_size > 0
    assert paths[0].read_bytes()[:3] == b"GIF"
    assert paths[0].read_bytes() == paths[1].read_bytes()


def test_cli_writes_default_names_and_reports_errors(tmp_path, capsys):
    assert animation.main(
        ["--all", "--num-agents", "6", "--num-steps", "1", "--output-dir", str(tmp_path)]
    ) == 0
    assert len(list(tmp_path.glob("*.gif"))) == 2
    with pytest.raises(SystemExit):
        animation.main(["--num-steps", "0", "--output-dir", str(tmp_path)])
    assert "num_steps" in capsys.readouterr().err


def test_cli_output_validation(tmp_path, capsys):
    out = tmp_path / "x.gif"
    for argv in (["--all", "--output", str(out)], ["--output", str(tmp_path / "x.png")]):
        with pytest.raises(SystemExit):
            animation.main(argv)
    assert capsys.readouterr().err.count("error") == 2
    assert animation.main(
        ["--num-agents", "6", "--num-steps", "1", "--output", str(out)]
    ) == 0
    assert out.stat().st_size > 0


def test_non_integer_inputs_rejected():
    with pytest.raises(ValueError):
        animation.build_model("baseline", num_agents="ten")
    with pytest.raises(ValueError):
        animation.collect_frames(animation.build_model("baseline", 6), None)
