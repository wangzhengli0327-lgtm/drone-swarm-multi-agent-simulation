from app.drone_sim import DEMO_PRESETS, OpenAICompatibleConfig, make_rng


def test_balanced_demo_preset_is_available() -> None:
    preset = DEMO_PRESETS["均衡巡逻演示"]
    assert preset["uav_count"] == 8
    assert preset["sim_steps"] == 120
    assert preset["task_count"] == 3


def test_seeded_rng_is_reproducible() -> None:
    first = make_rng(20250711).random()
    second = make_rng(20250711).random()
    assert first == second


def test_api_config_defaults_are_safe() -> None:
    config = OpenAICompatibleConfig(api_key="")
    assert config.requires_api_key is True
    assert config.api_key == ""
