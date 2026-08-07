import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from inverted_pendulum.mpc.discrete_model import natural_frequency_rad_s
from inverted_pendulum.utils.config_schema import (
    CONFIG_ROOT,
    DATA_GENERATION_CONFIG_SOURCES,
    INTELLIGENT_CONFIG_SOURCES,
    MPC_ONLY_CONFIG_SOURCES,
    ONLINE_TRAINING_CONFIG_SOURCES,
    DataGenerationRootConfig,
    RLRootConfig,
    RootConfig,
    load_data_generation_config,
    load_intelligent_config,
    load_mpc_only_config,
    load_online_training_config,
    load_visualization_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _replace_source(
    sources: tuple[Path, ...],
    filename: str,
    replacement: Path,
) -> tuple[Path, ...]:
    return tuple(replacement if source.name == filename else source for source in sources)


def test_mpc_only_config_composes_required_domains_without_rl_or_visualization() -> None:
    config = load_mpc_only_config()

    assert isinstance(config, RootConfig)
    assert {source.name for source in MPC_ONLY_CONFIG_SOURCES} == {
        "physics-config.yaml",
        "mission-config.yaml",
        "runtime-config.yaml",
        "mpc-config.yaml",
        "artifacts-config.yaml",
    }
    assert config.environment.simulation.pace_s == config.environment.simulation.timestep_s
    assert config.experiment.run_id == "pendulum_baseline"
    assert config.coordinator.event_trigger is True
    assert config.mpc.controller == "IP-dynamics-naturalPeriod"
    assert config.mpc.prediction_horizon_natural_periods == 5.0
    assert not hasattr(config, "rl")
    assert not hasattr(config, "visualization")
    assert not hasattr(config.mpc, "cost")

    # Preserve the resolved top-level layout consumed by experiment artifact readers.
    resolved = config.model_dump(mode="json", by_alias=True)
    assert set(resolved) == {"experiment", "coordinator", "environment", "mpc", "artifacts"}
    assert resolved["environment"]["simulation"]["timestep-s"] == 0.01


def test_intelligent_config_adds_only_the_deployment_rl_profile() -> None:
    config = load_intelligent_config()

    assert isinstance(config, RLRootConfig)
    assert {source.name for source in INTELLIGENT_CONFIG_SOURCES} == {
        "physics-config.yaml",
        "mission-config.yaml",
        "runtime-config.yaml",
        "mpc-config.yaml",
        "artifacts-config.yaml",
        "rl-config.yaml",
    }
    assert config.artifacts.alias == "intelligent"
    assert config.experiment.run_id == "pendulum_intelligent"
    assert config.environment.simulation.pace_s == config.environment.simulation.timestep_s
    assert config.coordinator.debug_log_every_n_steps == 25
    assert config.rl.critic_artifact_dir.endswith(
        "offline-results-structured-critic_20260614T215834"
    )
    assert config.rl.time_model_artifact_dir.endswith("current-surface-lasso-20260630T184805")
    assert config.rl.exploration_epsilon == 0.02
    assert config.rl.exploration_temperature == 0.5
    assert config.rl.training_epochs == 1
    assert not hasattr(config, "visualization")


def test_frozen_time_model_is_positive_on_deployment_grid() -> None:
    config = load_intelligent_config()
    critic_dir = REPO_ROOT / config.rl.critic_artifact_dir
    time_model_dir = REPO_ROOT / config.rl.time_model_artifact_dir
    critic_config = json.loads((critic_dir / "config.json").read_text(encoding="utf-8"))
    time_model = json.loads((time_model_dir / "final_model.json").read_text(encoding="utf-8"))
    bbar_axis = np.linspace(0.0, 1.0, int(critic_config["action-grid-count"]))
    hbar_axis = np.linspace(
        0.0,
        config.mpc.prediction_horizon_natural_periods,
        int(critic_config["action-grid-count"]),
    )
    bbar_grid, hbar_grid = np.meshgrid(bbar_axis, hbar_axis, indexing="xy")
    omega_n = natural_frequency_rad_s(config.environment.pendulum)
    horizon_steps = np.maximum(
        1,
        np.ceil(hbar_grid * np.pi * 2.0 / omega_n / config.environment.simulation.timestep_s),
    ).astype(np.int64)
    burst_steps = np.maximum(1, np.rint(bbar_grid * horizon_steps)).astype(np.int64)
    compute_time_s = np.full(bbar_grid.shape, float(time_model["intercept_s"]))

    for term in time_model["all_terms"]:
        if term["name"] == "horizon_steps":
            compute_time_s += float(term["coefficient_s"]) * horizon_steps
        elif term["name"] == "burst_steps":
            compute_time_s += float(term["coefficient_s"]) * burst_steps
        elif term["name"] == "burst_horizon_steps":
            compute_time_s += float(term["coefficient_s"]) * burst_steps * horizon_steps
        else:
            raise ValueError(f"Unsupported frozen time-model term: {term['name']}")

    assert float(compute_time_s.min()) > 0.0


def test_online_training_config_composes_training_specific_domain_overrides() -> None:
    config = load_online_training_config()

    assert isinstance(config, RLRootConfig)
    assert {source.name for source in ONLINE_TRAINING_CONFIG_SOURCES} == {
        "physics-config.yaml",
        "mission-config.yaml",
        "runtime-config.yaml",
        "mpc-config.yaml",
        "artifacts-config.yaml",
        "rl-config.yaml",
    }
    assert config.artifacts.alias == "online-training"
    assert config.experiment.run_id == "pendulum_online_training"
    assert config.environment.simulation.pace_s == 0.0
    assert config.coordinator.debug_log_every_n_steps == 25
    assert config.rl.exploration_epsilon == 0.10
    assert config.rl.exploration_temperature == 2.0
    assert config.rl.training_updates_per_transition == 8
    assert config.rl.training_epochs == 128


def test_data_generation_config_omits_unneeded_runtime_domains() -> None:
    config = load_data_generation_config()

    assert isinstance(config, DataGenerationRootConfig)
    assert {source.name for source in DATA_GENERATION_CONFIG_SOURCES} == {
        "physics-config.yaml",
        "mission-config.yaml",
        "runtime-config.yaml",
        "mpc-config.yaml",
        "artifacts-config.yaml",
        "data-generation-config.yaml",
    }
    assert not hasattr(config, "coordinator")
    assert not hasattr(config, "rl")
    assert not hasattr(config, "visualization")
    assert not hasattr(config.experiment, "initial_state")
    assert not hasattr(config.artifacts, "enabled")
    assert config.data_generation.seed is None
    assert config.data_generation.visual_artifacts is True
    assert 0.0 <= config.data_generation.bbar_min <= config.data_generation.bbar_max <= 1.0
    assert 0.0 <= config.data_generation.hbar_min <= config.data_generation.hbar_max
    assert config.data_generation.hbar_max == config.mpc.prediction_horizon_natural_periods
    assert config.mpc.split_ratios == [0.1]

    # Retain the historical JSON keys used by offline-training artifact ingestion.
    resolved = config.model_dump(mode="json", by_alias=True)
    assert set(resolved) == {"experiment", "environment", "mpc", "artifacts", "data-generation"}
    assert resolved["data-generation"]["hbar-max"] == 5.0


def test_visualization_config_loads_independently() -> None:
    visualization = load_visualization_config()

    assert visualization.update_every == 1


def test_missing_or_empty_source_list_fails_loudly() -> None:
    with pytest.raises(FileNotFoundError):
        load_mpc_only_config((CONFIG_ROOT / "missing-config.yaml",))
    with pytest.raises(ValueError, match="At least one config source"):
        load_mpc_only_config(())


def test_missing_domain_parameters_fail_composed_schema_validation(tmp_path) -> None:
    partial_path = tmp_path / "partial-config.yaml"
    partial_path.write_text(
        "artifacts:\n"
        "  root-dir: artifacts/inverted_pendulum/experiments\n"
        "  alias: null\n"
        "  enabled: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_mpc_only_config((partial_path,))


def test_unknown_mpc_controller_fails_composed_schema_validation(tmp_path) -> None:
    source = next(path for path in MPC_ONLY_CONFIG_SOURCES if path.name == "mpc-config.yaml")
    invalid_path = tmp_path / "invalid-mpc-config.yaml"
    invalid_path.write_text(
        source.read_text(encoding="utf-8").replace(
            "IP-dynamics-naturalPeriod", "missing-controller"
        ),
        encoding="utf-8",
    )
    invalid_sources = _replace_source(MPC_ONLY_CONFIG_SOURCES, source.name, invalid_path)

    with pytest.raises(ValidationError):
        load_mpc_only_config(invalid_sources)


def test_invalid_data_generation_bounds_fail_composed_schema_validation(tmp_path) -> None:
    source = next(
        path
        for path in DATA_GENERATION_CONFIG_SOURCES
        if path.name == "data-generation-config.yaml"
    )
    invalid_path = tmp_path / "invalid-data-generation-config.yaml"
    invalid_path.write_text(
        source.read_text(encoding="utf-8").replace("bbar-min: 0.01", "bbar-min: 1.1"),
        encoding="utf-8",
    )
    invalid_sources = _replace_source(DATA_GENERATION_CONFIG_SOURCES, source.name, invalid_path)

    with pytest.raises(ValidationError):
        load_data_generation_config(invalid_sources)
