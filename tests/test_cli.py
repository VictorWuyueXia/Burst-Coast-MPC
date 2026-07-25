from typer.testing import CliRunner

from bcmpc import app
from inverted_pendulum.utils.config_schema import (
    load_config,
    load_data_generation_config,
    load_online_training_config,
)
from inverted_pendulum.utils.messages import EpisodeResult, ExperimentSummary, RLStepRecord


def _require_matplotlib() -> None:
    import pytest

    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")


def test_cli_run_episode(monkeypatch) -> None:
    runner = CliRunner()
    config = load_config()
    config.experiment.max_steps = 2
    config.environment.simulation.timestep_s = 0.25
    config.environment.simulation.pace_s = 0.0
    config.artifacts.enabled = False

    def load_config_stub():
        return config

    monkeypatch.setattr("bringup.mpc_only_mode.load_config", load_config_stub)

    result = runner.invoke(
        app,
        ["--inverted-pendulum", "mpc-only", "--no-visual"],
    )

    assert result.exit_code == 0, result.output
    assert "pendulum_baseline" in result.output


def test_cli_run_episode_writes_artifacts_with_alias(tmp_path, monkeypatch) -> None:
    _require_matplotlib()
    runner = CliRunner()
    config = load_config()
    config.artifacts.root_dir = str(tmp_path)
    config.experiment.max_steps = 2
    config.environment.simulation.timestep_s = 0.25
    config.environment.simulation.pace_s = 0.0

    def load_config_stub():
        return config

    monkeypatch.setattr("bringup.mpc_only_mode.load_config", load_config_stub)

    result = runner.invoke(
        app,
        [
            "--inverted-pendulum",
            "mpc-only",
            "--alias",
            "smoke",
            "--no-visual",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "pendulum_baseline" in result.output
    assert "artifact_dir" in result.output
    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    for name in ["states", "energy", "phase", "commands"]:
        assert (run_dirs[0] / "figures" / f"{name}.png").exists()


def test_cli_intelligent_routes_weighted_exploration(monkeypatch) -> None:
    runner = CliRunner()
    calls = []

    def run_intelligent_stub(task, *, alias, no_visual, with_exploration, console):
        calls.append((task, alias, no_visual, with_exploration, console))

    monkeypatch.setattr("bcmpc.run_intelligent_mode", run_intelligent_stub)

    result = runner.invoke(
        app,
        ["--inverted-pendulum", "intelligent", "--with-exploration", "--no-visual"],
    )

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    task, alias, no_visual, with_exploration, _ = calls[0]
    assert task == "inverted_pendulum"
    assert alias is None
    assert no_visual is True
    assert with_exploration is True


def test_cli_train_config_epochs_create_separate_artifacts(tmp_path, monkeypatch) -> None:
    runner = CliRunner()
    config = load_online_training_config()
    config.artifacts.root_dir = str(tmp_path / "runs")
    config.artifacts.alias = "online"
    config.rl.training_epochs = 2
    config.experiment.max_steps = 1
    config.environment.simulation.pace_s = 0.0
    policy = object()
    coordinator_policy_ids = []
    episode_ids = []
    fit_policy_ids = []

    def load_online_training_config_stub():
        return config

    class EpochCoordinatorStub:
        def __init__(
            self,
            coordinator_config,
            environment_config,
            experiment_config,
            mpc_config,
            *,
            logger,
            rl_policy,
            rl_config,
            rl_explore,
        ) -> None:
            coordinator_policy_ids.append(id(rl_policy))
            episode_ids.append(experiment_config.episode_id)

        def run_episode(self, third_person_observers):
            episode_id = episode_ids[-1]
            summary = ExperimentSummary(
                run_id=config.experiment.run_id,
                episode_id=episode_id,
                status="max_steps_reached",
                total_steps=1,
                final_t_index=1,
                final_t_sec=config.environment.simulation.timestep_s,
                goal_reached=False,
                records_emitted=0,
                total_wall_time_s=0.0,
                final_observation=None,
            )
            record = RLStepRecord(
                run_id=config.experiment.run_id,
                episode_id=episode_id,
                replan_index=0,
                start_t_index=0,
                start_t_sec=0.0,
                end_t_index=1,
                end_t_sec=config.environment.simulation.timestep_s,
                s_sin_theta=1.0,
                s_cos_theta=0.0,
                s_omega_rad_s=0.0,
                bbar=0.5,
                hbar=0.5,
                burst_steps=1,
                horizon_steps=2,
                next_s_sin_theta=1.0,
                next_s_cos_theta=0.0,
                next_s_omega_rad_s=0.0,
                done=True,
                step_cost=1.0,
                return_cost=1.0,
                u_nm_json="[]",
                solve_time_s=0.01,
                plan_id=f"plan-{episode_id}",
            )
            return EpisodeResult(summary=summary, records=[], rl_records=[record])

    class OnlinePolicyTrainerStub:
        def __init__(self, rl_policy, rl_config) -> None:
            fit_policy_ids.append(id(rl_policy))
            self.snapshot_index = len(fit_policy_ids)

        def fit(self, records):
            assert records
            snapshot_dir = tmp_path / "snapshots" / f"epoch-{self.snapshot_index}"
            snapshot_dir.mkdir(parents=True)
            return snapshot_dir

    monkeypatch.setattr(
        "bringup.train_mode.load_online_training_config",
        load_online_training_config_stub,
    )
    monkeypatch.setattr("bringup.train_mode.StructuredCriticPolicy", lambda *_: policy)
    monkeypatch.setattr("bringup.train_mode.EpochCoordinator", EpochCoordinatorStub)
    monkeypatch.setattr("bringup.train_mode.OnlinePolicyTrainer", OnlinePolicyTrainerStub)
    monkeypatch.setattr("bringup.train_mode.create_artifact_figures", lambda *_: {})

    result = runner.invoke(app, ["--inverted-pendulum", "train", "--no-visual"])

    assert result.exit_code == 0, result.output
    run_dirs = sorted((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 2
    assert run_dirs[0].name.startswith("online-epoch-1_")
    assert run_dirs[1].name.startswith("online-epoch-2_")
    assert episode_ids == [0, 1]
    assert coordinator_policy_ids == [id(policy), id(policy)]
    assert fit_policy_ids == [id(policy), id(policy)]
    assert "training_epochs" in result.output
    assert "epoch_artifact_dirs" in result.output


def test_cli_generate_mc_data_writes_step_and_rl_artifacts(tmp_path, monkeypatch) -> None:
    _require_matplotlib()
    runner = CliRunner()
    config = load_data_generation_config()
    config.artifacts.root_dir = str(tmp_path)
    config.data_generation.episodes = 1
    config.data_generation.bbar_min = 1.0
    config.data_generation.bbar_max = 1.0
    config.data_generation.hbar_min = 1.0
    config.data_generation.hbar_max = 1.0
    config.data_generation.visual_artifacts = True
    config.experiment.max_steps = 2
    config.environment.simulation.timestep_s = 0.25
    config.environment.simulation.pace_s = 0.0
    config.environment.goal.hold_steps = 999

    def load_data_generation_config_stub():
        return config

    monkeypatch.setattr(
        "bringup.monte_carlo_mode.load_data_generation_config",
        load_data_generation_config_stub,
    )

    result = runner.invoke(app, ["--inverted-pendulum", "montecarlo"])

    assert result.exit_code == 0, result.output
    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "steps.csv").exists()
    assert (run_dirs[0] / "rl_steps.csv").exists()
    for name in ["states", "energy", "phase", "commands", "rl_timeseries"]:
        assert (run_dirs[0] / "figures" / f"{name}.png").exists()
    assert "epochs" in result.output
    assert "rl_steps" in result.output


def test_cli_generate_mc_data_epochs_create_separate_artifacts(tmp_path, monkeypatch) -> None:
    runner = CliRunner()
    config = load_data_generation_config()
    config.artifacts.root_dir = str(tmp_path)
    config.artifacts.alias = "mc"
    config.data_generation.episodes = 1
    config.data_generation.bbar_min = 1.0
    config.data_generation.bbar_max = 1.0
    config.data_generation.hbar_min = 1.0
    config.data_generation.hbar_max = 1.0
    config.data_generation.visual_artifacts = False
    config.experiment.max_steps = 1
    config.environment.simulation.timestep_s = 0.25
    config.environment.simulation.pace_s = 0.0
    config.environment.goal.hold_steps = 999

    def load_data_generation_config_stub():
        return config

    monkeypatch.setattr(
        "bringup.monte_carlo_mode.load_data_generation_config",
        load_data_generation_config_stub,
    )

    result = runner.invoke(
        app,
        ["--inverted-pendulum", "montecarlo", "--epochs", "2"],
    )

    assert result.exit_code == 0, result.output
    run_dirs = sorted(tmp_path.iterdir())
    assert len(run_dirs) == 2
    assert run_dirs[0].name.startswith("mc-epoch-1_")
    assert run_dirs[1].name.startswith("mc-epoch-2_")
    for run_dir in run_dirs:
        assert (run_dir / "steps.csv").exists()
        assert (run_dir / "rl_steps.csv").exists()
        assert (run_dir / "figures" / "rl_timeseries.png").exists()
        assert not (run_dir / "figures" / "states.png").exists()
