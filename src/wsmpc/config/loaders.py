"""Configuration composition for the CLI."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from wsmpc.config.schema import RootConfig
from wsmpc.core.time import realtime


def register_resolvers() -> None:
    """Register custom OmegaConf helpers used by human-editable YAML."""

    OmegaConf.register_new_resolver("realtime", realtime, replace=True)


def load_config(
    *,
    config_name: str = "config",
    config_dir: Path | None = None,
    experiment_name: str | None = None,
    overrides: Iterable[str] | None = None,
) -> RootConfig:
    """Load the root config and selected dedicated experiment/node config files."""

    # Register resolvers before loading any interpolation that references them.
    register_resolvers()

    # Read the root selector file; it decides which dedicated configs compose the run.
    root = Path(config_dir) if config_dir is not None else Path.cwd() / "configs"
    selector_cfg = _load_yaml(root / f"{config_name}.yaml")
    selectors = _to_plain_dict(selector_cfg)

    # Resolve config names with CLI experiment override support.
    selected_experiment = experiment_name or str(selectors.get("experiment", "pendulum_baseline"))
    selected_nodes = selectors.get("nodes", {})
    selected_runtime = str(selectors.get("runtime", "local_limited"))

    # Merge only fully formed config documents, keeping selector metadata out of RootConfig.
    merged = OmegaConf.merge(
        OmegaConf.create({"logging": selectors.get("logging", {"level": "INFO"})}),
        _load_yaml(root / "experiment" / f"{selected_experiment}.yaml"),
        _load_yaml(root / "nodes" / f"{selected_nodes.get('coordinator', 'coordinator')}.yaml"),
        _load_yaml(root / "nodes" / f"{selected_nodes.get('environment', 'environment')}.yaml"),
        _load_yaml(root / "runtime" / f"{selected_runtime}.yaml"),
    )

    # Apply exact dot-list overrides last so tests and CLI flags can pin parameters explicitly.
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(list(overrides)))

    # Resolve interpolation, then validate aliases and invariants with Pydantic.
    resolved = OmegaConf.to_container(merged, resolve=True)
    return RootConfig.model_validate(resolved)


def _load_yaml(path: Path) -> DictConfig:
    """Load one YAML document with a clear path-level failure if it is missing."""

    if not path.exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)
    return OmegaConf.load(path)


def _to_plain_dict(config: DictConfig) -> dict[str, Any]:
    """Convert selector config to plain containers without resolving node YAML."""

    value = OmegaConf.to_container(config, resolve=False)
    if not isinstance(value, dict):
        msg = "Root config must be a mapping"
        raise TypeError(msg)
    return value
