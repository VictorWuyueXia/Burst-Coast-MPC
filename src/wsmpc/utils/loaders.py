"""Strict atomic config-package loader."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from wsmpc.utils.config_schema import RootConfig
from wsmpc.utils.time import realtime

CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs"
STANDARD_PACKAGE = "standard"


def load_config(package_name: str) -> RootConfig:
    """Load one complete config package and validate every required field."""

    register_resolvers()
    package_path = _package_file(package_name)
    if not package_path.exists():
        msg = f"Config package not found: {package_path}"
        raise FileNotFoundError(msg)
    package_cfg = OmegaConf.load(package_path)
    resolved = OmegaConf.to_container(package_cfg, resolve=True)
    return RootConfig.model_validate(resolved)


def register_resolvers() -> None:
    """Register custom OmegaConf helpers used by human-editable YAML."""

    OmegaConf.register_new_resolver("realtime", realtime, replace=True)


def _package_file(package_name: str) -> Path:
    """Return the only YAML file that defines a config package."""

    return CONFIG_ROOT / package_name / "config.yaml"
