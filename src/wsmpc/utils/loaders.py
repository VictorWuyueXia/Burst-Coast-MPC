"""Small atomic config-package loader."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from wsmpc.utils.time import realtime
from wsmpc.utils.schema import RootConfig


CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs"
DEFAULT_PACKAGE = "default"
STANDARD_PACKAGE = "standard"


@dataclass(frozen=True)
class DefaultFallback:
    """One value supplied by the default config package."""

    parameter: str
    value: Any
    reason: str


def load_config(
    package_name: str = STANDARD_PACKAGE,
    logger: logging.Logger | None = None,
) -> RootConfig:
    """Load one atomic config package, using default values for missing parameters."""

    config, fallbacks = load_config_with_fallbacks(package_name)
    warn_default_fallbacks(logger or logging.getLogger("wsmpc"), fallbacks)
    return config


def load_config_with_fallbacks(
    package_name: str = STANDARD_PACKAGE,
) -> tuple[RootConfig, list[DefaultFallback]]:
    """Load a whole config package and report every default value that was used."""

    register_resolvers()
    default_cfg = _load_package(DEFAULT_PACKAGE)

    if package_name == DEFAULT_PACKAGE:
        selected_cfg = default_cfg
        fallbacks: list[DefaultFallback] = []
    else:
        package_path = _package_file(package_name)
        if package_path.exists():
            selected_cfg = OmegaConf.load(package_path)
            fallbacks = list(_find_defaulted_values(default_cfg, selected_cfg))
        else:
            selected_cfg = default_cfg
            fallbacks = [
                DefaultFallback(
                    parameter="config-package",
                    value=DEFAULT_PACKAGE,
                    reason=f"requested package '{package_name}' was not found",
                )
            ]

    merged = OmegaConf.merge(default_cfg, selected_cfg)
    resolved = OmegaConf.to_container(merged, resolve=True)
    return RootConfig.model_validate(resolved), fallbacks


def warn_default_fallbacks(
    logger: logging.Logger,
    fallbacks: list[DefaultFallback],
) -> None:
    """Log every default fallback with its parameter path and value."""

    for fallback in fallbacks:
        logger.warning(
            "identity=Config status=fallback action=use_default "
            "action_result=default_value parameter=%s value=%r reason=%s",
            fallback.parameter,
            fallback.value,
            fallback.reason,
        )


def register_resolvers() -> None:
    """Register custom OmegaConf helpers used by human-editable YAML."""

    OmegaConf.register_new_resolver("realtime", realtime, replace=True)


def _load_package(package_name: str) -> DictConfig:
    """Load a package-level config file."""

    path = _package_file(package_name)
    if not path.exists():
        msg = f"Config package not found: {path}"
        raise FileNotFoundError(msg)
    return OmegaConf.load(path)


def _package_file(package_name: str) -> Path:
    """Return the only YAML file that defines a config package."""

    return CONFIG_ROOT / package_name / "config.yaml"


def _find_defaulted_values(
    default_cfg: DictConfig,
    selected_cfg: DictConfig,
) -> Iterator[DefaultFallback]:
    """Yield every leaf that exists in default but is missing from selected."""

    default_data = OmegaConf.to_container(default_cfg, resolve=True)
    selected_data = OmegaConf.to_container(selected_cfg, resolve=False)
    yield from _walk_missing_defaults(default_data, selected_data, path="")


def _walk_missing_defaults(
    default_value: Any,
    selected_value: Any,
    *,
    path: str,
) -> Iterator[DefaultFallback]:
    """Recursively compare selected config against default config."""

    if isinstance(default_value, dict):
        selected_dict = selected_value if isinstance(selected_value, dict) else {}
        for key, value in default_value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if key not in selected_dict:
                yield from _walk_all_default_leaves(value, path=child_path)
            else:
                yield from _walk_missing_defaults(value, selected_dict[key], path=child_path)


def _walk_all_default_leaves(value: Any, *, path: str) -> Iterator[DefaultFallback]:
    """Yield all leaf defaults under a missing config branch."""

    if isinstance(value, dict):
        for key, child_value in value.items():
            yield from _walk_all_default_leaves(child_value, path=f"{path}.{key}")
    else:
        yield DefaultFallback(
            parameter=path,
            value=value,
            reason="parameter missing from selected config package",
        )
