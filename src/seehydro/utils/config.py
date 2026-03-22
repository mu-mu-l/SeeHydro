"""Utility helpers for locating the project root and loading OmegaConf-based configuration."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def get_project_root() -> Path:
    """Return the project root by searching upward for pyproject.toml."""
    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(
        f"无法从 '{current}' 定位项目根目录：父目录中未找到 pyproject.toml。"
    )


def load_config(
    config_path: str | Path | None = None,
    overrides: list[str] | None = None,
) -> DictConfig:
    """Load configuration from YAML and optionally merge dotlist overrides."""
    project_root = get_project_root()
    path = Path(config_path) if config_path is not None else project_root / "configs" / "default.yaml"

    if not path.is_absolute():
        path = project_root / path

    if not path.is_file():
        raise FileNotFoundError(
            f"配置文件不存在：'{path}'。请确认路径正确或提供有效的 YAML 配置文件。"
        )

    config: DictConfig = OmegaConf.load(path)
    if not isinstance(config, DictConfig):
        config = OmegaConf.create(config)

    if overrides:
        override_conf = OmegaConf.from_dotlist(overrides)
        config = OmegaConf.merge(config, override_conf)

    return config
