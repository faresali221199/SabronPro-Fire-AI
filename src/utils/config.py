"""Configuration management for SabronPro Fire AI."""

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv


class Config:
    """Loads and manages YAML configuration with environment overrides."""

    def __init__(self, config_dir: Optional[str] = None):
        load_dotenv()
        self.config_dir = Path(config_dir or os.getenv("SABRONPRO_CONFIG_DIR", "configs"))
        self._configs: dict[str, Any] = {}
        self._load_all()

    def _load_all(self) -> None:
        """Load all YAML config files from config directory."""
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {self.config_dir}")
        for yaml_file in self.config_dir.glob("*.yaml"):
            name = yaml_file.stem
            with open(yaml_file, "r") as f:
                self._configs[name] = yaml.safe_load(f) or {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value using dot notation: 'system.camera.fps'."""
        parts = key.split(".")
        # First part is the config file name
        config = self._configs.get(parts[0], {})
        for part in parts[1:]:
            if isinstance(config, dict):
                config = config.get(part, default)
            else:
                return default
        return config if config is not None else default

    def get_section(self, section: str) -> dict:
        """Get entire config section."""
        return self._configs.get(section, {})

    @property
    def system(self) -> dict:
        return self._configs.get("system", {})

    @property
    def model(self) -> dict:
        return self._configs.get("model", {})

    @property
    def zones(self) -> dict:
        return self._configs.get("zones", {})

    @property
    def sensors(self) -> dict:
        return self._configs.get("sensors", {})

    @property
    def decision(self) -> dict:
        return self._configs.get("decision", {})

    def __repr__(self) -> str:
        return f"Config(dir={self.config_dir}, sections={list(self._configs.keys())})"
