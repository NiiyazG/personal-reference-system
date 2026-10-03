"""Configuration for the reference system.

Everything that used to be a machine-specific constant (the storage root, the
storage quotas, the minimum free disk, the OCR engine and its language, the
embedding model and the operating mode) now has a generic default plus a way to
override it. Precedence, lowest to highest:

1. the built-in defaults defined here;
2. a JSON config file (``--config`` / ``REFERENCE_CONFIG``);
3. ``REFERENCE_*`` environment variables;
4. explicit CLI arguments.

The storage root is one directory; ``init`` creates the layout inside it. OCR
and embeddings are opt-in: the defaults choose neither, so a fresh install
indexes and searches full text with no model on disk and no network access.
An embedding model is never bundled — only its download is documented.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

ENV_PREFIX = "REFERENCE_"

DEFAULT_MODE = "LOCAL_ONLY"
DEFAULT_OCR_ENGINE = "none"
DEFAULT_OCR_LANGUAGE = "ru"
DEFAULT_EMBEDDING_MODEL = "none"
DEFAULT_ALLOWED_PROFILE = "default"

# Reasonable for a new single-user install: a 30 GiB envelope split into a
# 28 GiB live store and a 2 GiB temporary processing area, with a 20 GiB free
# disk floor so import never fills the volume it lives on.
DEFAULT_STORAGE_QUOTA: dict[str, int] = {
    "total_gib": 30,
    "live_gib": 28,
    "backup_gib": 0,
    "temporary_gib": 2,
    "minimum_free_disk_gib": 20,
}

# Environment variable suffix -> storage quota key.
_QUOTA_ENV = {
    "TOTAL_GIB": "total_gib",
    "LIVE_GIB": "live_gib",
    "BACKUP_GIB": "backup_gib",
    "TEMPORARY_GIB": "temporary_gib",
    "MINIMUM_FREE_DISK_GIB": "minimum_free_disk_gib",
}


class ConfigError(ValueError):
    """The supplied configuration is missing or malformed."""


def default_root() -> Path:
    """The storage root used when nothing else is configured.

    ``REFERENCE_ROOT`` wins; otherwise a ``data`` directory next to the
    current working directory, which is what running the CLI from the project
    checkout produces.
    """
    value = os.environ.get(f"{ENV_PREFIX}ROOT")
    if value:
        return Path(value).expanduser()
    return Path.cwd() / "data"


def default_config_file() -> Path | None:
    """The config file used when ``--config`` is not given.

    ``REFERENCE_CONFIG`` wins; otherwise ``reference.config.json`` in the
    current working directory, if it exists.
    """
    value = os.environ.get(f"{ENV_PREFIX}CONFIG")
    if value:
        return Path(value).expanduser()
    candidate = Path.cwd() / "reference.config.json"
    return candidate if candidate.is_file() else None


def _as_non_negative_int(value: Any, label: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} must be an integer, got {value!r}") from exc
    if number < 0:
        raise ConfigError(f"{label} must not be negative")
    return number


def _quota_from(mapping: Mapping[str, Any], quota: dict[str, int], label: str) -> None:
    if not isinstance(mapping, Mapping):
        raise ConfigError(f"{label} must be an object of quota values")
    for key, value in mapping.items():
        if key not in DEFAULT_STORAGE_QUOTA:
            raise ConfigError(f"unknown storage quota key: {key!r}")
        quota[key] = _as_non_negative_int(value, f"{label}.{key}")


@dataclass(frozen=True)
class ReferenceConfig:
    """The effective configuration resolved from all sources."""

    root: Path
    mode: str = DEFAULT_MODE
    storage_quota: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_STORAGE_QUOTA))
    ocr_engine: str = DEFAULT_OCR_ENGINE
    ocr_language: str = DEFAULT_OCR_LANGUAGE
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    allowed_profile: str = DEFAULT_ALLOWED_PROFILE

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "mode": self.mode,
            "storage_quota": dict(self.storage_quota),
            "ocr_engine": self.ocr_engine,
            "ocr_language": self.ocr_language,
            "embedding_model": self.embedding_model,
            "allowed_profile": self.allowed_profile,
        }


def load_config(
    *,
    root: str | os.PathLike[str] | None = None,
    config_file: str | os.PathLike[str] | None = None,
    mode: str | None = None,
    ocr_engine: str | None = None,
    ocr_language: str | None = None,
    embedding_model: str | None = None,
    allowed_profile: str | None = None,
    storage_quota: Mapping[str, Any] | None = None,
) -> ReferenceConfig:
    """Resolve the effective configuration.

    ``config_file=None`` means "use the default lookup"; an explicitly named
    file that does not exist is an error rather than a silent fallback.
    Keyword arguments are the highest-precedence source.
    """
    file_values: dict[str, Any] = {}
    path = Path(config_file).expanduser() if config_file is not None else default_config_file()
    if path is not None:
        if not path.is_file():
            raise ConfigError(f"config file not found: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ConfigError(f"cannot read config file {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ConfigError("config file must contain a JSON object")
        unknown = set(payload) - {
            "root",
            "mode",
            "storage_quota",
            "ocr_engine",
            "ocr_language",
            "embedding_model",
            "allowed_profile",
        }
        if unknown:
            raise ConfigError(f"unknown config keys: {', '.join(sorted(unknown))}")
        file_values = payload

    quota: dict[str, int] = dict(DEFAULT_STORAGE_QUOTA)
    _quota_from(file_values.get("storage_quota") or {}, quota, "storage_quota")
    env_quota: dict[str, int] = {}
    for suffix, key in _QUOTA_ENV.items():
        value = os.environ.get(f"{ENV_PREFIX}{suffix}")
        if value is not None:
            env_quota[key] = _as_non_negative_int(value, f"{ENV_PREFIX}{suffix}")
    quota.update(env_quota)
    if storage_quota is not None:
        _quota_from(storage_quota, quota, "storage_quota")

    env = os.environ
    resolved_root = root
    if resolved_root is None:
        resolved_root = env.get(f"{ENV_PREFIX}ROOT") or file_values.get("root")
    if resolved_root is None:
        resolved_root = default_root()

    return ReferenceConfig(
        root=Path(resolved_root).expanduser(),
        mode=mode or env.get(f"{ENV_PREFIX}MODE") or file_values.get("mode") or DEFAULT_MODE,
        storage_quota=quota,
        ocr_engine=(
            ocr_engine
            or env.get(f"{ENV_PREFIX}OCR_ENGINE")
            or file_values.get("ocr_engine")
            or DEFAULT_OCR_ENGINE
        ),
        ocr_language=(
            ocr_language
            or env.get(f"{ENV_PREFIX}OCR_LANGUAGE")
            or file_values.get("ocr_language")
            or DEFAULT_OCR_LANGUAGE
        ),
        embedding_model=(
            embedding_model
            or env.get(f"{ENV_PREFIX}EMBEDDING_MODEL")
            or file_values.get("embedding_model")
            or DEFAULT_EMBEDDING_MODEL
        ),
        allowed_profile=(
            allowed_profile
            or env.get(f"{ENV_PREFIX}ALLOWED_PROFILE")
            or file_values.get("allowed_profile")
            or DEFAULT_ALLOWED_PROFILE
        ),
    )
