"""YAML config loader with `_base_` inheritance and dotted-key overrides.

A config may declare `_base_: path/to/parent.yaml` (resolved relative to the
child file's directory). The parent is loaded first and then deep-merged with
the child — child keys win on conflict. This gives us a clean local / hipergator
override pattern without duplicating shared fields.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_with_base(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base_rel = cfg.pop("_base_", None)
    if base_rel:
        base_path = (path.parent / base_rel).resolve()
        base_cfg = _load_with_base(base_path)
        cfg = _deep_merge(base_cfg, cfg)
    return cfg


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Apply CLI-style `key.sub=value` overrides."""
    if not overrides:
        return cfg
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item!r}")
        key, raw = item.split("=", 1)
        try:
            value = yaml.safe_load(raw)
        except Exception:
            value = raw
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    cfg = _load_with_base(Path(path).resolve())
    return apply_overrides(cfg, overrides)


def save_config(cfg: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for p in dotted.split("."):
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node
