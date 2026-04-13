from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


class ConfigError(RuntimeError):
    """Chyba konfigurácie (neplatný YAML, chýbajúce kľúče, env premennej, atď.)."""


def _deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Rekurzívne spojí (merge) dve mapy; override vyhráva. (bez zmeny pôvodných dictov)"""
    out: Dict[str, Any] = dict(base)
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_merge(dict(out[k]), v)
        else:
            out[k] = v
    return out


def _expand_env(obj: Any) -> Any:
    """Rekurzívne expanduje ${VAR} v string hodnotách.

    Poznámka (nešpecifikované v YAML štandarde):
    - Toto je aplikačná konvencia, nie priamo vlastnosť YAML špecifikácie.
    """
    if isinstance(obj, str):
        def repl(m: re.Match) -> str:
            var = m.group(1)
            if var not in os.environ:
                raise ConfigError(f"Chýba env premenná: {var}")
            return os.environ[var]
        return _ENV_PATTERN.sub(repl, obj)

    if isinstance(obj, list):
        return [_expand_env(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    return obj


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Načíta YAML config vrátane base_config vrstvenia (ak je uvedené).

    Pravidlá:
    - base_config môže byť string alebo list[string]
    - relatívne base_config cesty sa vyhodnocujú voči adresáru aktuálneho configu
    - po načítaní a mergovaní sa aplikuje substitúcia env premenných
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config neexistuje: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"Neplatný YAML v {path}: {e}") from e

    if not isinstance(data, dict):
        raise ConfigError(f"Top-level YAML musí byť map/dict: {path}")

    base_cfg = data.pop("base_config", None)
    if base_cfg:
        if isinstance(base_cfg, (str, Path)):
            base_paths = [Path(base_cfg)]
        elif isinstance(base_cfg, list) and all(isinstance(x, (str, Path)) for x in base_cfg):
            base_paths = [Path(x) for x in base_cfg]
        else:
            raise ConfigError("base_config musí byť string alebo list[string]")

        merged: Dict[str, Any] = {}
        for bp in base_paths:
            if not bp.is_absolute():
                bp = (path.parent / bp).resolve()
            merged = _deep_merge(merged, load_yaml(bp))
        data = _deep_merge(merged, data)

    data = _expand_env(data)
    return data


def _device_to_arg(device_value: Any) -> Any:
    """Pre mapovanie runtime.device z YAML do train()/generate() argumentu.

    - "auto" alebo None => nechám None (pôvodná logika vyberie cuda/cpu)
    - inak vrátim string (napr. "cpu", "cuda", "cuda:0")
    """
    if device_value is None:
        return None
    s = str(device_value).strip()
    if s.lower() == "auto":
        return None
    return s


def build_train_kwargs(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Preloží YAML config do kwargs pre pôvodnú funkciu train(...).

    Poznámka:
    - podpis train() je považovaný za "hook" a v tomto refaktore ho nemením.
    """
    if "data" not in cfg or "train" not in cfg:
        raise ConfigError("Config musí obsahovať sekcie 'data' a 'train'.")

    d = cfg["data"]
    t = cfg["train"]
    r = cfg.get("runtime", {}) or {}
    out = cfg.get("output", {}) or {}

    # Povinné poli pre tréning v našom referenčnom projekte:
    # (ak je váš pôvodný skript iný, je to nešpecifikované a upravíte mapovanie)
    for key_path in [
        ("data", "train", "healthy_dir"),
        ("data", "train", "mask_dir"),
    ]:
        cur = cfg
        for k in key_path:
            cur = cur.get(k) if isinstance(cur, dict) else None
        if cur in (None, ""):
            raise ConfigError(f"Chýba povinná hodnota v configu: {'.'.join(key_path)}")

    return dict(
        train_healthy_dir=d["train"].get("healthy_dir"),
        train_mask_dir=d["train"].get("mask_dir"),
        train_corrupted_dir=d["train"].get("corrupted_dir"),
        val_healthy_dir=(d.get("val") or {}).get("healthy_dir"),
        val_mask_dir=(d.get("val") or {}).get("mask_dir"),
        val_corrupted_dir=(d.get("val") or {}).get("corrupted_dir"),
        test_healthy_dir=(d.get("test") or {}).get("healthy_dir"),
        test_mask_dir=(d.get("test") or {}).get("mask_dir"),
        test_corrupted_dir=(d.get("test") or {}).get("corrupted_dir"),
        epochs=int(t.get("epochs", 20)),
        batch_size=int(d.get("batch_size", 8)),
        eval_batch_size=d.get("eval_batch_size"),
        crop_size=int(d.get("crop_size", 256)),
        save_dir=str(out.get("save_dir", "runs/pgan_run")),
        best_metric_name=str(t.get("best_metric_name", "lpips")),
        device=_device_to_arg(r.get("device", "auto")),
        num_workers=int(r.get("num_workers", 0)),
        lr_g=float(t.get("lr_g", 1e-4)),
        lr_d=float(t.get("lr_d", 1e-4)),
        lambda_hole=float(t.get("lambda_hole", 20.0)),
        lambda_bg=float(t.get("lambda_bg", 5.0)),
        lambda_boundary=float(t.get("lambda_boundary", 15.0)),
        lambda_fm=float(t.get("lambda_fm", 5.0)),
        r1_gamma=float(t.get("r1_gamma", 5.0)),
        r1_every=int(t.get("r1_every", 8)),
        sample_count=int(t.get("sample_count", 8)),
    )


def build_generate_kwargs(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Preloží YAML config do kwargs pre pôvodnú funkciu generate(...)."""
    if "generate" not in cfg or "data" not in cfg:
        raise ConfigError("Config musí obsahovať sekcie 'generate' a 'data'.")

    g = cfg["generate"]
    d = cfg["data"]
    r = cfg.get("runtime", {}) or {}

    for key in ["healthy_dir", "mask_dir", "model_path", "output_dir"]:
        if g.get(key) in (None, ""):
            raise ConfigError(f"Chýba povinná hodnota v configu: generate.{key}")

    return dict(
        healthy_dir=g.get("healthy_dir"),
        mask_dir=g.get("mask_dir"),
        corrupted_dir=g.get("corrupted_dir"),
        model_path=g.get("model_path", "generator.pt"),
        output_dir=g.get("output_dir", "generated"),
        crop_size=int(d.get("crop_size", 256)),
        device=_device_to_arg(r.get("device", "auto")),
    )
