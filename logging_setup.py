from __future__ import annotations

import logging.config
from typing import Any, Mapping


def configure_logging(cfg: Mapping[str, Any]) -> None:
    log_cfg = dict(cfg.get("logging", {}) or {})
    level = str(log_cfg.get("level", "INFO")).upper()
    fmt = str(log_cfg.get("format", "%(asctime)s | %(levelname)s | %(name)s | %(message)s"))

    config_dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {"format": fmt},
        },
        "handlers": {
            "console": {"class": "logging.StreamHandler", "formatter": "default"},
        },
        "root": {"level": level, "handlers": ["console"]},
    }
    logging.config.dictConfig(config_dict)
