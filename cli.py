from __future__ import annotations

import logging
from pathlib import Path

import click

from config import ConfigError, load_yaml, build_train_kwargs, build_generate_kwargs
from logging_setup import configure_logging


logger = logging.getLogger(__name__)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """PGAN/ML nástroj: tréning a generovanie cez YAML config."""
    pass


@cli.command("train")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Cesta ku YAML konfigurácii pre tréning (napr. configs/config_train.yaml).",
)
def train_cmd(config_path: Path) -> None:
    """Spustí tréning podľa YAML konfigurácie."""
    try:
        cfg = load_yaml(config_path)
        configure_logging(cfg)

        # Lazy import: aby `pgan generate --help` nevyžadovalo všetky tréningové závislosti.
        from train import train  # noqa: WPS433 (lokálny import je zámerný)

        kwargs = build_train_kwargs(cfg)
        logger.info("Spúšťam tréning: save_dir=%s", kwargs.get("save_dir"))
        train(**kwargs)

    except ConfigError as e:
        raise click.ClickException(str(e)) from e


@cli.command("generate")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Cesta ku YAML konfigurácii pre generovanie (napr. configs/config_generate.yaml).",
)
def generate_cmd(config_path: Path) -> None:
    """Spustí generovanie podľa YAML konfigurácie."""
    try:
        cfg = load_yaml(config_path)
        configure_logging(cfg)

        from generate import generate

        kwargs = build_generate_kwargs(cfg)
        logger.info("Spúšťam generovanie: model=%s out=%s", kwargs.get("model_path"), kwargs.get("output_dir"))
        generate(**kwargs)

    except ConfigError as e:
        raise click.ClickException(str(e)) from e


if __name__ == "__main__":
    cli()
