"""Shared helpers: config loading, paths, logging and a stage-timing decorator."""
from __future__ import annotations

import functools
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_path(cfg: dict, key: str, *parts: str, create: bool = True) -> Path:
    """Resolve cfg['paths'][key] (relative to repo root) and optionally create it."""
    p = ROOT / cfg["paths"][key]
    p = p.joinpath(*parts)
    target = p if not p.suffix else p.parent
    if create:
        target.mkdir(parents=True, exist_ok=True)
    return p


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def logged_stage(stage_name: str):
    """Decorator: time a pipeline stage and append the outcome to reports/run_log.jsonl.

    The log is later turned into the 'workflow run' graphic and is useful evidence of
    which stages ran, in which order, and how long each took.
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(cfg: dict, *args, **kwargs):
            log = get_logger(stage_name)
            started = time.time()
            start_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
            log.info("START %s", stage_name)
            status, result = "success", None
            try:
                result = fn(cfg, *args, **kwargs)
                return result
            except Exception:
                status = "failed"
                raise
            finally:
                seconds = round(time.time() - started, 2)
                log.info("%s %s in %.2fs", "DONE" if status == "success" else "FAILED", stage_name, seconds)
                log_path = get_path(cfg, "reports_dir", "run_log.jsonl")
                label = stage_name
                if args and isinstance(args[0], str):
                    label = f"{stage_name}[{args[0]}]"
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"stage": label, "status": status, "started_utc": start_iso, "seconds": seconds}) + "\n")

        return wrapper

    return decorator


def reset_run_log(cfg: dict) -> None:
    log_path = get_path(cfg, "reports_dir", "run_log.jsonl")
    if log_path.exists():
        log_path.unlink()
