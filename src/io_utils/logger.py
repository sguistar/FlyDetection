from __future__ import annotations

import json
import logging
from pathlib import Path


def setup_logger(log_path: str | Path) -> logging.Logger:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("mot_project")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def log_kv(logger: logging.Logger, level: int, message: str, **kwargs) -> None:
    if kwargs:
        logger.log(level, "%s | %s", message, json.dumps(kwargs, sort_keys=True, ensure_ascii=False))
    else:
        logger.log(level, message)
