"""
core/logging_config.py
Structured JSON logging for GrocerAI.
In production: outputs JSON to stdout (picked up by Railway/Docker log aggregators).
In development: outputs human-readable coloured logs.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any


class JSONFormatter(logging.Formatter):
    """
    Formats log records as single-line JSON objects.
    Compatible with Datadog, Railway, CloudWatch, GCP Logging.
    """

    def format(self, record: logging.LogRecord) -> str:
        log: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log["exception"] = self.formatException(record.exc_info)

        # Add any extra fields passed via extra={}
        for key, value in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "taskName",
            ):
                log[key] = value

        return json.dumps(log, default=str)


class DevFormatter(logging.Formatter):
    """Human-readable formatter for local development."""
    COLOURS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        colour = self.COLOURS.get(record.levelname, "")
        ts = datetime.now().strftime("%H:%M:%S")
        msg = record.getMessage()
        base = f"{colour}[{ts}] {record.levelname:8s}{self.RESET} {record.name}: {msg}"
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def setup_logging() -> None:
    """
    Configure root logger based on environment.
    Call once at app startup.
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    env = os.getenv("APP_ENV", "development")
    numeric_level = getattr(logging, log_level, logging.INFO)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Remove existing handlers
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)

    if env == "production":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(DevFormatter())

    root.addHandler(handler)

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "streamlit"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger("grocerai").setLevel(numeric_level)
    logging.info("Logging configured | level=%s env=%s", log_level, env)
