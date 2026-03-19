"""
core/metrics.py
Application metrics for GrocerAI.
- Prometheus counters/histograms (if prometheus_client installed)
- OpenAI cost tracking
- Falls back to in-memory counters if Prometheus not available
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

_PROMETHEUS_AVAILABLE = False
_counters: dict[str, Any] = {}
_histograms: dict[str, Any] = {}
_mem_counters: dict[str, float] = defaultdict(float)

# OpenAI cost per 1K tokens (gpt-4o as of 2024)
OPENAI_COST_PER_1K_INPUT = 0.005
OPENAI_COST_PER_1K_OUTPUT = 0.015


def _init_prometheus() -> None:
    global _PROMETHEUS_AVAILABLE, _counters, _histograms
    try:
        from prometheus_client import Counter, Histogram, Gauge
        _counters["pipeline_runs"] = Counter(
            "grocerai_pipeline_runs_total",
            "Total pipeline runs",
            ["status"],
        )
        _counters["api_calls"] = Counter(
            "grocerai_api_calls_total",
            "Total external API calls",
            ["service", "endpoint", "status"],
        )
        _counters["openai_tokens"] = Counter(
            "grocerai_openai_tokens_total",
            "Total OpenAI tokens used",
            ["type"],  # input | output
        )
        _counters["openai_cost"] = Counter(
            "grocerai_openai_cost_usd_total",
            "Total OpenAI cost in USD",
        )
        _histograms["pipeline_duration"] = Histogram(
            "grocerai_pipeline_duration_seconds",
            "Pipeline execution duration",
            buckets=[1, 5, 10, 20, 30, 60, 120],
        )
        _histograms["agent_duration"] = Histogram(
            "grocerai_agent_duration_seconds",
            "Individual agent execution duration",
            ["agent"],
            buckets=[0.5, 1, 2, 5, 10, 30],
        )
        _PROMETHEUS_AVAILABLE = True
        logger.info("Prometheus metrics initialised")
    except ImportError:
        logger.info("prometheus_client not installed — using in-memory counters")


_init_prometheus()


# ── Pipeline metrics ──────────────────────────────────────────────────────────

def record_pipeline_run(success: bool, duration_seconds: float) -> None:
    status = "success" if success else "failure"
    if _PROMETHEUS_AVAILABLE:
        _counters["pipeline_runs"].labels(status=status).inc()
        _histograms["pipeline_duration"].observe(duration_seconds)
    _mem_counters[f"pipeline_runs.{status}"] += 1
    _mem_counters["pipeline_duration_total"] += duration_seconds


def record_agent_duration(agent: str, duration_seconds: float) -> None:
    if _PROMETHEUS_AVAILABLE:
        _histograms["agent_duration"].labels(agent=agent).observe(duration_seconds)
    _mem_counters[f"agent_duration.{agent}"] += duration_seconds


# ── API call metrics ──────────────────────────────────────────────────────────

def record_api_call(service: str, endpoint: str, success: bool) -> None:
    status = "success" if success else "failure"
    if _PROMETHEUS_AVAILABLE:
        _counters["api_calls"].labels(
            service=service, endpoint=endpoint, status=status
        ).inc()
    _mem_counters[f"api_calls.{service}.{status}"] += 1


# ── OpenAI cost tracking ──────────────────────────────────────────────────────

def record_openai_usage(
    input_tokens: int,
    output_tokens: int,
    session_id: str = "",
) -> float:
    """Record token usage and return cost in USD."""
    cost = (
        (input_tokens / 1000) * OPENAI_COST_PER_1K_INPUT
        + (output_tokens / 1000) * OPENAI_COST_PER_1K_OUTPUT
    )
    if _PROMETHEUS_AVAILABLE:
        _counters["openai_tokens"].labels(type="input").inc(input_tokens)
        _counters["openai_tokens"].labels(type="output").inc(output_tokens)
        _counters["openai_cost"].inc(cost)

    _mem_counters["openai_tokens_input"] += input_tokens
    _mem_counters["openai_tokens_output"] += output_tokens
    _mem_counters["openai_cost_usd"] += cost

    # Log to DB if available
    try:
        from core.db import log_api_cost
        log_api_cost(
            session_id=session_id,
            service="openai",
            endpoint="chat.completions",
            tokens_used=input_tokens + output_tokens,
            cost_usd=cost,
        )
    except Exception:
        pass

    logger.debug(
        "OpenAI usage | input=%d output=%d cost=$%.4f",
        input_tokens, output_tokens, cost,
    )
    return cost


# ── In-memory stats summary (for /health endpoint) ───────────────────────────

def get_stats() -> dict[str, Any]:
    return {
        "pipeline_runs": {
            "success": _mem_counters.get("pipeline_runs.success", 0),
            "failure": _mem_counters.get("pipeline_runs.failure", 0),
        },
        "openai": {
            "tokens_input": _mem_counters.get("openai_tokens_input", 0),
            "tokens_output": _mem_counters.get("openai_tokens_output", 0),
            "cost_usd": round(_mem_counters.get("openai_cost_usd", 0), 4),
        },
        "api_calls": {
            k.replace("api_calls.", ""): v
            for k, v in _mem_counters.items()
            if k.startswith("api_calls.")
        },
        "prometheus_enabled": _PROMETHEUS_AVAILABLE,
    }


# ── Context manager for timing ────────────────────────────────────────────────

class timer:
    """Usage: with timer("vision_agent", session_id) as t: ..."""
    def __init__(self, agent: str, session_id: str = ""):
        self.agent = agent
        self.session_id = session_id
        self.start = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *_):
        self.elapsed = time.time() - self.start
        record_agent_duration(self.agent, self.elapsed)
