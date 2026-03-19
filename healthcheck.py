"""
healthcheck.py
Lightweight HTTP health check server for GrocerAI.
Run alongside Streamlit: python healthcheck.py
Exposes:
  GET /health  — liveness check (returns 200 OK)
  GET /ready   — readiness check (checks DB, Redis, OpenAI key)
  GET /metrics — Prometheus metrics (if prometheus_client installed)
  GET /stats   — JSON stats summary

Railway / Docker healthcheck: GET http://localhost:8502/health
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.logging_config import setup_logging
from core.metrics import get_stats

setup_logging()
logger = logging.getLogger(__name__)

HEALTH_PORT = int(os.getenv("HEALTH_PORT", "8502"))


class HealthHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress access logs

    def _send_json(self, status: int, body: dict) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "service": "grocerai"})

        elif self.path == "/ready":
            checks = {}
            overall = True

            # OpenAI key
            checks["openai_key"] = bool(os.getenv("OPENAI_API_KEY"))
            if not checks["openai_key"]:
                overall = False

            # Kroger credentials
            checks["kroger_credentials"] = bool(
                os.getenv("KROGER_CLIENT_ID") and os.getenv("KROGER_CLIENT_SECRET")
            )

            # PostgreSQL
            try:
                from core.db import is_available
                checks["postgres"] = is_available()
            except Exception:
                checks["postgres"] = False

            # Redis
            try:
                from core.redis_client import is_available as redis_ok
                checks["redis"] = redis_ok()
            except Exception:
                checks["redis"] = False

            status = 200 if overall else 503
            self._send_json(status, {
                "status": "ready" if overall else "not_ready",
                "checks": checks,
            })

        elif self.path == "/metrics":
            try:
                from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
                payload = generate_latest()
                self.send_response(200)
                self.send_header("Content-Type", CONTENT_TYPE_LATEST)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except ImportError:
                self._send_json(404, {"error": "prometheus_client not installed"})

        elif self.path == "/stats":
            self._send_json(200, get_stats())

        else:
            self._send_json(404, {"error": "not found"})


def start_health_server(port: int = HEALTH_PORT, daemon: bool = True) -> threading.Thread:
    """Start health server in a background thread."""
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=daemon)
    thread.start()
    logger.info("Health server running on port %d", port)
    return thread


if __name__ == "__main__":
    logger.info("Starting GrocerAI health server on port %d", HEALTH_PORT)
    server = HTTPServer(("0.0.0.0", HEALTH_PORT), HealthHandler)
    server.serve_forever()
