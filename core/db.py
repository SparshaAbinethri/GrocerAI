"""
core/db.py
PostgreSQL database layer for GrocerAI.
Replaces FAISS JSON for user preferences in production.
Uses psycopg2 with connection pooling.
Falls back gracefully to JSON file if DB is not configured.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_pool = None
_DB_AVAILABLE = False


def _init_pool() -> None:
    global _pool, _DB_AVAILABLE
    db_url = os.getenv("DATABASE_URL", "")
    if not db_url:
        logger.info("DATABASE_URL not set — using JSON file fallback for preferences")
        return
    try:
        from psycopg2 import pool as pg_pool
        _pool = pg_pool.SimpleConnectionPool(1, 10, dsn=db_url)
        _DB_AVAILABLE = True
        _run_migrations()
        logger.info("PostgreSQL connection pool initialised")
    except Exception as e:
        logger.warning("PostgreSQL unavailable (%s) — falling back to JSON file", e)


def _get_conn():
    if _pool is None:
        _init_pool()
    if not _DB_AVAILABLE:
        return None
    return _pool.getconn()


def _release_conn(conn) -> None:
    if _pool and conn:
        _pool.putconn(conn)


def _run_migrations() -> None:
    """Create tables if they don't exist."""
    conn = _get_conn()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id     VARCHAR(64) PRIMARY KEY,
                    prefs       JSONB       NOT NULL DEFAULT '{}',
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    session_id   VARCHAR(64) PRIMARY KEY,
                    user_id      VARCHAR(64) NOT NULL,
                    grocery_list JSONB,
                    cart_items   JSONB,
                    cart_total   FLOAT,
                    errors       JSONB,
                    duration_ms  INTEGER,
                    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS api_costs (
                    id          SERIAL PRIMARY KEY,
                    session_id  VARCHAR(64),
                    service     VARCHAR(32),  -- 'openai' | 'kroger'
                    endpoint    VARCHAR(128),
                    tokens_used INTEGER,
                    cost_usd    FLOAT,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)
            conn.commit()
        logger.info("DB migrations applied")
    except Exception as e:
        logger.error("Migration failed: %s", e)
        conn.rollback()
    finally:
        _release_conn(conn)


# ── Preferences ───────────────────────────────────────────────────────────────

def get_preferences(user_id: str) -> dict[str, Any]:
    conn = _get_conn()
    if not conn:
        return {}
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT prefs FROM user_preferences WHERE user_id = %s",
                (user_id,)
            )
            row = cur.fetchone()
            return row[0] if row else {}
    except Exception as e:
        logger.error("get_preferences failed: %s", e)
        return {}
    finally:
        _release_conn(conn)


def save_preferences(user_id: str, prefs: dict[str, Any]) -> bool:
    conn = _get_conn()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO user_preferences (user_id, prefs, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (user_id) DO UPDATE
                SET prefs = EXCLUDED.prefs, updated_at = NOW()
            """, (user_id, json.dumps(prefs)))
            conn.commit()
        return True
    except Exception as e:
        logger.error("save_preferences failed: %s", e)
        conn.rollback()
        return False
    finally:
        _release_conn(conn)


# ── Pipeline run logging ──────────────────────────────────────────────────────

def log_pipeline_run(state: dict, duration_ms: int) -> None:
    conn = _get_conn()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO pipeline_runs
                    (session_id, user_id, grocery_list, cart_items, cart_total, errors, duration_ms)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (session_id) DO NOTHING
            """, (
                state.get("session_id"),
                state.get("user_id"),
                json.dumps(state.get("grocery_list", [])),
                json.dumps(state.get("cart_items", [])),
                state.get("cart_total", 0.0),
                json.dumps(state.get("errors", [])),
                duration_ms,
            ))
            conn.commit()
    except Exception as e:
        logger.error("log_pipeline_run failed: %s", e)
        conn.rollback()
    finally:
        _release_conn(conn)


# ── API cost tracking ─────────────────────────────────────────────────────────

def log_api_cost(
    session_id: str,
    service: str,
    endpoint: str,
    tokens_used: int = 0,
    cost_usd: float = 0.0,
) -> None:
    conn = _get_conn()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO api_costs (session_id, service, endpoint, tokens_used, cost_usd)
                VALUES (%s, %s, %s, %s, %s)
            """, (session_id, service, endpoint, tokens_used, cost_usd))
            conn.commit()
    except Exception as e:
        logger.error("log_api_cost failed: %s", e)
        conn.rollback()
    finally:
        _release_conn(conn)


def is_available() -> bool:
    return _DB_AVAILABLE
