"""
core/redis_client.py
Redis client for GrocerAI.
Used for:
  - Kroger OAuth token storage (access + refresh)
  - Session data caching
  - Rate limiting counters
Falls back gracefully to in-memory dict if Redis is not configured.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

_redis = None
_REDIS_AVAILABLE = False

# In-memory fallback store
_mem_store: dict[str, tuple[Any, float]] = {}  # key → (value, expires_at)


def _init_redis() -> None:
    global _redis, _REDIS_AVAILABLE
    redis_url = os.getenv("REDIS_URL", "")
    if not redis_url:
        logger.info("REDIS_URL not set — using in-memory fallback for sessions")
        return
    try:
        import redis
        _redis = redis.from_url(redis_url, decode_responses=True)
        _redis.ping()
        _REDIS_AVAILABLE = True
        logger.info("Redis connection established")
    except Exception as e:
        logger.warning("Redis unavailable (%s) — falling back to in-memory store", e)


def _get_client():
    global _redis
    if _redis is None and not _REDIS_AVAILABLE:
        _init_redis()
    return _redis if _REDIS_AVAILABLE else None


# ── Core get/set/delete ───────────────────────────────────────────────────────

def set(key: str, value: Any, ttl_seconds: int = 3600) -> None:
    """Store a value with optional TTL."""
    serialised = json.dumps(value)
    client = _get_client()
    if client:
        try:
            client.setex(key, ttl_seconds, serialised)
            return
        except Exception as e:
            logger.error("Redis set failed: %s", e)
    # Fallback
    _mem_store[key] = (value, time.time() + ttl_seconds)


def get(key: str) -> Any | None:
    """Retrieve a value, returns None if missing or expired."""
    client = _get_client()
    if client:
        try:
            raw = client.get(key)
            return json.loads(raw) if raw else None
        except Exception as e:
            logger.error("Redis get failed: %s", e)
    # Fallback
    entry = _mem_store.get(key)
    if entry:
        value, expires_at = entry
        if time.time() < expires_at:
            return value
        del _mem_store[key]
    return None


def delete(key: str) -> None:
    client = _get_client()
    if client:
        try:
            client.delete(key)
            return
        except Exception as e:
            logger.error("Redis delete failed: %s", e)
    _mem_store.pop(key, None)


# ── Kroger OAuth token helpers ────────────────────────────────────────────────

def store_kroger_tokens(
    user_id: str,
    access_token: str,
    refresh_token: str | None,
    expires_in: int = 1800,
) -> None:
    """Store Kroger OAuth tokens for a user."""
    set(f"kroger:access:{user_id}", access_token, ttl_seconds=expires_in - 60)
    if refresh_token:
        set(f"kroger:refresh:{user_id}", refresh_token, ttl_seconds=60 * 60 * 24 * 30)


def get_kroger_access_token(user_id: str) -> str | None:
    return get(f"kroger:access:{user_id}")


def get_kroger_refresh_token(user_id: str) -> str | None:
    return get(f"kroger:refresh:{user_id}")


def clear_kroger_tokens(user_id: str) -> None:
    delete(f"kroger:access:{user_id}")
    delete(f"kroger:refresh:{user_id}")


# ── Rate limiting ─────────────────────────────────────────────────────────────

def check_rate_limit(key: str, max_requests: int, window_seconds: int) -> tuple[bool, int]:
    """
    Sliding window rate limiter.
    Returns (allowed: bool, remaining: int).
    """
    client = _get_client()
    full_key = f"ratelimit:{key}"

    if client:
        try:
            pipe = client.pipeline()
            now = int(time.time())
            window_start = now - window_seconds
            pipe.zremrangebyscore(full_key, 0, window_start)
            pipe.zadd(full_key, {str(now): now})
            pipe.zcard(full_key)
            pipe.expire(full_key, window_seconds)
            results = pipe.execute()
            count = results[2]
            allowed = count <= max_requests
            remaining = max(0, max_requests - count)
            return allowed, remaining
        except Exception as e:
            logger.error("Rate limit check failed: %s", e)
            return True, max_requests  # fail open

    # Fallback: simple in-memory counter
    count_key = f"{full_key}:count"
    entry = _mem_store.get(count_key)
    if entry:
        count, expires_at = entry
        if time.time() < expires_at:
            count += 1
            _mem_store[count_key] = (count, expires_at)
        else:
            count = 1
            _mem_store[count_key] = (1, time.time() + window_seconds)
    else:
        count = 1
        _mem_store[count_key] = (1, time.time() + window_seconds)

    return count <= max_requests, max(0, max_requests - count)


def is_available() -> bool:
    return _REDIS_AVAILABLE
