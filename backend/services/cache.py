"""
Redis client, shared by session_store.py (chat sessions) and
routes/forecast.py (forecast response caching).

Uses redis-py's standard client against REDIS_URL rather than Upstash's
REST SDK (as the plan originally suggested) -- redis-py works
identically against local Redis now and Upstash's TCP+TLS endpoint at
deployment (rediss://...), so this builds and tests fully today without
needing new Upstash credentials, and avoids a vendor-specific SDK.

All cache operations degrade gracefully if Redis is unreachable --
caching is a performance optimization here, not a correctness
requirement, so a Redis outage should never break a request.
"""

import os

import redis

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=2)


def cache_get(key: str):
    try:
        return redis_client.get(key)
    except redis.RedisError as exc:
        print(f"[cache] GET failed for {key}: {exc}. Continuing without cache.")
        return None


def cache_set(key: str, value: str, ttl_seconds: int) -> None:
    try:
        redis_client.setex(key, ttl_seconds, value)
    except redis.RedisError as exc:
        print(f"[cache] SET failed for {key}: {exc}. Continuing without cache.")
