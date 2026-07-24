"""Pydantic v2 配置 + YAML 加载 + 热重载"""
from __future__ import annotations
import os, threading, logging
from pathlib import Path
from typing import Optional
import yaml
from pydantic import BaseModel, Field, model_validator
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(SCRIPT_DIR / ".env")
logger = logging.getLogger("SmartProxy.Config")

# ─── Pydantic models ─────────────────────────────────────────────

class BackendConf(BaseModel):
    url: str
    key_env: str
    models: dict[str, str]
    key: Optional[str] = None
    @model_validator(mode="after")
    def _resolve_key(self):
        self.key = os.getenv(self.key_env)
        return self


class RoutingConf(BaseModel):
    flash_first: bool = True
    fallback_tier_map: dict[str, str] = Field(default_factory=dict)
    session_sticky: dict = Field(default_factory=lambda: {"enabled": False, "ttl_seconds": 600})
    token_length: dict = Field(default_factory=lambda: {"enabled": False})


class CircuitConf(BaseModel):
    block_seconds_quota: int = 3600
    block_seconds_server: int = 300
    block_seconds_network: int = 60
    block_seconds_permanent: int = 18000   # 配额耗尽等永久错误：5h
    backoff_max_seconds: int = 7200
    backoff_multiplier: int = 2
    probe_interval: int = 30
    probe_timeout: int = 3
    min_block_seconds: int = 60

class LatencyConf(BaseModel):
    window: int = 8
    slow_threshold_ms: int = 8000
    degraded_sticky_s: int = 30

class TimeoutsConf(BaseModel):
    total: float = 120.0
    connect: float = 10.0
    read: float = 120.0
    pool: float = 10.0

class PoolConf(BaseModel):
    max_connections: int = 100
    max_keepalive_connections: int = 20
    keepalive_expiry: float = 60.0

class CacheConf(BaseModel):
    enabled: bool = True
    max_entries: int = 128
    ttl_seconds: int = 300

class SecurityConf(BaseModel):
    auth_token_env: str = "SP_AUTH_TOKEN"
    max_body_bytes: int = 33554432
    error_body_truncate: int = 500

class ServerConf(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000

class Settings(BaseModel):
    server: ServerConf
    backends: dict[str, BackendConf]
    routing: RoutingConf
    circuit: CircuitConf = Field(default_factory=CircuitConf)
    latency: LatencyConf = Field(default_factory=LatencyConf)
    timeouts: TimeoutsConf = Field(default_factory=TimeoutsConf)
    pool: PoolConf = Field(default_factory=PoolConf)
    cache: CacheConf = Field(default_factory=CacheConf)
    security: SecurityConf = Field(default_factory=SecurityConf)
    http_proxy: Optional[str] = None
    state_file: str = "proxy_state.json"
    usage_stats_file: str = "usage_stats.json"
    log_level: str = "INFO"

# ─── Load + hot-reload ───────────────────────────────────────────

_cached: list = [None]
_mtime: list = [None]
_lock = threading.Lock()

def load_settings() -> Settings:
    cfg_path = SCRIPT_DIR / "proxy_config.yaml"
    if not cfg_path.exists():
        logger.warning("proxy_config.yaml not found, using defaults")
        return Settings(server={"host": "127.0.0.1", "port": 8000},
                        backends={},
                        routing=RoutingConf())
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    routing_conf = RoutingConf()
    if isinstance(raw.get("routing"), dict):
        routing_conf = RoutingConf(**raw["routing"])
    return Settings(
        server=raw.get("server", {"host": "127.0.0.1", "port": 8000}),
        backends=raw.get("backends", {}),
        routing=routing_conf,
        circuit=CircuitConf(**raw.get("circuit", {})) if raw.get("circuit") else CircuitConf(),
        latency=LatencyConf(**raw.get("latency", {})) if raw.get("latency") else LatencyConf(),
        timeouts=TimeoutsConf(**raw.get("timeouts", {})) if raw.get("timeouts") else TimeoutsConf(),
        pool=PoolConf(**raw.get("pool", {})) if raw.get("pool") else PoolConf(),
        cache=CacheConf(**raw.get("cache", {})) if raw.get("cache") else CacheConf(),
        security=SecurityConf(**raw.get("security", {})) if raw.get("security") else SecurityConf(),
        http_proxy=raw.get("http_proxy"),
        state_file=raw.get("state_file", "proxy_state.json"),
        usage_stats_file=raw.get("usage_stats_file", "usage_stats.json"),
        log_level=raw.get("log_level", "INFO"),
    )

def get_settings() -> Settings:
    cfg_path = SCRIPT_DIR / "proxy_config.yaml"
    if not cfg_path.exists():
        return load_settings()
    m = cfg_path.stat().st_mtime
    with _lock:
        if _cached[0] is None or m != _mtime[0]:
            _cached[0] = load_settings()
            _mtime[0] = m
        return _cached[0]
