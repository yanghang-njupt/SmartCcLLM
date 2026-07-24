"""JSON 状态管理 (熔断/过载/用量) + TTL"""
import os, json, time, threading, logging
from pathlib import Path

logger = logging.getLogger("SmartProxy.State")

_lock = threading.Lock()
_cache: dict = {}
_loaded = False

OVERLOAD_TTL = 600  # overloaded 标记超 10min 自动过期

def _path(state_file="proxy_state.json"):
    return Path(__file__).resolve().parent.parent / state_file

def _load(state_file="proxy_state.json"):
    global _cache, _loaded
    p = _path(state_file)
    if p.exists():
        try:
            _cache = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            _cache = {"_version": 2, "circuit": {}, "overload": {}}
    else:
        _cache = {"_version": 2, "circuit": {}, "overload": {}}
    _loaded = True

def _read(state_file="proxy_state.json"):
    if not _loaded:
        _load(state_file)
    return _cache

def _write(state, state_file="proxy_state.json"):
    p = _path(state_file)
    tmp = str(p) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)
    os.replace(tmp, str(p))

# ─── 通用 API ────────────────────────────────────────────────────

def is_blocked(name, state_file="proxy_state.json"):
    s = _read(state_file)
    # v2 format: {"circuit": {"kimi": {"blocked_until": ...}}}
    blocked = s.get("circuit", {}).get(name, {}).get("blocked_until", 0)
    # Legacy format: {"kimi_blocked_until": ...}
    if blocked <= 0:
        blocked = s.get(f"{name}_blocked_until", 0)
    return time.time() < blocked

def block(name, seconds=3600, state_file="proxy_state.json"):
    now = time.time()
    with _lock:
        s = dict(_read(state_file))
        s.setdefault("circuit", {})[name] = {
            "state": "OPEN",
            "blocked_until": now + seconds,
            "blocked_at": now,
            "fail_count": s.get("circuit", {}).get(name, {}).get("fail_count", 0) + 1,
        }
        _cache.update(s)
        _write(s, state_file)
    logger.warning(f"{name} circuit OPEN until {time.strftime('%H:%M:%S', time.localtime(now + seconds))}")

def unblock(name, state_file="proxy_state.json"):
    with _lock:
        s = dict(_read(state_file))
        if s.get("circuit", {}).pop(name, None) is not None:
            _cache.update(s)
            _write(s, state_file)
    logger.info(f"{name} circuit CLOSED")

def blocked_at(name, state_file="proxy_state.json"):
    s = _read(state_file)
    ba = s.get("circuit", {}).get(name, {}).get("blocked_at", 0)
    if ba <= 0:
        ba = s.get(f"{name}_blocked_at", 0)
    return ba

def fail_count(name, state_file="proxy_state.json"):
    return _read(state_file).get("circuit", {}).get(name, {}).get("fail_count", 0)

# ─── Overload (429) ──────────────────────────────────────────────

def is_overloaded(name, state_file="proxy_state.json"):
    s = _read(state_file)
    m = s.get("overload", {}).get(name)
    if not m:
        return False
    if time.time() - m.get("marked_at", 0) > OVERLOAD_TTL:
        clear_overloaded(name, state_file)
        return False
    return True

def mark_overloaded(name, state_file="proxy_state.json"):
    with _lock:
        s = dict(_read(state_file))
        s.setdefault("overload", {})[name] = {"marked_at": time.time()}
        _cache.update(s)
        _write(s, state_file)
    logger.warning(f"{name} overloaded (429)")

def clear_overloaded(name, state_file="proxy_state.json"):
    with _lock:
        s = dict(_read(state_file))
        if s.get("overload", {}).pop(name, None) is not None:
            _cache.update(s)
            _write(s, state_file)
    logger.info(f"{name} overload cleared")
