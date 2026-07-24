"""响应缓存 —— LRU + TTL，仅缓存非流式幂等请求"""
import hashlib, json, time, threading, logging

logger = logging.getLogger("SmartProxy.Cache")


class ResponseCache:
    def __init__(self, max_entries=128, ttl_seconds=300):
        self._store: dict[str, tuple[float, bytes, str]] = {}  # key → (expires_at, body, content_type)
        self._max = max_entries
        self._ttl = ttl_seconds
        self._lock = threading.Lock()

    def _key(self, model: str, messages: list) -> str:
        """基于 model + messages 的 SHA256 摘要。"""
        raw = json.dumps({"model": model, "messages": messages}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, model: str, messages: list) -> tuple[bytes, str] | None:
        """命中返回 (body_bytes, content_type)，否则 None。"""
        key = self._key(model, messages)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires, body, ct = entry
            if time.time() > expires:
                del self._store[key]
                return None
            logger.info(f"Cache hit for {key[:12]}...")
            return body, ct

    def put(self, model: str, messages: list, body: bytes, content_type: str = "application/json"):
        key = self._key(model, messages)
        with self._lock:
            if len(self._store) >= self._max:
                # 淘汰最老的 entry
                oldest = min(self._store.items(), key=lambda x: x[1][0])
                del self._store[oldest[0]]
            self._store[key] = (time.time() + self._ttl, body, content_type)

    def stats(self) -> dict:
        with self._lock:
            now = time.time()
            active = sum(1 for _, (e, _, _) in self._store.items() if e > now)
            return {"entries": len(self._store), "active": active, "max": self._max}


# 全局单例
cache = ResponseCache()
