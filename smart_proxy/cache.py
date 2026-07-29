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
        """基于 model + 最后一条 user message 文本的 SHA256 摘要。
        v4.5: 不再序列化完整 messages 数组（长对话可达数百 KB），
        只用最后 user text 哈希——同 session 内相同 prompt 重复发送概率极低。
        """
        text = ""
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            break
                elif isinstance(content, str):
                    text = content
                break
        raw = json.dumps({"model": model, "text": text[:200]}, sort_keys=True, ensure_ascii=False)
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
