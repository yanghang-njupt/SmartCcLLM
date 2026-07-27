"""LatencyTracker + SimilarityBuffer + UpgradeStore(升级信号学习)"""
import json, os, re, time, threading, logging
from collections import deque, Counter
from pathlib import Path

logger = logging.getLogger("SmartProxy.Metrics")


class LatencyTracker:
    def __init__(self, window=8, slow_ms=8000, sticky_s=30):
        self._hist: dict = {}
        self._degraded_until: dict = {}
        self._window = window
        self._slow_ms = slow_ms
        self._sticky_s = sticky_s

    def configure(self, window=None, slow_ms=None, sticky_s=None):
        if window is not None:
            self._window = window
            for k in list(self._hist):
                if self._hist[k].maxlen < window:
                    self._hist[k] = deque(list(self._hist[k]), maxlen=window)
        if slow_ms is not None:
            self._slow_ms = slow_ms
        if sticky_s is not None:
            self._sticky_s = sticky_s

    def record(self, name, lat_ms, ok):
        self._hist.setdefault(name, deque(maxlen=self._window)).append(
            (time.time(), lat_ms, ok)
        )

    def avg(self, name):
        q = self._hist.get(name, [])
        ok_entries = [lat for _, lat, k in q if k]
        return sum(ok_entries) / len(ok_entries) if ok_entries else 0

    def error_rate(self, name):
        q = self._hist.get(name, [])
        return sum(1 for _, _, k in q if not k) / len(q) if q else 0

    def is_degraded(self, name, slow_ms=None, sticky_s=None):
        slow_ms = slow_ms if slow_ms is not None else self._slow_ms
        sticky_s = sticky_s if sticky_s is not None else self._sticky_s
        if time.time() < self._degraded_until.get(name, 0):
            return True
        q = self._hist.get(name)
        if not q or len(q) < 3:
            return False
        return self.avg(name) > slow_ms or self.error_rate(name) >= 0.5

    def mark_degraded(self, name, sticky_s=None):
        sticky_s = sticky_s if sticky_s is not None else self._sticky_s
        self._degraded_until[name] = time.time() + sticky_s

    def stats(self, name):
        return {
            "avg_latency_ms": round(self.avg(name), 1),
            "error_rate": round(self.error_rate(name), 2),
            "sample_count": len(self._hist.get(name, [])),
            "degraded": self.is_degraded(name),
        }


class SimilarityBuffer:
    """基于 Jaccard 相似度的历史结果缓冲。

    参考 RouteLLM Similarity-Weighted Elo：
    历史相似请求的结果对新请求有参考价值。
    """

    def __init__(self, maxlen=128):
        self._buffer: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def record_outcome(self, text: str, backend: str, tier: str,
                       success: bool, latency_ms: float, stop_reason: str = ""):
        """记录一次请求的结果，用于相似请求路由。"""
        with self._lock:
            self._buffer.append({
                "text": text[:500],  # 截断存储
                "backend": backend,
                "tier": tier,
                "success": success,
                "latency_ms": latency_ms,
                "stop_reason": stop_reason,
            })

    def _trigrams(self, text: str):
        t = text.lower()
        if len(t) < 3:
            return {t}
        return {t[i:i + 3] for i in range(len(t) - 2)}

    def _jaccard(self, a: set, b: set):
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def get_stats(self, text: str) -> dict | None:
        """获取相似历史请求的统计，用于路由决策。"""
        target_tri = self._trigrams(text)
        with self._lock:
            if len(self._buffer) < 5:
                return None
            flash_items = []
            pro_items = []
            for entry in self._buffer:
                sim = self._jaccard(target_tri, self._trigrams(entry["text"]))
                if sim < 0.1:
                    continue
                if entry["tier"] == "flash":
                    flash_items.append((sim, entry))
                elif entry["tier"] == "pro":
                    pro_items.append((sim, entry))

            if not flash_items:
                return None

            # 相似度加权统计
            flash_total_weight = sum(s[0] for s in flash_items)
            flash_failures = sum(s[0] for s in flash_items if not s[1]["success"])
            flash_failure_rate = flash_failures / flash_total_weight if flash_total_weight > 0 else 0

            flash_lat_sum = sum(s[0] * s[1]["latency_ms"] for s in flash_items)
            flash_avg_lat = flash_lat_sum / flash_total_weight if flash_total_weight > 0 else 0

            return {
                "flash_failure_rate": round(flash_failure_rate, 2),
                "flash_avg_latency": round(flash_avg_lat, 0),
                "flash_samples": len(flash_items),
            }

    def predict_latency(self, text: str, backend_key: str) -> float | None:
        """基于相似历史请求预测延迟（兼容旧 API）。"""
        target_tri = self._trigrams(text)
        with self._lock:
            scored = []
            for entry in self._buffer:
                be_tier = f"{entry['backend']}_{entry['tier']}"
                if be_tier != backend_key:
                    continue
                sim = self._jaccard(target_tri, self._trigrams(entry["text"]))
                if sim > 0:
                    scored.append((sim, entry["latency_ms"]))
            if not scored:
                return None
            scored.sort(key=lambda x: -x[0])
            top = scored[:8]
            total_weight = sum(s[0] for s in top)
            return sum(s[0] * s[1] for s in top) / total_weight if total_weight > 0 else None

    def stats(self):
        with self._lock:
            return {"buffer_size": len(self._buffer)}

    # ── 持久化 ──────────────────────────────────────────────

    HISTORY_FILE = "smart_proxy_history.json"

    def persist(self, filepath: str | None = None):
        """保存历史数据到 JSON 文件。"""
        if filepath is None:
            filepath = str(Path(__file__).resolve().parent.parent / "logs" / self.HISTORY_FILE)
        with self._lock:
            data = list(self._buffer)
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            Path(filepath).write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info(f"History saved ({len(data)} records) → {filepath}")
        except Exception as e:
            logger.error(f"Failed to persist history: {e}")

    def load(self, filepath: str | None = None):
        """从 JSON 文件加载历史数据。"""
        if filepath is None:
            filepath = str(Path(__file__).resolve().parent.parent / "logs" / self.HISTORY_FILE)
        try:
            data = json.loads(Path(filepath).read_text(encoding="utf-8"))
            if not isinstance(data, list):
                return False
            with self._lock:
                for entry in data:
                    self._buffer.append(entry)
            logger.info(f"History loaded ({len(data)} records) ← {filepath}")
            return True
        except FileNotFoundError:
            return False
        except json.JSONDecodeError as e:
            logger.warning(f"History file corrupt, ignoring: {e}")
            return False

    # ── 关键词自动萃取 ──────────────────────────────────────

    FLASH_SLOW_MS = 10000  # flash 超过此毫秒数即触发实时学习

    def extract_keywords(self, min_samples: int = 3) -> list[str]:
        """从历史数据中提取高频关键词，用于自动扩展 skip_flash 列表。"""
        with self._lock:
            if len(self._buffer) < min_samples:
                return []

            failed_texts = []
            for entry in self._buffer:
                tier = entry.get("tier", "")
                success = entry.get("success", True)
                text = (entry.get("text", "") or "").strip()
                if not text:
                    continue
                if tier == "flash" and not success:
                    failed_texts.append(text)

            if len(failed_texts) < min_samples:
                return []

        from collections import Counter
        bi_counter: Counter = Counter()
        tri_counter: Counter = Counter()
        quad_counter: Counter = Counter()

        for t in failed_texts:
            chars = list(t)
            for i in range(len(chars) - 1):
                gram = "".join(chars[i:i+2])
                if self._is_good_gram(gram):
                    bi_counter[gram] += 1
            for i in range(len(chars) - 2):
                gram = "".join(chars[i:i+3])
                if self._is_good_gram(gram):
                    tri_counter[gram] += 1
            for i in range(len(chars) - 3):
                gram = "".join(chars[i:i+4])
                if self._is_good_gram(gram):
                    quad_counter[gram] += 1

        def score_gram(gram: str) -> float:
            return len(gram) * 1.5

        scored = []
        for gram, cnt in quad_counter.most_common(15):
            if cnt >= min_samples:
                scored.append((gram, cnt * score_gram(gram)))
        for gram, cnt in tri_counter.most_common(15):
            if cnt >= min_samples and gram not in {s[0] for s in scored}:
                scored.append((gram, cnt * score_gram(gram)))
        for gram, cnt in bi_counter.most_common(10):
            if cnt >= min_samples and gram not in {s[0] for s in scored}:
                scored.append((gram, cnt * score_gram(gram)))

        scored.sort(key=lambda x: -x[1])

        from .router import SKIP_FLASH_INDICATORS as existing_kw
        existing_set = {kw.lower() for kw in existing_kw}
        candidates = [g for g, _ in scored if g not in existing_set]

        count = len(candidates)
        logger.info(f"Auto-extracted {count} keywords from {len(failed_texts)} failed-flash records")
        return candidates[:15]

    def extract_keywords_from_text(self, text: str) -> list[str]:
        """从单条请求文本中提取关键词，用于实时学习。"""
        if not text or len(text) < 3:
            return []
        chars = list(text)
        candidates = set()
        # 2-gram
        for i in range(len(chars) - 1):
            gram = "".join(chars[i:i+2])
            if self._is_good_gram(gram):
                candidates.add(gram)
        # 3-gram
        for i in range(len(chars) - 2):
            gram = "".join(chars[i:i+3])
            if self._is_good_gram(gram):
                candidates.add(gram)
        # 4-gram
        for i in range(len(chars) - 3):
            gram = "".join(chars[i:i+4])
            if self._is_good_gram(gram):
                candidates.add(gram)
        # 过滤：去掉已有的关键词
        from .router import get_all_indicators
        existing = {kw.lower() for kw in get_all_indicators()}
        result = [g for g in candidates if g not in existing]
        logger.info(f"Extracted {len(result)} keyword candidates from text ({len(text)} chars)")
        return result[:5]  # 最多取 5 个

    @staticmethod
    def _is_good_gram(gram: str) -> bool:
        """过滤掉无意义的 n-gram：纯标点、纯空格、纯数字。"""
        if len(gram) < 2:
            return False
        # 至少包含一个中文或英文字母
        has_chinese = any("一" <= c <= "鿿" for c in gram)
        has_english = any(c.isascii() and c.isalpha() for c in gram)
        if not has_chinese and not has_english:
            return False
        # 排除纯数字
        digits = sum(1 for c in gram if c.isdigit())
        if digits == len(gram):
            return False
        return True


# 全局单例
latency_tracker = LatencyTracker()
similarity_buffer = SimilarityBuffer()


class UpgradeStore:
    """记录 flash 被安全网判定不足的事件, 用 TF-DF 提取真实领域词。

    学习信号 = 安全网升级事件(不是 flash 报错 —— flash 几乎不报错)。
    维护两个文本集: U(升级/不足) vs S(flash 成功), 取 U 中高频且 S 中低频的词,
    作为"flash 不够"的指示词, 反哺 router 评分。
    """

    HISTORY_FILE = "upgrade_store.json"
    _MAX = 200

    def __init__(self):
        self._upgraded: list[str] = []
        self._success: list[str] = []
        self._keywords: list[str] = []
        self._lock = threading.Lock()
        self._dirty = False
        self.load()

    def _tokenize(self, text: str) -> list[str]:
        """英文按词(>=3 字符), 中文按 2/3-gram(连续中文段)。"""
        tokens: list[str] = []
        low = (text or "").lower()
        for m in re.findall(r"[a-z][a-z0-9_]{2,}", low):
            tokens.append(m)
        for seg in re.findall(r"[\u4e00-\u9fff]+", text or ""):
            for i in range(len(seg) - 1):
                tokens.append(seg[i:i+2])
            for i in range(len(seg) - 2):
                tokens.append(seg[i:i+3])
        return tokens

    def record_upgrade(self, text: str, reason: str = ""):
        if not text or len(text.strip()) < 3:
            return
        with self._lock:
            self._upgraded.append(text[:500])
            if len(self._upgraded) > self._MAX:
                self._upgraded = self._upgraded[-self._MAX:]
            self._recompute_locked()
            self._dirty = True
        logger.info(f"[UpgradeLearn] upgrade recorded ({reason}) "
                    f"U={len(self._upgraded)} kw={len(self._keywords)}")

    def record_success(self, text: str):
        if not text or len(text.strip()) < 3:
            return
        with self._lock:
            self._success.append(text[:500])
            if len(self._success) > self._MAX:
                self._success = self._success[-self._MAX:]
            if len(self._success) % 10 == 0:
                self._recompute_locked()
                self._dirty = True

    def _recompute_locked(self):
        if len(self._upgraded) < 3:
            return
        u_tokens = [t for txt in self._upgraded for t in self._tokenize(txt)]
        s_tokens = [t for txt in self._success for t in self._tokenize(txt)]
        u_freq = Counter(u_tokens)
        s_freq = Counter(s_tokens)
        u_total = max(1, len(u_tokens))
        s_total = max(1, len(s_tokens))
        scored = []
        for tok, cnt in u_freq.items():
            if cnt < 2:
                continue
            u_rate = cnt / u_total
            s_rate = s_freq.get(tok, 0) / s_total
            if u_rate > s_rate * 2:
                scored.append((tok, u_rate - s_rate))
        scored.sort(key=lambda x: -x[1])
        new_kw = [t for t, _ in scored[:20]]
        if new_kw != self._keywords:
            self._keywords = new_kw

    def learned_keywords(self) -> list[str]:
        with self._lock:
            return list(self._keywords)

    def upgrade_risk(self, text: str) -> bool:
        """新请求是否与历史升级请求相似 → 高风险(应避免 flash)。"""
        with self._lock:
            if not self._upgraded:
                return False
            target = self._trigrams(text)
            recent = list(self._upgraded[-50:])
        if not target:
            return False
        for txt in recent:
            sim = len(target & self._trigrams(txt)) / max(1, len(target | self._trigrams(txt)))
            if sim > 0.3:
                return True
        return False

    @staticmethod
    def _trigrams(text: str) -> set:
        t = (text or "").lower()
        if len(t) < 3:
            return {t} if t else set()
        return {t[i:i+3] for i in range(len(t) - 2)}

    def stats(self) -> dict:
        with self._lock:
            return {"upgraded": len(self._upgraded), "success": len(self._success),
                    "learned_keywords": len(self._keywords)}

    def persist(self, filepath: str | None = None):
        with self._lock:
            if not self._dirty:
                return
            data = {"upgraded": self._upgraded, "success": self._success, "keywords": self._keywords}
            self._dirty = False
        if filepath is None:
            filepath = str(Path(__file__).resolve().parent.parent / "logs" / self.HISTORY_FILE)
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            tmp = filepath + ".tmp"
            Path(tmp).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, filepath)
            logger.info(f"UpgradeStore saved → {filepath}")
        except Exception as e:
            logger.error(f"UpgradeStore persist failed: {e}")

    def load(self, filepath: str | None = None):
        if filepath is None:
            filepath = str(Path(__file__).resolve().parent.parent / "logs" / self.HISTORY_FILE)
        try:
            data = json.loads(Path(filepath).read_text(encoding="utf-8"))
            with self._lock:
                self._upgraded = list(data.get("upgraded", []))[-self._MAX:]
                self._success = list(data.get("success", []))[-self._MAX:]
                self._keywords = list(data.get("keywords", []))
            logger.info(f"UpgradeStore loaded: U={len(self._upgraded)} "
                        f"S={len(self._success)} kw={len(self._keywords)}")
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"UpgradeStore load failed: {e}")


upgrade_store = UpgradeStore()
