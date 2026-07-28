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
                if sim < 0.25:
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
            logger.info(f"History saved ({len(data)} records) -> {filepath}")
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
            logger.info(f"History loaded ({len(data)} records) <- {filepath}")
            return True
        except FileNotFoundError:
            return False
        except json.JSONDecodeError as e:
            logger.warning(f"History file corrupt, ignoring: {e}")
            return False

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

    v4.4: 中文 tokenize 优先用 jieba 分词, fallback 停用词过滤 n-gram。
    """

    HISTORY_FILE = "upgrade_store.json"
    _MAX = 200

    # 中文停用词: 代词/助词/连词/量词 — 不参与 TF-DF
    _CN_STOP = frozenset([
        "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一",
        "个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没",
        "看", "好", "自己", "这", "那", "他", "她", "它", "们", "些", "什么",
        "怎么", "如何", "因为", "所以", "但是", "但", "却", "而", "或", "如果",
        "虽然", "可以", "还是", "这个", "那个", "吗", "呢", "吧", "啊", "哦",
        "嗯", "哈", "呀", "被", "把", "让", "从", "对", "与", "及", "其",
        "以", "之", "为", "所", "能", "将", "已", "正在", "已经", "还", "又",
        "再", "才", "刚", "便", "则", "只", "无", "各", "每", "某", "请", "谢",
        "现", "老", "次", "搞", "现在", "是不是", "搞错", "错了",
    ])

    @staticmethod
    def _has_stop(gram: str) -> bool:
        """n-gram 中任一字符是停用字符则过滤。"""
        for ch in gram:
            if ch in UpgradeStore._CN_STOP:
                return True
        return False

    def __init__(self):
        self._upgraded: list[str] = []
        self._success: list[str] = []
        self._keywords: list[str] = []
        self._lock = threading.Lock()
        self._dirty = False
        self.load()

    def _tokenize(self, text: str) -> list[str]:
        """英文按词(>=3字母), 中文优先 jieba 分词, fallback 停用词过滤 n-gram。

        jieba:  "手部映射的角度有问题" -> [手部, 映射, 角度, 问题]
        n-gram: 同句 + 停用词过滤      -> [手部, 映射, 角度, 问题]
        """
        tokens: list[str] = []

        # 1. 英文:  >=3 字母/数字组合词
        low = (text or "").lower()
        for m in re.findall(r"[a-z][a-z0-9_]{2,}", low):
            tokens.append(m)

        # 2. 中文: jieba 优先, 未安装时退化
        try:
            import jieba as _jieba
            for w in _jieba.cut(text, cut_all=False):
                w = w.strip()
                if len(w) >= 2 and w not in self._CN_STOP:
                    tokens.append(w)
        except ImportError:
            for seg in re.findall(r"[一-鿿]+", text or ""):
                for i in range(len(seg) - 1):
                    bi = seg[i:i+2]
                    if not self._has_stop(bi):
                        tokens.append(bi)
                for i in range(len(seg) - 2):
                    tri = seg[i:i+3]
                    if not self._has_stop(tri):
                        tokens.append(tri)

        return tokens

    def _fallback_extract(self, u_all: str, s_all: str) -> list[str]:
        """jieba 不可用时的降级方案：字符 n-gram + 严格去噪。"""
        u_tokens = [t for txt in self._upgraded for t in self._tokenize(txt)]
        s_tokens = [t for txt in self._success[-len(self._upgraded):]
                    for t in self._tokenize(txt)]
        u_freq = Counter(u_tokens)
        s_freq = Counter(s_tokens)
        u_total = max(1, len(u_tokens))
        s_total = max(1, len(s_tokens))
        scored = []
        for tok, cnt in u_freq.items():
            if cnt < 3:
                continue
            u_rate = cnt / u_total
            s_rate = s_freq.get(tok, 0) / s_total
            if u_rate > s_rate * 3 and len(tok) >= 2:
                scored.append((tok, u_rate - s_rate))
        scored.sort(key=lambda x: -x[1])
        return [t for t, _ in scored[:20]]

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

    # 通用名词——高频但对任务难度无指示意义
    _CN_GENERIC_NOUNS = frozenset([
        "问题", "情况", "方法", "东西", "时候", "原因", "结果", "内容",
        "部分", "方面", "方式", "过程", "地方", "时间", "事情",
    ])

    def _recompute_locked(self):
        if len(self._upgraded) < 3:
            return
        u_all = "\n".join(self._upgraded)
        s_all = "\n".join(self._success[-len(self._upgraded):])

        try:
            import jieba.posseg as _ps
            # 只用 U 中的名词做候选(领域词都是名词: 手部/角度/弧度/串口/波特率)
            u_nouns = Counter()
            for txt in self._upgraded:
                seen = set()
                for w, flag in _ps.cut(txt):
                    if flag.startswith("n") and len(w) >= 2 and w not in self._CN_STOP:
                        if w not in seen:
                            u_nouns[w] += 1
                            seen.add(w)
            # 去掉在成功文本中也出现的词 + 通用名词
            s_nouns = set()
            for txt in self._success[-len(self._upgraded):]:
                for w, flag in _ps.cut(txt):
                    if flag.startswith("n") and len(w) >= 2:
                        s_nouns.add(w)
            candidates = [(w, c) for w, c in u_nouns.items()
                         if c >= 3 and w not in s_nouns
                         and w not in self._CN_GENERIC_NOUNS]
            candidates.sort(key=lambda x: -x[1])
            new_kw = [w for w, _ in candidates[:20]]
        except ImportError:
            new_kw = self._fallback_extract(u_all, s_all)
        if new_kw != self._keywords:
            self._keywords = new_kw
            # 实时注入到 router 关键词库, 不等重启
            try:
                from .router import extend_keywords as _extend
                added = _extend(new_kw)
                if added:
                    logger.info(f"[UpgradeLearn] injected {len(added)} keywords to router: {added}")
            except Exception:
                pass

    def learned_keywords(self) -> list[str]:
        with self._lock:
            return list(self._keywords)

    def upgrade_risk(self, text: str) -> bool:
        """新请求是否与历史升级请求相似 -> 高风险(应避免 flash)。"""
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
            logger.info(f"UpgradeStore saved -> {filepath}")
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
