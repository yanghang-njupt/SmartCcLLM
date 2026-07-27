"""路由引擎 —— 分层难度评估(置信度 + 经济闸门)

决策层次:
  0. 请求清洗: 剥离 system-reminder/transcript/tool_result 噪声, 识别续传 vs 新问题
  1. 启发式预筛(<1ms, 零成本): 小核心关键词 + 结构信号 → (score, confidence)
     - 高置信 easy/hard → 直接路由, 跳过分类器
     - 低置信 → difficulty="uncertain", 交 controller 调 LLM 分类器(经济闸门)
  2. (controller 侧) uncertain → 200 token flash 分类器 → easy/medium/hard

难度 → 后端:
  easy   → deepseek flash
  medium → deepseek pro
  hard   → kimi(default) 不可用降级 pro
  uncertain → 暂定 flash, 待 controller 分类
  continuation(工具结果/续传) → flash, 不分类
"""
import hashlib, re, threading, time, logging
from dataclasses import dataclass
from .config import Settings
from . import state
from .metrics import similarity_buffer

logger = logging.getLogger("SmartProxy.Router")

# ── 小核心关键词集(只留高置信信号, 不再穷举) ───────────────
# easy: 读/搜/列/找文件 类动词(词边界匹配, 防 "concatenate"→"cat")
EASY_VERBS = [
    "ls", "grep", "cat", "find", "read", "list", "show", "open", "browse", "glob",
    "查看", "列出", "查找", "搜索", "读取", "看一下", "找一下", "浏览",
]
# medium: 单文件修复/小重构类动词
EDIT_VERBS = ["fix", "implement", "refactor", "rewrite", "debug", "编写", "修改", "重构", "修复", "重写"]
# hard: 多文件/架构/迁移/审计
HARD_CORE = [
    "架构", "设计", "迁移", "migrate", "distributed", "分布式",
    "审计", "oauth", "end-to-end", "kafka", "kubernetes", "微服务",
]

# ── 动态扩展: UpgradeStore 学到的词(按 medium 计分) ────────
_extra_keywords: list[str] = []


def extend_keywords(keywords: list[str]) -> list[str]:
    """从历史/学习数据扩展关键词。"""
    global _extra_keywords
    existing = {k.lower() for k in EASY_VERBS + EDIT_VERBS + HARD_CORE + _extra_keywords}
    added = [kw for kw in keywords if kw and kw.lower() not in existing]
    _extra_keywords.extend(added)
    if added:
        logger.info(f"Extended keywords ({len(added)}): {added}")
    return added


def get_all_indicators() -> dict:
    return {"easy": len(EASY_VERBS), "edit": len(EDIT_VERBS),
            "hard": len(HARD_CORE), "learned": len(_extra_keywords)}


# ── 文本提取 ─────────────────────────────────────────────

def _extract_user_text(messages):
    if not isinstance(messages, list):
        return ""
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    t = block.get("text", "")
                    if t.strip():
                        return t
        elif isinstance(content, str) and content.strip():
            return content
    return ""


# ── 请求清洗: 剥离 Claude Code 注入的噪声 ─────────────────
_SYSREM_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL | re.IGNORECASE)
_TRANSCRIPT_RE = re.compile(r"<transcript>.*?</transcript>", re.DOTALL | re.IGNORECASE)
_FUNCRESULTS_RE = re.compile(r"<function_results>.*?</function_results>", re.DOTALL | re.IGNORECASE)
_ANTML_FUNCRESULTS_RE = re.compile(r"<antml:function_results>.*?</antml:function_results>", re.DOTALL | re.IGNORECASE)


def _sanitize(text: str) -> tuple[str, bool]:
    """剥离系统噪声, 返回 (clean_text, is_continuation)。

    is_continuation=True 表示这是工具结果回传/系统文本主导(非新问题),
    直接走 flash, 不进分类器。
    """
    if not text:
        return "", True
    original_len = len(text)
    text = _SYSREM_RE.sub("", text)
    text = _TRANSCRIPT_RE.sub("", text)
    text = _FUNCRESULTS_RE.sub("", text)
    text = _ANTML_FUNCRESULTS_RE.sub("", text)
    clean = text.strip()
    # 仅当剥离后几乎为空(纯系统文本/工具结果回传, 无真实指令)才判续传。
    # 短但真实的指令("ls -la"/"看一下 app.py")交给启发式, 不误杀。
    is_continuation = len(clean) < 3
    return clean, is_continuation


# ── 词边界匹配(中英通用, 防 "concatenate"→"cat") ──────────

def _is_word_boundary(text: str, idx: int) -> bool:
    if idx < 0 or idx >= len(text):
        return True
    ch = text[idx]
    if '一' <= ch <= '鿿' or '㐀' <= ch <= '䶿':
        return True  # CJK 天然边界
    return not ch.isalnum()


def _match(text: str, kw: str) -> bool:
    """子串 + 词边界匹配。"""
    i = text.find(kw)
    while i != -1:
        if _is_word_boundary(text, i - 1) and _is_word_boundary(text, i + len(kw)):
            return True
        i = text.find(kw, i + 1)
    return False


# ── 会话粘性 ─────────────────────────────────────────────

_sticky_store: dict = {}
_sticky_lock = threading.Lock()


def _session_key(messages, text=None):
    if text is None:
        text = _extract_user_text(messages)
    return hashlib.md5((text or "")[:200].encode("utf-8")).hexdigest()[:16] if text else None


def _check_sticky(settings, messages, text):
    sticky = settings.routing.session_sticky
    if not sticky.get("enabled"):
        return None
    key = _session_key(messages, text)
    if not key:
        return None
    now = time.time()
    with _sticky_lock:
        entry = _sticky_store.get(key)
        if entry and now < entry[2]:
            be, t = entry[0], entry[1]
            if _is_available(be):
                return be, t
        expired = [k for k, v in _sticky_store.items() if v[2] <= now]
        for k in expired:
            del _sticky_store[k]
    return None


def _set_sticky(settings, messages, backend, tier):
    sticky = settings.routing.session_sticky
    if not sticky.get("enabled"):
        return
    key = _session_key(messages)
    if not key:
        return
    ttl = sticky.get("ttl_seconds", 600)
    with _sticky_lock:
        _sticky_store[key] = (backend, tier, time.time() + ttl)


# ── 后端可用性 ───────────────────────────────────────────

def _is_available(name):
    return not state.is_blocked(name) and not state.is_overloaded(name)


def _pick_best(settings):
    if "kimi" in settings.backends and _is_available("kimi"):
        return "kimi", "default"
    return "deepseek", "pro"


# ── 启发式评分 ───────────────────────────────────────────

THRESHOLD_MEDIUM = 12
THRESHOLD_HARD = 40
EXTRA_LONG_PENALTY = 20


def _heuristic_score(clean: str, settings: Settings) -> tuple[int, str, str]:
    """返回 (score, confidence, difficulty_hint)。

    confidence: "high" 直接路由; "low" 交 controller 调分类器
    """
    lower = clean.lower()
    score = 0

    has_easy = any(_match(lower, v) for v in EASY_VERBS)
    has_edit = any(_match(lower, v) for v in EDIT_VERBS)
    hard_hits = [v for v in HARD_CORE if _match(lower, v)]
    for _ in hard_hits:
        score += 30
    if has_edit:
        score += 12
    for kw in _extra_keywords:
        if kw and kw in lower:
            score += 12

    # 超长上下文惩罚
    tl = settings.routing.token_length
    if tl.get("enabled"):
        est = len(clean) // 4
        if est >= tl.get("threshold_tokens", 100000):
            score += EXTRA_LONG_PENALTY

    long_text = len(clean) > 2000

    # 高置信判定
    if hard_hits:
        return score, "high", "hard"
    if has_edit and (long_text or score >= THRESHOLD_MEDIUM):
        return score, "high", ("medium" if score < THRESHOLD_HARD else "hard")
    if has_easy and not has_edit and not long_text:
        return score, "high", "easy"
    if score >= THRESHOLD_HARD:
        return score, "high", "hard"
    if score >= THRESHOLD_MEDIUM:
        return score, "high", "medium"
    # 拿不准
    return score, "low", "easy"


# ── 路由结果 ─────────────────────────────────────────────

@dataclass
class RoutingScore:
    backend: str
    tier: str            # "flash" | "pro" | "default" | ...
    difficulty: str      # "easy"|"medium"|"hard"|"uncertain"|"continuation"|"sticky"
    reason: str
    score: int = 0
    clean_text: str = "" # 供 controller 分类/学习用


# ── 主路由入口 ───────────────────────────────────────────

def route(settings: Settings, messages) -> RoutingScore:
    raw = _extract_user_text(messages)
    clean, is_continuation = _sanitize(raw)
    preview = (clean[:80].replace("\n", " ")) if clean else "(no text)"

    # 0. 会话粘性
    sticky_result = _check_sticky(settings, messages, clean)
    if sticky_result:
        be, t = sticky_result
        return RoutingScore(be, t, "sticky", f"Sticky: {preview}", clean_text=clean)

    # 1. 续传/工具结果 → flash, 不分类
    if is_continuation:
        return RoutingScore("deepseek", "flash", "continuation",
                            f"Continuation: {preview}", clean_text=clean)

    # 2. 启发式评分
    score, conf, hint = _heuristic_score(clean, settings)

    # 3. 升级风险(历史学习): 相似请求曾被升级 → 提分 + 强制分类
    try:
        from .metrics import upgrade_store
        if upgrade_store.upgrade_risk(clean):
            score += 20
            conf = "low"
    except Exception:
        pass

    # 4. 历史相似统计(保留原 sw_ranking 思路)
    hist = similarity_buffer.get_stats(clean)
    if hist and hist.get("flash_failure_rate", 0) > 0.3 and hist.get("flash_samples", 0) >= 3:
        score += 20
        conf = "low"

    if conf == "high":
        diff = hint
    else:
        diff = "uncertain"

    if diff == "hard":
        be, t = _pick_best(settings)
        return RoutingScore(be, t, "hard", f"Hard({score}): {preview}", score, clean)
    if diff == "medium":
        return RoutingScore("deepseek", "pro", "medium", f"Medium({score}): {preview}", score, clean)
    if diff == "easy":
        return RoutingScore("deepseek", "flash", "easy", f"Easy({score}): {preview}", score, clean)
    # uncertain: 暂定 flash, controller 会调分类器再定
    return RoutingScore("deepseek", "flash", "uncertain",
                        f"Uncertain({score}): {preview}", score, clean)
