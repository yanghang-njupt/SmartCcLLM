"""路由引擎 —— 三级难度评估路由

根据任务内容在**发送前**评估难度，不再事后补救：
  - 简单（easy, score<12）   → deepseek-v4-flash
  - 中等（medium, score≥12） → deepseek-v4-pro
  - 困难（hard, score≥40）   → kimi-k2.7（不可用时降级 pro）

评分维度：
  ① 关键词命中（medium +12, hard +30）
  ② 超长上下文（>100k tokens +20）
  ③ 历史学习（相似请求 flash 失败 +20）
"""
import hashlib, threading, time, logging
from dataclasses import dataclass
from .config import Settings
from . import state
from .metrics import similarity_buffer

logger = logging.getLogger("SmartProxy.Router")

# ── 中等难度关键词（+12/命中）：需要较好理解能力 ──────────
MEDIUM_INDICATORS = [
    # 架构/设计/方案
    "设计文档", "架构设计", "整体架构", "系统设计",
    "技术方案", "方案设计", "设计方案", "架构分析",
    "设计思路", "实现方案", "设计评审", "架构评审",
    "架构决策", "方案对比", "总体方案", "详细方案",
    # 审查/评估
    "审查整个", "review entire",
    "审计",
    # 优化
    "optimize memory", "optimize performance",
    # 英文设计/架构
    "design a", "design the",
    # 中等复杂度
    "docker", "kubernetes", "ci/cd",
    "pipeline with",
]

# ── 困难难度关键词（+30/命中）：复杂工程任务 ────────────
HARD_INDICATORS = [
    # 重构/重写
    "重构", "重写", "refactor", "rewrite",
    # 安全
    "安全审计", "security audit", "安全漏洞",
    "xss", "sql injection", "csrf", "encrypt",
    # 复杂实现
    "implement oauth", "implement auth",
    "implement a distributed", "implement distributed",
    "implement end-to-end",
    # 迁移
    "migrate", "迁移",
    # 分布式系统
    "distributed", "分布式", "kafka",
    # 高难度工程
    "fix the race", "fix race", "race condition",
    "build a",
    "audit log", "audit trail", "implement tamper",
]

# ── 动态扩展：历史学习自动生成的补充关键词（按 medium 计分）─
_extra_keywords: list[str] = []


def extend_keywords(keywords: list[str]) -> list[str]:
    """从历史数据自动学习，扩展中等难度关键词。"""
    global _extra_keywords
    existing = {k.lower() for k in MEDIUM_INDICATORS}
    existing.update(k.lower() for k in HARD_INDICATORS)
    existing.update(k.lower() for k in _extra_keywords)
    added = [kw for kw in keywords if kw.lower() not in existing]
    _extra_keywords.extend(added)
    if added:
        logger.info(f"Auto-extended keywords ({len(added)}): {added}")
    return added


def get_all_indicators() -> dict:
    return {"medium": len(MEDIUM_INDICATORS),
            "hard": len(HARD_INDICATORS),
            "auto": len(_extra_keywords)}


# ── 文本提取 ─────────────────────────────────────────────

def _extract_user_text(messages):
    """提取最后一条 user 消息的纯文本。"""
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


# ── token 估计 ───────────────────────────────────────────

def _token_estimate(messages):
    total = 0
    for msg in (messages if isinstance(messages, list) else []):
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    total += len(block.get("text", ""))
        elif isinstance(content, str):
            total += len(content)
    return total // 4


# ── 后端可用性 ───────────────────────────────────────────

def _is_available(name):
    return not state.is_blocked(name) and not state.is_overloaded(name)


def _pick_best(settings):
    """选择最强可用后端。"""
    if "kimi" in settings.backends and _is_available("kimi"):
        return "kimi", "default"
    return "deepseek", "pro"


# ── 三级难度评估 ─────────────────────────────────────────

THRESHOLD_MEDIUM = 12   # 命中一个 medium 关键词就升级到 pro
THRESHOLD_HARD = 40     # 命中 hard 关键词 + 可能的中等加权 → kimi
TOKEN_LONG = 100000     # tokens
EXTRA_LONG_PENALTY = 20


def _assess(text: str, settings: Settings) -> tuple[int, str]:
    """返回 (score, difficulty)。"""
    lower = text.lower()
    score = 0

    # ① 关键词评分
    for kw in MEDIUM_INDICATORS:
        if kw in lower:
            score += 12
    for kw in HARD_INDICATORS:
        if kw in lower:
            score += 30
    for kw in _extra_keywords:
        if kw in lower:
            score += 12

    # ② 超长上下文惩罚
    tl = settings.routing.token_length
    if tl.get("enabled"):
        est = len(text) // 4  # 粗略 token 估算
        if est >= tl.get("threshold_tokens", TOKEN_LONG):
            score += EXTRA_LONG_PENALTY

    # ③ 历史学习（留到 route() 中做，因为需要 similarity_buffer）
    # 由 route() 在调用 _assess 后补充

    if score >= THRESHOLD_HARD:
        return score, "hard"
    if score >= THRESHOLD_MEDIUM:
        return score, "medium"
    return score, "easy"


# ── 路由结果 ─────────────────────────────────────────────

@dataclass
class RoutingScore:
    backend: str
    tier: str            # "flash" | "pro" | "kimi"
    difficulty: str      # "easy" | "medium" | "hard"
    reason: str
    score: int = 0


# ── 主路由入口 ───────────────────────────────────────────

def route(settings: Settings, messages) -> RoutingScore:
    text = _extract_user_text(messages).lower()
    preview = text[:80].replace("\n", " ") if text else "(no text)"

    # ── 0. 会话粘性 ────────────────────────────────────
    sticky_result = _check_sticky(settings, messages, text)
    if sticky_result:
        be, t = sticky_result
        return RoutingScore(backend=be, tier=t, difficulty="sticky",
                            reason=f"Sticky: {preview}")

    # ── 1. 基础关键词 + token 长度评分 ─────────────────
    score, difficulty = _assess(text, settings)

    # ── 2. 历史数据补充评分 ────────────────────────────
    hist = similarity_buffer.get_stats(text)
    if hist and hist["flash_failure_rate"] > 0.3 and hist["flash_samples"] >= 3:
        score += 20
        if score >= THRESHOLD_HARD:
            difficulty = "hard"
        elif score >= THRESHOLD_MEDIUM:
            difficulty = "medium"

    # ── 3. 按难度分配后端 + tier ──────────────────────
    if difficulty == "hard":
        be, t = _pick_best(settings)
        return RoutingScore(backend=be, tier=t, difficulty="hard",
                            reason=f"Hard({score}): {preview}", score=score)

    if difficulty == "medium":
        return RoutingScore(backend="deepseek", tier="pro", difficulty="medium",
                            reason=f"Medium({score}): {preview}", score=score)

    # easy
    return RoutingScore(backend="deepseek", tier="flash", difficulty="easy",
                        reason=f"Easy({score}): {preview}", score=score)
