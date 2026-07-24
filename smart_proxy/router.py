"""路由引擎 —— 三层分流

需求：
  - 简单任务 → flash（最便宜，省 token）
  - 中等任务 → kimi（最强，K2.7 能力强且便宜）
  - 复杂任务 → kimi 不可用 → pro（次强，当做 fallback）

分流逻辑：
  - flash_first: 所有请求先试 flash
  - flash 成功（stop_reason=end_turn 且非截断）→ 返回 flash 结果
  - flash 失败/截断/过慢 → 升级
    - kimi 可用 → kimi
    - kimi 不可用 → pro
  - 极高难度（重构/重写/安全审计等）→ 跳过 flash，直接 kimi（或 pro）
"""
import hashlib, threading, time, logging
from dataclasses import dataclass, field
from .config import Settings
from . import state
from .metrics import similarity_buffer

logger = logging.getLogger("SmartProxy.Router")

# ── 极高难度：跳过 flash，直接用更强模型 ──────────────────
SKIP_FLASH_INDICATORS = [
    # 重构/重写
    "重构", "重写", "refactor", "rewrite",
    # 安全
    "安全审计", "security audit", "安全漏洞",
    "xss", "sql injection", "csrf", "encrypt",
    "审查整个", "review entire", "review the entire", "review the entire",
    # 复杂设计/实现
    "design a", "design the", "设计一个",
    "implement oauth", "implement auth",
    "implement a distributed", "implement distributed",
    "implement end-to-end", "implement the",
    # 方案设计/架构决策（需要强模型的理解能力）
    "方案设计", "设计方案", "架构设计", "设计文档",
    "整体架构", "系统设计", "技术方案", "架构评审",
    "总体方案", "详细方案", "架构分析", "方案对比",
    "设计评审", "架构决策", "设计思路", "实现方案",
    # 性能优化（复杂级别）
    "optimize memory", "optimize performance",
    # 迁移
    "migrate", "迁移",
    # DevOps
    "docker", "kubernetes", "ci/cd",
    # 审计
    "audit log", "audit trail", "implement tamper",
    # 并发/锁问题修复
    "fix the race", "fix race", "race condition",
    # 分布式系统/数据处理
    "distributed", "分布式",
    "build a", "kafka", "pipeline with",
]

# 动态扩展：由历史学习模块在启动时自动生成
_extra_keywords: list[str] = []


def extend_keywords(keywords: list[str]) -> list[str]:
    """从历史数据中自动学习，动态扩展跳过 flash 的关键词列表。"""
    global _extra_keywords
    existing = {k.lower() for k in SKIP_FLASH_INDICATORS}
    existing.update(k.lower() for k in _extra_keywords)
    added = [kw for kw in keywords if kw.lower() not in existing]
    _extra_keywords.extend(added)
    if added:
        logger.info(f"Auto-extended skip-flash keywords ({len(added)}): {added}")
    return added


def get_all_indicators() -> list[str]:
    """返回完整关键词列表（静态 + 动态）。"""
    return SKIP_FLASH_INDICATORS + _extra_keywords


def _skip_flash(text):
    lower = text.lower()
    if any(s in lower for s in SKIP_FLASH_INDICATORS):
        return True
    if any(s in lower for s in _extra_keywords):
        return True
    return False


# ── 文本提取 ─────────────────────────────────────────────

def _extract_user_text(messages):
    """提取最后一条 user 消息的纯文本（跳过系统 prompt 注入等前面的 user 消息）。"""
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
_sticky_store: dict[str, tuple[str, str, float]] = {}
_sticky_lock = threading.Lock()


def _session_key(messages, text=None):
    if text is None:
        text = _extract_user_text(messages)
    if not text:
        return None
    return hashlib.md5(text[:200].encode("utf-8")).hexdigest()[:16]


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
            return None
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
    """后端是否可用。"""
    return not state.is_blocked(name) and not state.is_overloaded(name)


def _pick_best(settings):
    """选择最佳后端：kimi 优先，不可用则 pro。用于 flash 升级 / skip flash 场景。"""
    if "kimi" in settings.backends and _is_available("kimi"):
        return "kimi", "default"
    return "deepseek", "pro"


def _pick_any(settings):
    """最后降级：任何可用的后端。"""
    for name in settings.backends:
        if _is_available(name):
            return name, "default"
    return "deepseek", "flash"


# ── 路由结果 ─────────────────────────────────────────────

@dataclass
class RoutingScore:
    backend: str
    tier: str
    strategy: str   # "flash_first" | "skip_flash"
    reason: str
    # flash 升级目标（controller 在 flash 不够时使用）
    upgrade_backend: str = ""
    upgrade_tier: str = ""


# ── 主路由入口 ───────────────────────────────────────────

def route(settings: Settings, messages) -> RoutingScore:
    text = _extract_user_text(messages).lower()
    preview = text[:80].replace("\n", " ") if text else "(no text)"

    # ── 0. 超长上下文 → 跳过 flash，用最强 ─────────────
    tl = settings.routing.token_length
    if tl.get("enabled"):
        est = _token_estimate(messages)
        if est >= tl.get("threshold_tokens", 100000):
            be, t = _pick_best(settings)
            return RoutingScore(backend=be, tier=t, strategy="skip_flash",
                                reason=f"TokenLength({est}t): {preview}")

    # ── 1. 会话粘性 ────────────────────────────────────
    sticky_result = _check_sticky(settings, messages, text)
    if sticky_result:
        be, t = sticky_result
        return RoutingScore(backend=be, tier=t, strategy="flash_first",
                            reason=f"Sticky: {preview}")

    # ── 2. 历史数据：flash 在相似请求上表现差？─────────
    hist = similarity_buffer.get_stats(text)
    if hist and hist["flash_failure_rate"] > 0.3 and hist["flash_samples"] >= 3:
        be, t = _pick_best(settings)
        return RoutingScore(backend=be, tier=t, strategy="skip_flash",
                            reason=f"History(fail={hist['flash_failure_rate']:.0%}): {preview}")

    # ── 3. 极高难度 → 跳过 flash ──────────────────────
    if _skip_flash(text):
        be, t = _pick_best(settings)
        return RoutingScore(backend=be, tier=t, strategy="skip_flash",
                            reason=f"SkipFlash: {preview}")

    # ── 4. 默认：flash first ──────────────────────────
    up_be, up_t = _pick_best(settings)
    return RoutingScore(backend="deepseek", tier="flash",
                         strategy="flash_first",
                         reason=f"FlashFirst: {preview}",
                         upgrade_backend=up_be, upgrade_tier=up_t)
