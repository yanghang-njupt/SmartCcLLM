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
    "check", "look", "search", "get",
    "查看", "列出", "查找", "搜索", "读取", "看一下", "找一下", "浏览", "看看", "看",
]
# medium: 单文件修复/小重构类动词
EDIT_VERBS = ["fix", "implement", "refactor", "rewrite", "debug",
              "编写", "修改", "重构", "修复", "重写", "修"]
# hard: 多文件/架构/迁移/审计
HARD_CORE = [
    "架构", "设计", "迁移", "migrate", "distributed", "分布式",
    "审计", "oauth", "end-to-end", "kafka", "kubernetes", "微服务",
    "集群", "k8s",
]

# ── 动态扩展: UpgradeStore 学到的词(按 medium 计分) ────────
_extra_keywords: list[str] = []
_extra_kw_hits: dict[str, int] = {}      # v4.5: 命中计数，用于衰减淘汰
_EXTRA_KW_MAX = 200
_EXTRA_KW_DECAY_INTERVAL = 500            # 每 500 次评分触发衰减检查
_EXTRA_KW_MIN_HITS = 2                    # 低于此命中数 → 移除
_extra_kw_score_count: int = 0            # 评分调用计数器


def extend_keywords(keywords: list[str]) -> list[str]:
    """从历史/学习数据扩展关键词。超出上限时淘汰最旧的。"""
    global _extra_keywords
    existing = {k.lower() for k in EASY_VERBS + EDIT_VERBS + HARD_CORE + _extra_keywords}
    added = [kw for kw in keywords if kw and kw.lower() not in existing]
    if added:
        _extra_keywords.extend(added)
        if len(_extra_keywords) > _EXTRA_KW_MAX:
            _extra_keywords = _extra_keywords[-_EXTRA_KW_MAX:]
        logger.info(f"Extended keywords (+{len(added)}, total {len(_extra_keywords)}): {added}")
    return added


def _mark_kw_hits(hit_keywords: list[str]) -> None:
    """v4.5: 记录学到的关键词命中次数，用于衰减淘汰。"""
    global _extra_kw_hits, _extra_kw_score_count
    _extra_kw_score_count += 1
    for kw in hit_keywords:
        _extra_kw_hits[kw] = _extra_kw_hits.get(kw, 0) + 1
    # 每 N 次评分衰减一次
    if _extra_kw_score_count >= _EXTRA_KW_DECAY_INTERVAL:
        _extra_kw_score_count = 0
        _decay_unused_keywords()


def _decay_unused_keywords() -> None:
    """v4.5: 移除长期未被命中的学到的关键词。"""
    global _extra_keywords, _extra_kw_hits
    before = len(_extra_keywords)
    _extra_keywords = [kw for kw in _extra_keywords
                       if _extra_kw_hits.get(kw, 0) >= _EXTRA_KW_MIN_HITS]
    _extra_kw_hits = {}  # 重置，下一轮重新计数
    removed = before - len(_extra_keywords)
    if removed:
        logger.info(f"[KW-Decay] pruned {removed} unused learned keywords "
                    f"({before} → {len(_extra_keywords)})")


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
# ── 系统噪声的正则(两遍: 先删闭合块, 再删无闭合的孤立标签) ──
_SYSREM_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL | re.IGNORECASE)
_SYSREM_OPEN_RE = re.compile(r"<system-reminder[^>]*>", re.IGNORECASE)  # 处理未关闭的标签
_TRANSCRIPT_RE = re.compile(r"<transcript>.*?</transcript>", re.DOTALL | re.IGNORECASE)
_TRANSCRIPT_OPEN_RE = re.compile(r"<transcript[^>]*>", re.IGNORECASE)
_FUNCRESULTS_RE = re.compile(r"<function_results>.*?</function_results>", re.DOTALL | re.IGNORECASE)
_ANTML_FUNCRESULTS_RE = re.compile(r"<antml:function_results>.*?</antml:function_results>", re.DOTALL | re.IGNORECASE)


def _drop_noise(text: str) -> str:
    """多遍删除所有已知系统标签(无论是否闭合)。"""
    text = _SYSREM_RE.sub("", text)
    text = _SYSREM_OPEN_RE.sub("", text)     # 抓遗漏的未关闭标签
    text = _TRANSCRIPT_RE.sub("", text)
    text = _TRANSCRIPT_OPEN_RE.sub("", text)
    text = _FUNCRESULTS_RE.sub("", text)
    text = _ANTML_FUNCRESULTS_RE.sub("", text)
    return text


# ── Claude Code 系统上下文剥离: 去掉 CLAUDE.md / skill / MCP 指令 ──
# XML 标签层面的 _drop_noise 已经处理了 <system-reminder>...</system-reminder>。
# 但 Claude Code 注入的 CLAUDE.md 内容、可用 skills 列表、MCP Server 指令
# 是纯文本块，会溢出到 _sanitize() 之后。这些文本天然包含 "架构""设计""分布式"
# 等 HARD_CORE 词 → 每个 +30 分 → 请求误判 Hard(60) 走 Pro/Kimi。
#
# 这些上下文块都有高度可辨的头部行，匹配后整块删除。
_CC_HEADERS = [
    # CLAUDE.md 注入
    r"Contents of\s+\S*CLAUDE\.md[^\n]*\n",
    r"# CLAUDE\.md\s*\n",
    r"Behavioral guidelines to reduce common LLM coding mistakes[^\n]*\n",
    r"Codebase and user instructions are shown below[^\n]*\n",
    r"IMPORTANT: [Tt]hese (?:instructions|guidelines) OVERRIDE[^\n]*\n",
    r"IMPORTANT: this context may or may not be relevant[^\n]*\n",
    r"# currentDate\nToday'?s date is \d{4}-\d{2}-\d{2}\.\n?",
    # —— skill / MCP / agent 清单 ——
    r"The following skills are available for use with the Skill tool:\n(?:- [^\n]+\n)+",
    r"(?:# )?MCP [Ss]erver [Ii]nstructions\n"
    r"(?:The following MCP servers[^\n]*\n)?"
    r"(?:## \w+\n)?",
    r"Available agent types for the Agent tool:\n(?:- [^\n]+\n)+",
    # —— session 标签 ——
    r"<session>.*?</session>\s*",
    # —— 系统提醒残余行 ——
    r"As you answer the user'?s questions[^\n]*\n",
]

_CC_RE = re.compile("|".join(_CC_HEADERS), re.DOTALL | re.IGNORECASE)


def _strip_cc_context(text: str) -> str:
    """剥离 Claude Code 注入的系统上下文（CLAUDE.md/skill/MCP/agent 列表）。

    在 _drop_noise() 之后调用 —— XML 标签已清除，
    只剩纯文本系统指令，用已知头部行匹配并移除。
    """
    if not text or len(text) < 30:
        return text
    before = text
    text = _CC_RE.sub("", text)
    # 防范未标记的残留：头部行下面的内容可能不匹配，再做一次全扫描，
    # 删除任何以已知头部开头的行及其后续缩进行
    for header in [
        "Contents of ",
        "Behavioral guidelines to reduce",
        "Codebase and user instructions",
        "IMPORTANT: These instructions",
        "IMPORTANT: these instructions",
        "IMPORTANT: This context",
        "The following skills are available",
        "MCP Server Instructions",
        "The following MCP servers",
        "Available agent types for the Agent",
        "# currentDate",
        "As you answer",
    ]:
        idx = text.find(header)
        if idx != -1:
            # 从 header 行开始到下一个非缩进行结束，或到文本末尾
            rest = text[idx:].replace("\r\n", "\n")  # 规范化 Windows 行尾
            lines = rest.split("\n")
            # 吃 header 行
            consumed = 1
            # 吃后续缩进行 / 连续列表项行
            for i, line in enumerate(lines[1:], 1):
                stripped_line = line.strip()
                if not stripped_line:
                    consumed += 1
                    continue
                if stripped_line.startswith("- ") or stripped_line.startswith("# "):
                    consumed += 1
                    continue
                if line and line[0] in (" ", "\t"):
                    consumed += 1
                    continue
                break
            text = text[:idx] + "\n".join(lines[consumed:])
    if text != before:
        logger.debug(f"Stripped CC context: {len(before)} → {len(text)} chars")
    return text.strip()


def _sanitize(text: str) -> tuple[str, bool]:
    """剥离系统噪声, 返回 (clean_text, is_continuation)。

    is_continuation=True 表示这是工具结果回传/系统文本主导(非新问题),
    直接走 flash, 不进分类器。

    关键处理: 遇到无 </system-reminder> 闭合的标签(Claude Code 常见),
    内容中混有架构/设计/分布式等关键词会误判 hard。
    判据: 文本以 <system-reminder 开头但无闭合 → 续传 flash。
    """
    if not text:
        return "", True

    # 检测未闭合的 <system-reminder: 以标签开头但没有 </system-reminder>
    stripped = text.strip()
    has_closed = "</system-reminder>" in text
    no_closing_sysrem = (
        stripped.lower().startswith("<system-reminder")
        and not has_closed
    )

    clean = _drop_noise(text).strip()
    clean = _strip_cc_context(clean)  # v4.4: 剥离 CLAUDE.md/skill/MCP 系统上下文

    # 未闭合标签 → 内容全是系统提示残渣, 真实指令在下一条消息
    if no_closing_sysrem:
        return "", True

    # 防范混合场景: 文本同时包含闭合 </system-reminder> 和未闭合 <system-reminder。
    # _drop_noise 去掉了闭合块, 但未闭合的 <system-reminder 仍残留在 clean 中。
    if has_closed and "<system-reminder" in clean.lower():
        clean = _SYSREM_OPEN_RE.sub("", clean).strip()

    # 仅当剥离后为空(纯系统文本/工具结果回传, 无真实指令)才判续传。
    # "ls" / "cd" / "grep" 等短命令不走续传路径, 它们需要启发式评分 + sticky.
    is_continuation = len(clean) == 0
    return clean, is_continuation


# ── 词边界匹配(中英通用, 防 "concatenate"→"cat") ──────────

def _is_word_boundary(text: str, idx: int) -> bool:
    if idx < 0 or idx >= len(text):
        return True
    ch = text[idx]
    if '一' <= ch <= '鿿' or '㐀' <= ch <= '䶿':
        return True  # CJK 天然边界
    if ch.isdigit():
        return True  # 数字与CJK/英文之间也是边界(防 "Vue2迁移" 不匹配 "迁移")
    return not ch.isalnum()


def _match(text: str, kw: str) -> bool:
    """子串 + 词边界匹配。

    CJK 结尾的 keyword 右边界天然成立(不管后面是英文/数字),
    解决 "看一下config" 中 "看一下" 不匹配的问题。
    """
    i = text.find(kw)
    while i != -1:
        left_ok = _is_word_boundary(text, i - 1)
        right_ok = _is_word_boundary(text, i + len(kw))
        # CJK 结尾: 右边界天然成立(如 "看一下" 后跟英文也不截断)
        if not right_ok and kw and '一' <= kw[-1] <= '鿿':
            right_ok = True
        if left_ok and right_ok:
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
    # v4.5: 活跃度续期——如果同一 session 连续命中且间隔 < 30s，
    # 自动延长 TTL 避免 active 讨论被切成多次分类器调用。
    now = time.time()
    ACTIVE_THRESHOLD = 30  # 30 秒内命中视为活跃会话
    if tier == "pro":
        base_ttl = sticky.get("pro_ttl_seconds", 120)
        with _sticky_lock:
            existing = _sticky_store.get(key)
            if existing and existing[1] == "pro":
                last_set_time = existing[2] - base_ttl  # 反推上次设置时间
                if now - last_set_time < ACTIVE_THRESHOLD:
                    # 活跃会话：TTL 翻倍（最多 300s），避免 medium 任务被频繁重新评估
                    ttl = min(base_ttl * 2, 300)
                else:
                    ttl = base_ttl
            else:
                ttl = base_ttl
            _sticky_store[key] = (backend, tier, now + ttl)
    else:
        ttl = sticky.get("ttl_seconds", 600)
        with _sticky_lock:
            _sticky_store[key] = (backend, tier, now + ttl)


# ── 后端可用性 ───────────────────────────────────────────

def _is_available(name):
    return not state.is_blocked(name) and not state.is_overloaded(name)


def _pick_best(settings):
    """选择最优 hard 后端: kimi 优先, 不可用时降级 deepseek pro。

    熔断器会自动处理 kimi 403 quota → block 5h → _is_available 返回 False,
    之后请求自动走 deepseek pro, 无需手动切换优先级。
    """
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

    关键设计 — 信号密度: 一个"修改"夹在一堆"看看/找找"中间，
    大概率只是多步简单任务(改个变量名), 不应升级到 pro。
    用 easy/edit 信号比例来区分。
    """
    lower = clean.lower()
    score = 0

    # 去重计数: 长词包含短词时只计长词(防 "修改"+"修" 双倍)
    def _dedup_matches(words: list[str], target: str) -> list[str]:
        positioned = []
        for w in words:
            pos = target.find(w)
            while pos != -1:
                left_ok = _is_word_boundary(target, pos - 1)
                right_ok = _is_word_boundary(target, pos + len(w))
                if not right_ok and w and '一' <= w[-1] <= '鿿':
                    right_ok = True  # CJK 结尾天然边界
                if left_ok and right_ok:
                    positioned.append((w, pos, len(w)))
                pos = target.find(w, pos + 1)
        positioned.sort(key=lambda x: -x[2])  # by length desc
        kept = []
        occupied = set()
        for word, pos, ln in positioned:
            span = set(range(pos, pos + ln))
            if not span & occupied:
                kept.append(word)
                occupied |= span
        return kept

    hard_hits = [v for v in HARD_CORE if _match(lower, v)]
    edit_hits = _dedup_matches(EDIT_VERBS, lower)
    easy_hits = _dedup_matches(EASY_VERBS, lower)
    edit_count = len(edit_hits)
    easy_count = len(easy_hits)

    # hard 关键词是确定性的(架构/审计/迁移), 不会被 easy 信号稀释
    for _ in hard_hits:
        score += 30

    # edit 关键词可能只是改一行。easy 信号密集时降权:
    # "修改变量名 + 看一下代码 + 找一下文件" → 改个变量名而已, easy
    # "重构模块 + implement oauth" → 真正的工程任务, medium
    has_edit = edit_count > 0
    if has_edit:
        if easy_count >= 2 * edit_count:
            # easy 信号压倒性多 → 一个 trivial edit 夹在浏览/搜索中
            has_edit = False  # 不按 medium 计分
        else:
            score += 12

    extra_hits = [kw for kw in _extra_keywords if kw and kw in lower]
    for _ in extra_hits:
        score += 12
    if extra_hits:
        _mark_kw_hits(extra_hits)

    # 超长上下文惩罚
    tl = settings.routing.token_length
    if tl.get("enabled"):
        est = len(clean) // 4
        if est >= tl.get("threshold_tokens", 100000):
            score += EXTRA_LONG_PENALTY

    long_text = len(clean) > 2000

    has_easy = easy_count > 0

    # 高置信判定
    # v4.4: 不再因单个 hard 关键词就提前返回 hard。
    # "设计"和"架构"在中文里太泛("设计一个登录页面"≠hard),
    # 现在统一走 score 阈值: single hit=30<40→medium, 2+hits≥60→hard。
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
