"""LLM 分类器 —— 经济闸门下的小成本难度判定

借鉴 RouteLLM 的 causal_llm router 思路: 用一个便宜模型做 2-token 分类。
关键经济设计: 分类器只看"任务描述"(最后一条 user 消息, 剥噪声, 截 800 字),
不看完整对话上下文, 故输入恒定 ~200 token, 与会话长度无关。
判错代价(一次完整上下文双读/错档)远大于 200 token, 故对非极小请求稳赚。
"""
import json, hashlib, time, threading, logging
import httpx
from .config import Settings

logger = logging.getLogger("SmartProxy.Classifier")

# ── 分类 prompt(中英通用, 极简) ──────────────────────────
_PROMPT = (
    "判断下面编码任务复杂度, 只回一个词: easy medium hard\n"
    "类别定义:\n"
    "- easy: 读、搜、列文件、解释代码、简单问答、改一行、找文件\n"
    "- medium: 修一个函数、写一个新功能、写单元测试、写文档、小重构\n"
    "- hard: 多文件修改、架构设计、建新系统、跨项目迁移、安全审计\n"
    "任务:"
)

_VALID = ("easy", "medium", "hard")

# ── v4.5: 分类器误判反馈闭环 ──────────────────────────
# 当分类器判 easy 但安全网升级到 pro 时，记录文本的 trigram 指纹。
# 同一 trigram 模式被误判 ≥3 次后，分类器自动将 easy 提升为 medium。
_mispredict: dict[str, int] = {}    # trigram → 误判次数
_mispredict_lock = threading.Lock()
_MISPREDICT_THRESHOLD = 3
_MISPREDICT_MAX = 500               # 上限防止内存泄漏


def _trigrams(text: str) -> set:
    t = (text or "").lower()
    if len(t) < 3:
        return {t} if t else set()
    return {t[i:i + 3] for i in range(len(t) - 2)}


def record_mispredict(text: str) -> None:
    """安全网从 flash 升级到 pro → 分类器可能判错了 easy。记录 trigram 指纹。"""
    if not text:
        return
    tri = _trigrams(text)
    with _mispredict_lock:
        for t in tri:
            _mispredict[t] = _mispredict.get(t, 0) + 1
        # 超出上限时清理低频项
        if len(_mispredict) > _MISPREDICT_MAX:
            keep = {t: c for t, c in _mispredict.items() if c >= _MISPREDICT_THRESHOLD}
            if keep:
                _mispredict.clear()
                _mispredict.update(keep)


def check_easy_override(text: str) -> str | None:
    """如果文本与已知误判模式高度重叠，返回 'medium' 覆写 easy 判定。"""
    tri = _trigrams(text)
    if not tri:
        return None
    with _mispredict_lock:
        if not _mispredict:
            return None
        bad_count = sum(1 for t in tri if _mispredict.get(t, 0) >= _MISPREDICT_THRESHOLD)
        # ≥40% 的 trigrams 来自已知误判模式 → 覆写
        if bad_count >= len(tri) * 0.4:
            return "medium"
    return None
_cache: dict = {}
_cache_lock = threading.Lock()
_cache_ttl = 3600


def _key(text: str) -> str:
    return hashlib.md5(text[:500].encode("utf-8")).hexdigest()


def _parse(resp_text: str) -> str | None:
    """从 Anthropic messages 响应里取分类标签, 容错。"""
    try:
        body = json.loads(resp_text)
    except Exception:
        return None
    for block in body.get("content", []) or []:
        if isinstance(block, dict) and block.get("type") == "text":
            t = (block.get("text") or "").strip().lower()
            for w in _VALID:
                if w in t:
                    return w
    # DeepSeek Flash 默认启用 thinking，max_tokens 不够时 text 块可能为空，
    # 但 thinking 块里或 stop_reason 里也可能带答案
    low = resp_text.lower()
    for w in _VALID:
        if w in low:
            return w
    return None


def cache_stats() -> dict:
    with _cache_lock:
        return {"size": len(_cache)}


def configure(cache_size: int | None = None) -> None:
    """热调整缓存容量(淘汰最老)。"""
    if cache_size is None:
        return
    with _cache_lock:
        while len(_cache) > cache_size:
            old = min(_cache, key=lambda k: _cache[k][1])
            del _cache[old]


async def classify(client: httpx.AsyncClient, settings: Settings, text: str) -> str | None:
    """返回 easy/medium/hard。失败/超时/禁用 → None(由调用方默认 easy, 安全网兜底)。

    注意: 经济闸门(min_input_tokens)由调用方在 controller 判断, 这里只做缓存 + 调用。
    """
    conf = settings.routing.classifier
    if not conf.get("enabled", True):
        return None
    if not text or len(text.strip()) < 3:
        return None

    k = _key(text)
    now = time.time()
    with _cache_lock:
        hit = _cache.get(k)
        if hit and now - hit[1] < _cache_ttl:
            return hit[0]

    be = settings.backends.get("deepseek")
    if not be or not be.key:
        return None
    model = be.models.get("flash") or be.models.get("default")
    timeout = conf.get("timeout", 1.5)
    resp = None
    try:
        resp = await client.post(
            be.url,
            json={
                "model": model,
                "max_tokens": 100,   # DeepSeek Flash 默认 thinking 会吃掉 ~50 token
                "temperature": 0,
                "messages": [{"role": "user", "content": _PROMPT + text[:800]}],
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {be.key}",
                "anthropic-version": "2023-06-01",
            },
            timeout=timeout,
        )
        if resp.status_code != 200:
            logger.debug(f"Classifier upstream {resp.status_code}, fallback")
            return None
        label = _parse(resp.text)
    except (httpx.TimeoutException, httpx.HTTPError) as e:
        logger.debug(f"Classifier timeout/error: {type(e).__name__}, fallback")
        return None
    except Exception as e:
        logger.warning(f"Classifier unexpected error: {e}")
        return None
    finally:
        if resp is not None:
            try:
                await resp.aclose()
            except Exception:
                pass

    if label:
        # v4.5: 检查是否是已知误判模式——分类器倾向于判 easy 但安全网频繁升级
        if label == "easy":
            override = check_easy_override(text)
            if override:
                logger.info(f"Classifier easy→{override} (mispredict feedback override)")
                label = override
        with _cache_lock:
            _cache[k] = (label, now)
            cap = conf.get("cache_size", 512)
            while len(_cache) > cap:
                old = min(_cache, key=lambda kk: _cache[kk][1])
                del _cache[old]
    return label
