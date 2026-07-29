"""请求编排 —— 分层路由 + LLM 分类器 + 流式安全网 + 升级信号学习

流程:
  1. route() 预判难度(easy/medium/hard/uncertain/continuation)
  2. uncertain → 经济闸门(上下文太小则跳过) → flash 2-token 分类器 → 定档
  3. 按难度选 tier 发送
  4. flash 响应:
     - 非流式: 读 body 检查 inadequate → 升级 + record_upgrade
     - 流式:   转发同时后置抓 stop_reason/usage → record_upgrade/success
       (流式无法中途升级, 安全网在此作"学习信号采集器")
  5. 失败 → 降级链
"""
import json, time, logging, traceback
import httpx
from fastapi import Response
from fastapi.responses import StreamingResponse
from .config import get_settings, Settings
from .router import route, _pick_best, _sanitize
from . import state
from . import circuit as circuit_mod
from .metrics import latency_tracker, similarity_buffer, upgrade_store
from .cache import cache
from .logging_setup import request_id_ctx
from . import classifier

logger = logging.getLogger("SmartProxy.Controller")


# ── 工具 ─────────────────────────────────────────────────

def _is_permanent_error(err_text):
    if not err_text:
        return False
    lower = err_text.lower()
    markers = ["usage limit", "quota", "billing cycle", "billing",
               "rate limit exceeded", "insufficient_quota",
               "exceeded your current quota", "upgrade your plan"]
    return any(m in lower for m in markers)


def _strip_thinking_blocks(messages):
    cleaned = []
    for msg in messages:
        if not isinstance(msg, dict):
            cleaned.append(msg)
            continue
        content = msg.get("content")
        if isinstance(content, list):
            new_content = [b for b in content
                           if not (isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking"))]
            new_msg = dict(msg)
            new_msg["content"] = new_content
            cleaned.append(new_msg)
        else:
            cleaned.append(msg)
    return cleaned


def _get_model(backend_conf, tier="default"):
    models = backend_conf.models
    return models.get(tier, models["default"])


def _backend_ready(name):
    return not state.is_blocked(name) and not state.is_overloaded(name)


def _fallback_chain(difficulty: str, settings) -> list[tuple[str, str]]:
    """返回降级候选链（不含原始尝试），已在函数内去重。"""
    if difficulty == "medium":
        # 原始是 deepseek/pro，降级就是 flash
        return [("deepseek", "flash")]
    if difficulty == "hard":
        # 原始可能是 Kimi（可用时），降级先 pro 再 flash
        return [("deepseek", "pro"), ("deepseek", "flash")]
    # easy / uncertain / continuation → flash 失败 → 先 pro 再 Kimi
    return [("deepseek", "pro"), _pick_best(settings)]

def _extract_text_from_body(body):
    try:
        msgs = body.get("messages", [])
        if msgs:
            last = msgs[-1]
            content = last.get("content", "")
            if isinstance(content, list):
                return " ".join(b.get("text", "") for b in content if isinstance(b, dict))
            return str(content)
    except Exception:
        pass
    return ""


# ── 安全网判定 ───────────────────────────────────────────

def _is_truncated(stop_reason: str, output_tokens: int, requested_max_tokens: int = 0) -> bool:
    """判断是否因 max_tokens 导致输出被截断。
    排除短输出误判: 输出 ≥50% 请求上限时不认为截断。
    """
    if stop_reason != "max_tokens":
        return False
    if output_tokens <= 0:
        return False
    if requested_max_tokens > 0 and output_tokens >= requested_max_tokens * 0.5:
        return False
    return output_tokens < 100


def _check_inadequate(resp_body_bytes, latency_ms, requested_max_tokens: int = 0):
    """非流式: 检查 flash 响应是否不足。"""
    try:
        body = json.loads(resp_body_bytes)
        stop_reason = body.get("stop_reason", "")
        output_tokens = body.get("usage", {}).get("output_tokens", 0)
        if _is_truncated(stop_reason, output_tokens, requested_max_tokens):
            return True, "truncated"
    except Exception:
        pass
    if latency_ms > 15000:
        return True, "slow"
    return False, ""


def _check_inadequate_streaming(stop_reason: str, output_tokens: int, total_ms: float,
                                 requested_max_tokens: int = 0):
    """流式: 用后置抓到的 stop_reason/usage 判定(逻辑与非流式一致)。"""
    if _is_truncated(stop_reason, output_tokens, requested_max_tokens):
        return True, "truncated"
    if total_ms > 15000:
        return True, "slow"
    return False, ""


# ── uncertain → 分类器(经济闸门) ──────────────────────────

def _apply_difficulty(score, diff, settings):
    score.difficulty = diff
    if diff == "hard":
        be, t = _pick_best(settings)
        score.backend, score.tier = be, t
    elif diff == "medium":
        score.backend, score.tier = "deepseek", "pro"
    else:  # easy
        score.backend, score.tier = "deepseek", "flash"
    return score


async def _resolve_uncertain(score, settings, client, messages, text, rid):
    conf = settings.routing.classifier
    if not conf.get("enabled", True):
        return _apply_difficulty(score, "easy", settings)
    # 经济闸门: 检查 cleaned text 长度而非 messages 总量。
    # 分类器只看 text[:800]，成本恒定 ~200 token，与会话无关。
    # 只有用户真没打字（空文本/纯标点）才跳过分分类器。
    clean = text.strip()
    min_len = conf.get("min_input_chars", 3)
    if len(clean) < min_len:
        logger.info(f"[{rid}] Uncertain but empty text({len(clean)}ch) → easy, skip classifier")
        return _apply_difficulty(score, "easy", settings)
    label = await classifier.classify(client, settings, text)
    if label is None:
        logger.info(f"[{rid}] Classifier fallback → easy")
        label = "easy"
    else:
        logger.info(f"[{rid}] Classifier → {label}")
    return _apply_difficulty(score, label, settings)


# ── 请求发送 ─────────────────────────────────────────────

async def _send_request(rid, candidate, tier, conf, body, req_headers,
                        settings, client, messages):
    model = _get_model(conf, tier)
    req_body = {**body, "model": model}
    if candidate == "deepseek":
        req_body["messages"] = _strip_thinking_blocks(messages)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {conf.key}",
        "anthropic-version": req_headers.get("anthropic-version", "2023-06-01"),
    }
    beta = req_headers.get("anthropic-beta", "")
    if beta:
        headers["anthropic-beta"] = beta

    t_start = time.time()
    try:
        request = client.build_request("POST", conf.url, json=req_body, headers=headers)
        response = await client.send(request, stream=True)
        latency_ms = (time.time() - t_start) * 1000
        if response.status_code != 200:
            err_body = await response.aread()
            err_text = err_body.decode(errors='replace')
            max_trunc = settings.security.error_body_truncate
            err_display = err_text[:max_trunc] if len(err_text) > max_trunc else err_text
            logger.error(f"[{rid}] Error {candidate}/{model} {response.status_code}: {err_display}")
            latency_tracker.record(candidate, latency_ms, ok=False)
            await response.aclose()
            return None, False, (candidate, response.status_code, err_display)
        return response, True, (candidate, latency_ms, "")
    except Exception as e:
        latency_ms = (time.time() - t_start) * 1000
        latency_tracker.record(candidate, latency_ms, ok=False)
        err_text = str(e) or repr(e)
        logger.error(f"[{rid}] Relay Error ({candidate}): {type(e).__name__}: {err_text}\n{traceback.format_exc()}")
        return None, False, (candidate, 0, err_text)


async def _return_response(resp, latency_ms, is_stream, model, messages, backend, settings):
    latency_tracker.record(backend, latency_ms, ok=True)
    if is_stream:
        async def stream_wrapper(resp=resp):
            try:
                async for chunk in resp.aiter_raw():
                    yield chunk
            finally:
                try:
                    await resp.aclose()
                except Exception:
                    pass
        return StreamingResponse(stream_wrapper(), media_type="text/event-stream")
    resp_body = await resp.aread()
    resp_ct = resp.headers.get("content-type", "application/json") if hasattr(resp, 'headers') else "application/json"
    if settings.cache.enabled:
        cache.put(model, messages, resp_body, resp_ct)
    return Response(content=resp_body, status_code=200, media_type=resp_ct)


def _handle_error(rid, candidate_name, status_code, err_display, settings):
    latency_tracker.mark_degraded(candidate_name)
    if status_code == 429:
        state.mark_overloaded(candidate_name)
        state.block(candidate_name, circuit_mod.effective_block(settings.circuit, "quota", state.fail_count(candidate_name)))
        return (candidate_name, 429, err_display)
    if status_code >= 500 or status_code in (403, 503):
        fc = state.fail_count(candidate_name)
        if status_code == 403 and _is_permanent_error(err_display):
            state.block(candidate_name, settings.circuit.block_seconds_permanent)
        else:
            state.block(candidate_name, circuit_mod.effective_block(settings.circuit, "server", fc))
        return (candidate_name, status_code, err_display)
    if status_code == 0:
        state.block(candidate_name, circuit_mod.effective_block(settings.circuit, "network", state.fail_count(candidate_name)))
        return (candidate_name, 0, err_display)
    return None


# ── 流式安全网: 转发 + 后置抓 stop_reason/usage 喂学习库 ───

def _feed_learning(text: str, is_upgrade: bool, reason: str = "") -> None:
    """清洗文本后喂 upgrade_store。防止系统噪声污染学习数据。"""
    if not text:
        return
    clean, _ = _sanitize(text)
    if not clean or len(clean) < 3:
        return
    if is_upgrade:
        upgrade_store.record_upgrade(clean, reason)
    else:
        upgrade_store.record_success(clean)

async def _stream_with_learning(resp, text: str, settings, requested_max_tokens: int = 0):
    """转发流式响应(零修改), 同时解析 SSE 抓 stop_reason/usage,
    流结束后判定 flash 是否不足 → 喂 upgrade_store。

    v4.4: 捕获上游流式断连(并行 agent + Enter 触发连接复用冲突导致
    DeepSeek 侧 ReadError/RemoteProtocolError), 不冒泡到 uvicorn。
    客户端已收到 200 header + 部分内容, 断连后流正常结束。
    """
    stop_reason = ""
    output_tokens = 0
    buf = b""
    t_start = time.time()
    try:
        async for chunk in resp.aiter_raw():
            yield chunk
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == b"[DONE]":
                    continue
                try:
                    evt = json.loads(payload)
                except Exception:
                    continue
                if evt.get("type") == "message_delta":
                    d = evt.get("delta", {}) or {}
                    sr = d.get("stop_reason")
                    if sr:
                        stop_reason = sr
                    u = evt.get("usage", {}) or {}
                    if u.get("output_tokens"):
                        output_tokens = u["output_tokens"]
    except (httpx.ReadError, httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadTimeout) as e:
        # 上游流式中途断连: 并行 agent 连接复用冲突 / HTTP proxy 抖动
        # 客户端已收到 200 + 部分内容, 不回退不重试, 静默结束
        logger.warning(f"[stream] Upstream disconnected mid-stream: {type(e).__name__}")
    finally:
        total_ms = (time.time() - t_start) * 1000
        try:
            await resp.aclose()
        except Exception:
            pass
        if text:
            inadequate, reason = _check_inadequate_streaming(
                stop_reason, output_tokens, total_ms, requested_max_tokens)
            if inadequate:
                _feed_learning(text, True, reason or stop_reason or "stream_inadequate")
                # v4.5: 流式不足也反馈到分类器误判闭环
                if text:
                    from . import classifier as classifier_mod
                    classifier_mod.record_mispredict(text)
            else:
                _feed_learning(text, False)
        logger.debug(f"[stream-end] stop={stop_reason} out_tok={output_tokens} "
                     f"total={total_ms:.0f}ms inadequate={inadequate if text else 'n/a'}")


async def _upgrade_flash_response(rid, resp_body, resp_ct, text, settings, is_stream,
                                  body, req_headers, client, messages, conf):
    """非流式 flash 被判不足 → 升级到 pro 重发。"""
    _feed_learning(text, True, "nonstream_inadequate")
    # v4.5: 记录误判反馈——flash 升级说明之前的判定（关键词或分类器）偏轻了
    if text:
        from . import classifier as classifier_mod
        classifier_mod.record_mispredict(text)
    up_dest = _fallback_chain("easy", settings)
    if up_dest:
        up_candidate, up_tier = up_dest[0]
        up_conf = settings.backends.get(up_candidate)
        if up_conf and _backend_ready(up_candidate):
            up_resp, up_ok, up_err = await _send_request(
                rid, up_candidate, up_tier, up_conf, body, req_headers, settings, client, messages)
            if up_ok:
                return await _return_response(up_resp, up_err[1], is_stream,
                                              _get_model(up_conf, up_tier), messages, up_candidate, settings)
            _handle_error(rid, up_candidate, up_err[1], up_err[2], settings)
    # 升级失败 → 返回原 flash 结果
    if settings.cache.enabled:
        cache.put(_get_model(conf, "flash"), messages, resp_body, resp_ct)
    return Response(content=resp_body, status_code=200, media_type=resp_ct)


# ── 主入口 ───────────────────────────────────────────────

async def handle_request(body: dict, req_headers: dict, client: httpx.AsyncClient) -> Response:
    settings = get_settings()
    rid = request_id_ctx.get()
    messages = body.get("messages", [])
    is_stream = body.get("stream", False)
    requested_max_tokens = body.get("max_tokens", 0)

    score = route(settings, messages)
    text = score.clean_text or _extract_text_from_body(body)

    # uncertain → 经济闸门 + LLM 分类器
    if score.difficulty == "uncertain":
        score = await _resolve_uncertain(score, settings, client, messages, text, rid)

    logger.info(f"[{rid}] Route: {score.backend}/{score.tier} ({score.reason}) difficulty={score.difficulty}")

    # 会话粘性记录(成功后会刷新)
    from .router import _set_sticky

    # 缓存检查(仅非流式 flash)
    if not is_stream and settings.cache.enabled and score.tier == "flash":
        model = _get_model(settings.backends["deepseek"], "flash")
        cached = cache.get(model, messages)
        if cached is not None:
            logger.info(f"[{rid}] Cache hit")
            return Response(content=cached[0], status_code=200, media_type=cached[1],
                            headers={"X-SP-Cache": "hit"})

    # 降级链
    attempts = [(score.backend, score.tier)] + _fallback_chain(score.difficulty, settings)
    unique = [attempts[0]]
    for a in attempts[1:]:
        if a != unique[-1]:
            unique.append(a)

    last_error = None
    for candidate, tier in unique:
        conf = settings.backends.get(candidate)
        if not conf or not _backend_ready(candidate):
            continue

        resp, success, err_info = await _send_request(
            rid, candidate, tier, conf, body, req_headers, settings, client, messages)

        if success:
            latency_ms = err_info[1]
            similarity_buffer.record_outcome(text, candidate, tier, True, latency_ms,
                                              "stream" if is_stream else "")

            # flash 特殊处理: 安全网 + 学习
            if candidate == "deepseek" and tier == "flash":
                if is_stream:
                    # 流式: 转发 + 后置学习(无法中途升级)
                    latency_tracker.record(candidate, latency_ms, ok=True)
                    _set_sticky(settings, messages, candidate, tier)
                    return StreamingResponse(_stream_with_learning(resp, text, settings, requested_max_tokens),
                                             media_type="text/event-stream")
                # 非流式: 读 body 检查
                resp_body = await resp.aread()
                resp_ct = resp.headers.get("content-type", "application/json") if hasattr(resp, 'headers') else "application/json"
                inadequate, reason = _check_inadequate(resp_body, latency_ms, requested_max_tokens)
                if inadequate:
                    logger.info(f"[{rid}] Flash inadequate ({reason}) → upgrading")
                    similarity_buffer.record_outcome(text, candidate, tier, False, latency_ms, reason)
                    return await _upgrade_flash_response(rid, resp_body, resp_ct, text, settings,
                                                         is_stream, body, req_headers, client, messages, conf)
                # flash 正常
                _feed_learning(text, False)
                _set_sticky(settings, messages, candidate, tier)
                latency_tracker.record(candidate, latency_ms, ok=True)
                if settings.cache.enabled:
                    cache.put(_get_model(conf, "flash"), messages, resp_body, resp_ct)
                return Response(content=resp_body, status_code=200, media_type=resp_ct)

            # 非 flash: 正常返回(不喂 flash 学习库)
            _set_sticky(settings, messages, candidate, tier)
            return await _return_response(resp, latency_ms, is_stream,
                                          _get_model(conf, tier), messages, candidate, settings)

        # 失败
        _, status_code, err_display = err_info
        similarity_buffer.record_outcome(text, candidate, tier, False, 0.0, f"http_{status_code}")
        last_error = _handle_error(rid, candidate, status_code, err_display, settings)

    return Response(
        content=json.dumps({"error": "All backends failed"}).encode(),
        status_code=502, media_type="application/json")
