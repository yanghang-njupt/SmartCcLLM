"""请求编排 —— 三级难度路由 + 安全网降级

流程：
  1. route() 在发送前评估难度 → (easy | medium | hard)
  2. 按难度选择 tier 发送请求
  3. 成功 → 非流式检查安全网 → 返回
  4. 失败 → 按降级链依次尝试下一级
"""
import json, time, logging, traceback
import httpx
from fastapi import Response
from fastapi.responses import StreamingResponse
from .config import get_settings
from .router import route
from . import state
from . import circuit as circuit_mod
from .metrics import latency_tracker, similarity_buffer
from .cache import cache
from .logging_setup import request_id_ctx

logger = logging.getLogger("SmartProxy.Controller")


def _is_permanent_error(err_text):
    if not err_text:
        return False
    lower = err_text.lower()
    markers = [
        "usage limit", "quota", "billing cycle", "billing",
        "rate limit exceeded", "insufficient_quota",
        "exceeded your current quota", "upgrade your plan",
    ]
    return any(m in lower for m in markers)


def _strip_thinking_blocks(messages):
    cleaned = []
    for msg in messages:
        if not isinstance(msg, dict):
            cleaned.append(msg)
            continue
        content = msg.get("content")
        if isinstance(content, list):
            new_content = [
                block for block in content
                if not (isinstance(block, dict) and block.get("type") in ("thinking", "redacted_thinking"))
            ]
            new_msg = dict(msg)
            new_msg["content"] = new_content
            cleaned.append(new_msg)
        else:
            cleaned.append(msg)
    return cleaned


def _get_model(backend_conf, tier="default"):
    models = backend_conf.models
    return models.get(tier, models["default"])


def _check_inadequate(resp_body_bytes, latency_ms):
    """安全网：检查 flash 响应是否不足（仅非流式）。"""
    try:
        body = json.loads(resp_body_bytes)
        stop_reason = body.get("stop_reason", "")
        output_tokens = body.get("usage", {}).get("output_tokens", 0)
        if stop_reason == "max_tokens" and output_tokens < 100:
            return True, "truncated"
        if stop_reason == "tool_use":
            return True, "tool_use"
    except Exception:
        pass
    if latency_ms > 15000:
        return True, "slow"
    return False, ""


def _backend_ready(name):
    return not state.is_blocked(name) and not state.is_overloaded(name)


def _pick_best(settings):
    if "kimi" in settings.backends and _backend_ready("kimi"):
        return "kimi", "default"
    return "deepseek", "pro"


def _fallback_chain(difficulty: str, settings) -> list[tuple[str, str]]:
    """返回降级路径：失败时依次尝试。"""
    if difficulty == "easy":
        # flash → pro → kimi
        return [("deepseek", "pro"), _pick_best(settings)]
    if difficulty == "medium":
        # pro → kimi → flash
        return [_pick_best(settings), ("deepseek", "flash")]
    # hard → pro → flash
    return [("deepseek", "pro"), ("deepseek", "flash")]


async def _send_request(rid, candidate, tier, conf, body, req_headers,
                        settings, client, messages):
    """发送请求，返回 (response_or_none, success, error_info)."""
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
    else:
        resp_body = await resp.aread()
        resp_ct = resp.headers.get("content-type", "application/json") if hasattr(resp, 'headers') else "application/json"
        if settings.cache.enabled:
            cache.put(model, messages, resp_body, resp_ct)
        return Response(content=resp_body, status_code=200, media_type=resp_ct)


def _handle_error(rid, candidate_name, status_code, err_display, settings):
    """熔断 + 退避。返回 (candidate, status_code, err_display) 或 None（非重试错误）。"""
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


def _extract_text_from_body(body):
    text = ""
    try:
        msgs = body.get("messages", [])
        if msgs:
            last = msgs[-1]
            content = last.get("content", "")
            if isinstance(content, list):
                text = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
            else:
                text = str(content)
    except Exception:
        pass
    return str(text)


# ── 主入口 ───────────────────────────────────────────────

async def handle_request(body: dict, req_headers: dict, client: httpx.AsyncClient) -> Response:
    settings = get_settings()
    rid = request_id_ctx.get()
    messages = body.get("messages", [])
    is_stream = body.get("stream", False)

    score = route(settings, messages)
    text = _extract_text_from_body(body)

    logger.info(f"[{rid}] Route: {score.backend}/{score.tier} ({score.reason}) difficulty={score.difficulty}")

    # ── 缓存检查（仅非流式 flash） ─────────────────────
    if not is_stream and settings.cache.enabled and score.tier == "flash":
        model = _get_model(settings.backends["deepseek"], "flash")
        cached = cache.get(model, messages)
        if cached is not None:
            logger.info(f"[{rid}] Cache hit")
            return Response(content=cached[0], status_code=200, media_type=cached[1],
                          headers={"X-SP-Cache": "hit"})

    # ── 构造降级链 ─────────────────────────────────────
    attempts = [(score.backend, score.tier)]
    attempts += _fallback_chain(score.difficulty, settings)
    # 去重（相邻重复的去掉）
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

            # 记录历史
            similarity_buffer.record_outcome(text, candidate, tier, True, latency_ms, "stream" if is_stream else "")

            # 安全网：flash 非流式检查不足
            if candidate == "deepseek" and tier == "flash" and not is_stream:
                resp_body = await resp.aread()
                inadequate, reason = _check_inadequate(resp_body, latency_ms)
                if inadequate:
                    logger.info(f"[{rid}] Flash inadequate ({reason}) → upgrading")
                    similarity_buffer.record_outcome(text, candidate, tier, False, latency_ms, reason)
                    # 直接使用第一个降级目标
                    up_dest = _fallback_chain("easy", settings)
                    if up_dest:
                        up_candidate, up_tier = up_dest[0]
                        up_conf = settings.backends.get(up_candidate)
                        if up_conf and _backend_ready(up_candidate):
                            up_resp, up_ok, up_err = await _send_request(
                                rid, up_candidate, up_tier, up_conf, body, req_headers,
                                settings, client, messages)
                            if up_ok:
                                return await _return_response(up_resp, up_err[1], is_stream,
                                                             _get_model(up_conf, up_tier),
                                                             messages, up_candidate, settings)
                            _handle_error(rid, up_candidate, up_err[1], up_err[2], settings)
                    # 升级失败，回退到非流式缓存结果
                    if settings.cache.enabled:
                        cache.put(_get_model(conf, "flash"), messages, resp_body,
                                  resp.headers.get("content-type", "application/json") if hasattr(resp, 'headers') else "application/json")
                    return Response(content=resp_body, status_code=200,
                                  media_type=resp.headers.get("content-type", "application/json") if hasattr(resp, 'headers') else "application/json")

                # flash 正常 → 缓存 + 返回
                if settings.cache.enabled:
                    cache.put(_get_model(conf, "flash"), messages, resp_body,
                              resp.headers.get("content-type", "application/json") if hasattr(resp, 'headers') else "application/json")
                return Response(content=resp_body, status_code=200,
                              media_type=resp.headers.get("content-type", "application/json") if hasattr(resp, 'headers') else "application/json")

            return await _return_response(resp, latency_ms, is_stream,
                                         _get_model(conf, tier), messages, candidate, settings)

        # 失败
        _, status_code, err_display = err_info
        similarity_buffer.record_outcome(text, candidate, tier, False, 0.0, f"http_{status_code}")
        last_error = _handle_error(rid, candidate, status_code, err_display, settings)

    return Response(
        content=json.dumps({"error": "All backends failed"}).encode(),
        status_code=502, media_type="application/json")
