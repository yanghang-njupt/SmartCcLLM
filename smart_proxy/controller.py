"""请求编排 —— Flash-First + 智能升级

策略：
  flash_first → 先发 flash，成功则返回，失败/截断则升级到 kimi（或 pro）
  skip_flash → 跳过 flash，直接发 kimi（或 pro）
"""
import json, time, logging, traceback, uuid
import httpx
from fastapi import Response
from fastapi.responses import StreamingResponse
from .config import get_settings
from .router import route, RoutingScore
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


def _check_flash_inadequate(resp_body_bytes, latency_ms):
    """检查 flash 响应是否不足，需要升级。"""
    try:
        body = json.loads(resp_body_bytes)
        stop_reason = body.get("stop_reason", "")
        usage = body.get("usage", {})
        output_tokens = usage.get("output_tokens", 0)
        if stop_reason == "max_tokens" and output_tokens < 100:
            return True, "flash_truncated"
        if stop_reason == "tool_use":
            return True, "flash_tool_use"
    except Exception:
        pass
    if latency_ms > 15000:
        return True, "flash_slow"
    return False, ""


def _backend_ready(name):
    """检查后端是否可发起请求（未被熔断 / 未被 overloaded）。"""
    return not state.is_blocked(name) and not state.is_overloaded(name)


async def _send_request(rid, candidate, tier, conf, body, req_headers,
                        settings, client, messages):
    """发送请求，返回 (response_or_none, success, error_info)。"""
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
            logger.error(f"[{rid}] Upstream Error {candidate}/{model} {response.status_code}: {err_display}")
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
    """处理成功响应并返回。"""
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


async def _handle_error(rid, candidate_name, status_code, err_display, settings):
    """处理错误：熔断 + 退避。返回 (error_info_or_none_meaning_non_retryable)。"""
    latency_tracker.mark_degraded(candidate_name)
    if status_code == 429:
        state.mark_overloaded(candidate_name)
        fc = state.fail_count(candidate_name)
        state.block(candidate_name, circuit_mod.effective_block(settings.circuit, "quota", fc))
        return (candidate_name, 429, err_display)
    if status_code >= 500 or status_code in (403, 503):
        fc = state.fail_count(candidate_name)
        if status_code == 403 and _is_permanent_error(err_display):
            logger.warning(f"[{rid}] {candidate_name} permanent error (quota/billing) "
                         f"— block {settings.circuit.block_seconds_permanent // 3600}h")
            state.block(candidate_name, settings.circuit.block_seconds_permanent)
        else:
            state.block(candidate_name, circuit_mod.effective_block(settings.circuit, "server", fc))
        return (candidate_name, status_code, err_display)
    if status_code == 0:
        fc = state.fail_count(candidate_name)
        state.block(candidate_name, circuit_mod.effective_block(settings.circuit, "network", fc))
        return (candidate_name, 0, err_display)
    # 4xx non-retryable → 不在这里处理，由调用者返回
    return None


# ── 主入口 ───────────────────────────────────────────────

async def handle_request(body: dict, req_headers: dict, client: httpx.AsyncClient) -> Response:
    settings = get_settings()
    rid = request_id_ctx.get()
    messages = body.get("messages", [])
    is_stream = body.get("stream", False)

    score = route(settings, messages)

    # 跳过 flash：kimi 优先，不可用则 pro
    if score.strategy == "skip_flash":
        conf = settings.backends.get(score.backend)
        if conf and _backend_ready(score.backend):
            model = _get_model(conf, score.tier)
            logger.info(f"[{rid}] Route: {score.backend}/{model} ({score.reason}) strategy=skip_flash")
            resp, success, err_info = await _send_request(
                rid, score.backend, score.tier, conf, body, req_headers,
                settings, client, messages)
            if success:
                return await _return_response(resp, err_info[1], is_stream, model, messages, score.backend, settings)
            candidate_name, status_code, err_display = err_info
            last_error = await _handle_error(rid, candidate_name, status_code, err_display, settings)
            if last_error is None:
                return Response(content=json.dumps({"error": err_display}).encode(),
                              status_code=status_code, media_type="application/json")
            logger.info(f"[{rid}] skip_flash backend failed → trying flash")
        # fallthrough: 发 flash
        conf = settings.backends["deepseek"]
        resp, success, err_info = await _send_request(
            rid, "deepseek", "flash", conf, body, req_headers,
            settings, client, messages)
        if success:
            return await _return_response(resp, err_info[1], is_stream, _get_model(conf, "flash"), messages, "deepseek", settings)
        return Response(json.dumps({"error": "All backends failed"}).encode(), 502, media_type="application/json")

    # ── 策略: flash_first ──────────────────────────────
    conf = settings.backends["deepseek"]

    # 缓存
    if not is_stream and settings.cache.enabled:
        model = _get_model(conf, "flash")
        cached = cache.get(model, messages)
        if cached is not None:
            cached_body, cached_ct = cached
            logger.info(f"[{rid}] Cache hit for deepseek/{model}")
            return Response(content=cached_body, status_code=200, media_type=cached_ct,
                          headers={"X-SP-Cache": "hit"})

    logger.info(f"[{rid}] FlashFirst: deepseek/{_get_model(conf, 'flash')} ({score.reason}) "
                f"upgrade={score.upgrade_backend}/{score.upgrade_tier}")

    # 发 flash
    resp, success, err_info = await _send_request(
        rid, "deepseek", "flash", conf, body, req_headers,
        settings, client, messages)

    if success:
        latency_ms = err_info[1]

        if not is_stream:
            resp_body = await resp.aread()
            # 记录到 SimilarityBuffer
            text = _extract_text_from_body(body)
            try:
                d = json.loads(resp_body)
                sr = d.get("stop_reason", "")
            except:
                sr = ""
            similarity_buffer.record_outcome(
                text, "deepseek", "flash", success=True,
                latency_ms=latency_ms, stop_reason=sr)

            # 检查是否需要升级
            upgrade, reason = _check_flash_inadequate(resp_body, latency_ms)
            if upgrade:
                logger.info(f"[{rid}] Flash inadequate ({reason}) → upgrading to {score.upgrade_backend}")
                similarity_buffer.record_outcome(
                    text, "deepseek", "flash", success=False,
                    latency_ms=latency_ms, stop_reason=reason)

                # 升级到 kimi（或 pro）
                up_conf = settings.backends.get(score.upgrade_backend)
                if up_conf:
                    up_resp, up_ok, up_err = await _send_request(
                        rid, score.upgrade_backend, score.upgrade_tier, up_conf,
                        body, req_headers, settings, client, messages)
                    if up_ok:
                        similarity_buffer.record_outcome(
                            text, score.upgrade_backend, score.upgrade_tier,
                            success=True, latency_ms=up_err[1], stop_reason="")
                        return await _return_response(
                            up_resp, up_err[1], is_stream,
                            _get_model(up_conf, score.upgrade_tier),
                            messages, score.upgrade_backend, settings)
                    # 升级失败 → 处理错误
                    candidate_name, status_code, err_display = up_err
                    last_error = await _handle_error(rid, candidate_name, status_code, err_display, settings)
                    if last_error is None:
                        return Response(content=json.dumps({"error": err_display}).encode(),
                                      status_code=status_code, media_type="application/json")
                    # 升级后端也失败了 → 最终降级到任何可用
                    fb_name, fb_tier = "deepseek", "flash"
                    for name in settings.backends:
                        if name != score.upgrade_backend and name != "deepseek":
                            if _backend_ready(name):
                                fb_name, fb_tier = name, "default"
                                break
                    fb_conf = settings.backends[fb_name]
                    logger.info(f"[{rid}] Ultimate fallback to {fb_name}")
                    fb_resp, fb_ok, fb_err = await _send_request(
                        rid, fb_name, fb_tier, fb_conf, body, req_headers,
                        settings, client, messages)
                    if fb_ok:
                        return await _return_response(fb_resp, fb_err[1], is_stream,
                                                     _get_model(fb_conf, fb_tier),
                                                     messages, fb_name, settings)
                    return Response(
                        content=json.dumps({"error": "All backends failed"}).encode(),
                        status_code=502, media_type="application/json")

            # flash 成功 → 直接返回
            resp_ct = resp.headers.get("content-type", "application/json") if hasattr(resp, 'headers') else "application/json"
            if settings.cache.enabled:
                cache.put(_get_model(conf, "flash"), messages, resp_body, resp_ct)
            return Response(content=resp_body, status_code=200, media_type=resp_ct)

        # 流式：flash 直接返回（注：流式无法检查 stop_reason，依赖历史数据规避）
        text = _extract_text_from_body(body)
        similarity_buffer.record_outcome(
            text, "deepseek", "flash", success=True,
            latency_ms=latency_ms, stop_reason="stream")
        return await _return_response(resp, latency_ms, is_stream, _get_model(conf, "flash"), messages, "deepseek", settings)

    # flash 失败 → 升级
    candidate_name, status_code, err_display = err_info
    text = _extract_text_from_body(body)
    similarity_buffer.record_outcome(
        text, "deepseek", "flash", success=False,
        latency_ms=0, stop_reason=f"http_{status_code}")

    logger.info(f"[{rid}] Flash failed ({status_code}) → upgrading to {score.upgrade_backend}")

    up_conf = settings.backends.get(score.upgrade_backend)
    if up_conf:
        up_resp, up_ok, up_err = await _send_request(
            rid, score.upgrade_backend, score.upgrade_tier, up_conf,
            body, req_headers, settings, client, messages)
        if up_ok:
            return await _return_response(up_resp, up_err[1], is_stream,
                                         _get_model(up_conf, score.upgrade_tier),
                                         messages, score.upgrade_backend, settings)
        # 升级也失败 → 最终降级
        _, up_status, up_display = up_err
        await _handle_error(rid, score.upgrade_backend, up_status, up_display, settings)
        # 尝试 deepseek pro 作为最后手段
        pro_resp, pro_ok, pro_err = await _send_request(
            rid, "deepseek", "pro", conf, body, req_headers,
            settings, client, messages)
        if pro_ok:
            pro_lat = pro_err[1]
            latency_tracker.record("deepseek", pro_lat, ok=True)
            return await _return_response(pro_resp, pro_lat, is_stream,
                                         _get_model(conf, "pro"),
                                         messages, "deepseek", settings)

    return Response(
        content=json.dumps({"error": "All backends failed"}).encode(),
        status_code=502, media_type="application/json")


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
