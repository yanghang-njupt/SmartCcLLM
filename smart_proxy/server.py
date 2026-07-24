"""FastAPI handler + 鉴权 + Body 校验 —— 薄层，编排委托给 controller"""
import json, uuid
from fastapi import Request, Response
from fastapi.responses import StreamingResponse
from .config import get_settings
from .controller import handle_request
from .logging_setup import request_id_ctx


async def handler(req: Request):
    settings = get_settings()
    rid = uuid.uuid4().hex[:8]
    request_id_ctx.set(rid)

    # ── Body 大小检查 ───────────────────────────────
    content_length = req.headers.get("content-length")
    if content_length and int(content_length) > settings.security.max_body_bytes:
        return Response(content=json.dumps({"error": "Request body too large"}), status_code=413, media_type="application/json")

    # ── Body 解析 ────────────────────────────────────
    try:
        body = await req.json()
    except Exception:
        return Response(content=json.dumps({"error": "Invalid JSON body"}), status_code=400, media_type="application/json")
    if not isinstance(body, dict):
        return Response(content=json.dumps({"error": "Body must be a JSON object"}), status_code=400, media_type="application/json")

    # ── 委托给 controller ────────────────────────────
    return await handle_request(body, req.headers, req.app.state.client)
