"""FastAPI 装配 + 生命周期"""
import asyncio, logging, time, os, json
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import Response
import httpx
from .config import get_settings, Settings
from .server import handler
from . import state as state_mod

logger = logging.getLogger("SmartProxy")

HISTORY_SAVE_INTERVAL = 300  # 每 5 分钟持久化一次历史数据


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info(f"SmartProxy v4.2 — Starting on {settings.server.host}:{settings.server.port}")

    # 注入 YAML 配置到全局 LatencyTracker
    from .metrics import latency_tracker, similarity_buffer
    latency_tracker.configure(
        window=settings.latency.window,
        slow_ms=settings.latency.slow_threshold_ms,
        sticky_s=settings.latency.degraded_sticky_s,
    )

    # 加载历史数据 → 自动萃取关键词 → 动态扩展
    similarity_buffer.load()
    keywords = similarity_buffer.extract_keywords(min_samples=2)
    if keywords:
        from .router import extend_keywords
        extend_keywords(keywords)

    app.state.client = httpx.AsyncClient(
        timeout=httpx.Timeout(
            settings.timeouts.total,
            connect=settings.timeouts.connect,
            read=settings.timeouts.read,
            pool=settings.timeouts.pool,
        ),
        limits=httpx.Limits(
            max_connections=settings.pool.max_connections,
            max_keepalive_connections=settings.pool.max_keepalive_connections,
            keepalive_expiry=settings.pool.keepalive_expiry,
        ),
        proxy=settings.http_proxy,
    )
    probe = asyncio.create_task(_probe_loop(app.state.client, settings))
    # 定时保存历史
    persist_task = asyncio.create_task(_persist_loop(similarity_buffer))
    yield
    probe.cancel()
    persist_task.cancel()
    try:
        await probe
    except asyncio.CancelledError:
        pass
    try:
        await persist_task
    except asyncio.CancelledError:
        pass
    # 关闭前保存一次
    similarity_buffer.persist()
    await app.state.client.aclose()
    logger.info("SmartProxy — Shutdown complete")


async def _persist_loop(buf):
    """定时将历史数据持久化到磁盘。"""
    while True:
        try:
            await asyncio.sleep(HISTORY_SAVE_INTERVAL)
            buf.persist()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"History persist error: {e}")


async def _probe_loop(client, settings: Settings):
    """后台探活: 定期检查被熔断/overloaded 的后端是否恢复。"""
    while True:
        try:
            await asyncio.sleep(settings.circuit.probe_interval)
            for name, conf in settings.backends.items():
                blocked = state_mod.is_blocked(name)
                overloaded = state_mod.is_overloaded(name)
                if not blocked and not overloaded:
                    continue
                if blocked:
                    blocked_at = state_mod.blocked_at(name)
                    if not blocked_at or time.time() < blocked_at + settings.circuit.min_block_seconds:
                        continue
                await _probe_one(client, name, conf, settings)
        except Exception as e:
            logger.error(f"Probe error: {e}")


async def _probe_one(client, name, conf, settings):
    try:
        default_model = conf.models.get("default", list(conf.models.values())[0])
        resp = await client.post(
            conf.url,
            json={"model": default_model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {conf.key}",
                "anthropic-version": "2023-06-01",
            },
            timeout=settings.circuit.probe_timeout,
        )
        if resp.status_code == 200:
            state_mod.unblock(name)
            state_mod.clear_overloaded(name)
        await resp.aclose()
    except Exception:
        pass


app = FastAPI(lifespan=lifespan, title="SmartProxy")


# ── 鉴权中间件 ──────────────────────────────────────────
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    settings = get_settings()
    token = os.getenv(settings.security.auth_token_env, "")
    if token and request.url.path != "/health":
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or auth[7:] != token:
            return Response(content=json.dumps({"error": "Unauthorized"}),
                          status_code=401, media_type="application/json")
    return await call_next(request)


app.post("/v1/messages")(handler)


@app.get("/health")
async def health():
    """健康检查 + 熔断状态一览。"""
    settings = get_settings()
    backends_status = {}
    for name in settings.backends:
        blocked = state_mod.is_blocked(name)
        overloaded = state_mod.is_overloaded(name)
        from .metrics import latency_tracker, similarity_buffer
        backends_status[name] = {
            "blocked": blocked,
            "overloaded": overloaded,
            "latency": latency_tracker.stats(name),
        }
    from .metrics import similarity_buffer
    from .router import get_all_indicators
    kw = get_all_indicators()
    return {
        "status": "ok",
        "backends": backends_status,
        "similarity_buffer": similarity_buffer.stats(),
        "skip_flash_keywords": kw,
    }
