"""结构化日志 + request_id 上下文"""
import logging, json, contextvars
from pathlib import Path

# 上下文变量：proxy.py 在每次请求入口设置，日志自动携带
request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")


class RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id_ctx.get()
        return True


def setup_logging(level="INFO"):
    """配置 JSON Lines 结构化日志（stdout + 文件双写）。"""
    logger = logging.getLogger("SmartProxy")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if logger.handlers:
        return logger

    fmt = _StructuredFormatter()
    flt = RequestIdFilter()

    # stdout
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.addFilter(flt)
    logger.addHandler(sh)

    # file —— 写入 logs/smart_proxy.log，供分析脚本消费
    log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(str(log_dir / "smart_proxy.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    fh.addFilter(flt)
    logger.addHandler(fh)

    return logger


class _StructuredFormatter(logging.Formatter):
    def format(self, record):
        base = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        rid = getattr(record, "request_id", "")
        if rid:
            base["request_id"] = rid
        return json.dumps(base, ensure_ascii=False)
