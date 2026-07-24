"""入口: python -m smart_proxy"""
from .config import get_settings
from .logging_setup import setup_logging


def main():
    settings = get_settings()
    setup_logging(level=settings.log_level)
    import uvicorn
    uvicorn.run(
        "smart_proxy.app:app",
        host=settings.server.host,
        port=settings.server.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
