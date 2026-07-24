"""兼容 re-export —— handler 已迁移到 server.py，编排逻辑在 controller.py"""
from .server import handler
from .controller import handle_request
