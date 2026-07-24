<p align="center">
  <img src="https://img.shields.io/badge/version-4.0.0-blue" alt="version 4.0.0">
  <img src="https://img.shields.io/badge/python-3.10+-green" alt="python 3.10+">
  <img src="https://img.shields.io/badge/license-MIT-orange" alt="license MIT">
</p>

# SmartProxy — 智能 LLM 请求分流代理

> **Flash-First 策略：简单任务用便宜模型，复杂任务自动升级到强模型。省钱又省心。**

## 解决的问题

调用 LLM API 时，每次都选最强模型（如 deepseek-v4-pro、kimi-k2.7）成本高，每次都选最便宜的 flash 模型质量又没保障。

SmartProxy 位于你的客户端与 API 提供商之间，**自动判断每个请求的难度**：

- 日常对话、简单修改 → **deepseek-v4-flash**（省 80%+ 成本）
- 重构、方案设计、安全检查 → **跳过 flash，直接 kimi/pro**（保障质量）
- flash 响应不足（截断/太慢） → **自动升级到更强模型**
- kimi 配额耗尽 → **无缝降级到 deepseek pro**，零中断

## 核心特性

| 特性 | 说明 |
|------|------|
| ⚡ **Flash-First 分流** | 简单任务 flash，复杂任务自动升级 |
| 🔀 **双路由模式** | 智能分流 / 直连，cc-switch 热切换 |
| 🛡️ **断路器自动熔断** | 4 级熔断 + 指数退避 + 探活恢复 |
| 🧠 **历史自学习** | 记录请求结果，自动扩展关键词，越用越准 |
| 📊 **可视化报告** | 一键生成 HTML 分析报告 |
| 🔌 **即插即用** | 兼容 OpenAI / Anthropic 格式 API |

## 架构总览

```
Claude Code / 任意 LLM 客户端
    │
    ▼
cc-switch（供应商管理）
    │
    ├── 智能分流 ──→ SmartProxy(:8000)
    │                       │
    │                 ┌─────┴─────┐
    │                 │   Router  │  ← 关键词 / token 评估 / 历史
    │                 └─────┬─────┘
    │                 ┌─────┴─────┐
    │                 │ Controller │  ← flash-first + 升级编排
    │                 └─────┬─────┘
    │                 ┌─────┴─────┐
    │                 │  后端 API  │  ← deepseek / kimi
    │                 └───────────┘
    │
    └── 直连 ──→ api.deepseek.com（跳过分流层）
```

## 快速开始

### 1. 配置 API Key

复制模板并填入你的密钥：

```bash
cp .env.example .env
```

```ini
DEEPSEEK_API_KEY=sk-your-deepseek-key    # 必填
MOONSHOT_API_KEY=sk-your-kimi-key        # 可选
```

### 2. 启动

**以管理员身份运行** `start_proxy.bat`。

### 3. 验证

```bash
curl http://127.0.0.1:8000/health
```

看到 `{"status": "ok"}` 即运行正常。

### 4. 配置客户端

在 cc-switch 中将供应商 URL 设为 `http://127.0.0.1:8000/v1/messages`，即可启用智能分流。

## 分流效果示例

| 请求类型 | 路由决策 | 节省成本 |
|----------|---------|----------|
| "帮我改个变量名" | ⚡ flash 直接返回 | ~90% |
| "查看项目架构" | ⚡ flash 直接返回 | ~90% |
| "帮我重构这个模块" | ⏭️ 跳过 flash → pro | 保障质量 |
| "设计一个支付系统" | ⏭️ 跳过 flash → pro/kimi | 保障质量 |
| 3000 行上下文的重写请求 | ⏭️ 超长上下文 → pro | 保障质量 |
| flash 响应超 15s 或截断 | ⬆️ 自动升级 | 自动 |

## 生成可视化报告

```bash
python analyze_routing.py                  # 今天的数据
python analyze_routing.py 2026-07-24       # 指定日期
```

在浏览器打开 `routing_report_2026-07-24.html` 查看分流统计。

## 项目结构

```
smart-cllm/
├── smart_proxy/              # ★ 核心源码
│   ├── app.py                #   FastAPI 装配 + 自学习生命周期
│   ├── controller.py         #   请求编排（flash-first 升级逻辑）
│   ├── router.py             #   路由引擎（关键词 + 历史评估）
│   ├── metrics.py            #   延迟追踪 + 历史缓冲 + 关键词萃取
│   ├── config.py             #   YAML 热加载配置
│   ├── server.py             #   请求 handler + 鉴权
│   ├── state.py              #   熔断/过载 TTL 管理
│   ├── circuit.py            #   指数退避计算
│   ├── cache.py              #   LRU 响应缓存
│   └── logging_setup.py      #   结构化日志
├── proxy_config.yaml         # 路由策略配置
├── analyze_routing.py        # 日志分析脚本
├── start_proxy.bat           # Windows 启动脚本
├── .env.example              # API Key 配置模板
├── 使用指南.md                # 使用文档
└── 架构设计.md                # 架构设计文档
```

## 配置参考

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `routing.flash_first` | `true` | 启用 flash-first 策略 |
| `circuit.block_seconds_permanent` | 18000 | 配额耗尽熔断（5h） |
| `latency.slow_threshold_ms` | 8000 | flash 慢查询阈值 |
| `cache.ttl_seconds` | 300 | 响应缓存 TTL |
| 关键词列表 | `router.py` | 静态 + 自学习动态扩展 |

## 技术栈

- **Python 3.10+** — 异步 runtime
- **FastAPI + uvicorn** — HTTP 服务
- **httpx** — 异步 HTTP 客户端
- **Pydantic v2** — 配置验证
- **Chart.js** — 可视化报告

## License

MIT
