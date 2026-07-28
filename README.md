# SmartProxy — 智能 LLM 请求分流代理

> **一句话：自动判断每个请求的难度，简单问题用便宜模型省钱，复杂问题自动升级到强模型保质量。**

版本 4.4 · Python 3.10+ · MIT

---

## 目录

1. [它做什么](#1-它做什么)
2. [架构总览](#2-架构总览)
3. [核心链路详解](#3-核心链路详解)
4. [快速开始](#4-快速开始)
5. [文件清单：哪些上传了、哪些没上传](#5-文件清单哪些上传了哪些没上传)
6. [你需要自己准备什么](#6-你需要自己准备什么)
7. [如何优化关键词库和评分](#7-如何优化关键词库和评分)
8. [配置参数参考](#8-配置参数参考)
9. [日常运维](#9-日常运维)
10. [故障排查](#10-故障排查)

---

## 1. 它做什么

你用 Claude Code 写代码时，每次请求都会经过 SmartProxy。SmartProxy 会**在请求发出之前**判断这个任务的难度，然后决定用哪个模型：

| 难度 | 条件 | 用的模型 | 为什么 |
|------|------|----------|--------|
| 🟢 **简单** | 日常对话、小修改 | `deepseek-v4-flash` | 便宜，速度够 |
| 🟡 **中等** | 设计方案、代码审查 | `deepseek-v4-pro` | 需要更强的理解能力 |
| 🔴 **困难** | 重构、安全审计、分布式系统 | `kimi-k2.7`（或降级 pro） | 需要最强的模型 |

**省钱的逻辑**：flash 的价格大约是 pro 的 1/10，大部分日常请求走 flash 就够了。

**保质量的逻辑**：当请求涉及复杂工程任务时，直接跳过 flash，避免 flash 答不好还要重来。

---

## 2. 架构总览

### 2.1 整体拓扑

```
你（Claude Code 客户端）
    │
    ▼
cc-switch（桌面应用，管理供应商切换）
    │
    ├── 智能分流模式 ──→ SmartProxy (:8000) ──→ deepseek / kimi API
    │
    └── 直连模式 ──→ api.deepseek.com（跳过分流层）
```

你可以随时在 cc-switch 里热切换这两种模式，SmartProxy 不需要重启。

### 2.2 SmartProxy 内部结构

SmartProxy 本身是一个 FastAPI 服务，内部有 13 个模块：

```
请求进来（POST /v1/messages）
    │
    ▼
┌─────────────────────────────────────────────┐
│  server.py   鉴权 + Body 校验               │  薄层，不做业务逻辑
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│  router.py   分层难度评估                    │  ★ 核心决策层
│                                              │
│  ⓪ 请求清洗: 剥离 <system-reminder>/        │
│     <transcript> 噪声, 识别续传 vs 新问题     │
│  ① 启发式预筛(<1ms, 零成本): 小核心关键词    │
│     + 结构信号 → (score, confidence)         │
│     - 高置信 easy/hard → 直接路由            │
│     - 低置信 → difficulty=uncertain          │
└──────────┬──────────────────────┬───────────┘
           │ uncertain             │
           ▼                      │
┌──────────────────────────┐     │
│  classifier.py  LLM 分类器│ ◀───┘  仅低置信请求调用
│  flash · max_tokens=2     │  只看任务描述(~200 token)
│  缓存 · 经济闸门          │  → easy/medium/hard
└──────────┬───────────────┘
           ▼
┌─────────────────────────────────────────────┐
│  controller.py   请求编排                    │  ★ 核心执行层
│                                              │
│  发请求 → flash 成功？→ 安全网判定           │
│            · 非流式: 读 body 检查 → 升级     │
│            · 流式: 转发同时抓 stop_reason     │
│              喂学习库(无法中途升级)           │
│         → 失败？→ 熔断 + 降级重试             │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│  metrics.py   延迟追踪 + 升级信号学习         │  ★ 自适应层
│   · UpgradeStore: TF-DF 提取真实领域词       │
│   · SimilarityBuffer: 相似度历史             │
│  cache.py     响应缓存                       │
│  state.py     熔断状态管理                    │
│  circuit.py   指数退避计算                    │
└─────────────────────────────────────────────┘
```

### 2.3 各模块职责

| 文件 | 一句话职责 |
|------|-----------|
| `__main__.py` | 启动入口，调 uvicorn |
| `app.py` | FastAPI 装配、生命周期管理、后台探活、学习库持久化 |
| `server.py` | 接收 HTTP 请求、鉴权、Body 解析 |
| `router.py` | **请求清洗 + 信号密度评分**（去重、密度、边界），决定走 flash / pro / kimi |
| `classifier.py` | LLM 分类器（flash 2-token），默认关闭，有需在 YAML 开 |
| `controller.py` | **执行路由决策**，分类器集成、流式安全网、升级降级重试 |
| `config.py` | 读 YAML 配置，支持热加载 |
| `state.py` | 熔断/过载状态的 JSON 持久化 |
| `circuit.py` | 熔断冷却时长的指数退避计算 |
| `metrics.py` | 延迟追踪 + **UpgradeStore (jieba 词性标注+五层过滤)** + 相似度缓冲 |
| `cache.py` | LRU 响应缓存（仅非流式） |
| `logging_setup.py` | JSON 结构化日志 |

---

## 3. 核心链路详解

### 3.1 分层难度评估（router.py + classifier.py）

v4.4 起改为**分层路由 + 信号密度 + 系统噪声清洗**。每次请求依次经过：

```
最后一条 user 消息文本
    │
    ▼
⓪ 请求清洗 _sanitize()
    ├── 剥离 <system-reminder>/<transcript>/<function_results> 噪声
    │   (修 v4.2/v4.3 的假 hard 问题: 无闭合 <system-reminder 的残渣内容
    │    含 distributed/architecture → 直接判续传 flash, 不分类)
    └── 剥离后几乎为空 → 续传(tool_result 回传) → 直接 flash, 不分类
    │
    ▼
① 启发式预筛 _heuristic_score() (<1ms, 零成本)
    ├── 词边界匹配 + 去重计数(长词覆盖短词, 防 "修"+"修改" 双倍)
    ├── CJK 结尾关键词右边界天然成立(防 "看一下config" 漏检)
    ├── 数字+CJK 交界判边界("Vue2迁移" 能匹配 "迁移")
    │
    ├── 信号密度: easy ≥ 2× edit → 降级(修改变量+看看+找找 → easy)
    │   · EASY_VERBS(ls/grep/cat/find/read/check/look/search/查看/看看/看…)
    │   · EDIT_VERBS(fix/implement/refactor/rewrite/debug/编写/修改/重构/修复/修…)
    │   · HARD_CORE(架构/设计/迁移/migrate/distributed/分布式/审计/kubernetes/微服务/集群/k8s…)
    ├── 从 UpgradeStore 实时注入的领域词(+12/个)
    ├── 超长上下文(>100k tokens) → +20
    └── 历史升级风险(UpgradeStore trigram 相似) → +20
    │
    ├── 高置信 easy  → flash
    ├── 高置信 hard  → kimi/pro
    └── 低置信       → uncertain (分类器兜底, 默认关闭)
    │
    ▼
最终难度 → 后端映射
    easy   → deepseek-v4-flash
    medium → deepseek-v4-pro
    hard   → kimi-k2.7(不可用则 pro)
```

**为什么这样设计**（经济账）：分类器只看任务描述，不看完整上下文，故成本恒定 ~200 token。判错代价（一次完整上下文双读/错档）远大于 200 token，所以对非极小请求稳赚；极小请求由经济闸门直接走 flash。

**关键词库**（写在 `router.py`，已大幅精简为高置信核心集）：

- `EASY_VERBS`：ls/grep/cat/find/read/list/show/open/browse/查看/列出/查找…（18 个）
- `EDIT_VERBS`：fix/implement/refactor/rewrite/debug/编写/修改/重构/修复/重写（10 个）
- `HARD_CORE`：架构/设计/迁移/migrate/distributed/分布式/审计/oauth/end-to-end/kafka…（12 个）
- 学到的词（`_extra_keywords`）：由 UpgradeStore 自动注入，无需手改代码

### 3.2 请求执行与升级链路（controller.py）

路由决策之后，controller 负责执行。核心是 **flash-first + 安全网**：

```
路由结果
    │
    ├── easy（flash）
    │     ├── 先查缓存 → 命中直接返回
    │     ├── 发 flash 请求
    │     │     ├── 成功 + 流式 → 转发, 同时后置抓 stop_reason/usage
    │     │     │     (流式无法中途升级; 安全网作"学习信号采集器")
    │     │     │     ├── 被截断(max_tokens + output<100) → record_upgrade
    │     │     │     ├── 太慢(>15s)                      → record_upgrade
    │     │     │     └── 正常 → record_success + 返回
    │     │     ├── 成功 + 非流式 → 检查响应质量
    │     │     │     ├── 被截断 / 太慢 → 升级到 pro + record_upgrade
    │     │     │     └── 正常 → record_success + 写入缓存
    │     │     └── 失败 → 熔断 flash + 升级到 pro/kimi
    │     └── 升级后还失败 → 尝试所有可用后端 → 502
    │
    ├── medium（pro）
    │     └── 降级链：pro → kimi → flash → 502
    │
    └── hard（kimi/pro）
          └── 降级链：kimi → pro → flash → 502
```

**v4.4 安全网**：
- 流式请求后置解析 SSE `message_delta` 抓 `stop_reason`/`usage`（Claude Code 全流式）。
- `stop_reason=tool_use` **不再**当"不足"（agentic 流正常会调工具）。
- 流式下只**采集学习信号**；当场补救靠前置分类器（默认关闭，有需再开）。
- 非流式 flash 不足 → 立即升级 pro/kimi 重发。

### 3.3 熔断系统

当后端返回错误时，系统会自动熔断该后端，避免反复重试浪费资源：

| 错误类型 | 触发条件 | 冷却时间 | 说明 |
|----------|----------|----------|------|
| 配额耗尽 | 429 / 403 + billing 关键词 | 1 小时 | 指数退避 ×2 |
| 服务端错误 | 5xx / 403（非 billing） | 5 分钟 | 指数退避 ×2 |
| 网络错误 | 连接超时/断开 | 1 分钟 | 指数退避 ×2 |
| 永久错误 | 403 + 配额关键词 | 5 小时 | 一次性，不退避 |

**指数退避**：同一个后端连续失败次数越多，冷却时间越长。比如网络错误：60s → 120s → 240s → ... → 最大 7200s。

**自动恢复**：后台每 30 秒对熔断的后端发一条探测请求，成功就自动解除熔断。

### 3.4 升级信号自学习（metrics.py）

v4.4 用 **jieba 词性标注 + 五层过滤** 替代了 v4.2 的 n-gram 萃取和 v4.3 的手写 TF-DF（原方案产出大量"的角/但我/在的"等无意义碎片）。

**核心思路**：学习信号 = **安全网升级事件**（不是 flash 报错）。flash 被判不足 → 说明这个任务 flash 不够 → 记下来，下次类似任务提前避开 flash。

```
flash 被安全网判不足 ──→ record_upgrade(text) ──→ U 集合(升级文本)
flash 正常返回      ──→ record_success(text) ──→ S 集合(成功文本)
                                              │
                     五层过滤提取领域名词:
                     ① jieba 词性标注 → 只留名词(n*)
                         "检查"vn→剔  "现在"t→剔  "手部"n→留
                     ② 停用词过滤 → 剔掉的/我/你/是/很…
                     ③ 频次阈值 → ≥3 次升级事件
                     ④ 对比过滤 → 不在 S 集合中
                     ⑤ 通用名词黑名单 → 剔问题/情况/方法
                                              │
                                              ▼
                              学到的领域词(手部/角度/弧度/单位)
                                              │
                    ① 实时注入 router._extra_keywords(+12 分)
                    ② upgrade_risk: trigram 相似度 > 0.3 → +20 分
```

**五层过滤的效果**：

| 版本 | 输入文本 | 产出关键词 |
|------|---------|-----------|
| v4.2 n-gram | "手部映射的角度有问题…" | `的角, 但我, 我现, 在的, 的问题` |
| v4.3 TF-DF | 同上 | `角度, 问题, 检查, 现在, 单位` |
| v4.4 jieba POS | 同上 | **`手部, 角度, 弧度, 单位`** |

**持久化**：`logs/upgrade_store.json`，启动加载、每 5 分钟 + 关闭时落盘。`SimilarityBuffer`（相似度延迟预测）保留，但学习主源已切到 UpgradeStore。

### 3.5 响应缓存

- 仅缓存非流式请求（流式请求无法缓存）
- LRU 淘汰，最多 128 条
- TTL 5 分钟
- Key = SHA256(model + messages)
- 命中时返回 `X-SP-Cache: hit` 响应头

---

## 4. 快速开始

### 4.1 准备环境

- Windows 10/11
- Python 3.10+（推荐 Anaconda）
- DeepSeek API Key（必填，去 https://platform.deepseek.com/api_keys 申请）
- Kimi API Key（可选，去 https://platform.moonshot.cn/console/api-keys 申请）

### 4.2 安装

```bash
# 1. 安装依赖
pip install fastapi uvicorn httpx pydantic pyyaml python-dotenv jieba

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env，填入你的 API Key

# 3. 启动
python -m smart_proxy
# 或者 Windows 上双击 start_proxy.bat（需管理员权限）
```

### 4.3 验证

浏览器访问 `http://127.0.0.1:8000/health`，看到 `{"status": "ok"}` 就成功了。

### 4.4 配置客户端

在 cc-switch 中创建供应商，目标 URL 设为 `http://127.0.0.1:8000/v1/messages`。

---

## 5. 文件清单：哪些上传了、哪些没上传

### ✅ 已上传（Git 跟踪）

| 文件 | 作用 |
|------|------|
| `smart_proxy/` 全部 13 个 `.py` 文件 | 核心源码（含 classifier.py） |
| `proxy_config.yaml` | 路由策略配置 |
| `.env.example` | API Key 配置模板 |
| `start_proxy.bat` | Windows 启动脚本 |
| `analyze_routing.py` | 日志分析 + HTML 报告生成 |
| `.gitignore` | Git 忽略规则 |
| `LICENSE` | MIT 许可证 |

### ❌ 未上传（在 `.gitignore` 中排除）

| 文件 | 作用 | 不上传的原因 | 影响 |
|------|------|-------------|------|
| **`.env`** | API Key | 包含密钥，安全考虑 | 新用户需自己创建 |
| **`logs/smart_proxy_history.json`** | **历史学习数据** | 运行时生成，在 `logs/` 目录下 | ⚠️ **影响最大** — 见下方说明 |
| **`logs/upgrade_store.json`** | **升级信号学习库** | 运行时生成 | ⚠️ 影响 UpgradeStore 自学习（见 §3.4） |
| `logs/smart_proxy.log` | 路由日志 | 运行时生成 | 不影响功能 |
| `proxy_state.json` | 熔断状态 | 运行时自动生成 | 不影响功能 |
| `cc-switch.db` | cc-switch 本地数据库 | 桌面应用数据 | 不影响 SmartProxy |
| `backups/` | 数据库备份 | 历史备份文件 | 不影响功能 |
| `routing_report_*.html` | 分析报告 | 运行脚本生成 | 不影响功能 |
| `settings.json` | cc-switch 桌面配置 | 个人桌面设置 | 不影响功能 |

### ⚠️ 关于学习数据文件的重要说明

`logs/smart_proxy_history.json`（相似度历史）和 `logs/upgrade_store.json`（升级信号学习库）是运行时积累的学习数据，都不在仓库里。

**它们不在仓库里的后果**：
- 每个新用户启动时，SimilarityBuffer 与 UpgradeStore 都是空的
- 升级信号自学习（§3.4）暂时不生效，需要积累足够的安全网升级事件后才开始工作
- **但分流功能完全正常**：分类器（§3.1 ②）和静态关键词（§7.1）是主判据，不依赖历史数据
- `upgrade_store.json` 出现后，学到的领域词会自动注入并随请求增长

**需要多长时间才能"学够"**：
- 学习库需要 flash 被**安全网判不足**才会积累（升级事件）
- 几十次 flash 请求后，若有少量升级事件，UpgradeStore 开始产出领域词
- 日常使用几天后，系统就能达到比较好的自适应效果
- 即使学习库为空，分类器仍能正确判断难度（只是少了"领域词加分"这个增强项）

---

## 6. 你需要自己准备什么

1. **API Key**（必须）：复制 `.env.example` 为 `.env`，填入真实的 Key
2. **Python 环境**（必须）：Python 3.10+，安装依赖包
3. **cc-switch**（推荐）：如果没有，任何能发 Anthropic 格式请求的客户端都可以
4. **耐心**（建议）：让系统跑几天积累历史数据，自学习才会生效

---

## 7. 如何优化关键词库和评分

v4.3 起，**关键词不再是主要判据**——主判据是 LLM 分类器。关键词降级为"高置信快速通道"，只保留小核心集。

### 7.1 调整高置信核心关键词（可选，影响快速通道）

打开 `smart_proxy/router.py`，找到三个小列表：

```python
EASY_VERBS = ["ls", "grep", "cat", "find", "read", "查看", "查找", ...]  # 命中→高置信 easy
EDIT_VERBS = ["fix", "implement", "refactor", "重构", "修复", ...]      # 命中→+12 偏 medium
HARD_CORE  = ["架构", "设计", "迁移", "distributed", "分布式", ...]     # 命中→高置信 hard
```

**添加原则**：只放**确定无疑**的词（命中即该档）。拿不准的不要放——拿不准的请求应交给分类器。词边界自动匹配（中英通用），不会出现 "concatenate" 误命中 "cat"。

### 7.2 让系统自动学习（零配置，主学习源）

系统的主学习源是 **UpgradeStore**（见 §3.4），你不需要手动操作：

1. flash 被安全网判不足 → 自动记入升级集
2. TF-DF 分析自动提取"flash 不够"的领域词 → 注入 `_extra_keywords`
3. 下次类似请求 → `upgrade_risk` 触发 → 强制走分类器确认

跑几天后，`/health` 的 `upgrade_store.learned_keywords` 会逐渐增长，系统越用越准。

### 7.3 调整评分阈值

在 `smart_proxy/router.py` 中：

```python
THRESHOLD_MEDIUM = 12   # 降低 → 更多请求走 pro（保守）
THRESHOLD_HARD = 40     # 降低 → 更多请求走 kimi（保守）
```

### 7.4 分类器开关与调参（A/B 对比）

在 `proxy_config.yaml` 的 `routing.classifier`：

```yaml
routing:
  classifier:
    enabled: true            # false → 完全关闭分类器, 回退纯启发式(可做 A/B)
    timeout: 1.5             # 分类器超时(秒), 超时回退 easy
    cache_size: 512          # LRU 缓存条目
    min_input_tokens: 200    # 上下文小于此值不调(经济闸门)
```

**如何 A/B**：先记下当前 `analyze_routing.py` 报告的升级率，把 `enabled` 改 `false` 跑一段，再对比。分类器开启应让分流更准（medium 不再误判 easy）。

### 7.5 扩展后端

目前支持 deepseek 和 kimi 两个后端。要添加新后端（如 OpenAI），需要：

1. 在 `proxy_config.yaml` 的 `backends:` 下添加配置
2. 在 `router.py` 的 `_pick_best()` 函数中加入新后端的逻辑
3. 在 `controller.py` 的 `_fallback_chain()` 中加入新后端的降级路径

---

## 8. 配置参数参考

`proxy_config.yaml` 中所有可配置项（已上传，可按需修改）：

### 路由策略
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `routing.flash_first` | `true` | 是否启用 flash-first |
| `routing.session_sticky.enabled` | `true` | 同一会话是否复用后端 |
| `routing.session_sticky.ttl_seconds` | `600` | Flash 会话粘性有效期 |
| `routing.session_sticky.pro_ttl_seconds` | `120` | Pro 会话粘性（防 easy 追问绑定贵模型） |
| `routing.token_length.enabled` | `true` | 是否启用超长上下文检测 |
| `routing.token_length.threshold_tokens` | `100000` | 超长阈值 |
| `routing.classifier.enabled` | `false` | LLM 分类器开关（默认关，仅直连路由用） |
| `routing.classifier.timeout` | `1.5` | 分类器超时（秒），超时回退 easy |
| `routing.classifier.cache_size` | `512` | 分类结果 LRU 缓存条目 |
| `routing.classifier.min_input_tokens` | `200` | 经济闸门：上下文小于此值不调分类器 |

### 熔断参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `circuit.block_seconds_quota` | `3600` | 配额耗尽冷却（1h） |
| `circuit.block_seconds_server` | `300` | 服务端错误冷却（5min） |
| `circuit.block_seconds_network` | `60` | 网络错误冷却（1min） |
| `circuit.block_seconds_permanent` | `18000` | 永久错误冷却（5h） |
| `circuit.backoff_max_seconds` | `7200` | 指数退避上限 |
| `circuit.probe_interval` | `30` | 探活间隔（秒） |

### 延迟与性能
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `latency.window` | `8` | 延迟统计滑动窗口大小 |
| `latency.slow_threshold_ms` | `8000` | flash 慢查询判定阈值 |
| `latency.degraded_sticky_s` | `30` | 降级标记粘性时长 |
| `cache.enabled` | `true` | 是否启用响应缓存 |
| `cache.max_entries` | `128` | 缓存最大条目数 |
| `cache.ttl_seconds` | `300` | 缓存过期时间 |

### 超时
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `timeouts.total` | `120` | 总超时（秒） |
| `timeouts.connect` | `10` | 连接超时（秒） |
| `timeouts.read` | `120` | 读取超时（秒） |

---

## 9. 日常运维

### 启动
```bash
python -m smart_proxy
# Windows: 双击 start_proxy.bat（管理员权限）
```

### 查看状态
```bash
curl http://127.0.0.1:8000/health
```
返回各后端熔断状态、延迟统计、关键词库大小、`upgrade_store`（升级学习库，含 learned_keywords 数量）、`classifier_cache`（分类器缓存命中数）等。重点看：
- `upgrade_store.learned_keywords`：学到的领域词数量，应随使用增长
- `classifier_cache.size`：分类器缓存条目数

### 生成分流报告
```bash
python analyze_routing.py              # 今天的数据
python analyze_routing.py 2026-07-24   # 指定日期
```
在浏览器打开生成的 `routing_report_*.html`。

### 健康指标参考
| 指标 | 正常 | 需关注 |
|------|------|--------|
| flash 直接返回率 | >60% | <30%（关键词库可能太宽） |
| flash 升级率 | <20% | >50%（flash 质量不够或关键词太窄） |
| 上游错误率 | 0% | >5%（检查 API Key 或网络） |

### 重启清除熔断
直接重启 SmartProxy 即可清零所有熔断状态。

---

## 10. 故障排查

### 启动报端口被占用
以管理员身份运行 `start_proxy.bat`，脚本会自动清理 8000 端口。

### 所有请求都走 pro（不走 flash）
检查是否触发了系统 prompt 误判。v4.3 已用 `_sanitize()` 剥离 `<system-reminder>` 等噪声，若仍异常，确认 `router.py` 的 `_extract_user_text()` 只检查最后一条 user 消息。也可临时把 `routing.classifier.enabled` 设 `false` 排除分类器影响。

### 请求返回 502 "All backends failed"
1. 检查 `http://127.0.0.1:8000/health` 看后端状态
2. 检查 `.env` 中的 API Key 是否有效
3. 查看 `logs/smart_proxy.log` 中的错误详情

### kimi 一直熔断
通常是 kimi 额度用完了。系统会自动熔断 5 小时并降级到 deepseek pro，不影响使用。充值后等探活自动恢复，或重启 SmartProxy。

### 想看更详细的日志
修改 `proxy_config.yaml` 中的 `log_level` 为 `DEBUG`，然后重启。

---

## 项目结构速查

```
.cc-switch/
├── smart_proxy/              ★ 核心源码（13 个 .py 文件）
│   ├── router.py             ★★★ 请求清洗 + 信号密度评分(去重/边界/密度)
│   ├── classifier.py               LLM 分类器(默认关闭)
│   ├── controller.py         ★★★ 请求编排 + 流式安全网 + 升级降级
│   ├── metrics.py            ★★  jieba 词性标注+五层过滤 + 延迟追踪
│   ├── app.py                ★    FastAPI 装配 + 探活 + 学习库持久化
│   ├── server.py                 请求 handler + 鉴权
│   ├── config.py                 YAML 热加载配置
│   ├── state.py                  熔断/过载状态管理
│   ├── circuit.py                指数退避计算
│   ├── cache.py                  LRU 响应缓存
│   ├── logging_setup.py          结构化日志
│   ├── __main__.py               启动入口
│   └── __init__.py               版本号
│
├── proxy_config.yaml          ★ 路由/熔断/缓存配置
├── .env.example               ★ API Key 模板
├── start_proxy.bat               启动脚本
├── analyze_routing.py            可视化分析
│
├── .env                       ❌ 未上传 — 你的 API Key
├── logs/                      ❌ 未上传 — 日志和历史学习数据
├── proxy_state.json           ❌ 未上传 — 自动生成
└── cc-switch.db               ❌ 未上传 — 桌面应用数据
```

---

*有问题看 `logs/smart_proxy.log`，改配置看 `proxy_config.yaml`，优化关键词看 `smart_proxy/router.py`。*
