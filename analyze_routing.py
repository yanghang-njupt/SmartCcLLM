#!/usr/bin/env python3
"""SmartProxy 分流可视化分析脚本

解析 cc-switch.log + smart_proxy.log，生成交互式 HTML 报告。
用法: python analyze_routing.py [日期]   # 默认今天
     python analyze_routing.py 2026-07-24
"""

import json, re, sys, os
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, Counter

# 修复 Windows 控制台 GBK 编码
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = Path(__file__).resolve().parent
LOGS_DIR = SCRIPT_DIR / "logs"

# ── 日志解析器 ────────────────────────────────────────────

CC_LOG_PAT = re.compile(
    r"\[(\d{4}-\d{2}-\d{2})\]\[(\d{2}:\d{2}:\d{2})\]\[INFO\]"
    r"\[cc_switch_lib::(proxy::forwarder|services::provider)\]"
)


def parse_cc_switch_log(target_date: str) -> list[dict]:
    """解析 Rust 主程序日志，提取请求目标和供应商切换事件。"""
    events = []
    log_file = LOGS_DIR / "cc-switch.log"
    if not log_file.exists():
        print(f"[WARN] {log_file} not found")
        return events

    for line in log_file.read_text(encoding="utf-8", errors="replace").splitlines():
        m = CC_LOG_PAT.match(line)
        if not m:
            continue
        date_str, time_str, module = m.group(1), m.group(2), m.group(3)
        if date_str != target_date:
            continue

        ts = f"{date_str} {time_str}"

        # 供应商切换
        if module == "services::provider":
            prov_match = re.search(r"供应商为\s+(\S+)", line)
            if prov_match:
                events.append({
                    "time": ts,
                    "type": "provider_switch",
                    "provider_id": prov_match.group(1),
                })

        # 请求转发
        if module == "proxy::forwarder":
            target_match = re.search(r"请求目标:\s+(\S+).*?model=([\w.-]+)", line)
            if target_match:
                url, model = target_match.group(1), target_match.group(2)
                via_proxy = "127.0.0.1:8000" in url or "localhost:8000" in url
                vendor = "deepseek"
                if "deepseek.com" in url:
                    vendor = "deepseek"
                elif "kimi.com" in url:
                    vendor = "kimi"
                elif via_proxy:
                    vendor = "smart_proxy"

                events.append({
                    "time": ts,
                    "type": "request",
                    "vendor": vendor,
                    "model": model,
                    "via_proxy": via_proxy,
                    "url": url,
                })

    return events


SP_LOG_PAT = re.compile(r'"ts"\s*:\s*"([^"]+)"')


def parse_smart_proxy_log(target_date: str) -> list[dict]:
    """解析 Python smart_proxy 日志，提取路由决策。"""
    events = []
    log_file = LOGS_DIR / "smart_proxy.log"
    if not log_file.exists():
        print(f"[INFO] {log_file} not found (smart_proxy hasn't written logs yet)")
        return events

    for line in log_file.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = entry.get("ts", "")
        if not ts.startswith(target_date):
            continue
        msg = entry.get("msg", "")

        # 路由决策: FlashFirst
        if "FlashFirst:" in msg:
            model_match = re.search(r"deepseek/([\w.-]+)", msg)
            upgrade_match = re.search(r"upgrade=(\w+)/(\w+)", msg)
            events.append({
                "time": ts,
                "type": "route_flash",
                "model": model_match.group(1) if model_match else "?",
                "upgrade_target": f"{upgrade_match.group(1)}/{upgrade_match.group(2)}" if upgrade_match else "?",
                "detail": msg,
            })

        # 路由决策: skip_flash
        elif "Route:" in msg and "strategy=skip_flash" in msg:
            backend_match = re.search(r"Route:\s+(\w+)/([\w.-]+)", msg)
            events.append({
                "time": ts,
                "type": "route_skip_flash",
                "backend": backend_match.group(1) if backend_match else "?",
                "model": backend_match.group(2) if backend_match else "?",
                "detail": msg,
            })

        # 升级: flash 不够
        elif "Flash inadequate" in msg:
            reason_match = re.search(r"Flash inadequate \(([^)]+)\).*upgrading to (\w+)", msg)
            events.append({
                "time": ts,
                "type": "flash_upgrade",
                "reason": reason_match.group(1) if reason_match else "?",
                "upgrade_to": reason_match.group(2) if reason_match else "?",
                "detail": msg,
            })

        # Flash 失败 → 升级
        elif "Flash failed" in msg:
            code_match = re.search(r"Flash failed \((\d+)\).*upgrading to (\w+)", msg)
            events.append({
                "time": ts,
                "type": "flash_error_upgrade",
                "status_code": int(code_match.group(1)) if code_match else 0,
                "upgrade_to": code_match.group(2) if code_match else "?",
                "detail": msg,
            })

        # 最终降级
        elif "Ultimate fallback to" in msg:
            fb_match = re.search(r"fallback to (\w+)", msg)
            events.append({
                "time": ts,
                "type": "ultimate_fallback",
                "backend": fb_match.group(1) if fb_match else "?",
                "detail": msg,
            })

        # 上游错误
        elif "Upstream Error" in msg or "Relay Error" in msg:
            events.append({
                "time": ts,
                "type": "error",
                "detail": msg,
            })

        # 缓存命中
        elif "Cache hit" in msg:
            events.append({
                "time": ts,
                "type": "cache_hit",
                "detail": msg,
            })

    return events


# ── 统计计算 ──────────────────────────────────────────────

def compute_stats(cc_events, sp_events):
    """聚合统计。"""
    # cc-switch 层
    total_req = sum(1 for e in cc_events if e["type"] == "request")
    direct_req = sum(1 for e in cc_events if e["type"] == "request" and not e["via_proxy"])
    proxy_req = sum(1 for e in cc_events if e["type"] == "request" and e["via_proxy"])

    # 按供应商分布
    model_dist = Counter()
    vendor_dist = Counter()
    for e in cc_events:
        if e["type"] == "request":
            model_dist[e["model"]] += 1
            vendor_dist[e["vendor"]] += 1

    # 供应商切换
    switches = [(e["time"], e["provider_id"]) for e in cc_events if e["type"] == "provider_switch"]

    # smart_proxy 层
    sp_total = len([e for e in sp_events if e["type"] in ("route_flash", "route_skip_flash")])
    flash_direct = len([
        e for e in sp_events
        if e["type"] == "route_flash"
        and not any(
            u["time"] == e["time"] for u in sp_events
            if u["type"] in ("flash_upgrade", "flash_error_upgrade")
        )
    ])
    flash_upgraded = len([e for e in sp_events if e["type"] in ("flash_upgrade", "flash_error_upgrade")])
    skip_flash = len([e for e in sp_events if e["type"] == "route_skip_flash"])
    upstream_errors = len([e for e in sp_events if e["type"] == "error"])
    cache_hits = len([e for e in sp_events if e["type"] == "cache_hit"])

    # 小时分布
    hour_buckets = defaultdict(lambda: {"direct": 0, "proxy": 0})
    for e in cc_events:
        if e["type"] == "request":
            hour = e["time"][11:13]
            key = "proxy" if e["via_proxy"] else "direct"
            hour_buckets[hour][key] += 1

    return {
        "total_req": total_req,
        "direct_req": direct_req,
        "proxy_req": proxy_req,
        "model_dist": dict(model_dist.most_common()),
        "vendor_dist": dict(vendor_dist),
        "switches": switches,
        "sp_total": sp_total,
        "flash_direct": flash_direct,
        "flash_upgraded": flash_upgraded,
        "skip_flash": skip_flash,
        "upstream_errors": upstream_errors,
        "cache_hits": cache_hits,
        "hour_buckets": dict(sorted(hour_buckets.items())),
    }


# ── HTML 报告生成 ─────────────────────────────────────────

def generate_html(stats, cc_events, sp_events, target_date):
    """生成可视化 HTML 报告。"""

    # 小时分布数据
    hours = list(stats["hour_buckets"].keys())
    h_direct = [stats["hour_buckets"][h]["direct"] for h in hours]
    h_proxy = [stats["hour_buckets"][h]["proxy"] for h in hours]

    # SP 路由分解
    sp_pie = [stats["flash_direct"], stats["flash_upgraded"], stats["skip_flash"]]
    sp_pie_labels = ["flash 直接返回", "flash 升级", "跳过 flash"]

    # 模型分布 (top 5)
    top_models = list(stats["model_dist"].items())[:8]

    # 供应商切换时间线
    switch_rows = ""
    for ts, pid in stats["switches"]:
        short_pid = pid[:20] + "..." if len(pid) > 20 else pid
        label = "🌐 smart_proxy" if "universal" in pid else "🔗 直连 deepseek"
        switch_rows += f'<tr><td class="ts">{ts}</td><td>{label}</td><td class="mono">{short_pid}</td></tr>'

    # SP 事件列表
    sp_rows = ""
    for e in sp_events[-60:]:  # 最近 60 条
        icon_map = {
            "route_flash": "⚡",
            "route_skip_flash": "⏭️",
            "flash_upgrade": "⬆️",
            "flash_error_upgrade": "⚠️⬆️",
            "ultimate_fallback": "🔻",
            "error": "❌",
            "cache_hit": "💾",
        }
        icon = icon_map.get(e["type"], "•")
        detail = e.get("detail", "")[:100]
        sp_rows += f'<tr><td class="ts">{e["time"]}</td><td>{icon} {e["type"]}</td><td class="detail">{detail}</td></tr>'

    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SmartProxy 分流报告 — {target_date}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  :root {{ --bg: #0d1117; --card: #161b22; --border: #30363d; --text: #e6edf3; --dim: #8b949e;
          --green: #3fb950; --orange: #d2991d; --red: #f85149; --blue: #58a6ff; --purple: #a371f7; }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; background:var(--bg); color:var(--text); padding:24px; }}
  h1 {{ font-size:24px; margin-bottom:4px; }}
  .subtitle {{ color:var(--dim); font-size:13px; margin-bottom:24px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:16px; margin-bottom:24px; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:8px; padding:16px; }}
  .card h3 {{ font-size:13px; color:var(--dim); text-transform:uppercase; letter-spacing:0.5px; margin-bottom:12px; }}
  .kpi {{ font-size:36px; font-weight:700; }}
  .kpi.green {{ color:var(--green); }}
  .kpi.blue {{ color:var(--blue); }}
  .kpi.orange {{ color:var(--orange); }}
  .kpi.purple {{ color:var(--purple); }}
  .kpi.red {{ color:var(--red); }}
  .kpi-desc {{ font-size:12px; color:var(--dim); }}
  .chart-wrap {{ position:relative; height:200px; margin:8px 0; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th,td {{ padding:6px 10px; text-align:left; border-bottom:1px solid var(--border); }}
  th {{ color:var(--dim); font-weight:600; }}
  .ts {{ font-family:monospace; font-size:12px; color:var(--dim); white-space:nowrap; }}
  .mono {{ font-family:monospace; font-size:11px; }}
  .detail {{ font-size:12px; color:var(--dim); max-width:400px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  .section {{ margin-top:32px; }}
  .section h2 {{ font-size:18px; margin-bottom:12px; }}
  .arrow-box {{ display:flex; align-items:center; gap:12px; flex-wrap:wrap; padding:16px; background:var(--card); border-radius:8px; border:1px solid var(--border); font-family:monospace; font-size:13px; }}
  .arrow-box .node {{ background:#1c2128; padding:8px 14px; border-radius:6px; border:1px solid var(--border); }}
  .arrow-box .arrow {{ color:var(--dim); font-size:18px; }}
  .tag {{ display:inline-block; padding:2px 8px; border-radius:12px; font-size:11px; font-weight:600; }}
  .tag.green {{ background:var(--green); color:#000; }}
  .tag.orange {{ background:var(--orange); color:#000; }}
  .tag.red {{ background:var(--red); color:#fff; }}
  .tag.blue {{ background:var(--blue); color:#000; }}
</style>
</head>
<body>

<h1>🔀 SmartProxy 分流报告</h1>
<div class="subtitle">日期: {target_date} &nbsp;|&nbsp; 生成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>

<div class="grid">
  <div class="card">
    <h3>📊 总请求量</h3>
    <div class="kpi blue">{stats['total_req']}</div>
    <div class="kpi-desc">直连 {stats['direct_req']} 条 &middot; 经 smart_proxy {stats['proxy_req']} 条</div>
  </div>
  <div class="card">
    <h3>⚡ flash 直接返回</h3>
    <div class="kpi green">{stats['flash_direct'] if stats['sp_total'] > 0 else '—'}</div>
    <div class="kpi-desc">无需升级，flash 质量足够</div>
  </div>
  <div class="card">
    <h3>⬆️ flash 升级</h3>
    <div class="kpi orange">{stats['flash_upgraded'] if stats['sp_total'] > 0 else '—'}</div>
    <div class="kpi-desc">截断/太慢/工具调用 触发升级</div>
  </div>
  <div class="card">
    <h3>⏭️ 跳过 flash</h3>
    <div class="kpi purple">{stats['skip_flash'] if stats['sp_total'] > 0 else '—'}</div>
    <div class="kpi-desc">高难度/长上下文，直接最强模型</div>
  </div>
</div>

<div class="grid">
  <div class="card">
    <h3>📈 请求量按小时分布</h3>
    <div class="chart-wrap"><canvas id="hourlyChart"></canvas></div>
  </div>
  <div class="card">
    <h3>🍩 分流器路由分解</h3>
    <div class="chart-wrap"><canvas id="routePie"></canvas></div>
  </div>
</div>

<div class="grid">
  <div class="card">
    <h3>🔗 模型调用分布</h3>
    <div class="chart-wrap"><canvas id="modelChart"></canvas></div>
  </div>
  <div class="card">
    <h3>🗺️ 当前分流路径</h3>
    <div class="arrow-box">
      <span class="node">Claude Code</span>
      <span class="arrow">→</span>
      <span class="node">{'smart_proxy' if stats['proxy_req'] > 0 else 'api.deepseek.com'}</span>
      <span class="arrow">→</span>
      <span class="node">deepseek flash</span>
      {'<span class="arrow">→</span><span class="node">pro/kimi</span>' if stats['flash_upgraded'] > 0 else ''}
    </div>
  </div>
</div>

<div class="section">
  <h2>🔄 供应商切换记录</h2>
  <table>
    <tr><th>时间</th><th>切换至</th><th>Provider ID</th></tr>
    {switch_rows or '<tr><td colspan="3" style="color:var(--dim)">今日无切换记录</td></tr>'}
  </table>
</div>

<div class="section">
  <h2>📋 smart_proxy 路由事件</h2>
  <table>
    <tr><th>时间</th><th>事件</th><th>详情</th></tr>
    {sp_rows or '<tr><td colspan="3" style="color:var(--dim)">smart_proxy 暂无日志（请重启 smart_proxy 后生效）</td></tr>'}
  </table>
</div>

<script>
new Chart(document.getElementById('hourlyChart'), {{
  type:'bar',
  data:{{
    labels:{json.dumps(hours)},
    datasets:[
      {{label:'直连',data:{json.dumps(h_direct)},backgroundColor:'#58a6ff55',borderColor:'#58a6ff',borderWidth:1}},
      {{label:'smart_proxy',data:{json.dumps(h_proxy)},backgroundColor:'#a371f755',borderColor:'#a371f7',borderWidth:1}},
    ]
  }},
  options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{labels:{{color:'#8b949e'}}}}}},scales:{{x:{{ticks:{{color:'#8b949e'}},grid:{{color:'#21262d'}}}},y:{{ticks:{{color:'#8b949e'}},grid:{{color:'#21262d'}},beginAtZero:true}}}}}}
}});

new Chart(document.getElementById('routePie'), {{
  type:'doughnut',
  data:{{
    labels:{json.dumps(sp_pie_labels)},
    datasets:[{{data:{json.dumps(sp_pie)},backgroundColor:['#3fb950','#d2991d','#a371f7'],borderColor:'#161b22'}}]
  }},
  options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{labels:{{color:'#8b949e'}}}}}}}}
}});

new Chart(document.getElementById('modelChart'), {{
  type:'bar',
  data:{{
    labels:{json.dumps([m for m,_ in top_models])},
    datasets:[{{label:'请求数',data:{json.dumps([c for _,c in top_models])},backgroundColor:'#f0883e88',borderColor:'#f0883e',borderWidth:1}}]
  }},
  options:{{indexAxis:'y',responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},scales:{{x:{{ticks:{{color:'#8b949e'}},grid:{{color:'#21262d'}},beginAtZero:true}},y:{{ticks:{{color:'#8b949e'}},grid:{{color:'#21262d'}}}}}}}}
}});
</script>
</body>
</html>"""


# ── 主函数 ────────────────────────────────────────────────

def main():
    target_date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")

    print(f"📊 分析日期: {target_date}")
    print(f"─" * 50)

    cc_events = parse_cc_switch_log(target_date)
    sp_events = parse_smart_proxy_log(target_date)

    print(f"cc-switch 日志: {len(cc_events)} 条事件")
    print(f"smart_proxy 日志: {len(sp_events)} 条路由事件")

    stats = compute_stats(cc_events, sp_events)

    # 终端摘要
    print()
    print(f"━━ 总览 ━━")
    print(f"  总请求: {stats['total_req']}  (直连 {stats['direct_req']} | smart_proxy {stats['proxy_req']})")
    print(f"  模型: {stats['model_dist']}")
    if stats['sp_total'] > 0:
        print(f"  ── smart_proxy 层 ──")
        print(f"    flash 直接返回: {stats['flash_direct']}")
        print(f"    flash 升级   : {stats['flash_upgraded']}")
        print(f"    跳过 flash   : {stats['skip_flash']}")
        print(f"    上游错误     : {stats['upstream_errors']}")
        print(f"    缓存命中     : {stats['cache_hits']}")
    else:
        print(f"  smart_proxy: 暂未记录到路由事件（需重启 smart_proxy 启用日志）")
    print(f"  供应商切换: {len(stats['switches'])} 次")
    for ts, pid in stats["switches"]:
        label = "smart_proxy" if "universal" in pid else "直连"
        print(f"    {ts} → {label} ({pid})")

    # 生成 HTML
    html = generate_html(stats, cc_events, sp_events, target_date)
    out_path = SCRIPT_DIR / f"routing_report_{target_date}.html"
    # 用带 BOM 的 utf-8 防止 Windows 浏览器乱码
    out_path.write_text(html, encoding="utf-8")
    print(f"\n✅ 报告已生成: {out_path}")
    print(f"   在浏览器中打开即可查看可视化报告")


if __name__ == "__main__":
    main()
