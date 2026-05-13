#!/usr/bin/env python3
"""
GitHub Copilot Agent Log Analyzer
Analyzes VS Code Copilot chat sessions, transcripts, and debug logs
to generate a rich HTML insights report.

Usage:
    python analyze_agent_log.py [--output report.html] [--logs-dir /path/to/logs]

By default it auto-discovers all VS Code Copilot logs on this Mac.
Drop your own *.jsonl files into this folder and they will be picked up too.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


# ─────────────────────────────────────────────
#  Data collection helpers
# ─────────────────────────────────────────────

def ts_to_dt(ts):
    """Convert millisecond epoch or ISO string to datetime (UTC-aware)."""
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        try:
            return datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        except (OSError, OverflowError):
            return None
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def find_vscode_log_dirs():
    """Locate VS Code workspaceStorage on macOS / Linux / Windows."""
    candidates = []
    home = Path.home()
    # macOS
    candidates.append(home / "Library" / "Application Support" / "Code" / "User" / "workspaceStorage")
    candidates.append(home / "Library" / "Application Support" / "Code - Insiders" / "User" / "workspaceStorage")
    # Linux
    candidates.append(home / ".config" / "Code" / "User" / "workspaceStorage")
    # Windows
    appdata = os.environ.get("APPDATA", "")
    if appdata:
        candidates.append(Path(appdata) / "Code" / "User" / "workspaceStorage")

    return [p for p in candidates if p.exists()]


def collect_jsonl_files(extra_dirs):
    """Yield (path, kind) for every relevant JSONL file."""
    roots = find_vscode_log_dirs()
    roots += [Path(d) for d in extra_dirs if Path(d).exists()]

    # Also scan the script's own directory for any dropped-in JSONL files
    script_dir = Path(__file__).parent
    roots.append(script_dir)

    seen = set()
    for root in roots:
        for path in root.rglob("*.jsonl"):
            if path in seen:
                continue
            seen.add(path)
            parts = path.parts
            if "chatSessions" in parts:
                yield path, "chatSession"
            elif "transcripts" in parts:
                yield path, "transcript"
            elif "debug-logs" in parts:
                yield path, "debugLog"
            else:
                # Unknown – still try to parse
                yield path, "unknown"


# ─────────────────────────────────────────────
#  Parsers
# ─────────────────────────────────────────────

def parse_chat_session(path):
    """Parse a chatSessions/*.jsonl file. Returns a list of request dicts."""
    requests = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                v = obj.get("v", {})
                if not isinstance(v, dict):
                    continue
                session_id = v.get("sessionId", str(path.stem))
                creation_date = v.get("creationDate")
                mode = v.get("inputState", {}).get("mode", {}).get("kind", "unknown")
                for req in v.get("requests", []):
                    if not isinstance(req, dict):
                        continue
                    result_meta = req.get("result", {})
                    if isinstance(result_meta, dict):
                        metadata = result_meta.get("metadata", {})
                        timings = result_meta.get("timings", {})
                    else:
                        metadata = {}
                        timings = {}

                    tool_rounds = metadata.get("toolCallRounds", [])
                    tool_calls = []
                    for rnd in tool_rounds if isinstance(tool_rounds, list) else []:
                        for tc in rnd.get("toolCalls", []) if isinstance(rnd, dict) else []:
                            tool_calls.append(tc.get("function", {}).get("name") or tc.get("name", "unknown"))

                    requests.append({
                        "source": "chatSession",
                        "session_id": session_id,
                        "session_created": ts_to_dt(creation_date),
                        "request_id": req.get("requestId"),
                        "timestamp": ts_to_dt(req.get("timestamp")),
                        "model_id": req.get("modelId", ""),
                        "resolved_model": metadata.get("resolvedModel", ""),
                        "mode": req.get("modeInfo", {}).get("kind", mode) if isinstance(req.get("modeInfo"), dict) else mode,
                        "message_text": req.get("message", {}).get("text", "") if isinstance(req.get("message"), dict) else "",
                        "prompt_tokens": metadata.get("promptTokens"),
                        "output_tokens": metadata.get("outputTokens") or req.get("completionTokens"),
                        "elapsed_ms": timings.get("totalElapsed") or req.get("elapsedMs"),
                        "first_progress_ms": timings.get("firstProgress"),
                        "tool_calls": tool_calls,
                        "result_details": result_meta.get("details", "") if isinstance(result_meta, dict) else "",
                        "agent_version": req.get("agent", {}).get("extensionVersion", "") if isinstance(req.get("agent"), dict) else "",
                        "workspace_path": str(path),
                    })
    except (OSError, UnicodeDecodeError):
        pass
    return requests


def parse_transcript(path):
    """Parse a transcripts/*.jsonl file. Returns list of event dicts."""
    events = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                events.append(obj)
    except (OSError, UnicodeDecodeError):
        pass
    return events


def parse_debug_log(path):
    """Parse a debug-logs/*/main.jsonl file. Returns list of event dicts."""
    events = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    events.append(obj)
    except (OSError, UnicodeDecodeError):
        pass
    return events


# ─────────────────────────────────────────────
#  Aggregation
# ─────────────────────────────────────────────

def aggregate(extra_dirs):
    all_requests = []
    all_transcript_events = []
    all_debug_events = []

    for path, kind in collect_jsonl_files(extra_dirs):
        if kind == "chatSession":
            all_requests.extend(parse_chat_session(path))
        elif kind == "transcript":
            all_transcript_events.extend(parse_transcript(path))
        elif kind == "debugLog":
            all_debug_events.extend(parse_debug_log(path))

    return all_requests, all_transcript_events, all_debug_events


# ─────────────────────────────────────────────
#  Insight computation
# ─────────────────────────────────────────────

def compute_insights(requests, transcript_events, debug_events):
    ins = {}

    # ── Session-level ──────────────────────────────────────────────────
    session_ids = {r["session_id"] for r in requests}
    session_ids.update(
        e["data"]["sessionId"] for e in transcript_events
        if e.get("type") == "session.start" and isinstance(e.get("data"), dict)
    )
    ins["total_sessions"] = len(session_ids)
    ins["total_requests"] = len(requests)

    timestamps = [r["timestamp"] for r in requests if r["timestamp"]]
    if timestamps:
        ins["first_request"] = min(timestamps)
        ins["last_request"] = max(timestamps)
        delta = ins["last_request"] - ins["first_request"]
        ins["active_days"] = max(1, delta.days + 1)
    else:
        ins["first_request"] = ins["last_request"] = None
        ins["active_days"] = 0

    # ── Model usage ────────────────────────────────────────────────────
    model_counts = Counter()
    model_tokens = defaultdict(lambda: {"prompt": 0, "output": 0, "requests": 0})
    for r in requests:
        mid = r.get("resolved_model") or r.get("model_id") or "unknown"
        mid = mid.replace("copilot/", "")
        model_counts[mid] += 1
        model_tokens[mid]["requests"] += 1
        if r["prompt_tokens"]:
            model_tokens[mid]["prompt"] += r["prompt_tokens"]
        if r["output_tokens"]:
            model_tokens[mid]["output"] += r["output_tokens"]
    ins["model_counts"] = dict(model_counts.most_common())
    ins["model_tokens"] = dict(model_tokens)

    # ── Token totals ───────────────────────────────────────────────────
    ins["total_prompt_tokens"] = sum(r["prompt_tokens"] or 0 for r in requests)
    ins["total_output_tokens"] = sum(r["output_tokens"] or 0 for r in requests)

    # ── Latency ───────────────────────────────────────────────────────
    elapsed = [r["elapsed_ms"] for r in requests if r["elapsed_ms"]]
    ttft = [r["first_progress_ms"] for r in requests if r["first_progress_ms"]]
    ins["avg_elapsed_ms"] = int(sum(elapsed) / len(elapsed)) if elapsed else 0
    ins["median_elapsed_ms"] = sorted(elapsed)[len(elapsed) // 2] if elapsed else 0
    ins["avg_ttft_ms"] = int(sum(ttft) / len(ttft)) if ttft else 0
    ins["p95_elapsed_ms"] = sorted(elapsed)[int(len(elapsed) * 0.95)] if elapsed else 0

    # ── Tool usage ─────────────────────────────────────────────────────
    all_tool_calls = []
    for r in requests:
        all_tool_calls.extend(r["tool_calls"])
    # Also from transcripts
    for e in transcript_events:
        if e.get("type") == "tool.execution_start":
            d = e.get("data", {})
            if isinstance(d, dict) and d.get("toolName"):
                all_tool_calls.append(d["toolName"])
    ins["tool_call_counts"] = dict(Counter(all_tool_calls).most_common(20))
    ins["total_tool_calls"] = len(all_tool_calls)

    tool_success = sum(
        1 for e in transcript_events
        if e.get("type") == "tool.execution_complete"
        and isinstance(e.get("data"), dict)
        and e["data"].get("success", False)
    )
    tool_failures = sum(
        1 for e in transcript_events
        if e.get("type") == "tool.execution_complete"
        and isinstance(e.get("data"), dict)
        and not e["data"].get("success", True)
    )
    ins["tool_success_rate"] = (
        round(tool_success / (tool_success + tool_failures) * 100, 1)
        if (tool_success + tool_failures) > 0 else None
    )

    # ── Mode breakdown ─────────────────────────────────────────────────
    ins["mode_counts"] = dict(Counter(r["mode"] for r in requests).most_common())

    # ── Time-of-day heatmap ────────────────────────────────────────────
    hour_counts = Counter(r["timestamp"].hour for r in requests if r["timestamp"])
    ins["hour_counts"] = {h: hour_counts.get(h, 0) for h in range(24)}

    # ── Day-of-week usage ─────────────────────────────────────────────
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_counts = Counter(r["timestamp"].weekday() for r in requests if r["timestamp"])
    ins["dow_counts"] = {dow_names[i]: dow_counts.get(i, 0) for i in range(7)}

    # ── Requests per session ───────────────────────────────────────────
    reqs_per_session = Counter(r["session_id"] for r in requests)
    counts = list(reqs_per_session.values())
    ins["avg_requests_per_session"] = round(sum(counts) / len(counts), 1) if counts else 0
    ins["max_requests_per_session"] = max(counts) if counts else 0
    ins["sessions_with_multiple_requests"] = sum(1 for c in counts if c > 1)

    # ── Tool calls per session ─────────────────────────────────────────
    tool_calls_per_session = defaultdict(int)
    for r in requests:
        tool_calls_per_session[r["session_id"]] += len(r["tool_calls"])
    tcps = list(tool_calls_per_session.values())
    ins["avg_tool_calls_per_session"] = round(sum(tcps) / len(tcps), 1) if tcps else 0

    # ── Message length analysis ────────────────────────────────────────
    msg_lengths = [len(r["message_text"]) for r in requests if r["message_text"]]
    ins["avg_message_length"] = int(sum(msg_lengths) / len(msg_lengths)) if msg_lengths else 0
    ins["max_message_length"] = max(msg_lengths) if msg_lengths else 0

    # ── Top keywords from user messages ───────────────────────────────
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "as", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "i", "you", "it", "this",
        "that", "my", "your", "me", "we", "us", "he", "she", "they", "them",
        "from", "by", "not", "so", "if", "up", "out", "what", "how", "all",
        "just", "like", "make", "please", "use", "using", "add", "new"
    }
    word_freq = Counter()
    for r in requests:
        words = re.findall(r"[a-z]{3,}", r["message_text"].lower())
        word_freq.update(w for w in words if w not in stop_words)
    ins["top_keywords"] = dict(word_freq.most_common(20))

    # ── Copilot / VS Code version history ─────────────────────────────
    versions = []
    for e in debug_events:
        attrs = e.get("attrs", {})
        if isinstance(attrs, dict) and attrs.get("copilotVersion"):
            versions.append({
                "copilot": attrs.get("copilotVersion", ""),
                "vscode": attrs.get("vscodeVersion", ""),
                "ts": ts_to_dt(e.get("ts")),
            })
    for e in transcript_events:
        if e.get("type") == "session.start":
            d = e.get("data", {})
            if isinstance(d, dict) and d.get("copilotVersion"):
                versions.append({
                    "copilot": d.get("copilotVersion", ""),
                    "vscode": d.get("vscodeVersion", ""),
                    "ts": ts_to_dt(d.get("startTime")),
                })
    versions.sort(key=lambda x: x["ts"] or datetime.min.replace(tzinfo=timezone.utc))
    unique_copilot = []
    seen_v = None
    for v in versions:
        if v["copilot"] != seen_v:
            unique_copilot.append(v)
            seen_v = v["copilot"]
    ins["version_history"] = unique_copilot[-10:]  # last 10 unique versions

    # ── Daily request trend (last 30 unique dates) ─────────────────────
    day_counts = Counter(
        r["timestamp"].date() for r in requests if r["timestamp"]
    )
    sorted_days = sorted(day_counts.keys())[-30:]
    ins["daily_trend"] = {str(d): day_counts[d] for d in sorted_days}

    return ins


# ─────────────────────────────────────────────
#  HTML report generation
# ─────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>GitHub Copilot Agent Log Insights</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0d1117;
    --surface: #161b22;
    --border: #30363d;
    --accent: #58a6ff;
    --accent2: #3fb950;
    --accent3: #d29922;
    --accent4: #f78166;
    --text: #e6edf3;
    --muted: #8b949e;
    --card-radius: 12px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 32px 24px;
  }
  h1 { font-size: 1.8rem; font-weight: 700; margin-bottom: 4px; }
  .subtitle { color: var(--muted); margin-bottom: 32px; font-size: 0.9rem; }
  .section-title {
    font-size: 1.1rem; font-weight: 600;
    margin: 32px 0 16px;
    color: var(--accent);
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
  }
  .grid { display: grid; gap: 16px; }
  .grid-2 { grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); }
  .grid-3 { grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); }
  .grid-chart { grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); }
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--card-radius);
    padding: 20px;
  }
  .stat-label { font-size: 0.78rem; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }
  .stat-value { font-size: 2rem; font-weight: 700; line-height: 1.2; margin-top: 4px; }
  .stat-sub { font-size: 0.82rem; color: var(--muted); margin-top: 4px; }
  .accent-blue { color: var(--accent); }
  .accent-green { color: var(--accent2); }
  .accent-yellow { color: var(--accent3); }
  .accent-red { color: var(--accent4); }
  .chart-wrap { position: relative; height: 240px; }
  .chart-wrap-tall { position: relative; height: 300px; }
  table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
  th { text-align: left; padding: 8px 12px; color: var(--muted); font-weight: 600;
       border-bottom: 1px solid var(--border); }
  td { padding: 8px 12px; border-bottom: 1px solid var(--border); }
  tr:last-child td { border-bottom: none; }
  .bar-cell { display: flex; align-items: center; gap: 8px; }
  .mini-bar { height: 8px; border-radius: 4px; background: var(--accent); flex-shrink: 0; }
  .badge {
    display: inline-block; padding: 2px 8px; border-radius: 12px;
    font-size: 0.75rem; font-weight: 600;
    background: rgba(88,166,255,.15); color: var(--accent);
  }
  .tag { display: inline-block; padding: 3px 10px; border-radius: 6px; font-size: 0.8rem;
          margin: 3px; background: var(--border); color: var(--text); }
  .version-row { display: flex; justify-content: space-between; padding: 6px 0;
                  border-bottom: 1px solid var(--border); font-size: 0.85rem; }
  .version-row:last-child { border-bottom: none; }
  footer { text-align: center; color: var(--muted); font-size: 0.8rem; margin-top: 48px; }
</style>
</head>
<body>
<h1>GitHub Copilot Agent Log Insights</h1>
<p class="subtitle">Generated {generated_at} &nbsp;·&nbsp; Analyzing logs from {first_date} to {last_date}</p>

<!-- ── OVERVIEW STATS ────────────────────────────────────── -->
<div class="section-title">Overview</div>
<div class="grid grid-3">
  <div class="card">
    <div class="stat-label">Total Sessions</div>
    <div class="stat-value accent-blue">{total_sessions}</div>
    <div class="stat-sub">{sessions_multi} with multiple requests</div>
  </div>
  <div class="card">
    <div class="stat-label">Total Requests</div>
    <div class="stat-value accent-green">{total_requests}</div>
    <div class="stat-sub">avg {avg_rps} per session &nbsp;·&nbsp; max {max_rps}</div>
  </div>
  <div class="card">
    <div class="stat-label">Active Days</div>
    <div class="stat-value accent-yellow">{active_days}</div>
    <div class="stat-sub">{requests_per_day} requests / day avg</div>
  </div>
  <div class="card">
    <div class="stat-label">Total Tokens Used</div>
    <div class="stat-value accent-blue">{total_tokens}</div>
    <div class="stat-sub">{prompt_tokens} prompt &nbsp;·&nbsp; {output_tokens} output</div>
  </div>
  <div class="card">
    <div class="stat-label">Total Tool Calls</div>
    <div class="stat-value accent-green">{total_tool_calls}</div>
    <div class="stat-sub">{avg_tools_per_session} per session avg{success_rate_str}</div>
  </div>
  <div class="card">
    <div class="stat-label">Avg Response Time</div>
    <div class="stat-value accent-yellow">{avg_elapsed}</div>
    <div class="stat-sub">TTFT {avg_ttft} &nbsp;·&nbsp; p95 {p95_elapsed}</div>
  </div>
</div>

<!-- ── CHARTS ROW 1 ──────────────────────────────────────── -->
<div class="section-title">Usage Trends</div>
<div class="grid grid-chart">
  <div class="card">
    <div style="font-weight:600;margin-bottom:12px">Daily Request Activity</div>
    <div class="chart-wrap-tall"><canvas id="dailyChart"></canvas></div>
  </div>
  <div class="card">
    <div style="font-weight:600;margin-bottom:12px">Requests by Hour of Day</div>
    <div class="chart-wrap-tall"><canvas id="hourChart"></canvas></div>
  </div>
</div>

<!-- ── MODEL USAGE ───────────────────────────────────────── -->
<div class="section-title">Model Usage</div>
<div class="grid grid-chart">
  <div class="card">
    <div style="font-weight:600;margin-bottom:12px">Requests per Model</div>
    <div class="chart-wrap"><canvas id="modelChart"></canvas></div>
  </div>
  <div class="card">
    <div style="font-weight:600;margin-bottom:12px">Token Distribution per Model</div>
    <div class="chart-wrap"><canvas id="tokenChart"></canvas></div>
  </div>
</div>

<!-- ── TOOL USAGE ────────────────────────────────────────── -->
<div class="section-title">Tool Usage</div>
<div class="grid grid-chart">
  <div class="card">
    <div style="font-weight:600;margin-bottom:16px">Top Tools Called</div>
    <table>
      <thead><tr><th>Tool</th><th>Calls</th><th>Share</th></tr></thead>
      <tbody id="toolTable"></tbody>
    </table>
  </div>
  <div class="card">
    <div style="font-weight:600;margin-bottom:12px">Day-of-Week Activity</div>
    <div class="chart-wrap"><canvas id="dowChart"></canvas></div>
  </div>
</div>

<!-- ── USER MESSAGES ─────────────────────────────────────── -->
<div class="section-title">User Message Patterns</div>
<div class="grid grid-2">
  <div class="card">
    <div style="font-weight:600;margin-bottom:12px">Top Keywords</div>
    <div id="keywordTags"></div>
  </div>
  <div class="card">
    <table>
      <thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>
        <tr><td>Avg message length</td><td class="accent-blue">{avg_msg_len} chars</td></tr>
        <tr><td>Max message length</td><td class="accent-blue">{max_msg_len} chars</td></tr>
        <tr><td>Most used mode</td><td class="accent-green">{top_mode}</td></tr>
        <tr><td>Avg tokens / request</td><td class="accent-yellow">{avg_output_tokens}</td></tr>
      </tbody>
    </table>
  </div>
</div>

<!-- ── VERSION HISTORY ───────────────────────────────────── -->
<div class="section-title">Version History</div>
<div class="card">
  <div id="versionList"></div>
</div>

<footer>Generated by analyze_agent_log.py on {generated_at}</footer>

<script>
const BLUE   = '#58a6ff';
const GREEN  = '#3fb950';
const YELLOW = '#d29922';
const RED    = '#f78166';
const PURPLE = '#bc8cff';
const CYAN   = '#39d353';
const PALETTE = [BLUE, GREEN, YELLOW, RED, PURPLE, CYAN,
                  '#ff8c00','#e07b39','#88d4ab','#b392f0'];

const gridColor = 'rgba(48,54,61,0.8)';
const textColor = '#8b949e';

Chart.defaults.color = textColor;
Chart.defaults.font.family = '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';

function barChart(id, labels, data, color, opts={}) {
  new Chart(document.getElementById(id), {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data,
        backgroundColor: color || BLUE,
        borderRadius: 4,
        borderSkipped: false,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, ...opts.plugins },
      scales: {
        x: { grid: { color: gridColor }, ticks: { maxRotation: 45, font: { size: 11 } } },
        y: { grid: { color: gridColor }, beginAtZero: true, ticks: { precision: 0 } }
      },
      ...opts
    }
  });
}

function doughnutChart(id, labels, data) {
  new Chart(document.getElementById(id), {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{ data, backgroundColor: PALETTE, borderWidth: 2, borderColor: '#161b22' }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { position: 'right', labels: { boxWidth: 12, padding: 12, font: { size: 11 } } }
      }
    }
  });
}

// ── Data injected by Python ───────────────────────────────────────
const dailyLabels  = {daily_labels_json};
const dailyValues  = {daily_values_json};
const hourLabels   = {hour_labels_json};
const hourValues   = {hour_values_json};
const modelLabels  = {model_labels_json};
const modelValues  = {model_values_json};
const tokenLabels  = {token_labels_json};
const promptValues = {prompt_values_json};
const outputValues = {output_values_json};
const toolData     = {tool_data_json};
const dowLabels    = {dow_labels_json};
const dowValues    = {dow_values_json};
const keywords     = {keywords_json};
const versions     = {versions_json};

// ── Render charts ─────────────────────────────────────────────────
barChart('dailyChart', dailyLabels, dailyValues, BLUE);

barChart('hourChart', hourLabels, hourValues,
  hourValues.map((v, i) => i >= 9 && i <= 17 ? GREEN : BLUE));

doughnutChart('modelChart', modelLabels, modelValues);

// Stacked bar for tokens
new Chart(document.getElementById('tokenChart'), {
  type: 'bar',
  data: {
    labels: tokenLabels,
    datasets: [
      { label: 'Prompt', data: promptValues, backgroundColor: BLUE, borderRadius: 4 },
      { label: 'Output', data: outputValues, backgroundColor: GREEN, borderRadius: 4 }
    ]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { position: 'top', labels: { boxWidth: 12 } } },
    scales: {
      x: { grid: { color: gridColor }, stacked: true },
      y: { grid: { color: gridColor }, beginAtZero: true, stacked: true, ticks: { precision: 0 } }
    }
  }
});

barChart('dowChart', dowLabels, dowValues, YELLOW);

// ── Tool table ────────────────────────────────────────────────────
const toolTotal = toolData.reduce((s, t) => s + t[1], 0);
const tbody = document.getElementById('toolTable');
toolData.forEach(([name, count]) => {
  const pct = toolTotal > 0 ? Math.round(count / toolTotal * 100) : 0;
  const row = document.createElement('tr');
  row.innerHTML = `
    <td>${name}</td>
    <td><span class="badge">${count}</span></td>
    <td>
      <div class="bar-cell">
        <div class="mini-bar" style="width:${pct * 1.5}px;max-width:120px"></div>
        <span style="font-size:.8rem;color:var(--muted)">${pct}%</span>
      </div>
    </td>`;
  tbody.appendChild(row);
});

// ── Keywords ──────────────────────────────────────────────────────
const kwContainer = document.getElementById('keywordTags');
const kwMax = keywords.length > 0 ? keywords[0][1] : 1;
keywords.forEach(([word, count]) => {
  const span = document.createElement('span');
  const size = 0.8 + (count / kwMax) * 0.6;
  span.className = 'tag';
  span.style.fontSize = size + 'rem';
  span.textContent = `${word} (${count})`;
  kwContainer.appendChild(span);
});

// ── Version history ───────────────────────────────────────────────
const vContainer = document.getElementById('versionList');
if (versions.length === 0) {
  vContainer.innerHTML = '<span style="color:var(--muted)">No version history found in logs.</span>';
} else {
  versions.forEach(v => {
    const div = document.createElement('div');
    div.className = 'version-row';
    div.innerHTML = `
      <span>Copilot <strong>${v.copilot}</strong> &nbsp;·&nbsp; VS Code <strong>${v.vscode}</strong></span>
      <span style="color:var(--muted)">${v.ts || ''}</span>`;
    vContainer.appendChild(div);
  });
}
</script>
</body>
</html>
"""


def fmt_ms(ms):
    if not ms:
        return "N/A"
    if ms >= 60000:
        return f"{ms/60000:.1f} min"
    if ms >= 1000:
        return f"{ms/1000:.1f} s"
    return f"{ms} ms"


def fmt_num(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}k"
    return str(n)


def render_html(ins, output_path):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    first = ins["first_request"].strftime("%b %d, %Y") if ins["first_request"] else "N/A"
    last = ins["last_request"].strftime("%b %d, %Y") if ins["last_request"] else "N/A"

    # Daily trend
    daily_labels = list(ins["daily_trend"].keys())
    daily_values = list(ins["daily_trend"].values())

    # Hour of day
    hour_labels = [f"{h:02d}:00" for h in range(24)]
    hour_values = [ins["hour_counts"][h] for h in range(24)]

    # Models
    model_labels = list(ins["model_counts"].keys())[:8]
    model_values = [ins["model_counts"][m] for m in model_labels]

    # Token chart
    token_labels = list(ins["model_tokens"].keys())[:6]
    prompt_values = [ins["model_tokens"][m]["prompt"] for m in token_labels]
    output_values = [ins["model_tokens"][m]["output"] for m in token_labels]

    # Tools
    tool_data = list(ins["tool_call_counts"].items())[:12]

    # Day of week
    dow_labels = list(ins["dow_counts"].keys())
    dow_values = list(ins["dow_counts"].values())

    # Keywords
    keywords = list(ins["top_keywords"].items())[:20]

    # Version history
    versions = [
        {"copilot": v["copilot"], "vscode": v["vscode"],
         "ts": v["ts"].strftime("%Y-%m-%d") if v["ts"] else ""}
        for v in ins["version_history"]
    ]

    top_mode = max(ins["mode_counts"], key=ins["mode_counts"].get, default="N/A")
    success_str = (
        f" · {ins['tool_success_rate']}% success"
        if ins["tool_success_rate"] is not None else ""
    )
    avg_output = (
        round(ins["total_output_tokens"] / ins["total_requests"])
        if ins["total_requests"] > 0 else 0
    )
    rpd = round(ins["total_requests"] / ins["active_days"], 1) if ins["active_days"] > 0 else 0

    substitutions = {
        "generated_at": now,
        "first_date": first,
        "last_date": last,
        "total_sessions": str(ins["total_sessions"]),
        "sessions_multi": str(ins["sessions_with_multiple_requests"]),
        "total_requests": str(ins["total_requests"]),
        "avg_rps": str(ins["avg_requests_per_session"]),
        "max_rps": str(ins["max_requests_per_session"]),
        "active_days": str(ins["active_days"]),
        "requests_per_day": str(rpd),
        "total_tokens": fmt_num(ins["total_prompt_tokens"] + ins["total_output_tokens"]),
        "prompt_tokens": fmt_num(ins["total_prompt_tokens"]),
        "output_tokens": fmt_num(ins["total_output_tokens"]),
        "total_tool_calls": str(ins["total_tool_calls"]),
        "avg_tools_per_session": str(ins["avg_tool_calls_per_session"]),
        "success_rate_str": success_str,
        "avg_elapsed": fmt_ms(ins["avg_elapsed_ms"]),
        "avg_ttft": fmt_ms(ins["avg_ttft_ms"]),
        "p95_elapsed": fmt_ms(ins["p95_elapsed_ms"]),
        "avg_msg_len": str(ins["avg_message_length"]),
        "max_msg_len": str(ins["max_message_length"]),
        "top_mode": top_mode,
        "avg_output_tokens": str(avg_output),
        # JSON data for charts
        "daily_labels_json": json.dumps(daily_labels),
        "daily_values_json": json.dumps(daily_values),
        "hour_labels_json": json.dumps(hour_labels),
        "hour_values_json": json.dumps(hour_values),
        "model_labels_json": json.dumps(model_labels),
        "model_values_json": json.dumps(model_values),
        "token_labels_json": json.dumps(token_labels),
        "prompt_values_json": json.dumps(prompt_values),
        "output_values_json": json.dumps(output_values),
        "tool_data_json": json.dumps(tool_data),
        "dow_labels_json": json.dumps(dow_labels),
        "dow_values_json": json.dumps(dow_values),
        "keywords_json": json.dumps(keywords),
        "versions_json": json.dumps(versions),
    }
    html = HTML_TEMPLATE
    for key, val in substitutions.items():
        html = html.replace("{" + key + "}", val)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path


# ─────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze GitHub Copilot agent logs")
    parser.add_argument(
        "--output", "-o",
        default=str(Path(__file__).parent / "agent_insights_report.html"),
        help="Path for the HTML report (default: agent_insights_report.html)"
    )
    parser.add_argument(
        "--logs-dir", "-d", nargs="*", default=[],
        help="Additional directories to scan for *.jsonl log files"
    )
    parser.add_argument(
        "--print-summary", action="store_true",
        help="Also print a text summary to stdout"
    )
    args = parser.parse_args()

    print("Collecting log files …", flush=True)
    requests, transcript_events, debug_events = aggregate(args.logs_dir)
    print(f"  Found {len(requests)} requests  |  "
          f"{len(transcript_events)} transcript events  |  "
          f"{len(debug_events)} debug events")

    if not requests and not transcript_events and not debug_events:
        print("\nNo log data found. Make sure VS Code with GitHub Copilot Chat is installed,")
        print("or drop *.jsonl log files into the same folder as this script.")
        sys.exit(1)

    print("Computing insights …", flush=True)
    ins = compute_insights(requests, transcript_events, debug_events)

    print("Rendering HTML report …", flush=True)
    out = render_html(ins, args.output)
    print(f"\nReport saved to: {out}")

    if args.print_summary:
        print("\n──────────────── TEXT SUMMARY ────────────────")
        print(f"Sessions         : {ins['total_sessions']}")
        print(f"Total requests   : {ins['total_requests']}")
        print(f"Active days      : {ins['active_days']}")
        print(f"Prompt tokens    : {fmt_num(ins['total_prompt_tokens'])}")
        print(f"Output tokens    : {fmt_num(ins['total_output_tokens'])}")
        print(f"Avg response time: {fmt_ms(ins['avg_elapsed_ms'])}")
        print(f"Total tool calls : {ins['total_tool_calls']}")
        print(f"Top model        : {next(iter(ins['model_counts']), 'N/A')}")
        print(f"Top tool         : {next(iter(ins['tool_call_counts']), 'N/A')}")
        print(f"Top keywords     : {', '.join(list(ins['top_keywords'])[:5])}")
        if ins["version_history"]:
            latest = ins["version_history"][-1]
            print(f"Latest Copilot   : {latest['copilot']} / VS Code {latest['vscode']}")


if __name__ == "__main__":
    main()
