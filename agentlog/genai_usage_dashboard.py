#!/usr/bin/env python3
"""
GenAI Usage Dashboard — Last 100 Days
Analyzes VS Code Copilot chat sessions, transcripts, and debug logs
and renders a rich HTML dashboard scoped to the past 100 days.

Usage:
    python genai_usage_dashboard.py
    python genai_usage_dashboard.py --days 60 --output my_report.html
    python genai_usage_dashboard.py --logs-dir /extra/path --print-summary
"""

import argparse
import html
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path


# ─────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────

def ts_to_dt(ts):
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
    candidates = []
    home = Path.home()
    candidates.append(home / "Library" / "Application Support" / "Code" / "User" / "workspaceStorage")
    candidates.append(home / "Library" / "Application Support" / "Code - Insiders" / "User" / "workspaceStorage")
    candidates.append(home / ".config" / "Code" / "User" / "workspaceStorage")
    appdata = os.environ.get("APPDATA", "")
    if appdata:
        candidates.append(Path(appdata) / "Code" / "User" / "workspaceStorage")
    return [p for p in candidates if p.exists()]


def collect_jsonl_files(extra_dirs):
    roots = find_vscode_log_dirs()
    roots += [Path(d) for d in extra_dirs if Path(d).exists()]
    roots.append(Path(__file__).parent)
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
                yield path, "unknown"


IDENTITY_FIELD_KEYS = {
    "account",
    "accountid",
    "attid",
    "attuid",
    "email",
    "employeeid",
    "login",
    "loginid",
    "mail",
    "objectid",
    "oid",
    "upn",
    "useremail",
    "userid",
    "username",
    "userprincipalname",
}


def normalize_identity_key(key):
    return re.sub(r"[^a-z0-9]+", "", str(key).lower())


def is_scalar_identity_value(value):
    return isinstance(value, (str, int, float, bool))


def extract_identity_fields(obj, prefix=""):
    fields = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_str = str(key)
            path = f"{prefix}.{key_str}" if prefix else key_str
            if normalize_identity_key(key_str) in IDENTITY_FIELD_KEYS and is_scalar_identity_value(value):
                value_str = str(value).strip()
                if value_str:
                    fields[path] = value_str
            if isinstance(value, (dict, list)):
                fields.update(extract_identity_fields(value, path))
    elif isinstance(obj, list):
        for item in obj:
            fields.update(extract_identity_fields(item, prefix))
    return fields


def merge_identity_fields(*sources):
    merged = {}
    for source in sources:
        merged.update(extract_identity_fields(source))
    return merged


def update_identity_catalog(catalog, identity_fields):
    for field, value in identity_fields.items():
        catalog[field].add(value)


def identity_field_matches(field_name, requested_field):
    if not requested_field:
        return True
    requested_norm = normalize_identity_key(requested_field)
    field_leaf = field_name.rsplit(".", 1)[-1]
    return requested_norm in {
        normalize_identity_key(field_name),
        normalize_identity_key(field_leaf),
    }


def record_matches_user_filter(record, user_field="", user_value=""):
    if not user_field and not user_value:
        return True
    identity_fields = record.get("identity_fields", {})
    if not identity_fields:
        return False
    requested_value = str(user_value).strip().lower()
    for field_name, value in identity_fields.items():
        field_ok = identity_field_matches(field_name, user_field)
        value_ok = not requested_value or str(value).strip().lower() == requested_value
        if field_ok and value_ok:
            return True
    return False


def transcript_session_id(event):
    data = event.get("data", {})
    return data.get("sessionId") if isinstance(data, dict) else None


def format_identity_catalog(identity_catalog, max_values=5):
    if not identity_catalog:
        return ["  No identity/user fields detected in scanned logs."]
    lines = []
    for field in sorted(identity_catalog):
        values = sorted(identity_catalog[field])
        sample = ", ".join(values[:max_values])
        suffix = f" (+{len(values) - max_values} more)" if len(values) > max_values else ""
        lines.append(f"  {field}: {sample}{suffix}")
    return lines


def describe_user_filter(user_field="", user_value=""):
    if user_field and user_value:
        return f"{user_field}={user_value}"
    if user_field:
        return f"field {user_field}"
    if user_value:
        return f"any identity field == {user_value}"
    return "none"


# ─────────────────────────────────────────────────────────
#  Parsers
# ─────────────────────────────────────────────────────────

def parse_chat_session(path):
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
                    identity_fields = merge_identity_fields(obj, v, req, result_meta, metadata)
                    tool_rounds = metadata.get("toolCallRounds", [])
                    tool_calls = []
                    for rnd in (tool_rounds if isinstance(tool_rounds, list) else []):
                        for tc in (rnd.get("toolCalls", []) if isinstance(rnd, dict) else []):
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
                        "agent_version": req.get("agent", {}).get("extensionVersion", "") if isinstance(req.get("agent"), dict) else "",
                        "identity_fields": identity_fields,
                    })
    except (OSError, UnicodeDecodeError):
        pass
    return requests


def parse_transcript(path):
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
                  obj["identity_fields"] = extract_identity_fields(obj)
                  events.append(obj)
    except (OSError, UnicodeDecodeError):
        pass
    return events


def parse_debug_log(path):
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
                  obj["identity_fields"] = extract_identity_fields(obj)
                  events.append(obj)
    except (OSError, UnicodeDecodeError):
        pass
    return events


# ─────────────────────────────────────────────────────────
#  Aggregation
# ─────────────────────────────────────────────────────────

def aggregate(extra_dirs, days, user_field="", user_value=""):
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
    all_requests = []
    all_transcript_events = []
    all_debug_events = []
    identity_catalog = defaultdict(set)

    for path, kind in collect_jsonl_files(extra_dirs):
        if kind == "chatSession":
            all_requests.extend(parse_chat_session(path))
        elif kind == "transcript":
            all_transcript_events.extend(parse_transcript(path))
        elif kind == "debugLog":
            all_debug_events.extend(parse_debug_log(path))

    for record in all_requests:
        update_identity_catalog(identity_catalog, record.get("identity_fields", {}))
    for event in all_transcript_events:
        update_identity_catalog(identity_catalog, event.get("identity_fields", {}))
    for event in all_debug_events:
        update_identity_catalog(identity_catalog, event.get("identity_fields", {}))

    window_requests = [
        record for record in all_requests
        if record["timestamp"] and record["timestamp"] >= cutoff
    ]
    filtered_requests = [
        record for record in window_requests
        if record_matches_user_filter(record, user_field, user_value)
    ]

    if user_field or user_value:
        matched_session_ids = {record["session_id"] for record in filtered_requests}
        filtered_transcript_events = [
            event for event in all_transcript_events
            if record_matches_user_filter(event, user_field, user_value)
            or transcript_session_id(event) in matched_session_ids
        ]
        filtered_debug_events = [
            event for event in all_debug_events
            if record_matches_user_filter(event, user_field, user_value)
        ]
    else:
        filtered_transcript_events = all_transcript_events
        filtered_debug_events = all_debug_events

    scan_stats = {
        "raw_requests": len(all_requests),
        "window_requests": len(window_requests),
        "transcript_events": len(all_transcript_events),
        "debug_events": len(all_debug_events),
    }

    return filtered_requests, filtered_transcript_events, filtered_debug_events, cutoff, identity_catalog, scan_stats


# ─────────────────────────────────────────────────────────
#  Insight computation
# ─────────────────────────────────────────────────────────

def compute_insights(requests, transcript_events, debug_events, cutoff, days):
    ins = {}
    ins["days_window"] = days
    ins["cutoff"] = cutoff

    # ── Overview ──────────────────────────────────────────
    session_ids = {r["session_id"] for r in requests}
    ins["total_sessions"] = len(session_ids)
    ins["total_requests"] = len(requests)

    timestamps = [r["timestamp"] for r in requests if r["timestamp"]]
    if timestamps:
        ins["first_request"] = min(timestamps)
        ins["last_request"] = max(timestamps)
        delta = ins["last_request"] - ins["first_request"]
        ins["active_days"] = len({r["timestamp"].date() for r in requests if r["timestamp"]})
    else:
        ins["first_request"] = ins["last_request"] = None
        ins["active_days"] = 0

    # ── Token totals ──────────────────────────────────────
    ins["total_prompt_tokens"] = sum(r["prompt_tokens"] or 0 for r in requests)
    ins["total_output_tokens"] = sum(r["output_tokens"] or 0 for r in requests)
    ins["total_tokens"] = ins["total_prompt_tokens"] + ins["total_output_tokens"]

    # ── Model usage ───────────────────────────────────────
    model_counts = Counter()
    model_tokens = defaultdict(lambda: {"prompt": 0, "output": 0})
    for r in requests:
        mid = (r.get("resolved_model") or r.get("model_id") or "unknown").replace("copilot/", "")
        model_counts[mid] += 1
        model_tokens[mid]["prompt"] += r["prompt_tokens"] or 0
        model_tokens[mid]["output"] += r["output_tokens"] or 0
    ins["model_counts"] = dict(model_counts.most_common())
    ins["model_tokens"] = dict(model_tokens)

    # ── Latency ───────────────────────────────────────────
    elapsed = sorted(r["elapsed_ms"] for r in requests if r["elapsed_ms"])
    ttft    = [r["first_progress_ms"] for r in requests if r["first_progress_ms"]]
    ins["avg_elapsed_ms"]    = int(sum(elapsed) / len(elapsed)) if elapsed else 0
    ins["median_elapsed_ms"] = elapsed[len(elapsed) // 2] if elapsed else 0
    ins["p95_elapsed_ms"]    = elapsed[int(len(elapsed) * 0.95)] if elapsed else 0
    ins["avg_ttft_ms"]       = int(sum(ttft) / len(ttft)) if ttft else 0

    # ── Tool usage ────────────────────────────────────────
    all_tool_calls = []
    for r in requests:
        all_tool_calls.extend(r["tool_calls"])
    for e in transcript_events:
        if e.get("type") == "tool.execution_start":
            d = e.get("data", {})
            if isinstance(d, dict) and d.get("toolName"):
                all_tool_calls.append(d["toolName"])
    ins["tool_call_counts"] = dict(Counter(all_tool_calls).most_common(15))
    ins["total_tool_calls"]  = len(all_tool_calls)

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

    # ── Mode breakdown ────────────────────────────────────
    ins["mode_counts"] = dict(Counter(r["mode"] for r in requests).most_common())

    # ── Hour-of-day heatmap ───────────────────────────────
    hour_counts = Counter(r["timestamp"].hour for r in requests if r["timestamp"])
    ins["hour_counts"] = {h: hour_counts.get(h, 0) for h in range(24)}

    # ── Day-of-week ───────────────────────────────────────
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_counts = Counter(r["timestamp"].weekday() for r in requests if r["timestamp"])
    ins["dow_counts"] = {dow_names[i]: dow_counts.get(i, 0) for i in range(7)}

    # ── Full daily trend (all 100 days, 0s for missing) ───
    today = datetime.now(tz=timezone.utc).date()
    day_requests = Counter(r["timestamp"].date() for r in requests if r["timestamp"])
    day_tokens   = defaultdict(int)
    for r in requests:
        if r["timestamp"]:
            day_tokens[r["timestamp"].date()] += (r["prompt_tokens"] or 0) + (r["output_tokens"] or 0)
    all_days = [(today - timedelta(days=days - 1 - i)) for i in range(days)]
    ins["daily_labels"]   = [str(d) for d in all_days]
    ins["daily_requests"] = [day_requests.get(d, 0) for d in all_days]
    ins["daily_tokens"]   = [day_tokens.get(d, 0) for d in all_days]

    # ── Cumulative tokens over window ─────────────────────
    cumulative = 0
    ins["cumulative_tokens"] = []
    for d in all_days:
        cumulative += day_tokens.get(d, 0)
        ins["cumulative_tokens"].append(cumulative)

    # ── Weekly buckets (for bar chart) ───────────────────
    week_labels  = []
    week_requests= []
    week_tokens_list = []
    for w in range(0, days, 7):
        week_days = all_days[w:w+7]
        label = str(week_days[0])
        week_labels.append(label)
        week_requests.append(sum(day_requests.get(d, 0) for d in week_days))
        week_tokens_list.append(sum(day_tokens.get(d, 0) for d in week_days))
    ins["week_labels"]   = week_labels
    ins["week_requests"] = week_requests
    ins["week_tokens"]   = week_tokens_list

    # ── Session depth distribution ────────────────────────
    reqs_per_session = Counter(r["session_id"] for r in requests)
    counts = list(reqs_per_session.values())
    buckets = {"1": 0, "2-5": 0, "6-10": 0, "11-20": 0, "21+": 0}
    for c in counts:
        if c == 1:       buckets["1"]    += 1
        elif c <= 5:     buckets["2-5"]  += 1
        elif c <= 10:    buckets["6-10"] += 1
        elif c <= 20:    buckets["11-20"]+= 1
        else:            buckets["21+"]  += 1
    ins["session_depth_labels"] = list(buckets.keys())
    ins["session_depth_values"] = list(buckets.values())
    ins["avg_requests_per_session"] = round(sum(counts) / len(counts), 1) if counts else 0
    ins["max_requests_per_session"] = max(counts, default=0)
    ins["sessions_with_multiple_requests"] = sum(1 for c in counts if c > 1)

    # ── Tool calls per session ────────────────────────────
    tcps_map = defaultdict(int)
    for r in requests:
        tcps_map[r["session_id"]] += len(r["tool_calls"])
    tcps = list(tcps_map.values())
    ins["avg_tool_calls_per_session"] = round(sum(tcps) / len(tcps), 1) if tcps else 0

    # ── Message length ────────────────────────────────────
    msg_lengths = [len(r["message_text"]) for r in requests if r["message_text"]]
    ins["avg_message_length"] = int(sum(msg_lengths) / len(msg_lengths)) if msg_lengths else 0
    ins["max_message_length"] = max(msg_lengths, default=0)

    # ── Top keywords ──────────────────────────────────────
    stop_words = {
        "the","a","an","and","or","but","in","on","at","to","for","of","with","as",
        "is","are","was","were","be","been","being","have","has","had","do","does",
        "did","will","would","could","should","may","might","shall","can","i","you",
        "it","this","that","my","your","me","we","us","he","she","they","them","from",
        "by","not","so","if","up","out","what","how","all","just","like","make",
        "please","use","using","add","new","get","into","want","need","can","file",
        "code","there","then","also","some","any","when","here","about","after","before"
    }
    word_freq = Counter()
    for r in requests:
        words = re.findall(r"[a-z]{3,}", r["message_text"].lower())
        word_freq.update(w for w in words if w not in stop_words)
    ins["top_keywords"] = dict(word_freq.most_common(25))

    # ── Activity heatmap grid (last N days as calendar) ───
    # 15 columns × 7 rows, newest day = bottom-right
    heatmap_days = min(days, 105)  # round up to 15 weeks
    heatmap_grid = []
    max_reqs = max(day_requests.values(), default=1)
    for i in range(days - 1, -1, -1):
        d = today - timedelta(days=i)
        cnt = day_requests.get(d, 0)
        heatmap_grid.append({"date": str(d), "count": cnt, "max": max_reqs})
    ins["heatmap_grid"] = heatmap_grid[-105:]   # at most 15×7 cells

    # ── Hourly-DoW 2-D heatmap ────────────────────────────
    hour_dow = defaultdict(int)
    for r in requests:
        if r["timestamp"]:
            hour_dow[(r["timestamp"].hour, r["timestamp"].weekday())] += 1
    ins["hour_dow_matrix"] = [[hour_dow.get((h, d), 0) for d in range(7)] for h in range(24)]

    # ── Version info ──────────────────────────────────────
    versions = []
    for e in debug_events:
        attrs = e.get("attrs", {})
        if isinstance(attrs, dict) and attrs.get("copilotVersion"):
            versions.append({"copilot": attrs.get("copilotVersion",""), "vscode": attrs.get("vscodeVersion",""), "ts": ts_to_dt(e.get("ts"))})
    for e in transcript_events:
        if e.get("type") == "session.start":
            d = e.get("data", {})
            if isinstance(d, dict) and d.get("copilotVersion"):
                versions.append({"copilot": d.get("copilotVersion",""), "vscode": d.get("vscodeVersion",""), "ts": ts_to_dt(d.get("startTime"))})
    versions.sort(key=lambda x: x["ts"] or datetime.min.replace(tzinfo=timezone.utc))
    seen_v = None
    unique_v = []
    for v in versions:
        if v["copilot"] != seen_v:
            unique_v.append(v)
            seen_v = v["copilot"]
    ins["latest_copilot"] = unique_v[-1]["copilot"] if unique_v else "N/A"
    ins["latest_vscode"]  = unique_v[-1]["vscode"]  if unique_v else "N/A"

    # ── Cost estimation (API equivalent value) ─────────────
    MODEL_PRICING = {
        "gpt-4o":            (2.50,  10.00),
        "gpt-4o-mini":       (0.15,   0.60),
        "gpt-4-turbo":       (10.00, 30.00),
        "gpt-4":             (30.00, 60.00),
        "gpt-3.5-turbo":     (0.50,   1.50),
        "o1":                (15.00, 60.00),
        "o1-mini":           (1.10,   4.40),
        "o3-mini":           (1.10,   4.40),
        "o3":                (10.00, 40.00),
        "gpt-5-mini":        (0.15,   0.60),
        "gpt-5":             (15.00, 60.00),
        "claude-3-5-sonnet": (3.00,  15.00),
        "claude-sonnet-4":   (3.00,  15.00),
        "claude-sonnet":     (3.00,  15.00),
        "claude-3-haiku":    (0.25,   1.25),
        "claude-3-opus":     (15.00, 75.00),
        "gemini-1.5-pro":    (1.25,   5.00),
        "gemini-2.0-flash":  (0.10,   0.40),
    }
    DEFAULT_PRICING = (3.00, 15.00)
    total_cost_usd = 0.0
    cost_by_model = {}
    for mid, tkns in model_tokens.items():
        pricing = next(
            (v for k, v in MODEL_PRICING.items() if k in mid.lower()),
            DEFAULT_PRICING
        )
        cost = (tkns["prompt"] / 1_000_000 * pricing[0]) + (tkns["output"] / 1_000_000 * pricing[1])
        cost_by_model[mid] = round(cost, 4)
        total_cost_usd += cost
    ins["estimated_cost_usd"] = round(total_cost_usd, 2)
    ins["cost_by_model"] = dict(sorted(cost_by_model.items(), key=lambda x: x[1], reverse=True))

    # ── Streak analysis ────────────────────────────────────
    active_date_set = sorted({r["timestamp"].date() for r in requests if r["timestamp"]})
    current_streak = 0
    longest_streak = 0
    if active_date_set:
        streak = 1
        for i in range(1, len(active_date_set)):
            if (active_date_set[i] - active_date_set[i-1]).days == 1:
                streak += 1
                longest_streak = max(longest_streak, streak)
            else:
                streak = 1
        longest_streak = max(longest_streak, 1)
        check = today
        for d in reversed(active_date_set):
            if d == check:
                current_streak += 1
                check = check - timedelta(days=1)
            elif d < check:
                break
    ins["current_streak"] = current_streak
    ins["longest_streak"] = longest_streak

    # ── Request intent categorization ─────────────────────
    INTENT_PATTERNS = [
        ("Code Generation", r"\bgenerate|write|create|implement|build|scaffold\b"),
        ("Debugging",       r"\bdebug|fix|error|bug|issue|broken|fail|crash|exception|traceback\b"),
        ("Explanation",     r"\bexplain|what is|how does|describe|understand|meaning|why\b"),
        ("Refactoring",     r"\brefactor|rewrite|improve|optimize|clean|simplify|restructure\b"),
        ("Code Review",     r"\breview|check|validate|verify|audit|look at\b"),
        ("Testing",         r"\btest|unit test|coverage|mock|assert|spec|pytest\b"),
        ("Documentation",   r"\bdoc|comment|readme|docstring|annotate|changelog\b"),
    ]
    intent_counter = Counter()
    for r in requests:
        text = r["message_text"].lower()
        matched = False
        for intent, pattern in INTENT_PATTERNS:
            if re.search(pattern, text):
                intent_counter[intent] += 1
                matched = True
                break
        if not matched:
            intent_counter["Chat / Q&A"] += 1
    ins["intent_counts"] = dict(intent_counter.most_common())

    # ── Latency percentiles ────────────────────────────────
    def _pct(lst, p):
        return lst[int(len(lst) * p / 100)] if lst else 0
    ins["p50_elapsed_ms"] = _pct(elapsed, 50)
    ins["p75_elapsed_ms"] = _pct(elapsed, 75)
    ins["p90_elapsed_ms"] = _pct(elapsed, 90)
    ins["p99_elapsed_ms"] = _pct(elapsed, 99)

    # ── 7-day rolling average ──────────────────────────────
    rolling = []
    for i in range(len(ins["daily_requests"])):
        window = ins["daily_requests"][max(0, i - 6):i + 1]
        rolling.append(round(sum(window) / len(window), 2))
    ins["rolling_avg_7d"] = rolling

    # ── Token efficiency ratio ─────────────────────────────
    ins["token_efficiency"] = (
        round(ins["total_output_tokens"] / ins["total_prompt_tokens"], 3)
        if ins["total_prompt_tokens"] > 0 else 0
    )

    # ── Top 10 sessions by request count ──────────────────
    session_req_counts = Counter(r["session_id"] for r in requests)
    session_first_ts   = {}
    session_models_map = defaultdict(set)
    for r in requests:
        sid = r["session_id"]
        ts  = r["timestamp"]
        if ts and (sid not in session_first_ts or ts < session_first_ts[sid]):
            session_first_ts[sid] = ts
        mid = (r.get("resolved_model") or r.get("model_id") or "?").replace("copilot/", "")
        session_models_map[sid].add(mid)
    ins["top_sessions"] = [
        {
            "id":       sid[:8],
            "requests": cnt,
            "date":     session_first_ts[sid].strftime("%Y-%m-%d") if sid in session_first_ts else "",
            "model":    ", ".join(sorted(session_models_map[sid])),
            "tokens":   sum((r["prompt_tokens"] or 0) + (r["output_tokens"] or 0)
                            for r in requests if r["session_id"] == sid),
        }
        for sid, cnt in session_req_counts.most_common(10)
    ]

    # ── Language / file extension mentions ────────────────
    LANG_PATTERNS = [
        ("Python",      r"\.py\b|\bpython\b"),
        ("TypeScript",  r"\.ts\b|\btypescript\b"),
        ("JavaScript",  r"\.js\b|\bjavascript\b"),
        ("React/JSX",   r"\.jsx\b|\.tsx\b|\breact\b"),
        ("Rust",        r"\.rs\b|\brust\b"),
        ("Go",          r"\bgolang\b|\.go\b"),
        ("Java",        r"\.java\b|\bjava\b"),
        ("C/C++",       r"\.cpp\b|\.c\b|\.h\b|\bc\+\+\b"),
        ("SQL",         r"\.sql\b|\bsql\b|\bquery\b"),
        ("YAML/JSON",   r"\.ya?ml\b|\.json\b|\byaml\b|\bjson\b"),
        ("Shell",       r"\.sh\b|\bbash\b|\bshell\b|\bpowershell\b"),
        ("CSS/HTML",    r"\.css\b|\.html\b|\bcss\b|\bhtml\b"),
        ("Markdown",    r"\.md\b|\bmarkdown\b"),
        ("Docker/K8s",  r"\bdocker\b|\bkubernetes\b|\bk8s\b|dockerfile"),
    ]
    lang_counter = Counter()
    for r in requests:
        text = r["message_text"].lower()
        for lang, pat in LANG_PATTERNS:
            if re.search(pat, text):
                lang_counter[lang] += 1
    ins["lang_counts"] = dict(lang_counter.most_common())

    # ── Intensity score (0–100) ────────────────────────────
    adr       = min(ins["active_days"] / max(days, 1), 1.0) * 40
    rps_score = min(ins["avg_requests_per_session"] / 10.0, 1.0) * 30
    tcs_score = min(ins["avg_tool_calls_per_session"] / 20.0, 1.0) * 30
    ins["intensity_score"] = round(adr + rps_score + tcs_score, 1)

    return ins


# ─────────────────────────────────────────────────────────
#  HTML template
# ─────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>GenAI Usage Dashboard — Last {days} Days</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script>
(() => {
  try {
    const key = 'genai-dashboard-theme';
    const saved = localStorage.getItem(key);
    const preferred = window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches
      ? 'light'
      : 'dark';
    document.documentElement.dataset.theme = saved || preferred;
  } catch (error) {
    document.documentElement.dataset.theme = 'dark';
  }
})();
</script>
<style>
@import url("https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap");
:root{
  --bg:#07111a;
  --bg-deep:#0b1822;
  --surface:rgba(13,28,39,.88);
  --surface-strong:rgba(18,37,50,.94);
  --surface-soft:rgba(18,37,50,.72);
  --border:rgba(145,214,205,.16);
  --line:rgba(255,255,255,.08);
  --accent:#53d1c1;
  --accent-soft:#baf4ed;
  --green:#7fe39d;
  --yellow:#ffca7a;
  --red:#ff8d73;
  --purple:#8fb0ff;
  --cyan:#6fe7ff;
  --orange:#ff9e64;
  --text:#f3f7f6;
  --muted:#97afb8;
  --muted-strong:#c7d8dd;
  --shadow:0 24px 80px rgba(0,0,0,.34);
  --wash-left:rgba(83,209,193,.18);
  --wash-right:rgba(255,158,100,.16);
  --bg-start:#06111a;
  --bg-mid:#0a1721;
  --bg-end:#08121a;
  --grid-overlay:rgba(255,255,255,.025);
  --hero-start:rgba(16,39,53,.96);
  --hero-end:rgba(11,24,34,.92);
  --hero-wash:rgba(255,158,100,.18);
  --hero-gloss:rgba(255,255,255,.08);
  --hero-accent-glow:rgba(83,209,193,.08);
  --chart-grid:rgba(255,255,255,.08);
  --chart-text:#97afb8;
  --tooltip-bg:rgba(7,17,26,.94);
  --tooltip-title:#f3f7f6;
  --tooltip-body:#d7e6ea;
  --tooltip-border:rgba(83,209,193,.28);
  --heatmap-empty:#161b22;
  --heatmap-1:#0e4429;
  --heatmap-2:#006d32;
  --heatmap-3:#26a641;
  --heatmap-4:#39d353;
  --intensity-rgb:88,166,255;
  --r:18px;
}
[data-theme="light"]{
  --bg:#f4fbf8;
  --bg-deep:#edf8f4;
  --surface:rgba(255,255,255,.84);
  --surface-strong:rgba(255,255,255,.92);
  --surface-soft:rgba(243,250,248,.88);
  --border:rgba(16,35,46,.10);
  --line:rgba(16,35,46,.08);
  --accent:#0f8b8d;
  --accent-soft:#0a6063;
  --green:#1c9a5f;
  --yellow:#a86b00;
  --red:#cf5d37;
  --purple:#4769d8;
  --cyan:#0b88bc;
  --orange:#db7b1d;
  --text:#10232e;
  --muted:#617983;
  --muted-strong:#36505b;
  --shadow:0 18px 54px rgba(27,61,73,.12);
  --wash-left:rgba(15,139,141,.14);
  --wash-right:rgba(219,123,29,.12);
  --bg-start:#eef8f5;
  --bg-mid:#f7fbfa;
  --bg-end:#edf6f3;
  --grid-overlay:rgba(16,35,46,.05);
  --hero-start:rgba(255,255,255,.96);
  --hero-end:rgba(239,248,245,.94);
  --hero-wash:rgba(15,139,141,.12);
  --hero-gloss:rgba(255,255,255,.62);
  --hero-accent-glow:rgba(15,139,141,.08);
  --chart-grid:rgba(16,35,46,.12);
  --chart-text:#617983;
  --tooltip-bg:rgba(255,255,255,.96);
  --tooltip-title:#10232e;
  --tooltip-body:#45606a;
  --tooltip-border:rgba(15,139,141,.18);
  --heatmap-empty:#dfeae6;
  --heatmap-1:#c2e4ce;
  --heatmap-2:#8cd5a4;
  --heatmap-3:#47bc76;
  --heatmap-4:#189a57;
  --intensity-rgb:15,139,141;
}
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{
  font-family:"Manrope","Segoe UI",sans-serif;
  background:
    radial-gradient(circle at 0% 0%, var(--wash-left), transparent 28%),
    radial-gradient(circle at 100% 0%, var(--wash-right), transparent 24%),
    linear-gradient(180deg, var(--bg-start) 0%, var(--bg-mid) 42%, var(--bg-end) 100%);
  color:var(--text);
  line-height:1.6;
  padding:40px 20px 72px;
  min-height:100vh;
  overflow-x:hidden;
}
body::before{
  content:"";
  position:fixed;
  inset:0;
  background-image:
    linear-gradient(var(--grid-overlay) 1px, transparent 1px),
    linear-gradient(90deg, var(--grid-overlay) 1px, transparent 1px);
  background-size:28px 28px;
  mask-image:radial-gradient(circle at center, black 30%, transparent 90%);
  opacity:.35;
  pointer-events:none;
}
.page-shell{max-width:1460px;margin:0 auto;position:relative;z-index:1}
.hero{
  display:grid;
  grid-template-columns:minmax(0,1.4fr) minmax(320px,.9fr);
  gap:18px;
  padding:26px;
  margin-bottom:28px;
  background:
    linear-gradient(135deg, var(--hero-start), var(--hero-end)),
    radial-gradient(circle at top right, var(--hero-wash), transparent 30%);
  border:1px solid rgba(145,214,205,.22);
  border-radius:28px;
  box-shadow:var(--shadow);
  overflow:hidden;
  position:relative;
}
.hero::before{
  content:"";
  position:absolute;
  inset:0;
  background:linear-gradient(115deg, var(--hero-gloss), transparent 35%, transparent 60%, var(--hero-accent-glow));
  pointer-events:none;
}
.hero-copy,.hero-stat-grid{position:relative;z-index:1}
.kicker{
  font-size:.78rem;
  letter-spacing:.18em;
  text-transform:uppercase;
  color:var(--accent-soft);
  font-weight:700;
  margin-bottom:10px;
}
h1{
  font-family:"Space Grotesk","Segoe UI",sans-serif;
  font-size:clamp(2rem,4vw,3.3rem);
  line-height:1.05;
  margin-bottom:10px;
  letter-spacing:-.04em;
}
.subtitle{
  color:var(--muted-strong);
  margin-bottom:16px;
  font-size:1rem;
  max-width:860px;
}
.hero-meta{display:flex;flex-wrap:wrap;gap:10px;align-items:center}
.badge{
  display:inline-flex;
  align-items:center;
  gap:6px;
  padding:6px 12px;
  border-radius:999px;
  font-size:.76rem;
  font-weight:700;
  background:linear-gradient(180deg, rgba(83,209,193,.16), rgba(83,209,193,.08));
  color:var(--accent-soft);
  border:1px solid rgba(83,209,193,.22);
  backdrop-filter:blur(10px);
}
.theme-toggle{
  appearance:none;
  border:1px solid rgba(255,255,255,.10);
  background:linear-gradient(180deg, rgba(255,255,255,.10), rgba(255,255,255,.04));
  color:var(--text);
  padding:8px 14px;
  border-radius:999px;
  font-family:"Space Grotesk","Segoe UI",sans-serif;
  font-size:.76rem;
  font-weight:700;
  letter-spacing:.08em;
  text-transform:uppercase;
  cursor:pointer;
  transition:transform .18s ease,border-color .18s ease,background .18s ease,box-shadow .18s ease;
  box-shadow:0 10px 24px rgba(0,0,0,.12);
}
.theme-toggle:hover{
  transform:translateY(-1px);
  border-color:rgba(83,209,193,.34);
  background:linear-gradient(180deg, rgba(83,209,193,.16), rgba(83,209,193,.06));
}
.theme-toggle:focus-visible{
  outline:2px solid var(--accent);
  outline-offset:2px;
}
.hero-stat-grid{
  display:grid;
  grid-template-columns:repeat(2,minmax(0,1fr));
  gap:12px;
  align-content:start;
}
.hero-stat{
  padding:16px 18px;
  border-radius:20px;
  background:linear-gradient(180deg, rgba(255,255,255,.09), rgba(255,255,255,.03));
  border:1px solid rgba(255,255,255,.1);
  backdrop-filter:blur(8px);
}
.hero-stat-label{
  display:block;
  font-size:.72rem;
  letter-spacing:.12em;
  text-transform:uppercase;
  color:var(--muted);
  margin-bottom:8px;
}
.hero-stat strong{
  display:block;
  font-family:"Space Grotesk","Segoe UI",sans-serif;
  font-size:2rem;
  line-height:1;
  color:var(--text);
  margin-bottom:4px;
}
.hero-stat span:last-child{color:var(--muted-strong);font-size:.82rem}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
.section{margin-top:26px}
.section-title{
  display:flex;
  align-items:center;
  gap:12px;
  font-family:"Space Grotesk","Segoe UI",sans-serif;
  font-size:1.15rem;
  font-weight:700;
  color:var(--text);
  margin-bottom:18px;
}
.section-title::before{
  content:"";
  width:38px;
  height:4px;
  border-radius:99px;
  background:linear-gradient(90deg,var(--accent),var(--orange));
}
.grid{display:grid;gap:16px}
.g2{grid-template-columns:repeat(auto-fit,minmax(260px,1fr))}
.g3{grid-template-columns:repeat(auto-fit,minmax(220px,1fr))}
.g4{grid-template-columns:repeat(auto-fit,minmax(180px,1fr))}
.gc{grid-template-columns:repeat(auto-fit,minmax(360px,1fr))}
.card{
  background:linear-gradient(180deg, var(--surface-strong), rgba(11,24,34,.92));
  border:1px solid var(--border);
  border-radius:var(--r);
  padding:22px;
  box-shadow:var(--shadow);
  position:relative;
  overflow:hidden;
  backdrop-filter:blur(14px);
  transition:transform .24s ease,border-color .24s ease,box-shadow .24s ease;
}
.card::before{
  content:"";
  position:absolute;
  inset:0;
  background:linear-gradient(135deg, rgba(255,255,255,.05), transparent 34%);
  pointer-events:none;
}
.card:hover{
  transform:translateY(-4px);
  border-color:rgba(83,209,193,.34);
  box-shadow:0 30px 90px rgba(0,0,0,.38);
}
.stat-label{font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.14em}
.stat-value{
  font-family:"Space Grotesk","Segoe UI",sans-serif;
  font-size:2.1rem;
  font-weight:700;
  line-height:1.08;
  margin-top:8px;
  letter-spacing:-.04em;
}
.stat-sub{font-size:.84rem;color:var(--muted-strong);margin-top:6px}
.blue{color:var(--accent)}.green{color:var(--green)}.yellow{color:var(--yellow)}
.red{color:var(--red)}.purple{color:var(--purple)}.cyan{color:var(--cyan)}
.chart-h200{position:relative;height:200px}
.chart-h260{position:relative;height:260px}
.chart-h300{position:relative;height:300px}
.chart-h360{position:relative;height:360px}
table{
  width:100%;
  border-collapse:separate;
  border-spacing:0;
  font-size:.87rem;
  overflow:hidden;
  border:1px solid rgba(255,255,255,.05);
  border-radius:14px;
  background:rgba(255,255,255,.02);
}
th{
  text-align:left;
  padding:10px 12px;
  color:var(--muted);
  font-weight:700;
  border-bottom:1px solid var(--line);
  background:rgba(255,255,255,.03);
}
td{padding:10px 12px;border-bottom:1px solid var(--line);color:var(--muted-strong)}
tbody tr:nth-child(even) td{background:rgba(255,255,255,.015)}
tbody tr:hover td{background:rgba(83,209,193,.05)}
tr:last-child td{border-bottom:none}
.bar-cell{display:flex;align-items:center;gap:8px}
.mini-bar{height:7px;border-radius:999px;background:linear-gradient(90deg,var(--accent),var(--cyan));flex-shrink:0}
.tag{
  display:inline-block;
  padding:5px 11px;
  border-radius:999px;
  font-size:.81rem;
  margin:4px;
  background:linear-gradient(180deg, rgba(255,255,255,.08), rgba(255,255,255,.03));
  border:1px solid rgba(255,255,255,.07);
  color:var(--text);
}
code{
  font-family:"Space Grotesk",Consolas,monospace;
  background:rgba(83,209,193,.08);
  color:var(--accent-soft);
  padding:4px 8px;
  border-radius:999px;
  border:1px solid rgba(83,209,193,.12);
}
.heat-legend{
  width:13px;
  height:13px;
  border-radius:4px;
  display:inline-block;
  border:1px solid rgba(255,255,255,.06);
}
.heat-0{background:var(--heatmap-empty)}
.heat-1{background:var(--heatmap-1)}
.heat-2{background:var(--heatmap-2)}
.heat-3{background:var(--heatmap-3)}
.heat-4{background:var(--heatmap-4)}
#heatmap{display:flex;flex-wrap:wrap;gap:5px;justify-content:flex-start;padding:6px 0}
.hm-cell{
  width:14px;
  height:14px;
  border-radius:4px;
  flex-shrink:0;
  cursor:default;
  border:1px solid rgba(255,255,255,.04);
  box-shadow:inset 0 0 0 1px rgba(255,255,255,.02);
}
.hm-cell[title]:hover{outline:1px solid rgba(255,255,255,.6)}
.intensity-grid{
  display:grid;
  gap:4px;
  padding:10px;
  border:1px solid rgba(255,255,255,.05);
  border-radius:16px;
  background:rgba(255,255,255,.02);
}
.ig-cell{border-radius:4px;aspect-ratio:1/1;border:1px solid rgba(255,255,255,.03)}
footer{
  text-align:center;
  color:var(--muted);
  font-size:.8rem;
  margin-top:56px;
  padding-top:18px;
  border-top:1px solid rgba(255,255,255,.07);
}
.pill{
  display:inline-block;
  padding:4px 10px;
  border-radius:999px;
  font-size:.75rem;
  font-weight:700;
  margin:2px;
  background:rgba(255,255,255,.06);
  border:1px solid rgba(255,255,255,.08);
}
/* User Filter UI */
.user-filter-panel{
  display:flex;
  flex-wrap:wrap;
  gap:8px;
  align-items:center;
  margin-top:12px;
  padding:12px 16px;
  background:linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.02));
  border:1px solid rgba(255,255,255,.08);
  border-radius:14px;
}
.user-filter-panel label{
  font-size:.72rem;
  letter-spacing:.1em;
  text-transform:uppercase;
  color:var(--muted);
  font-weight:700;
}
.user-filter-panel select,
.user-filter-panel input[type="text"]{
  appearance:none;
  border:1px solid rgba(255,255,255,.12);
  background:rgba(255,255,255,.06);
  color:var(--text);
  padding:7px 12px;
  border-radius:8px;
  font-family:inherit;
  font-size:.82rem;
  min-width:140px;
}
.user-filter-panel select:focus,
.user-filter-panel input[type="text"]:focus{
  outline:none;
  border-color:var(--accent);
  box-shadow:0 0 0 2px rgba(83,209,193,.18);
}
.user-filter-panel select option{
  background:var(--bg-deep);
  color:var(--text);
}
.user-filter-panel .filter-btn{
  appearance:none;
  border:1px solid rgba(83,209,193,.3);
  background:linear-gradient(180deg, rgba(83,209,193,.18), rgba(83,209,193,.08));
  color:var(--accent-soft);
  padding:7px 16px;
  border-radius:8px;
  font-family:"Space Grotesk","Segoe UI",sans-serif;
  font-size:.78rem;
  font-weight:700;
  cursor:pointer;
  transition:transform .15s ease,background .15s ease;
}
.user-filter-panel .filter-btn:hover{
  transform:translateY(-1px);
  background:linear-gradient(180deg, rgba(83,209,193,.26), rgba(83,209,193,.12));
}
.user-filter-panel .filter-btn.clear-btn{
  border-color:rgba(255,141,115,.3);
  background:linear-gradient(180deg, rgba(255,141,115,.14), rgba(255,141,115,.06));
  color:var(--red);
}
.user-filter-panel .filter-btn.clear-btn:hover{
  background:linear-gradient(180deg, rgba(255,141,115,.22), rgba(255,141,115,.10));
}
.filter-status{
  font-size:.78rem;
  color:var(--muted-strong);
  margin-left:auto;
}
.filter-status.active{
  color:var(--accent);
  font-weight:600;
}
.no-identity-hint{
  font-size:.78rem;
  color:var(--muted);
  font-style:italic;
}
@media (max-width:900px){
  .hero{grid-template-columns:1fr;padding:22px}
  .hero-stat-grid{grid-template-columns:repeat(2,minmax(0,1fr))}
}
@media (max-width:640px){
  body{padding:22px 14px 56px}
  .hero{border-radius:22px}
  .hero-stat-grid{grid-template-columns:1fr}
  .gc,.g4,.g3,.g2{grid-template-columns:1fr}
  .card{padding:18px}
  .section-title{font-size:1.02rem}
}
</style>
</head>
<body>
<main class="page-shell">
<header class="hero">
  <div class="hero-copy">
    <div class="kicker">Personal GenAI Telemetry</div>
    <h1>GenAI Usage Dashboard</h1>
    <p class="subtitle">
      Last <strong>{days} days</strong> &nbsp;·&nbsp;
      {first_date} → {last_date} &nbsp;·&nbsp;
      Generated {generated_at}
    </p>
    <div class="hero-meta">
      <span class="badge">Copilot {latest_copilot}</span>
      <span class="badge">VS Code {latest_vscode}</span>
      <span class="badge">Intensity {intensity_score}/100</span>
      <span class="badge">Estimated value ${estimated_cost_usd}</span>
      {identity_filter_badge}
      <button id="themeToggle" class="theme-toggle" type="button">Theme: Dark</button>
    </div>
    <div class="user-filter-panel" id="userFilterPanel">
      <label for="filterField">Filter by User</label>
      <select id="filterField">
        <option value="">All Users</option>
      </select>
      <input type="text" id="filterValue" placeholder="Enter value (e.g., ATTUID)" />
      <button class="filter-btn" id="applyFilterBtn" type="button">Apply Filter</button>
      <button class="filter-btn clear-btn" id="clearFilterBtn" type="button" style="display:none">Clear</button>
      <span class="filter-status" id="filterStatus"></span>
    </div>
  </div>
  <div class="hero-stat-grid">
    <div class="hero-stat">
      <span class="hero-stat-label">Requests</span>
      <strong>{total_requests}</strong>
      <span>{avg_rps} per session average</span>
    </div>
    <div class="hero-stat">
      <span class="hero-stat-label">Tokens</span>
      <strong>{total_tokens}</strong>
      <span>{prompt_tokens} prompt / {output_tokens} output</span>
    </div>
    <div class="hero-stat">
      <span class="hero-stat-label">Active Days</span>
      <strong>{active_days}</strong>
      <span>{current_streak} day current streak</span>
    </div>
    <div class="hero-stat">
      <span class="hero-stat-label">Tool Calls</span>
      <strong>{total_tool_calls}</strong>
      <span>{avg_tcs} per session average</span>
    </div>
  </div>
</header>

<!-- ── KPI CARDS ────────────────────────────────────────── -->
<div class="section">
  <div class="section-title">Overview — Last {days} Days</div>
  <div class="grid g4">
    <div class="card">
      <div class="stat-label">Active Days</div>
      <div class="stat-value blue">{active_days}</div>
      <div class="stat-sub">out of {days} in window</div>
    </div>
    <div class="card">
      <div class="stat-label">Sessions</div>
      <div class="stat-value green">{total_sessions}</div>
      <div class="stat-sub">{sessions_multi} multi-turn</div>
    </div>
    <div class="card">
      <div class="stat-label">Requests</div>
      <div class="stat-value yellow">{total_requests}</div>
      <div class="stat-sub">avg {avg_rps} / session · max {max_rps}</div>
    </div>
    <div class="card">
      <div class="stat-label">Requests / Active Day</div>
      <div class="stat-value purple">{rpd}</div>
      <div class="stat-sub">over {active_days} active days</div>
    </div>
    <div class="card">
      <div class="stat-label">Total Tokens</div>
      <div class="stat-value blue">{total_tokens}</div>
      <div class="stat-sub">{prompt_tokens} prompt · {output_tokens} output</div>
    </div>
    <div class="card">
      <div class="stat-label">Tool Calls</div>
      <div class="stat-value green">{total_tool_calls}</div>
      <div class="stat-sub">avg {avg_tcs} / session{success_rate_str}</div>
    </div>
    <div class="card">
      <div class="stat-label">Avg Response Time</div>
      <div class="stat-value yellow">{avg_elapsed}</div>
      <div class="stat-sub">TTFT {avg_ttft} · p95 {p95_elapsed}</div>
    </div>
    <div class="card">
      <div class="stat-label">Avg Output Tokens</div>
      <div class="stat-value purple">{avg_output_tokens}</div>
      <div class="stat-sub">per request · {top_mode} mode</div>
    </div>
  </div>
</div>

<!-- ── ACTIVITY HEATMAP ──────────────────────────────────── -->
<div class="section">
  <div class="section-title">Daily Activity Heatmap</div>
  <div class="card">
    <div style="font-size:.8rem;color:var(--muted);margin-bottom:10px">
      Each cell = one day &nbsp;|&nbsp; darker = more requests &nbsp;|&nbsp; oldest → newest left → right
    </div>
    <div id="heatmap"></div>
    <div style="font-size:.75rem;color:var(--muted);margin-top:8px;display:flex;align-items:center;gap:6px">
      Less
      <span style="display:flex;gap:3px">
        <span class="heat-legend heat-0"></span>
        <span class="heat-legend heat-1"></span>
        <span class="heat-legend heat-2"></span>
        <span class="heat-legend heat-3"></span>
        <span class="heat-legend heat-4"></span>
      </span>
      More
    </div>
  </div>
</div>

<!-- ── TRENDS ────────────────────────────────────────────── -->
<div class="section">
  <div class="section-title">Usage Trends</div>
  <div class="grid gc">
    <div class="card">
      <div style="font-weight:600;margin-bottom:10px">Daily Requests (last {days} days)</div>
      <div class="chart-h260"><canvas id="dailyReqChart"></canvas></div>
    </div>
    <div class="card">
      <div style="font-weight:600;margin-bottom:10px">Cumulative Tokens</div>
      <div class="chart-h260"><canvas id="cumulativeChart"></canvas></div>
    </div>
  </div>
  <div class="grid gc" style="margin-top:14px">
    <div class="card">
      <div style="font-weight:600;margin-bottom:10px">Weekly Request Volume</div>
      <div class="chart-h260"><canvas id="weeklyChart"></canvas></div>
    </div>
    <div class="card">
      <div style="font-weight:600;margin-bottom:10px">Daily Token Usage</div>
      <div class="chart-h260"><canvas id="dailyTokenChart"></canvas></div>
    </div>
  </div>
</div>

<!-- ── MODEL USAGE ───────────────────────────────────────── -->
<div class="section">
  <div class="section-title">Model Usage</div>
  <div class="grid gc">
    <div class="card">
      <div style="font-weight:600;margin-bottom:10px">Requests per Model</div>
      <div class="chart-h260"><canvas id="modelChart"></canvas></div>
    </div>
    <div class="card">
      <div style="font-weight:600;margin-bottom:10px">Token Distribution per Model</div>
      <div class="chart-h260"><canvas id="tokenModelChart"></canvas></div>
    </div>
  </div>
</div>

<!-- ── TIME PATTERNS ─────────────────────────────────────── -->
<div class="section">
  <div class="section-title">Time Patterns</div>
  <div class="grid gc">
    <div class="card">
      <div style="font-weight:600;margin-bottom:10px">Requests by Hour of Day</div>
      <div class="chart-h260"><canvas id="hourChart"></canvas></div>
    </div>
    <div class="card">
      <div style="font-weight:600;margin-bottom:10px">Day-of-Week Breakdown</div>
      <div class="chart-h260"><canvas id="dowChart"></canvas></div>
    </div>
  </div>
  <div class="grid gc" style="margin-top:14px">
    <div class="card">
      <div style="font-weight:600;margin-bottom:12px">Hour × Day-of-Week Intensity (requests)</div>
      <div style="font-size:.78rem;color:var(--muted);margin-bottom:8px">Rows = hours 0–23 &nbsp;|&nbsp; Cols = Mon–Sun</div>
      <div id="intensityGrid" class="intensity-grid" style="grid-template-columns:repeat(7,1fr);max-width:340px"></div>
      <div style="display:flex;gap:16px;margin-top:8px;font-size:.78rem;color:var(--muted)">
        <span>Mon</span><span>Tue</span><span>Wed</span><span>Thu</span><span>Fri</span><span>Sat</span><span>Sun</span>
      </div>
    </div>
    <div class="card">
      <div style="font-weight:600;margin-bottom:10px">Session Depth Distribution</div>
      <div class="chart-h260"><canvas id="sessionDepthChart"></canvas></div>
    </div>
  </div>
</div>

<!-- ── TOOL USAGE ────────────────────────────────────────── -->
<div class="section">
  <div class="section-title">Tool Usage</div>
  <div class="grid gc">
    <div class="card">
      <div style="font-weight:600;margin-bottom:16px">Top Tools Called</div>
      <table>
        <thead><tr><th>Tool</th><th>Calls</th><th>Share</th></tr></thead>
        <tbody id="toolTable"></tbody>
      </table>
    </div>
    <div class="card">
      <div style="font-weight:600;margin-bottom:10px">Tool Call Distribution</div>
      <div class="chart-h260"><canvas id="toolChart"></canvas></div>
    </div>
  </div>
</div>

<!-- ── MESSAGE PATTERNS ──────────────────────────────────── -->
<div class="section">
  <div class="section-title">Message Patterns</div>
  <div class="grid g2">
    <div class="card">
      <div style="font-weight:600;margin-bottom:12px">Top Keywords</div>
      <div id="keywordTags"></div>
    </div>
    <div class="card">
      <table>
        <thead><tr><th>Metric</th><th>Value</th></tr></thead>
        <tbody>
          <tr><td>Avg message length</td><td class="blue">{avg_msg_len} chars</td></tr>
          <tr><td>Max message length</td><td class="blue">{max_msg_len} chars</td></tr>
          <tr><td>Most used mode</td><td class="green">{top_mode}</td></tr>
          <tr><td>Avg prompt tokens / req</td><td class="yellow">{avg_prompt_tokens}</td></tr>
          <tr><td>Avg output tokens / req</td><td class="yellow">{avg_output_tokens}</td></tr>
          <tr><td>Median response time</td><td class="purple">{median_elapsed}</td></tr>
          <tr><td>Token efficiency (out/prompt)</td><td class="cyan">{token_efficiency}</td></tr>
        </tbody>
      </table>
    </div>
  </div>
</div>

<!-- ── STREAKS &amp; PRODUCTIVITY ─────────────────────────── -->
<div class="section">
  <div class="section-title">Streaks &amp; Productivity</div>
  <div class="grid g4">
    <div class="card">
      <div class="stat-label">Current Streak</div>
      <div class="stat-value green">{current_streak}</div>
      <div class="stat-sub">consecutive active days</div>
    </div>
    <div class="card">
      <div class="stat-label">Longest Streak</div>
      <div class="stat-value yellow">{longest_streak}</div>
      <div class="stat-sub">consecutive active days</div>
    </div>
    <div class="card">
      <div class="stat-label">Intensity Score</div>
      <div class="stat-value blue">{intensity_score}</div>
      <div class="stat-sub">out of 100 · composite metric</div>
    </div>
    <div class="card">
      <div class="stat-label">Est. API Value</div>
      <div class="stat-value purple">${estimated_cost_usd}</div>
      <div class="stat-sub">based on public API rates</div>
    </div>
  </div>
  <div class="grid gc" style="margin-top:14px">
    <div class="card">
      <div style="font-weight:600;margin-bottom:10px">Request Intent Breakdown</div>
      <div class="chart-h260"><canvas id="intentChart"></canvas></div>
    </div>
    <div class="card">
      <div style="font-weight:600;margin-bottom:10px">Daily Requests with 7-Day Rolling Avg</div>
      <div class="chart-h260"><canvas id="rollingChart"></canvas></div>
    </div>
  </div>
</div>

<!-- ── LATENCY DEEP DIVE ───────────────────────────────────── -->
<div class="section">
  <div class="section-title">Latency Deep Dive</div>
  <div class="grid gc">
    <div class="card">
      <div style="font-weight:600;margin-bottom:10px">Response Time Percentiles</div>
      <div class="chart-h260"><canvas id="latencyPercChart"></canvas></div>
    </div>
    <div class="card">
      <div style="font-weight:600;margin-bottom:10px">Mode Distribution</div>
      <div class="chart-h260"><canvas id="modeChart"></canvas></div>
    </div>
  </div>
</div>

<!-- ── LANGUAGE &amp; FILE INSIGHTS ─────────────────────────── -->
<div class="section">
  <div class="section-title">Language &amp; File Insights</div>
  <div class="grid gc">
    <div class="card">
      <div style="font-weight:600;margin-bottom:10px">Languages / Files Mentioned in Requests</div>
      <div class="chart-h260"><canvas id="langChart"></canvas></div>
    </div>
    <div class="card">
      <div style="font-weight:600;margin-bottom:10px">Estimated API Value by Model (USD)</div>
      <div class="chart-h260"><canvas id="costChart"></canvas></div>
      <div style="font-size:.75rem;color:var(--muted);margin-top:8px">* Based on public API pricing. Copilot is subscription-based — this shows compute value only.</div>
    </div>
  </div>
</div>

<!-- ── TOP SESSIONS ────────────────────────────────────────── -->
<div class="section">
  <div class="section-title">Top Sessions by Request Count</div>
  <div class="card">
    <table>
      <thead><tr><th>#</th><th>Session ID</th><th>Started</th><th>Requests</th><th>Tokens</th><th>Model(s)</th></tr></thead>
      <tbody id="topSessionsTable"></tbody>
    </table>
  </div>
</div>

<footer>
  GenAI Usage Dashboard &nbsp;·&nbsp; Generated {generated_at} &nbsp;·&nbsp;
  Powered by <a href="https://github.com/features/copilot" target="_blank">GitHub Copilot</a> logs
</footer>

</main>

<script>
// ── Constants ────────────────────────────────────────────
const B='#58a6ff',G='#3fb950',Y='#d29922',R='#f78166',P='#bc8cff',C='#39d353',O='#ff8c00';
const PALETTE=[B,G,Y,R,P,C,O,'#e07b39','#88d4ab','#b392f0'];
const THEME_KEY='genai-dashboard-theme';
const root=document.documentElement;
const themeToggle=document.getElementById('themeToggle');
let GRID='';
let TXT='';

function getThemeColor(name, fallback=''){
  return getComputedStyle(root).getPropertyValue(name).trim() || fallback;
}

function syncChartDefaults(){
  GRID = getThemeColor('--chart-grid', 'rgba(48,54,61,.8)');
  TXT = getThemeColor('--chart-text', '#8b949e');
  Chart.defaults.color = TXT;
  Chart.defaults.font.family = '"Manrope","Segoe UI",sans-serif';
  Chart.defaults.plugins.tooltip.backgroundColor = getThemeColor('--tooltip-bg', 'rgba(7,17,26,.94)');
  Chart.defaults.plugins.tooltip.titleColor = getThemeColor('--tooltip-title', '#f3f7f6');
  Chart.defaults.plugins.tooltip.bodyColor = getThemeColor('--tooltip-body', '#d7e6ea');
  Chart.defaults.plugins.tooltip.borderColor = getThemeColor('--tooltip-border', 'rgba(83,209,193,.28)');
  Chart.defaults.plugins.tooltip.borderWidth = 1;
  Chart.defaults.plugins.tooltip.padding = 12;
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
}

function syncThemeToggleLabel(){
  if(!themeToggle) return;
  const isLight = root.dataset.theme === 'light';
  themeToggle.textContent = `Theme: ${isLight ? 'Light' : 'Dark'}`;
  themeToggle.setAttribute('title', `Switch to ${isLight ? 'dark' : 'light'} theme`);
  themeToggle.setAttribute('aria-label', `Switch to ${isLight ? 'dark' : 'light'} theme`);
}

function updateChartsTheme(){
  Object.values(Chart.instances).forEach(chart=>{
    if(chart.options.scales){
      ['x','y','r'].forEach(axis=>{
        if(chart.options.scales[axis]){
          if(chart.options.scales[axis].grid){
            chart.options.scales[axis].grid.color = GRID;
          }
          if(chart.options.scales[axis].ticks){
            chart.options.scales[axis].ticks.color = TXT;
          }
        }
      });
    }
    if(chart.options.plugins?.legend?.labels){
      chart.options.plugins.legend.labels.color = TXT;
    }
    if(chart.options.plugins?.tooltip){
      chart.options.plugins.tooltip.backgroundColor = getThemeColor('--tooltip-bg', chart.options.plugins.tooltip.backgroundColor);
      chart.options.plugins.tooltip.titleColor = getThemeColor('--tooltip-title', chart.options.plugins.tooltip.titleColor);
      chart.options.plugins.tooltip.bodyColor = getThemeColor('--tooltip-body', chart.options.plugins.tooltip.bodyColor);
      chart.options.plugins.tooltip.borderColor = getThemeColor('--tooltip-border', chart.options.plugins.tooltip.borderColor);
    }
    if(chart.config.type === 'doughnut'){
      chart.data.datasets.forEach(dataset=>{
        dataset.borderColor = getThemeColor('--surface-strong', '#161b22');
      });
    }
    chart.update();
  });
}

syncChartDefaults();
syncThemeToggleLabel();

function mkBar(id,labels,data,color,opts={}){
  new Chart(document.getElementById(id),{
    type:'bar',
    data:{labels,datasets:[{data,backgroundColor:color||B,borderRadius:3,borderSkipped:false}]},
    options:{
      responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:false},...(opts.plugins||{})},
      scales:{
        x:{grid:{color:GRID},ticks:{maxRotation:45,font:{size:10}}},
        y:{grid:{color:GRID},beginAtZero:true,ticks:{precision:0}}
      },...opts
    }
  });
}

function mkLine(id,labels,datasets,opts={}){
  new Chart(document.getElementById(id),{
    type:'line',
    data:{labels,datasets},
    options:{
      responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:datasets.length>1,...(opts.legendOpts||{})}},
      scales:{
        x:{grid:{color:GRID},ticks:{maxRotation:45,font:{size:10}}},
        y:{grid:{color:GRID},beginAtZero:true,ticks:{precision:0}}
      },...opts
    }
  });
}

function mkDoughnut(id,labels,data){
  new Chart(document.getElementById(id),{
    type:'doughnut',
    data:{labels,datasets:[{data,backgroundColor:PALETTE,borderWidth:2,borderColor:getThemeColor('--surface-strong', '#161b22')}]},
    options:{
      responsive:true,maintainAspectRatio:false,
      plugins:{legend:{position:'right',labels:{boxWidth:12,padding:10,font:{size:10}}}}
    }
  });
}

// ── Data ─────────────────────────────────────────────────
const dailyLabels   = DAILY_LABELS_JSON;
const dailyReqs     = DAILY_REQUESTS_JSON;
const dailyToks     = DAILY_TOKENS_JSON;
const cumToks       = CUMULATIVE_TOKENS_JSON;
const weekLabels    = WEEK_LABELS_JSON;
const weekReqs      = WEEK_REQUESTS_JSON;
const weekToks      = WEEK_TOKENS_JSON;
const modelLabels   = MODEL_LABELS_JSON;
const modelValues   = MODEL_VALUES_JSON;
const tokenLabels   = TOKEN_LABELS_JSON;
const modelPromptValues = MODEL_PROMPT_VALUES_JSON;
const modelOutputValues = MODEL_OUTPUT_VALUES_JSON;
const dailyPromptValues = DAILY_PROMPT_VALUES_JSON;
const dailyOutputValues = DAILY_OUTPUT_VALUES_JSON;
const hourValues    = HOUR_VALUES_JSON;
const dowLabels     = DOW_LABELS_JSON;
const dowValues     = DOW_VALUES_JSON;
const sessionDepthLabels = SESSION_DEPTH_LABELS_JSON;
const sessionDepthValues = SESSION_DEPTH_VALUES_JSON;
const toolData      = TOOL_DATA_JSON;
const keywords      = KEYWORDS_JSON;
const heatmap       = HEATMAP_JSON;
const hourDowMatrix = HOUR_DOW_MATRIX_JSON;

// ── Identity/User Filter Data ─────────────────────────────
const identityCatalog = IDENTITY_CATALOG_JSON;
const rawRequests     = RAW_REQUESTS_JSON;

// ── User Filter State & Logic ─────────────────────────────
let activeFilter = { field: '', value: '' };
const filterFieldSelect = document.getElementById('filterField');
const filterValueInput = document.getElementById('filterValue');
const applyFilterBtn = document.getElementById('applyFilterBtn');
const clearFilterBtn = document.getElementById('clearFilterBtn');
const filterStatus = document.getElementById('filterStatus');
const userFilterPanel = document.getElementById('userFilterPanel');

// Common identity field types for filtering
const COMMON_IDENTITY_FIELDS = ['attuid', 'attid', 'email', 'userid', 'username', 'employeeid'];

function initUserFilter() {
  // Always add common identity field options
  COMMON_IDENTITY_FIELDS.forEach(f => {
    const opt = document.createElement('option');
    opt.value = f;
    opt.textContent = f.toUpperCase();
    filterFieldSelect.appendChild(opt);
  });
  
  // Also add any fields found in the catalog
  const fields = Object.keys(identityCatalog).sort();
  fields.forEach(f => {
    const leafName = f.includes('.') ? f.split('.').pop() : f;
    // Skip if already in common fields
    if (COMMON_IDENTITY_FIELDS.includes(leafName.toLowerCase())) return;
    const opt = document.createElement('option');
    opt.value = f;
    const valCount = identityCatalog[f].length;
    opt.textContent = `${leafName} (${valCount} user${valCount !== 1 ? 's' : ''})`;
    filterFieldSelect.appendChild(opt);
  });
  
  // Show hint if no identity data exists but filter is still usable
  if (fields.length === 0) {
    filterStatus.textContent = 'Enter ATTUID to filter';
    filterStatus.classList.add('no-identity-hint');
  }
  
  // Auto-fill dropdown when a field with known values is selected
  filterFieldSelect.addEventListener('change', () => {
    const field = filterFieldSelect.value;
    if (field && identityCatalog[field]) {
      const vals = identityCatalog[field];
      if (vals.length <= 20) {
        let dl = document.getElementById('filterValueList');
        if (!dl) {
          dl = document.createElement('datalist');
          dl.id = 'filterValueList';
          filterValueInput.setAttribute('list', 'filterValueList');
          filterValueInput.parentNode.appendChild(dl);
        }
        dl.innerHTML = '';
        vals.forEach(v => {
          const o = document.createElement('option');
          o.value = v;
          dl.appendChild(o);
        });
      }
    }
  });
}

function getFilteredRequests() {
  if (!activeFilter.field && !activeFilter.value) {
    return rawRequests;
  }
  const fieldNorm = activeFilter.field.toLowerCase().replace(/[^a-z0-9]/g, '');
  const valueNorm = activeFilter.value.trim().toLowerCase();
  
  // If no value entered, return all
  if (!valueNorm) {
    return rawRequests;
  }
  
  return rawRequests.filter(r => {
    // Search in identity fields if they exist
    if (r.identity && Object.keys(r.identity).length > 0) {
      for (const [fName, fVal] of Object.entries(r.identity)) {
        const fNameNorm = fName.toLowerCase().replace(/[^a-z0-9]/g, '');
        const fLeafNorm = fName.split('.').pop().toLowerCase().replace(/[^a-z0-9]/g, '');
        const fieldMatch = !fieldNorm || fNameNorm.includes(fieldNorm) || fLeafNorm.includes(fieldNorm);
        const valMatch = String(fVal).toLowerCase() === valueNorm || String(fVal).toLowerCase().includes(valueNorm);
        if (fieldMatch && valMatch) return true;
      }
    }
    
    // Also search in session_id for partial matches (useful when no identity fields)
    if (r.session_id && r.session_id.toLowerCase().includes(valueNorm)) {
      return true;
    }
    
    return false;
  });
}

function recalculateStats(filteredReqs) {
  // Recalculate daily requests/tokens
  const dayReqMap = {};
  const dayTokMap = {};
  const dayPromptMap = {};
  const dayOutputMap = {};
  dailyLabels.forEach(d => { dayReqMap[d] = 0; dayTokMap[d] = 0; dayPromptMap[d] = 0; dayOutputMap[d] = 0; });
  
  const modelCounts = {};
  const modelPrompt = {};
  const modelOutput = {};
  const hourCounts = Array(24).fill(0);
  const dowCounts = [0,0,0,0,0,0,0];
  const hourDow = Array.from({length:24}, ()=>Array(7).fill(0));
  const sessionReqCounts = {};
  const toolCalls = {};
  let totalPrompt = 0, totalOutput = 0;
  
  filteredReqs.forEach(r => {
    if (r.date && dayReqMap.hasOwnProperty(r.date)) {
      dayReqMap[r.date]++;
      dayTokMap[r.date] += (r.prompt_tokens || 0) + (r.output_tokens || 0);
      dayPromptMap[r.date] += r.prompt_tokens || 0;
      dayOutputMap[r.date] += r.output_tokens || 0;
    }
    const model = r.model || 'unknown';
    modelCounts[model] = (modelCounts[model] || 0) + 1;
    modelPrompt[model] = (modelPrompt[model] || 0) + (r.prompt_tokens || 0);
    modelOutput[model] = (modelOutput[model] || 0) + (r.output_tokens || 0);
    totalPrompt += r.prompt_tokens || 0;
    totalOutput += r.output_tokens || 0;
    if (r.hour !== undefined) {
      hourCounts[r.hour]++;
      if (r.dow !== undefined) {
        dowCounts[r.dow]++;
        hourDow[r.hour][r.dow]++;
      }
    }
    sessionReqCounts[r.session_id] = (sessionReqCounts[r.session_id] || 0) + 1;
    if (r.tools) {
      r.tools.forEach(t => { toolCalls[t] = (toolCalls[t] || 0) + 1; });
    }
  });
  
  return {
    dailyReqs: dailyLabels.map(d => dayReqMap[d]),
    dailyToks: dailyLabels.map(d => dayTokMap[d]),
    dailyPrompt: dailyLabels.map(d => dayPromptMap[d]),
    dailyOutput: dailyLabels.map(d => dayOutputMap[d]),
    cumToks: dailyLabels.reduce((acc, d, i) => {
      acc.push((acc[i-1] || 0) + dayTokMap[d]);
      return acc;
    }, []),
    modelCounts,
    modelPrompt,
    modelOutput,
    hourCounts,
    dowCounts,
    hourDow,
    totalRequests: filteredReqs.length,
    totalPrompt,
    totalOutput,
    activeDays: Object.values(dayReqMap).filter(v => v > 0).length,
    totalSessions: Object.keys(sessionReqCounts).length,
    toolCalls,
    heatmap: dailyLabels.map(d => ({date: d, count: dayReqMap[d]})),
  };
}

function updateDashboardWithFilter() {
  const filtered = getFilteredRequests();
  const stats = recalculateStats(filtered);
  
  // Update hero stats
  document.querySelectorAll('.hero-stat strong').forEach((el, i) => {
    if (i === 0) el.textContent = stats.totalRequests;
    if (i === 1) el.textContent = formatNum(stats.totalPrompt + stats.totalOutput);
    if (i === 2) el.textContent = stats.activeDays;
  });
  
  // Update KPI cards
  const kpiCards = document.querySelectorAll('.section .card .stat-value');
  if (kpiCards[0]) kpiCards[0].textContent = stats.activeDays;
  if (kpiCards[1]) kpiCards[1].textContent = stats.totalSessions;
  if (kpiCards[2]) kpiCards[2].textContent = stats.totalRequests;
  if (kpiCards[4]) kpiCards[4].textContent = formatNum(stats.totalPrompt + stats.totalOutput);
  
  // Update charts
  updateChartData('dailyReqChart', stats.dailyReqs);
  updateChartData('cumulativeChart', stats.cumToks);
  updateStackedChartData('dailyTokenChart', [stats.dailyPrompt, stats.dailyOutput]);
  updateChartWithNewLabels('modelChart', Object.keys(stats.modelCounts), Object.values(stats.modelCounts));
  updateStackedChartWithNewLabels('tokenModelChart', 
    Object.keys(stats.modelPrompt),
    [Object.values(stats.modelPrompt), Object.values(stats.modelOutput)]
  );
  updateChartData('hourChart', stats.hourCounts);
  updateChartData('dowChart', stats.dowCounts);
  
  // Update heatmap
  renderHeatmapWithData(stats.heatmap);
  
  // Update intensity grid
  renderIntensityGridWithData(stats.hourDow);
  
  // Update tool chart/table
  const sortedTools = Object.entries(stats.toolCalls).sort((a,b) => b[1] - a[1]).slice(0, 15);
  updateToolTable(sortedTools);
  updateChartWithNewLabels('toolChart', sortedTools.map(t=>t[0]), sortedTools.map(t=>t[1]));
  
  // Update filter status
  filterStatus.textContent = `Showing ${stats.totalRequests} of ${rawRequests.length} requests`;
  filterStatus.classList.add('active');
  clearFilterBtn.style.display = 'inline-block';
}

function formatNum(n) {
  if (n >= 1000000) return (n/1000000).toFixed(1) + 'M';
  if (n >= 1000) return (n/1000).toFixed(1) + 'k';
  return String(n);
}

function updateChartData(chartId, newData) {
  const chart = Chart.getChart(chartId);
  if (chart) {
    chart.data.datasets[0].data = newData;
    chart.update();
  }
}

function updateStackedChartData(chartId, datasetsData) {
  const chart = Chart.getChart(chartId);
  if (chart) {
    datasetsData.forEach((data, i) => {
      if (chart.data.datasets[i]) chart.data.datasets[i].data = data;
    });
    chart.update();
  }
}

function updateChartWithNewLabels(chartId, labels, data) {
  const chart = Chart.getChart(chartId);
  if (chart) {
    chart.data.labels = labels;
    chart.data.datasets[0].data = data;
    chart.update();
  }
}

function updateStackedChartWithNewLabels(chartId, labels, datasetsData) {
  const chart = Chart.getChart(chartId);
  if (chart) {
    chart.data.labels = labels;
    datasetsData.forEach((data, i) => {
      if (chart.data.datasets[i]) chart.data.datasets[i].data = data;
    });
    chart.update();
  }
}

function renderHeatmapWithData(data) {
  hmContainer.innerHTML = '';
  const maxCount = Math.max(...data.map(d => d.count), 1);
  const colors = [
    getThemeColor('--heatmap-empty', '#161b22'),
    getThemeColor('--heatmap-1', '#0e4429'),
    getThemeColor('--heatmap-2', '#006d32'),
    getThemeColor('--heatmap-3', '#26a641'),
    getThemeColor('--heatmap-4', '#39d353'),
  ];
  data.forEach(d => {
    const cell = document.createElement('div');
    cell.className = 'hm-cell';
    cell.title = `${d.date}: ${d.count} requests`;
    const pct = d.count / maxCount;
    if (d.count === 0) cell.style.background = colors[0];
    else if (pct < 0.25) cell.style.background = colors[1];
    else if (pct < 0.50) cell.style.background = colors[2];
    else if (pct < 0.75) cell.style.background = colors[3];
    else cell.style.background = colors[4];
    hmContainer.appendChild(cell);
  });
}

function renderIntensityGridWithData(matrix) {
  igContainer.innerHTML = '';
  const maxVal = Math.max(...matrix.flat(), 1);
  const intensityRgb = getThemeColor('--intensity-rgb', '88,166,255');
  matrix.forEach((row, h) => {
    row.forEach((val, d) => {
      const cell = document.createElement('div');
      cell.className = 'ig-cell';
      cell.title = `${String(h).padStart(2,'0')}:00 ${['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][d]}: ${val}`;
      const a = val / maxVal;
      cell.style.background = `rgba(${intensityRgb},${a.toFixed(2)})`;
      cell.style.minHeight = '14px';
      igContainer.appendChild(cell);
    });
  });
}

function updateToolTable(toolData) {
  const tbody = document.getElementById('toolTable');
  tbody.innerHTML = '';
  const total = toolData.reduce((s, t) => s + t[1], 0);
  toolData.forEach(([name, count]) => {
    const pct = total > 0 ? Math.round(count / total * 100) : 0;
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${name}</td>
      <td><span class="badge">${count}</span></td>
      <td>
        <div class="bar-cell">
          <div class="mini-bar" style="width:${Math.min(pct*1.8,120)}px"></div>
          <span style="font-size:.75rem;color:var(--muted)">${pct}%</span>
        </div>
      </td>`;
    tbody.appendChild(row);
  });
}

function clearFilter() {
  activeFilter = { field: '', value: '' };
  filterFieldSelect.value = '';
  filterValueInput.value = '';
  filterStatus.textContent = '';
  filterStatus.classList.remove('active');
  clearFilterBtn.style.display = 'none';
  
  // Restore original data
  updateChartData('dailyReqChart', dailyReqs);
  updateChartData('cumulativeChart', cumToks);
  updateStackedChartData('dailyTokenChart', [dailyPromptValues, dailyOutputValues]);
  updateChartWithNewLabels('modelChart', modelLabels, modelValues);
  updateStackedChartWithNewLabels('tokenModelChart', tokenLabels, [modelPromptValues, modelOutputValues]);
  updateChartData('hourChart', hourValues);
  updateChartData('dowChart', dowValues);
  renderHeatmap();
  renderIntensityGrid();
  updateToolTable(toolData);
  updateChartWithNewLabels('toolChart', toolData.map(t=>t[0]), toolData.map(t=>t[1]));
  
  // Restore hero stats (reload page for full restore, or store originals)
  location.reload();
}

// Initialize filter UI
initUserFilter();
if (applyFilterBtn) {
  applyFilterBtn.addEventListener('click', () => {
    activeFilter.field = filterFieldSelect.value;
    activeFilter.value = filterValueInput.value;
    if (activeFilter.field || activeFilter.value) {
      updateDashboardWithFilter();
    }
  });
}
if (clearFilterBtn) {
  clearFilterBtn.addEventListener('click', clearFilter);
}
// Allow Enter key to apply filter
if (filterValueInput) {
  filterValueInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') applyFilterBtn.click();
  });
}

// ── Activity Heatmap ──────────────────────────────────────
const hmContainer = document.getElementById('heatmap');
const hmMax = Math.max(...heatmap.map(d=>d.count), 1);
function renderHeatmap(){
  hmContainer.innerHTML = '';
  const colors = [
    getThemeColor('--heatmap-empty', '#161b22'),
    getThemeColor('--heatmap-1', '#0e4429'),
    getThemeColor('--heatmap-2', '#006d32'),
    getThemeColor('--heatmap-3', '#26a641'),
    getThemeColor('--heatmap-4', '#39d353'),
  ];
  heatmap.forEach(d=>{
    const cell = document.createElement('div');
    cell.className = 'hm-cell';
    cell.title = `${d.date}: ${d.count} requests`;
    const pct = d.count / hmMax;
    if(d.count===0) cell.style.background = colors[0];
    else if(pct<0.25) cell.style.background = colors[1];
    else if(pct<0.50) cell.style.background = colors[2];
    else if(pct<0.75) cell.style.background = colors[3];
    else cell.style.background = colors[4];
    hmContainer.appendChild(cell);
  });
}
renderHeatmap();

// ── Daily Requests ────────────────────────────────────────
mkLine('dailyReqChart', dailyLabels,
  [{label:'Requests',data:dailyReqs,borderColor:B,backgroundColor:'rgba(88,166,255,.12)',
    fill:true,tension:.3,pointRadius:dailyReqs.length>60?0:3}]);

// ── Cumulative Tokens ─────────────────────────────────────
mkLine('cumulativeChart', dailyLabels,
  [{label:'Cumulative Tokens',data:cumToks,borderColor:G,backgroundColor:'rgba(63,185,80,.1)',
    fill:true,tension:.4,pointRadius:0}]);

// ── Weekly Volume ─────────────────────────────────────────
mkBar('weeklyChart', weekLabels, weekReqs, B);

// ── Daily Token Usage ─────────────────────────────────────
new Chart(document.getElementById('dailyTokenChart'),{
  type:'bar',
  data:{
    labels:dailyLabels,
    datasets:[
      {label:'Prompt',data:dailyPromptValues,backgroundColor:'rgba(88,166,255,.7)',borderRadius:2,stack:'tokens'},
      {label:'Output',data:dailyOutputValues,backgroundColor:'rgba(63,185,80,.7)',borderRadius:2,stack:'tokens'}
    ]
  },
  options:{
    responsive:true,maintainAspectRatio:false,
    plugins:{legend:{position:'top',labels:{boxWidth:12,font:{size:10}}}},
    scales:{
      x:{grid:{color:GRID},stacked:true,ticks:{maxRotation:45,font:{size:10}}},
      y:{grid:{color:GRID},beginAtZero:true,stacked:true,ticks:{precision:0}}
    }
  }
});

// ── Model Charts ──────────────────────────────────────────
mkDoughnut('modelChart', modelLabels, modelValues);
new Chart(document.getElementById('tokenModelChart'),{
  type:'bar',
  data:{
    labels:tokenLabels,
    datasets:[
      {label:'Prompt',data:modelPromptValues,backgroundColor:B,borderRadius:3,stack:'t'},
      {label:'Output',data:modelOutputValues,backgroundColor:G,borderRadius:3,stack:'t'}
    ]
  },
  options:{
    responsive:true,maintainAspectRatio:false,
    plugins:{legend:{position:'top',labels:{boxWidth:12,font:{size:10}}}},
    scales:{
      x:{grid:{color:GRID},stacked:true,ticks:{font:{size:10}}},
      y:{grid:{color:GRID},beginAtZero:true,stacked:true,ticks:{precision:0}}
    }
  }
});

// ── Hour Chart ────────────────────────────────────────────
const hourLabels=[...Array(24)].map((_,i)=>`${String(i).padStart(2,'0')}:00`);
mkBar('hourChart', hourLabels, hourValues,
  hourValues.map((_,i)=>i>=9&&i<=17?G:B));

// ── DoW Chart ─────────────────────────────────────────────
mkBar('dowChart', dowLabels, dowValues, Y);

// ── Session Depth ─────────────────────────────────────────
mkBar('sessionDepthChart', sessionDepthLabels, sessionDepthValues, P);

// ── Hour × Day-of-Week Intensity ──────────────────────────
const igMax = Math.max(...hourDowMatrix.flat(), 1);
const igContainer = document.getElementById('intensityGrid');
igContainer.style.gridTemplateRows = `repeat(24,16px)`;
function renderIntensityGrid(){
  igContainer.innerHTML = '';
  const intensityRgb = getThemeColor('--intensity-rgb', '88,166,255');
  hourDowMatrix.forEach((row,h)=>{
    row.forEach((val,d)=>{
      const cell = document.createElement('div');
      cell.className = 'ig-cell';
      cell.title = `${String(h).padStart(2,'0')}:00 ${['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][d]}: ${val}`;
      const a = val/igMax;
      cell.style.background = `rgba(${intensityRgb},${a.toFixed(2)})`;
      cell.style.minHeight = '14px';
      igContainer.appendChild(cell);
    });
  });
}
renderIntensityGrid();

// ── Tool Table + Chart ────────────────────────────────────
const toolTotal = toolData.reduce((s,t)=>s+t[1],0);
const tbody = document.getElementById('toolTable');
toolData.forEach(([name,count])=>{
  const pct = toolTotal>0?Math.round(count/toolTotal*100):0;
  const row = document.createElement('tr');
  row.innerHTML=`
    <td>${name}</td>
    <td><span class="badge">${count}</span></td>
    <td>
      <div class="bar-cell">
        <div class="mini-bar" style="width:${Math.min(pct*1.8,120)}px"></div>
        <span style="font-size:.75rem;color:var(--muted)">${pct}%</span>
      </div>
    </td>`;
  tbody.appendChild(row);
});
mkDoughnut('toolChart', toolData.map(t=>t[0]), toolData.map(t=>t[1]));

// ── Keywords ──────────────────────────────────────────────
const kwMax = keywords.length>0?keywords[0][1]:1;
keywords.forEach(([w,c])=>{
  const span = document.createElement('span');
  const sz = 0.75 + (c/kwMax)*0.65;
  span.className='tag';
  span.style.fontSize = sz+'rem';
  span.style.opacity  = 0.6+(c/kwMax)*0.4;
  span.textContent=`${w} (${c})`;
  document.getElementById('keywordTags').appendChild(span);
});

// ── New section data ──────────────────────────────────────
const intentLabels  = INTENT_LABELS_JSON;
const intentValues  = INTENT_VALUES_JSON;
const rollingAvg    = ROLLING_AVG_JSON;
const latencyPercs  = LATENCY_PERCS_JSON;
const modeLabels2   = MODE_LABELS_JSON;
const modeValues2   = MODE_VALUES_JSON;
const langLabels    = LANG_LABELS_JSON;
const langValues    = LANG_VALUES_JSON;
const costLabels    = COST_LABELS_JSON;
const costValues    = COST_VALUES_JSON;
const topSessions   = TOP_SESSIONS_JSON;

// ── Intent Doughnut ───────────────────────────────────────
mkDoughnut('intentChart', intentLabels, intentValues);

// ── Rolling 7-day avg line ─────────────────────────────────
new Chart(document.getElementById('rollingChart'),{
  type:'line',
  data:{
    labels:dailyLabels,
    datasets:[
      {label:'Daily Requests',data:dailyReqs,borderColor:'rgba(88,166,255,.45)',
       backgroundColor:'rgba(88,166,255,.07)',fill:true,tension:.3,pointRadius:0},
      {label:'7-day Avg',data:rollingAvg,borderColor:G,backgroundColor:'transparent',
       tension:.4,pointRadius:0,borderWidth:2.5}
    ]
  },
  options:{
    responsive:true,maintainAspectRatio:false,
    plugins:{legend:{position:'top',labels:{boxWidth:12,font:{size:10}}}},
    scales:{
      x:{grid:{color:GRID},ticks:{maxRotation:45,font:{size:10}}},
      y:{grid:{color:GRID},beginAtZero:true,ticks:{precision:0}}
    }
  }
});

// ── Latency percentiles ─────────────────────────────────────
mkBar('latencyPercChart',
  ['p50','p75','p90','p95','p99'],
  latencyPercs,
  [G, G, Y, R, '#ff4444']);

// ── Mode Doughnut ──────────────────────────────────────────
mkDoughnut('modeChart', modeLabels2, modeValues2);

// ── Language bar ───────────────────────────────────────────
mkBar('langChart', langLabels, langValues, P);

// ── Cost by model bar ──────────────────────────────────────
new Chart(document.getElementById('costChart'),{
  type:'bar',
  data:{
    labels:costLabels,
    datasets:[{data:costValues,backgroundColor:O,borderRadius:3,borderSkipped:false}]
  },
  options:{
    responsive:true,maintainAspectRatio:false,
    plugins:{legend:{display:false},tooltip:{callbacks:{label:ctx=>`$${ctx.parsed.y.toFixed(4)}`}}},
    scales:{
      x:{grid:{color:GRID},ticks:{font:{size:10}}},
      y:{grid:{color:GRID},beginAtZero:true,ticks:{callback:v=>`$${v.toFixed(3)}`}}
    }
  }
});

// ── Top Sessions table ─────────────────────────────────────
const tsBody = document.getElementById('topSessionsTable');
topSessions.forEach((s,i)=>{
  const row = document.createElement('tr');
  const toks = s.tokens >= 1000 ? (s.tokens/1000).toFixed(1)+'k' : String(s.tokens);
  row.innerHTML=`
    <td>${i+1}</td>
    <td><code style="font-size:.78rem;color:var(--muted)">${s.id}&hellip;</code></td>
    <td>${s.date}</td>
    <td><span class="badge">${s.requests}</span></td>
    <td style="color:var(--yellow)">${toks}</td>
    <td style="font-size:.8rem;color:var(--muted)">${s.model}</td>`;
  tsBody.appendChild(row);
});

if(themeToggle){
  themeToggle.addEventListener('click', ()=>{
    const nextTheme = root.dataset.theme === 'light' ? 'dark' : 'light';
    root.dataset.theme = nextTheme;
    try {
      localStorage.setItem(THEME_KEY, nextTheme);
    } catch (error) {
      // Ignore storage failures for local file previews.
    }
    syncChartDefaults();
    syncThemeToggleLabel();
    renderHeatmap();
    renderIntensityGrid();
    updateChartsTheme();
  });
}
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────
#  Formatting helpers
# ─────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────
#  Daily token split by model (for stacked chart)
# ─────────────────────────────────────────────────────────

def daily_model_token_split(requests, daily_labels):
    """Return per-day prompt/output sums aligned to daily_labels."""
    day_prompt = defaultdict(int)
    day_output = defaultdict(int)
    for r in requests:
        if r["timestamp"]:
            d = str(r["timestamp"].date())
            day_prompt[d] += r["prompt_tokens"] or 0
            day_output[d] += r["output_tokens"] or 0
    return (
        [day_prompt.get(d, 0) for d in daily_labels],
        [day_output.get(d, 0) for d in daily_labels],
    )


# ─────────────────────────────────────────────────────────
#  HTML rendering
# ─────────────────────────────────────────────────────────

def render_html(ins, requests, output_path, days, identity_filter_label="", identity_catalog=None):
    if identity_catalog is None:
        identity_catalog = {}
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    first = ins["first_request"].strftime("%b %d, %Y") if ins["first_request"] else "N/A"
    last  = ins["last_request"].strftime("%b %d, %Y")  if ins["last_request"]  else "N/A"

    top_mode = max(ins["mode_counts"], key=ins["mode_counts"].get, default="N/A")
    success_str = f" · {ins['tool_success_rate']}% success" if ins["tool_success_rate"] is not None else ""
    avg_output  = round(ins["total_output_tokens"] / ins["total_requests"]) if ins["total_requests"] else 0
    avg_prompt  = round(ins["total_prompt_tokens"]  / ins["total_requests"]) if ins["total_requests"] else 0
    rpd = round(ins["total_requests"] / ins["active_days"], 1) if ins["active_days"] > 0 else 0

    model_labels = list(ins["model_counts"].keys())[:8]
    model_values = [ins["model_counts"][m] for m in model_labels]
    token_labels = list(ins["model_tokens"].keys())[:8]
    prompt_values_model = [ins["model_tokens"][m]["prompt"] for m in token_labels]
    output_values_model = [ins["model_tokens"][m]["output"] for m in token_labels]

    daily_prompt_vals, daily_output_vals = daily_model_token_split(requests, ins["daily_labels"])

    # Prepare raw request data for client-side filtering
    raw_requests_js = []
    for r in requests:
        raw_requests_js.append({
            "session_id": r.get("session_id", ""),
            "date": str(r["timestamp"].date()) if r.get("timestamp") else "",
            "hour": r["timestamp"].hour if r.get("timestamp") else None,
            "dow": r["timestamp"].weekday() if r.get("timestamp") else None,
            "model": (r.get("resolved_model") or r.get("model_id") or "unknown").replace("copilot/", ""),
            "prompt_tokens": r.get("prompt_tokens") or 0,
            "output_tokens": r.get("output_tokens") or 0,
            "tools": r.get("tool_calls", []),
            "identity": r.get("identity_fields", {}),
        })

    # Convert identity_catalog sets to sorted lists for JSON
    identity_catalog_js = {k: sorted(list(v)) for k, v in identity_catalog.items()}

    html = HTML
    # Replace JS placeholders (avoiding { } conflicts in f-string)
    placeholders = {
        "DAILY_LABELS_JSON":        json.dumps(ins["daily_labels"]),
        "DAILY_REQUESTS_JSON":      json.dumps(ins["daily_requests"]),
        "DAILY_TOKENS_JSON":        json.dumps(ins["daily_tokens"]),
        "CUMULATIVE_TOKENS_JSON":   json.dumps(ins["cumulative_tokens"]),
        "WEEK_LABELS_JSON":         json.dumps(ins["week_labels"]),
        "WEEK_REQUESTS_JSON":       json.dumps(ins["week_requests"]),
        "WEEK_TOKENS_JSON":         json.dumps(ins["week_tokens"]),
        "MODEL_LABELS_JSON":        json.dumps(model_labels),
        "MODEL_VALUES_JSON":        json.dumps(model_values),
        "TOKEN_LABELS_JSON":        json.dumps(token_labels),
        "MODEL_PROMPT_VALUES_JSON": json.dumps(prompt_values_model),
        "MODEL_OUTPUT_VALUES_JSON": json.dumps(output_values_model),
        "HOUR_VALUES_JSON":         json.dumps([ins["hour_counts"][h] for h in range(24)]),
        "DOW_LABELS_JSON":          json.dumps(list(ins["dow_counts"].keys())),
        "DOW_VALUES_JSON":          json.dumps(list(ins["dow_counts"].values())),
        "SESSION_DEPTH_LABELS_JSON":json.dumps(ins["session_depth_labels"]),
        "SESSION_DEPTH_VALUES_JSON":json.dumps(ins["session_depth_values"]),
        "TOOL_DATA_JSON":           json.dumps(list(ins["tool_call_counts"].items())),
        "KEYWORDS_JSON":            json.dumps(list(ins["top_keywords"].items())[:25]),
        "HEATMAP_JSON":             json.dumps(ins["heatmap_grid"]),
        "HOUR_DOW_MATRIX_JSON":     json.dumps(ins["hour_dow_matrix"]),
        # daily stacked token data for the chart
        "DAILY_PROMPT_VALUES_JSON": json.dumps(daily_prompt_vals),
        "DAILY_OUTPUT_VALUES_JSON": json.dumps(daily_output_vals),
        # new sections
        "INTENT_LABELS_JSON":       json.dumps(list(ins["intent_counts"].keys())),
        "INTENT_VALUES_JSON":       json.dumps(list(ins["intent_counts"].values())),
        "ROLLING_AVG_JSON":         json.dumps(ins["rolling_avg_7d"]),
        "LATENCY_PERCS_JSON":       json.dumps([
                                        ins["p50_elapsed_ms"], ins["p75_elapsed_ms"],
                                        ins["p90_elapsed_ms"], ins["p95_elapsed_ms"],
                                        ins["p99_elapsed_ms"],
                                    ]),
        "MODE_LABELS_JSON":         json.dumps(list(ins["mode_counts"].keys())),
        "MODE_VALUES_JSON":         json.dumps(list(ins["mode_counts"].values())),
        "LANG_LABELS_JSON":         json.dumps(list(ins["lang_counts"].keys())),
        "LANG_VALUES_JSON":         json.dumps(list(ins["lang_counts"].values())),
        "COST_LABELS_JSON":         json.dumps(list(ins["cost_by_model"].keys())[:8]),
        "COST_VALUES_JSON":         json.dumps([ins["cost_by_model"][k] for k in list(ins["cost_by_model"].keys())[:8]]),
        "TOP_SESSIONS_JSON":        json.dumps(ins["top_sessions"]),
        # Identity/user filter data
        "IDENTITY_CATALOG_JSON":    json.dumps(identity_catalog_js),
        "RAW_REQUESTS_JSON":        json.dumps(raw_requests_js),
    }
    for key, val in placeholders.items():
        html = html.replace(key, val)

    # String template substitutions
    subs = {
        "{days}":             str(days),
        "{first_date}":       first,
        "{last_date}":        last,
        "{generated_at}":     now,
        "{latest_copilot}":   ins["latest_copilot"],
        "{latest_vscode}":    ins["latest_vscode"],
        "{active_days}":      str(ins["active_days"]),
        "{total_sessions}":   str(ins["total_sessions"]),
        "{sessions_multi}":   str(ins["sessions_with_multiple_requests"]),
        "{total_requests}":   str(ins["total_requests"]),
        "{avg_rps}":          str(ins["avg_requests_per_session"]),
        "{max_rps}":          str(ins["max_requests_per_session"]),
        "{rpd}":              str(rpd),
        "{total_tokens}":     fmt_num(ins["total_tokens"]),
        "{prompt_tokens}":    fmt_num(ins["total_prompt_tokens"]),
        "{output_tokens}":    fmt_num(ins["total_output_tokens"]),
        "{total_tool_calls}": str(ins["total_tool_calls"]),
        "{avg_tcs}":          str(ins["avg_tool_calls_per_session"]),
        "{success_rate_str}": success_str,
        "{avg_elapsed}":      fmt_ms(ins["avg_elapsed_ms"]),
        "{avg_ttft}":         fmt_ms(ins["avg_ttft_ms"]),
        "{p95_elapsed}":      fmt_ms(ins["p95_elapsed_ms"]),
        "{median_elapsed}":   fmt_ms(ins["median_elapsed_ms"]),
        "{avg_output_tokens}":str(avg_output),
        "{avg_prompt_tokens}":str(avg_prompt),
        "{top_mode}":         top_mode,
        "{avg_msg_len}":         str(ins["avg_message_length"]),
        "{max_msg_len}":         str(ins["max_message_length"]),
        "{token_efficiency}":    str(ins["token_efficiency"]),
        "{current_streak}":      str(ins["current_streak"]),
        "{longest_streak}":      str(ins["longest_streak"]),
        "{intensity_score}":     str(ins["intensity_score"]),
        "{estimated_cost_usd}": f"{ins['estimated_cost_usd']:.2f}",
        "{identity_filter_badge}": (
          f'<span class="badge">Filter {html.escape(identity_filter_label)}</span>'
          if identity_filter_label else ""
        ),
    }
    for key, val in subs.items():
        html = html.replace(key, val)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path


# ─────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GenAI Usage Dashboard — last N days")
    parser.add_argument("--days", "-n", type=int, default=100,
                        help="Number of days to include (default: 100)")
    parser.add_argument("--output", "-o",
                        default=str(Path(__file__).parent / "genai_dashboard.html"),
                        help="Output HTML file (default: genai_dashboard.html)")
    parser.add_argument("--logs-dir", "-d", nargs="*", default=[],
                        help="Extra directories to scan for *.jsonl files")
    parser.add_argument("--user-field", default="",
                        help="Identity field to filter on when present, e.g. attid, email, userId")
    parser.add_argument("--user-value", default="",
                        help="Identity field value to match (case-insensitive)")
    parser.add_argument("--show-user-fields", action="store_true",
                        help="Print identity-like fields discovered in the scanned logs")
    parser.add_argument("--print-summary", action="store_true",
                        help="Print text summary to stdout")
    args = parser.parse_args()
    filter_label = describe_user_filter(args.user_field, args.user_value)
    filter_active = bool(args.user_field or args.user_value)

    print(f"Scanning logs (last {args.days} days) …", flush=True)
    requests, transcript_events, debug_events, cutoff, identity_catalog, scan_stats = aggregate(
        args.logs_dir,
        args.days,
        args.user_field,
        args.user_value,
    )
    print(f"  {len(requests)} requests  |  "
          f"{len(transcript_events)} transcript events  |  "
          f"{len(debug_events)} debug events")
    if filter_active:
        print(f"  User filter: {filter_label}")
        print(f"  Matched {len(requests)} of {scan_stats['window_requests']} requests in the time window")
    if args.show_user_fields:
        print("\nDetected identity-like fields:")
        for line in format_identity_catalog(identity_catalog):
            print(line)

    if not requests:
        if filter_active:
            print(
                "\nNo requests matched the user filter ({}) in the last {} days.".format(
                    filter_label,
                    args.days,
                )
            )
            if not identity_catalog:
                print("No identity/user fields were detected in the scanned logs, so ATTID-style filtering is not available for this profile.")
        else:
            print(
                "\nNo requests found in the last {} days.\n"
                "Make sure VS Code with GitHub Copilot Chat is installed,\n"
                "or drop *.jsonl files into the same folder as this script.".format(args.days)
            )
        sys.exit(1)

    print("Computing insights …", flush=True)
    ins = compute_insights(requests, transcript_events, debug_events, cutoff, args.days)

    print("Rendering HTML dashboard …", flush=True)
    out = render_html(ins, requests, args.output, args.days, filter_label if filter_active else "", identity_catalog)
    print(f"\nDashboard saved → {out}")

    if args.print_summary:
        sep = "─" * 44
        print(f"\n{sep}")
        print(f"  GenAI Usage Summary — Last {args.days} Days")
        print(sep)
        print(f"  Active days          : {ins['active_days']}")
        print(f"  Sessions             : {ins['total_sessions']}")
        print(f"  Requests             : {ins['total_requests']}")
        print(f"  Requests / active day: {round(ins['total_requests']/max(ins['active_days'],1),1)}")
        print(f"  Prompt tokens        : {fmt_num(ins['total_prompt_tokens'])}")
        print(f"  Output tokens        : {fmt_num(ins['total_output_tokens'])}")
        print(f"  Avg response time    : {fmt_ms(ins['avg_elapsed_ms'])}")
        print(f"  p95 response time    : {fmt_ms(ins['p95_elapsed_ms'])}")
        print(f"  Total tool calls     : {ins['total_tool_calls']}")
        print(f"  Top model            : {next(iter(ins['model_counts']), 'N/A')}")
        print(f"  Top tool             : {next(iter(ins['tool_call_counts']), 'N/A')}")
        print(f"  Top keywords         : {', '.join(list(ins['top_keywords'])[:6])}")
        print(f"  Copilot version      : {ins['latest_copilot']}")
        print(sep)


if __name__ == "__main__":
    main()
