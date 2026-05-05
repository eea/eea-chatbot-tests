"""HTML report generator for JSONL test results.

Takes an AnalysisResult from analyze.py and produces a self-contained
dark-theme HTML report with Chart.js visualisations, a production-readiness
verdict, and full per-test detail tables.

Usage
-----
    from chatbot_tests.analyze import analyze_jsonl
    from chatbot_tests.html_report import generate_html, generate_report

    # High-level: parse + write in one call
    out = generate_report(Path("reports/run.jsonl"))

    # Low-level: get the HTML string and do what you want with it
    analysis = analyze_jsonl(Path("reports/run.jsonl"))
    html = generate_html(analysis)
    Path("report.html").write_text(html, encoding="utf-8")

CLI (wired in main.py):
    chatbot_tests analyze ./results.jsonl --html
"""

import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from chatbot_tests.analyze import AnalysisResult, analyze_jsonl


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(
    jsonl_path: Path,
    output_path: Optional[Path] = None,
) -> Path:
    """Parse *jsonl_path*, generate HTML, write to *output_path*, return it.

    If *output_path* is None the file is written next to the JSONL with
    the same stem and a ``.html`` extension.
    """
    analysis = analyze_jsonl(jsonl_path)
    html = generate_html(analysis)

    if output_path is None:
        output_path = jsonl_path.with_suffix(".html")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def generate_html(analysis: AnalysisResult) -> str:
    """Return a complete HTML document string for *analysis*."""
    ctx = _build_context(analysis)
    return _render(ctx)


# ---------------------------------------------------------------------------
# Context builder — all derived values in one place
# ---------------------------------------------------------------------------

def _build_context(a: AnalysisResult) -> dict:
    """Derive every value needed by the renderer from *a*."""

    # --- basic counts ---
    total = a.total_tests
    passed = a.passed_tests
    failed = a.failed_tests
    skipped = a.skipped_tests
    warns = a.passed_with_warnings
    evaluated = total - skipped
    pass_rate = a.pass_rate
    step_pass_rate = a.step_pass_rate
    all_steps_count = a.total_steps

    # --- timing ---
    dur_str = _fmt_duration(a.total_duration_seconds)
    session_label = _fmt_timestamp(a.session_start)
    session_end_label = _fmt_timestamp(a.session_end)

    # --- health ---
    health = a.health_status
    health_color = {
        "healthy": "#22c55e",
        "degraded": "#f59e0b",
        "unstable": "#f97316",
        "critical": "#ef4444",
    }.get(health, "#6b7280")

    # --- Halloumi scores ---
    scores = _extract_halloumi_scores(a)
    score_vals = [s for _, s in scores]
    avg_score = sum(score_vals) / len(score_vals) if score_vals else 0.0
    below_threshold = [(n, s) for n, s in scores if s < 60]
    score_buckets = _bucket_scores(score_vals)

    # --- warning breakdown ---
    halloumi_timeouts = _warning_tests_with(a, "504")
    low_score_tests = _warning_tests_with(a, "Quality score")
    citation_fail_tests = _warning_tests_with(a, "missing citations")

    # --- failed / skipped / warning test detail ---
    failed_tests = [t for t in a.tests if t.outcome == "failed"]
    warning_tests = [t for t in a.tests if t.warn]
    skipped_tests = [t for t in a.tests if t.outcome == "skipped"]

    # --- topic markers (exclude priority / set / meta markers) ---
    _meta = {"high", "medium", "low", "basic", "always", "question",
             "set_1_50", "set_51_100", "set_101_150", "set_151_200",
             "test", "halloumi", "ui", "sources", "related_questions",
             "deep_research", "error_handling", "feedback", "follow_up"}
    topic_markers = {k: v for k, v in a.tests_by_marker.items()
                     if k not in _meta}

    # --- production-readiness verdict ---
    verdict = _build_verdict(
        pass_rate=pass_rate,
        failed_tests=failed_tests,
        warning_tests=warning_tests,
        skipped_tests=skipped_tests,
        halloumi_timeouts=halloumi_timeouts,
        low_score_tests=low_score_tests,
        avg_score=avg_score,
    )

    return dict(
        # meta
        source_file=a.source_file or "",
        session_label=session_label,
        session_end_label=session_end_label,
        dur_str=dur_str,
        health=health,
        health_color=health_color,
        # counts
        total=total, passed=passed, failed=failed,
        skipped=skipped, warns=warns, evaluated=evaluated,
        pass_rate=pass_rate, step_pass_rate=step_pass_rate,
        all_steps_count=all_steps_count,
        passed_steps=a.passed_steps, failed_steps=a.failed_steps,
        # halloumi
        scores=scores, avg_score=avg_score,
        below_threshold=below_threshold,
        score_buckets=score_buckets,
        # test lists
        failed_tests=failed_tests,
        warning_tests=warning_tests,
        skipped_tests=skipped_tests,
        halloumi_timeouts=halloumi_timeouts,
        low_score_tests=low_score_tests,
        citation_fail_tests=citation_fail_tests,
        # markers
        topic_markers=topic_markers,
        # slowest
        slowest_tests=a.slowest_tests,
        # llm verdicts
        llm_verdicts=a.llm_verdicts,
        # verdict
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m {s}s"


def _fmt_timestamp(iso: Optional[str]) -> str:
    if not iso:
        return "N/A"
    try:
        return datetime.fromisoformat(iso).strftime("%B %d, %Y at %H:%M")
    except (ValueError, TypeError):
        return iso


def _score_color(s: float) -> str:
    if s >= 80:
        return "#22c55e"
    if s >= 60:
        return "#f59e0b"
    return "#ef4444"


def _outcome_badge(outcome: str, warn: bool = False) -> str:
    if outcome == "passed" and warn:
        return '<span class="badge badge-warn">WARN</span>'
    badges = {
        "passed":  '<span class="badge badge-pass">PASS</span>',
        "failed":  '<span class="badge badge-fail">FAIL</span>',
        "skipped": '<span class="badge badge-skip">SKIP</span>',
    }
    return badges.get(outcome, f'<span class="badge">{outcome.upper()}</span>')


def _h(text: str) -> str:
    """HTML-escape."""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _extract_halloumi_scores(a: AnalysisResult):
    scores = []
    for t in a.tests:
        for s in t.steps:
            if s.step_type == "info" and "Halloumi quality fact-check score:" in s.step_name:
                m = re.search(r"(\d+)%", s.step_name)
                if m:
                    scores.append((t.name, int(m.group(1))))
    return scores


def _bucket_scores(vals):
    buckets = {"<40": 0, "40–59": 0, "60–74": 0, "75–89": 0, "90–100": 0}
    for s in vals:
        if s < 40:
            buckets["<40"] += 1
        elif s < 60:
            buckets["40–59"] += 1
        elif s < 75:
            buckets["60–74"] += 1
        elif s < 90:
            buckets["75–89"] += 1
        else:
            buckets["90–100"] += 1
    return buckets


def _warning_tests_with(a: AnalysisResult, keyword: str):
    result = []
    for t in a.tests:
        if t.warn:
            for s in t.steps:
                if s.outcome == "failed" and keyword in (s.step_name + (s.message or "")):
                    result.append(t)
                    break
    return result


# ---------------------------------------------------------------------------
# Production-readiness verdict (data-driven)
# ---------------------------------------------------------------------------

def _build_verdict(
    pass_rate, failed_tests, warning_tests, skipped_tests,
    halloumi_timeouts, low_score_tests, avg_score,
) -> dict:
    """Return a dict describing production readiness."""

    hard_failures = len(failed_tests)
    infra_noise = len(halloumi_timeouts)
    quality_warnings = len(low_score_tests)
    knowledge_gaps = len(skipped_tests)

    # Determine level
    if hard_failures == 0 and pass_rate >= 98:
        level = "ready"
        headline = "Production Ready"
        icon = "✅"
        border_color = "rgba(34,197,94,0.35)"
        bg = "linear-gradient(135deg,rgba(34,197,94,0.12),rgba(99,102,241,0.08))"
    elif hard_failures <= 2 and pass_rate >= 95:
        level = "ready_with_fixes"
        headline = f"Production Ready — {hard_failures} fix{'es' if hard_failures > 1 else ''} required"
        icon = "✅"
        border_color = "rgba(34,197,94,0.35)"
        bg = "linear-gradient(135deg,rgba(34,197,94,0.12),rgba(99,102,241,0.08))"
    elif hard_failures <= 5 and pass_rate >= 85:
        level = "needs_fixes"
        headline = f"Not Ready — {hard_failures} failures blocking release"
        icon = "⚠️"
        border_color = "rgba(245,158,11,0.35)"
        bg = "linear-gradient(135deg,rgba(245,158,11,0.10),rgba(99,102,241,0.06))"
    else:
        level = "critical"
        headline = f"Critical — Do Not Release ({hard_failures} failures, {pass_rate:.0f}% pass rate)"
        icon = "🚨"
        border_color = "rgba(239,68,68,0.4)"
        bg = "linear-gradient(135deg,rgba(239,68,68,0.12),rgba(99,102,241,0.06))"

    # Build summary sentence
    if level in ("ready", "ready_with_fixes"):
        summary = (
            f"The chatbot is performing well. "
            f"<strong style=\"color:#22c55e\">{pass_rate:.0f}% of evaluated tests pass.</strong> "
            f"All core UI flows work. "
            + (f"The system correctly handles unknown questions by admitting it lacks information "
               f"rather than hallucinating. " if knowledge_gaps > 0 else "")
            + "No systemic failures across topic categories."
        )
    elif level == "needs_fixes":
        summary = (
            f"<strong style=\"color:#f59e0b\">{pass_rate:.1f}% pass rate</strong> with "
            f"{hard_failures} hard failures. Fix the failures listed below before releasing. "
            + (f"{quality_warnings} responses also scored below the quality threshold. " if quality_warnings else "")
        )
    else:
        summary = (
            f"<strong style=\"color:#ef4444\">{pass_rate:.1f}% pass rate.</strong> "
            f"{hard_failures} tests failed outright. The system is not stable enough to release."
        )

    # Build blocker list
    blockers = []
    for t in failed_tests:
        failed_steps = [s for s in t.steps if s.outcome == "failed"]
        first_step = failed_steps[0] if failed_steps else None
        msg = (first_step.message or t.message or "")[:300] if first_step else (t.message or "")[:300]
        q_id = re.search(r'\[(\w+-\d+)', t.name)
        label = q_id.group(1) if q_id else t.name
        desc = _h(msg).replace("\n", " ") if msg else "No error message captured."
        step_name = _h(first_step.step_name) if first_step else ""
        blockers.append({"label": label, "step": step_name, "desc": desc})

    # Build known-issues list
    known_issues = []
    if infra_noise > 0:
        known_issues.append({
            "title": f"Halloumi service timeouts ({infra_noise} tests)",
            "body": "External fact-check API returned 504 on these questions. "
                    "Infrastructure issue — chatbot responses themselves were fine.",
        })
    if quality_warnings > 0:
        q_labels = ", ".join(
            re.search(r'\[(\w+-\d+)', t.name).group(1)
            if re.search(r'\[(\w+-\d+)', t.name) else t.name
            for t in low_score_tests
        )
        known_issues.append({
            "title": f"Low Halloumi quality scores ({quality_warnings} tests)",
            "body": f"Scores below 60% threshold. Affected: {q_labels}. "
                    "Review whether the knowledge base needs enrichment for these topics.",
        })
    if knowledge_gaps > 0:
        known_issues.append({
            "title": f"Knowledge-gap skips ({knowledge_gaps} tests)",
            "body": "Chatbot correctly admitted it lacked information for these questions. "
                    "Expected behaviour — not failures.",
        })

    # One-line bottom verdict
    if level == "ready":
        bottom = "All checks pass. Good to ship."
    elif level == "ready_with_fixes":
        fix_labels = ", ".join(b["label"] for b in blockers)
        bottom = (
            f"Fix {fix_labels}, then ship. "
            + (f"Everything else is infrastructure noise or acceptable knowledge limits." if infra_noise > 0 or knowledge_gaps > 0 else "")
            + (f" Halloumi quality averages <strong style=\"color:#22c55e\">{avg_score:.1f}%</strong> across scored responses." if avg_score > 0 else "")
        )
    elif level == "needs_fixes":
        bottom = f"Address all {hard_failures} failures before releasing. Re-run after fixes."
    else:
        bottom = "Major quality issues. Do not deploy without a full investigation."

    return dict(
        level=level,
        icon=icon,
        headline=headline,
        border_color=border_color,
        bg=bg,
        summary=summary,
        blockers=blockers,
        known_issues=known_issues,
        bottom=bottom,
    )


# ---------------------------------------------------------------------------
# HTML renderer
# ---------------------------------------------------------------------------

def _render(c: dict) -> str:
    pass_rate = c["pass_rate"]
    step_pass_rate = c["step_pass_rate"]
    avg_score = c["avg_score"]

    # --- verdict HTML ---
    verdict = c["verdict"]

    blockers_html = ""
    if verdict["blockers"]:
        items = ""
        for i, b in enumerate(verdict["blockers"], 1):
            items += f"""
            <div style="margin-bottom:{'10px' if i < len(verdict['blockers']) else '0'}">
              <div style="font-weight:600;color:#f1f5f9;margin-bottom:2px">{i}. {b['label']} — {b['step']}</div>
              <div style="color:#94a3b8;font-size:12px">{b['desc']}</div>
            </div>"""
        blockers_html = f"""
        <div style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);border-radius:10px;padding:16px 18px;">
          <div style="font-size:12px;font-weight:700;color:#ef4444;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:10px;">🚨 Must Fix Before Release ({len(verdict['blockers'])})</div>
          {items}
        </div>"""

    issues_html = ""
    if verdict["known_issues"]:
        items = ""
        for ki in verdict["known_issues"]:
            items += f"""
            <div style="margin-bottom:8px">
              <div style="font-weight:600;color:#f1f5f9;margin-bottom:2px">{ki['title']}</div>
              <div style="color:#94a3b8;font-size:12px">{ki['body']}</div>
            </div>"""
        issues_html = f"""
        <div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.3);border-radius:10px;padding:16px 18px;">
          <div style="font-size:12px;font-weight:700;color:#f59e0b;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:10px;">⚠️ Known Issues — Not Blocking</div>
          {items}
        </div>"""

    panels_grid = ""
    if blockers_html or issues_html:
        panels_grid = f"""
      <div style="display:grid;grid-template-columns:{'1fr 1fr' if blockers_html and issues_html else '1fr'};gap:16px;margin-bottom:20px;">
        {blockers_html}
        {issues_html}
      </div>"""

    verdict_section = f"""
  <div class="section" style="margin-bottom:32px">
    <div style="background:{verdict['bg']};border:1px solid {verdict['border_color']};border-radius:14px;padding:28px 32px;">
      <div style="display:flex;align-items:center;gap:14px;margin-bottom:18px;">
        <div style="font-size:36px">{verdict['icon']}</div>
        <div>
          <div style="font-size:20px;font-weight:800;color:#fff;">{verdict['headline']}</div>
          <div style="color:#94a3b8;font-size:13px;margin-top:2px;">{_h(c['source_file'])} · {c['session_label']}</div>
        </div>
      </div>
      <p style="font-size:15px;line-height:1.75;color:#e2e8f0;margin-bottom:20px;">{verdict['summary']}</p>
      {panels_grid}
      <div style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.25);border-radius:10px;padding:14px 18px;">
        <div style="font-size:12px;font-weight:700;color:#818cf8;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px;">📋 Verdict</div>
        <div style="color:#e2e8f0;font-size:14px;line-height:1.7;">{verdict['bottom']}</div>
      </div>
    </div>
  </div>"""

    # --- KPIs ---
    kpi_pass_color = "#22c55e" if pass_rate >= 95 else ("#f59e0b" if pass_rate >= 80 else "#ef4444")
    kpi_step_color = "#22c55e" if step_pass_rate >= 95 else ("#f59e0b" if step_pass_rate >= 80 else "#ef4444")

    kpi_section = f"""
  <div class="kpi-grid">
    <div class="kpi"><div class="label">Total Tests</div><div class="value">{c['total']}</div></div>
    <div class="kpi"><div class="label">Pass Rate</div><div class="value" style="color:{kpi_pass_color}">{pass_rate:.1f}%</div><div class="sub">{c['passed']} passed / {c['evaluated']} evaluated</div></div>
    <div class="kpi"><div class="label">Failed</div><div class="value" style="color:#ef4444">{c['failed']}</div></div>
    <div class="kpi"><div class="label">Warnings</div><div class="value" style="color:#f59e0b">{c['warns']}</div><div class="sub">passed w/ failed steps</div></div>
    <div class="kpi"><div class="label">Skipped</div><div class="value" style="color:#6b7280">{c['skipped']}</div><div class="sub">knowledge gaps</div></div>
    <div class="kpi"><div class="label">Step Pass Rate</div><div class="value" style="color:{kpi_step_color}">{step_pass_rate:.1f}%</div><div class="sub">{c['passed_steps']}/{c['all_steps_count']} steps</div></div>
    <div class="kpi"><div class="label">Avg Halloumi Score</div><div class="value" style="color:{_score_color(avg_score)}">{avg_score:.1f}%</div><div class="sub">{len(c['scores'])} tests scored</div></div>
    <div class="kpi"><div class="label">Duration</div><div class="value" style="font-size:24px">{c['dur_str']}</div></div>
  </div>"""

    # --- insights ---
    insights = [
        ("good", f"<strong>{pass_rate:.1f}% pass rate</strong> across {c['evaluated']} evaluated tests.")
        if pass_rate >= 95 else
        ("warn", f"<strong>{pass_rate:.1f}% pass rate</strong> across {c['evaluated']} evaluated tests."),
    ]
    if c["warns"] > 0:
        timeout_n = len(c["halloumi_timeouts"])
        low_n = len(c["low_score_tests"])
        parts = []
        if timeout_n:
            parts.append(f"{timeout_n} Halloumi service timeouts (504 — infrastructure, not chatbot quality)")
        if low_n:
            parts.append(f"{low_n} responses below the 60% Halloumi quality threshold")
        if len(c["citation_fail_tests"]):
            parts.append(f"{len(c['citation_fail_tests'])} missing-citations verdicts")
        insights.append(("warn", f"<strong>{c['warns']} tests passed with warnings</strong> — " + "; ".join(parts) + "."))
    if c["skipped"] > 0:
        insights.append(("info", f"<strong>{c['skipped']} questions skipped</strong> — chatbot correctly admitted it lacked information. Expected behaviour, not bugs."))
    if avg_score > 0:
        insights.append(("info", f"<strong>Halloumi quality:</strong> average {avg_score:.1f}% across {len(c['scores'])} scored responses. {len(c['below_threshold'])} below 60% threshold."))

    insights_html = "".join(
        f'<div class="insight-box {kind}">{msg}</div>'
        for kind, msg in insights
    )

    # --- topic marker table ---
    marker_rows = ""
    for mk, mts in sorted(c["topic_markers"].items()):
        mp = sum(1 for t in mts if t.outcome == "passed")
        mf = sum(1 for t in mts if t.outcome == "failed")
        ms = sum(1 for t in mts if t.outcome == "skipped")
        mt = len(mts)
        denom = mt - ms
        mr = (mp / denom * 100) if denom > 0 else 0.0
        bar_color = "#22c55e" if mr >= 90 else ("#f59e0b" if mr >= 70 else "#ef4444")
        marker_rows += f"""
    <tr>
      <td><code>{mk}</code></td>
      <td class="num">{mt}</td>
      <td class="num" style="color:#22c55e">{mp}</td>
      <td class="num" style="color:#ef4444">{mf}</td>
      <td class="num" style="color:#888">{ms}</td>
      <td>
        <div class="bar-wrap"><div class="bar" style="width:{mr:.0f}%;background:{bar_color}"></div></div>
        <span class="pct">{mr:.0f}%</span>
      </td>
    </tr>"""

    # --- failed tests table ---
    failed_rows = _test_table_rows(c["failed_tests"], outcome="failed")

    # --- warning tests table ---
    warning_rows = _test_table_rows(c["warning_tests"], warn=True)

    # --- skipped tests table ---
    skipped_rows = ""
    for t in c["skipped_tests"]:
        msg = _h((t.message or "")[:300])
        skipped_rows += f"""
    <tr>
      <td>{_outcome_badge("skipped")}</td>
      <td>{_h(t.name)}<div class="test-msg">{msg}</div></td>
    </tr>"""

    # --- Halloumi below-threshold table ---
    score_table_rows = "".join(
        f'<tr><td>{_q_label(name)}</td><td style="color:{_score_color(s)};font-weight:bold">{s}%</td></tr>'
        for name, s in sorted(c["below_threshold"], key=lambda x: x[1])
    )

    total_scored = len(c["scores"])
    score_dist_rows = "".join(
        f'<tr><td>{b}</td><td class="num">{cnt}</td>'
        f'<td><div class="bar-wrap"><div class="bar" style="width:{cnt/total_scored*100:.0f}%;'
        f'background:{_score_color({"<40":20,"40–59":50,"60–74":67,"75–89":82,"90–100":95}[b])}"></div></div>'
        f'<span class="pct">{cnt/total_scored*100:.0f}%</span></td></tr>'
        for b, cnt in c["score_buckets"].items()
    ) if total_scored > 0 else ""

    # --- slowest tests table ---
    slowest_rows = "".join(
        f'<tr><td>{_h(name)}</td>'
        f'<td class="num" style="color:{"#ef4444" if dur>90 else "#f59e0b" if dur>60 else "var(--text)"}">{dur:.1f}s</td></tr>'
        for name, dur in (c["slowest_tests"] or [])[:10]
    )

    # --- LLM verdicts table ---
    llm_verdict_section = _render_llm_verdicts(c["llm_verdicts"])

    # --- Chart.js data ---
    passes_no_warn = c["passed"] - c["warns"]
    score_bucket_vals = list(c["score_buckets"].values())
    score_bucket_keys = list(c["score_buckets"].keys())

    topic_labels = sorted(c["topic_markers"].keys())
    topic_pass = [sum(1 for t in c["topic_markers"][m] if t.outcome == "passed") for m in topic_labels]
    topic_fail = [sum(1 for t in c["topic_markers"][m] if t.outcome == "failed") for m in topic_labels]
    topic_skip = [sum(1 for t in c["topic_markers"][m] if t.outcome == "skipped") for m in topic_labels]

    charts_js = f"""
const pieData = {{
  labels: ['Passed','Passed (warnings)','Failed','Skipped'],
  datasets: [{{
    data: [{passes_no_warn},{c['warns']},{c['failed']},{c['skipped']}],
    backgroundColor: ['#22c55e','#f59e0b','#ef4444','#6b7280'],
    borderWidth: 2, borderColor: '#1a1a2e',
  }}]
}};
const scoreData = {{
  labels: {json.dumps(score_bucket_keys)},
  datasets: [{{
    label: 'Tests',
    data: {json.dumps(score_bucket_vals)},
    backgroundColor: ['#ef4444','#f59e0b','#eab308','#22c55e','#16a34a'],
    borderRadius: 4,
  }}]
}};
const topicLabels = {json.dumps(topic_labels)};
const topicPass  = {json.dumps(topic_pass)};
const topicFail  = {json.dumps(topic_fail)};
const topicSkip  = {json.dumps(topic_skip)};

Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';

new Chart(document.getElementById('pieChart'), {{
  type: 'doughnut', data: pieData,
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{
      legend: {{ position: 'bottom', labels: {{ boxWidth: 12, padding: 12 }} }},
      tooltip: {{ callbacks: {{ label: ctx => ' ' + ctx.label + ': ' + ctx.raw }} }}
    }},
    cutout: '65%',
  }}
}});
new Chart(document.getElementById('scoreChart'), {{
  type: 'bar', data: scoreData,
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{ x: {{ grid: {{ display: false }} }}, y: {{ beginAtZero: true }} }}
  }}
}});
new Chart(document.getElementById('topicChart'), {{
  type: 'bar',
  data: {{
    labels: topicLabels,
    datasets: [
      {{ label: 'Passed', data: topicPass, backgroundColor: 'rgba(34,197,94,0.7)', stack: 'a', borderRadius: 2 }},
      {{ label: 'Failed', data: topicFail, backgroundColor: 'rgba(239,68,68,0.7)', stack: 'a', borderRadius: 2 }},
      {{ label: 'Skipped', data: topicSkip, backgroundColor: 'rgba(107,114,128,0.5)', stack: 'a', borderRadius: 2 }},
    ]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ position: 'bottom', labels: {{ boxWidth: 10, padding: 10 }} }} }},
    scales: {{
      x: {{ stacked: true, grid: {{ display: false }}, ticks: {{ font: {{ size: 10 }} }} }},
      y: {{ stacked: true, beginAtZero: true }}
    }}
  }}
}});"""

    health_label = c["health"].upper()

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EEA Chatbot Test Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
{_CSS}
</head>
<body>

<div class="header">
  <h1>EEA Chatbot Test Report</h1>
  <div class="meta">
    <span><code>{_h(c['source_file'])}</code></span>
    <span>{c['session_label']}</span>
    <span>Duration: {c['dur_str']}</span>
    <span>Health: <span style="color:{c['health_color']};font-weight:600">{health_label}</span></span>
  </div>
</div>

<div class="container">

{verdict_section}

{kpi_section}

  <div class="section">
    <div class="section-title"><span class="icon">💡</span> Key Insights</div>
    {insights_html}
  </div>

  <div class="section">
    <div class="section-title"><span class="icon">📊</span> Visual Overview</div>
    <div class="grid-3">
      <div class="card">
        <div class="chart-label">Outcome Distribution</div>
        <div class="chart-wrap"><canvas id="pieChart"></canvas></div>
      </div>
      <div class="card">
        <div class="chart-label">Halloumi Score Distribution</div>
        <div class="chart-wrap"><canvas id="scoreChart"></canvas></div>
      </div>
      <div class="card">
        <div class="chart-label">Pass Rate by Topic</div>
        <div class="chart-wrap"><canvas id="topicChart"></canvas></div>
      </div>
    </div>
  </div>

  <div class="section">
    <div class="section-title"><span class="icon">🏷️</span> Results by Topic</div>
    <div class="card">
      <table>
        <thead><tr><th>Marker</th><th class="num">Total</th><th class="num">Passed</th><th class="num">Failed</th><th class="num">Skipped</th><th>Pass Rate</th></tr></thead>
        <tbody>{marker_rows}</tbody>
      </table>
    </div>
  </div>

  {'<div class="section"><div class="section-title"><span class="icon">❌</span> Failed Tests (' + str(c["failed"]) + ')</div><div class="card"><table><thead><tr><th style="width:80px">Status</th><th>Test</th><th class="num">Duration</th></tr></thead><tbody>' + failed_rows + '</tbody></table></div></div>' if c["failed"] > 0 else ''}

  {'<div class="section"><div class="section-title"><span class="icon">⚠️</span> Tests Passed with Warnings (' + str(c["warns"]) + ')</div><div class="card"><table><thead><tr><th style="width:80px">Status</th><th>Test / Failed Step</th><th class="num">Duration</th></tr></thead><tbody>' + warning_rows + '</tbody></table></div></div>' if c["warns"] > 0 else ''}

  {'_halloumi_section(c, score_table_rows, score_dist_rows, total_scored)' if c["below_threshold"] else ''}

  {'<div class="section"><div class="section-title"><span class="icon">⏭️</span> Skipped — Knowledge Gaps (' + str(c["skipped"]) + ')</div><div class="card"><div style="color:var(--muted);font-size:12px;margin-bottom:12px">Chatbot correctly admitted lacking information. Expected behaviour, not bugs.</div><table><thead><tr><th style="width:80px">Status</th><th>Test / Reason</th></tr></thead><tbody>' + skipped_rows + '</tbody></table></div></div>' if c["skipped"] > 0 else ''}

  {llm_verdict_section}

  {'<div class="section"><div class="section-title"><span class="icon">⏱️</span> Slowest Tests (Top 10)</div><div class="card"><table><thead><tr><th>Test</th><th class="num">Duration</th></tr></thead><tbody>' + slowest_rows + '</tbody></table></div></div>' if c["slowest_tests"] else ''}

  <div style="color:var(--muted);font-size:12px;text-align:center;padding:24px 0;border-top:1px solid var(--border);margin-top:8px">
    Generated from <code>{_h(c['source_file'])}</code> · {c['session_label']} → {c['session_end_label']}
  </div>

</div>

<script>
{charts_js}
</script>
</body>
</html>"""


def _halloumi_section(c, score_table_rows, score_dist_rows, total_scored):
    return f"""
  <div class="section">
    <div class="section-title"><span class="icon">🧪</span> Halloumi Quality — Below Threshold (&lt;60%)</div>
    <div class="grid-2">
      <div class="card">
        <table>
          <thead><tr><th>Question</th><th class="num">Score</th></tr></thead>
          <tbody>{score_table_rows}</tbody>
        </table>
      </div>
      <div class="card">
        <div style="font-size:13px;color:var(--muted);margin-bottom:10px">Score distribution — {total_scored} scored responses:</div>
        <table>
          <thead><tr><th>Range</th><th class="num">Count</th><th>Share</th></tr></thead>
          <tbody>{score_dist_rows}</tbody>
        </table>
      </div>
    </div>
  </div>"""


def _test_table_rows(tests, outcome=None, warn=False) -> str:
    rows = ""
    for t in sorted(tests, key=lambda x: x.name):
        failed_steps = [s for s in t.steps if s.outcome == "failed"]
        steps_html = ""
        for s in failed_steps:
            msg = _h((s.message or "")[:300])
            steps_html += (
                f'<div class="step-fail"><b>{_h(s.step_name)}</b>'
                + (f'<div class="step-msg">{msg}</div>' if msg else "")
                + "</div>"
            )
        tmsg = _h((t.message or "")[:300])
        dur = f"{t.duration_seconds:.1f}s" if t.duration_seconds else ""
        badge = _outcome_badge(outcome or t.outcome, warn=warn or t.warn)
        rows += f"""
    <tr>
      <td>{badge}</td>
      <td>{_h(t.name)}
        {'<div class="test-msg">' + tmsg + '</div>' if tmsg and not warn else ''}
        <div class="step-details">{steps_html}</div>
      </td>
      <td class="num">{dur}</td>
    </tr>"""
    return rows


def _render_llm_verdicts(verdicts: dict) -> str:
    if not verdicts:
        return ""
    dimension_labels = {
        "relevance": "Relevance (on-topic)",
        "specificity": "Specificity (not vague)",
        "citations": "Citations",
        "information": "Information sufficiency",
    }
    rows = ""
    for dim in ["relevance", "specificity", "citations", "information"]:
        counts = verdicts.get(dim)
        if not counts:
            continue
        p = counts.get("passed", 0)
        f = counts.get("failed", 0)
        tot = p + f
        rate = (p / tot * 100) if tot > 0 else 0.0
        bar_color = _score_color(rate)
        rows += f"""
    <tr>
      <td>{dimension_labels.get(dim, dim.capitalize())}</td>
      <td class="num">{p}/{tot}</td>
      <td>
        <div class="bar-wrap"><div class="bar" style="width:{rate:.0f}%;background:{bar_color}"></div></div>
        <span class="pct">{rate:.0f}%</span>
      </td>
    </tr>"""
    if not rows:
        return ""
    return f"""
  <div class="section">
    <div class="section-title"><span class="icon">🤖</span> LLM Quality Verdicts</div>
    <div class="card">
      <table>
        <thead><tr><th>Dimension</th><th class="num">Pass / Total</th><th>Rate</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
  </div>"""


def _q_label(name: str) -> str:
    m = re.search(r'\[(\w+-\d+)', name)
    return m.group(1) if m else name


# ---------------------------------------------------------------------------
# CSS (single source of truth — embed inline so the file is self-contained)
# ---------------------------------------------------------------------------

_CSS = """<style>
  :root {
    --bg: #0f0f1a; --card: #1a1a2e; --card2: #16213e;
    --border: #2d2d44; --text: #e2e8f0; --muted: #94a3b8;
    --pass: #22c55e; --warn: #f59e0b; --fail: #ef4444;
    --skip: #6b7280; --accent: #6366f1;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; font-size: 14px; line-height: 1.6; }

  .header { background: var(--card2); border-bottom: 1px solid var(--border); padding: 24px 32px; }
  .header h1 { font-size: 22px; font-weight: 700; color: #fff; margin-bottom: 4px; }
  .header .meta { color: var(--muted); font-size: 13px; }
  .header .meta span { margin-right: 20px; }

  .container { max-width: 1400px; margin: 0 auto; padding: 24px 32px; }

  .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 16px; margin-bottom: 28px; }
  .kpi { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 18px 20px; }
  .kpi .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }
  .kpi .value { font-size: 32px; font-weight: 700; line-height: 1; }
  .kpi .sub { color: var(--muted); font-size: 12px; margin-top: 4px; }

  .section { margin-bottom: 28px; }
  .section-title { font-size: 15px; font-weight: 600; color: #fff; margin-bottom: 14px; padding-bottom: 8px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 8px; }

  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; }
  @media(max-width: 900px) { .grid-2, .grid-3 { grid-template-columns: 1fr; } }

  .card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 20px; }
  .chart-label { font-size: 13px; font-weight: 600; color: var(--muted); margin-bottom: 12px; }
  .chart-wrap { position: relative; height: 260px; }

  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { color: var(--muted); font-weight: 600; text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--border); font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; }
  td { padding: 8px 12px; border-bottom: 1px solid rgba(255,255,255,0.04); vertical-align: top; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: rgba(255,255,255,0.02); }
  .num { text-align: right; font-variant-numeric: tabular-nums; }

  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; letter-spacing: 0.04em; }
  .badge-pass { background: rgba(34,197,94,0.15); color: #22c55e; }
  .badge-fail { background: rgba(239,68,68,0.15); color: #ef4444; }
  .badge-warn { background: rgba(245,158,11,0.15); color: #f59e0b; }
  .badge-skip { background: rgba(107,114,128,0.15); color: #9ca3af; }

  .bar-wrap { display: inline-block; width: 80px; height: 6px; background: rgba(255,255,255,0.08); border-radius: 3px; vertical-align: middle; margin-right: 6px; }
  .bar { height: 100%; border-radius: 3px; }
  .pct { font-size: 12px; color: var(--muted); }

  .step-details { margin-top: 6px; }
  .step-fail { background: rgba(239,68,68,0.07); border-left: 3px solid #ef4444; padding: 6px 10px; border-radius: 0 4px 4px 0; margin-bottom: 4px; font-size: 12px; }
  .step-msg { color: var(--muted); font-family: monospace; font-size: 11px; margin-top: 4px; word-break: break-word; white-space: pre-wrap; }
  .test-msg { color: var(--muted); font-family: monospace; font-size: 11px; margin-top: 4px; word-break: break-word; }

  .insight-box { background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.3); border-radius: 8px; padding: 14px 16px; margin-bottom: 10px; font-size: 13px; }
  .insight-box.warn { background: rgba(245,158,11,0.08); border-color: rgba(245,158,11,0.3); }
  .insight-box.fail { background: rgba(239,68,68,0.08); border-color: rgba(239,68,68,0.3); }
  .insight-box.good { background: rgba(34,197,94,0.08); border-color: rgba(34,197,94,0.3); }

  code { background: rgba(255,255,255,0.07); padding: 1px 5px; border-radius: 3px; font-size: 12px; font-family: monospace; }
</style>"""
