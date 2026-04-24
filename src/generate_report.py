"""
Generate a self-contained HTML report from eval results.

Takes the summary.json + scores.json + raw_outputs.json written by run_eval
and produces report.html — a single file (no external assets) suitable for
sharing or embedding in a GitHub repo via GitHub Pages.

Usage:
    python -m src.generate_report --results-dir results/run_01
"""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path
from typing import Any


def generate_report(results_dir: Path) -> Path:
    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    scores = json.loads((results_dir / "scores.json").read_text(encoding="utf-8"))
    raw_outputs = json.loads((results_dir / "raw_outputs.json").read_text(encoding="utf-8"))

    html = _render_html(summary, scores, raw_outputs)
    report_path = results_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path


def _render_html(summary: dict, scores: list[dict], raw_outputs: list[dict]) -> str:
    strategies = summary["strategies"]
    ranking = summary["ranking_by_composite"]

    # --- Metric table ---
    metric_rows = _render_metric_table(strategies, ranking)

    # --- Per-case breakdown ---
    case_breakdown = _render_case_breakdown(scores, raw_outputs)

    # --- Bar chart via inline SVG ---
    composite_chart = _render_bar_chart(
        title="Composite score by strategy",
        data=[(s, strategies[s]["composite_mean"]) for s in ranking],
        max_val=1.0,
    )

    cost_chart = _render_bar_chart(
        title="Total cost by strategy (USD)",
        data=[(s, strategies[s]["total_cost_usd"]) for s in ranking],
        max_val=max((strategies[s]["total_cost_usd"] for s in ranking), default=0.01),
        fmt="${:.4f}",
    )

    latency_chart = _render_bar_chart(
        title="Avg latency per call (ms)",
        data=[(s, strategies[s]["avg_latency_ms"]) for s in ranking],
        max_val=max((strategies[s]["avg_latency_ms"] for s in ranking), default=1),
        fmt="{:.0f}ms",
    )

    # --- Dimension-specific charts ---
    dim_charts = []
    for metric, label in [
        ("escalation_accuracy", "Escalation accuracy"),
        ("no_hallucination_rate", "No-hallucination rate"),
        ("classification_accuracy", "Classification accuracy"),
        ("completeness_mean", "Completeness"),
    ]:
        dim_charts.append(
            _render_bar_chart(
                title=label,
                data=[(s, strategies[s][metric] or 0) for s in ranking],
                max_val=1.0,
            )
        )

    judge_note = ""
    if summary.get("judge_model"):
        judge_note = f"<p class=meta>LLM-as-judge scoring enabled — judge model: <code>{escape(summary['judge_model'])}</code></p>"

    return f"""<!doctype html>
<html lang=en>
<head>
<meta charset=utf-8>
<title>Triage Bench — Results</title>
<style>
  :root {{
    --bg: #0f1419;
    --surface: #1a1f26;
    --surface-2: #232a33;
    --text: #e6e8eb;
    --text-dim: #9aa4ae;
    --accent: #d97706;
    --good: #10b981;
    --bad: #ef4444;
    --border: #2d3540;
  }}
  body {{
    font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem 1.5rem 4rem;
    line-height: 1.5;
  }}
  h1 {{ font-size: 2rem; margin-bottom: 0.25rem; }}
  h2 {{ font-size: 1.4rem; margin-top: 2.5rem; border-bottom: 1px solid var(--border); padding-bottom: 0.4rem; }}
  h3 {{ font-size: 1.1rem; color: var(--text-dim); font-weight: 500; margin-top: 1.5rem; }}
  p.meta {{ color: var(--text-dim); margin: 0.25rem 0; }}
  code {{ background: var(--surface-2); padding: 0.1rem 0.4rem; border-radius: 3px; font-size: 0.9em; }}
  table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: 0.92rem; }}
  th, td {{ padding: 0.6rem 0.75rem; text-align: left; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--text-dim); font-weight: 600; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.05em; }}
  tr:hover td {{ background: var(--surface); }}
  .winner {{ color: var(--good); font-weight: 600; }}
  .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1rem; margin: 1rem 0; }}
  .chart {{ background: var(--surface); padding: 1rem; border-radius: 6px; border: 1px solid var(--border); }}
  .chart h4 {{ margin: 0 0 0.75rem; font-size: 0.95rem; color: var(--text-dim); font-weight: 500; }}
  .bar-row {{ display: flex; align-items: center; gap: 0.5rem; margin: 0.3rem 0; font-size: 0.85rem; }}
  .bar-label {{ width: 130px; color: var(--text-dim); text-align: right; font-family: ui-monospace, monospace; font-size: 0.8rem; }}
  .bar-track {{ flex: 1; height: 20px; background: var(--surface-2); border-radius: 3px; position: relative; overflow: hidden; }}
  .bar-fill {{ height: 100%; background: var(--accent); }}
  .bar-value {{ width: 70px; font-family: ui-monospace, monospace; font-size: 0.8rem; }}
  details {{ margin: 0.5rem 0; padding: 0.75rem; background: var(--surface); border: 1px solid var(--border); border-radius: 4px; }}
  details summary {{ cursor: pointer; font-weight: 500; user-select: none; }}
  details[open] summary {{ margin-bottom: 0.75rem; }}
  .pill {{ display: inline-block; padding: 0.1rem 0.5rem; border-radius: 10px; font-size: 0.75rem; font-weight: 600; margin-right: 0.4rem; }}
  .pill-good {{ background: rgba(16,185,129,0.15); color: var(--good); }}
  .pill-bad {{ background: rgba(239,68,68,0.15); color: var(--bad); }}
  .pill-neutral {{ background: var(--surface-2); color: var(--text-dim); }}
  pre {{ background: var(--surface-2); padding: 0.75rem; border-radius: 4px; overflow-x: auto; font-size: 0.82rem; white-space: pre-wrap; word-wrap: break-word; }}
  .ticket-text {{ font-style: italic; color: var(--text-dim); padding-left: 0.75rem; border-left: 3px solid var(--border); margin: 0.5rem 0; }}
</style>
</head>
<body>

<h1>Triage Bench — Results</h1>
<p class=meta>Model under test: <code>{escape(summary["model"])}</code></p>
{judge_note}
<p class=meta>Strategies compared: {escape(", ".join(ranking))}</p>

<h2>Strategy ranking</h2>
<p>Ranked by composite score (weighted: escalation 25%, no-hallucination 25%, classification 20%, tone 15%, completeness 10%, parse 5%). See the <code>composite</code> property in <code>evaluators.py</code> for the full weighting rationale.</p>
{metric_rows}

<h2>Headline charts</h2>
<div class=chart-grid>
{composite_chart}
{cost_chart}
{latency_chart}
</div>

<h2>Per-dimension breakdown</h2>
<div class=chart-grid>
{"".join(dim_charts)}
</div>

<h2>Per-case breakdown</h2>
<p class=meta>Click a case to expand and see the actual model output across strategies.</p>
{case_breakdown}

<p class=meta style="margin-top:3rem;font-size:0.85rem">
Generated by <code>src/generate_report.py</code>. Source and methodology: see README.md.
</p>

</body>
</html>
"""


def _render_metric_table(strategies: dict[str, dict], ranking: list[str]) -> str:
    headers = [
        "Strategy",
        "Composite",
        "Classif.",
        "Escalation",
        "No-hallu.",
        "Complete.",
        "Tone",
        "Quality",
        "Parse",
        "Cost",
        "Latency",
    ]
    header_html = "".join(f"<th>{h}</th>" for h in headers)

    rows = []
    for i, strat in enumerate(ranking):
        m = strategies[strat]
        cls = "winner" if i == 0 else ""
        cells = [
            f"<td class='{cls}'>{escape(strat)}</td>",
            f"<td class='{cls}'>{m['composite_mean']:.3f}</td>",
            f"<td>{_fmt_pct(m['classification_accuracy'])}</td>",
            f"<td>{_fmt_pct(m['escalation_accuracy'])}</td>",
            f"<td>{_fmt_pct(m['no_hallucination_rate'])}</td>",
            f"<td>{_fmt_pct(m['completeness_mean'])}</td>",
            f"<td>{_fmt_score(m['tone_score_mean'])}</td>",
            f"<td>{_fmt_score(m['quality_score_mean'])}</td>",
            f"<td>{_fmt_pct(m['parse_success_rate'])}</td>",
            f"<td>${m['total_cost_usd']:.4f}</td>",
            f"<td>{m['avg_latency_ms']}ms</td>",
        ]
        rows.append(f"<tr>{''.join(cells)}</tr>")

    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _render_bar_chart(
    title: str, data: list[tuple[str, float]], max_val: float, fmt: str = "{:.3f}"
) -> str:
    rows = []
    for label, value in data:
        pct = (value / max_val * 100) if max_val > 0 else 0
        rows.append(
            f"""<div class=bar-row>
            <span class=bar-label>{escape(label)}</span>
            <div class=bar-track><div class=bar-fill style="width:{pct:.1f}%"></div></div>
            <span class=bar-value>{fmt.format(value)}</span>
            </div>"""
        )
    return f"<div class=chart><h4>{escape(title)}</h4>{''.join(rows)}</div>"


def _render_case_breakdown(scores: list[dict], raw_outputs: list[dict]) -> str:
    # Group outputs by case_id
    by_case: dict[str, list[dict]] = {}
    for out in raw_outputs:
        by_case.setdefault(out["case_id"], []).append(out)

    scores_by_key: dict[tuple[str, str], dict] = {
        (s["case_id"], s["strategy"]): s for s in scores
    }

    html_parts = []
    for case_id in sorted(by_case.keys()):
        outputs = by_case[case_id]
        ticket = outputs[0]["ticket"]
        expected_cat = outputs[0]["expected_category"]
        expected_esc = outputs[0]["expected_escalate"]

        # Summary pills — how many strategies got it right
        strat_results = []
        for out in outputs:
            key = (case_id, out["strategy"])
            s = scores_by_key.get(key, {})
            cls_ok = s.get("classification_correct", 0)
            esc_ok = s.get("escalation_correct", 0)
            nohall = s.get("no_hallucination", 0)
            pill_class = "pill-good" if (cls_ok and esc_ok and nohall) else "pill-bad"
            label = f"{out['strategy']}: "
            label += "✓" if cls_ok else "✗"
            label += "✓" if esc_ok else "✗"
            label += "✓" if nohall else "✗"
            strat_results.append(
                f"<span class='pill {pill_class}'>{escape(label)}</span>"
            )

        # Per-strategy detail
        detail_parts = []
        for out in outputs:
            key = (case_id, out["strategy"])
            s = scores_by_key.get(key, {})
            parsed = out.get("parsed") or {}

            pills = []
            pills.append(
                f"<span class='pill {'pill-good' if s.get('classification_correct') else 'pill-bad'}'>cls: {escape(str(parsed.get('category', '—')))}</span>"
            )
            pills.append(
                f"<span class='pill {'pill-good' if s.get('escalation_correct') else 'pill-bad'}'>esc: {escape(str(parsed.get('escalate', '—')))}</span>"
            )
            pills.append(
                f"<span class='pill {'pill-good' if s.get('no_hallucination') else 'pill-bad'}'>no-hall</span>"
            )
            pills.append(
                f"<span class='pill pill-neutral'>complete: {s.get('completeness', 0):.2f}</span>"
            )
            if s.get("tone_score") is not None:
                pills.append(
                    f"<span class='pill pill-neutral'>tone: {s['tone_score']:.2f}</span>"
                )

            response_text = parsed.get("response", "") or out.get("raw_response", "")
            reasoning_text = parsed.get("reasoning", "")

            notes_html = ""
            if s.get("notes"):
                notes_html = f"<p style='color:var(--bad);font-size:0.85rem'>Notes: {escape('; '.join(s['notes']))}</p>"

            judge_html = ""
            if s.get("judge_rationale"):
                judge_html = f"<p style='color:var(--text-dim);font-size:0.85rem'><strong>Judge:</strong> {escape(s['judge_rationale'])}</p>"

            reasoning_html = ""
            if reasoning_text:
                reasoning_html = f"<p style='color:var(--text-dim);font-size:0.85rem'><strong>Model reasoning:</strong> {escape(reasoning_text)}</p>"

            detail_parts.append(f"""
            <div style="margin-top:1rem;padding-top:0.75rem;border-top:1px solid var(--border)">
              <strong>{escape(out['strategy'])}</strong><br>
              {''.join(pills)}
              <pre>{escape(response_text)}</pre>
              {reasoning_html}
              {notes_html}
              {judge_html}
            </div>
            """)

        html_parts.append(f"""
        <details>
          <summary>
            <strong>{escape(case_id)}</strong> — expected: {escape(expected_cat)}, escalate={str(expected_esc).lower()}
            &nbsp;&nbsp;{''.join(strat_results)}
          </summary>
          <div class=ticket-text>{escape(ticket)}</div>
          {''.join(detail_parts)}
        </details>
        """)

    return "\n".join(html_parts)


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v*100:.1f}%"


def _fmt_score(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent.parent / "results" / "latest",
    )
    args = parser.parse_args()

    path = generate_report(args.results_dir)
    print(f"Report written: {path}")


if __name__ == "__main__":
    main()
