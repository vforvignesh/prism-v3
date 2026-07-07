"""Self-contained HTML report: rankings, dimension bars, portfolios, shock heatmap."""
from datetime import datetime

import numpy as np
import pandas as pd

from .shock import SHOCK_SCENARIOS

CSS = """
body { font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; margin: 24px;
       background: #f7f8fa; color: #1a1d23; }
h1 { font-size: 22px; } h2 { font-size: 17px; margin-top: 32px; }
.meta { color: #667; font-size: 13px; }
table { border-collapse: collapse; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,.08);
        font-size: 13px; margin-top: 10px; }
th { background: #232936; color: #fff; padding: 6px 10px; text-align: right;
     position: sticky; top: 0; }
th:first-child, td:first-child { text-align: left; }
td { padding: 5px 10px; border-bottom: 1px solid #eceef2; text-align: right;
     white-space: nowrap; }
tr:hover td { background: #f0f4ff; }
.bar { display: inline-block; height: 10px; border-radius: 2px;
       background: linear-gradient(90deg,#4f8ef7,#6fc3ff); vertical-align: middle; }
.barwrap { display: inline-block; width: 70px; background: #e9ecf2;
           border-radius: 2px; margin-right: 6px; }
.frag { color: #c0392b; font-weight: 600; }
.pill { display: inline-block; padding: 1px 8px; border-radius: 10px;
        font-size: 11px; background: #e8f0fe; color: #29508a; margin: 1px; }
.warn { background: #fff3cd; border: 1px solid #ffe08a; border-radius: 6px;
        padding: 10px 14px; margin: 12px 0; font-size: 13px; }
.section { overflow-x: auto; }
"""


def _score_bar(v, vmax=100.0):
    pct = 0 if pd.isna(v) else max(0.0, min(1.0, v / vmax)) * 100
    return (f"<span class='barwrap'><span class='bar' style='width:{pct:.0f}%'>"
            f"</span></span>{v:.0f}")


def _heat(v, lo=-60, hi=0):
    """Red→white heat colour for drawdown percentages."""
    if pd.isna(v):
        return "#fff"
    t = max(0.0, min(1.0, (v - lo) / (hi - lo)))
    g = int(120 + t * 135)
    b = int(110 + t * 145)
    return f"rgb(245,{g},{b})"


def _fmt(v, fmt, na="–"):
    return na if pd.isna(v) else format(v, fmt)


def render_html_report(df, portfolios, shock_results, quality_df, cfg,
                       fragile_stocks=(), run_date=None):
    run_date = run_date or datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = [f"<style>{CSS}</style>"]
    parts.append("<h1>PRISM — Scored Watchlist</h1>")
    parts.append(
        f"<div class='meta'>Generated {run_date} · {len(df)} stocks · weights "
        f"G {cfg['weight_growth']:.0%} / V {cfg['weight_value']:.0%} / "
        f"M {cfg['weight_momentum']:.0%} / R {cfg['weight_resilience']:.0%}</div>")

    # Data quality banner
    defaulted = quality_df[(quality_df["G26_conf"] == "DEFAULT")
                           | (quality_df["G27_conf"] == "DEFAULT")]["Stock"].tolist()
    if defaulted:
        parts.append(
            f"<div class='warn'>⚠️ No growth estimate for {', '.join(defaulted)} — "
            "a flat 4% was assumed. Consider <code>growth_overrides</code> in "
            "config.yaml.</div>")

    # Rankings table
    parts.append("<h2>Rankings</h2><div class='section'><table>")
    parts.append("<tr><th>Stock</th><th>Rank</th><th>Score</th><th>Growth</th>"
                 "<th>Value</th><th>Momentum</th><th>Resilience</th><th>G26</th>"
                 "<th>G27</th><th>Fwd P/E</th><th>PEG</th><th>Beta</th><th>6M Ret</th>"
                 "<th>52w</th><th>Upside</th><th>Sector</th><th>Fragile</th></tr>")
    for _, r in df.iterrows():
        frag = (f"<span class='frag'>⚑ {r['fragile_points']:.1f}</span>"
                if r["fragile_flag"] else f"{r['fragile_points']:.1f}")
        ret6 = r.get("Ret 6M", np.nan)
        parts.append(
            f"<tr><td><b>{r['Stock']}</b></td><td>{int(r['prism_rank'])}</td>"
            f"<td>{_score_bar(r['prism_score'])}</td>"
            f"<td>{_fmt(r['dim_growth'], '.0f')}</td><td>{_fmt(r['dim_value'], '.0f')}</td>"
            f"<td>{_fmt(r['dim_momentum'], '.0f')}</td><td>{_fmt(r['dim_resilience'], '.0f')}</td>"
            f"<td>{_fmt(r['2026 Growth Rate'], '+.0%')}</td>"
            f"<td>{_fmt(r['2027 Growth Rate'], '+.0%')}</td>"
            f"<td>{_fmt(r['P/E.1'], '.1f')}</td><td>{_fmt(r['peg_proxy'], '.2f')}</td>"
            f"<td>{_fmt(r['Beta'], '.2f')}</td><td>{_fmt(ret6, '+.0%')}</td>"
            f"<td>{_fmt(r['52w_position'], '.0%')}</td>"
            f"<td>{_fmt(r['Average Outcome'], '+.0%')}</td>"
            f"<td>{r['shock_sector']}</td><td>{frag}</td></tr>")
    parts.append("</table></div>")

    # Portfolios
    parts.append("<h2>Portfolios</h2>")
    for pname, sl in portfolios.items():
        pills = "".join(f"<span class='pill'>{s}</span>" for s in sl)
        w = 100 / len(sl) if sl else 0
        parts.append(f"<p><b>{pname}</b> ({len(sl)} × {w:.1f}%)<br>{pills}</p>")

    # Shock heatmap
    parts.append("<h2>Shock stress test</h2><div class='section'><table>")
    pnames = list(shock_results.keys())
    parts.append("<tr><th>Scenario</th><th>Prob</th>"
                 + "".join(f"<th>{pn}</th>" for pn in pnames) + "<th>SPY</th></tr>")
    for sn, sp in SHOCK_SCENARIOS.items():
        cells = "".join(
            f"<td style='background:{_heat(shock_results[pn].get(sn, np.nan))}'>"
            f"{_fmt(shock_results[pn].get(sn, np.nan), '+.0f')}%</td>"
            for pn in pnames)
        parts.append(f"<tr><td>{sn}</td><td>{sp['probability']:.0%}</td>{cells}"
                     f"<td style='background:{_heat(sp['market_dd']*100)}'>"
                     f"{sp['market_dd']*100:+.0f}%</td></tr>")
    spy_score = sum(s["probability"] * s["market_dd"] * 100 for s in SHOCK_SCENARIOS.values())
    parts.append("<tr><td><b>SHOCK Score</b></td><td></td>" + "".join(
        f"<td><b>{shock_results[pn]['SHOCK Score']:+.1f}%</b></td>" for pn in pnames)
        + f"<td><b>{spy_score:+.1f}%</b></td></tr>")
    parts.append("<tr><td><b>Edge Ratio</b></td><td></td>" + "".join(
        f"<td><b>{shock_results[pn]['Edge Ratio']:.2f}x</b></td>" for pn in pnames)
        + "<td></td></tr>")
    parts.append("</table></div>")

    best = max(shock_results, key=lambda x: shock_results[x]["Edge Ratio"])
    b = shock_results[best]
    parts.append(f"<h2>Recommendation: {best}</h2>"
                 f"<p>Edge Ratio <b>{b['Edge Ratio']:.2f}x</b> · growth "
                 f"{b['Growth']:.0%} · expected shock {b['SHOCK Score']:+.1f}% · "
                 f"{b['N']} positions</p>")

    # Quality appendix
    parts.append("<h2>Data quality</h2><div class='section'><table>")
    parts.append("<tr><th>Stock</th><th>G26 conf</th><th>G27 conf</th>"
                 "<th>Final g26</th><th>Final g27</th></tr>")
    for _, q in quality_df.iterrows():
        parts.append(
            f"<tr><td>{q['Stock']}</td><td>{q['G26_conf']}</td><td>{q['G27_conf']}</td>"
            f"<td>{_fmt(q['Final_g26'], '+.0%')}</td>"
            f"<td>{_fmt(q['Final_g27'], '+.0%')}</td></tr>")
    parts.append("</table></div>")

    return ("<!doctype html><html><head><meta charset='utf-8'>"
            "<title>PRISM report</title></head><body>"
            + "".join(parts) + "</body></html>")


def export_html(df, portfolios, shock_results, quality_df, cfg, out_dir,
                fragile_stocks=()):
    html = render_html_report(df, portfolios, shock_results, quality_df, cfg,
                              fragile_stocks=fragile_stocks)
    path = out_dir / "prism_report.html"
    path.write_text(html, encoding="utf-8")
    print(f"  HTML report: {path}")
    return path
