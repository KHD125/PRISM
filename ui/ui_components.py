"""
Multibagger Discovery System — UI Components
=============================================
Reusable Streamlit UI widgets, cards, and charts.
Premium dark-mode design system.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import html as _html
from config import COLORS, TIER_COLORS, CONVICTION_TIERS, UI


def inject_css():
    """Inject the premium dark-mode CSS design system."""
    st.markdown(f"""
    <style>
    @import url('{UI["font_url"]}');

    /* ── Global ── */
    html, body, [data-testid="stAppViewContainer"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    /* ── Responsive Layout ── */
    /* padding-top must clear Streamlit's fixed ~3.75rem header bar, otherwise the first
       content row (the Mandate selector buttons) is hidden underneath it. */
    section[data-testid="stMain"] > div.block-container {{
        max-width: 100%; overflow-x: hidden; box-sizing: border-box;
        padding-top: 3.5rem;
    }}
    /* Blend Streamlit's header bar into the dark theme (keeps the toolbar usable, no white bar) */
    [data-testid="stHeader"] {{
        background: transparent;
    }}
    /* Tighten the default top-of-app gap so the extra header clearance doesn't feel empty */
    section[data-testid="stMain"] > div.block-container > div:first-child {{
        padding-top: 0;
    }}

    /* ── Hero Banner ── */
    .hero-banner {{
        text-align: center; padding: 2rem 1.5rem 1.8rem;
        background: linear-gradient(135deg, {COLORS['gradient_start']} 0%,
                    {COLORS['gradient_mid']} 40%, {COLORS['gradient_end']} 100%);
        border: 1px solid rgba(88,166,255,0.15);
        border-radius: 16px; margin-bottom: 1.5rem;
        position: relative; overflow: hidden;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05);
    }}
    .hero-banner::before {{
        content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(ellipse at 30% 20%, rgba(228,179,65,0.08) 0%, transparent 50%),
                    radial-gradient(ellipse at 70% 80%, rgba(139,92,246,0.06) 0%, transparent 50%);
        pointer-events: none;
    }}
    .hero-icon {{ font-size: 3rem; line-height: 1; position: relative;
        filter: drop-shadow(0 0 14px rgba(255,215,0,0.5)); margin-bottom: 6px; }}
    .hero-title {{
        font-size: 2.4rem; font-weight: 900; position: relative;
        background: linear-gradient(120deg, {COLORS['gold']} 0%, {COLORS['blue']} 50%, {COLORS['green']} 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        letter-spacing: 1.5px; line-height: 1.15; margin: 0;
    }}
    .hero-sub {{
        font-size: 0.85rem; font-weight: 500; color: {COLORS['text_secondary']};
        letter-spacing: 2.5px; text-transform: uppercase;
        position: relative; margin-top: 8px;
    }}
    .hero-badge {{
        display: inline-block; font-size: 0.65rem; font-weight: 700;
        color: {COLORS['gold']}; background: rgba(228,179,65,0.10);
        border: 1px solid rgba(228,179,65,0.25); padding: 4px 16px;
        border-radius: 12px; margin-top: 12px; position: relative;
        letter-spacing: 1px;
    }}

    /* ── Metric Strip ── */
    .m-strip {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }}
    .m-chip {{
        background: {COLORS['bg_secondary']}; border: 1px solid {COLORS['border']};
        border-radius: 12px; padding: 14px 0; text-align: center; flex: 1; min-width: 100px;
        transition: all 0.2s ease;
    }}
    .m-chip:hover {{ border-color: {COLORS['border_hover']}; transform: translateY(-1px); }}
    .m-val {{ font-size: 1.5rem; font-weight: 700; color: {COLORS['text_primary']}; line-height: 1; }}
    .m-lbl {{ font-size: 0.65rem; color: {COLORS['text_secondary']}; text-transform: uppercase;
              letter-spacing: 0.6px; margin-top: 4px; }}
    .m-green .m-val {{ color: {COLORS['green']}; }}
    .m-red .m-val {{ color: {COLORS['red']}; }}
    .m-gold .m-val {{ color: {COLORS['gold']}; }}
    .m-blue .m-val {{ color: {COLORS['blue']}; }}
    .m-purple .m-val {{ color: {COLORS['purple']}; }}

    /* ── Stock Cards ── */
    .stock-card {{
        background: {COLORS['bg_secondary']}; border: 1px solid {COLORS['border']};
        border-radius: 14px; padding: 18px 20px; margin-bottom: 10px;
        transition: all 0.2s ease; cursor: default;
    }}
    .stock-card:hover {{
        border-color: {COLORS['border_hover']};
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }}
    .stock-card-gold {{ border-left: 3px solid {COLORS['gold']}; }}
    .stock-card-green {{ border-left: 3px solid {COLORS['green']}; }}
    .stock-card-blue {{ border-left: 3px solid {COLORS['blue']}; }}

    /* ── Score Bar ── */
    .score-bar-wrap {{
        background: {COLORS['bg_tertiary']}; border-radius: 4px; height: 6px;
        margin-top: 4px; overflow: hidden;
    }}
    .score-bar {{
        height: 6px; border-radius: 4px; transition: width 0.5s ease;
    }}

    /* ── Pill Tags ── */
    .pill {{
        display: inline-block; padding: 3px 10px; border-radius: 10px;
        font-size: 0.7rem; font-weight: 600; margin: 2px 3px; border: 1px solid;
    }}
    .pill-green {{ color: {COLORS['green']}; border-color: rgba(63,185,80,0.3);
                   background: rgba(63,185,80,0.08); }}
    .pill-red {{ color: {COLORS['red']}; border-color: rgba(248,81,73,0.3);
                 background: rgba(248,81,73,0.08); }}
    .pill-gold {{ color: {COLORS['gold']}; border-color: rgba(228,179,65,0.3);
                  background: rgba(228,179,65,0.08); }}
    .pill-blue {{ color: {COLORS['blue']}; border-color: rgba(88,166,255,0.3);
                  background: rgba(88,166,255,0.08); }}
    .pill-purple {{ color: {COLORS['purple']}; border-color: rgba(139,92,246,0.3);
                    background: rgba(139,92,246,0.08); }}

    /* ── Tier Card ── */
    .tier-card {{
        border-radius: 12px; padding: 16px 20px; margin-bottom: 10px;
        transition: all 0.2s ease;
    }}
    .tier-card:hover {{ transform: translateY(-1px); }}
    .tier-header {{
        display: flex; justify-content: space-between; align-items: center;
        margin-bottom: 8px;
    }}
    .tier-name {{ font-weight: 800; font-size: 1rem; }}
    .tier-count {{
        font-size: 0.75rem; font-weight: 600; padding: 3px 10px;
        border-radius: 8px; background: rgba(255,255,255,0.08);
    }}

    /* ── Section Headers ── */
    .sec-head {{
        font-size: 0.9rem; font-weight: 700; color: {COLORS['text_primary']};
        letter-spacing: 0.3px; margin: 24px 0 10px 0;
        display: flex; align-items: center; gap: 8px;
    }}
    .sec-cap {{
        font-size: 0.72rem; color: {COLORS['text_muted']};
        margin-top: -6px; margin-bottom: 12px;
    }}

    /* ── DataFrames ── */
    div[data-testid="stDataFrame"] > div {{ border-radius: 10px; overflow: hidden; }}

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background: {COLORS['bg_secondary']};
        border-radius: 12px 12px 0 0;
        padding: 6px 6px 0 6px;
        border-bottom: 2px solid {COLORS['border']};
    }}
    .stTabs [data-baseweb="tab"] {{
        padding: 10px 20px;
        font-weight: 600;
        border-radius: 10px 10px 0 0;
        font-size: 0.85rem;
        color: {COLORS['text_secondary']} !important;
        background: transparent;
        border: 1px solid transparent;
        transition: all 0.2s ease;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        color: {COLORS['text_primary']} !important;
        background: rgba(255,255,255,0.05);
    }}
    .stTabs [aria-selected="true"] {{
        color: {COLORS['gold']} !important;
        background: rgba(228,179,65,0.08) !important;
        border-color: rgba(228,179,65,0.3) !important;
        border-bottom-color: transparent !important;
    }}

    /* ── Sidebar ── */
    .sb-brand {{
        background: linear-gradient(135deg, rgba(139,92,246,0.08) 0%,
                    rgba(228,179,65,0.10) 50%, rgba(63,185,80,0.08) 100%);
        border: 1px solid rgba(228,179,65,0.3); border-radius: 16px;
        padding: 20px 14px 14px; text-align: center; margin-bottom: 16px;
        position: relative; overflow: hidden;
    }}
    .sb-brand::before {{
        content: ''; position: absolute; top: -40%; left: -40%;
        width: 180%; height: 180%;
        background: radial-gradient(circle, rgba(228,179,65,0.08) 0%, transparent 70%);
        animation: sb-pulse 6s ease-in-out infinite;
    }}
    @keyframes sb-pulse {{ 0%,100% {{ opacity: 0.4; }} 50% {{ opacity: 1; }} }}
    .sb-brand-icon {{ font-size: 2.2rem; position: relative; line-height: 1;
        filter: drop-shadow(0 0 8px rgba(228,179,65,0.4)); margin-bottom: 4px; }}
    .sb-brand-title {{
        font-size: 1.15rem; font-weight: 800; position: relative;
        background: linear-gradient(120deg, {COLORS['gold']} 0%, {COLORS['purple']} 50%, {COLORS['green']} 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .sb-brand-ver {{
        display: inline-block; font-size: 0.6rem; font-weight: 700;
        color: {COLORS['gold']}; background: rgba(228,179,65,0.10);
        border: 1px solid rgba(228,179,65,0.3); padding: 2px 10px;
        border-radius: 10px; margin-top: 6px; position: relative;
    }}

    /* ── Tsunami Card ── */
    .tsunami-card {{
        background: linear-gradient(135deg, {COLORS['gradient_start']} 0%, {COLORS['gradient_mid']} 100%);
        border: 1px solid rgba(139,92,246,0.4); border-radius: 14px;
        padding: 18px 20px; margin-bottom: 10px;
        box-shadow: 0 4px 20px rgba(139,92,246,0.15);
    }}
    .tsunami-card:hover {{ box-shadow: 0 8px 32px rgba(139,92,246,0.25); }}

    /* ── Forensic Risk Badge ── */
    .risk-clean {{ color: {COLORS['green']}; background: rgba(63,185,80,0.1);
                   border: 1px solid rgba(63,185,80,0.3); }}
    .risk-watch {{ color: {COLORS['gold']}; background: rgba(228,179,65,0.1);
                   border: 1px solid rgba(228,179,65,0.3); }}
    .risk-caution {{ color: {COLORS['orange']}; background: rgba(255,107,53,0.1);
                     border: 1px solid rgba(255,107,53,0.3); }}
    .risk-high {{ color: {COLORS['red']}; background: rgba(248,81,73,0.1);
                  border: 1px solid rgba(248,81,73,0.3); }}

    /* ══════════════════════════════════════════════
       TEARSHEET — Premium section styles
       ══════════════════════════════════════════════ */

    /* Hero header card */
    .ts-hero {{
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
        border: 1px solid {COLORS['border']};
        border-radius: 20px; padding: 28px 30px; margin-bottom: 14px;
        position: relative; overflow: hidden;
        box-shadow: 0 8px 32px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.04);
    }}
    .ts-hero::before {{
        content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(ellipse at 85% 50%, rgba(88,166,255,0.06) 0%, transparent 55%),
                    radial-gradient(ellipse at 10% 80%, rgba(139,92,246,0.04) 0%, transparent 50%);
        pointer-events: none;
    }}
    .ts-score-ring {{
        width: 110px; height: 110px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        flex-direction: column; flex-shrink: 0;
        border-width: 3px; border-style: solid;
        box-shadow: 0 0 28px rgba(0,0,0,0.5);
        position: relative;
    }}
    .ts-score-val {{
        font-size: 2.6rem; font-weight: 900; line-height: 1;
    }}
    .ts-score-lbl {{
        font-size: 0.5rem; letter-spacing: 1px; text-transform: uppercase;
        margin-top: 2px; opacity: 0.7;
    }}

    /* Score strip grid */
    .ts-score-strip {{
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 8px; margin: 10px 0 16px 0;
    }}
    .ts-score-cell {{
        background: {COLORS['bg_secondary']};
        border: 1px solid {COLORS['border']};
        border-radius: 12px; padding: 12px 10px;
        text-align: center; transition: border-color 0.2s ease;
    }}
    .ts-score-cell:hover {{ border-color: {COLORS['border_hover']}; }}
    .ts-score-cell-lbl {{
        font-size: 0.58rem; color: {COLORS['text_muted']};
        text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 4px;
    }}
    .ts-score-cell-val {{
        font-size: 1.9rem; font-weight: 900; line-height: 1;
    }}
    .ts-score-bar-bg {{
        height: 3px; background: {COLORS['bg_tertiary']};
        border-radius: 2px; margin-top: 7px; overflow: hidden;
    }}
    .ts-score-bar-fill {{
        height: 3px; border-radius: 2px;
    }}

    /* Framework grid cards */
    .ts-fw-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
        gap: 8px; margin-top: 8px;
    }}
    .ts-fw-card {{
        border-radius: 10px; padding: 11px 13px;
        border-width: 1px; border-style: solid; transition: transform 0.15s ease;
    }}
    .ts-fw-card:hover {{ transform: translateY(-2px); }}
    .ts-fw-card-head {{
        display: flex; align-items: center; gap: 8px; margin-bottom: 5px;
    }}
    .ts-fw-card-name {{ font-weight: 700; font-size: 0.78rem; }}
    .ts-fw-card-desc {{ font-size: 0.65rem; color: {COLORS['text_muted']}; line-height: 1.4; }}

    /* Flag rows in forensics */
    .ts-flag-row {{
        display: flex; align-items: flex-start; gap: 12px;
        padding: 9px 14px; border-radius: 6px; margin: 4px 0;
        border-left-width: 3px; border-left-style: solid;
    }}
    .ts-flag-sev {{ font-size: 1rem; flex-shrink: 0; margin-top: 1px; }}
    .ts-flag-title {{ font-size: 0.78rem; font-weight: 600; line-height: 1.3; }}
    .ts-flag-sub {{ font-size: 0.66rem; color: {COLORS['text_muted']}; margin-top: 2px; line-height: 1.4; }}

    /* Fisher proxy cards */
    .ts-fisher-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 8px; margin-top: 10px;
    }}
    .ts-fisher-card {{
        border-radius: 9px; padding: 10px 12px;
        border-width: 1px; border-style: solid;
    }}
    .ts-fisher-head {{
        display: flex; align-items: center; gap: 7px; margin-bottom: 4px;
    }}
    .ts-fisher-key {{ font-size: 0.68rem; font-weight: 700; }}
    .ts-fisher-val {{ font-size: 0.85rem; font-weight: 800; }}

    /* Sell alert banners */
    .ts-sell-banner {{
        border-radius: 12px; padding: 14px 18px; margin: 6px 0;
        border-width: 1px; border-style: solid;
        display: flex; align-items: flex-start; gap: 12px;
    }}
    .ts-sell-icon {{ font-size: 1.4rem; flex-shrink: 0; }}
    .ts-sell-title {{ font-size: 0.88rem; font-weight: 800; margin-bottom: 2px; }}
    .ts-sell-body {{ font-size: 0.75rem; }}

    /* Raw data metric cell */
    .ts-raw-cell {{
        background: {COLORS['bg_secondary']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px; padding: 10px 12px; text-align: center;
    }}
    .ts-raw-lbl {{
        font-size: 0.6rem; color: {COLORS['text_muted']};
        text-transform: uppercase; letter-spacing: 0.5px;
    }}
    .ts-raw-val {{
        font-size: 1.1rem; font-weight: 800;
        color: {COLORS['text_primary']}; margin-top: 3px;
    }}

    /* KPI strip in forensics */
    .ts-kpi-strip {{
        display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 14px;
    }}
    .ts-kpi-cell {{
        background: {COLORS['bg_secondary']};
        border: 1px solid {COLORS['border']};
        border-radius: 10px; padding: 10px 16px;
        min-width: 100px; text-align: center; flex: 1;
    }}
    .ts-kpi-val {{ font-size: 1.5rem; font-weight: 900; line-height: 1; }}
    .ts-kpi-lbl {{
        font-size: 0.58rem; color: {COLORS['text_muted']};
        text-transform: uppercase; letter-spacing: 0.6px; margin-top: 3px;
    }}
    </style>
    """, unsafe_allow_html=True)


def render_hero_banner(total_stocks: int, gate_passed: int, tier1_count: int):
    """Render the main hero banner."""
    st.markdown(f"""
    <div class="hero-banner">
        <div class="hero-icon">🏆</div>
        <h1 class="hero-title">{UI['app_title']}</h1>
        <p class="hero-sub">{UI['app_subtitle']}</p>
        <div class="hero-badge">v{UI['version']} · {total_stocks} STOCKS SCANNED · {gate_passed} QUALIFIED · {tier1_count} CROWN JEWELS</div>
    </div>
    """, unsafe_allow_html=True)


def render_metric_strip(metrics: list):
    """Render a horizontal metric strip. Each metric: (value, label, color_class)."""
    chips = ""
    for val, label, cls in metrics:
        chips += f'<div class="m-chip {cls}"><div class="m-val">{val}</div><div class="m-lbl">{label}</div></div>'
    st.markdown(f'<div class="m-strip">{chips}</div>', unsafe_allow_html=True)


def render_score_bar(score: float, color: str = "#3fb950", label: str = ""):
    """Render a horizontal score bar. Clamps width to [0, 100] so bars never overflow.
    Negative values (e.g. governance_bonus < 0) show a red penalty badge instead of invisible bar."""
    _width = min(100, max(0, float(score or 0)))
    _is_negative = float(score or 0) < 0
    _display_val = f'<span style="color:{COLORS["red"]};font-weight:700;">⚠ {score:.0f}</span>' if _is_negative else f'<span style="font-size:0.75rem;font-weight:700;color:{color};min-width:30px;">{score:.0f}</span>'
    html = f"""
    <div style="display:flex; align-items:center; gap:8px; margin:2px 0;">
        <span style="font-size:0.7rem; color:{COLORS['text_secondary']}; min-width:50px;">{label}</span>
        <div class="score-bar-wrap" style="flex:1;">
            <div class="score-bar" style="width:{_width:.1f}%; background:{color};"></div>
        </div>
        {_display_val}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_stock_card(row: pd.Series, show_scores: bool = True):
    """Render a premium stock card."""
    tier = int(row.get("conviction_tier", 5))
    tc = TIER_COLORS.get(tier, TIER_COLORS[5])

    gate_status = "✅ All gates passed" if row.get("gate_pass", 0) == 1 else f"❌ {int(row.get('gates_failed', 0))} gates failed"

    pills = ""
    # Catalysts
    if row.get("cat_capacity", 0) == 1:
        pills += '<span class="pill pill-blue">🔥 Capacity Explosion</span>'
    if row.get("cat_oplev", 0) == 1:
        pills += '<span class="pill pill-green">🔥 OpLev Inflection</span>'
    if row.get("cat_inst_discovery", 0) == 1:
        pills += '<span class="pill pill-purple">🔥 Inst Discovery</span>'
    if row.get("cat_deleveraging", 0) == 1:
        pills += '<span class="pill pill-gold">🔥 Deleveraging Cycle</span>'
        
    # Frameworks — generic gray pills for every passed framework.
    # EXCEPT those that have a dedicated colour pill below (avoids duplicate display).
    _DEDICATED_FW = {"100x Candidate", "Bruised Blue Chip 29"}
    fw_str = row.get("frameworks_passed", "None")
    if fw_str != "None":
        for fw in fw_str.split(", "):
            if fw.strip() in _DEDICATED_FW:
                continue
            pills += f'<span class="pill" style="border-color:rgba(255,255,255,0.4); color:#eee; background:rgba(255,255,255,0.05);">🏛️ {_html.escape(fw)}</span>'
            
    # Legacy specific tags
    if row.get("tsunami_signal", 0) == 1:
        pills += '<span class="pill pill-gold">🌊 Tsunami</span>'
    if row.get("net_debt_negative", 0) == 1:
        pills += '<span class="pill pill-green">Net Cash</span>'
        
    # ── ALPHA VECTORS ──
    # 1. Moat Growth Matrix
    mg_quad = row.get("moat_growth_quad", "")
    if "Wealth Creator" in mg_quad:
        pills += f'<span class="pill pill-green">{mg_quad}</span>'
    elif "Quality Trap" in mg_quad:
        pills += f'<span class="pill pill-gold">{mg_quad}</span>'
    elif "Growth Trap" in mg_quad:
        pills += f'<span class="pill pill-blue">{mg_quad}</span>'
    elif "Destroyer" in mg_quad:
        pills += f'<span class="pill pill-red">{mg_quad}</span>'
        
    # 2. Cash Machine (Accrual Anomaly)
    cm_label = row.get("cash_machine_label", "")
    if "Cash Machine" in cm_label:
        pills += f'<span class="pill pill-green">{cm_label}</span>'
    elif "Paper Profits" in cm_label:
        pills += f'<span class="pill pill-red">{cm_label}</span>'
        
    # 3. Buy Zone (Actionability)
    bz_label = row.get("buy_zone_label", "")
    if "Perfect Entry" in bz_label:
        pills += f'<span class="pill pill-green">{bz_label}</span>'
    elif "Extended" in bz_label:
        pills += f'<span class="pill pill-red">{bz_label}</span>'

    # 4. Bruised Blue Chip (WCS 29): use the engine-computed flag (P/B ≤2.0 + sector tailwind)
    #    Emoji 🏛️ matches the _FW_META "Bruised Blue Chip 29" entry (zero-duplicate matrix).
    if row.get("bruised_blue_chip_29", 0) == 1:
        pills += '<span class="pill pill-blue">🏛️ Bruised Blue Chip</span>'

    # 5. 100x Candidate (17th WCS Mouse-to-Elephant) — rare, high-conviction alpha tag
    #    Emoji 🐘 (literal "Mouse-to-Elephant") matches the _FW_META "100x Candidate" entry.
    if row.get("mosl_100x_candidate", 0) == 1:
        pills += '<span class="pill pill-gold">🐘 100x Candidate</span>'

    # 6. Atoms→Bits business design (26th WCS) — surface asset-light "Bits" vs capital-heavy "Atoms"
    _atb = str(row.get("atoms_to_bits_label", "") or "")
    if _atb == "Bits":
        pills += '<span class="pill pill-blue">💡 Bits</span>'
    elif _atb == "Atoms":
        pills += ('<span class="pill" style="border-color:rgba(228,179,65,0.3);'
                  'color:#e3b341;background:rgba(228,179,65,0.06);">🏭 Atoms</span>')

    # ── Verdict strip: gate + forensic + moat quadrant in one scannable line ──
    _gate_ok = row.get("gate_pass", 0) == 1
    _failed_esc = _html.escape(str(row.get("failed_gates", "") or ""))
    if _gate_ok:
        _gate_v = f'<span style="color:{COLORS["green"]};font-size:0.72rem;font-weight:700;">✅ All Gates</span>'
    else:
        _short = (_failed_esc[:38] + "…") if len(_failed_esc) > 38 else _failed_esc
        _gate_v = f'<span style="color:{COLORS["red"]};font-size:0.72rem;font-weight:700;">❌ {_short}</span>'

    _f_lbl = str(row.get("forensic_label", "🟢 Clean") or "🟢 Clean")
    _f_cnt = int(row.get("red_flag_count", 0) or 0)
    if "Clean" in _f_lbl:
        _forensic_v = f'<span style="color:{COLORS["green"]};font-size:0.72rem;">🟢 Clean</span>'
    else:
        _forensic_v = f'<span style="color:{COLORS["red"]};font-size:0.72rem;">🔴 Risk — {_f_cnt}⚑</span>'

    # The moat-growth quadrant is already a coloured pill in the tag row — don't duplicate it here.
    # The status strip is now just gate + forensic, complementing (not repeating) the verdict chip.
    _dot = '<span style="color:#555;font-size:0.65rem;margin:0 3px;">·</span>'
    _verdict_strip = _dot.join([_gate_v, _forensic_v])

    if show_scores:
        _bars = [
            ("Moat",       row.get("moat_score",      0), COLORS["purple"]),
            ("Growth",     row.get("growth_score",     0), COLORS["green"]),
            ("Cash",       row.get("cash_score",       0), COLORS["blue"]),
            ("Momentum",   row.get("momentum_score",   0), COLORS["orange"]),
            ("Governance", row.get("governance_bonus", 0), COLORS["gold"]),
        ]
        _bar_items = ""
        for lbl, sc, clr in _bars:
            _w = min(100, max(0, float(sc or 0)))
            _bar_items += (
                f'<div style="flex:1;">'
                f'<div style="font-size:0.57rem;color:{COLORS["text_secondary"]};margin-bottom:3px;'
                f'text-transform:uppercase;letter-spacing:0.5px;">{lbl}</div>'
                f'<div style="background:{COLORS["bg_tertiary"]};border-radius:4px;height:5px;overflow:hidden;">'
                f'<div style="width:{_w:.0f}%;height:5px;border-radius:4px;background:{clr};"></div>'
                f'</div>'
                f'</div>'
            )
        _score_bars_html = (
            f'<div style="display:flex;gap:6px;margin-top:11px;padding-top:10px;'
            f'border-top:1px solid rgba(255,255,255,0.05);">{_bar_items}</div>'
        )
    else:
        _score_bars_html = ""

    _esc_name     = _html.escape(str(row.get('name', 'N/A') or 'N/A'))
    _esc_sector   = _html.escape(str(row.get('sector', '') or ''))
    _esc_industry = _html.escape(str(row.get('industry', '') or ''))
    _esc_mcat     = _html.escape(str(row.get('market_category', '') or ''))

    # ── Engine verdict chip: the card LEADS with the decision (scan the list by BUY/WATCH/AVOID) ──
    _vdir   = str(row.get("verdict_direction", "") or "")
    _vemoji = str(row.get("verdict_emoji", "") or "")
    _vclr = {"BUY": COLORS["green"], "WATCH": COLORS["gold"], "AVOID": COLORS["text_muted"]}.get(
        _vdir, COLORS["text_muted"])
    _verdict_chip = (
        f'<div style="display:inline-block;font-size:0.78rem;font-weight:900;color:{_vclr};'
        f'letter-spacing:0.6px;white-space:nowrap;margin-bottom:5px;padding:3px 11px;border-radius:11px;'
        f'background:{_vclr}1f;border:1px solid {_vclr}66;">{_vemoji} {_html.escape(_vdir)}</div>'
    ) if _vdir else ""

    card_html = f"""
    <div class="stock-card" style="border-left: 3px solid {tc['border']};">
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div style="flex:1;min-width:0;">
                <div style="font-weight:800; font-size:1.05rem; color:{COLORS['text_primary']};">
                    {row.get('tier_emoji', '')} #{int(row.get('rank', 0))} · {_esc_name}
                </div>
                <div style="font-size:0.75rem; color:{COLORS['text_secondary']}; margin-top:2px;">
                    {_esc_sector} · {_esc_industry} · ₹{row.get('market_cap', 0):,.0f} Cr · {_esc_mcat}
                </div>
                <div style="margin-top:5px;">{_verdict_strip}</div>
            </div>
            <div style="text-align:right;flex-shrink:0;margin-left:12px;">
                {_verdict_chip}
                <div style="font-size:1.8rem; font-weight:900; color:{tc['text']};">{row.get('composite_score', 0):.0f}</div>
                <div style="font-size:0.65rem; color:{COLORS['text_muted']};">COMPOSITE</div>
            </div>
        </div>
        <div style="margin-top:8px;">{pills}</div>
        {_score_bars_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_radar_chart(row: pd.Series, title: str = "Quality Radar") -> go.Figure:
    """Create a radar chart for a stock's quality sub-scores."""
    categories = ['Moat', 'Growth', 'Cash Quality', 'Margins', 'Balance Sheet']
    values = [
        row.get("moat_score", 0),
        row.get("growth_score", 0),
        row.get("cash_score", 0),
        row.get("margin_score", 0),
        row.get("balance_sheet_score", 0),
    ]
    values += [values[0]]  # close the polygon
    categories += [categories[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories,
        fill='toself',
        fillcolor='rgba(139,92,246,0.15)',
        line=dict(color=COLORS['purple'], width=2),
        marker=dict(size=6, color=COLORS['purple']),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor=COLORS['bg_secondary'],
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=True,
                          tickfont=dict(size=9, color=COLORS['text_muted']),
                          gridcolor=COLORS['border']),
            angularaxis=dict(tickfont=dict(size=11, color=COLORS['text_primary']),
                           gridcolor=COLORS['border']),
        ),
        showlegend=False,
        title=dict(text=title, font=dict(size=14, color=COLORS['text_primary'])),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50, b=30, l=60, r=60),
        height=350,
    )
    return fig


def render_tier_summary(df: pd.DataFrame):
    """Render conviction tier summary cards."""
    for tier_cfg in CONVICTION_TIERS:
        tier_num = tier_cfg["tier"]
        count = (df["conviction_tier"] == tier_num).sum()
        gate_passed = ((df["conviction_tier"] == tier_num) & (df["gate_pass"] == 1)).sum()
        tc = TIER_COLORS[tier_num]

        st.markdown(f"""
        <div class="tier-card" style="background:{tc['bg']}; border: 1px solid {tc['border']};">
            <div class="tier-header">
                <span class="tier-name" style="color:{tc['text']};">
                    {tier_cfg['emoji']} Tier {tier_num} — {tier_cfg['label']}
                </span>
                <span class="tier-count" style="color:{tc['text']};">{count} stocks</span>
            </div>
            <div style="font-size:0.75rem; color:{COLORS['text_secondary']};">
                {tier_cfg['description']} · {gate_passed} gate-qualified
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_sidebar_brand():
    """Render the sidebar brand card."""
    st.markdown(f"""
    <div class="sb-brand">
        <div class="sb-brand-icon">🏆</div>
        <div class="sb-brand-title">Multibagger<br>Discovery</div>
        <div class="sb-brand-ver">v{UI['version']} · QUANTAMENTAL ENGINE</div>
    </div>
    """, unsafe_allow_html=True)


def render_bruised_blue_chips(df: pd.DataFrame):
    """
    Bruised Blue Chips tracker (29th WCS).
    Large-cap quality compounders (ROCE ≥ 20% 10Y, Market Cap ≥ ₹20,000 Cr)
    trading at a ≥ 20% discount to their 10Y mean P/E — temporary bruising, not structural damage.
    """
    st.markdown("<div class='sec-head'>💙 Bruised Blue Chips (29th WCS)</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sec-cap'>Established quality compounders with ROCE ≥ 20% over 10Y, Market Cap ≥ ₹20,000 Cr, "
        "currently trading ≥ 20% below their 10Y mean P/E. Temporary bruising — not structural damage.</div>",
        unsafe_allow_html=True,
    )

    bbc = df[
        df.get("bruised_blue_chip_29", pd.Series(0, index=df.index)).fillna(0) == 1
    ].sort_values("pe_discount", ascending=False)

    if bbc.empty:
        st.info("💙 No Bruised Blue Chips detected. Quality large-caps are either fairly valued or not yet discounted enough.")
        return

    st.success(f"💙 **{len(bbc)} Bruised Blue Chips** — quality at a discount.")
    _disp_cols = [c for c in ["rank", "name", "sector", "market_cap", "roce_med_10y", "pe_discount",
                               "pb_ratio", "composite_score", "conviction_tier"] if c in bbc.columns]
    st.dataframe(
        bbc[_disp_cols].reset_index(drop=True),
        use_container_width=True,
        height=min(400, 80 + len(bbc) * 35),
        column_config={
            "pe_discount":   st.column_config.ProgressColumn("PE Discount %", min_value=0, max_value=60, format="%.1f"),
            "roce_med_10y":  st.column_config.NumberColumn("ROCE 10Y %", format="%.1f"),
            "market_cap":    st.column_config.NumberColumn("MCap (Cr)", format="₹%.0f"),
            "composite_score": st.column_config.ProgressColumn("Composite", min_value=0, max_value=100, format="%.0f"),
        },
    )


def render_multi_trillion_tipping_points(df: pd.DataFrame):
    """
    Multi-Trillion Macro Tipping Points (30th WCS).
    Sectors with structural tailwinds to become next multi-trillion market cap clusters.
    Filters for momentum-ready stocks in selected sunrise sectors.
    """
    st.markdown("<div class='sec-head'>🚀 Multi-Trillion Tipping Points (30th WCS)</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sec-cap'>Sunrise sectors with structural tailwinds to reach multi-trillion INR market cap. "
        "Shows stocks with institutional momentum, earnings acceleration, and technical readiness.</div>",
        unsafe_allow_html=True,
    )

    _SECTOR_GROUPS = {
        "Financial Services": ["Financial", "Bank", "NBFC", "Insurance", "Fintech"],
        "Consumer Discretionary": ["Consumer", "Retail", "Automobile", "Durables", "Hotel", "Tourism"],
        "Healthcare & Pharma": ["Pharma", "Health", "Hospital", "Diagnostic"],
        "Infrastructure & Capital Goods": ["Infrastructure", "Capital Goods", "Construction", "Defence", "Engineering"],
    }

    selected_group = st.radio(
        "Select Sunrise Sector",
        options=list(_SECTOR_GROUPS.keys()),
        horizontal=True,
        label_visibility="collapsed",
    )
    keywords = _SECTOR_GROUPS[selected_group]
    _sector_pat = "|".join(keywords)

    if "sector" in df.columns:
        _sector_mask = df["sector"].str.contains(_sector_pat, case=False, na=False)
    else:
        st.warning("Sector column not available.")
        return

    _pat_gr   = df["pat_gr_yoy"].fillna(0)  if "pat_gr_yoy"      in df.columns else pd.Series(0,  index=df.index)
    _vstop    = df["vstop_green"].fillna(0)  if "vstop_green"     in df.columns else pd.Series(0,  index=df.index)
    _inst     = df["inst_convergence"].fillna(0) if "inst_convergence" in df.columns else pd.Series(0, index=df.index)
    _brk      = df["breakout_score"].fillna(0)   if "breakout_score"   in df.columns else pd.Series(0, index=df.index)

    # Core trigger: in sector AND (earning acceleration > 15%) AND (technical OR institutional signal)
    _earnings_acc = _pat_gr > 15
    _tech_or_inst = (_vstop == 1) | (_inst == 1) | (_brk >= 70)

    mttp = df[_sector_mask & _earnings_acc & _tech_or_inst].sort_values("composite_score", ascending=False)

    if mttp.empty:
        st.info(f"No {selected_group} stocks currently meet the multi-trillion tipping point criteria (earnings acceleration + technical/institutional signal).")
        return

    render_metric_strip([
        (str(len(mttp)),                                    f"{selected_group} triggers",  "m-purple"),
        (f"{mttp['composite_score'].mean():.0f}",           "Avg Composite",               "m-blue"),
        (str(int((_vstop[mttp.index] == 1).sum())),         "VSTOP Green",                 "m-green"),
        (str(int((_inst[mttp.index] == 1).sum())),          "Inst Convergence",            "m-gold"),
    ])

    _disp = [c for c in ["rank", "name", "sector", "market_cap", "pat_gr_yoy",
                          "vstop_green", "inst_convergence", "breakout_score",
                          "composite_score", "conviction_tier"] if c in mttp.columns]
    st.dataframe(
        mttp[_disp].reset_index(drop=True),
        use_container_width=True,
        height=min(500, 80 + len(mttp) * 35),
        column_config={
            "pat_gr_yoy":     st.column_config.NumberColumn("PAT Gr YoY %", format="%.1f"),
            "composite_score": st.column_config.ProgressColumn("Composite", min_value=0, max_value=100, format="%.0f"),
            "breakout_score": st.column_config.ProgressColumn("Breakout", min_value=0, max_value=100, format="%.0f"),
            "market_cap":     st.column_config.NumberColumn("MCap (Cr)", format="₹%.0f"),
        },
    )
