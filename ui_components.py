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
    section[data-testid="stMain"] > div.block-container {{
        max-width: 100%; overflow-x: hidden; box-sizing: border-box;
        padding-top: 1rem;
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
        background: linear-gradient(135deg, #12095C 0%, #3C3489 100%);
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
    """Render a horizontal score bar."""
    html = f"""
    <div style="display:flex; align-items:center; gap:8px; margin:2px 0;">
        <span style="font-size:0.7rem; color:{COLORS['text_secondary']}; min-width:50px;">{label}</span>
        <div class="score-bar-wrap" style="flex:1;">
            <div class="score-bar" style="width:{score}%; background:{color};"></div>
        </div>
        <span style="font-size:0.75rem; font-weight:700; color:{color}; min-width:30px;">{score:.0f}</span>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_stock_card(row: pd.Series, show_scores: bool = True):
    """Render a premium stock card."""
    tier = int(row.get("conviction_tier", 5))
    tc = TIER_COLORS.get(tier, TIER_COLORS[5])

    gate_status = "✅ All gates passed" if row.get("gate_pass", 0) == 1 else f"❌ {int(row.get('gates_failed', 0))} gates failed"

    pills = ""
    if row.get("promoter_buying", 0) == 1:
        pills += '<span class="pill pill-green">Promoter Buying</span>'
    if row.get("inst_convergence", 0) == 1:
        pills += '<span class="pill pill-blue">FII+DII Convergence</span>'
    if row.get("vstop_green", 0) == 1:
        pills += '<span class="pill pill-purple">VSTOP Green</span>'
    if row.get("tsunami_signal", 0) == 1:
        pills += '<span class="pill pill-gold">🌊 Tsunami</span>'
    if row.get("net_debt_negative", 0) == 1:
        pills += '<span class="pill pill-green">Net Cash</span>'

    card_html = f"""
    <div class="stock-card" style="border-left: 3px solid {tc['border']};">
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div>
                <div style="font-weight:800; font-size:1.05rem; color:{COLORS['text_primary']};">
                    {row.get('tier_emoji', '')} #{int(row.get('rank', 0))} · {row.get('name', 'N/A')}
                </div>
                <div style="font-size:0.75rem; color:{COLORS['text_secondary']}; margin-top:2px;">
                    {row.get('sector', '')} · {row.get('industry', '')} · ₹{row.get('market_cap', 0):,.0f} Cr · {row.get('market_category', '')}
                </div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:1.8rem; font-weight:900; color:{tc['text']};">{row.get('composite_score', 0):.0f}</div>
                <div style="font-size:0.65rem; color:{COLORS['text_muted']};">COMPOSITE</div>
            </div>
        </div>
        <div style="margin-top:8px;">{pills}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

    if show_scores:
        cols = st.columns(5)
        scores = [
            ("Moat", row.get("moat_score", 0), COLORS["purple"]),
            ("Growth", row.get("growth_score", 0), COLORS["green"]),
            ("Cash", row.get("cash_score", 0), COLORS["blue"]),
            ("Momentum", row.get("momentum_score", 0), COLORS["orange"]),
            ("Governance", row.get("governance_bonus", 0), COLORS["gold"]),
        ]
        for col, (label, score, color) in zip(cols, scores):
            with col:
                render_score_bar(score, color, label)


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
