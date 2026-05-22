"""
The Systematic Architect's Index v2.0
======================================
Adaptive Quantamental Engine — Regime-Aware, Master-Driven
Dr. Malik + Raamdeo Agrawal + O'Neil + Mukherjea + Marks + Fisher + Lynch
"""
import os
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

import streamlit as st
st.set_page_config(page_title="Systematic Architect's Index", page_icon="🏛️", layout="wide", initial_sidebar_state="expanded")

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core import fetch_and_clean_data, run_full_scoring, run_forensic_analysis
from ui import (render_scanner_grid, render_moat_growth_matrix, render_fisher_module,
                render_ep_power_curve_module, render_bruised_blue_chip_badge,
                render_multitrillioncap_card, render_forensic_perimeter, render_guru_frameworks,
                render_financial_insights, render_stock_hero, render_score_strip,
                render_sell_alerts_panel, render_raw_signals,
                inject_css, render_hero_banner, render_metric_strip, render_stock_card,
                render_radar_chart, render_score_bar, render_sidebar_brand,
                render_bruised_blue_chips, render_multi_trillion_tipping_points)
from config import (COLORS, TIER_COLORS, CONVICTION_TIERS, UI, HARD_GATES,
                    QUALITY_WEIGHTS, MOMENTUM_WEIGHTS, COMPOSITE_WEIGHTS,
                    VALUATION_SIGNALS, MARKS_CYCLE, DEFAULT_CYCLE_TEMPERATURE,
                    BAID_SELL_TRIGGERS, MEAN_REVERSION, PEG_ZONES,
                    MASTER_PROFILES, ANALYSIS_MODES)


# ═══════════════════════════════════════════════════════════════
# 3-TIER CACHE SPLIT
# Tier 1: fetch_and_clean_data — CACHED. Only reruns on Clear Cache or new sheet.
# Tier 2: run_full_scoring     — NOT cached. Instant on dropdown change.
# Tier 3: run_forensic_analysis— NOT cached. Instant.
# ═══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def get_clean_data(data_source, _file_signature: str, sheet_id, _uploaded_dict=None):
    """Tier-1: Expensive data fetch + clean. Heavily cached.
    _uploaded_dict is prefixed with _ so Streamlit skips hashing the raw stream objects.
    _file_signature (stable string: name+size per file) is the actual cache key for uploads.
    """
    t0 = time.time()
    df = fetch_and_clean_data(data_source, _uploaded_dict, sheet_id)
    elapsed = time.time() - t0
    return df, elapsed

def get_scored_data(clean_df, analysis_mode, scoring_profile):
    """Tier-2+3: Instant scoring. NOT cached — runs in <0.5s on dropdowns change."""
    df = run_full_scoring(clean_df, analysis_mode, scoring_profile)
    df = run_forensic_analysis(df)
    return df

inject_css()

# Data Source UI
if "data_source" not in st.session_state:
    st.session_state.data_source = "sheet"

with st.sidebar:
    render_sidebar_brand()

    st.markdown("### 📂 Data Source")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Google Sheets", type="primary" if st.session_state.data_source == "sheet" else "secondary", use_container_width=True):
            st.session_state.data_source = "sheet"
            st.rerun()
    with col2:
        if st.button("📁 Upload CSV", type="primary" if st.session_state.data_source == "upload" else "secondary", use_container_width=True):
            st.session_state.data_source = "upload"
            st.rerun()

    if st.button("🔄 Clear Cache & Reload", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    sheet_id = None
    uploaded_dict = None
    data_ready = False

    if st.session_state.data_source == "sheet":
        sheet_id = st.text_input("Google Sheets URL or ID", placeholder="Enter Google Sheet ID...")
        if sheet_id:
            data_ready = True
    elif st.session_state.data_source == "upload":
        uploaded_files = st.file_uploader("Upload CSV files (Ratio, Income, Balance, Cashflow, Shareholding, Tech)", type="csv", accept_multiple_files=True)
        if uploaded_files and len(uploaded_files) > 0:
            uploaded_dict = {}
            for f in uploaded_files:
                name = f.name.lower()
                if "ratio" in name: uploaded_dict["ratio"] = f
                elif "income" in name: uploaded_dict["income"] = f
                elif "balance" in name: uploaded_dict["balance"] = f
                elif "cashflow" in name: uploaded_dict["cashflow"] = f
                elif "shareholding" in name: uploaded_dict["shareholding"] = f
                elif "technical" in name: uploaded_dict["technical"] = f
            if len(uploaded_dict) >= 1:
                data_ready = True

    # ══ Sidebar Data Source Ends Here ══
    # (Analysis Mode and Scoring Profile moved to Main Command Center)

if not data_ready:
    st.info("👋 Welcome! Please select a data source from the sidebar (Google Sheets or Upload CSV) to begin scanning.")
    st.stop()

with st.spinner("🔄 Loading data..."):
    try:
        if uploaded_dict:
            file_sig = "|".join(
                f"{k}:{v.name}:{v.size}"
                for k, v in uploaded_dict.items()
                if v is not None
            )
        else:
            file_sig = f"local_{sheet_id or 'default'}"
        clean_df, load_time = get_clean_data(
            st.session_state.data_source, file_sig, sheet_id, _uploaded_dict=uploaded_dict
        )
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.stop()

# ═══════════════════════════════════════════════════════════════
# 🏛️ THE COMMAND CENTER — Mandate-Driven Investment Philosophy
# ═══════════════════════════════════════════════════════════════
_MANDATES = {
    "QGLP Balanced": {
        "icon": "🎯", "mode": "Hybrid", "profile": "Balanced",
        "desc": "Raamdeo's all-weather formula — Quality · Growth · Longevity · Price in harmony",
    },
    "Coffee Can": {
        "icon": "🛡️", "mode": "Fundamental", "profile": "Quality",
        "desc": "Buffett / Mukherjea: ROCE 20%+ sustained, zero debt stress — buy and hold forever",
    },
    "Lynch GARP": {
        "icon": "📈", "mode": "Hybrid", "profile": "GARP",
        "desc": "Peter Lynch: PEG ≤ 1.0 mandatory — earnings growth at a price no one else will pay",
    },
    "Deep Value": {
        "icon": "💰", "mode": "Hybrid", "profile": "Value",
        "desc": "Howard Marks / Vijay Kedia: beaten-down quality at maximum margin of safety",
    },
    "Breakout": {
        "icon": "⚡", "mode": "Technical", "profile": "Momentum",
        "desc": "O'Neil CAN-SLIM: institutional accumulation into Stage 2 breakouts — follow smart money",
    },
    "Turnaround": {
        "icon": "🔄", "mode": "Technical", "profile": "Turnaround",
        "desc": "QoQ earnings revival + promoter buying + volume surge — asymmetric risk/reward",
    },
}
_MANDATE_KEYS = list(_MANDATES.keys())

# ── Mandate Selector — button row ─────────────────────────────
if "sel_mandate" not in st.session_state:
    st.session_state["sel_mandate"] = _MANDATE_KEYS[0]

_sel_mandate = st.session_state["sel_mandate"]
_mb_cols = st.columns(len(_MANDATES))
for _mi, (_mk, _mv) in enumerate(_MANDATES.items()):
    with _mb_cols[_mi]:
        _is_active = (_sel_mandate == _mk)
        if st.button(
            f"{_mv['icon']} {_mk}",
            key=f"_mb_{_mk}",
            type="primary" if _is_active else "secondary",
            use_container_width=True,
        ):
            st.session_state["sel_mandate"] = _mk
            st.session_state["adv_mode"]    = _mv["mode"]
            st.session_state["adv_profile"] = _mv["profile"]
            st.rerun()

_sel_mandate = st.session_state["sel_mandate"]

# Mandate description strip
st.markdown(
    f'<div style="font-size:0.75rem;color:{COLORS["text_secondary"]};'
    f'padding:4px 2px 10px 2px;border-bottom:1px solid {COLORS["border"]};margin-bottom:6px;">'
    f'{_MANDATES[_sel_mandate]["desc"]}</div>',
    unsafe_allow_html=True,
)

# ── Advanced Override (collapsed — power users only) ───────────
with st.expander("⚙️ Advanced: Override Mandate Defaults", expanded=False):
    analysis_mode = st.selectbox(
        "Analysis Mode",
        options=list(ANALYSIS_MODES.keys()),
        format_func=lambda k: ANALYSIS_MODES[k]["label"],
        key="adv_mode",
    )
    st.caption(ANALYSIS_MODES[analysis_mode]["description"])

    _ov_allowed = ANALYSIS_MODES[analysis_mode]["allowed_profiles"]
    if st.session_state.get("adv_profile") not in _ov_allowed:
        st.session_state["adv_profile"] = _ov_allowed[0]
    scoring_profile = st.selectbox(
        "Scoring Profile",
        options=_ov_allowed,
        format_func=lambda k: f"{MASTER_PROFILES[k]['icon']} {MASTER_PROFILES[k]['label']}",
        key="adv_profile",
    )
    st.caption(MASTER_PROFILES[scoring_profile]["description"])

# Final values — expander code always executes even when collapsed
analysis_mode   = st.session_state.get("adv_mode",    _MANDATES[_sel_mandate]["mode"])
scoring_profile = st.session_state.get("adv_profile", _MANDATES[_sel_mandate]["profile"])
profile_cfg     = MASTER_PROFILES[scoring_profile]

# ── Scoring ────────────────────────────────────────────────────
_score_key = f"{file_sig}::{analysis_mode}::{scoring_profile}"
if st.session_state.get("_score_key") != _score_key or "_scored_df" not in st.session_state:
    _spin_icon = _MANDATES.get(_sel_mandate, {}).get("icon", "🧭")
    with st.spinner(f"{_spin_icon} Running {_sel_mandate} mandate — {scoring_profile}..."):
        try:
            _df_scored = get_scored_data(clean_df, analysis_mode, scoring_profile)
            st.session_state["_scored_df"] = _df_scored
            st.session_state["_score_key"] = _score_key
        except Exception as e:
            st.error(f"❌ Scoring error: {e}")
            st.stop()
df = st.session_state["_scored_df"]

if df is None or df.empty:
    st.warning("⚠️ No data returned after scoring. Check your data source or filters.")
    st.stop()

adaptive_w = df.attrs.get("adaptive_weights", {})
# Key metrics
total = len(df)
gate_passed = int(df["gate_pass"].sum())
tier1 = int((df["conviction_tier"] == 1).sum())
tier2 = int((df["conviction_tier"] == 2).sum())
tsunami_count = int(df["tsunami_signal"].sum())
avg_quality = df["quality_score"].mean()
qualified = df[df["gate_pass"] == 1]


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="background:{COLORS['bg_secondary']}; border:1px solid {COLORS['border']};
                border-radius:12px; padding:12px 14px; margin:10px 0;">
        <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:{COLORS['text_primary']}; padding:3px 0;">
            <span>📊 Universe</span><span style="font-weight:700;">{total}</span>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:{COLORS['green']}; padding:3px 0;">
            <span>✅ Gate Passed</span><span style="font-weight:700;">{gate_passed}</span>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:{COLORS['gold']}; padding:3px 0;">
            <span>🏆 Crown Jewels</span><span style="font-weight:700;">{tier1}</span>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:{COLORS['purple']}; padding:3px 0;">
            <span>🌊 Tsunami</span><span style="font-weight:700;">{tsunami_count}</span>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:{COLORS['text_muted']}; padding:3px 0;">
            <span>⏱️ Load Time</span><span style="font-weight:700;">{load_time:.1f}s</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    regime = df.attrs.get("detected_market_regime", "SIDEWAYS")
    regime_color = COLORS['green'] if regime == "BULL" else COLORS['red'] if regime == "BEAR" else COLORS['gold']
    st.markdown(f"""
    <div style="background:{COLORS['bg_tertiary']}; border-left:4px solid {regime_color}; padding:8px 12px; margin-bottom:15px; border-radius:4px;">
        <div style="font-size:0.75rem; color:{COLORS['text_muted']}; text-transform:uppercase; letter-spacing:1px;">Detected Regime</div>
        <div style="font-size:1.1rem; font-weight:800; color:{regime_color};">{regime} MARKET</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div class='sec-head'>🎯 Filters</div>", unsafe_allow_html=True)
    sectors = ["All"] + sorted(df["sector"].dropna().unique().tolist())
    sel_sector = st.selectbox("Sector", sectors, key="sb_sector")
    sel_tier = st.multiselect("Conviction Tier", [1,2,3,4,5], default=[1,2,3], key="sb_tier")
    sel_mcap = st.multiselect("Market Category", ["Mega Cap", "Large Cap", "Mid Cap", "Small Cap", "Micro Cap", "Nano Cap"], default=["Mega Cap", "Large Cap", "Mid Cap", "Small Cap", "Micro Cap", "Nano Cap"], key="sb_mcap")
    
    # The All-Time Best Filter: Moat-Growth Matrix
    moat_options = ["⭐ Wealth Creator", "🛡️ Quality Trap", "⚡ Growth Trap", "💀 Wealth Destroyer"]
    sel_moat = st.multiselect("Moat", moat_options, default=["⭐ Wealth Creator"], key="sb_moat")
    
    # Institutional Sweep Vector
    st.markdown("---")
    st.markdown("<div style='font-size:0.8rem; font-weight:700; color:#8b5cf6; margin-bottom:5px;'>🌊 ALPHA VECTORS</div>", unsafe_allow_html=True)
    smart_sweep = st.checkbox("🎯 Smart Money Sweep (FII+DII + Breakout)", value=False, key="sb_sweep")
    
    gate_only = st.checkbox("Gate-passed only", value=True, key="sb_gate")
    min_quality = st.slider("Min Quality Score", 0, 100, 0, key="sb_minq")

# Apply filters
filt = df.copy()
if sel_sector != "All":
    filt = filt[filt["sector"] == sel_sector]
if sel_tier:
    filt = filt[filt["conviction_tier"].isin(sel_tier)]
if sel_mcap:
    filt = filt[filt["market_category"].isin(sel_mcap)]
if sel_moat:
    filt = filt[filt["moat_growth_quad"].isin(sel_moat)]
if smart_sweep:
    # Requires simultaneous FII + DII buying AND a Tsunami signal
    filt = filt[(filt["inst_convergence"] == 1) & (filt["tsunami_signal"] == 1)]
if gate_only:
    filt = filt[filt["gate_pass"] == 1]
if min_quality > 0:
    filt = filt[filt["quality_score"] >= min_quality]


# ═══════════════════════════════════════════════════════════════
# BANNER (above tabs — always visible)
# ═══════════════════════════════════════════════════════════════
render_hero_banner(total, gate_passed, tier1)
render_metric_strip([
    (f"{total}", "Universe", "m-blue"),
    (f"{gate_passed}", "Gate Passed", "m-green"),
    (f"{tier1}", "Crown Jewels", "m-gold"),
    (f"{tier2}", "Strong", "m-green"),
    (f"{tsunami_count}", "Tsunami", "m-purple"),
    (f"{avg_quality:.0f}", "Avg Quality", "m-blue"),
])

# ── Live Engine Weights Strip (always visible) ────────────────
if adaptive_w:
    _qw         = adaptive_w.get("quality_w", 0)
    _gw         = adaptive_w.get("growth_w", 0)
    _lw         = adaptive_w.get("longevity_w", 0)
    _pw         = adaptive_w.get("price_w", 0)
    _det_regime = df.attrs.get("detected_market_regime", "SIDEWAYS")
    _reg_clr    = COLORS["green"] if _det_regime == "BULL" else COLORS["red"] if _det_regime == "BEAR" else COLORS["gold"]
    _reg_emoji  = "🟢" if _det_regime == "BULL" else "🔴" if _det_regime == "BEAR" else "🟡"
    _m_icon     = _MANDATES.get(_sel_mandate, {}).get("icon", "⚖️")
    _prof_icon  = profile_cfg.get("icon", "⚖️")
    _wbars = [
        ("⚡ Quality",     _qw, COLORS["purple"]),
        ("🌱 Growth",      _gw, COLORS["green"]),
        ("🏛️ Longevity",  _lw, COLORS["blue"]),
        ("💰 Price",       _pw, COLORS["gold"]),
    ]
    _bars_html = "".join(
        f'<div style="flex:1;min-width:55px;">'
        f'<div style="font-size:0.57rem;color:{COLORS["text_muted"]};text-transform:uppercase;'
        f'letter-spacing:0.6px;margin-bottom:4px;font-weight:600;">{lbl}</div>'
        f'<div style="background:{COLORS["bg_tertiary"]};border-radius:4px;height:5px;overflow:hidden;">'
        f'<div style="width:{pct*100:.0f}%;height:5px;border-radius:4px;background:{clr};"></div>'
        f'</div>'
        f'<div style="font-size:0.7rem;font-weight:700;color:{clr};margin-top:3px;">{pct:.0%}</div>'
        f'</div>'
        for lbl, pct, clr in _wbars
    )
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{COLORS['bg_secondary']},{COLORS['bg_tertiary']});
         border:1px solid {COLORS['border']};border-radius:10px;padding:12px 18px;margin:8px 0 16px 0;">
      <div style="display:flex;align-items:center;justify-content:space-between;
           margin-bottom:10px;flex-wrap:wrap;gap:6px;">
        <span style="font-size:0.8rem;font-weight:700;color:{COLORS['text_primary']};">
          {_m_icon} {_sel_mandate}
          <span style="color:{COLORS['text_muted']};font-weight:400;"> · </span>
          <span style="color:{COLORS['text_secondary']};font-weight:400;font-size:0.74rem;">
            {_prof_icon} {scoring_profile}
          </span>
        </span>
        <span style="font-size:0.71rem;font-weight:700;padding:2px 10px;border-radius:20px;
             background:{_reg_clr}18;color:{_reg_clr};border:1px solid {_reg_clr}50;">
          {_reg_emoji} {_det_regime}
        </span>
      </div>
      <div style="display:flex;gap:12px;margin-bottom:8px;">{_bars_html}</div>
      <div style="font-size:0.62rem;color:{COLORS['text_muted']};">
        Hard Gates — ROCE≥{adaptive_w.get('roce_gate', 15):.0f}% ·
        Growth≥{adaptive_w.get('growth_gate', 15):.0f}% ·
        PEG≤{adaptive_w.get('peg_gate', 1.5):.1f}
      </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════
tabs = st.tabs(["🏠 Discovery", "🔍 Deep Scanner", "🔬 The Tear-Sheet", "🌊 Market Pulse", "⚙️ Config"])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1: DISCOVERY DASHBOARD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[0]:

    # ── Compact tier strip (replaces 5 stacked tier cards) ────────
    _tier_strip_html = ""
    for _tc in CONVICTION_TIERS:
        _tn   = _tc["tier"]
        _fcnt = int((filt["conviction_tier"] == _tn).sum())
        _acnt = int((df["conviction_tier"] == _tn).sum())
        if _acnt == 0:
            continue
        _ts = TIER_COLORS.get(_tn, TIER_COLORS[5])
        _tier_strip_html += (
            f'<div style="flex:1;min-width:90px;background:{_ts["bg"]};border:1px solid {_ts["border"]};'
            f'border-radius:10px;padding:11px 8px;text-align:center;">'
            f'<div style="font-size:1.5rem;font-weight:900;color:{_ts["text"]};line-height:1;">{_fcnt}</div>'
            f'<div style="font-size:0.67rem;font-weight:700;color:{_ts["text"]};margin-top:3px;'
            f'text-transform:uppercase;letter-spacing:0.4px;">{_tc["emoji"]} {_tc["label"]}</div>'
            f'<div style="font-size:0.57rem;color:{COLORS["text_muted"]};margin-top:2px;">of {_acnt} total</div>'
            f'</div>'
        )
    st.markdown(
        f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px;">{_tier_strip_html}</div>',
        unsafe_allow_html=True,
    )

    # ── Controls: sort + count ─────────────────────────────────────
    _dc1, _dc2 = st.columns([6, 2])
    with _dc1:
        _disc_sort = st.pills(
            "Sort by",
            ["🏆 Score", "📊 Quality", "📈 Momentum", "💰 PEG"],
            default="🏆 Score",
            key="disc_sort",
        )
    with _dc2:
        _disc_n = st.selectbox(
            "Show", [10, 20, 30, 50], index=1, key="disc_n",
        )

    # Sort the filtered data
    _sort_map = {
        "🏆 Score":    ("composite_score", False),
        "📊 Quality":  ("quality_score",   False),
        "📈 Momentum": ("momentum_score",  False),
        "💰 PEG":      ("peg",             True),
    }
    if not _disc_sort:
        _disc_sort = "🏆 Score"
    _sc, _sa = _sort_map.get(_disc_sort, ("composite_score", False))
    _disc_df  = filt.sort_values(_sc, ascending=_sa) if _sc in filt.columns else filt.copy()
    _shown_n  = int(_disc_n or 20)

    st.markdown(
        f'<div class="sec-head">🏆 Top Picks — {len(_disc_df)} stocks'
        f'{"" if _disc_sort == "🏆 Score" else f" &nbsp;·&nbsp; sorted by {_disc_sort}"}</div>',
        unsafe_allow_html=True,
    )

    # ── Stock cards with tearsheet shortcut ────────────────────────
    for _di, (_, _drow) in enumerate(_disc_df.head(_shown_n).iterrows()):
        render_stock_card(_drow, show_scores=True)
        _, _btn_c = st.columns([8, 2])
        with _btn_c:
            if st.button(
                "🔬 Open Analysis →",
                key=f"disc_ts_{_di}",
                use_container_width=True,
                type="secondary",
                help=f"View full tearsheet for {_drow.get('name', '')}",
            ):
                st.session_state["xray_stock"] = _drow.get("name", "")
                st.toast(f"🔬 {_drow.get('name', '')} ready — click The Tear-Sheet tab")

    if len(_disc_df) > _shown_n:
        st.markdown(
            f'<div style="text-align:center;padding:12px 0 4px;font-size:0.73rem;'
            f'color:{COLORS["text_muted"]};">'
            f'{len(_disc_df) - _shown_n} more stocks — increase "Show" above</div>',
            unsafe_allow_html=True,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2: DEEP SCANNER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[1]:

    # ── Column view presets ────────────────────────────────────────
    _DS_VIEWS = {
        "🏆 Core":      ["rank","name","sector","market_category","composite_score",
                         "conviction_tier","gate_pass","moat_growth_quad","smart_money_flow"],
        "📊 Quality":   ["name","quality_score","moat_score","growth_score","cash_score",
                         "governance_bonus","piotroski_fscore","roce","opm","cfo_to_pat"],
        "💰 Valuation": ["name","composite_score","pe_ratio","pb_ratio","peg",
                         "earnings_yield","fcf_yield","market_cap","buy_zone_label"],
        "🔬 Forensic":  ["name","piotroski_fscore","forensic_score","cfo_to_pat",
                         "debt_to_equity","promoter_holdings","pledged_percentage"],
        "📈 Technical": ["name","momentum_score","rsi","from_high_pct",
                         "breakout_score","smart_money_flow","tsunami_signal","vstop_green"],
    }
    _DS_SORTS = {
        "Score ↓":    ("composite_score", False),
        "Quality ↓":  ("quality_score",   False),
        "Momentum ↓": ("momentum_score",  False),
        "PEG ↑":      ("peg",             True),
        "MCap ↓":     ("market_cap",      False),
    }

    # ── Control bar ────────────────────────────────────────────────
    st.markdown(
        f'<div style="font-size:0.7rem;font-weight:700;color:{COLORS["text_muted"]};'
        f'text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">'
        f'🔍 Deep Scanner &nbsp;·&nbsp; {profile_cfg.get("icon","⚖️")} {scoring_profile}</div>',
        unsafe_allow_html=True,
    )
    _ds_c1, _ds_c2, _ds_c3 = st.columns([2, 5, 2])
    with _ds_c1:
        ds_search = st.text_input(
            "Search", placeholder="Search stock name…",
            key="ds_search", label_visibility="collapsed",
        )
    with _ds_c2:
        ds_view = st.pills(
            "Column View", list(_DS_VIEWS.keys()),
            default="🏆 Core", key="ds_view",
        )
        if not ds_view:
            ds_view = "🏆 Core"
    with _ds_c3:
        ds_sort_label = st.selectbox(
            "Sort", list(_DS_SORTS.keys()),
            key="ds_sort", label_visibility="collapsed",
        )

    # ── Filter + sort ──────────────────────────────────────────────
    ds_df = filt.copy()
    if ds_search and ds_search.strip():
        ds_df = ds_df[ds_df["name"].str.contains(ds_search.strip(), case=False, na=False)]
    _sort_col, _sort_asc = _DS_SORTS[ds_sort_label]
    if _sort_col in ds_df.columns:
        ds_df = ds_df.sort_values(_sort_col, ascending=_sort_asc)

    # ── Stats strip ────────────────────────────────────────────────
    _ds_t1   = int((ds_df["conviction_tier"] == 1).sum()) if "conviction_tier" in ds_df.columns else 0
    _ds_tsun = int(ds_df["tsunami_signal"].sum()) if "tsunami_signal" in ds_df.columns else 0
    _ds_avg  = ds_df["composite_score"].mean() if "composite_score" in ds_df.columns and len(ds_df) else 0
    _ds_gate = int(ds_df["gate_pass"].sum()) if "gate_pass" in ds_df.columns else len(ds_df)
    st.markdown(f"""
    <div style="display:flex;gap:20px;padding:8px 2px 12px 2px;
         border-bottom:1px solid {COLORS['border']};margin-bottom:10px;flex-wrap:wrap;
         align-items:center;">
      <span style="font-size:0.82rem;font-weight:800;color:{COLORS['text_primary']};">
        {len(ds_df)} stocks
      </span>
      <span style="font-size:0.78rem;color:{COLORS['text_muted']};">
        Avg&nbsp;<strong style="color:{COLORS['blue']};font-size:0.86rem;">{_ds_avg:.0f}</strong>
      </span>
      <span style="font-size:0.78rem;color:{COLORS['green']};">
        ✅ {_ds_gate} gate&nbsp;passed
      </span>
      <span style="font-size:0.78rem;color:{COLORS['gold']};">
        🏆 {_ds_t1} Crown&nbsp;Jewels
      </span>
      <span style="font-size:0.78rem;color:{COLORS['purple']};">
        🌊 {_ds_tsun} Tsunami
      </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Column selection ───────────────────────────────────────────
    _view_cols = [c for c in _DS_VIEWS.get(ds_view, []) if c in ds_df.columns]
    if not _view_cols:
        _view_cols = [c for c in ["rank", "name", "composite_score"] if c in ds_df.columns]
    _display_df = ds_df[_view_cols].reset_index(drop=True)

    # ── Column config ──────────────────────────────────────────────
    _CC: dict = {}
    for _sc, _sl in {
        "composite_score": "Score", "quality_score": "Quality",
        "moat_score": "Moat", "growth_score": "Growth",
        "cash_score": "Cash", "momentum_score": "Momentum",
        "forensic_score": "Forensic", "governance_bonus": "Governance",
        "breakout_score": "Breakout",
    }.items():
        if _sc in _display_df.columns:
            _CC[_sc] = st.column_config.ProgressColumn(_sl, min_value=0, max_value=100, format="%.0f")
    for _bc in ("gate_pass", "tsunami_signal", "vstop_green"):
        if _bc in _display_df.columns:
            _lbl = {"gate_pass": "✅ Gate", "tsunami_signal": "🌊", "vstop_green": "VSTOP"}[_bc]
            _CC[_bc] = st.column_config.CheckboxColumn(_lbl)
    _num_fmt = {
        "conviction_tier": ("Tier",     "T%.0f"),
        "piotroski_fscore":("F-Score",  "%.0f/9"),
        "peg":             ("PEG",      "%.2f×"),
        "pe_ratio":        ("P/E",      "%.1f×"),
        "pb_ratio":        ("P/B",      "%.1f×"),
        "cfo_to_pat":      ("CFO/PAT",  "%.0f%%"),
        "opm":             ("OPM",      "%.1f%%"),
        "roce":            ("ROCE",     "%.1f%%"),
        "debt_to_equity":  ("D/E",      "%.2f"),
        "promoter_holdings":("Promoter","%.1f%%"),
        "pledged_percentage":("Pledged","%.1f%%"),
        "rsi":             ("RSI",      "%.0f"),
        "from_high_pct":   ("52WH Δ",  "%.1f%%"),
        "earnings_yield":  ("E.Yield",  "%.1f%%"),
        "fcf_yield":       ("FCF Yld",  "%.1f%%"),
        "market_cap":      ("MCap ₹Cr", "%.0f"),
        "rank":            ("Rank",     "%.0f"),
    }
    for _nc, (_nl, _nf) in _num_fmt.items():
        if _nc in _display_df.columns:
            _CC[_nc] = st.column_config.NumberColumn(_nl, format=_nf)

    # ── Render table ───────────────────────────────────────────────
    _sel = st.dataframe(
        _display_df,
        column_config=_CC,
        use_container_width=True,
        height=580,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    # ── Tearsheet shortcut on row select ──────────────────────────
    _sel_rows = _sel.selection.rows if _sel and hasattr(_sel, "selection") else []
    if _sel_rows:
        _picked = ds_df.iloc[_sel_rows[0]]["name"]
        st.session_state["xray_stock"] = _picked
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;padding:10px 14px;
             background:rgba(88,166,255,0.06);border:1px solid rgba(88,166,255,0.25);
             border-radius:8px;margin-top:8px;">
          <span style="font-size:1rem;">🔬</span>
          <span style="font-size:0.8rem;color:{COLORS['text_secondary']};">
            <strong style="color:{COLORS['text_primary']};">{_picked}</strong>
            set as active stock —
            <strong style="color:{COLORS['blue']};">click The Tear-Sheet tab</strong> to view full analysis.
          </span>
        </div>
        """, unsafe_allow_html=True)

    # ── Export ─────────────────────────────────────────────────────
    st.markdown("<div style='margin-top:12px;'>", unsafe_allow_html=True)
    _safe_mandate = _sel_mandate.replace(" ", "_").lower()
    st.download_button(
        f"📥 Export {len(ds_df)} stocks — {_sel_mandate} / {scoring_profile}",
        data=filt.to_csv(index=False),
        file_name=f"scan_{_safe_mandate}_{scoring_profile.lower()}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3: THE TEAR-SHEET
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[2]:
    all_stock_names = df["name"].dropna().tolist()
    if not all_stock_names:
        st.info("No stocks available. Check your data source.")
    else:
        # ── Stock search bar ──────────────────────────────────────────────
        c_search, c_sel = st.columns([1, 3])
        with c_search:
            search_ticker = st.text_input(
                "🔍 Search", placeholder="HDFC, Infosys, TATA…",
                key="search_ticker", label_visibility="collapsed",
            )
        _term  = (search_ticker or "").strip().upper()
        _names = [n for n in all_stock_names if _term in n.upper()] if _term else all_stock_names
        if not _names:
            _names = all_stock_names
        _prev = st.session_state.get("xray_stock")
        _idx  = _names.index(_prev) if _prev in _names else 0
        with c_sel:
            selected = st.selectbox(
                "Stock",
                _names, index=_idx, key="xray_stock",
                label_visibility="collapsed",
            )

        stock = df[df["name"] == selected].iloc[0]
        _regime = df.attrs.get("detected_market_regime", "SIDEWAYS")

        # ── Hard-gate rejection banner (above everything) ─────────────────
        if stock.get("gate_pass", 0) == 0:
            st.markdown(f"""
            <div style="background:rgba(248,81,73,0.08);border:1px solid rgba(248,81,73,0.5);
                        border-radius:10px;padding:14px 18px;margin:6px 0 12px 0;">
              <div style="font-size:0.88rem;font-weight:800;color:{COLORS['red']};margin-bottom:4px;">
                ❌ SYSTEM REJECTED — Hard Gate Failure
              </div>
              <div style="font-size:0.78rem;color:{COLORS['text_secondary']};">
                <strong>Reason(s):</strong> {stock.get('failed_gates', 'Unknown')}
              </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Sell alerts (prominent, above hero) ───────────────────────────
        render_sell_alerts_panel(stock)

        # ── Mean reversion warning ────────────────────────────────────────
        if stock.get('mean_reversion_risk', 0) == 1:
            st.markdown(f"""
            <div style="background:rgba(228,179,65,0.07);border:1px solid rgba(228,179,65,0.4);
                        border-radius:8px;padding:10px 14px;margin-bottom:10px;font-size:0.78rem;">
              ⚠️ <strong style="color:{COLORS['gold']};">Marks Mean Reversion Risk:</strong>
              <span style="color:{COLORS['text_secondary']};">
                Current margins significantly above 5Y median — cyclical peak risk detected.
                Quality score penalized by 15%.
              </span>
            </div>
            """, unsafe_allow_html=True)

        # ── Hero card ─────────────────────────────────────────────────────
        render_stock_hero(stock, regime=_regime)

        # ── Score strip ───────────────────────────────────────────────────
        render_score_strip(stock)

        # ── Inner tabs ────────────────────────────────────────────────────
        _itabs = st.tabs([
            "📋 Overview",
            "🔬 Forensics & Accounting",
            "🏛️ Frameworks",
            "📈 Matrix & WCS",
            "📊 All Data",
        ])

        # ── Tab A: Overview ───────────────────────────────────────────────
        with _itabs[0]:
            # Null-safe getter scoped to this stock row
            def _sg(k, d=0):
                v = stock.get(k, d)
                return d if (v is None or (isinstance(v, float) and np.isnan(v))) else v

            tier_num_ov = int(_sg("conviction_tier", 5))
            tc_ov       = TIER_COLORS.get(tier_num_ov, TIER_COLORS[5])
            tier_cfg_ov = next((t for t in CONVICTION_TIERS if t["tier"] == tier_num_ov),
                               CONVICTION_TIERS[-1])
            comp_ov     = float(_sg("composite_score", 0))

            col1, col2 = st.columns([1, 1])

            with col1:
                fig = render_radar_chart(stock, f"{selected} — Quality Radar")
                st.plotly_chart(fig, use_container_width=True)

                # ── Elevated signal badges (replaces tiny caption text) ────
                pio_raw = stock.get("piotroski_fscore", None)
                pio_val = None
                if pio_raw is not None and not (isinstance(pio_raw, float) and np.isnan(pio_raw)):
                    try:
                        pio_val = int(float(pio_raw))
                    except Exception:
                        pio_val = None
                pio_str = f"{pio_val}/9" if pio_val is not None else "N/A"
                pio_lbl = str(stock.get("piotroski_label", "") or "")
                pio_clr = (COLORS["green"] if pio_val is not None and pio_val >= 7 else
                           COLORS["gold"]  if pio_val is not None and pio_val >= 5 else
                           COLORS["text_muted"] if pio_val is None else
                           COLORS["red"])
                smart  = str(stock.get("smart_money_flow", "⚪ Neutral") or "⚪ Neutral")
                cf_tri = str(stock.get("cf_triangle", "") or "")
                badge_items = [(f"F-Score {pio_str}", pio_clr),
                               (smart, COLORS["purple"])]
                if cf_tri:
                    badge_items.append((cf_tri, COLORS["blue"]))
                bdgs = "".join(
                    f'<span style="display:inline-block;padding:3px 9px;border-radius:6px;'
                    f'font-size:0.68rem;font-weight:700;margin:2px 3px 2px 0;'
                    f'background:{c}18;border:1px solid {c}40;color:{c};">{lbl}</span>'
                    for lbl, c in badge_items
                )
                st.markdown(bdgs, unsafe_allow_html=True)

            with col2:
                # ── Key Decision Metrics — replaces redundant stock card ───
                # Shows the 7 signals that answer "is this worth buying?" —
                # new information not already visible in the hero or score strip.
                _roce_r = _sg("roce_med_10y", None)
                roce_ov = float(_roce_r if _roce_r is not None else _sg("roce", 0))
                pat5_ov = float(_sg("pat_gr_5y", 0))
                peg_ov  = float(_sg("peg", 0))
                cfo_ov  = float(_sg("cfo_to_pat", 0))
                de_ov   = float(_sg("debt_to_equity", 0))
                prom_ov = float(_sg("promoter_holdings", 0))
                plg_ov  = float(_sg("pledged_percentage", 0))
                fcfy_ov = float(_sg("fcf_yield", 0))

                def _krow(label, val_str, passed, note=""):
                    ico, clr = (("✅", COLORS["green"]) if passed is True else
                                ("❌", COLORS["red"])   if passed is False else
                                ("⚪", COLORS["text_muted"]))
                    nh = (f'<span style="font-size:0.63rem;color:{COLORS["text_muted"]};'
                          f'flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;'
                          f'white-space:nowrap;margin-left:6px;">{note}</span>') if note else ""
                    return (
                        f'<div style="display:flex;align-items:center;gap:8px;padding:6px 0;'
                        f'border-bottom:1px solid rgba(255,255,255,0.04);">'
                        f'<span style="width:18px;text-align:center;flex-shrink:0;">{ico}</span>'
                        f'<span style="font-size:0.74rem;color:{COLORS["text_secondary"]};'
                        f'width:128px;flex-shrink:0;">{label}</span>'
                        f'<span style="font-size:0.82rem;font-weight:700;color:{clr};'
                        f'white-space:nowrap;flex-shrink:0;">'
                        f'{val_str}</span>{nh}</div>'
                    )

                peg_str  = f"{peg_ov:.2f}×" if peg_ov > 0 else "N/A"
                peg_pass = (0 < peg_ov <= 1.0) if peg_ov > 0 else None
                pr_note  = f"⚠️ {plg_ov:.0f}% pledged" if plg_ov > 10 else "≥50% aligned"

                kpi_html = (
                    _krow("ROCE 10Y Median", f"{roce_ov:.1f}%",  roce_ov >= 15,    "≥15%")      +
                    _krow("PAT CAGR 5Y",     f"{pat5_ov:.1f}%",  pat5_ov >= 15,    "≥15%")      +
                    _krow("PEG Ratio",         peg_str,           peg_pass,        "Lynch ≤1.0") +
                    _krow("CFO / PAT",       f"{cfo_ov:.1f}%",   cfo_ov >= 70,     "≥70% cash") +
                    _krow("D / E Ratio",     f"{de_ov:.2f}",      de_ov < 0.5,     "<0.5 safe") +
                    _krow("Promoter Hold.",  f"{prom_ov:.1f}%",   prom_ov >= 50,   pr_note)     +
                    _krow("FCF Yield",       f"{fcfy_ov:.1f}%",   fcfy_ov >= 3,    "≥3% solid")
                )

                st.markdown(f"""
                <div style="background:{COLORS['bg_secondary']};border:1px solid {COLORS['border']};
                            border-left:3px solid {tc_ov['text']};border-radius:10px;
                            padding:14px 16px;">
                  <div style="font-size:0.64rem;font-weight:800;color:{tc_ov['text']};
                              text-transform:uppercase;letter-spacing:1.2px;margin-bottom:8px;">
                    {tier_cfg_ov['emoji']} {tier_cfg_ov['label']} &nbsp;·&nbsp; Score {comp_ov:.0f} / 100
                  </div>
                  {kpi_html}
                </div>
                """, unsafe_allow_html=True)

            st.markdown(
                f"<div class='sec-head'>📊 Business & Financial Analysis</div>",
                unsafe_allow_html=True,
            )
            render_financial_insights(stock)

        # ── Tab B: Forensics & Accounting ─────────────────────────────────
        with _itabs[1]:
            st.markdown(
                f"<div class='sec-head'>🔬 Forensic Fraud Perimeter (25-Flag Cascade)</div>",
                unsafe_allow_html=True,
            )
            render_forensic_perimeter(stock)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='sec-head'>🧠 Systematic Fisher Proxy — 7 Automated Checks</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='sec-cap'>Phil Fisher's 15 qualitative points translated into "
                f"strict quantitative proxies using pre-derived CSV columns. "
                f"100% automated — zero manual input.</div>",
                unsafe_allow_html=True,
            )
            render_fisher_module(stock)

        # ── Tab C: Guru Frameworks ────────────────────────────────────────
        with _itabs[2]:
            st.markdown(
                f"<div class='sec-head'>🏛️ Guru Framework Alignment — 32 Frameworks</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='sec-cap'>Pre-computed framework badges from scoring engine. "
                f"Each represents a complete quantamental screen from a master investor's methodology.</div>",
                unsafe_allow_html=True,
            )
            render_guru_frameworks(stock)

        # ── Tab D: Matrix & WCS ───────────────────────────────────────────
        with _itabs[3]:
            render_moat_growth_matrix(filt, highlight_stock=selected)
            st.markdown("<br>", unsafe_allow_html=True)
            render_ep_power_curve_module(stock)
            render_bruised_blue_chip_badge(stock)
            render_multitrillioncap_card(stock)

        # ── Tab E: All Data ───────────────────────────────────────────────
        with _itabs[4]:
            st.markdown(
                f"<div class='sec-head'>📊 Raw Signal Data — Full Universe Output</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='sec-cap'>Every computed signal across all 6 data sheets. "
                f"Grouped by category. All values are engine-computed — no re-calculation here.</div>",
                unsafe_allow_html=True,
            )
            render_raw_signals(stock)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4: MARKET PULSE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[3]:

    # ── Pre-compute section datasets ───────────────────────────────
    _mp_ts   = (df[df["tsunami_signal"] == 1].sort_values("composite_score", ascending=False)
                if "tsunami_signal" in df.columns else df.iloc[:0])
    _mp_qglp = (filt[filt["qglp_pass"] == 1].sort_values("qglp_score", ascending=False)
                if "qglp_pass" in filt.columns else filt.iloc[:0])
    _mp_qual = df[df["gate_pass"] == 1] if "gate_pass" in df.columns else df

    _bbc_mask = (
        (df["market_cap"].fillna(0)   >= 20_000 if "market_cap"   in df.columns else pd.Series(False, index=df.index)) &
        (df["roce_med_10y"].fillna(0) >= 20      if "roce_med_10y" in df.columns else pd.Series(False, index=df.index)) &
        (df["pe_discount"].fillna(0)  >= 20      if "pe_discount"  in df.columns else pd.Series(False, index=df.index))
    )
    _bbc_cnt = int(_bbc_mask.sum())

    # ── Top summary strip ──────────────────────────────────────────
    render_metric_strip([
        (str(len(_mp_ts)),   "🌊 Tsunami",     "m-purple"),
        (str(len(_mp_qglp)), "🏛️ QGLP",        "m-gold"),
        (str(_mp_qual["sector"].nunique() if "sector" in _mp_qual.columns else 0), "📈 Sectors", "m-blue"),
        (str(_bbc_cnt),      "💙 Blue Chips",  "m-green"),
    ])

    # ── Inner navigation tabs ──────────────────────────────────────
    _mp_tabs = st.tabs([
        "🌊 Tsunami",
        "🏛️ QGLP",
        "📈 Sectors",
        "💙 Blue Chips",
        "🚀 Tipping Points",
    ])

    # ══ Tsunami ════════════════════════════════════════════════════
    with _mp_tabs[0]:
        st.markdown(
            f"<div class='sec-cap'>All 7 conviction conditions fire together: Quality + Momentum + "
            f"Governance + Technical. Rare by design.</div>",
            unsafe_allow_html=True,
        )
        if len(_mp_ts) == 0:
            st.info("🌊 No tsunami signals in current conditions — all 7 gates must fire simultaneously.")
        else:
            _ts_undi = int(_mp_ts["tsunami_undiscovered"].sum()) if "tsunami_undiscovered" in _mp_ts.columns else 0
            _ts_avg  = float(_mp_ts["composite_score"].mean())   if "composite_score"      in _mp_ts.columns else 0
            st.markdown(f"""
            <div style="display:flex;gap:20px;padding:8px 2px 12px 2px;
                 border-bottom:1px solid {COLORS['border']};margin-bottom:10px;flex-wrap:wrap;">
              <span style="font-size:0.82rem;font-weight:800;color:{COLORS['purple']};">
                🌊 {len(_mp_ts)} Tsunami signals
              </span>
              <span style="font-size:0.78rem;color:{COLORS['gold']};">
                🏆 {_ts_undi} undiscovered
              </span>
              <span style="font-size:0.78rem;color:{COLORS['text_muted']};">
                Avg score <strong style="color:{COLORS['green']}">{_ts_avg:.0f}</strong>
              </span>
            </div>
            """, unsafe_allow_html=True)

            _ts_cols = [c for c in ["rank","name","sector","market_category","market_cap",
                                    "composite_score","quality_score","momentum_score",
                                    "piotroski_fscore","smart_money_flow","buy_zone_label"]
                        if c in _mp_ts.columns]
            _ts_sel = st.dataframe(
                _mp_ts[_ts_cols].reset_index(drop=True),
                column_config={
                    "composite_score": st.column_config.ProgressColumn("Score",    min_value=0, max_value=100, format="%.0f"),
                    "quality_score":   st.column_config.ProgressColumn("Quality",  min_value=0, max_value=100, format="%.0f"),
                    "momentum_score":  st.column_config.ProgressColumn("Momentum", min_value=0, max_value=100, format="%.0f"),
                    "piotroski_fscore": st.column_config.NumberColumn("F-Score",   format="%.0f/9"),
                    "market_cap":      st.column_config.NumberColumn("MCap ₹Cr",   format="%.0f"),
                    "rank":            st.column_config.NumberColumn("Rank",        format="%.0f"),
                },
                use_container_width=True,
                height=min(480, 80 + len(_mp_ts) * 35 + 40),
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
            )
            _ts_rows = _ts_sel.selection.rows if _ts_sel and hasattr(_ts_sel, "selection") else []
            if _ts_rows:
                _ts_pick = _mp_ts.iloc[_ts_rows[0]]["name"]
                st.session_state["xray_stock"] = _ts_pick
                st.markdown(f"""
                <div style="padding:9px 14px;margin-top:8px;background:rgba(139,92,246,0.07);
                     border:1px solid rgba(139,92,246,0.3);border-radius:8px;font-size:0.8rem;">
                  🔬 <strong style="color:{COLORS['text_primary']};">{_ts_pick}</strong>
                  set — <strong style="color:{COLORS['blue']};">click The Tear-Sheet tab</strong> for full analysis.
                </div>
                """, unsafe_allow_html=True)

    # ══ QGLP ═══════════════════════════════════════════════════════
    with _mp_tabs[1]:
        st.markdown(
            "<div class='sec-cap'>Raamdeo Agrawal's framework: ROCE>15%, PAT growth>15%, "
            "Promoter>50%, reasonable valuation. Strict gates. Filtered by sidebar.</div>",
            unsafe_allow_html=True,
        )
        if len(_mp_qglp) == 0:
            st.info("No stocks pass the strict QGLP gates with current sidebar filters.")
        else:
            _q_avg = float(_mp_qglp["qglp_score"].mean()) if "qglp_score" in _mp_qglp.columns else 0
            st.markdown(f"""
            <div style="display:flex;gap:20px;padding:8px 2px 12px 2px;
                 border-bottom:1px solid {COLORS['border']};margin-bottom:10px;flex-wrap:wrap;">
              <span style="font-size:0.82rem;font-weight:800;color:{COLORS['gold']};">
                🏛️ {len(_mp_qglp)} QGLP compounders
              </span>
              <span style="font-size:0.78rem;color:{COLORS['text_muted']};">
                Avg QGLP score <strong style="color:{COLORS['blue']}">{_q_avg:.0f}</strong>
              </span>
            </div>
            """, unsafe_allow_html=True)

            _q_cols = [c for c in ["rank","name","sector","market_cap","qglp_score",
                                   "qglp_quality","qglp_growth","qglp_longevity","qglp_price",
                                   "smart_money_flow","buy_zone_label"]
                       if c in _mp_qglp.columns]
            _q_sel = st.dataframe(
                _mp_qglp[_q_cols].reset_index(drop=True),
                column_config={
                    "qglp_score":     st.column_config.ProgressColumn("QGLP",      min_value=0, max_value=100, format="%.0f"),
                    "qglp_quality":   st.column_config.ProgressColumn("Quality",   min_value=0, max_value=100, format="%.0f"),
                    "qglp_growth":    st.column_config.ProgressColumn("Growth",    min_value=0, max_value=100, format="%.0f"),
                    "qglp_longevity": st.column_config.ProgressColumn("Longevity", min_value=0, max_value=100, format="%.0f"),
                    "qglp_price":     st.column_config.ProgressColumn("Price/PEG", min_value=0, max_value=100, format="%.0f"),
                    "market_cap":     st.column_config.NumberColumn("MCap ₹Cr",    format="%.0f"),
                    "rank":           st.column_config.NumberColumn("Rank",         format="%.0f"),
                },
                use_container_width=True,
                height=min(500, 80 + len(_mp_qglp) * 35 + 40),
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
            )
            _q_rows = _q_sel.selection.rows if _q_sel and hasattr(_q_sel, "selection") else []
            if _q_rows:
                _q_pick = _mp_qglp.iloc[_q_rows[0]]["name"]
                st.session_state["xray_stock"] = _q_pick
                st.markdown(f"""
                <div style="padding:9px 14px;margin-top:8px;background:rgba(228,179,65,0.07);
                     border:1px solid rgba(228,179,65,0.3);border-radius:8px;font-size:0.8rem;">
                  🔬 <strong style="color:{COLORS['text_primary']};">{_q_pick}</strong>
                  set — <strong style="color:{COLORS['blue']};">click The Tear-Sheet tab</strong> for full analysis.
                </div>
                """, unsafe_allow_html=True)

    # ══ Sectors ════════════════════════════════════════════════════
    with _mp_tabs[2]:
        if len(_mp_qual) == 0:
            st.info("No gate-qualified stocks to analyse.")
        else:
            _sec_stats = (
                _mp_qual.groupby("sector").agg(
                    stocks=("name", "count"),
                    avg_quality=("quality_score",   "mean"),
                    avg_momentum=("momentum_score",  "mean"),
                    avg_composite=("composite_score","mean"),
                    crown_jewels=("conviction_tier", lambda x: (x == 1).sum()),
                ).sort_values("avg_composite", ascending=False).head(15)
            )

            _sc1, _sc2 = st.columns([2, 3])
            with _sc1:
                st.dataframe(
                    _sec_stats.reset_index(),
                    column_config={
                        "avg_quality":   st.column_config.ProgressColumn("Quality",  min_value=0, max_value=100, format="%.0f"),
                        "avg_momentum":  st.column_config.ProgressColumn("Momentum", min_value=0, max_value=100, format="%.0f"),
                        "avg_composite": st.column_config.ProgressColumn("Score",    min_value=0, max_value=100, format="%.0f"),
                        "crown_jewels":  st.column_config.NumberColumn("👑 T1",      format="%.0f"),
                        "stocks":        st.column_config.NumberColumn("Count",      format="%.0f"),
                    },
                    use_container_width=True,
                    height=470,
                    hide_index=True,
                )
            with _sc2:
                _bar_clrs = [
                    COLORS["green"] if v >= 65 else COLORS["gold"] if v >= 50 else COLORS["red"]
                    for v in _sec_stats["avg_composite"]
                ]
                _fig_sec = go.Figure(go.Bar(
                    x=list(_sec_stats.index),
                    y=list(_sec_stats["avg_composite"]),
                    marker_color=_bar_clrs,
                    text=[f"{v:.0f}" for v in _sec_stats["avg_composite"]],
                    textposition="outside",
                    textfont=dict(color=COLORS["text_primary"], size=11),
                ))
                _fig_sec.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=COLORS["text_primary"], size=10),
                    height=470,
                    margin=dict(t=10, b=90, l=30, r=10),
                    xaxis=dict(tickangle=-35, gridcolor=COLORS["border"], tickfont=dict(size=10)),
                    yaxis=dict(gridcolor=COLORS["border"], range=[0, 105]),
                    showlegend=False,
                )
                st.plotly_chart(_fig_sec, use_container_width=True)

    # ══ Bruised Blue Chips ═════════════════════════════════════════
    with _mp_tabs[3]:
        render_bruised_blue_chips(df)

    # ══ Tipping Points ═════════════════════════════════════════════
    with _mp_tabs[4]:
        render_multi_trillion_tipping_points(df)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5: CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[4]:
    st.markdown(f"<div class='sec-head'>⚙️ System Configuration</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='sec-cap'>Current scoring weights and gate thresholds. Modify config.py to adjust.</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Composite Blend Weights**")
        for k, v in COMPOSITE_WEIGHTS.items():
            st.markdown(f"- {k.title()}: **{v*100:.0f}%**")
        st.markdown("**Quality Sub-Weights (6 Layers)**")
        for k, v in QUALITY_WEIGHTS.items():
            src = {"moat": "SQGLP", "growth": "SQGLP", "cash": "Coffee Can",
                   "margin": "Fisher", "balance_sheet": "Baid", "valuation": "Marks+Baid"}
            st.markdown(f"- {k.replace('_',' ').title()} ({src.get(k,'')}): **{v*100:.0f}%**")
    with c2:
        st.markdown("**Hard Gates (7 Frameworks)**")
        for name, cfg in HARD_GATES.items():
            st.markdown(f"- {cfg['description']}")
        st.markdown("**Momentum Sub-Weights (CAN-SLIM)**")
        for k, v in MOMENTUM_WEIGHTS.items():
            st.markdown(f"- {k.replace('_',' ').title()}: **{v*100:.0f}%**")

    # ── MARKS CYCLE TEMPERATURE GAUGE ──
    st.markdown("---")
    st.markdown(f"<div class='sec-head'>🌡️ Marks Cycle Temperature Gauge</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='sec-cap'>Howard Marks' 5-Dimension Market Cycle Assessment. "
                f"Score each dimension 1 (cold/fear) to 5 (hot/greed). Total 5-25.</div>", unsafe_allow_html=True)

    tc1, tc2 = st.columns(2)
    with tc1:
        t_val = st.slider("📊 Valuations (1=PE<17, 5=PE>25)", 1, 5,
                          DEFAULT_CYCLE_TEMPERATURE["valuations"], key="ct_val")
        t_credit = st.slider("🏦 Credit Conditions (1=tight, 5=loose)", 1, 5,
                             DEFAULT_CYCLE_TEMPERATURE["credit_conditions"], key="ct_credit")
        t_psych = st.slider("🧠 Investor Psychology (1=fear, 5=greed)", 1, 5,
                            DEFAULT_CYCLE_TEMPERATURE["investor_psychology"], key="ct_psych")
    with tc2:
        t_cap = st.slider("📈 Capital Markets (1=no IPOs, 5=IPO mania)", 1, 5,
                          DEFAULT_CYCLE_TEMPERATURE["capital_markets"], key="ct_cap")
        t_qual = st.slider("⚖️ Market Quality (1=quality leads, 5=junk leads)", 1, 5,
                           DEFAULT_CYCLE_TEMPERATURE["market_quality"], key="ct_qual")

    cycle_total = t_val + t_credit + t_psych + t_cap + t_qual
    if cycle_total <= MARKS_CYCLE["posture_aggressive"]["max_score"]:
        posture = MARKS_CYCLE["posture_aggressive"]
    elif cycle_total <= MARKS_CYCLE["posture_neutral"]["max_score"]:
        posture = MARKS_CYCLE["posture_neutral"]
    else:
        posture = MARKS_CYCLE["posture_defensive"]

    posture_color = "#3fb950" if "Aggressive" in posture["label"] else "#d29922" if "Neutral" in posture["label"] else "#f85149"
    st.markdown(f"""
    <div style="background:{COLORS['bg_secondary']}; border:2px solid {posture_color};
                border-radius:12px; padding:20px; margin:10px 0; text-align:center;">
        <div style="font-size:2.5rem; font-weight:900; color:{posture_color};">{cycle_total}/25</div>
        <div style="font-size:1.3rem; font-weight:700; color:{posture_color}; margin-top:4px;">{posture["label"]}</div>
        <div style="font-size:0.85rem; color:{COLORS['text_muted']}; margin-top:8px;">{posture["action"]}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── BAID SELL TRIGGERS INFO ──
    st.markdown("---")
    st.markdown(f"<div class='sec-head'>🚨 Baid Sell Trigger Rules</div>", unsafe_allow_html=True)
    sell_alert_count = int(df.get("sell_alert_any", pd.Series(0)).sum())
    st.info(f"**{sell_alert_count}** stocks currently have active sell alerts.")
    for trigger_name, trigger_cfg in BAID_SELL_TRIGGERS.items():
        st.markdown(f"- **{trigger_name.replace('_', ' ').title()}:** {trigger_cfg['description']}")

    # ── MEAN REVERSION INFO ──
    st.markdown("---")
    st.markdown(f"<div class='sec-head'>📉 Mean Reversion Risk (Marks)</div>", unsafe_allow_html=True)
    mr_count = int(df.get("mean_reversion_risk", pd.Series(0)).sum())
    st.info(f"**{mr_count}** stocks flagged with cyclical peak margins (OPM or NPM > {MEAN_REVERSION['opm_spike_threshold']}× their 5Y median).")
    st.markdown(f"Quality score penalty: **{(1-MEAN_REVERSION['penalty_factor'])*100:.0f}%** reduction for flagged stocks.")

    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center; padding:20px; color:{COLORS['text_muted']}; font-size:0.75rem;">
        The Systematic Architect's Index v{UI['version']} · Adaptive Quantamental Engine<br>
        Dr. Malik (SSGR+8 Params) · Raamdeo (QGLP) · O'Neil (CAN-SLIM) · Mukherjea (Coffee Can)<br>
        Howard Marks (Cycles) · Philip Fisher · Peter Lynch (PEG) · Schilit (Forensics)<br>
        {total} stocks · {len(df.columns)} signals · {load_time:.1f}s pipeline<br>
        <strong>Marks Cycle Posture: {posture['label']}</strong>
    </div>
    """, unsafe_allow_html=True)
