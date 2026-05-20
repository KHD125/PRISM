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
                inject_css, render_hero_banner, render_metric_strip, render_stock_card,
                render_radar_chart, render_tier_summary, render_score_bar, render_sidebar_brand)
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
# 🏛️ THE COMMAND CENTER (Global Strategy Controls)
# ═══════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="background:linear-gradient(135deg, {COLORS['bg_secondary']}, #0d1117); border:1px solid {COLORS['border']}; border-radius:8px; padding:15px 20px; margin-bottom:20px;">
    <div style="color:{COLORS['text_muted']}; font-size:0.85rem; font-weight:700; letter-spacing:1px; text-transform:uppercase; margin-bottom:10px;">
        ⚙️ Quantamental Command Center
    </div>
</div>
""", unsafe_allow_html=True)

cc_col1, cc_col2 = st.columns([1, 1])

with cc_col1:
    analysis_mode = st.selectbox(
        "🎯 Master Analysis Mode",
        options=list(ANALYSIS_MODES.keys()),
        index=0,
        format_func=lambda k: ANALYSIS_MODES[k]["label"],
        key="sel_mode",
    )
    st.caption(ANALYSIS_MODES[analysis_mode]["description"])

with cc_col2:
    allowed = ANALYSIS_MODES[analysis_mode]["allowed_profiles"]
    scoring_profile = st.selectbox(
        "📊 Scoring Profile (Regime Aware)",
        options=allowed,
        index=0,
        format_func=lambda k: f"{MASTER_PROFILES[k]['icon']} {MASTER_PROFILES[k]['label']}",
        key="sel_profile",
    )
    profile_cfg = MASTER_PROFILES[scoring_profile]
    st.caption(profile_cfg["description"])

st.markdown("<br>", unsafe_allow_html=True)

with st.spinner(f"🧭 Scoring with {scoring_profile} profile..."):
    try:
        df = get_scored_data(clean_df, analysis_mode, scoring_profile)
    except Exception as e:
        st.error(f"❌ Scoring error: {e}")
        st.stop()

if df is None or df.empty:
    st.warning("⚠️ No data returned after scoring. Check your data source or filters.")
    st.stop()

# Active Engine Weights - Rendered in Expander below Command Center
adaptive_w = df.attrs.get("adaptive_weights", {})
if adaptive_w:
    with st.expander("🔬 View Active Engine Weights & Thresholds", expanded=False):
        regime_label = adaptive_w.get("regime_label", "🟡 Sideways")
        w_cols = st.columns(5)
        w_cols[0].metric("Regime", regime_label)
        w_cols[1].metric("Quality Weight", f"{adaptive_w.get('quality_w', 0):.0%}")
        w_cols[2].metric("Growth Weight", f"{adaptive_w.get('growth_w', 0):.0%}")
        w_cols[3].metric("Longevity Weight", f"{adaptive_w.get('longevity_w', 0):.0%}")
        w_cols[4].metric("Price Weight", f"{adaptive_w.get('price_w', 0):.0%}")
        st.caption(
            f"**Active Hard Gates:** ROCE ≥ {adaptive_w.get('roce_gate', 15):.0f}% | "
            f"Growth ≥ {adaptive_w.get('growth_gate', 15):.0f}% | "
            f"PEG ≤ {adaptive_w.get('peg_gate', 1.5):.1f}"
        )
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

# ── Profile Context Badge (always visible above tabs) ──
if adaptive_w:
    profile_icon = MASTER_PROFILES.get(scoring_profile, {}).get('icon', '⚖️')
    regime = df.attrs.get("detected_market_regime", "SIDEWAYS")
    regime_emoji = "🟢" if regime == "BULL" else "🔴" if regime == "BEAR" else "🟡"
    st.markdown(f"""
    <div style="background:linear-gradient(135deg, {COLORS['bg_secondary']}, {COLORS['bg_tertiary']});
                border:1px solid {COLORS['border']}; border-radius:10px; padding:10px 18px;
                margin:8px 0 16px 0; display:flex; align-items:center; gap:20px; flex-wrap:wrap;">
        <span style="font-size:0.8rem; color:{COLORS['text_muted']};">
            <b>{profile_icon} {scoring_profile}</b> · {analysis_mode} Mode · {regime_emoji} {regime}
        </span>
        <span style="font-size:0.75rem; color:{COLORS['text_secondary']};">
            Q:{adaptive_w['quality_w']:.0%} · G:{adaptive_w['growth_w']:.0%} ·
            L:{adaptive_w['longevity_w']:.0%} · P:{adaptive_w['price_w']:.0%}
        </span>
        <span style="font-size:0.72rem; color:{COLORS['text_muted']};">
            Gates: ROCE≥{adaptive_w['roce_gate']:.0f}% · Growth≥{adaptive_w['growth_gate']:.0f}% · PEG≤{adaptive_w['peg_gate']:.1f}
        </span>
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
    st.markdown(f"<div class='sec-head'>📋 Conviction Tier Overview</div>", unsafe_allow_html=True)
    render_tier_summary(df)

    st.markdown(f"<div class='sec-head'>🏆 Top Conviction Stocks ({len(filt)} filtered)</div>", unsafe_allow_html=True)
    for _, row in filt.head(20).iterrows():
        render_stock_card(row, show_scores=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2: DEEP SCANNER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[1]:
    st.markdown(f"<div class='sec-head'>🔍 Deep Scanner — {len(filt)} Stocks · {profile_cfg.get('icon', '⚖️')} {scoring_profile}</div>", unsafe_allow_html=True)

    priority = profile_cfg.get("priority_cols", [])
    render_scanner_grid(filt, priority_cols=priority)
    
    csv_data = filt.to_csv(index=False)
    st.download_button("📥 Export CSV", csv_data, "multibagger_scan.csv", "text/csv")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3: THE TEAR-SHEET (X-Ray + Forensic + Fisher)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[2]:
    st.markdown(f"<div class='sec-head'>🔬 The Ultimate Tear-Sheet</div>", unsafe_allow_html=True)
    # Quantamental Best Practice: Tear Sheet searches the FULL Universe.
    all_stock_names = df["name"].dropna().tolist()
    if all_stock_names:
        c_sel, c_blank = st.columns([1, 2])
        with c_sel:
            selected = st.selectbox("Select Stock for Deep Dive (Full 2,000+ Universe)", all_stock_names, key="xray_stock")
        stock = df[df["name"] == selected].iloc[0]
        
        # ELITE ARCHITECTURE: Explicit Rejection Banner for failed stocks
        if stock.get("gate_pass", 0) == 0:
            st.markdown(f"""
            <div style="background:rgba(248, 81, 73, 0.1); border:1px solid #f85149; border-radius:8px; padding:15px; margin-bottom:20px;">
                <h3 style="color:#f85149; margin:0 0 5px 0;">❌ SYSTEM REJECTED</h3>
                <p style="color:#eee; margin:0; font-size:0.95rem;">
                    This stock failed the Quantitative Hard Gates. It is excluded from the main scanner.<br>
                    <b>Failure Reason(s):</b> {stock.get('failed_gates', 'Unknown')}
                </p>
            </div>
            """, unsafe_allow_html=True)

        # TOP LEVEL: Radar & Core Profile
        c1, c2 = st.columns([1, 1])
        with c1:
            render_stock_card(stock, show_scores=True)
            st.markdown(f"**Gate Status:** {'✅ All gates passed' if stock.get('gate_pass',0)==1 else '❌ Failed: ' + str(stock.get('failed_gates',''))}")
            st.markdown(f"**Forensic:** {stock.get('forensic_label','')} · F-Score: {stock.get('piotroski_fscore','N/A')}/9")
            st.markdown(f"**Smart Money Flow:** {stock.get('smart_money_flow', '⚪ Neutral')}")
            st.markdown(f"**Cashflow Triangle:** {stock.get('cf_triangle','')}")
            st.markdown(f"**Moat-Growth:** {stock.get('moat_growth_quad', 'N/A')}")
            if stock.get('red_flag_count', 0) > 0:
                st.warning(f"🚨 Red Flags: {stock.get('red_flag_list','')}")
        with c2:
            fig = render_radar_chart(stock, f"{selected} — Quality Profile")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        
        # MID LEVEL: Visualizations (Matrix)
        render_moat_growth_matrix(filt, highlight_stock=selected)
        
        st.markdown("---")

        # ── Dr. Malik + WCS Signals ──
        st.markdown(f"<div class='sec-head'>🕉️ Peaceful Investing Signals (Dr. Malik)</div>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Malik Score", f"{stock.get('malik_score', 0):.0f}/100", stock.get('malik_label', ''))
        m2.metric("SSGR", f"{stock.get('ssgr', 0):.1f}%")
        m3.metric("SSGR Cushion", f"{stock.get('ssgr_cushion', 0):.1f}%",
                  "Self-Funded ✅" if stock.get('ssgr_self_funded', 0) == 1 else "Debt-Dependent ⚠️")
        m4.metric("Interest Coverage", f"{stock.get('interest_coverage', 0):.1f}x")

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Tax Rate Est.", f"{stock.get('tax_rate_est', 0):.1f}%")
        m6.metric("CFO/PAT", f"{stock.get('cfo_to_pat', 0):.1f}%")
        m7.metric("Earnings Yield", f"{stock.get('earnings_yield', 0):.1f}%")
        m8.metric("High Cash+Debt", "⚠️ Flagged" if stock.get('high_cash_high_debt', 0) == 1 else "Clean ✅")

        st.markdown(f"<div class='sec-head'>💰 Wealth Creation Signals (MOSL WCS)</div>", unsafe_allow_html=True)
        w1, w2, w3, w4 = st.columns(4)
        w1.metric("Economic Profit", f"₹{stock.get('economic_profit', 0):.0f} Cr",
                  "EP Positive ✅" if stock.get('economic_profit_positive', 0) == 1 else "EP Negative ❌")
        w2.metric("QGLP Score", f"{stock.get('qglp_score', 0):.0f}/100")
        w3.metric("Moat-Growth", stock.get('moat_growth_quad', 'N/A'))
        w4.metric("Sales→Profit Conv", "✅ Positive Leverage" if stock.get('operating_leverage', 0) == 1 else "❌ Negative Leverage")

        w5, w6, w7, w8 = st.columns(4)
        pe_roe_mos = stock.get('pe_vs_roe_mos', 0)
        w5.metric("P/E vs ROE (MoS)", f"{pe_roe_mos:.1f}", "Value Zone ✅" if pe_roe_mos > 0 else "Premium Zone ⚠️")
        w6.metric("PEG Zone", stock.get('peg_zone', 'N/A'))
        w7.metric("Capex Productive", "✅ Yes" if stock.get('capex_productive', 0) == 1 else "❌ No / Wait")
        w8.metric("QGLP Pass", "✅ Yes" if stock.get('qglp_pass', 0) == 1 else "❌ No")

        st.markdown(f"<div class='sec-head'>📋 Key Financials</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROCE Med 10Y", f"{stock.get('roce_med_10y', 0):.1f}%")
        c2.metric("ROE Med 10Y", f"{stock.get('roe_med_10y', 0):.1f}%")
        c3.metric("CFO/PAT", f"{stock.get('cfo_to_pat', 0):.1f}%")
        c4.metric("D/E Ratio", f"{stock.get('debt_to_equity', 0):.2f}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("PAT Gr 5Y", f"{stock.get('pat_gr_5y', 0):.1f}%")
        c2.metric("Rev Gr 5Y", f"{stock.get('rev_gr_5y', 0):.1f}%")
        c3.metric("NPM Med 5Y", f"{stock.get('npm_med_5y', 0):.1f}%")
        c4.metric("PEG", f"{stock.get('peg', 0):.2f}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Promoter %", f"{stock.get('promoter_holdings', 0):.1f}%")
        c2.metric("Pledge %", f"{stock.get('pledged_percentage', 0):.1f}%")
        c3.metric("FII %", f"{stock.get('fii_holdings', 0):.1f}%")
        c4.metric("CRS 50D", f"{stock.get('crs_50d', 0):.0f}")

        # ── SELL ALERTS (Baid's 3 Triggers) ──
        has_sell_alert = stock.get('sell_alert_any', 0) == 1
        if has_sell_alert:
            st.markdown(f"<div class='sec-head' style='color:#f85149;'>🚨 BAID SELL TRIGGERS ACTIVE</div>", unsafe_allow_html=True)
            triggers = []
            if stock.get('sell_alert_thesis_broken', 0) == 1:
                triggers.append("**Thesis Broken:** ROCE trajectory declining structurally")
            if stock.get('sell_alert_mgmt_deteriorated', 0) == 1:
                triggers.append("**Management Deteriorated:** Pledge rising + promoter selling + D/E rising")
            if stock.get('sell_alert_cash_collapse', 0) == 1:
                triggers.append("**Cash Quality Collapse:** CFO/PAT dropped below 50%")
            for t in triggers:
                st.error(t)

        # ── MEAN REVERSION WARNING (Marks) ──
        if stock.get('mean_reversion_risk', 0) == 1:
            st.warning("⚠️ **Marks Mean Reversion Risk:** Current margins are significantly above 5Y median — cyclical peak risk detected. Quality score penalized by 15%.")

        # ── VALUATION ATTRACTIVENESS ──
        st.markdown(f"<div class='sec-head'>💰 Valuation Attractiveness (Marks + Baid)</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Valuation Score", f"{stock.get('valuation_score', 0):.0f}/100")
        c2.metric("PE Discount vs 10Y", f"{stock.get('pe_discount', 0):.1f}%")
        c3.metric("EV Compression", f"{stock.get('ev_compression', 0):.1f}")
        c4.metric("FCF Yield", f"{stock.get('fcf_yield', 0):.1f}%")
        
        st.markdown("---")
        # BOTTOM LEVEL: The Systematic Fisher Proxy
        render_fisher_module(stock)
    else:
        st.info("No stocks match current filters.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4: MARKET PULSE (Tsunami, QGLP, Sectors)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[3]:
    st.markdown(f"<div class='sec-head'>🌊 Market Pulse — Alpha Strategies</div>", unsafe_allow_html=True)
    
    st.markdown("#### 🌊 Tsunami Signals (Max Conviction)")
    st.markdown(f"<div class='sec-cap'>Stocks where ALL conviction layers fire simultaneously: Quality + Momentum + Governance + Technical</div>", unsafe_allow_html=True)

    tsunami_df = df[df["tsunami_signal"] == 1].sort_values("composite_score", ascending=False)

    if len(tsunami_df) == 0:
        st.info("🌊 No tsunami signals detected in current market conditions. This is rare and by design — these are the highest-conviction setups.")
    else:
        render_metric_strip([
            (str(len(tsunami_df)), "Tsunami Signals", "m-purple"),
            (str(int(tsunami_df["tsunami_undiscovered"].sum())), "Undiscovered (Tier C)", "m-gold"),
            (f"{tsunami_df['composite_score'].mean():.0f}", "Avg Score", "m-green"),
        ])
        for _, row in tsunami_df.iterrows():
            st.markdown(f"""<div class="tsunami-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div style="font-weight:800; font-size:1.1rem; color:#E8E3FF;">
                            🌊 {row['name']}
                        </div>
                        <div style="font-size:0.75rem; color:#A8A3D8; margin-top:2px;">
                            {row.get('sector','')} · ₹{row.get('market_cap',0):,.0f} Cr · {row.get('market_category','')}
                        </div>
                    </div>
                    <div style="font-size:2rem; font-weight:900; color:#FFD700;">{row['composite_score']:.0f}</div>
                </div>
            </div>""", unsafe_allow_html=True)
            render_stock_card(row, show_scores=True)


    st.markdown("---")
    st.markdown("#### 🏛️ QGLP Compounders (Motilal Oswal Framework)")
    st.markdown(f"<div class='sec-cap'>Quality + Growth + Longevity + Price. Exclusive to strictly vetted compounders with ROCE > 15%, EPS/PAT Growth > 15%, and reasonable valuations.</div>", unsafe_allow_html=True)
    
    qglp_df = filt[filt["qglp_pass"] == 1].sort_values("qglp_score", ascending=False)
    
    if len(qglp_df) == 0:
        st.info("No stocks pass the strict QGLP gates right now.")
    else:
        render_metric_strip([
            (str(len(qglp_df)), "QGLP Stocks", "m-gold"),
            (f"{qglp_df['qglp_score'].mean():.0f}", "Avg QGLP Score", "m-blue"),
        ])
        
        display_cols = ["rank","name","sector","market_cap","qglp_score","qglp_quality","qglp_growth", "qglp_longevity", "qglp_price", "smart_money_flow"]
        avail_cols = [c for c in display_cols if c in qglp_df.columns]
        st.dataframe(qglp_df[avail_cols].reset_index(drop=True), use_container_width=True, height=500,
                     column_config={
                         "qglp_score": st.column_config.ProgressColumn("QGLP", min_value=0, max_value=100, format="%.0f"),
                         "qglp_quality": st.column_config.ProgressColumn("Quality", min_value=0, max_value=100, format="%.0f"),
                         "qglp_growth": st.column_config.ProgressColumn("Growth", min_value=0, max_value=100, format="%.0f"),
                         "qglp_price": st.column_config.ProgressColumn("Price (PEG)", min_value=0, max_value=100, format="%.0f"),
                     })


    st.markdown("---")
    st.markdown(f"#### 📈 Sector Intelligence", unsafe_allow_html=True)
    qual_df = df[df["gate_pass"] == 1]

    if len(qual_df) > 0:
        sector_stats = qual_df.groupby("sector").agg(
            count=("name", "count"),
            avg_quality=("quality_score", "mean"),
            avg_momentum=("momentum_score", "mean"),
            avg_composite=("composite_score", "mean"),
            best_stock=("composite_score", "idxmax"),
            tier1_count=("conviction_tier", lambda x: (x == 1).sum()),
        ).sort_values("avg_composite", ascending=False).head(20)

        # Get best stock names
        sector_stats["top_stock"] = sector_stats["best_stock"].map(df["name"])
        sector_stats = sector_stats.drop(columns=["best_stock"])

        st.dataframe(sector_stats.reset_index(), use_container_width=True, height=500,
                     column_config={
                         "avg_quality": st.column_config.ProgressColumn("Avg Quality", min_value=0, max_value=100, format="%.0f"),
                         "avg_composite": st.column_config.ProgressColumn("Avg Composite", min_value=0, max_value=100, format="%.0f"),
                     })

        fig_sec = px.bar(sector_stats.head(15).reset_index(), x="sector", y="avg_composite",
                        color="avg_quality", color_continuous_scale="Viridis",
                        title="Top 15 Sectors by Average Composite Score")
        fig_sec.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                             font=dict(color=COLORS['text_primary']), height=400,
                             xaxis_tickangle=-45)
        st.plotly_chart(fig_sec, use_container_width=True)


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
