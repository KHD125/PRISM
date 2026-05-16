"""
Multibagger Discovery System v1.0
==================================
Quantamental Compounding Engine — 7-Tab Streamlit Application
"""
import os
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

import streamlit as st
st.set_page_config(page_title="Multibagger Discovery System", page_icon="🏆", layout="wide", initial_sidebar_state="expanded")

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import warnings
warnings.filterwarnings('ignore')

from data_engine import fetch_and_clean_data
from scoring_engine import run_full_scoring
from forensic_engine import run_forensic_analysis
from ui_components import (inject_css, render_hero_banner, render_metric_strip,
                           render_stock_card, render_radar_chart, render_tier_summary,
                           render_score_bar, render_sidebar_brand)
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
def get_clean_data(data_source, uploaded_dict, sheet_id):
    """Tier-1: Expensive data fetch + clean. Heavily cached."""
    t0 = time.time()
    df = fetch_and_clean_data(data_source, uploaded_dict, sheet_id)
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

    st.markdown("---")
    # ══ Analysis Mode ══
    st.markdown("### 🎯 Analysis Mode")
    analysis_mode = st.selectbox(
        "Mode",
        options=list(ANALYSIS_MODES.keys()),
        index=0,
        format_func=lambda k: ANALYSIS_MODES[k]["label"],
        key="sel_mode",
        label_visibility="collapsed",
    )
    st.caption(ANALYSIS_MODES[analysis_mode]["description"])

    # ══ Scoring Profile — filtered by selected mode ══
    allowed = ANALYSIS_MODES[analysis_mode]["allowed_profiles"]
    st.markdown("### 📊 Scoring Profile")
    scoring_profile = st.selectbox(
        "Profile",
        options=allowed,
        index=0,
        format_func=lambda k: f"{MASTER_PROFILES[k]['icon']} {MASTER_PROFILES[k]['label']}",
        key="sel_profile",
        label_visibility="collapsed",
    )
    profile_cfg = MASTER_PROFILES[scoring_profile]
    st.caption(profile_cfg["description"])

if not data_ready:
    st.info("👋 Welcome! Please select a data source from the sidebar (Google Sheets or Upload CSV) to begin scanning.")
    st.stop()

with st.spinner("🔄 Loading data..."):
    try:
        clean_df, load_time = get_clean_data(st.session_state.data_source, uploaded_dict, sheet_id)
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.stop()

with st.spinner(f"🧭 Scoring with {scoring_profile} profile..."):
    try:
        df = get_scored_data(clean_df, analysis_mode, scoring_profile)
    except Exception as e:
        st.error(f"❌ Scoring error: {e}")
        st.stop()

# Show active engine weights in sidebar AFTER scoring
adaptive_w = df.attrs.get("adaptive_weights", {})
if adaptive_w:
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ⚙️ Active Engine Weights")
        regime_label = adaptive_w.get("regime_label", "🟡 Sideways")
        st.markdown(f"**Regime:** {regime_label}")
        w_cols = st.columns(4)
        w_cols[0].metric("Quality", f"{adaptive_w.get('quality_w', 0):.0%}")
        w_cols[1].metric("Growth", f"{adaptive_w.get('growth_w', 0):.0%}")
        w_cols[2].metric("Longevity", f"{adaptive_w.get('longevity_w', 0):.0%}")
        w_cols[3].metric("Price", f"{adaptive_w.get('price_w', 0):.0%}")
        st.caption(
            f"Gates: ROCE≥{adaptive_w.get('roce_gate', 15):.0f}% · "
            f"Growth≥{adaptive_w.get('growth_gate', 15):.0f}% · "
            f"PEG≤{adaptive_w.get('peg_gate', 1.5):.1f}"
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

# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════
tabs = st.tabs(["🏠 Discovery", "🔍 Scanner", "🛡️ Forensic", "📊 X-Ray", "🌊 Tsunami", "🏛️ QGLP Compounders", "📈 Sectors", "⚙️ Config"])

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
    st.markdown(f"<div class='sec-head'>🔍 Deep Scanner — {len(filt)} Stocks</div>", unsafe_allow_html=True)
    display_cols = ["rank","name","sector","industry","market_category","market_cap","composite_score",
                    "quality_score","valuation_score","moat_score","growth_score","cash_score","momentum_score",
                    "governance_bonus","piotroski_fscore","forensic_label","tier_label",
                    "mean_reversion_risk","sell_alert_any",
                    "gate_pass","gates_failed","close_price","roe_med_10y","roce_med_10y",
                    "cfo_to_pat","pat_gr_5y","rev_gr_5y","debt_to_equity","peg","pledged_percentage",
                    "promoter_holdings","crs_50d","ret_vs_industry_1y"]
    available = [c for c in display_cols if c in filt.columns]
    st.dataframe(
        filt[available].reset_index(drop=True),
        use_container_width=True, height=600,
        column_config={
            "composite_score": st.column_config.ProgressColumn("Composite", min_value=0, max_value=100, format="%.0f"),
            "quality_score": st.column_config.ProgressColumn("Quality", min_value=0, max_value=100, format="%.0f"),
            "momentum_score": st.column_config.ProgressColumn("Momentum", min_value=0, max_value=100, format="%.0f"),
            "market_cap": st.column_config.NumberColumn("MCap ₹Cr", format="%.0f"),
        }
    )
    csv_data = filt[available].to_csv(index=False)
    st.download_button("📥 Export CSV", csv_data, "multibagger_scan.csv", "text/csv")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3: FORENSIC AUDIT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[2]:
    st.markdown(f"<div class='sec-head'>🛡️ Forensic Audit Dashboard</div>", unsafe_allow_html=True)
    forensic_df = filt[filt["conviction_tier"].isin([1,2,3])].copy()

    c1, c2, c3, c4 = st.columns(4)
    clean = int((forensic_df["forensic_label"] == "🟢 Clean").sum())
    watch = int((forensic_df["forensic_label"] == "🟡 Watch").sum())
    caution = int((forensic_df["forensic_label"] == "🟠 Caution").sum())
    high_risk = int((forensic_df["forensic_label"] == "🔴 High Risk").sum())
    c1.metric("🟢 Clean", clean)
    c2.metric("🟡 Watch", watch)
    c3.metric("🟠 Caution", caution)
    c4.metric("🔴 High Risk", high_risk)

    # F-Score histogram
    if len(forensic_df) > 0:
        fig_f = px.histogram(forensic_df, x="piotroski_fscore", nbins=10, color_discrete_sequence=[COLORS['purple']])
        fig_f.update_layout(title="Piotroski F-Score Distribution", xaxis_title="F-Score (0-9)",
                           yaxis_title="Count", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           font=dict(color=COLORS['text_primary']), height=300)
        st.plotly_chart(fig_f, use_container_width=True)

    forensic_cols = ["rank","name","tier_label","composite_score","piotroski_fscore","piotroski_label",
                     "forensic_score","forensic_label","red_flag_count","red_flag_list","cf_triangle"]
    avail_f = [c for c in forensic_cols if c in forensic_df.columns]
    st.dataframe(forensic_df[avail_f].reset_index(drop=True), use_container_width=True, height=400)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4: STOCK X-RAY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[3]:
    st.markdown(f"<div class='sec-head'>📊 Stock X-Ray — Deep Dive Analysis</div>", unsafe_allow_html=True)
    stock_names = filt["name"].dropna().tolist()
    if stock_names:
        selected = st.selectbox("Select Stock", stock_names, key="xray_stock")
        stock = filt[filt["name"] == selected].iloc[0]

        c1, c2 = st.columns([1, 1])
        with c1:
            render_stock_card(stock, show_scores=True)
            st.markdown(f"**Gate Status:** {'✅ All gates passed' if stock.get('gate_pass',0)==1 else '❌ Failed: ' + str(stock.get('failed_gates',''))}")
            st.markdown(f"**Forensic:** {stock.get('forensic_label','')} · F-Score: {stock.get('piotroski_fscore','N/A')}/9")
            st.markdown(f"**Smart Money Flow:** {stock.get('smart_money_flow', '⚪ Neutral')}")
            st.markdown(f"**Cashflow Triangle:** {stock.get('cf_triangle','')}")
            if stock.get('red_flag_count', 0) > 0:
                st.warning(f"🚨 Red Flags: {stock.get('red_flag_list','')}")
        with c2:
            fig = render_radar_chart(stock, f"{selected} — Quality Profile")
            st.plotly_chart(fig, use_container_width=True)

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
                triggers.append("**Cash Quality Collapse:** CFO/PAT dropped below 0.5")
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
    else:
        st.info("No stocks match current filters.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5: TSUNAMI SIGNALS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[4]:
    st.markdown(f"<div class='sec-head'>🌊 Tsunami Signals — Maximum Conviction Setups</div>", unsafe_allow_html=True)
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 6: QGLP COMPOUNDERS (MOTILAL OSWAL)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[5]:
    st.markdown(f"<div class='sec-head'>🏛️ QGLP Compounders (Motilal Oswal Framework)</div>", unsafe_allow_html=True)
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 7: SECTOR INTELLIGENCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[6]:
    st.markdown(f"<div class='sec-head'>📈 Sector Intelligence</div>", unsafe_allow_html=True)
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
# TAB 8: CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[7]:
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
        Multibagger Discovery System v{UI['version']} · 7 Frameworks Fused<br>
        SQGLP + Coffee Can + Fisher + CAN-SLIM + Forensic Shenanigans + Howard Marks + Compounding Codex<br>
        {total} stocks · {len(df.columns)} signals · {load_time:.1f}s pipeline<br>
        <strong>Marks Cycle Posture: {posture['label']}</strong>
    </div>
    """, unsafe_allow_html=True)
