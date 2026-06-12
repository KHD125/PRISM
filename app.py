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
import re
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core import fetch_and_clean_data, run_full_scoring, compute_forensic_signals, apply_forensic_penalty
from ui import (render_scanner_grid, render_moat_growth_matrix, render_fisher_module,
                render_ep_power_curve_module, render_bruised_blue_chip_badge,
                render_multitrillioncap_card, render_forensic_perimeter, render_guru_frameworks,
                render_financial_insights, render_stock_hero, render_score_strip,
                render_sell_alerts_panel, render_raw_signals,
                render_canslim_radar, render_sepa_radar, render_schilit_shield, render_dorsey_radar,
                render_outsider_radar, render_marks_radar, render_malik_radar,
                render_lynch_radar, render_mauboussin_radar, render_mosl_wealth_matrix,
                render_valuation_inversion_and_sizing_cockpit,
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

def get_scored_data(clean_df: pd.DataFrame, analysis_mode: str, scoring_profile: str) -> pd.DataFrame:
    """Tier-2+3: Instant scoring + forensic pass. NOT cached — runs in <0.5s on dropdown change.

    3-step sequencing contract (non-negotiable order):
      1. compute_forensic_signals : Piotroski F-Score → 27 red flags → Schilit 4-checkers →
                                    Cashflow Triangle. Writes forensic_score, forensic_label,
                                    red_flag_count, piotroski_fscore, schilit_forensic_score.
                                    MUST run first: 5 framework gates read these columns.
      2. run_full_scoring         : Hard gates → Quality → Momentum → Governance → Composite →
                                    Framework flags (Diamond, Dhandho, SQGLP Engine, Schilit,
                                    Fisher all read forensic columns from step 1). → Tsunami.
      3. apply_forensic_penalty   : Cascading multiplier on composite_score → conviction tier
                                    reassignment. MUST run last: composite_score only exists
                                    after step 2.
    """
    df = compute_forensic_signals(clean_df)
    df = run_full_scoring(df, analysis_mode, scoring_profile)
    df = apply_forensic_penalty(df)
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
        # Full refresh: clear the Tier-1 data cache AND the Tier-2 scored-df session cache,
        # so a re-score runs from scratch (picks up engine code changes, not stale labels).
        st.cache_data.clear()
        st.session_state.pop("_scored_df", None)
        st.session_state.pop("_score_key", None)
        st.rerun()

    sheet_id = None
    uploaded_dict = None
    data_ready = False

    if st.session_state.data_source == "sheet":
        sheet_id = st.text_input("Google Sheets URL or ID", placeholder="Enter Google Sheet ID...")
        if sheet_id:
            data_ready = True
    elif st.session_state.data_source == "upload":
        uploaded_files = st.file_uploader("Upload all 6 CSV files (Ratio, Income, Balance, Cashflow, Shareholding, Technical)", type="csv", accept_multiple_files=True)
        if uploaded_files and len(uploaded_files) > 0:
            uploaded_dict = {}
            _unmatched = []
            for f in uploaded_files:
                fname = f.name.lower()
                # Most-specific keywords first — prevents "cashflow_ratios.csv" misrouting to "ratio"
                if   "shareholding" in fname: uploaded_dict["shareholding"] = f
                elif "technical"    in fname: uploaded_dict["technical"]    = f
                elif "cashflow"     in fname or "cash_flow" in fname: uploaded_dict["cashflow"] = f
                elif "balance"      in fname: uploaded_dict["balance"]      = f
                elif "income"       in fname: uploaded_dict["income"]       = f
                elif "ratio"        in fname: uploaded_dict["ratio"]        = f
                else: _unmatched.append(f.name)
            # Show slot-by-slot match status so user sees exactly what mapped where
            _slots = ["ratio", "income", "balance", "cashflow", "shareholding", "technical"]
            _status_lines = []
            for _s in _slots:
                if _s in uploaded_dict:
                    _status_lines.append(f"✅ **{_s}** ← `{uploaded_dict[_s].name}`")
                else:
                    _status_lines.append(f"❌ **{_s}** — not matched")
            if _unmatched:
                for _u in _unmatched:
                    _status_lines.append(f"⚠️ `{_u}` — unrecognized (rename to include the sheet type)")
            st.markdown("\n".join(_status_lines))
            # All 6 required — load_all_csvs raises FileNotFoundError on any missing slot
            if all(_s in uploaded_dict for _s in _slots):
                data_ready = True
            else:
                _missing = [s for s in _slots if s not in uploaded_dict]
                st.warning(f"Missing sheets: {', '.join(_missing)}. Upload all 6 to proceed.")

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

profile_cfg = MASTER_PROFILES[scoring_profile]

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

    # ── SMART INTERCONNECTED FILTER CASCADE ──────────────────────────────────
    # Every option-based filter narrows the OPTIONS of every filter BELOW it. A single
    # progressively-narrowed frame (_cf) drives all option lists AND the final filtered
    # dataframe — so what a dropdown shows is exactly what survives the filter (zero
    # drift between options and results).
    # Defensive: stored multiselect selections are pruned to the current valid options
    # before each widget renders, preventing Streamlit's "value not in options" crash.
    def _ms_cascade(label, options, key, default, help=None):
        """Cascade-safe multiselect. Fully manages session_state (no `default=` arg, which
        avoids Streamlit's default-plus-session-state warning) and prunes any stale stored
        selection down to the current options each run. Empty selection = no filter."""
        if key not in st.session_state:
            st.session_state[key] = [v for v in default if v in options]
        else:
            st.session_state[key] = [v for v in st.session_state[key] if v in options]
        return st.multiselect(label, options, key=key, help=help)

    def _ordered_present(frame, col, order):
        """Labels present in frame[col], in canonical `order`; unknown labels appended last."""
        if col not in frame.columns:
            return []
        present = set(frame[col].dropna().astype(str).unique())
        opts = [v for v in order if v in present]
        opts += [v for v in sorted(present) if v not in opts]
        return opts

    _cf = df   # progressively-narrowed cascade frame — drives every option list below

    # 1. Market Category — cascade root (only categories present in the data)
    _ALL_MCAPS = ["Mega Cap", "Large Cap", "Mid Cap", "Small Cap", "Micro Cap", "Nano Cap"]
    _mcap_opts = _ordered_present(_cf, "market_category", _ALL_MCAPS)
    sel_mcap = _ms_cascade("Market Category", _mcap_opts, "sb_mcap", default=_mcap_opts)
    if sel_mcap:
        _cf = _cf[_cf["market_category"].isin(sel_mcap)]

    # 2. Sector — only sectors within the chosen market categories
    _sector_opts = ["All"] + sorted(_cf["sector"].dropna().unique().tolist())
    if st.session_state.get("sb_sector", "All") not in _sector_opts:
        st.session_state["sb_sector"] = "All"
    sel_sector = st.selectbox("Sector", _sector_opts, key="sb_sector")
    if sel_sector != "All":
        _cf = _cf[_cf["sector"] == sel_sector]

    # 3. Industry — only industries within the chosen categories AND sector
    _industry_opts = ["All"] + sorted(_cf["industry"].dropna().unique().tolist())
    if st.session_state.get("sb_industry", "All") not in _industry_opts:
        st.session_state["sb_industry"] = "All"
    sel_industry = st.selectbox(
        "Industry", _industry_opts, key="sb_industry",
        help="Granular industry within the selected sector. Narrows with Sector above.",
    )
    if sel_industry != "All":
        _cf = _cf[_cf["industry"] == sel_industry]

    # 4. Conviction Tier — only tiers present in the remaining stocks
    _tier_opts = sorted(int(t) for t in _cf["conviction_tier"].dropna().unique())
    sel_tier = _ms_cascade("Conviction Tier", _tier_opts, "sb_tier",
                           default=[t for t in (1, 2, 3) if t in _tier_opts])
    if sel_tier:
        _cf = _cf[_cf["conviction_tier"].isin(sel_tier)]

    # ── 3-TIER FRAMEWORK FILTER ENGINE ───────────────────────────────────────
    # Set Algebra: (Universe − Excluded) ∩ (Included ∪ ∅) ∩ (∀ Combined)
    # Evaluation order: Exclude first → Include second → Combination last.
    # Each panel's options cascade from the panel above — zero impossible combos.
    #
    # Panel 1 — Exclude (NOT): remove stocks passing ANY selected framework
    # Panel 2 — Include (OR) : show stocks passing ANY selected framework (current behavior)
    # Panel 3 — Combination (AND): keep only stocks passing ALL selected simultaneously

    def _extract_frameworks(frame):
        """Extract sorted unique framework names from the cascade frame.
        Splitter uses ', ' — the exact separator written by scoring_engine fw_str builder."""
        if "frameworks_passed" not in frame.columns:
            return []
        return sorted(set(
            fw.strip()
            for cell in frame["frameworks_passed"].dropna()
            if cell != "None"
            for fw in cell.split(", ")
            if fw.strip()
        ))

    def _fw_match_mask(frame, fw_list, logic="or"):
        """Build a boolean mask for stocks matching framework list.
        logic='or'  → stock passes ANY of the selected frameworks
        logic='and' → stock passes ALL of the selected frameworks simultaneously
        Regex is boundary-safe: prevents 'Bruised Blue Chip' from substring-matching
        'Bruised Blue Chip 29' via anchored exact-token pattern."""
        if not fw_list or "frameworks_passed" not in frame.columns:
            return pd.Series(True, index=frame.index)
        if logic == "or":
            mask = pd.Series(False, index=frame.index)
            for fw in fw_list:
                _pat = r"(?:^|, )" + re.escape(fw) + r"(?:,|$)"
                mask = mask | frame["frameworks_passed"].str.contains(_pat, regex=True, na=False)
            return mask
        else:  # logic == "and"
            mask = pd.Series(True, index=frame.index)
            for fw in fw_list:
                _pat = r"(?:^|, )" + re.escape(fw) + r"(?:,|$)"
                mask = mask & frame["frameworks_passed"].str.contains(_pat, regex=True, na=False)
            return mask

    st.markdown(
        f'<div style="font-size:0.72rem;font-weight:800;color:{COLORS["purple"]};'
        f'text-transform:uppercase;letter-spacing:1.2px;margin:12px 0 6px 0;'
        f'padding-bottom:4px;border-bottom:1px solid {COLORS["border"]};"'
        f'>🧬 Framework Filter Engine</div>',
        unsafe_allow_html=True,
    )

    # 5a. EXCLUDE Framework — NOT logic (applied first, narrows universe)
    _all_fw_excl = _extract_frameworks(_cf)
    sel_fw_exclude = _ms_cascade(
        "🚫 Exclude Framework", _all_fw_excl, "sb_fw_exclude", default=[],
        help="Remove stocks passing ANY of these frameworks. Applied first.",
    )
    if sel_fw_exclude:
        _excl_mask = _fw_match_mask(_cf, sel_fw_exclude, logic="or")
        _cf = _cf[~_excl_mask]
    # Exclude badge
    _n_excl_fw = len(sel_fw_exclude)
    if _n_excl_fw > 0:
        st.markdown(
            f'<div style="font-size:0.6rem;color:{COLORS["red"]};padding:0 0 6px 2px;"'
            f'>⛔ {_n_excl_fw} excluded · {len(_cf)} stocks remaining</div>',
            unsafe_allow_html=True,
        )

    # 5b. INCLUDE Framework — OR logic (show stocks passing ANY selected)
    _all_fw_incl = _extract_frameworks(_cf)
    sel_fw_include = _ms_cascade(
        "✅ Include Framework", _all_fw_incl, "sb_fw_include", default=[],
        help="Show stocks passing ANY of these. Empty = all remaining stocks.",
    )
    if sel_fw_include:
        _incl_mask = _fw_match_mask(_cf, sel_fw_include, logic="or")
        _cf = _cf[_incl_mask]
    # Include badge
    _n_incl_fw = len(sel_fw_include)
    if _n_incl_fw > 0:
        st.markdown(
            f'<div style="font-size:0.6rem;color:{COLORS["green"]};padding:0 0 6px 2px;"'
            f'>✅ {_n_incl_fw} included (OR) · {len(_cf)} stocks remaining</div>',
            unsafe_allow_html=True,
        )

    # 5c. COMBINATION Framework — AND logic (stock must pass ALL selected)
    _all_fw_comb = _extract_frameworks(_cf)
    sel_fw_combine = _ms_cascade(
        "🔗 Combination Framework", _all_fw_comb, "sb_fw_combine", default=[],
        help="Stock must pass ALL of these simultaneously. AND logic.",
    )
    if sel_fw_combine:
        _comb_mask = _fw_match_mask(_cf, sel_fw_combine, logic="and")
        _cf = _cf[_comb_mask]
    # Combination badge
    _n_comb_fw = len(sel_fw_combine)
    if _n_comb_fw > 0:
        st.markdown(
            f'<div style="font-size:0.6rem;color:{COLORS["blue"]};padding:0 0 6px 2px;"'
            f'>🔗 {_n_comb_fw} required (AND) · {len(_cf)} stocks remaining</div>',
            unsafe_allow_html=True,
        )

    # 6. Moat-Growth quadrant — only quadrants present in the remaining stocks
    _MOAT_ORDER = ["⭐ Wealth Creator", "🛡️ Quality Trap", "⚡ Growth Trap", "💀 Wealth Destroyer"]
    _moat_opts = _ordered_present(_cf, "moat_growth_quad", _MOAT_ORDER)
    sel_moat = _ms_cascade("Moat", _moat_opts, "sb_moat", default=[])
    if sel_moat:
        _cf = _cf[_cf["moat_growth_quad"].isin(sel_moat)]

    # 7. PEG Zone — valuation tier (options present in remaining stocks, canonical order)
    _PEG_ZONE_ORDER = [
        "💎 Deep Value", "🟢 Fair PEG", "🟡 Stretched",
        "🟠 Expensive", "🔴 Overpriced", "🔴 Declining",
    ]
    _peg_opts = _ordered_present(_cf, "peg_zone", _PEG_ZONE_ORDER)
    sel_peg_zone = _ms_cascade("PEG Zone", _peg_opts, "sb_peg_zone", default=[],
                               help="Valuation tier from the PEG ratio. Empty = all stocks.")
    if sel_peg_zone and "peg_zone" in _cf.columns:
        _cf = _cf[_cf["peg_zone"].isin(sel_peg_zone)]

    # 8. Buy Zone — entry timing vs Volatility Stop (options present in remaining stocks)
    _BUY_ZONE_ORDER = [
        "🟢 Perfect Entry (Low Risk)", "🟡 Standard Zone",
        "🔴 Extended (Wait for Pullback)", "⚪ Uncharted",
    ]
    _buy_opts = _ordered_present(_cf, "buy_zone_label", _BUY_ZONE_ORDER)
    sel_buy_zone = _ms_cascade("Buy Zone", _buy_opts, "sb_buy_zone", default=[],
                               help="Entry timing vs the Volatility Stop. Empty = all stocks.")
    if sel_buy_zone and "buy_zone_label" in _cf.columns:
        _cf = _cf[_cf["buy_zone_label"].isin(sel_buy_zone)]

    # Institutional Sweep Vector
    st.markdown("---")
    st.markdown("<div style='font-size:0.8rem; font-weight:700; color:#8b5cf6; margin-bottom:5px;'>🌊 ALPHA VECTORS</div>", unsafe_allow_html=True)
    smart_sweep = st.checkbox("🎯 Smart Money Sweep (FII+DII + Breakout)", value=False, key="sb_sweep")
    
    gate_only = st.checkbox("Gate-passed only", value=True, key="sb_gate")
    min_quality = st.slider("Min Quality Score", 0, 100, 0, key="sb_minq")

# Apply filters — the cascade frame (_cf) already encodes every option-based filter
# (Market Category → Sector → Industry → Tier → Framework → Moat → PEG Zone → Buy Zone),
# so the dropdown options and the result set are guaranteed identical. Only the bottom
# Alpha-Vector toggles remain to apply.
filt = _cf.copy()
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
    _disc_slice = _disc_df.head(_shown_n)
    for _di in range(len(_disc_slice)):
        _drow = _disc_slice.iloc[_di]
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
        # ── Stock selector: search + dropdown ────────────────────────
        _ts_c1, _ts_c2 = st.columns([2, 5])
        with _ts_c1:
            search_ticker = st.text_input(
                "Search", placeholder="🔍  HDFC, Infosys, TATA…",
                key="search_ticker", label_visibility="collapsed",
            )
        _term  = (search_ticker or "").strip().upper()
        _names = [n for n in all_stock_names if _term in n.upper()] if _term else all_stock_names
        if not _names:
            _names = all_stock_names
        _prev_sel = st.session_state.get("xray_stock")
        _ts_idx   = _names.index(_prev_sel) if _prev_sel in _names else 0
        with _ts_c2:
            selected = st.selectbox(
                "Stock", _names, index=_ts_idx, key="xray_stock",
                label_visibility="collapsed",
            )

        stock   = df[df["name"] == selected].iloc[0]
        _regime = df.attrs.get("detected_market_regime", "SIDEWAYS")

        # ── Null-safe getter — available to ALL inner tabs ─────────────────
        def _sg(k, d=0):
            v = stock.get(k, d)
            return d if (v is None or (isinstance(v, float) and np.isnan(v))) else v

        # Pre-compute verdict inputs once — reused across tabs
        _gate_ok  = stock.get("gate_pass", 0) == 1
        _sell_any = stock.get("sell_alert_any", 0) == 1
        _mr_risk  = stock.get("mean_reversion_risk", 0) == 1
        _tier_num = int(_sg("conviction_tier", 5))
        _tc       = TIER_COLORS.get(_tier_num, TIER_COLORS[5])
        _tier_cfg = next((t for t in CONVICTION_TIERS if t["tier"] == _tier_num), CONVICTION_TIERS[-1])
        _comp_sc  = float(_sg("composite_score", 0))

        # ── Verdict logic: one clear signal, always at the top ─────────────
        if not _gate_ok:
            _verdict     = "SYSTEM REJECTED"
            _verdict_clr = COLORS["red"]
            _verdict_bg  = "rgba(248,81,73,0.09)"
            _verdict_reason = f"Hard Gate Failure — {stock.get('failed_gates', 'Unknown')}"
        elif _sell_any:
            _verdict     = "SELL ALERT"
            _verdict_clr = COLORS["red"]
            _verdict_bg  = "rgba(248,81,73,0.07)"
            _verdict_reason = "One or more Baid sell triggers have fired — review Forensics tab."
        elif _tier_num <= 2 and _comp_sc >= 70:
            _verdict     = "BUY CANDIDATE"
            _verdict_clr = COLORS["green"]
            _verdict_bg  = "rgba(63,185,80,0.08)"
            _verdict_reason = (
                f"{_tier_cfg['emoji']} {_tier_cfg['label']} · "
                f"Score {_comp_sc:.0f}/100 · All hard gates passed"
            )
        elif _tier_num <= 3 and _comp_sc >= 55:
            _verdict     = "MONITOR"
            _verdict_clr = COLORS["gold"]
            _verdict_bg  = "rgba(228,179,65,0.07)"
            _verdict_reason = (
                f"{_tier_cfg['emoji']} {_tier_cfg['label']} · "
                f"Score {_comp_sc:.0f}/100 · Watch for better entry point"
            )
        else:
            _verdict     = "AVOID"
            _verdict_clr = COLORS["text_muted"]
            _verdict_bg  = "rgba(110,118,129,0.06)"
            _verdict_reason = (
                f"Tier {_tier_num} · Score {_comp_sc:.0f}/100 · "
                f"Does not meet investment threshold"
            )

        _mr_pill = (
            f'<span style="font-size:0.67rem;font-weight:700;padding:2px 10px;'
            f'border-radius:12px;background:rgba(228,179,65,0.15);color:{COLORS["gold"]};'
            f'border:1px solid rgba(228,179,65,0.4);white-space:nowrap;">⚠️ Mean Reversion</span>'
        ) if _mr_risk else ""

        st.markdown(f"""
        <div style="background:{_verdict_bg};border:1px solid {_verdict_clr}55;
             border-left:4px solid {_verdict_clr};border-radius:10px;
             padding:11px 16px;margin:6px 0 10px 0;display:flex;
             align-items:center;gap:14px;flex-wrap:wrap;">
          <span style="font-size:0.77rem;font-weight:900;color:{_verdict_clr};
               letter-spacing:1.2px;white-space:nowrap;">{_verdict}</span>
          <span style="font-size:0.75rem;color:{COLORS['text_secondary']};
               flex:1;min-width:160px;">{_verdict_reason}</span>
          {_mr_pill}
        </div>
        """, unsafe_allow_html=True)

        # Sell alerts panel — only rendered when active
        if _sell_any:
            render_sell_alerts_panel(stock)

        # ── Hero + score strip ────────────────────────────────────────────
        render_stock_hero(stock, regime=_regime)
        render_score_strip(stock)

        # ── Inner tabs ────────────────────────────────────────────────────
        _itabs = st.tabs([
            "📋 Overview",
            "🔬 Forensics",
            "🏛️ Frameworks",
            "📈 Matrix & WCS",
            "📊 All Data",
        ])

        # ══ Tab A: Overview ════════════════════════════════════════════════
        with _itabs[0]:
            col1, col2 = st.columns([1, 1])

            with col1:
                fig = render_radar_chart(stock, f"{selected} — Quality Radar")
                st.plotly_chart(fig, use_container_width=True)

                # Signal badges row
                pio_raw = stock.get("piotroski_fscore", None)
                pio_val = None
                if pio_raw is not None and not (isinstance(pio_raw, float) and np.isnan(pio_raw)):
                    try:
                        pio_val = int(float(pio_raw))
                    except Exception:
                        pio_val = None
                pio_str = f"{pio_val}/9" if pio_val is not None else "N/A"
                pio_clr = (COLORS["green"] if pio_val is not None and pio_val >= 7 else
                           COLORS["gold"]  if pio_val is not None and pio_val >= 5 else
                           COLORS["text_muted"] if pio_val is None else COLORS["red"])
                smart  = str(stock.get("smart_money_flow", "⚪ Neutral") or "⚪ Neutral")
                cf_tri = str(stock.get("cf_triangle", "") or "")
                quad   = str(stock.get("moat_growth_quad", "") or "")
                badge_items = [(f"F-Score {pio_str}", pio_clr), (smart, COLORS["purple"])]
                if cf_tri:
                    badge_items.append((cf_tri, COLORS["blue"]))
                if quad:
                    badge_items.append((quad, _tc["text"]))
                bdgs = "".join(
                    f'<span style="display:inline-block;padding:3px 9px;border-radius:6px;'
                    f'font-size:0.68rem;font-weight:700;margin:2px 3px 2px 0;'
                    f'background:{c}18;border:1px solid {c}40;color:{c};">{lbl}</span>'
                    for lbl, c in badge_items
                )
                st.markdown(bdgs, unsafe_allow_html=True)

            with col2:
                # 7-KPI buy decision checklist
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
                    nh = (
                        f'<span style="font-size:0.63rem;color:{COLORS["text_muted"]};'
                        f'flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;'
                        f'white-space:nowrap;margin-left:6px;">{note}</span>'
                    ) if note else ""
                    return (
                        f'<div style="display:flex;align-items:center;gap:8px;padding:6px 0;'
                        f'border-bottom:1px solid rgba(255,255,255,0.04);">'
                        f'<span style="width:18px;text-align:center;flex-shrink:0;">{ico}</span>'
                        f'<span style="font-size:0.74rem;color:{COLORS["text_secondary"]};'
                        f'width:128px;flex-shrink:0;">{label}</span>'
                        f'<span style="font-size:0.82rem;font-weight:700;color:{clr};'
                        f'white-space:nowrap;flex-shrink:0;">{val_str}</span>{nh}</div>'
                    )

                peg_str  = f"{peg_ov:.2f}×" if peg_ov > 0 else "N/A"
                peg_pass = (0 < peg_ov <= 1.0) if peg_ov > 0 else None
                pr_note  = f"⚠️ {plg_ov:.0f}% pledged" if plg_ov > 10 else "≥50% aligned"

                kpi_html = (
                    _krow("ROCE 10Y Median", f"{roce_ov:.1f}%",  roce_ov >= 15,   "≥15%")      +
                    _krow("PAT CAGR 5Y",     f"{pat5_ov:.1f}%",  pat5_ov >= 15,   "≥15%")      +
                    _krow("PEG Ratio",         peg_str,           peg_pass,       "Lynch ≤1.0") +
                    _krow("CFO / PAT",       f"{cfo_ov:.1f}%",   cfo_ov >= 70,    "≥70% cash") +
                    _krow("D / E Ratio",     f"{de_ov:.2f}",      de_ov < 0.5,    "<0.5 safe") +
                    _krow("Promoter Hold.",  f"{prom_ov:.1f}%",   prom_ov >= 50,  pr_note)     +
                    _krow("FCF Yield",       f"{fcfy_ov:.1f}%",   fcfy_ov >= 3,   "≥3% solid")
                )

                st.markdown(f"""
                <div style="background:{COLORS['bg_secondary']};border:1px solid {COLORS['border']};
                            border-left:3px solid {_tc['text']};border-radius:10px;
                            padding:14px 16px;">
                  <div style="font-size:0.64rem;font-weight:800;color:{_tc['text']};
                              text-transform:uppercase;letter-spacing:1.2px;margin-bottom:8px;">
                    {_tier_cfg['emoji']} {_tier_cfg['label']} &nbsp;·&nbsp; Score {_comp_sc:.0f} / 100
                  </div>
                  {kpi_html}
                </div>
                """, unsafe_allow_html=True)

            st.markdown(
                f"<div class='sec-head'>📊 Business & Financial Analysis</div>",
                unsafe_allow_html=True,
            )
            render_financial_insights(stock)

        # ══ Tab B: Forensics ═══════════════════════════════════════════════
        with _itabs[1]:
            # ── Forensic KPI strip — F-Score · Red Flags · Forensic% · CF Triangle
            _f_fscore  = int(_sg("piotroski_fscore", 0))
            _f_flags   = int(_sg("red_flag_count", 0))
            _f_forensic = float(_sg("forensic_score", 0))
            _f_cftri   = str(stock.get("cf_triangle", "—") or "—")
            _fkpi_data = [
                (str(_f_fscore), "/9",  "F-Score",
                 COLORS["green"] if _f_fscore >= 7 else COLORS["gold"] if _f_fscore >= 5 else COLORS["red"]),
                (str(_f_flags),  "",    "Red Flags",
                 COLORS["green"] if _f_flags == 0 else COLORS["gold"] if _f_flags <= 2 else COLORS["red"]),
                (f"{_f_forensic:.0f}", "%", "Forensic",
                 COLORS["green"] if _f_forensic >= 80 else COLORS["gold"] if _f_forensic >= 60 else COLORS["red"]),
                (_f_cftri, "", "CF Triangle",
                 COLORS["green"] if any(x in _f_cftri for x in ("✅", "🟢")) else
                 COLORS["gold"]  if "🟡" in _f_cftri else COLORS["red"]),
            ]
            _fkpi_html = "".join(
                f'<div style="flex:1;min-width:80px;background:{COLORS["bg_secondary"]};'
                f'border:1px solid {COLORS["border"]};border-radius:10px;'
                f'padding:10px 14px;text-align:center;">'
                f'<div style="font-size:1.4rem;font-weight:900;color:{clr};line-height:1;">'
                f'{val}<span style="font-size:0.75rem;color:{COLORS["text_muted"]};">{suf}</span></div>'
                f'<div style="font-size:0.57rem;color:{COLORS["text_muted"]};text-transform:uppercase;'
                f'letter-spacing:0.6px;margin-top:4px;">{lbl}</div>'
                f'</div>'
                for val, suf, lbl, clr in _fkpi_data
            )
            st.markdown(
                f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px;">'
                f'{_fkpi_html}</div>',
                unsafe_allow_html=True,
            )

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
                f"<div class='sec-cap'>Phil Fisher's 15 qualitative points translated into strict "
                f"quantitative proxies using pre-derived CSV columns. "
                f"100% automated — zero manual input.</div>",
                unsafe_allow_html=True,
            )
            render_fisher_module(stock)

            st.markdown("<br>", unsafe_allow_html=True)
            render_schilit_shield(stock)

        # ══ Tab C: Guru Frameworks ═════════════════════════════════════════
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

            st.markdown("<br>", unsafe_allow_html=True)
            render_canslim_radar(stock)

            st.markdown("<br>", unsafe_allow_html=True)
            render_sepa_radar(stock)

            st.markdown("<br>", unsafe_allow_html=True)
            render_dorsey_radar(stock)

            st.markdown("<br>", unsafe_allow_html=True)
            render_outsider_radar(stock)

            st.markdown("<br>", unsafe_allow_html=True)
            render_marks_radar(stock)

            st.markdown("<br>", unsafe_allow_html=True)
            render_malik_radar(stock)

            st.markdown("<br>", unsafe_allow_html=True)
            render_lynch_radar(stock)

            st.markdown("<br>", unsafe_allow_html=True)
            render_mauboussin_radar(stock)

            st.markdown("<br>", unsafe_allow_html=True)
            render_mosl_wealth_matrix(stock)

        # ══ Tab D: Matrix & WCS ════════════════════════════════════════════
        with _itabs[3]:
            render_moat_growth_matrix(filt, highlight_stock=selected)
            st.markdown("<br>", unsafe_allow_html=True)
            render_ep_power_curve_module(stock)
            st.markdown("<br>", unsafe_allow_html=True)
            render_valuation_inversion_and_sizing_cockpit(stock)
            render_bruised_blue_chip_badge(stock)
            render_multitrillioncap_card(stock)

        # ══ Tab E: All Data ════════════════════════════════════════════════
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
            _stock_export = pd.DataFrame({
                "Signal": df[df["name"] == selected].iloc[0].index,
                "Value":  df[df["name"] == selected].iloc[0].values,
            })
            st.download_button(
                f"📥 Export {selected} — All Signals",
                data=_stock_export.to_csv(index=False),
                file_name=f"{selected.replace(' ','_').replace(':','').lower()}_signals.csv",
                mime="text/csv",
                use_container_width=True,
            )


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

    _bbc_mask = df.get("bruised_blue_chip_29", pd.Series(0, index=df.index)).fillna(0) == 1
    _bbc_cnt  = int(_bbc_mask.sum())

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
    st.markdown(f"<div class='sec-head'>⚙️ System Configuration — The Engine Rulebook</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='sec-cap'>Read-only view of the live, deterministic scoring weights and hard "
        f"gates every stock is measured against. To change them, edit <code>config.py</code> — the "
        f"single source of truth.</div>",
        unsafe_allow_html=True,
    )

    # ── Presentation helpers (pure display — no data mutation) ──────────────
    def _cfg_wbar(label: str, frac: float, color: str, note: str = "") -> str:
        """A labelled horizontal weight bar, clamped to [0,100]%."""
        w = max(0.0, min(100.0, float(frac) * 100.0))
        _note = (f'<span style="color:{COLORS["text_muted"]};font-weight:400;"> · {note}</span>'
                 if note else "")
        return (
            f'<div style="margin-bottom:9px;">'
            f'<div style="display:flex;justify-content:space-between;font-size:0.72rem;margin-bottom:3px;">'
            f'<span style="color:{COLORS["text_secondary"]};font-weight:600;">{label}{_note}</span>'
            f'<span style="color:{color};font-weight:800;">{frac*100:.0f}%</span></div>'
            f'<div style="background:{COLORS["bg_tertiary"]};border-radius:4px;height:6px;overflow:hidden;">'
            f'<div style="width:{w:.0f}%;height:6px;border-radius:4px;background:{color};"></div></div>'
            f'</div>'
        )

    def _cfg_card(title: str, icon: str, body_html: str, accent: str) -> str:
        return (
            f'<div style="background:{COLORS["bg_secondary"]};border:1px solid {COLORS["border"]};'
            f'border-left:3px solid {accent};border-radius:10px;padding:14px 16px;margin-bottom:12px;">'
            f'<div style="font-size:0.66rem;font-weight:800;color:{accent};text-transform:uppercase;'
            f'letter-spacing:1.2px;margin-bottom:10px;">{icon} &nbsp;{title}</div>{body_html}</div>'
        )

    _q_src = {"moat": "SQGLP", "growth": "SQGLP", "cash": "Coffee Can",
              "margin": "Fisher", "balance_sheet": "Baid", "valuation": "Marks+Baid"}
    _q_clr = {"moat": COLORS["purple"], "growth": COLORS["green"], "cash": COLORS["blue"],
              "margin": COLORS["orange"], "balance_sheet": COLORS["gold"], "valuation": COLORS["cyan"]}

    cc1, cc2 = st.columns(2)
    with cc1:
        _qbody = "".join(
            _cfg_wbar(k.replace("_", " ").title(), v, _q_clr.get(k, COLORS["blue"]), _q_src.get(k, ""))
            for k, v in QUALITY_WEIGHTS.items()
        )
        st.markdown(_cfg_card("Quality Sub-Weights · 6 Layers", "🏭", _qbody, COLORS["purple"]),
                    unsafe_allow_html=True)
    with cc2:
        _mbody = "".join(
            _cfg_wbar(k.replace("_", " ").title(), v, COLORS["orange"])
            for k, v in MOMENTUM_WEIGHTS.items()
        )
        _mbody += (
            f'<div style="border-top:1px solid {COLORS["border"]};margin-top:8px;padding-top:10px;">'
            + _cfg_wbar("Governance Blend (composite)", COMPOSITE_WEIGHTS["governance"], COLORS["gold"])
            + '</div>'
        )
        st.markdown(_cfg_card("Momentum Sub-Weights · CAN-SLIM", "⚡", _mbody, COLORS["orange"]),
                    unsafe_allow_html=True)

    # Hard gates — clean grid of pass-criteria chips
    _gate_cells = "".join(
        f'<div style="flex:1;min-width:210px;background:{COLORS["bg_tertiary"]};'
        f'border:1px solid {COLORS["border"]};border-radius:8px;padding:9px 12px;">'
        f'<span style="color:{COLORS["green"]};font-size:0.82rem;font-weight:800;">✓</span> '
        f'<span style="color:{COLORS["text_secondary"]};font-size:0.71rem;">{cfg["description"]}</span></div>'
        for _name, cfg in HARD_GATES.items()
    )
    st.markdown(
        _cfg_card(f"Hard Gates · {len(HARD_GATES)} Criteria — Every Stock Must Pass ALL", "🚨",
                  f'<div style="display:flex;gap:6px;flex-wrap:wrap;">{_gate_cells}</div>', COLORS["red"]),
        unsafe_allow_html=True,
    )

    # ── MARKS CYCLE TEMPERATURE GAUGE (standalone qualitative macro lens) ──
    # Deliberately display-only: a subjective human cycle read must NOT re-rank the deterministic,
    # vectorized quant scores. Macro adaptation is already handled objectively by detect_market_regime.
    st.markdown("---")
    st.markdown(f"<div class='sec-head'>🌡️ Marks Cycle Temperature Gauge</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='sec-cap'>Howard Marks' 5-dimension cycle read — score each 1 (cold/fear) "
        f"to 5 (hot/greed). A <strong>qualitative lens for your own conviction &amp; position "
        f"sizing</strong>; it deliberately does <strong>not</strong> alter the quant rankings, "
        f"which adapt objectively via the auto-detected market regime.</div>",
        unsafe_allow_html=True,
    )

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

    posture_color = ("#3fb950" if "Aggressive" in posture["label"]
                     else "#d29922" if "Neutral" in posture["label"] else "#f85149")
    # Marker position on the 5-25 scale → 0-100% (clamped so it can never overflow the bar)
    _marker_pct = max(0.0, min(100.0, (cycle_total - 5) / 20.0 * 100.0))

    st.markdown(f"""
    <div style="background:{COLORS['bg_secondary']};border:2px solid {posture_color};
                border-radius:14px;padding:18px 22px;margin:12px 0;">
      <div style="display:flex;align-items:center;justify-content:space-between;gap:16px;
                  flex-wrap:wrap;margin-bottom:16px;">
        <div>
          <div style="font-size:0.6rem;color:{COLORS['text_muted']};text-transform:uppercase;
                      letter-spacing:1.4px;font-weight:700;">Cycle Temperature</div>
          <div style="font-size:2.4rem;font-weight:900;color:{posture_color};line-height:1.05;">
            {cycle_total}<span style="font-size:1.1rem;color:{COLORS['text_muted']};
            font-weight:600;">/25</span></div>
        </div>
        <div style="text-align:right;flex:1;min-width:190px;">
          <div style="font-size:1.15rem;font-weight:800;color:{posture_color};">{posture['label']}</div>
          <div style="font-size:0.78rem;color:{COLORS['text_secondary']};margin-top:3px;">
            {posture['action']}</div>
        </div>
      </div>
      <div style="position:relative;height:12px;border-radius:6px;
                  background:linear-gradient(90deg,#3fb950 0%,#3fb950 25%,#e3b341 25%,
                  #e3b341 65%,#f85149 65%,#f85149 100%);">
        <div style="position:absolute;top:-6px;left:{_marker_pct:.1f}%;transform:translateX(-50%);
                    width:0;height:0;border-left:7px solid transparent;border-right:7px solid transparent;
                    border-top:10px solid {COLORS['text_primary']};"></div>
      </div>
      <div style="display:flex;justify-content:space-between;margin-top:10px;
                  font-size:0.58rem;font-weight:700;text-transform:uppercase;letter-spacing:0.4px;">
        <span style="color:#3fb950;">🟢 Aggressive · Deploy</span>
        <span style="color:#d29922;">🟡 Neutral · Hold</span>
        <span style="color:#f85149;">🔴 Defensive · Protect</span>
      </div>
    </div>
    <div style="font-size:0.66rem;color:{COLORS['text_muted']};margin:-2px 0 6px 2px;">
      🧠 A <strong>thinking tool</strong> — it shapes how aggressively <em>you</em> deploy capital.
      The engine's stock rankings stay 100% objective and untouched by these sliders.
    </div>
    """, unsafe_allow_html=True)

    # ── SYSTEM RISK MONITORS (Baid Sell Triggers + Mean Reversion) ──────────
    st.markdown("---")
    st.markdown(f"<div class='sec-head'>🛡️ System Risk Monitors</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='sec-cap'>Live, universe-wide risk counts computed by the engine this run.</div>",
        unsafe_allow_html=True,
    )
    _sell_cnt = int(df.get("sell_alert_any", pd.Series(0, dtype=int)).fillna(0).sum())
    _mr_cnt   = int(df.get("mean_reversion_risk", pd.Series(0, dtype=int)).fillna(0).sum())

    rm1, rm2 = st.columns(2)
    with rm1:
        _baid_clr  = COLORS["red"] if _sell_cnt else COLORS["green"]
        _baid_body = (
            f'<div style="font-size:1.7rem;font-weight:900;color:{_baid_clr};line-height:1;">'
            f'{_sell_cnt}<span style="font-size:0.7rem;color:{COLORS["text_muted"]};font-weight:600;">'
            f'&nbsp;stocks flagged</span></div>'
            f'<div style="margin-top:8px;">'
            + "".join(
                f'<div style="font-size:0.7rem;color:{COLORS["text_secondary"]};padding:3px 0;'
                f'border-bottom:1px solid rgba(255,255,255,0.04);">'
                f'<strong style="color:{COLORS["text_primary"]};">{n.replace("_"," ").title()}</strong> — '
                f'{c["description"]}</div>'
                for n, c in BAID_SELL_TRIGGERS.items()
            )
            + '</div>'
        )
        st.markdown(_cfg_card("Baid Sell Triggers", "📉", _baid_body, COLORS["red"]),
                    unsafe_allow_html=True)
    with rm2:
        _mr_clr  = COLORS["gold"] if _mr_cnt else COLORS["green"]
        _mr_body = (
            f'<div style="font-size:1.7rem;font-weight:900;color:{_mr_clr};line-height:1;">'
            f'{_mr_cnt}<span style="font-size:0.7rem;color:{COLORS["text_muted"]};font-weight:600;">'
            f'&nbsp;at cyclical peak</span></div>'
            f'<div style="font-size:0.72rem;color:{COLORS["text_secondary"]};margin-top:8px;">'
            f'OPM or NPM &gt; {MEAN_REVERSION["opm_spike_threshold"]}× their 5Y median — current '
            f'margins are likely unsustainable (Marks: extremes revert).</div>'
            f'<div style="font-size:0.72rem;color:{COLORS["text_muted"]};margin-top:8px;'
            f'border-top:1px solid {COLORS["border"]};padding-top:8px;">Quality-score penalty applied: '
            f'<strong style="color:{COLORS["gold"]};">−{(1-MEAN_REVERSION["penalty_factor"])*100:.0f}%</strong> '
            f'for each flagged stock.</div>'
        )
        st.markdown(_cfg_card("Mean Reversion Risk (Marks)", "🌡️", _mr_body, COLORS["gold"]),
                    unsafe_allow_html=True)

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
