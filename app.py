"""
PRISM — Quantamental Intelligence
=================================
Every lens. One verdict. — Regime-Aware, Master-Driven
Dr. Malik + Raamdeo Agrawal + O'Neil + Mukherjea + Marks + Fisher + Lynch
"""
import os
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

import html as _html

import streamlit as st


def _prism_favicon(size: int = 128):
    """Browser-tab favicon = the PRISM refracting-prism mark on a dark app-icon TILE (mirrors
    _PRISM_SVG in ui/ui_components.py). Drawn BOLD on a dark rounded background so it stays legible
    at 16px tab size — thin white strokes on transparency were invisible. PIL-only + inline so it
    runs BEFORE set_page_config without importing the (st-touching) ui package; page_icon takes a
    PIL.Image reliably (an SVG data-URI favicon is flaky across browsers). Strokes are sized as a %
    of the canvas so they survive the browser's downscale to 16/32px."""
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # Dark rounded tile — makes the mark pop on ANY browser tab (light or dark chrome).
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=round(size * 0.22),
                        fill=(13, 17, 23, 255), outline=(48, 54, 61, 255),
                        width=max(1, round(size * 0.015)))
    pad = size * 0.16
    s = (size - 2 * pad) / 72.0          # _PRISM_SVG viewBox is 72×56
    oy = (size - 56 * s) / 2.0
    P = lambda x, y: (pad + x * s, oy + y * s)
    WHITE = (230, 237, 243, 255)

    def _rline(p1, p2, rgb, w):          # round-capped line (emulates the SVG stroke-linecap)
        d.line([p1, p2], fill=rgb, width=w)
        r = w / 2.0
        for cx, cy in (p1, p2):
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rgb)

    tri = [P(28, 7), P(11, 47), P(45, 47)]
    d.polygon(tri, fill=WHITE)                                       # SOLID prism (legible at 16px)
    _rline(P(1, 29), P(20, 29), WHITE, max(2, round(size * 0.045)))  # bold incoming light beam
    ws = max(2, round(size * 0.058))                                 # bold refracted 5-axis spectrum
    for y, rgb in ((21, (163, 113, 247, 255)), (28, (63, 185, 80, 255)), (34, (88, 166, 255, 255)),
                   (40, (240, 136, 62, 255)), (47, (210, 153, 34, 255))):
        _rline(P(40, 33), P(70, y), rgb, ws)
    return img


st.set_page_config(page_title="PRISM — Quantamental Intelligence", page_icon=_prism_favicon(),
                   layout="wide", initial_sidebar_state="expanded")

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

from core import (fetch_and_clean_data, run_full_scoring, compute_forensic_signals,
                  apply_forensic_penalty, compute_verdict, run_scoring_pipeline)
from ui import (render_moat_growth_matrix, render_fisher_module,
                render_ep_power_curve_module, render_bruised_blue_chip_badge,
                render_multitrillioncap_card, render_forensic_perimeter, render_guru_frameworks,
                render_financial_insights, render_stock_hero, render_verdict_scorecard, render_score_strip,
                render_sell_alerts_panel, render_raw_signals,
                render_canslim_radar, render_sepa_radar, render_schilit_shield, render_dorsey_radar,
                render_outsider_radar, render_marks_radar, render_malik_radar,
                render_lynch_radar, render_mauboussin_radar, render_qglp_radar, render_mosl_wealth_matrix,
                render_piotroski_checklist,
                render_sector_peer_strip,
                render_trajectory_card,
                render_valuation_inversion_and_sizing_cockpit,
                inject_css, render_hero_banner, render_metric_strip, render_pulse_band,
                render_stock_card, help_chip,
                render_radar_chart, render_sidebar_brand,
                render_reference, render_concepts, render_flags, render_frameworks,
                build_reference_markdown)
from ui.ui_discovery import render_discovery_sidebar, clear_all_filters
from ui.ui_scanner import _SCANNER_HEADER_TIPS
from ui.ui_components import _RAW_GLOSSARY
from ui.ui_reference_data import CONCEPT_REFERENCE
from ui.ui_tearsheet import _FLAG_DISPLAY, _FW_META
from config import (COLORS, TIER_COLORS, CONVICTION_TIERS, UI, HARD_GATES,
                    QUALITY_WEIGHTS, MOMENTUM_WEIGHTS, COMPOSITE_WEIGHTS,
                    VALUATION_SIGNALS,
                    BAID_SELL_TRIGGERS, MEAN_REVERSION, PEG_ZONES,
                    MASTER_PROFILES, ANALYSIS_MODES, FORENSIC_MAX_FLAGS,
                    FORENSIC_PENALTY_TIERS, GOVERNANCE_RISK_MULTIPLIERS)


# ═══════════════════════════════════════════════════════════════
# 3-TIER CACHE SPLIT
# Tier 1: fetch_and_clean_data — CACHED. Only reruns on Clear Cache or new sheet.
# Tier 2: run_full_scoring     — NOT cached. Instant on dropdown change.
# Tier 3: run_forensic_analysis— NOT cached. Instant.
# ═══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def get_clean_data(data_source, file_signature: str, sheet_id, _uploaded_dict=None):
    """Tier-1: Expensive data fetch + clean. Heavily cached.
    file_signature (stable string: name+size per file) is NOT underscored, so Streamlit HASHES it —
    it is the real cache key that busts the cache when a different file is uploaded.
    _uploaded_dict IS underscored so Streamlit skips hashing the raw, unhashable stream objects.
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
                                    reassignment. MUST run last among scoring steps: composite_score
                                    only exists after step 2.
      4. compute_verdict          : Display-only decision-synthesis. Reads the POST-penalty
                                    composite_score / conviction_tier (consistent after step 3) +
                                    the 6 axes → verdict_direction / strength / narrative / risk.
                                    Adds ZERO scoring; only verdict_* label columns.
    """
    return run_scoring_pipeline(clean_df, analysis_mode, scoring_profile)

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
        # DEV CONVENIENCE: PRISM_SHEET_ID env var pre-fills the box so a local dev server boots
        # WITH data (no manual sidebar entry) → fast Playwright/visual-check loop. Unset in prod
        # (Streamlit Cloud never sets it) → identical behaviour to before. The legacy
        # STOCKSCAN_SHEET_ID is still honored (backward-compat) so existing dev/deploy envs keep working.
        _default_sheet = os.environ.get("PRISM_SHEET_ID") or os.environ.get("STOCKSCAN_SHEET_ID", "")
        sheet_id = st.text_input("Google Sheets URL or ID", value=_default_sheet,
                                 placeholder="Enter Google Sheet ID...")
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
# BRAND — compact strip at page top (identity → control → context)
# ═══════════════════════════════════════════════════════════════
render_hero_banner(compact=True)

# ═══════════════════════════════════════════════════════════════
# SCORING CONTROLS — two plain widgets, living in the ⚙️ Config tab
# ═══════════════════════════════════════════════════════════════
# The old Command Center (six mandate buttons + weights strip + Advanced Override) was REMOVED
# 2026-08-24 after measurement proved it a false promise: three of six mandates were ranking-
# identical (the profile feeds ONLY the QGLP screen — qglp_score/qglp_pass — never the
# composite), and the prominent Q/G/L/P weights strip implied engine re-weighting that never
# happened. The two REAL knobs remain as plain selectboxes in ⚙️ Config (widget-owned keys, no
# callbacks, no canonical/mirror dance — the machinery that caused the prod KeyError crash).
#
# Reading the widget keys HERE — before the Config tab renders them — is correct and current:
# Streamlit commits a changed widget's value to session_state BEFORE the rerun starts. Fresh
# cfg_* keys on purpose: resurrected sessions carrying the old adv_*/_w_* keys are ignored.
st.session_state.setdefault("cfg_mode", "Hybrid")
st.session_state.setdefault("cfg_profile", "Balanced")
# Snap the profile into the active mode's allowed set (a mode change can orphan the profile).
# Writing a widget's key before the widget instantiates is legal; the selectbox renders the value.
_allowed_profiles = ANALYSIS_MODES[st.session_state["cfg_mode"]]["allowed_profiles"]
if st.session_state["cfg_profile"] not in _allowed_profiles:
    st.session_state["cfg_profile"] = _allowed_profiles[0]

analysis_mode   = st.session_state["cfg_mode"]
scoring_profile = st.session_state["cfg_profile"]
profile_cfg = MASTER_PROFILES[scoring_profile]

# ── Scoring ────────────────────────────────────────────────────
_score_key = f"{file_sig}::{analysis_mode}::{scoring_profile}"
if st.session_state.get("_score_key") != _score_key or "_scored_df" not in st.session_state:
    with st.spinner(f"🧭 Scoring — {analysis_mode} / {scoring_profile}..."):
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

    # Scored-data export — placeholder here (prominent, in the data panel); FILLED after the filter
    # cascade runs (below) so it reflects the LIVE filter: it downloads exactly the stocks surviving
    # your sidebar filters (every column), or the whole universe when nothing is filtered. Distinct
    # from the Deep Scanner's curated (~40-col) export and the All-Data single-row export.
    from datetime import date as _date
    from ui.ui_export import scored_universe_csv
    _scored_dl_ph = st.empty()

    regime = df.attrs.get("detected_market_regime", "SIDEWAYS")
    regime_color = COLORS['green'] if regime == "BULL" else COLORS['red'] if regime == "BEAR" else COLORS['gold']
    st.markdown(f"""
    <div style="background:{COLORS['bg_tertiary']}; border-left:4px solid {regime_color}; padding:8px 12px; margin-bottom:15px; border-radius:4px;">
        <div style="font-size:0.75rem; color:{COLORS['text_muted']}; text-transform:uppercase; letter-spacing:1px;">Detected Regime</div>
        <div style="font-size:1.1rem; font-weight:800; color:{regime_color};">{regime} MARKET</div>
    </div>
    """, unsafe_allow_html=True)


# Discovery filter cascade — built in ui/ui_discovery.py (stateful counterpart to the
# stateless ui_tearsheet). Returns the fully-filtered frame the tabs render.
filt = render_discovery_sidebar(df)

# Fill the scored-data download now that the filtered frame exists. Filter-aware: exports the surviving
# rows (all columns), cached on a cheap (score + count + composite-sum) signature so it re-serializes
# ONLY when the filter actually changes — not on every rerun (the full 724-col frame is expensive to
# serialize). No filter active → the whole universe, exactly as before.
with _scored_dl_ph.container():
    _dl_sig = f"{_score_key}|{len(filt)}|{float(filt['composite_score'].sum()):.2f}"
    st.download_button(
        f"📥 Download {len(filt):,} stocks · all {df.shape[1]} cols",
        data=scored_universe_csv(_dl_sig, filt),
        file_name=f"prism_scored_{_date.today().isoformat()}_{len(filt)}stocks.csv",
        mime="text/csv",
        use_container_width=True,
        help="Downloads the CURRENTLY FILTERED stocks (every column) as Excel-safe CSV — reflects your "
             "sidebar filters (no filter = the full universe). For a curated column set, use the Deep "
             "Scanner's export.",
    )


# ═══════════════════════════════════════════════════════════════
# STATS STRIP (above tabs)
# ═══════════════════════════════════════════════════════════════
render_metric_strip([
    (f"{total}", "Universe", "m-blue"),
    (f"{gate_passed}", "Gate Passed", "m-green"),
    (f"{tier1}", "Crown Jewels", "m-gold"),
    (f"{tier2}", "Strong", "m-green"),
    (f"{tsunami_count}", "Tsunami", "m-purple"),
    (f"{avg_quality:.0f}", "Avg Quality", "m-blue"),
])

# (The Q/G/L/P weights strip was removed with the Command Center — those weights only ever
# drove the QGLP screen, not the composite; the screen line now lives beside its knobs in ⚙️ Config.)

# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════
tabs = st.tabs(["🏠 Discovery", "🔍 Deep Scanner", "🔬 The Tear-Sheet", "🌊 Market Pulse", "⚙️ Config", "📖 Reference"])

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

    # ── Control: how many cards to show ────────────────────────────
    # The "Sort by" pills (Quality / Momentum / PEG) were REMOVED 2026-08-24: they reordered the
    # cards by numbers the cards never display (only the composite is shown, in large type), which
    # also produced the confusing "#37 above #5" rank jumble. The Deep Scanner sorts by those
    # metrics AND shows each as a visible column — that is the view built for numeric comparison.
    # Discovery now always presents the engine's own ranking.
    _, _dc2 = st.columns([6, 2])
    with _dc2:
        _disc_n = st.selectbox(
            "Show", [10, 20, 30, 50], index=1, key="disc_n",
        )

    # LOAD-BEARING sort — not cosmetic. run_full_scoring sorts by the PRE-penalty composite and
    # resets the index; apply_forensic_penalty then multiplies composite_score and re-derives
    # `rank` WITHOUT re-sorting the frame. So `filt` arrives in stale pre-penalty ROW order while
    # its `rank` column is post-penalty (verified live 2026-08-24: rank is non-monotonic in frame
    # order, though it matches the composite ranking exactly). Dropping this sort would render
    # cards in an order contradicting their own #rank badge. Pinned by tests/test_page_order.py.
    _disc_df = (filt.sort_values("composite_score", ascending=False)
                if "composite_score" in filt.columns else filt.copy())
    _shown_n = int(_disc_n or 20)

    # ── No-match dead-end → actionable empty-state (filters can narrow to zero; the engine and
    # the non-empty path below are untouched — this only ADDS the empty branch) ──
    if _disc_df.empty:
        # Name the filter that emptied the list — the sidebar cascade records it (the first filter
        # to take a non-empty frame to zero) and publishes it on filt.attrs. Read from `filt`, not
        # `_disc_df`: attrs need not survive sort_values, but `filt` is the object it was set on.
        _culprit = str(filt.attrs.get("zero_culprit", "") or "")
        _culprit_line = (
            f'<div style="font-size:0.75rem;color:{COLORS["red"]};margin-top:8px;">'
            f'⚠️ <strong>{_html.escape(_culprit)}</strong> removed the last stocks — loosen that one first.'
            f'</div>'
        ) if _culprit else ""
        st.markdown(
            f'<div style="text-align:center;background:{COLORS["bg_secondary"]};'
            f'border:1px dashed {COLORS["border"]};border-radius:12px;padding:28px 18px;margin-top:6px;">'
            f'<div style="font-size:1.4rem;margin-bottom:6px;">🔍</div>'
            f'<div style="font-size:0.95rem;font-weight:800;color:{COLORS["text_primary"]};">'
            f'No stocks match these filters</div>'
            f'<div style="font-size:0.72rem;color:{COLORS["text_muted"]};margin-top:4px;">'
            f'Your active filters narrowed all {len(df):,} stocks out. Loosen one — or clear everything '
            f'and start fresh.</div>{_culprit_line}</div>',
            unsafe_allow_html=True,
        )
        _, _ec, _ = st.columns([3, 2, 3])
        with _ec:
            st.button("🧹 Clear all filters", key="disc_clear", use_container_width=True,
                      on_click=clear_all_filters)
    else:
        st.markdown(
            f'<div class="sec-head">🏆 Top Picks — {len(_disc_df)} stocks</div>',
            unsafe_allow_html=True,
        )

        # One-time legend for the cards' sub-score bars — explained ONCE here (the scan-friendly
        # alternative to repeating ~100 identical "?" chips, one on every card). Reuses the shared
        # glossary via help_chip, so these definitions never drift from the tearsheet's.
        _SS_LABELS = ("Moat", "Growth", "Cash", "Momentum", "Governance")
        _ss_legend = " &nbsp;·&nbsp; ".join(_l + help_chip(_l + " Score") for _l in _SS_LABELS)
        st.markdown(
            f'<div style="font-size:0.62rem;color:{COLORS["text_muted"]};margin:0 0 10px 2px;">'
            f'Card score bars &nbsp;—&nbsp; {_ss_legend}</div>',
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
        "🏆 Core":      ["rank","name","verdict_direction","wealth_tier","sector","market_category","composite_score",
                         "data_coverage_pct","conviction_tier","gate_pass","moat_growth_quad","smart_money_flow"],
        "📊 Quality":   ["name","quality_score","moat_score","growth_score","cash_score",
                         "governance_bonus","piotroski_fscore","roce","opm","cfo_to_pat"],
        "💰 Valuation": ["name","close_price","fair_value_qglp","valuation_score","expected_excess_return",
                         "pe","pb_ratio","peg","earnings_yield","fcf_yield","market_cap","buy_zone_label"],
        "🔬 Forensic":  ["name","red_flag_count","red_flag_list","piotroski_fscore","forensic_score",
                         "forensic_multiplier","cfo_to_pat","accruals_ratio","debt_to_equity",
                         "promoter_holdings","pledged_percentage"],
        "📈 Technical": ["name","close_price","dist_to_vstop","momentum_score","rsi_14d","dist_52wh",
                         "crs_52w","weinstein_stage","breakout_score","smart_money_flow","tsunami_signal","vstop_green"],
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
    _ds_c1, _ds_c2, _ds_c3 = st.columns([1.5, 5.5, 2])
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
        "breakout_score": "Breakout", "valuation_score": "Valuation",
    }.items():
        if _sc in _display_df.columns:
            _CC[_sc] = st.column_config.ProgressColumn(
                _sl, help=_SCANNER_HEADER_TIPS.get(_sc), min_value=0, max_value=100, format="%.0f")
    for _bc in ("gate_pass", "tsunami_signal", "vstop_green"):
        if _bc in _display_df.columns:
            _lbl = {"gate_pass": "✅ Gate", "tsunami_signal": "🌊", "vstop_green": "VSTOP"}[_bc]
            _CC[_bc] = st.column_config.CheckboxColumn(_lbl, help=_SCANNER_HEADER_TIPS.get(_bc))
    _num_fmt = {
        "conviction_tier": ("Tier",     "T%.0f"),
        "piotroski_fscore":("F-Score",  "%.0f/9"),
        "peg":             ("PEG",      "%.2f×"),
        "pe":              ("P/E",      "%.1f×"),
        "pb_ratio":        ("P/B",      "%.1f×"),
        "cfo_to_pat":      ("CFO/PAT",  "%.0f%%"),
        "opm":             ("OPM",      "%.1f%%"),
        "roce":            ("ROCE",     "%.1f%%"),
        "debt_to_equity":  ("D/E",      "%.2f"),
        "promoter_holdings":("Promoter","%.1f%%"),
        "pledged_percentage":("Pledged","%.1f%%"),
        "rsi_14d":         ("RSI",      "%.0f"),
        "dist_52wh":       ("52WH Δ",  "%.1f%%"),
        "earnings_yield":  ("E.Yield",  "%.1f%%"),
        "fcf_yield":       ("FCF Yld",  "%.1f%%"),
        "market_cap":      ("MCap ₹Cr", "%.0f"),
        "rank":            ("Rank",     "%.0f"),
        "red_flag_count":  ("Red Flags","%.0f"),
        "accruals_ratio":  ("Accruals", "%.2f"),
        "crs_52w":         ("RS 52W",   "%.0f"),
        "expected_excess_return": ("Edge %", "%.1f%%"),
        "close_price":     ("Price ₹",  "%.2f"),     # Valuation+Technical: the number every ₹ column is measured against
        "fair_value_qglp": ("Fair ₹",   "%.0f"),     # Valuation: QGLP fair PE × EPS (blank = loss-maker, undefined)
        "dist_to_vstop":   ("Stop Δ",   "%.1f%%"),   # Technical: % above(+)/below(−) the Volatility Stop
        "data_coverage_pct":      ("Evidence",   "%.0f%%"),   # Core: score-confidence % (high score on thin data = trap)
        "forensic_multiplier":    ("Forensic ×", "%.2f"),     # Forensic: the penalty cutting composite (1.00 clean → 0.50 high-risk)
    }
    for _nc, (_nl, _nf) in _num_fmt.items():
        if _nc in _display_df.columns:
            _CC[_nc] = st.column_config.NumberColumn(_nl, help=_SCANNER_HEADER_TIPS.get(_nc), format=_nf)
    # String decision-signal + identity columns get clean headers (else they show raw snake_case).
    for _tc, _tl in {
        "name": "Stock", "sector": "Sector", "market_category": "Market Cap",
        "verdict_direction": "Soundness", "wealth_tier": "Wealth", "weinstein_stage": "Trend",
        "moat_growth_quad": "Moat·Growth", "smart_money_flow": "Smart Money",
        "buy_zone_label": "Buy Zone",
        "red_flag_list": "Which Flags",
    }.items():
        if _tc in _display_df.columns:
            _CC[_tc] = st.column_config.TextColumn(_tl, help=_SCANNER_HEADER_TIPS.get(_tc))
    # Safety net: a future _DS_VIEWS column with a tip but no typed config above still gets its
    # hover tooltip (raw header). NOTE: Streamlit issue #10841 — header tooltips don't render in
    # the dataframe's FULL-SCREEN mode; they work in the normal embedded view.
    for _col in _display_df.columns:
        if _col not in _CC and _SCANNER_HEADER_TIPS.get(_col):
            _CC[_col] = st.column_config.Column(help=_SCANNER_HEADER_TIPS[_col])

    # ── Render table — or a smart, cause-specific empty-state ──────
    if filt.empty:
        # Sidebar filters narrowed everything out → the fix is Clear all filters.
        st.markdown(
            f'<div style="text-align:center;background:{COLORS["bg_secondary"]};'
            f'border:1px dashed {COLORS["border"]};border-radius:12px;padding:26px 18px;margin-top:6px;">'
            f'<div style="font-size:1.3rem;margin-bottom:6px;">🔍</div>'
            f'<div style="font-size:0.95rem;font-weight:800;color:{COLORS["text_primary"]};">'
            f'No stocks match your filters</div>'
            f'<div style="font-size:0.72rem;color:{COLORS["text_muted"]};margin-top:4px;">'
            f'Your sidebar filters narrowed all {len(df):,} stocks out — loosen them or clear everything.'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        _, _ec, _ = st.columns([3, 2, 3])
        with _ec:
            st.button("🧹 Clear all filters", key="ds_clear", use_container_width=True,
                      on_click=clear_all_filters)
    elif ds_df.empty:
        # Filters DO match stocks; the search box killed them → clear the search, not the filters.
        st.info(f"🔍 No stock matches “{ds_search}” among the {len(filt):,} filtered stocks — "
                f"clear the search box above to see them all.")
    else:
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

        # ── Export — the CURATED columns (the deduped union of all 5 view presets, ~40 meaningful
        # cols) instead of the ~500 raw internal columns (rf_/cat_/vqs_/proxies). Rows are the
        # searched/sorted ds_df; the column set is auto-derived from _DS_VIEWS so it never drifts,
        # and it's ~10x smaller to serialize on every rerun. ──
        _export_cols = [c for c in dict.fromkeys(_c for _v in _DS_VIEWS.values() for _c in _v)
                        if c in ds_df.columns]
        _safe_mode = analysis_mode.replace(" ", "_").lower()
        # Encode via the shared _to_csv_bytes (UTF-8-with-BOM) — the SAME Excel-safe path the sidebar
        # full-dump uses — so the export's emoji decision-columns (moat_growth_quad ⭐💀, smart_money_flow
        # ⚪✅❌, weinstein_stage, buy_zone_label) render in Excel instead of mojibaking on a BOM-less file.
        from ui.ui_export import _to_csv_bytes
        st.download_button(
            f"📥 Export {len(ds_df)} stocks · {len(_export_cols)} columns — {analysis_mode} / {scoring_profile}",
            data=_to_csv_bytes(ds_df[_export_cols]),
            file_name=f"scan_{_safe_mode}_{scoring_profile.lower()}.csv",
            mime="text/csv",
            use_container_width=True,
        )


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
        # Cross-tab handoff: tabs that render AFTER this one (Market Pulse) cannot assign the
        # xray_stock widget key directly — Streamlit raises StreamlitAPIException (set-after-
        # instantiation). They stage a transient _pending_xray + st.rerun() instead; consume it
        # HERE, before the selectbox below is instantiated, so its index reflects the jumped stock.
        if "_pending_xray" in st.session_state:
            st.session_state["xray_stock"] = st.session_state.pop("_pending_xray")
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

        # ── Verdict header: reads the pre-computed verdict_* columns (core/verdict_engine.py) ──
        # Hard overrides (gate fail / sell alert) take precedence; otherwise the engine's veto-aware
        # verdict drives the band. No verdict logic is computed here — single source of truth is the engine.
        _vdir  = str(_sg("verdict_direction", "FLAWED") or "FLAWED")
        # verdict_strength is deliberately NOT read here (removed 2026-08-27). It is a measured
        # 1:1 rename of conviction_tier, and the hero already carries the tier badge and the
        # score — so "🟡 MIXED · HIGH CONVICTION · Score 90/100" stated the same fact three ways
        # on one screen. The COLUMN still exists (snapshot-schema stability, orphan principle:
        # an unsurfaced column harms nobody); only the display is retired. Pinned by
        # tests/test_verdict_vocabulary.py.
        _vconf = str(stock.get("verdict_confidence", "") or "")
        _vnarr = str(stock.get("verdict_narrative", "") or "")
        _vrisk = str(stock.get("verdict_top_risk", "") or "")
        _vemoji = str(stock.get("verdict_emoji", "") or "")

        if not _gate_ok:
            _verdict, _verdict_clr, _verdict_bg = "SYSTEM REJECTED", COLORS["red"], "rgba(248,81,73,0.09)"
            _verdict_reason = f"Hard Gate Failure — {stock.get('failed_gates', 'Unknown')}"
        elif _sell_any:
            _verdict, _verdict_clr, _verdict_bg = "SELL ALERT", COLORS["red"], "rgba(248,81,73,0.07)"
            _verdict_reason = "One or more Baid sell triggers have fired — review Forensics tab."
        else:
            _dir_map = {
                "SOUND":  (COLORS["green"],      "rgba(63,185,80,0.08)"),
                "MIXED":  (COLORS["gold"],       "rgba(228,179,65,0.07)"),
                "FLAWED": (COLORS["text_muted"], "rgba(110,118,129,0.06)"),
            }
            _verdict_clr, _verdict_bg = _dir_map.get(_vdir, _dir_map["FLAWED"])
            _verdict = f"{_vemoji} {_vdir}".strip()
            _verdict_reason = _vnarr or f"Tier {_tier_num} · Score {_comp_sc:.0f}/100"

        # Score · Confidence subline (engine path only)
        _meta_bits = []
        if _gate_ok and not _sell_any:
            _meta_bits.append(f"Score {_comp_sc:.0f}/100")
            if _vconf:
                _meta_bits.append(f"🔍 {_vconf} data")
        _meta_line = " · ".join(_meta_bits)

        _pill_css = ("font-size:0.67rem;font-weight:700;padding:2px 10px;border-radius:12px;"
                     "white-space:nowrap;")
        _risk_pill = (
            f'<span style="{_pill_css}background:rgba(248,81,73,0.13);color:{COLORS["red"]};'
            f'border:1px solid rgba(248,81,73,0.4);">{_vrisk}</span>'
        ) if (_vrisk and _gate_ok and not _sell_any) else ""
        _mr_pill = (
            f'<span style="{_pill_css}background:rgba(228,179,65,0.15);color:{COLORS["gold"]};'
            f'border:1px solid rgba(228,179,65,0.4);">⚠️ Mean Reversion</span>'
        ) if _mr_risk else ""

        # ── WHAT-vs-WHEN reconciliation: the verdict is a FUNDAMENTAL call (own this business?);
        # Weinstein stage is the TECHNICAL trend (is the trend with you?). They're orthogonal and
        # can disagree — a BUY/WATCH on a stock below its falling 30-week MA (Stage 3/4) is a
        # watchlist candidate, not a buy-now. Surface that tension (display only — the verdict
        # engine is untouched; this never changes the direction). Fires only on real conflict.
        _wstage = str(stock.get("weinstein_stage", "") or "")
        _trend_conflict = (_gate_ok and not _sell_any and _vdir in ("SOUND", "MIXED")
                           and ("Stage 4" in _wstage or "Stage 3" in _wstage))
        _trend_pill = (
            f'<span style="{_pill_css}background:rgba(228,179,65,0.15);color:{COLORS["gold"]};'
            f'border:1px solid rgba(228,179,65,0.4);">⚠️ Against 30-wk trend</span>'
        ) if _trend_conflict else ""
        if _trend_conflict and "Stage 4" in _wstage:
            _trend_msg = ("📉 Strong business, weak trend — price is below a falling 30-week MA "
                          "(Stage 4). A watchlist candidate; wait for a Stage-2 base before buying.")
        elif _trend_conflict:  # Stage 3 Top
            _trend_msg = ("⚠️ Strong business, topping trend — price has slipped below its 30-week MA "
                          "(Stage 3). Don't chase; wait for the trend to reset.")
        else:
            _trend_msg = ""
        _trend_action = (
            f'<div style="font-size:0.72rem;color:{COLORS["gold"]};margin-top:6px;line-height:1.4;'
            f'background:rgba(228,179,65,0.08);border:1px solid rgba(228,179,65,0.25);'
            f'border-radius:7px;padding:6px 10px;">{_trend_msg}</div>'
        ) if _trend_conflict else ""

        # ── 💹 Wealth-tier pill: the third layer of the grammar, on the reading surface ──
        # A LABELED PILL, never a competing banner — the band is Soundness's home. Unlocked by
        # the vocabulary rename: "FLAWED · 💹 WATCH★" reads as two layers disagreeing openly
        # (the feature), where the old "AVOID · BUY★" read as a contradiction (the cf_triangle
        # defect). ⚠ rides with the tier per the wealth_warn contract — never blended.
        _wtier = str(stock.get("wealth_tier", "") or "")
        _wwarn = " ⚠" if int(stock.get("wealth_warn", 0) or 0) == 1 else ""
        _WT_PILL_CLR = {
            "BUY★":   ("rgba(63,185,80,0.13)",   COLORS["green"], "rgba(63,185,80,0.4)"),
            "BUY":    ("rgba(63,185,80,0.10)",   COLORS["green"], "rgba(63,185,80,0.3)"),
            "WATCH★": ("rgba(228,179,65,0.15)",  COLORS["gold"],  "rgba(228,179,65,0.4)"),
            "WATCH":  ("rgba(228,179,65,0.10)",  COLORS["gold"],  "rgba(228,179,65,0.3)"),
            "AVOID":  ("rgba(110,118,129,0.12)", COLORS["text_secondary"], "rgba(110,118,129,0.35)"),
        }
        _wealth_pill = ""
        if _wtier in _WT_PILL_CLR:          # N/A and missing render nothing — no pill over no data
            _bg, _fg, _bd = _WT_PILL_CLR[_wtier]
            _wealth_pill = (
                f'<span style="{_pill_css}background:{_bg};color:{_fg};'
                f'border:1px solid {_bd};">💹 {_wtier}{_wwarn}</span>'
            )

        st.markdown(f"""
        <div style="background:{_verdict_bg};border:1px solid {_verdict_clr}55;
             border-left:4px solid {_verdict_clr};border-radius:10px;
             padding:11px 16px;margin:6px 0 10px 0;">
          <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
            <span style="font-size:0.85rem;font-weight:900;color:{_verdict_clr};
                 letter-spacing:1.1px;white-space:nowrap;">{_verdict}</span>
            <span style="font-size:0.7rem;color:{COLORS['text_secondary']};
                 white-space:nowrap;">{_meta_line}</span>
            {_wealth_pill}{_risk_pill}{_mr_pill}{_trend_pill}
          </div>
          <div style="font-size:0.75rem;color:{COLORS['text_secondary']};margin-top:5px;">
            {_verdict_reason}</div>
          {_trend_action}
        </div>
        """, unsafe_allow_html=True)

        # ── Verdict scorecard: the 6-axis evidence grid (Layer 2, directly under the verdict) ──
        render_verdict_scorecard(stock)

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
        # Visual quality profile (radar) + signal badges → the deep financial breakdown.
        # The old 7-KPI buy-checklist was REMOVED (2026-06-14): every one of its metrics is shown,
        # with more depth, in the Business & Financial Analysis below, and its tier/score header
        # duplicated the verdict header above. The verdict + 6-axis scorecard are now the at-a-glance.
        with _itabs[0]:
            _ov1, _ov2 = st.columns([3, 2])

            with _ov1:
                fig = render_radar_chart(stock, f"{selected} — Quality Radar")
                st.plotly_chart(fig, use_container_width=True)

            with _ov2:
                # Quality facets — the radar's LEGEND (the polygon shows shape; these are the exact
                # scores). Cash + Margin are unique here — the orthogonal scorecard omits them.
                def _qfrow(lbl, key):
                    sc = _sg(key, None)
                    if sc is None:
                        clr, vs = COLORS["text_muted"], "—"
                    else:
                        sc = float(sc)
                        clr = (COLORS["green"] if sc >= 60 else
                               COLORS["gold"]  if sc >= 40 else COLORS["red"])
                        vs = f"{sc:.0f}"
                    return (
                        f'<div style="display:flex;justify-content:space-between;align-items:center;'
                        f'padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.04);">'
                        f'<span style="font-size:0.72rem;color:{COLORS["text_secondary"]};">{lbl}</span>'
                        f'<span style="font-size:0.82rem;font-weight:800;color:{clr};">{vs}'
                        f'<span style="font-size:0.6rem;color:{COLORS["text_muted"]};">/100</span></span></div>'
                    )
                _facets = (
                    _qfrow("🛡️ Moat",   "moat_score")          +
                    _qfrow("📈 Growth", "growth_score")        +
                    _qfrow("💰 Cash",   "cash_score")          +
                    _qfrow("📊 Margin", "margin_score")        +
                    _qfrow("⚖️ Balance","balance_sheet_score")
                )
                st.markdown(
                    f'<div style="font-size:0.62rem;font-weight:800;color:{COLORS["text_muted"]};'
                    f'text-transform:uppercase;letter-spacing:0.8px;margin:2px 0 4px 0;">Quality Facets</div>'
                    f'{_facets}',
                    unsafe_allow_html=True,
                )

                # Signal badges
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
                st.markdown(
                    f'<div style="font-size:0.62rem;font-weight:800;color:{COLORS["text_muted"]};'
                    f'text-transform:uppercase;letter-spacing:0.8px;margin:13px 0 8px 0;">Signals</div>'
                    f'{bdgs}',
                    unsafe_allow_html=True,
                )

            # vs Sector Peers — contextualizes the at-a-glance quality (radar + facets) against
            # the stock's OWN sector before the absolute financials below: the value-trap guard
            # (a high absolute score that is bottom-quartile for its sector, or vice-versa).
            render_sector_peer_strip(stock)

            # Trajectory — the second derivative, and the one question the rest of Overview does
            # not ask: not where this business stands, but which way it is moving and whether the
            # move is speeding up. Sits after the peer strip (absolute -> relative -> directional)
            # and before the detailed financials it summarises.
            render_trajectory_card(stock)

            st.markdown(
                f"<div class='sec-head'>📊 Business & Financial Analysis</div>",
                unsafe_allow_html=True,
            )
            render_financial_insights(stock)

        # ══ Tab B: Forensics ═══════════════════════════════════════════════
        with _itabs[1]:
            # The Fraud Perimeter renders its own richer KPI row (Red Flags · Forensic Score ·
            # Score Multiplier · Piotroski · Mgmt Integrity); a separate strip here just duplicated
            # F-Score/Red Flags/Forensic. CF Triangle still shows in the Overview "Signals" strip.
            st.markdown(
                f"<div class='sec-head'>🔬 Forensic Fraud Perimeter ({FORENSIC_MAX_FLAGS}-Flag Cascade)</div>",
                unsafe_allow_html=True,
            )
            render_forensic_perimeter(stock)

            # The F-Score is already shown as a NUMBER in the perimeter KPI row; this is the
            # checklist behind it. All nine components are computed at 100% coverage and none of
            # them reached the screen until now. Ordered weakest-to-strongest evidence:
            # red flags -> Piotroski -> Fisher.
            st.markdown("<br>", unsafe_allow_html=True)
            render_piotroski_checklist(stock)

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
                f"<div class='sec-head'>🏛️ Guru Framework Alignment — {len(_FW_META)} Frameworks</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='sec-cap'>Pre-computed framework badges from scoring engine. "
                f"Each represents a complete quantamental screen from a master investor's methodology.</div>",
                unsafe_allow_html=True,
            )
            render_guru_frameworks(stock)

            # Deep-dive guru radars — collapsed by default (Layer 3 evidence). The verdict header,
            # 6-axis scorecard and categorized frameworks above already SUMMARIZE these same
            # dimensions; expand a radar only to audit one methodology's detail. Nothing removed —
            # just decluttered (9 stacked radars → 9 collapsed expanders). Calls kept explicit so the
            # app-wiring contract tests (canslim→sepa→dorsey order) still hold.
            st.markdown(
                "<div class='sec-cap' style='margin-top:16px;'>🔬 Deep-dive radars — expand to audit "
                "a specific methodology (its signals are already summarized in the scorecard above).</div>",
                unsafe_allow_html=True,
            )
            with st.expander("👑 QGLP — Raamdeo's Process (Q·G·L·P)", expanded=False):
                render_qglp_radar(stock, scoring_profile)
            with st.expander("📊 CAN SLIM — Tactical Momentum (O'Neil)", expanded=False):
                render_canslim_radar(stock)
            with st.expander("⚡ Minervini SEPA — Momentum & VCP", expanded=False):
                render_sepa_radar(stock)
            with st.expander("🌊 Dorsey — Wide-Moat Pillars", expanded=False):
                render_dorsey_radar(stock)
            with st.expander("🎯 Outsider CEO — Capital Allocation", expanded=False):
                render_outsider_radar(stock)
            with st.expander("🛡️ Marks — Cycle Position", expanded=False):
                render_marks_radar(stock)
            with st.expander("📚 Malik — Quality Checklist", expanded=False):
                render_malik_radar(stock)
            with st.expander("👓 Lynch — Category & PEG", expanded=False):
                render_lynch_radar(stock)
            with st.expander("🔮 Mauboussin — Expectations & Payoff", expanded=False):
                render_mauboussin_radar(stock)
            with st.expander("🏛️ MOSL Wealth-Creation Matrix", expanded=False):
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
                f"<div class='sec-cap'>Every final, decision-grade signal the engine computes, grouped by "
                f"category — intermediate working columns are omitted here (the Export below carries the "
                f"complete machine-readable row). Engine-computed; nothing is re-calculated on this tab.</div>",
                unsafe_allow_html=True,
            )
            # The search box lives HERE, not in ui_tearsheet: that module is bound by the
            # stateless contract (app.py owns session_state). Same split as ds_search/ref_search.
            _ad_q = st.text_input(
                "Filter signals", value="", key="ad_search", label_visibility="collapsed",
                placeholder="🔎 Filter signals — by name (roce) or by meaning (cost of equity)…",
                help="Matches the signal's label AND its description, so you can search for what a "
                     "number MEANS, not only what it is called. Every word must appear. Blank = all.",
            )
            render_raw_signals(stock, query=_ad_q)
            # Breathing room before the Export so it doesn't crowd the last data section.
            st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)
            # `stock` IS df[df["name"] == selected].iloc[0] (assigned once at the top of this tab) and
            # is the very row the grid above just rendered. Reuse it instead of re-running the same
            # lookup twice more: one derivation cannot drift from what the tab displayed, two can.
            _stock_export = pd.DataFrame({"Signal": stock.index, "Value": stock.values})
            # Excel-safe UTF-8-with-BOM encode (the SAME path as the Deep Scanner + sidebar exports) —
            # this row's Value column is full of emoji decision-strings (corporate_class 🏆, smart_money
            # ⚪/✅/❌, weinstein_stage, verdict emojis) + Indian names that mojibake under a bare to_csv.
            from ui.ui_export import _to_csv_bytes
            st.download_button(
                f"📥 Export {selected} — full row · all {df.shape[1]} signals",
                data=_to_csv_bytes(_stock_export),
                file_name=f"{re.sub(r'[^A-Za-z0-9._-]+', '_', selected).lower()}_signals.csv",
                mime="text/csv",
                use_container_width=True,
                # Both sibling data exports state their column count in the label; this one said only
                # "(all columns)". The help names the one thing measurement showed a user WILL hit:
                # the CSV is keyed by ENGINE column name, and NONE of the 154 display labels the grid
                # above just taught them appear in it (0 of 154 — verified on live data).
                help=f"The complete machine-readable row — all {df.shape[1]} engine columns, including "
                     f"the intermediate working columns the grid above omits. Rows are keyed by engine "
                     f"column name (roce_med_10y), NOT the display labels shown above (ROCE 10Y Med). "
                     f"Excel-safe UTF-8. The filter box does not narrow this export.",
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4: MARKET PULSE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[3]:

    # ── Pre-compute section datasets ───────────────────────────────
    _mp_ts   = (df[df["tsunami_signal"] == 1].sort_values("composite_score", ascending=False)
                if "tsunami_signal" in df.columns else df.iloc[:0])
    _mp_qglp = (df[df["qglp_pass"] == 1].sort_values("qglp_score", ascending=False)
                if "qglp_pass" in df.columns else df.iloc[:0])   # market-wide, like the other 4 sections

    # ── Market-state Pulse band (breadth-led market vitals — what the tab's name promises) ──────
    render_pulse_band(df)

    # ── Inner navigation tabs ──────────────────────────────────────
    _mp_tabs = st.tabs([
        "🌊 Tsunami",
        "🏛️ QGLP",
        "🔭 MOSL",
        "💹 Wealth",
        "📈 Sectors",
        "🏭 Industry",
    ])   # Stage 3: dropped dead "💙 Blue Chips" (0% fires) + brittle "🚀 Tipping Points" (folded into Sectors)
         # 🏭 Industry APPENDED 2026-08-28 — appended, never inserted: each `with _mp_tabs[i]` body
         # binds by index, so inserting anywhere earlier renders existing content into a new tab.

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

            # Same ordering fix as QGLP below: 12 columns, ~8 fit. The Tsunami claim is that all 7
            # conviction conditions fire at once, so the EVIDENCE for that (scores, F-score, entry
            # zone) leads and the wide context columns follow.
            _ts_cols = [c for c in ["rank","name","verdict_direction",
                                    "composite_score","quality_score","momentum_score",
                                    "piotroski_fscore","buy_zone_label",
                                    "sector","market_category","market_cap","smart_money_flow"]
                        if c in _mp_ts.columns]
            _ts_sel = st.dataframe(
                _mp_ts[_ts_cols].reset_index(drop=True),
                column_config={
                    "verdict_direction": st.column_config.TextColumn("Soundness", help="The engine's overall SOUND / MIXED / FLAWED gate — a Tsunami setup can still be MIXED/FLAWED on valuation or entry timing."),
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
                # Stage a transient key + rerun (NOT a direct widget-key set — this tab renders
                # after the Tear-Sheet selectbox). The change-guard is essential: st.dataframe's
                # selection persists across reruns, so an unguarded set+rerun would loop forever.
                if _ts_pick != st.session_state.get("xray_stock"):
                    st.session_state["_pending_xray"] = _ts_pick
                    st.rerun()
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
            "Promoter>50%, reasonable valuation. Strict gates. Market-wide (ignores sidebar filters).</div>",
            unsafe_allow_html=True,
        )
        if len(_mp_qglp) == 0:
            st.info("No stocks currently pass the strict QGLP gates.")
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

            # COLUMN ORDER MATTERS HERE, and it is the fix for a real defect (2026-08-27): 13
            # columns are defined but only ~8 fit the container, and `sector` — the widest column
            # in the frame ("Infrastructure Developers & Operators") — sat at position 5. It shoved
            # qglp_price, the "P" in QGLP, off-screen entirely. A tab showcasing a four-leg
            # framework was showing one and a half legs.
            # Nothing is REMOVED (the table scrolls, so every column is still reachable) — the
            # framework's own components simply come before the context columns now.
            _q_cols = [c for c in ["rank","name","verdict_direction","red_flag_count",
                                   "qglp_score","qglp_quality","qglp_growth","qglp_longevity","qglp_price",
                                   "sector","market_cap","smart_money_flow","buy_zone_label"]
                       if c in _mp_qglp.columns]
            _q_sel = st.dataframe(
                _mp_qglp[_q_cols].reset_index(drop=True),
                column_config={
                    "verdict_direction": st.column_config.TextColumn("Soundness", help="The engine's overall SOUND / MIXED / FLAWED gate — most QGLP passers are MIXED/FLAWED on valuation, so this surfaces the few that are buyable now."),
                    # width="small" on the five legs + the name column: reordering alone left
                    # Longevity and Price/PEG off-screen at a 1793px viewport (verified in the
                    # browser). The legs need room for a bar and 2-3 digits, nothing more, and
                    # `name` is the widest text column in the frame.
                    "name":           st.column_config.TextColumn("name", width="medium"),
                    "qglp_score":     st.column_config.ProgressColumn("QGLP",      min_value=0, max_value=100, format="%.0f", width="small"),
                    "qglp_quality":   st.column_config.ProgressColumn("Quality",   min_value=0, max_value=100, format="%.0f", width="small"),
                    "qglp_growth":    st.column_config.ProgressColumn("Growth",    min_value=0, max_value=100, format="%.0f", width="small"),
                    "qglp_longevity": st.column_config.ProgressColumn("Longevity", min_value=0, max_value=100, format="%.0f", width="small"),
                    "qglp_price":     st.column_config.ProgressColumn("Price/PEG", min_value=0, max_value=100, format="%.0f", width="small"),
                    "red_flag_count": st.column_config.NumberColumn("🚩 Flags",    format="%.0f", help="Forensic red flags raised (0 = clean). QGLP gates on quality/growth, NOT forensics — so this is the risk check the screen itself doesn't do."),
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
                # Transient key + rerun + change-guard (see Tsunami above — same set-after-widget rule).
                if _q_pick != st.session_state.get("xray_stock"):
                    st.session_state["_pending_xray"] = _q_pick
                    st.rerun()
                st.markdown(f"""
                <div style="padding:9px 14px;margin-top:8px;background:rgba(228,179,65,0.07);
                     border:1px solid rgba(228,179,65,0.3);border-radius:8px;font-size:0.8rem;">
                  🔬 <strong style="color:{COLORS['text_primary']};">{_q_pick}</strong>
                  set — <strong style="color:{COLORS['blue']};">click The Tear-Sheet tab</strong> for full analysis.
                </div>
                """, unsafe_allow_html=True)

    # ══ MOSL — convergence across the Wealth Creation Study family ═════
    with _mp_tabs[2]:
        # EXACT TOKENS, NEVER SUBSTRINGS. `frameworks_passed` joins names with ", ", and "QGLP" is
        # a SUBSTRING of "SQGLP Century Stock" — matching by substring inflated a first measurement
        # of this very cohort by 37 stocks. Splitting on the ", " boundary yields whole tokens only
        # (the same discipline ui_tearsheet._parse_frameworks uses). Cross-checked against the
        # authoritative qglp_pass column: both give 328.
        #
        # WHY A VIEW AND NOT NEW ENGINE COLUMNS: the fw_* booleans are LOCALS inside
        # scoring_engine.run_full_scoring and are never persisted, so frameworks_passed is the only
        # surviving record. Persisting them would be the cleaner data model but it is an engine
        # change, and this is a display feature.
        _MOSL_LENSES = ["QGLP", "Economic Moat", "Consistent in Volatile", "EP Hockey Stick",
                        "CAP-GAP Compounder", "SQGLP Century Stock", "100x Candidate",
                        "Blue Chip Quality", "MOSL Wealth Creator", "Bruised Blue Chip 29"]
        _tok = df.get("frameworks_passed", pd.Series("", index=df.index)).fillna("").astype(str).map(
            lambda _s: {t.strip() for t in re.split(r"\s*,\s*", _s) if t.strip()})
        _mosl = df.copy()
        _mosl["mosl_n"] = _tok.map(lambda t: sum(1 for m in _MOSL_LENSES if m in t))
        _mosl["mosl_hits"] = _tok.map(lambda t: " · ".join(m for m in _MOSL_LENSES if m in t))
        # >=2 because ONE lens is not convergence -- the tab's whole claim is that independent
        # studies from the same house agree.
        _mosl = _mosl[_mosl["mosl_n"] >= 2].sort_values(
            ["mosl_n", "composite_score"], ascending=False)

        st.markdown(
            f"<div class='sec-cap'>How many of the <b>{len(_MOSL_LENSES)} Motilal Oswal Wealth "
            f"Creation lenses</b> (studies 16–30) a stock clears at once. Unlike a count across all "
            f"37 frameworks — where gate strictness varies by design and the numbers are not "
            f"comparable — these come from one research programme, so agreement between them means "
            f"something. Showing stocks that clear <b>2 or more</b>; one lens is not convergence."
            f"</div>",
            unsafe_allow_html=True,
        )
        # THE CAVEAT IS IN THE CAPTION, NOT A TOOLTIP. Measured 2026-08-27 with EXACT-TOKEN
        # matching: the 4-or-more cohort (175 stocks) carries a median of 6 red flags against the
        # universe's 5 — slightly WORSE, not better — and 80.6% of it is AVOID. These lenses gate
        # quality, growth and longevity; none of them reads the forensics.
        # (An earlier substring-matched pass reported 5 flags and 83.6%. It was wrong: "QGLP" is a
        # substring of "SQGLP Century Stock", which pulled 37 extra stocks into the cohort.)
        st.markdown(
            f"<div style='font-size:0.72rem;color:{COLORS['gold']};margin:-4px 0 10px 0;'>"
            f"⚠️ Convergence is <b>agreement, not safety</b>. These lenses test quality, growth and "
            f"longevity — none of them reads the forensics, so a high count carries no clean-books "
            f"claim. Check <b>Verdict</b> and <b>🚩 Flags</b> on every row. Unvalidated against "
            f"forward returns: read it as convergence, never as conviction.</div>",
            unsafe_allow_html=True,
        )

        if _mosl.empty:
            st.info("No stock clears 2 or more MOSL lenses in this universe.")
        else:
            _n4 = int((_mosl["mosl_n"] >= 4).sum())
            st.markdown(f"""
            <div style="display:flex;gap:20px;align-items:center;margin-bottom:6px;">
              <span style="font-size:1.05rem;font-weight:800;color:{COLORS['gold']};">
                🔭 {len(_mosl)} stocks clear 2+ lenses
              </span>
              <span style="font-size:0.8rem;color:{COLORS['text_secondary']};">
                {_n4} clear 4+ &nbsp;·&nbsp; deepest agreement: {int(_mosl['mosl_n'].max())} of {len(_MOSL_LENSES)}
              </span>
            </div>
            """, unsafe_allow_html=True)

            _m_cols = [c for c in ["rank", "name", "verdict_direction", "red_flag_count",
                                   "mosl_n", "composite_score", "mosl_hits"]
                       if c in _mosl.columns]
            st.dataframe(
                _mosl[_m_cols].reset_index(drop=True),
                column_config={
                    "rank":             st.column_config.NumberColumn("Rank", format="%.0f"),
                    "name":             st.column_config.TextColumn("name", width="medium"),
                    "verdict_direction": st.column_config.TextColumn("Soundness", help="The engine's overall SOUND / MIXED / FLAWED gate. Most high-convergence names are FLAWED — the MOSL lenses do not read forensics or entry timing, and the soundness gate does."),
                    "red_flag_count":   st.column_config.NumberColumn("🚩 Flags", format="%.0f", width="small", help="Forensic red flags. The MOSL lenses gate quality/growth/longevity and NOT forensics, so this is the risk check the convergence count itself does not do."),
                    "mosl_n":           st.column_config.ProgressColumn("MOSL", min_value=0, max_value=len(_MOSL_LENSES), format="%.0f", width="small", help="How many of the 10 Wealth Creation lenses this stock clears."),
                    "composite_score":  st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f", width="small"),
                    "mosl_hits":        st.column_config.TextColumn("Lenses cleared", width="large"),
                },
                use_container_width=True,
                height=min(500, 80 + len(_mosl) * 35 + 40),
                hide_index=True,
            )

    # ══ Wealth — the change-lens tiers (engine columns from verdict_engine) ═════
    with _mp_tabs[3]:
        # Everything here READS the wealth_* columns compute_verdict materialized — zero logic
        # lives in this tab, so the tier a snapshot captures is byte-identical to the tier shown.
        # Grammar, provenance and the four rules: core/verdict_engine.py + tests/test_wealth_tier.py.
        _WT_ORDER = ["BUY★", "BUY", "WATCH★", "WATCH", "AVOID", "N/A"]
        _wt_counts = df["wealth_tier"].value_counts() if "wealth_tier" in df.columns else {}
        st.markdown(
            f"<div class='sec-cap'>The <b>wealth-engine tier</b> — three clocks, nothing else: "
            f"<b>EP%</b> (economic profit ÷ reserves = ROE − cost of equity, so a ₹200 Cr and a "
            f"₹2,000 Cr business compare fairly) · <b>Vel%</b> (this year's change in that excess "
            f"return) · <b>tau</b> (the 5-year margin spine). BUY★ = earning, improving, confirmed; "
            f"WATCH★ = the confirmed turnaround (not earning yet, improving with a spine); AVOID = "
            f"nothing improving — the LEVEL may be fine. Market-wide (ignores sidebar filters).</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='font-size:0.72rem;color:{COLORS['gold']};margin:-4px 0 10px 0;'>"
            f"⚠️ <b>Price-blind and forensics-blind by design.</b> BUY means the wealth engine is "
            f"buy-grade — not that the price is right (check Valuation) and not that the books are "
            f"clean (the ⚠ marker and 🚩 count carry that; they never alter the tier, so the "
            f"tension stays visible). A description to decide WITH, not a recommendation.</div>",
            unsafe_allow_html=True,
        )
        if "wealth_tier" not in df.columns:
            st.info("wealth_tier not present — re-run the pipeline.")
        else:
            st.markdown(
                "<div style='font-size:0.78rem;font-weight:700;margin-bottom:6px;'>"
                + " &nbsp;·&nbsp; ".join(
                    f"{t} <span style='color:{COLORS['text_secondary']};'>{int(_wt_counts.get(t, 0))}</span>"
                    for t in _WT_ORDER)
                + "</div>",
                unsafe_allow_html=True,
            )
            _wt_pick = st.selectbox(
                "Tier", ["All"] + _WT_ORDER, key="mp_wealth_tier",
                label_visibility="collapsed",
                help="Filter to one tier. The table always sorts strongest tier first, then by Vel%.",
            )
            _wl = df.copy()
            _wl["_wt_ord"] = _wl["wealth_tier"].map({t: i for i, t in enumerate(_WT_ORDER)})
            _wl["_warn_txt"] = np.where(_wl["wealth_warn"].fillna(0) == 1, "⚠", "")
            if _wt_pick != "All":
                _wl = _wl[_wl["wealth_tier"] == _wt_pick]
            _wl = _wl.sort_values(["_wt_ord", "wealth_vel_pct"], ascending=[True, False])
            _wt_cols = [c for c in ["wealth_tier", "name", "_warn_txt", "wealth_ep_pct",
                                    "wealth_vel_pct", "moat_tau", "moat_score", "growth_score",
                                    "red_flag_count", "verdict_direction", "reserves"]
                        if c in _wl.columns]
            st.dataframe(
                _wl[_wt_cols].reset_index(drop=True),
                column_config={
                    "wealth_tier":       st.column_config.TextColumn("Tier", width="small"),
                    "name":              st.column_config.TextColumn("name", width="medium"),
                    "_warn_txt":         st.column_config.TextColumn("⚠", width="small", help="Forensic caution: 8+ red flags or a Schilit checker fail. Never changes the tier — it rides beside it."),
                    "wealth_ep_pct":     st.column_config.NumberColumn("EP%",  format="%+.1f", width="small", help="Excess return: ROE − cost of equity, in percentage points. The universe median is NEGATIVE — being above zero already beats the median listed company."),
                    "wealth_vel_pct":    st.column_config.NumberColumn("Vel%", format="%+.1f", width="small", help="This year's CHANGE in the excess return, in points of the reserves base. Direction beats level — the 28th WCS's own finding."),
                    "moat_tau":          st.column_config.NumberColumn("tau",  format="%+.2f", width="small", help="5-year margin trend (rank correlation, −1…+1). ≥ +0.25 confirms; ≤ −0.25 caps the tier at WATCH."),
                    "moat_score":        st.column_config.NumberColumn("Moat", format="%.0f", width="small"),
                    "growth_score":      st.column_config.NumberColumn("Gro",  format="%.0f", width="small"),
                    "red_flag_count":    st.column_config.NumberColumn("🚩",   format="%.0f", width="small"),
                    "verdict_direction": st.column_config.TextColumn("Soundness", width="small", help="The soundness gate (level + forensics + valuation) beside the wealth tier (pure change). They answer DIFFERENT questions and disagreeing openly is the point: the engine's rare SOUND names include stocks this lens reads as decaying, and its FLAWED pile hides confirmed turnarounds."),
                    "reserves":          st.column_config.NumberColumn("Eq ₹Cr", format="%.0f", width="small", help="Reserves — the equity base behind EP% and Vel% (the same base economic profit is computed on). A tiny base can make the percentages explode — check this before believing an extreme EP%."),
                },
                use_container_width=True,
                height=min(520, 80 + len(_wl) * 35 + 40),
                hide_index=True,
            )

    # ══ Sectors ════════════════════════════════════════════════════
    with _mp_tabs[4]:
        # The size floor is a CONTROL now, so this caption cannot hardcode it. It is rendered into
        # a placeholder AFTER the filter row has produced _min_n — the same set-after-the-widget
        # pattern the sidebar funnel uses. The first cut left "≥5 stocks" literal here and it went
        # stale the moment the dial moved to 15: a caption contradicting its own control, which is
        # the exact defect class this session has been removing.
        _sec_cap_ph = st.empty()
        # Cap-tier filter — Market Pulse is market-wide by design (ignores the sidebar filter), so this
        # slices the WHOLE-sector aggregation by size. selectbox (not pills): cleaner for 7 options +
        # always returns a value; format_func adds the tier emoji while the option value stays the exact
        # market_category string (zero-mapping filter). Guarded if the column is absent.
        # TWO KINDS OF CONTROL, and the caption says so because they look identical and are not:
        #   RE-AGGREGATING (market-cap, cyclicality) filter the STOCKS, so every average and the
        #     % Qualify are recomputed over the surviving subset.
        #   ROW filters (capital phase, minimum size) hide whole sectors and leave the numbers
        #     of the survivors untouched.
        # sector_capital_phase is CONSTANT within a sector today (measured: 0 of 81 vary), so
        # filtering stocks by it would be equivalent — but it is applied AFTER aggregation anyway,
        # so it stays correct if that ever stops being true rather than silently part-filtering a
        # sector and skewing its averages.
        _sec_src = df
        _c1, _c2, _c3, _c4, _c5 = st.columns([2, 2, 2, 2, 2])

        with _c1:
            if "market_category" in df.columns:
                from config import MCAP_TIERS
                _cap_opts = ["All"] + [t for t in MCAP_TIERS if (df["market_category"] == t).any()]
                _cap = st.selectbox(
                    "Market-cap tier", _cap_opts,
                    format_func=lambda t: t if t == "All" else f"{MCAP_TIERS[t]['emoji']} {t}",
                    key="mp_sec_cap",
                    help="Re-aggregates: keeps only stocks in this tier, then recomputes every sector average.",
                )
            else:
                _cap = "All"

        with _c2:
            # 2026-08-27: Cyclicality tier → Wealth Tier in this slot (user request). 2026-08-28:
            # cyclicality RETURNED as a third control (below) once measurement showed the swap had
            # quietly lost a capability — 42 of 81 sectors hold MORE THAN ONE cyclicality tier
            # (Chemicals only 44% its dominant one), so "sector averages over Defensive stocks
            # only" is a stock-level re-aggregation no Discovery filter or row-hide can rebuild.
            # Two-kinds architecture now: re-aggregating = Market-cap + Wealth + Cyclicality ·
            # row filters = Capital phase + Min size.
            _WT_SEC = ["BUY★", "BUY", "WATCH★", "WATCH", "AVOID", "N/A"]
            if "wealth_tier" in df.columns:
                _wt_sec_opts = ["All"] + [t for t in _WT_SEC if (df["wealth_tier"] == t).any()]
                _wt_sec = st.selectbox(
                    "Wealth tier", _wt_sec_opts, key="mp_sec_wealth",
                    help="Re-aggregates: keeps only stocks in this wealth tier, then recomputes every "
                         "sector average and % Qualify over the survivors. 'BUY★, grouped by sector' "
                         "shows where the improving-wealth names concentrate.",
                )
            else:
                _wt_sec = "All"

        with _c3:
            # Canonical tier order (defensive → deep cyclical), not alphabetical — the economic
            # spectrum reads left to right. Presence-checked like the other re-aggregators.
            _CYC_SEC = ["Defensive", "Sensitive / Structural-Growth", "Cyclical",
                        "Deep Cyclical / Commodity", "Financials", "Catch-all"]
            if "cyclicality_tier" in df.columns:
                _cyc_opts = ["All"] + [t for t in _CYC_SEC if (df["cyclicality_tier"] == t).any()]
                _cyc_sec = st.selectbox(
                    "Cyclicality tier", _cyc_opts, key="mp_sec_cyc",
                    help="Re-aggregates: keeps only stocks in this cyclicality tier, then recomputes "
                         "every sector average, % Qualify and the 💹 share over the survivors. "
                         "Tiers cross sector lines (42 of 81 sectors hold more than one), so this "
                         "is a stock filter — hiding whole sectors could not answer 'how do "
                         "sectors rank among Defensive stocks only'.",
                )
            else:
                _cyc_sec = "All"

        with _c4:
            if "sector_capital_phase" in df.columns:
                _ph_opts = ["All"] + sorted(df["sector_capital_phase"].dropna().unique().tolist())
                _phase = st.selectbox(
                    "Capital phase", _ph_opts, key="mp_sec_phase",
                    help="Hides rows. The phase is a SECTOR attribute (constant within a sector), so this "
                         "shows or hides whole sectors and changes no average.",
                )
            else:
                _phase = "All"

        with _c5:
            # THE FLOOR IS NOW A DIAL. It was hardcoded at 5, and that is why the ranking was
            # dominated by tiny sectors: an extreme % Qualify is easy at n=7 and near-impossible at
            # n=96. Measured 2026-08-27: 8 of the top 10 sectors held fewer than 12 stocks, and
            # raising this to 15 changes the top six COMPLETELY (0 of 6 in common). Default stays 5
            # so nothing moves for anyone who does not touch it.
            _min_n = st.selectbox(
                "Min stocks / sector", [5, 10, 15, 20, 30], index=0, key="mp_sec_minn",
                help="Hides rows. A sector's % Qualify is only as trustworthy as its sample: at n=7 one "
                     "company moves it 14 points, at n=96 it moves it 1. Raise this to see only sectors "
                     "big enough for the percentage to mean something.",
            )

        _sec_cap_ph.markdown(
            f"<div class='sec-cap'>Every sector with <strong>≥{_min_n} stocks</strong> — "
            f"Quality / Momentum / Valuation / Score averaged across <strong>all</strong> its stocks "
            f"(sample-robust, not just the gate-passers). <strong>% Qualify</strong> = the share "
            f"clearing the hard gates (the sector's quality breadth). Ranked by % Qualify "
            f"(most-investable first). Capital-cycle phase is named below: 🔥 hot (over-investing — "
            f"caution) · ❄️ starved (under-invested — opportunity). A sector average can hide up to "
            f"<strong>50 points</strong> of industry dispersion — see 🏭 Industry for the split.</div>",
            unsafe_allow_html=True,
        )

        if _cap != "All":
            _sec_src = _sec_src[_sec_src["market_category"] == _cap]
        if _cyc_sec != "All" and "cyclicality_tier" in _sec_src.columns:
            _sec_src = _sec_src[_sec_src["cyclicality_tier"] == _cyc_sec]
        # 💹 TIER-SHARE BASE — captured AFTER the cap and cyclicality filters, BEFORE the wealth
        # filter, so Defensive × BUY★ reads "BUY★ share among the sector's Defensive stocks". The share
        # column answers "how much of this group's FULL roster is tier X?"; computed after the
        # wealth filter it would read 100% everywhere the moment a tier is selected (the filter
        # keeps only that tier — the trap the design review caught). Denominator = the roster
        # under every OTHER filter. The tier shown follows the filter selection; All → BUY★,
        # the top of the forward-validated monotonic ladder.
        _sec_share_base = _sec_src
        _sec_share_tier = _wt_sec if _wt_sec != "All" else "BUY★"
        if _wt_sec != "All":
            _sec_src = _sec_src[_sec_src["wealth_tier"] == _wt_sec]
        if _cap != "All" or _wt_sec != "All" or _cyc_sec != "All":
            _bits = " · ".join(b for b in [_cap if _cap != "All" else "",
                                           _cyc_sec if _cyc_sec != "All" else "",
                                           _wt_sec if _wt_sec != "All" else ""] if b)
            st.caption(f"📊 {len(_sec_src):,} stocks ({_bits}) across "
                       f"{_sec_src['sector'].nunique()} sectors — averages recomputed on this subset.")

        # WHOLE-sector aggregation over ALL stocks — bigger samples = robust averages (the fix for
        # comparing a 3-stock sector to a 50-stock one). % Qualify = gate-pass rate, the sample-size-
        # immune breadth signal. The >=5-stock floor reuses the engine's own sector_capital_phase guard
        # ("median unstable below 5"). No top-N cap — every reliable sector is shown, sorted by Score.
        _sec_stats = _sec_src.groupby("sector").agg(
            stocks=("name", "count"),
            pct_qualify=("gate_pass", lambda s: 100.0 * s.mean()),
            avg_quality=("quality_score",    "mean"),
            avg_momentum=("momentum_score",  "mean"),
            avg_valuation=("valuation_score","mean"),
            avg_composite=("composite_score","mean"),
        )
        # 👑 T1 REMOVED 2026-08-28 (user call, sparsity-backed): nonzero in 7 of 81 sectors (9%)
        # and 7 of 355 industries (2%) — the same 7 names live in Discovery's tier filter.
        # 💹 tier share — over _sec_share_base (the pre-wealth-filter roster; see above). Exact
        # equality, never a contains match: "BUY" is a substring of "BUY★" (the QGLP⊂SQGLP class).
        if "wealth_tier" in _sec_share_base.columns:
            _sec_stats["pct_tier"] = (
                _sec_share_base.groupby("sector")["wealth_tier"]
                .apply(lambda s: 100.0 * (s == _sec_share_tier).mean())
                .reindex(_sec_stats.index)
            )
        # Sort by % Qualify (breadth), then Score — so the most-INVESTABLE sectors lead. Sorting by
        # Score alone would rank a 0%-qualify sector #1 (e.g. Financial Services scores high on
        # fundamentals but every stock fails a hard gate), which misleads at a glance.
        _sec_stats = (_sec_stats[_sec_stats["stocks"] >= _min_n]
                      .sort_values(["pct_qualify", "avg_composite"], ascending=False))
        # Phase applied AFTER aggregation — a row filter, so the survivors' averages are untouched.
        if _phase != "All" and "sector_capital_phase" in df.columns:
            _keep = set(df.loc[df["sector_capital_phase"] == _phase, "sector"].dropna().unique())
            _sec_stats = _sec_stats[_sec_stats.index.isin(_keep)]

        if _sec_stats.empty:
            st.info(f"No sector clears these filters at ≥{_min_n} stocks — widen the selection or "
                    f"lower the minimum.")
        else:
            # Score (avg_composite) sat second-to-last and rendered as a bar plus a single
            # truncated digit. The three figures a reader scans first — how many, what share
            # qualifies, and how they score — now lead; the component averages follow.
            _sec_order = [c for c in ["stocks", "pct_qualify", "avg_composite", "pct_tier",
                                      "avg_quality", "avg_momentum", "avg_valuation"]
                          if c in _sec_stats.columns]
            st.dataframe(
                _sec_stats[_sec_order].reset_index(),
                column_config={
                    "stocks":        st.column_config.NumberColumn("Count", format="%.0f"),
                    "pct_qualify":   st.column_config.ProgressColumn("% Qualify", min_value=0, max_value=100, format="%.0f%%",
                                       help="Share of the sector's stocks that clear all hard gates — its quality breadth. "
                                            "SCALE-FREE, not statistically robust: a percentage stops big sectors "
                                            "dominating, but small ones then reach extremes easily. Measured "
                                            "2026-08-27: 8 of the top 10 sectors hold fewer than 12 stocks (median 9 "
                                            "vs 19 universe-wide), and at n=7 a single stock moves this 14 points. "
                                            "Read it alongside Count."),
                    "pct_tier":      st.column_config.ProgressColumn(f"💹 {_sec_share_tier} %", min_value=0, max_value=100, format="%.0f%%",
                                       help=f"Share of the sector's FULL roster in the {_sec_share_tier} wealth tier. The tier "
                                            f"follows the Wealth-tier filter (All → BUY★, the top of the ladder); the "
                                            f"denominator deliberately IGNORES that filter — computed after it, this column "
                                            f"would read 100% everywhere. Unverifiable (N/A) stocks stay in the denominator "
                                            f"and dilute the share. Universe BUY★ base rate ≈ 12%. Price-blind and "
                                            f"forensics-blind, like the tier itself; read against Count."),
                    "avg_quality":   st.column_config.ProgressColumn("Quality",  min_value=0, max_value=100, format="%.0f"),
                    "avg_momentum":  st.column_config.ProgressColumn("Momentum", min_value=0, max_value=100, format="%.0f"),
                    "avg_valuation": st.column_config.ProgressColumn("Valuation",min_value=0, max_value=100, format="%.0f"),
                    "avg_composite": st.column_config.ProgressColumn("Score",    min_value=0, max_value=100, format="%.0f"),
                },
                use_container_width=True,
                height=min(700, 80 + len(_sec_stats) * 35),
                hide_index=True,
            )

        # Capital-cycle phase — NAMES the Hot/Starved sectors (the Pulse band only COUNTS them),
        # computed universe-wide; always shown, independent of the cap filter / floor above.
        if "sector_capital_phase" in df.columns:
            import html as _html
            _phase_by_sec = df.groupby("sector")["sector_capital_phase"].first().fillna("")
            _hot     = sorted(_phase_by_sec[_phase_by_sec.str.contains("Hot", na=False)].index)
            _starved = sorted(_phase_by_sec[_phase_by_sec.str.contains("Starved", na=False)].index)
            _join = lambda xs: " · ".join(_html.escape(str(s)) for s in xs) if xs else "—"
            st.markdown(
                f'<div style="font-size:0.72rem;line-height:1.7;margin-top:12px;'
                f'border-top:1px solid {COLORS["border"]};padding-top:10px;">'
                f'<span style="color:{COLORS["orange"]};font-weight:700;">🔥 Hot capital '
                f'({len(_hot)})</span>'
                f'<span style="color:{COLORS["text_muted"]};"> — over-investing, caution: </span>'
                f'<span style="color:{COLORS["text_secondary"]};">{_join(_hot)}</span><br>'
                f'<span style="color:{COLORS["blue"]};font-weight:700;">❄️ Capital-starved '
                f'({len(_starved)})</span>'
                f'<span style="color:{COLORS["text_muted"]};"> — under-invested, opportunity: </span>'
                f'<span style="color:{COLORS["text_secondary"]};">{_join(_starved)}</span></div>',
                unsafe_allow_html=True,
            )

    # ══ Industry ═══════════════════════════════════════════════════
    with _mp_tabs[5]:
        # WHY THIS IS NOT THE SECTORS TAB WITH A DIFFERENT GROUPBY KEY. `sector` has 81 values,
        # `industry` has 355 — and averaging up to the sector destroys real dispersion. Measured
        # 2026-08-28: the six sizeable industries inside Pharmaceuticals run 18.1 → 51.3 on average
        # composite, a 33-point spread the Sectors tab reports as one number (FMCG 22.9, Auto
        # Ancillaries 22.6). 20 of the 76 industries holding ≥8 stocks sit more than 5 points from
        # their parent sector's average. That gap is this tab's whole subject, so it is a COLUMN
        # (Δ vs Sector) and the table sorts by it rather than by % Qualify.
        #
        # SORTING BY THE DELTA DOES NOT FIX THE SMALL-SAMPLE PROBLEM — an earlier version of this
        # comment claimed it did, and measurement contradicted it. Δ is MORE small-sample sensitive
        # than % Qualify, not less: % Qualify is bounded 0–100 while Δ is unbounded, so a one-stock
        # industry sitting 38 points off its sector average tops the entire table. Measured with no
        # floor at all: 5 of the top 10 rows were single-stock industries, and the top 10 overlapped
        # a floored table by 1 of 10. There is deliberately NO floor here — see the block below for
        # the trade that was made and the mitigation (Count, first column).
        if "industry" not in df.columns:
            st.info("🏭 No `industry` column in the loaded frame — re-run the pipeline.")
        else:
            # NO SIZE FLOOR — every industry is shown, down to the ones holding a single stock.
            # This went dial (5..30) → fixed 8 → none, each step on the user's explicit call.
            #
            # THE COST IS REAL AND ACCEPTED, recorded here so nobody "fixes" it back. Sorting by Δ
            # does not neutralise small samples — Δ is MORE exposed to them than % Qualify, since
            # % Qualify is bounded 0–100 while Δ is not. Measured 2026-08-28 with no floor: 5 of the
            # top 10 rows are single-stock industries, the leader is "Auto Ancillaries - Seats"
            # (n=1, +32) — one company's score minus a sector average — and the unfloored top 10
            # shares 1 row with the floored one. The floor did not trim the tail; it decided who led.
            # The mitigation is Count, on every row, second from the left: a reader can see n=1 and
            # discount it. That is the trade the user chose, deliberately, twice.

            # Same two-kinds architecture as Sectors, minus one control:
            #   RE-AGGREGATING  Market-cap tier, Wealth tier — filter the STOCKS, every average and
            #                   the sector BASELINE are recomputed over the survivors.
            #   ROW FILTER      Min stocks/industry — hides rows, touches no number.
            # Capital phase is deliberately absent: sector_capital_phase is a SECTOR attribute with
            # no industry-level analogue, so carrying it over would attach a sector's phase to an
            # industry that only partly lives in it.
            _IND_KEEP = [c for c in ["industry", "sector", "name", "composite_score",
                                     "quality_score", "momentum_score", "valuation_score",
                                     "gate_pass", "conviction_tier", "market_category",
                                     "wealth_tier"] if c in df.columns]
            _ind_src = df[_IND_KEEP].copy()
            _ind_src["industry"] = _ind_src["industry"].astype(str).str.strip()
            _ind_src = _ind_src[~_ind_src["industry"].isin(["", "nan", "None"])]

            _i1, _i2, _i3 = st.columns([2, 2, 2])
            with _i1:
                if "market_category" in _ind_src.columns:
                    from config import MCAP_TIERS
                    _ind_cap_opts = ["All"] + [t for t in MCAP_TIERS
                                               if (_ind_src["market_category"] == t).any()]
                    _ind_cap = st.selectbox(
                        "Market-cap tier", _ind_cap_opts,
                        format_func=lambda t: t if t == "All" else f"{MCAP_TIERS[t]['emoji']} {t}",
                        key="mp_ind_cap",
                        help="Re-aggregates: keeps only stocks in this tier, then recomputes every "
                             "industry average AND its sector baseline over the same survivors.",
                    )
                else:
                    _ind_cap = "All"
            with _i2:
                if "wealth_tier" in _ind_src.columns:
                    _ind_wt_opts = ["All"] + [t for t in ["BUY★", "BUY", "WATCH★", "WATCH",
                                                          "AVOID", "N/A"]
                                              if (_ind_src["wealth_tier"] == t).any()]
                    _ind_wt = st.selectbox(
                        "Wealth tier", _ind_wt_opts, key="mp_ind_wealth",
                        help="Re-aggregates. 'BUY★, grouped by industry' shows where the "
                             "improving-wealth names actually concentrate — a far sharper answer "
                             "than the same question asked of an 81-value sector.",
                    )
                else:
                    _ind_wt = "All"
            with _i3:
                # SECTOR DRILL-DOWN (2026-08-28) — the navigation the tab pair implies: spot a
                # sector on 📈 Sectors, open 🏭 Industry, see its internal dispersion. A ROW
                # FILTER applied AFTER aggregation (matches on the DOMINANT sector), so every
                # number — averages, Δ, 💹 share — is exactly what the unfiltered table shows.
                # Sectors hold a median of 3 industries (max 38), so this turns 355 rows into a
                # focused split. sorted() — determinism mandate.
                _ind_sec_opts = ["All"] + sorted(_ind_src["sector"].dropna().astype(str).unique().tolist())
                _ind_sec = st.selectbox(
                    "Sector (drill-down)", _ind_sec_opts, key="mp_ind_sec",
                    help="Hides rows: shows only industries whose MAJORITY of stocks sit in this "
                         "sector (the table's own 'dominant sector'). Applied after aggregation — "
                         "no average, Δ or 💹 share changes. Industries that only partly touch "
                         "the sector (the ~ rows) stay under their dominant home.",
                )

            if _ind_cap != "All":
                _ind_src = _ind_src[_ind_src["market_category"] == _ind_cap]
            # 💹 TIER-SHARE BASE — same design as the Sectors tab: captured AFTER the cap filter,
            # BEFORE the wealth filter, so the share column keeps the FULL roster as denominator
            # (computed after the filter it would read 100% everywhere). All → BUY★.
            _ind_share_base = _ind_src
            _ind_share_tier = _ind_wt if _ind_wt != "All" else "BUY★"
            if _ind_wt != "All":
                _ind_src = _ind_src[_ind_src["wealth_tier"] == _ind_wt]

            st.markdown(
                f"<div class='sec-cap'>All <strong>{_ind_src['industry'].nunique()} industries</strong>, "
                f"ranked by <strong>Δ vs Sector</strong> — how far its average score sits above or "
                f"below its sector <strong>peers</strong> (the industry's own stocks are excluded "
                f"from the baseline, so a sector-dominating industry cannot damp its own gap). That "
                f"is the one thing the Sectors tab cannot show: inside Pharmaceuticals alone, "
                f"industry averages run from 18 to 51. Positive = this industry outscores the rest "
                f"of its sector; negative = the rest of the sector is carrying it.</div>",
                unsafe_allow_html=True,
            )

            # DOMINANT SECTOR, not parent — industry is NOT nested inside sector. 136 of the 355
            # industries span more than one (dominant-sector share across ALL 355: median 1.00,
            # minimum 0.33 — the 136 multi-sector ones are the impure tail). The modal sector is
            # picked with an explicit (count desc, sector asc) tie-break: an unsorted mode is
            # non-deterministic across processes (PYTHONHASHSEED), which would make the displayed
            # parent sector flicker between runs.
            _ind_stats = _ind_src.groupby("industry").agg(
                stocks=("name", "count"),
                pct_qualify=("gate_pass", lambda s: 100.0 * s.mean()),
                avg_composite=("composite_score", "mean"),
                avg_quality=("quality_score", "mean"),
                avg_momentum=("momentum_score", "mean"),
                avg_valuation=("valuation_score", "mean"),
            )
            # 💹 tier share — over _ind_share_base (pre-wealth-filter roster; see above). Exact
            # equality, never contains: "BUY" ⊂ "BUY★".
            if "wealth_tier" in _ind_share_base.columns and not _ind_stats.empty:
                _ind_stats["pct_tier"] = (
                    _ind_share_base.groupby("industry")["wealth_tier"]
                    .apply(lambda s: 100.0 * (s == _ind_share_tier).mean())
                    .reindex(_ind_stats.index)
                )
            if _ind_stats.empty:
                st.info("No stocks match these filters — widen the selection.")
            else:
                _ind_pair = (_ind_src.groupby(["industry", "sector"]).size().rename("n")
                             .reset_index()
                             .sort_values(["industry", "n", "sector"], ascending=[True, False, True]))
                _ind_dom  = _ind_pair.drop_duplicates("industry").set_index("industry")
                _dom_sec  = _ind_dom["sector"].reindex(_ind_stats.index)
                # pd.Series (not a bare array): the drill-down below row-filters _ind_stats, and
                # positional alignment would silently pair shares with the wrong industries.
                _dom_share = pd.Series(
                    np.where(_ind_stats["stocks"] > 0,
                             _ind_dom["n"].reindex(_ind_stats.index) / _ind_stats["stocks"],
                             np.nan),
                    index=_ind_stats.index)

                # BOTH TERMS OF THE DIFFERENCE COME FROM ONE POPULATION. The baseline is grouped off
                # `_ind_src` — the FILTERED frame — not off `df`. Comparing a Small-Cap-only industry
                # average against an all-cap sector average would report the cap effect as though it
                # were an industry effect: the cross-year-basis defect in different clothes.
                #
                # LEAVE-ONE-OUT BASELINE (2026-08-28; was the plain sector mean). Including the
                # industry's own stocks damps the delta by exactly (1 − its share of the sector) —
                # measured: QSR displayed −7.0 against a true peer gap of −48.1 (it IS 83% of its
                # sector, so the baseline was mostly itself), Paints +4.2 vs +33.5, max distortion
                # 41 points — and the damping factor appears nowhere on screen, so a reader could
                # neither see nor undo it. The baseline now excludes the industry's own in-sector
                # stocks: Δ is the gap to its sector PEERS. The aggregate sort barely moves (rank
                # corr 0.971 vs the old math); the muted tail rows were the point.
                #
                # THE DEGENERATE CASE NOW FALLS OUT OF THE MATH. A sector holding only this
                # industry leaves zero peers, the count guard emits np.nan, and the row renders
                # blank — never a 0.0 sentinel that reads "perfectly average" when the truth is
                # "nothing to compare against". The old explicit ≥2-industries test is gone because
                # it became REDUNDANT, not because the rule changed (live: 8 of 355 industries,
                # counted dynamically in the footer).
                _sec_agg = _ind_src.groupby("sector")["composite_score"].agg(["sum", "count"])
                _ind_dom_rows = _ind_src[_ind_src["sector"].values ==
                                         _ind_src["industry"].map(_dom_sec).values]
                _ind_own = (_ind_dom_rows.groupby("industry")["composite_score"]
                            .agg(["sum", "count"]).reindex(_ind_stats.index))
                _peer_sum = _dom_sec.map(_sec_agg["sum"])   - _ind_own["sum"].fillna(0.0)
                _peer_cnt = _dom_sec.map(_sec_agg["count"]) - _ind_own["count"].fillna(0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    _ind_stats["delta_vs_sector"] = np.where(
                        _peer_cnt > 0,
                        _ind_stats["avg_composite"] - _peer_sum / _peer_cnt, np.nan)
                # "~" flags an industry whose stocks are NOT mostly in the sector named beside it.
                _ind_stats["dom_sector"] = np.where(_dom_share < 0.8,
                                                    "~ " + _dom_sec.astype(str),
                                                    _dom_sec.astype(str))

                if _ind_cap != "All" or _ind_wt != "All":
                    _ind_bits = " · ".join(b for b in [_ind_cap if _ind_cap != "All" else "",
                                                       _ind_wt if _ind_wt != "All" else ""] if b)
                    st.caption(f"🏭 {len(_ind_src):,} stocks ({_ind_bits}) across "
                               f"{_ind_src['industry'].nunique()} industries — averages and the "
                               f"sector baseline both recomputed on this subset.")

                # Incomparable rows carry no signal, so they sink rather than heading a descending
                # sort; avg_composite breaks ties and orders that trailing group sensibly.
                _ind_stats = _ind_stats.sort_values(["delta_vs_sector", "avg_composite"],
                                                    ascending=[False, False], na_position="last")

                # SECTOR DRILL-DOWN — row filter, applied AFTER every number is computed: matches
                # the dominant sector, so no average, Δ or 💹 share moves (pinned).
                if _ind_sec != "All":
                    _ind_stats = _ind_stats[_dom_sec.reindex(_ind_stats.index) == _ind_sec]
                    if _ind_stats.empty:
                        st.info(f"No industry has {_ind_sec} as its dominant sector under these "
                                f"filters — its stocks live inside industries that mostly sit "
                                f"elsewhere (the ~ rows of their own homes).")

                # Signal before context — the same invariant tests/test_market_pulse_columns.py pins
                # for the other tables. Sector names are the widest strings in the frame
                # ("Infrastructure Developers & Operators"), so the sector goes last.
                _ind_order = [c for c in ["stocks", "pct_qualify", "avg_composite",
                                          "delta_vs_sector", "pct_tier", "avg_quality",
                                          "avg_momentum", "avg_valuation",
                                          "dom_sector"]
                              if c in _ind_stats.columns]
                st.dataframe(
                    _ind_stats[_ind_order].reset_index(),
                    column_config={
                        "industry":       st.column_config.TextColumn("Industry", width="medium"),
                        "stocks":         st.column_config.NumberColumn("Count", format="%.0f",
                                            help="Read every percentage on this row against this number first."),
                        "pct_qualify":    st.column_config.ProgressColumn("% Qualify", min_value=0, max_value=100, format="%.0f%%",
                                            help="Share of the industry's stocks clearing all hard gates. SCALE-FREE, "
                                                 "not statistically robust — and much less robust here than on the "
                                                 "Sectors tab: the median industry holds 3 stocks against 19 for "
                                                 "sectors. This is a column, not the sort key, for exactly that reason."),
                        "avg_composite":  st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
                        "delta_vs_sector": st.column_config.NumberColumn("Δ vs Sector", format="%+.1f", width="small",
                                            help="Average score minus the average of its sector PEERS — the OTHER stocks "
                                                 "in the sector this industry mostly sits in; its own stocks are excluded "
                                                 "from the baseline (including them shrinks a dominant industry's gap "
                                                 "toward zero by its own weight). Both terms are computed over the SAME "
                                                 "filtered stocks. BLANK means incomparable, not zero: that sector holds "
                                                 "no other industry, so no peers exist."),
                        "pct_tier":       st.column_config.ProgressColumn(f"💹 {_ind_share_tier} %", min_value=0, max_value=100, format="%.0f%%",
                                            help=f"Share of the industry's FULL roster in the {_ind_share_tier} wealth tier. "
                                                 f"Follows the Wealth-tier filter (All → BUY★); the denominator deliberately "
                                                 f"IGNORES that filter — computed after it, the column would read 100% "
                                                 f"everywhere. N/A stocks dilute the share. Universe BUY★ base rate ≈ 12%. "
                                                 f"Read against Count — even more so here than on Sectors (median industry "
                                                 f"holds 3 stocks)."),
                        "avg_quality":    st.column_config.ProgressColumn("Quality",   min_value=0, max_value=100, format="%.0f"),
                        "avg_momentum":   st.column_config.ProgressColumn("Momentum",  min_value=0, max_value=100, format="%.0f"),
                        "avg_valuation":  st.column_config.ProgressColumn("Valuation", min_value=0, max_value=100, format="%.0f"),
                        "dom_sector":     st.column_config.TextColumn("Sector (dominant)", width="medium",
                                            help="Industry is NOT nested inside sector — 136 of 355 span more than one. "
                                                 "This is where the MAJORITY of the industry's stocks sit; a leading '~' "
                                                 "means under 80% of them do, so read the Δ for that row loosely."),
                    },
                    use_container_width=True,
                    height=min(700, 80 + len(_ind_stats) * 35),
                    hide_index=True,
                )

                _ind_blank = int(_ind_stats["delta_vs_sector"].isna().sum())
                _ind_tilde = int((_dom_share.reindex(_ind_stats.index) < 0.8).sum())
                st.markdown(
                    f'<div style="font-size:0.72rem;line-height:1.7;margin-top:12px;'
                    f'border-top:1px solid {COLORS["border"]};padding-top:10px;'
                    f'color:{COLORS["text_muted"]};">'
                    f'<span style="color:{COLORS["text_secondary"]};font-weight:700;">Reading this '
                    f'table.</span> {len(_ind_stats)} industries, no size floor. Check '
                    f'<strong>Count</strong> before trusting a Δ — a one-stock industry is just '
                    f'that single stock measured against its sector average. '
                    f'<strong>{_ind_blank}</strong> show a blank Δ — their sector contains no other '
                    f'industry, so there is nothing to compare them against (blank, deliberately, '
                    f'rather than a 0.0 that would read as "average"). '
                    f'<strong>{_ind_tilde}</strong> carry a <strong>~</strong> — fewer than 80% of '
                    f'their stocks sit in the sector named beside them, because industry is not a '
                    f'clean subdivision of sector.</div>',
                    unsafe_allow_html=True,
                )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5: CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[4]:
    st.markdown(f"<div class='sec-head'>⚙️ System Configuration — The Engine Rulebook</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='sec-cap'>The two live scoring controls, then a read-only view of the "
        f"deterministic weights and hard gates every stock is measured against. To change the "
        f"constants, edit <code>config.py</code> — the single source of truth.</div>",
        unsafe_allow_html=True,
    )

    # ── Live scoring controls (moved from the front-page Command Center, 2026-08-24) ──
    # Plain widget-owned keys — the top of the script reads them next rerun. Honest labels:
    # only Analysis Mode re-ranks; the profile drives the QGLP screen, never the composite.
    _cfg_c1, _cfg_c2 = st.columns(2)
    with _cfg_c1:
        st.selectbox(
            "Analysis Mode", options=list(ANALYSIS_MODES.keys()),
            format_func=lambda k: ANALYSIS_MODES[k]["label"], key="cfg_mode",
            help="Fundamental-vs-momentum blend of the composite — the one control that re-ranks "
                 "the universe (Hybrid 70/30 · Fundamental 100/0 · Technical 10/90).",
        )
        st.caption(ANALYSIS_MODES[analysis_mode]["description"])
    with _cfg_c2:
        st.selectbox(
            "Scoring Profile", options=_allowed_profiles,
            format_func=lambda k: f"{MASTER_PROFILES[k]['icon']} {MASTER_PROFILES[k]['label']}",
            key="cfg_profile",
            help="Drives the QGLP screen — its gates, fit count and the tearsheet QGLP card. "
                 "It does NOT re-rank the composite (measured 2026-08-24).",
        )
        st.caption(MASTER_PROFILES[scoring_profile]["description"])
    _fit_cfg = int(((df["gate_pass"] == 1)
                    & (df.get("qglp_pass", pd.Series(0, index=df.index)) == 1)).sum())
    st.markdown(
        f'<div style="font-size:0.72rem;color:{COLORS["text_muted"]};margin:2px 0 14px 2px;">'
        f'🎯 QGLP screen ({scoring_profile}) — ROCE≥{adaptive_w.get("roce_gate", 15):.0f}% · '
        f'Growth≥{adaptive_w.get("growth_gate", 15):.0f}% · PEG≤{adaptive_w.get("peg_gate", 1.5):.1f} '
        f'&nbsp;→&nbsp;<span style="color:{COLORS["gold"]};font-weight:700;">{_fit_cfg} fit</span> '
        f'(of {gate_passed} gate-passed)</div>',
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

    # ── Composite Score Formula — the master blend the sub-weights below feed into ──
    # Mirrors scoring_engine: composite = quality·(F) + momentum·(M) + governance·gov_w, where
    # governance is fixed and F+M fill (1-gov_w), split by analysis mode (from ANALYSIS_MODES — DRY).
    _gov_w  = COMPOSITE_WEIGHTS.get("governance", 0.15)
    _scale  = 1.0 - _gov_w
    _mode_icon = {"Hybrid": "🧭", "Fundamental": "📊", "Technical": "📈"}
    _mode_rows = "".join(
        f'<div style="display:flex;justify-content:space-between;font-size:0.72rem;padding:3px 0;'
        f'border-bottom:1px solid rgba(255,255,255,0.04);">'
        f'<span style="color:{COLORS["text_secondary"]};">{_mode_icon.get(_m, "•")} {_m}</span>'
        f'<span style="color:{COLORS["text_muted"]};">Quality '
        f'<strong style="color:{COLORS["purple"]};">{_v["fundamental_w"]*100:.0f}%</strong> : '
        f'Momentum <strong style="color:{COLORS["orange"]};">{_v["momentum_w"]*100:.0f}%</strong></span></div>'
        for _m, _v in ANALYSIS_MODES.items()
    )
    _comp_body = (
        f'<div style="font-size:0.82rem;color:{COLORS["text_primary"]};font-weight:700;margin-bottom:4px;">'
        f'Composite = Quality × F &nbsp;+&nbsp; Momentum × M &nbsp;+&nbsp; '
        f'Governance × <span style="color:{COLORS["gold"]};">{_gov_w*100:.0f}%</span></div>'
        f'<div style="font-size:0.68rem;color:{COLORS["text_muted"]};margin-bottom:8px;">'
        f'Governance is fixed at {_gov_w*100:.0f}%; F and M split the remaining {_scale*100:.0f}% by analysis mode:</div>'
        f'{_mode_rows}'
        f'<div style="font-size:0.68rem;color:{COLORS["text_muted"]};margin-top:8px;'
        f'border-top:1px solid {COLORS["border"]};padding-top:8px;">'
        f'Then: <strong style="color:{COLORS["text_secondary"]};">+ framework boosts</strong> (e.g. SQGLP +15) '
        f'→ <strong style="color:{COLORS["text_secondary"]};">× forensic penalty</strong> multiplier '
        f'→ clamped to a final 0–100 score.</div>'
    )
    st.markdown(_cfg_card("Composite Score Formula — How the Final Score is Built", "🧮",
                          _comp_body, COLORS["green"]), unsafe_allow_html=True)

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

    # ── Conviction Tiers — the post-penalty composite_score → tier mapping (from CONVICTION_TIERS) ──
    _tier_rows = "".join(
        f'<div style="display:flex;align-items:center;gap:10px;padding:5px 0;'
        f'border-bottom:1px solid rgba(255,255,255,0.04);flex-wrap:wrap;">'
        f'<span style="font-size:0.78rem;font-weight:800;color:{t["color"]};min-width:150px;">'
        f'{t["emoji"]} {t["label"]}</span>'
        f'<span style="font-size:0.68rem;font-weight:700;color:{t["color"]};background:{t["color"]}1a;'
        f'border:1px solid {t["color"]}44;border-radius:5px;padding:1px 8px;white-space:nowrap;">'
        f'score ≥ {t["min"]}</span>'
        f'<span style="font-size:0.7rem;color:{COLORS["text_secondary"]};flex:1;min-width:200px;">'
        f'{t["description"]}</span></div>'
        for t in CONVICTION_TIERS
    )
    st.markdown(
        _cfg_card(f"Conviction Tiers · {len(CONVICTION_TIERS)} Bands — Score → Tier Mapping", "🏆",
                  _tier_rows, COLORS["gold"]),
        unsafe_allow_html=True,
    )

    # ── Asymmetric Penalty Multipliers — the two "× penalty" levers the formula card references ──
    # Both schedules render live from config (FORENSIC_PENALTY_TIERS + GOVERNANCE_RISK_MULTIPLIERS),
    # the SAME constants the engine applies — so this card can never drift from the real penalty.
    st.markdown(
        f'<div style="font-size:0.72rem;color:{COLORS["text_secondary"]};margin:2px 0 8px 2px;">'
        f'🔻 <strong>Negative signals don\'t subtract points — they MULTIPLY the composite down</strong>, '
        f'so the penalty scales with conviction (a 90-score loses more absolute points than a 20). '
        f'Forensic flags are <em>evidence</em> (harsher, ×0.50 floor); ownership signals are '
        f'<em>warnings</em> (milder, ×0.70 floor).</div>',
        unsafe_allow_html=True,
    )

    def _pen_color(m: float) -> str:
        """Severity tint for a penalty multiplier (display-only)."""
        return (COLORS["green"] if m >= 0.999 else COLORS["gold"] if m >= 0.85
                else COLORS["orange"] if m >= 0.70 else COLORS["red"])

    def _pen_row(left: str, mult: float, right: str = "") -> str:
        c = _pen_color(mult)
        _r = (f'<span style="font-size:0.66rem;color:{COLORS["text_muted"]};flex:1;">{right}</span>'
              if right else '<span style="flex:1;"></span>')
        return (
            f'<div style="display:flex;align-items:center;gap:10px;padding:4px 0;'
            f'border-bottom:1px solid rgba(255,255,255,0.04);">'
            f'<span style="font-size:0.72rem;color:{COLORS["text_secondary"]};min-width:92px;">{left}</span>'
            f'<span style="font-size:0.74rem;font-weight:800;color:{c};min-width:54px;">× {mult:.2f}</span>'
            f'{_r}</div>'
        )

    # Forensic cascade rows — derive the count RANGE from the ascending max_flags upper bounds.
    _fc_rows, _prev = "", -1
    for _t in FORENSIC_PENALTY_TIERS:
        _mx = _t["max_flags"]
        if _mx is None:
            _rng = f"{_prev + 1}+ flags"
        elif _mx == _prev + 1:
            _rng = f"{_mx} flag" + ("" if _mx == 1 else "s")
        else:
            _rng = f"{_prev + 1}–{_mx} flags"
        _fc_rows += _pen_row(_rng, _t["multiplier"], _t["label"])
        _prev = _mx if _mx is not None else _prev

    # Governance shield rows — exact count → multiplier; the highest key is the "N+" bucket.
    _gk = sorted(GOVERNANCE_RISK_MULTIPLIERS)
    _gmax = max(_gk)
    _g_lbl = {0: "clean", 1: "caution", 2: "structural concern", 3: "promoter signal"}
    _gov_rows = ""
    for _k in _gk:
        _lab = (f"{_k}+ signals" if (_k == _gmax and _k > 0)
                else "no signals" if _k == 0 else f"{_k} signal" + ("" if _k == 1 else "s"))
        _gov_rows += _pen_row(_lab, GOVERNANCE_RISK_MULTIPLIERS[_k], _g_lbl.get(_k, ""))

    pc1, pc2 = st.columns(2)
    with pc1:
        st.markdown(
            _cfg_card("Forensic Red-Flag Cascade — Evidence", "🔬",
                      f'<div style="font-size:0.64rem;color:{COLORS["text_muted"]};margin-bottom:6px;">'
                      f'red_flag_count → multiplier on composite_score</div>{_fc_rows}', COLORS["red"]),
            unsafe_allow_html=True,
        )
    with pc2:
        st.markdown(
            _cfg_card("Governance Risk Shield — Warnings", "🛡️",
                      f'<div style="font-size:0.64rem;color:{COLORS["text_muted"]};margin-bottom:6px;">'
                      f'hard ownership-risk signals → multiplier on composite_score</div>{_gov_rows}',
                      COLORS["gold"]),
            unsafe_allow_html=True,
        )

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

    # ── 🩺 DATA HEALTH — the source-sheet gaps the engine works around (LIVE, never hardcoded) ──
    # The engine rulebook's missing chapter: the two known sheet defects degrade real signals
    # (DPR → fabricated full retention; CR-1YB → dead Piotroski F6), and until now they lived only
    # in session memories. Every figure below is COMPUTED THIS RUN, so the day the sheet is fixed
    # the rows turn green by themselves — the card self-resolves, it cannot go stale.
    st.markdown("---")
    st.markdown(f"<div class='sec-head'>🩺 Data Health</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='sec-cap'>Known source-sheet gaps, measured live on this run's data. These are "
        f"fixed in the <strong>Google Sheet / CSVs</strong>, not in code — each row shows exactly "
        f"what it degrades and turns green on its own once the sheet carries the real figure.</div>",
        unsafe_allow_html=True,
    )

    def _dh_row(dot_clr: str, label: str, value: str, consequence: str) -> str:
        return (
            f'<div style="display:flex;align-items:baseline;gap:10px;padding:6px 0;'
            f'border-bottom:1px solid rgba(255,255,255,0.04);flex-wrap:wrap;">'
            f'<span style="color:{dot_clr};font-size:0.9rem;line-height:1;">●</span>'
            f'<span style="font-size:0.74rem;font-weight:700;color:{COLORS["text_primary"]};'
            f'min-width:150px;">{label}</span>'
            f'<span style="font-size:0.74rem;font-weight:800;color:{dot_clr};min-width:110px;">{value}</span>'
            f'<span style="font-size:0.68rem;color:{COLORS["text_muted"]};flex:1;min-width:260px;">'
            f'{consequence}</span></div>'
        )

    # 1. DPR coverage — missing payout is read as "retains everything" (RR fabricated at 1.0).
    _dpr_s = df.get("dividend_payout_ratio", pd.Series(np.nan, index=df.index))
    _dpr_cov = float(_dpr_s.notna().mean()) * 100.0
    _dpr_clr = (COLORS["green"] if _dpr_cov >= 90 else
                COLORS["gold"] if _dpr_cov >= 60 else COLORS["red"])
    _dh = _dh_row(
        _dpr_clr, "Dividend Payout (DPR)", f"{_dpr_cov:.0f}% populated",
        "Missing rows are read as full retention (RR = 1.0) — inflates Value Creation Velocity, "
        "g★ and the misallocation flag. Fix: the DPR column in the source sheet.",
    )

    # 2. CR one-year-back — the known copy bug: identical to current CR for every row.
    _cr0 = df.get("current_ratio", pd.Series(np.nan, index=df.index))
    _cr1 = df.get("current_ratio_1yb", pd.Series(np.nan, index=df.index))
    _cr_both = _cr0.notna() & _cr1.notna()
    _cr_same = float((_cr0[_cr_both] == _cr1[_cr_both]).mean()) * 100.0 if _cr_both.any() else np.nan
    if pd.isna(_cr_same):
        _dh += _dh_row(COLORS["text_muted"], "Current Ratio 1Y-back", "not reported",
                       "No prior-year liquidity figure at all — Piotroski F6 and the "
                       "liquidity-improving check cannot run.")
    else:
        _cr_clr = (COLORS["green"] if _cr_same < 50 else
                   COLORS["gold"] if _cr_same < 95 else COLORS["red"])
        _dh += _dh_row(
            _cr_clr, "Current Ratio 1Y-back", f"{_cr_same:.0f}% identical to CR",
            "A copy of the current figure carries no year-over-year information — Piotroski F6 "
            "and the liquidity-improving check stay dead until the sheet holds the real prior year.",
        )

    # 3. Overall evidence coverage — context, from the engine's own confidence input.
    _cov_s = df.get("data_coverage_pct", pd.Series(np.nan, index=df.index))
    if _cov_s.notna().any():
        _cov_med, _cov_p10 = float(_cov_s.median()), float(_cov_s.quantile(0.10))
        _cov_clr = COLORS["green"] if _cov_p10 >= 60 else COLORS["gold"]
        _dh += _dh_row(_cov_clr, "Evidence coverage", f"median {_cov_med:.0f}%",
                       f"The 44-input coverage behind the 🔍 confidence badge; the thinnest tenth "
                       f"of the universe sits at {_cov_p10:.0f}% or less.")

    # 4. Snapshot age — the validation series only exists if captures happen (monthly ritual).
    _snap_dir = os.path.join("Other Resources", "snapshots")
    _snaps = (sorted(f for f in os.listdir(_snap_dir) if f.endswith(".csv"))
              if os.path.isdir(_snap_dir) else [])
    if _snaps:
        _snap_age = (pd.Timestamp.now()
                     - pd.Timestamp(os.path.getmtime(os.path.join(_snap_dir, _snaps[-1])), unit="s")).days
        _snap_clr = (COLORS["green"] if _snap_age <= 35 else
                     COLORS["gold"] if _snap_age <= 70 else COLORS["red"])
        _dh += _dh_row(_snap_clr, "Last snapshot", f"{_snap_age} day{'s' if _snap_age != 1 else ''} ago",
                       "Forward-return validation needs a monthly capture: run tools/snapshot.py "
                       "on the first trading day of each month.")
    else:
        _dh += _dh_row(COLORS["red"], "Last snapshot", "none yet",
                       "No capture in Other Resources/snapshots — every conclusion about whether "
                       "the engine predicts returns waits on this. Run tools/snapshot.py.")

    st.markdown(_cfg_card("Source-Sheet Gaps & Validation Cadence — Live", "🩺", _dh, COLORS["cyan"]),
                unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center; padding:20px; color:{COLORS['text_muted']}; font-size:0.75rem;">
        PRISM v{UI['version']} · Quantamental Intelligence · Every lens, one verdict
    </div>
    """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 6: REFERENCE — searchable glossary (renders the _RAW_GLOSSARY single source, count shown live)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[5]:
    st.markdown(
        f'<div style="font-size:0.7rem;font-weight:700;color:{COLORS["text_muted"]};'
        f'text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">'
        f'📖 Reference — Glossary</div>',
        unsafe_allow_html=True,
    )
    _ref_q = st.text_input(
        "Search the glossary", key="ref_search",
        placeholder="Search any term or label (e.g. PEG, Wealth Creator, Stage 2)…",
        label_visibility="collapsed",
    )
    # Offline copy of the ENTIRE reference (ignores the search filter) — one generator, same
    # single-source dicts as the on-screen render, so the download can never drift from the app.
    # The 37-framework registry rides along: _FW_META (tuple (color, emoji, desc)) adapted to the
    # {emoji,name,desc} shape the builder emits — same single source the tearsheet renders.
    _fw_md = {name: {"emoji": meta[1], "name": name, "desc": meta[2]}
              for name, meta in _FW_META.items()}
    st.download_button(
        "📥 Download Reference (Markdown)",
        data=build_reference_markdown(_RAW_GLOSSARY, CONCEPT_REFERENCE, _FLAG_DISPLAY, frameworks=_fw_md),
        file_name="prism_reference.md", mime="text/markdown",
        use_container_width=True,
    )
    # Two corpora, one search: the term glossary (column NAMES) + the concept reference (the VALUE
    # labels you see on a cell — Wealth Creator, Deep Value, Stage 2…). The query filters both.
    _concepts_html = render_concepts(CONCEPT_REFERENCE, _ref_q)
    if _concepts_html:
        st.markdown(
            f'<div style="font-size:0.72rem;font-weight:800;color:{COLORS["text_secondary"]};'
            f'text-transform:uppercase;letter-spacing:1px;margin:6px 0 2px 0;">'
            f'Labels &amp; Verdicts — what each value means</div>',
            unsafe_allow_html=True,
        )
        st.markdown(_concepts_html, unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:0.72rem;font-weight:800;color:{COLORS["text_secondary"]};'
            f'text-transform:uppercase;letter-spacing:1px;margin:20px 0 2px 0;">Glossary — terms</div>',
            unsafe_allow_html=True,
        )
    st.markdown(render_reference(_RAW_GLOSSARY, _ref_q), unsafe_allow_html=True)
    # Forensic red flags — rendered straight from the engine's single-source _FLAG_DISPLAY (no copy).
    _flags_html = render_flags(_FLAG_DISPLAY, _ref_q)
    if _flags_html:
        st.markdown(
            f'<div style="font-size:0.72rem;font-weight:800;color:{COLORS["red"]};'
            f'text-transform:uppercase;letter-spacing:1px;margin:20px 0 2px 0;">'
            f'Forensic Red Flags — what each warning means</div>',
            unsafe_allow_html=True,
        )
        st.markdown(_flags_html, unsafe_allow_html=True)
    # Framework registry ON SCREEN (2026-08-28) — the same _fw_md the download consumes; before
    # this, the export documented all 37 while the search box could find only 6.
    _fw_html = render_frameworks(_fw_md, _ref_q)
    if _fw_html:
        st.markdown(
            f'<div style="font-size:0.72rem;font-weight:800;color:{COLORS["purple"]};'
            f'text-transform:uppercase;letter-spacing:1px;margin:20px 0 2px 0;">'
            f'Framework Registry — the {len(_fw_md)} lenses</div>',
            unsafe_allow_html=True,
        )
        st.markdown(_fw_html, unsafe_allow_html=True)