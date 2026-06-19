"""
PRISM — Quantamental Intelligence
=================================
Every lens. One verdict. — Regime-Aware, Master-Driven
Dr. Malik + Raamdeo Agrawal + O'Neil + Mukherjea + Marks + Fisher + Lynch
"""
import os
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

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
                render_lynch_radar, render_mauboussin_radar, render_mosl_wealth_matrix,
                render_sector_peer_strip,
                render_valuation_inversion_and_sizing_cockpit,
                inject_css, render_hero_banner, render_metric_strip, render_pulse_band,
                render_stock_card, help_chip,
                render_radar_chart, render_score_bar, render_sidebar_brand,
                render_reference, render_concepts, render_flags, build_reference_markdown)
from ui.ui_discovery import render_discovery_sidebar, clear_all_filters
from ui.ui_scanner import _SCANNER_HEADER_TIPS
from ui.ui_components import _RAW_GLOSSARY
from ui.ui_reference_data import CONCEPT_REFERENCE
from ui.ui_tearsheet import _FLAG_DISPLAY
from config import (COLORS, TIER_COLORS, CONVICTION_TIERS, UI, HARD_GATES,
                    QUALITY_WEIGHTS, MOMENTUM_WEIGHTS, COMPOSITE_WEIGHTS,
                    VALUATION_SIGNALS, MARKS_CYCLE, DEFAULT_CYCLE_TEMPERATURE,
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
# Single source of truth = (adv_mode, adv_profile). sel_mandate is DERIVED: the mandate whose
# (mode, profile) matches the active combo, or None = a "⚙️ Custom" override combo (e.g. Growth/
# Defensive, or any pair no mandate uses). Streamlit writes a changed widget's value into
# session_state BEFORE the script reruns, so deriving here (at the top) reflects the user's latest
# override and keeps the button highlight + the card consistent in a single pass.
_MANDATE_BY_COMBO = {(v["mode"], v["profile"]): k for k, v in _MANDATES.items()}

# Mandate state lives in CANONICAL keys adv_mode / adv_profile — which are deliberately NOT widget
# keys. Streamlit forbids sharing one key between a widget and your own programmatic writes: the
# selectbox then reverts / needs two clicks, version-dependently (streamlit#7649 + the widget-behavior
# docs). So: buttons mutate canonical state in on_click CALLBACKS; the Override selectboxes own their
# OWN keys (_w_mode/_w_profile) and write back via on_change; and we MIRROR canonical → those widget
# keys each run. Callbacks run BEFORE the rerun — the only safe moment to set state a widget reads.
if "adv_mode" not in st.session_state:                  # first load → default mandate (QGLP Balanced)
    _d = _MANDATE_KEYS[0]
    st.session_state["adv_mode"]    = _MANDATES[_d]["mode"]
    st.session_state["adv_profile"] = _MANDATES[_d]["profile"]
# Guard FIRST: snap the active profile into the active mode's allowed set (so a mode change that
# orphans the profile resolves cleanly AND the Scoring-Profile selectbox value stays valid below).
_allowed_now = ANALYSIS_MODES[st.session_state["adv_mode"]]["allowed_profiles"]
if st.session_state["adv_profile"] not in _allowed_now:
    st.session_state["adv_profile"] = _allowed_now[0]
_sel_mandate = _MANDATE_BY_COMBO.get((st.session_state["adv_mode"], st.session_state["adv_profile"]))
st.session_state["sel_mandate"] = _sel_mandate          # None = ⚙️ Custom
_mandate_label = _sel_mandate or "Custom"
# Mirror canonical → the selectbox widget keys BEFORE those widgets render below, so a button-driven
# change is reflected in the Override selectboxes too (safe: writing a widget key before its widget).
st.session_state["_w_mode"]    = st.session_state["adv_mode"]
st.session_state["_w_profile"] = st.session_state["adv_profile"]

def _pick_mandate(_mode, _profile):                     # button on_click — runs before the rerun
    st.session_state["adv_mode"]    = _mode
    st.session_state["adv_profile"] = _profile

_mb_cols = st.columns(len(_MANDATES))
for _mi, (_mk, _mv) in enumerate(_MANDATES.items()):
    with _mb_cols[_mi]:
        st.button(
            f"{_mv['icon']} {_mk}",
            key=f"_mb_{_mk}",
            type="primary" if _sel_mandate == _mk else "secondary",
            use_container_width=True,
            on_click=_pick_mandate, args=(_mv["mode"], _mv["profile"]),
        )

# Mandate description strip — None-safe (a Custom override shows the active profile's description)
_desc = (_MANDATES[_sel_mandate]["desc"] if _sel_mandate
         else f"⚙️ Custom override — {MASTER_PROFILES[st.session_state['adv_profile']]['description']}")
st.markdown(
    f'<div style="font-size:0.75rem;color:{COLORS["text_secondary"]};'
    f'padding:4px 2px 10px 2px;border-bottom:1px solid {COLORS["border"]};margin-bottom:6px;">'
    f'{_desc}</div>',
    unsafe_allow_html=True,
)

# ── Advanced Override (collapsed — power users only) ───────────
# The selectboxes own SEPARATE keys (_w_mode/_w_profile, mirrored from canonical above) and push the
# user's pick back into canonical via on_change — never sharing a key with the buttons' writes.
def _sync_mode():
    st.session_state["adv_mode"] = st.session_state["_w_mode"]

def _sync_profile():
    st.session_state["adv_profile"] = st.session_state["_w_profile"]

with st.expander("⚙️ Advanced: Override Mandate Defaults", expanded=False):
    st.selectbox(
        "Analysis Mode",
        options=list(ANALYSIS_MODES.keys()),
        format_func=lambda k: ANALYSIS_MODES[k]["label"],
        key="_w_mode", on_change=_sync_mode,
    )
    st.caption(ANALYSIS_MODES[st.session_state["adv_mode"]]["description"])

    _ov_allowed = ANALYSIS_MODES[st.session_state["adv_mode"]]["allowed_profiles"]
    # (the top guard already snapped adv_profile into _ov_allowed, so the value below is always valid)
    st.selectbox(
        "Scoring Profile",
        options=_ov_allowed,
        format_func=lambda k: f"{MASTER_PROFILES[k]['icon']} {MASTER_PROFILES[k]['label']}",
        key="_w_profile", on_change=_sync_profile,
    )
    st.caption(MASTER_PROFILES[st.session_state["adv_profile"]]["description"])

# Canonical state drives scoring + display (the selectbox returns can lag a rerun behind canonical).
analysis_mode   = st.session_state["adv_mode"]
scoring_profile = st.session_state["adv_profile"]
profile_cfg = MASTER_PROFILES[scoring_profile]

# ── Scoring ────────────────────────────────────────────────────
_score_key = f"{file_sig}::{analysis_mode}::{scoring_profile}"
if st.session_state.get("_score_key") != _score_key or "_scored_df" not in st.session_state:
    _spin_icon = _MANDATES.get(_sel_mandate, {}).get("icon", "🧭")
    with st.spinner(f"{_spin_icon} Running {_mandate_label} mandate — {scoring_profile}..."):
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
# Mandate Fit = clears the universal safety floor (gate_pass) AND fits the selected mandate's
# per-profile thesis screen (qglp_pass: ROCE/Growth/PEG) — the mandate-responsive qualified count
# (gate_pass is profile-invariant; qglp_pass alone can exceed it by including unsafe names).
mandate_fit = int(((df["gate_pass"] == 1) &
                   (df.get("qglp_pass", pd.Series(0, index=df.index)) == 1)).sum())
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

    # Full-universe export — the complete scored frame (all rows/cols) as Excel-safe CSV. Distinct
    # from the Deep Scanner's curated/filtered export and the All-Data single-row export. Imports are
    # co-located here (not top-of-file) to keep this change in one self-contained hunk.
    from datetime import date as _date
    from ui.ui_export import scored_universe_csv
    st.download_button(
        f"📥 Download full scored data — {total} × {df.shape[1]} cols",
        data=scored_universe_csv(_score_key, df),
        file_name=f"prism_scored_universe_{_date.today().isoformat()}_{total}stocks.csv",
        mime="text/csv",
        use_container_width=True,
        help="The complete scored universe (every column) as CSV, for your own Excel/Python "
             "analysis. For a curated, filtered list, use the Deep Scanner's export instead.",
    )

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


# ═══════════════════════════════════════════════════════════════
# STATS STRIP (above tabs — reflects the selected mandate)
# ═══════════════════════════════════════════════════════════════
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
    _m_icon     = _MANDATES.get(_sel_mandate, {}).get("icon", "⚙️")
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
          {_m_icon} {_mandate_label}
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
        🎯 Mandate Screen — ROCE≥{adaptive_w.get('roce_gate', 15):.0f}% ·
        Growth≥{adaptive_w.get('growth_gate', 15):.0f}% ·
        PEG≤{adaptive_w.get('peg_gate', 1.5):.1f}
        &nbsp;→&nbsp;<span style="color:{COLORS['gold']};font-weight:700;">{mandate_fit} fit</span>
        <span style="color:{COLORS['text_muted']};">(of {gate_passed} gate-passed)</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

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

    # ── No-match dead-end → actionable empty-state (filters can narrow to zero; the engine and
    # the non-empty path below are untouched — this only ADDS the empty branch) ──
    if _disc_df.empty:
        st.markdown(
            f'<div style="text-align:center;background:{COLORS["bg_secondary"]};'
            f'border:1px dashed {COLORS["border"]};border-radius:12px;padding:28px 18px;margin-top:6px;">'
            f'<div style="font-size:1.4rem;margin-bottom:6px;">🔍</div>'
            f'<div style="font-size:0.95rem;font-weight:800;color:{COLORS["text_primary"]};">'
            f'No stocks match these filters</div>'
            f'<div style="font-size:0.72rem;color:{COLORS["text_muted"]};margin-top:4px;">'
            f'Your active filters narrowed all {len(df):,} stocks out. Loosen one — or clear everything '
            f'and start fresh.</div></div>',
            unsafe_allow_html=True,
        )
        _, _ec, _ = st.columns([3, 2, 3])
        with _ec:
            if st.button("🧹 Clear all filters", key="disc_clear", use_container_width=True):
                clear_all_filters()
    else:
        st.markdown(
            f'<div class="sec-head">🏆 Top Picks — {len(_disc_df)} stocks'
            f'{"" if _disc_sort == "🏆 Score" else f" &nbsp;·&nbsp; sorted by {_disc_sort}"}</div>',
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
        "🏆 Core":      ["rank","name","verdict_direction","sector","market_category","composite_score",
                         "data_coverage_pct","conviction_tier","gate_pass","moat_growth_quad","smart_money_flow"],
        "📊 Quality":   ["name","quality_score","moat_score","growth_score","cash_score",
                         "governance_bonus","piotroski_fscore","roce","opm","cfo_to_pat"],
        "💰 Valuation": ["name","valuation_score","expected_excess_return","pe","pb_ratio","peg",
                         "earnings_yield","fcf_yield","market_cap","buy_zone_label"],
        "🔬 Forensic":  ["name","red_flag_count","piotroski_fscore","forensic_score","forensic_multiplier",
                         "cfo_to_pat","accruals_ratio","debt_to_equity","promoter_holdings","pledged_percentage"],
        "📈 Technical": ["name","momentum_score","rsi_14d","dist_52wh","crs_52w","weinstein_stage",
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
        "data_coverage_pct":      ("Evidence",   "%.0f%%"),   # Core: score-confidence % (high score on thin data = trap)
        "forensic_multiplier":    ("Forensic ×", "%.2f"),     # Forensic: the penalty cutting composite (1.00 clean → 0.50 high-risk)
    }
    for _nc, (_nl, _nf) in _num_fmt.items():
        if _nc in _display_df.columns:
            _CC[_nc] = st.column_config.NumberColumn(_nl, help=_SCANNER_HEADER_TIPS.get(_nc), format=_nf)
    # String decision-signal + identity columns get clean headers (else they show raw snake_case).
    for _tc, _tl in {
        "name": "Stock", "sector": "Sector", "market_category": "Market Cap",
        "verdict_direction": "Verdict", "weinstein_stage": "Trend",
        "moat_growth_quad": "Moat·Growth", "smart_money_flow": "Smart Money",
        "buy_zone_label": "Buy Zone",
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
            if st.button("🧹 Clear all filters", key="ds_clear", use_container_width=True):
                clear_all_filters()
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
        # _mandate_label is _sel_mandate-or-"Custom": _sel_mandate is None for the Custom mandate
        # (and when a profile switch clears it), so derive the filename from the None-safe label.
        _safe_mandate = _mandate_label.replace(" ", "_").lower()
        st.download_button(
            f"📥 Export {len(ds_df)} stocks · {len(_export_cols)} columns — {_mandate_label} / {scoring_profile}",
            data=ds_df[_export_cols].to_csv(index=False),
            file_name=f"scan_{_safe_mandate}_{scoring_profile.lower()}.csv",
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
        _vdir  = str(_sg("verdict_direction", "AVOID") or "AVOID")
        _vstr  = str(stock.get("verdict_strength", "") or "")
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
                "BUY":   (COLORS["green"],      "rgba(63,185,80,0.08)"),
                "WATCH": (COLORS["gold"],       "rgba(228,179,65,0.07)"),
                "AVOID": (COLORS["text_muted"], "rgba(110,118,129,0.06)"),
            }
            _verdict_clr, _verdict_bg = _dir_map.get(_vdir, _dir_map["AVOID"])
            _verdict = f"{_vemoji} {_vdir}".strip()
            _verdict_reason = _vnarr or f"Tier {_tier_num} · Score {_comp_sc:.0f}/100"

        # Strength · Score · Confidence subline (engine path only)
        _meta_bits = []
        if _gate_ok and not _sell_any:
            if _vstr:
                _meta_bits.append(_vstr)
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
        _trend_conflict = (_gate_ok and not _sell_any and _vdir in ("BUY", "WATCH")
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

        st.markdown(f"""
        <div style="background:{_verdict_bg};border:1px solid {_verdict_clr}55;
             border-left:4px solid {_verdict_clr};border-radius:10px;
             padding:11px 16px;margin:6px 0 10px 0;">
          <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
            <span style="font-size:0.85rem;font-weight:900;color:{_verdict_clr};
                 letter-spacing:1.1px;white-space:nowrap;">{_verdict}</span>
            <span style="font-size:0.7rem;color:{COLORS['text_secondary']};
                 white-space:nowrap;">{_meta_line}</span>
            {_risk_pill}{_mr_pill}{_trend_pill}
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
            render_raw_signals(stock)
            # Breathing room before the Export so it doesn't crowd the last data section.
            st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)
            _stock_export = pd.DataFrame({
                "Signal": df[df["name"] == selected].iloc[0].index,
                "Value":  df[df["name"] == selected].iloc[0].values,
            })
            st.download_button(
                f"📥 Export {selected} — Full Data Row (all columns)",
                data=_stock_export.to_csv(index=False),
                file_name=f"{re.sub(r'[^A-Za-z0-9._-]+', '_', selected).lower()}_signals.csv",
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
    _mp_qglp = (df[df["qglp_pass"] == 1].sort_values("qglp_score", ascending=False)
                if "qglp_pass" in df.columns else df.iloc[:0])   # market-wide, like the other 4 sections

    # ── Market-state Pulse band (breadth-led market vitals — what the tab's name promises) ──────
    render_pulse_band(df)

    # ── Inner navigation tabs ──────────────────────────────────────
    _mp_tabs = st.tabs([
        "🌊 Tsunami",
        "🏛️ QGLP",
        "📈 Sectors",
    ])   # Stage 3: dropped dead "💙 Blue Chips" (0% fires) + brittle "🚀 Tipping Points" (folded into Sectors)

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

            _ts_cols = [c for c in ["rank","name","verdict_direction","sector","market_category","market_cap",
                                    "composite_score","quality_score","momentum_score",
                                    "piotroski_fscore","smart_money_flow","buy_zone_label"]
                        if c in _mp_ts.columns]
            _ts_sel = st.dataframe(
                _mp_ts[_ts_cols].reset_index(drop=True),
                column_config={
                    "verdict_direction": st.column_config.TextColumn("Verdict", help="The engine's overall BUY / WATCH / AVOID call — a Tsunami setup can still be WATCH/AVOID on valuation or entry timing."),
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

            _q_cols = [c for c in ["rank","name","verdict_direction","red_flag_count","sector","market_cap",
                                   "qglp_score","qglp_quality","qglp_growth","qglp_longevity","qglp_price",
                                   "smart_money_flow","buy_zone_label"]
                       if c in _mp_qglp.columns]
            _q_sel = st.dataframe(
                _mp_qglp[_q_cols].reset_index(drop=True),
                column_config={
                    "verdict_direction": st.column_config.TextColumn("Verdict", help="The engine's overall BUY / WATCH / AVOID call — most QGLP passers are WATCH/AVOID on valuation, so this surfaces the few that are buyable now."),
                    "qglp_score":     st.column_config.ProgressColumn("QGLP",      min_value=0, max_value=100, format="%.0f"),
                    "qglp_quality":   st.column_config.ProgressColumn("Quality",   min_value=0, max_value=100, format="%.0f"),
                    "qglp_growth":    st.column_config.ProgressColumn("Growth",    min_value=0, max_value=100, format="%.0f"),
                    "qglp_longevity": st.column_config.ProgressColumn("Longevity", min_value=0, max_value=100, format="%.0f"),
                    "qglp_price":     st.column_config.ProgressColumn("Price/PEG", min_value=0, max_value=100, format="%.0f"),
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

    # ══ Sectors ════════════════════════════════════════════════════
    with _mp_tabs[2]:
        st.markdown(
            "<div class='sec-cap'>Every sector with ≥5 stocks — Quality / Momentum / Valuation / Score "
            "averaged across <strong>all</strong> its stocks (sample-robust, not just the gate-passers). "
            "<strong>% Qualify</strong> = the share clearing the hard gates (the sector's quality breadth). "
            "Ranked by % Qualify (most-investable first). Capital-cycle phase is named below: 🔥 hot (over-investing — caution) · "
            "❄️ starved (under-invested — opportunity).</div>",
            unsafe_allow_html=True,
        )
        # Cap-tier filter — Market Pulse is market-wide by design (ignores the sidebar filter), so this
        # slices the WHOLE-sector aggregation by size. selectbox (not pills): cleaner for 7 options +
        # always returns a value; format_func adds the tier emoji while the option value stays the exact
        # market_category string (zero-mapping filter). Guarded if the column is absent.
        _sec_src = df
        if "market_category" in df.columns:
            from config import MCAP_TIERS
            _cap_opts = ["All"] + [t for t in MCAP_TIERS if (df["market_category"] == t).any()]
            _cf1, _ = st.columns([2, 6])
            with _cf1:
                _cap = st.selectbox(
                    "Market-cap tier", _cap_opts,
                    format_func=lambda t: t if t == "All" else f"{MCAP_TIERS[t]['emoji']} {t}",
                    key="mp_sec_cap",
                )
            if _cap != "All":
                _sec_src = df[df["market_category"] == _cap]
                st.caption(f"📊 {len(_sec_src):,} {_cap} stocks across {_sec_src['sector'].nunique()} sectors.")

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
            crown_jewels=("conviction_tier", lambda x: (x == 1).sum()),
        )
        # Sort by % Qualify (breadth), then Score — so the most-INVESTABLE sectors lead. Sorting by
        # Score alone would rank a 0%-qualify sector #1 (e.g. Financial Services scores high on
        # fundamentals but every stock fails a hard gate), which misleads at a glance.
        _sec_stats = (_sec_stats[_sec_stats["stocks"] >= 5]
                      .sort_values(["pct_qualify", "avg_composite"], ascending=False))

        if _sec_stats.empty:
            st.info("No sector has ≥5 stocks in this view — sample too small for a reliable average.")
        else:
            _sec_order = [c for c in ["stocks", "pct_qualify", "avg_quality", "avg_momentum",
                                      "avg_valuation", "avg_composite", "crown_jewels"]
                          if c in _sec_stats.columns]
            st.dataframe(
                _sec_stats[_sec_order].reset_index(),
                column_config={
                    "stocks":        st.column_config.NumberColumn("Count", format="%.0f"),
                    "pct_qualify":   st.column_config.ProgressColumn("% Qualify", min_value=0, max_value=100, format="%.0f%%",
                                       help="Share of the sector's stocks that clear all hard gates — its quality breadth. Robust to sector size."),
                    "avg_quality":   st.column_config.ProgressColumn("Quality",  min_value=0, max_value=100, format="%.0f"),
                    "avg_momentum":  st.column_config.ProgressColumn("Momentum", min_value=0, max_value=100, format="%.0f"),
                    "avg_valuation": st.column_config.ProgressColumn("Valuation",min_value=0, max_value=100, format="%.0f"),
                    "avg_composite": st.column_config.ProgressColumn("Score",    min_value=0, max_value=100, format="%.0f"),
                    "crown_jewels":  st.column_config.NumberColumn("👑 T1",      format="%.0f"),
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
        PRISM v{UI['version']} · Quantamental Intelligence · Every lens, one verdict<br>
        Dr. Malik (SSGR+8 Params) · Raamdeo (QGLP) · O'Neil (CAN-SLIM) · Mukherjea (Coffee Can)<br>
        Howard Marks (Cycles) · Philip Fisher · Peter Lynch (PEG) · Schilit (Forensics)<br>
        {total} stocks · {len(df.columns)} signals · {load_time:.1f}s pipeline<br>
        <strong>Marks Cycle Posture: {posture['label']}</strong>
    </div>
    """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 6: REFERENCE — searchable glossary (renders the 173-term _RAW_GLOSSARY single source)
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
    st.download_button(
        "📥 Download Reference (Markdown)",
        data=build_reference_markdown(_RAW_GLOSSARY, CONCEPT_REFERENCE, _FLAG_DISPLAY),
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