"""
Multibagger Discovery System — Tearsheet Visualization Layer
=============================================================
Deep-dive charts and WCS 28/29/30 framework cards for individual stocks.
All functions are PURE DISPLAY — zero sorting, grouping, or math.
Pre-calculated vectors arrive from data_engine + scoring_engine + forensic_engine.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import html as _html
from config import COLORS, CONVICTION_TIERS, TIER_COLORS


# ─── Display Utilities ──────────────────────────────────────────────────────

def _esc(val) -> str:
    """Escape HTML special characters before raw-HTML markdown injection.
    Prevents XSS and broken layouts from corporate strings like industry descriptions
    or insider-trading comments that contain quotes, ampersands, or angle brackets.
    """
    return _html.escape(str(val)) if val is not None else ""


def _g(stock: pd.Series, key: str, default=0):
    """Null-safe Series lookup. Returns default if key is missing or NaN."""
    v = stock.get(key, default)
    return default if (v is None or (isinstance(v, float) and np.isnan(v))) else v


# ─── Forensic flag badge registry (mirrors flag_descriptions in forensic_engine) ──
_FLAG_DISPLAY = {
    "rf_low_cfo_pat":       ("Low CFO/PAT (<70%) — earnings not backed by cash",           "🔴"),
    "rf_high_receivables":  ("High DSO: >120d (services) / >75d (products)",                "🟠"),
    "rf_inventory_bloat":   ("Inventory growing faster than revenue",                        "🟡"),
    "rf_rising_debt":       ("D/E rising materially (>10% relative rise AND D/E >0.3)",     "🟠"),
    "rf_ccc_worsening":     ("Cash conversion cycle worsening by >10 days",                 "🟡"),
    "rf_expense_rising":    ("Expense ratio rising >3 percentage points",                    "🟡"),
    "rf_pledge_elevated":   ("Promoter pledge >10% of shares",                              "🔴"),
    "rf_dilution":          ("Meaningful share dilution (>3% Tier 2+)",                     "🟠"),
    "rf_negative_fcf":      ("Negative FCF AND negative OCF — true cash burn",              "🔴"),
    "rf_margin_squeeze":    ("Revenue growth +5% but PAT declining — margin collapse",      "🟠"),
    "rf_high_cash_debt":    ("High cash + high debt simultaneously (Malik Shenanigan 4)",   "🟡"),
    "rf_itr_declining":     ("Inventory turnover declining >10% YoY (Malik Shenanigan 3)", "🟡"),
    "rf_ssgr_deficit":      ("Actual growth exceeds SSGR by >5% — debt-dependent growth",  "🔴"),
    "rf_high_accruals":     ("High accruals >5% of assets — Beneish TATA forensic signal",  "🔴"),
    "rf_low_fcf_ebitda":    ("FCF/EBITDA <30% — EBITDA significantly overstates real cash", "🟠"),
    "rf_fcf_to_cfo_low":    ("FCF/CFO <15% — capital trap: capex consuming all OCF",        "🟠"),
    "rf_opm_volatile":      ("OPM >30% off 5Y median — commodity trap, no pricing power",   "🟡"),
    "rf_nfat_very_low":     ("NFAT <1.5 — extreme capital intensity, growth destroys value","🟡"),
    "rf_debt_ebitda_high":  ("Debt/EBITDA >5× — Amtek Auto collapse pattern",              "🔴"),
    "rf_cwip_bloat":        ("CWIP share of assets grew >50% YoY — IL&FS balance-sheet parking", "🟠"),
    "rf_capex_mirage":      ("Rev growth >20% but capex <0.5× dep — deferred-maintenance time bomb", "🟠"),
    "rf_tax_panic":         ("Effective tax rate <10% despite PAT >0 — Sharp Practices (WCS 24)", "🔴"),
    "rf_receivables_bloat":        ("DSO expansion >20 days above sector median — relative receivables manipulation", "🟡"),
    "rf_psu_value_destruction":    ("PSU Value-Destruction Loop — low capital spread + high payout + CWIP delays", "🟠"),
    "rf_lease_inflation":          ("Ind AS 116 lease mirage — EBITDA inflated by RoU capitalisation (QSR/Retail/Aviation)", "🟡"),
}


# ═══════════════════════════════════════════════════════════════
# MOAT-GROWTH MATRIX (22nd WCS)
# ═══════════════════════════════════════════════════════════════

def render_moat_growth_matrix(df: pd.DataFrame, highlight_stock: str = None):
    """
    2D scatter — ROCE (Moat) vs PAT CAGR (Growth) for the entire filtered universe.

    Data integrity notes:
    - Stocks with NaN in BOTH axes are excluded — this is a Plotly rendering requirement,
      NOT a data deletion bug. Scatter traces cannot plot undefined coordinates.
    - Stocks with Growth_X > viewport are fully RETAINED in the dataset; the axis range
      parameter clips the visual canvas only (G9 FIX). No rows are dropped for outliers.
    """
    st.markdown("<div class='sec-head'>🧭 Moat-Growth Matrix (22nd WCS)</div>", unsafe_allow_html=True)

    plot_df = df.copy()
    plot_df["Moat_Y"]   = plot_df["roce_med_5y"].fillna(plot_df["roce"]).fillna(0)
    plot_df["Growth_X"] = plot_df["pat_gr_5y"].fillna(plot_df["pat_gr_3y"]).fillna(0)

    # Plotly requirement: rows where EITHER axis is NaN cannot appear on a scatter trace.
    # Rows are NOT dropped for being outliers — only for being literally unplottable (both NaN).
    plot_df = plot_df[plot_df["Moat_Y"].notna() & plot_df["Growth_X"].notna()]

    if len(plot_df) == 0:
        st.warning("Not enough valid data to plot the matrix.")
        return

    # G9 FIX: viewport clips at 300% Growth — all points retained, canvas just zoomed.
    x_max = max(min(float(plot_df["Growth_X"].max()) * 1.05, 300), 50)  # floor 50 prevents axis collapse when all growth=0

    fig = px.scatter(
        plot_df, x="Growth_X", y="Moat_Y",
        color="moat_growth_quad",
        color_discrete_map={
            "⭐ Wealth Creator":    COLORS["green"],
            "🛡️ Quality Trap":     COLORS["gold"],
            "⚡ Growth Trap":      COLORS["blue"],
            "💀 Wealth Destroyer": COLORS["red"],
        },
        hover_name="name",
        hover_data={"Growth_X": ":.1f", "Moat_Y": ":.1f", "moat_growth_quad": False},
        labels={"Growth_X": "Growth (PAT CAGR %)", "Moat_Y": "Moat (ROCE %)"},
    )
    fig.add_vline(x=15, line_width=1, line_dash="dash", line_color=COLORS["border"])
    fig.add_hline(y=15, line_width=1, line_dash="dash", line_color=COLORS["border"])
    fig.add_annotation(x=80,  y=80,  text="⭐ Wealth Creators", showarrow=False,
                       font=dict(color=COLORS["green"], size=16), opacity=0.3)
    fig.add_annotation(x=-20, y=80,  text="🛡️ Quality Traps",  showarrow=False,
                       font=dict(color=COLORS["gold"],  size=16), opacity=0.3)
    fig.add_annotation(x=80,  y=-10, text="⚡ Growth Traps",   showarrow=False,
                       font=dict(color=COLORS["blue"],  size=16), opacity=0.3)
    fig.add_annotation(x=-20, y=-10, text="💀 Destroyers",     showarrow=False,
                       font=dict(color=COLORS["red"],   size=16), opacity=0.3)

    if highlight_stock:
        hl = plot_df[plot_df["name"] == highlight_stock]
        if not hl.empty:
            fig.add_trace(go.Scatter(
                x=hl["Growth_X"], y=hl["Moat_Y"],
                mode="markers+text",
                marker=dict(color="white", size=15, line=dict(color="black", width=2)),
                text=["🎯 " + _esc(highlight_stock)],
                textposition="top center",
                name="Selected Stock",
                showlegend=False,
            ))

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text_primary"]),
        margin=dict(l=0, r=0, t=30, b=0), height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=COLORS["border"],
                     zeroline=True, zerolinewidth=2, zerolinecolor=COLORS["border"],
                     range=[-50, x_max])
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=COLORS["border"],
                     zeroline=True, zerolinewidth=2, zerolinecolor=COLORS["border"],
                     range=[-25, 105])
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# EP POWER CURVE MODULE — 28th WCS
# ═══════════════════════════════════════════════════════════════

def render_ep_power_curve_module(stock: pd.Series):
    """
    Economic Profit Power Curve card (28th WCS).
    Displays the stock's quintile position, EP velocity, and the Hockey-Stick
    Breakthrough badge when ep_hockey_stick_breakout fires (Q2/Q3 ascending
    the curve with institutional volume confirmation).
    """
    st.markdown("<div class='sec-head'>📈 Economic Profit Power Curve (28th WCS)</div>",
                unsafe_allow_html=True)

    ep_val      = _g(stock, "economic_profit",          0)
    ep_vel      = _g(stock, "economic_profit_velocity",
                    _g(stock, "economic_profit_delta",  0))
    ep_curve    = stock.get("ep_power_curve", "📉 Value Trap") or "📉 Value Trap"
    ep_q        = stock.get("ep_quintile",    None)
    hs_breakout = int(_g(stock, "ep_hockey_stick_breakout", 0))
    ep_positive = int(_g(stock, "economic_profit_positive",  0))

    # ── Hockey-Stick Breakthrough Banner ──────────────────────────────────
    if hs_breakout:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0b2214,#0e3320);
                    border:2px solid {COLORS['green']}; border-radius:14px;
                    padding:16px 22px; margin-bottom:14px; text-align:center;
                    box-shadow:0 0 22px rgba(63,185,80,0.40);">
            <div style="font-size:1.7rem; margin-bottom:4px;">🚀</div>
            <div style="font-size:1.05rem; font-weight:900; color:{COLORS['green']};
                        letter-spacing:1.2px;">
                HOCKEY-STICK EP BREAKTHROUGH
            </div>
            <div style="font-size:0.75rem; color:{COLORS['text_muted']}; margin-top:6px;">
                Q2/Q3 company ascending the Economic Profit Power Curve with institutional
                volume confirmation — 28th WCS structural alpha inflection signal
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Quintile Position Bar ─────────────────────────────────────────────
    _Q_LABELS = {
        1: "Alpha Creators",
        2: "Value Creators",
        3: "Mediocre",
        4: "Destroyers",
        5: "Capital Destroyers",
    }
    _Q_COLORS = {
        1: COLORS["green"],
        2: "#6ec97f",
        3: COLORS["gold"],
        4: COLORS["orange"],
        5: COLORS["red"],
    }

    ep_q_int = None
    if ep_q is not None:
        try:
            ep_q_int = int(float(ep_q))
        except (TypeError, ValueError):
            ep_q_int = None

    segs_html = ""
    for q in range(1, 6):
        is_cur  = (ep_q_int == q)
        opacity = "1.0" if is_cur else "0.38"
        border  = (f"box-shadow:0 0 0 3px #0d1117,0 0 0 5px {_Q_COLORS[q]};"
                   "font-size:0.82rem;") if is_cur else "font-size:0.7rem;"
        segs_html += f"""
            <div style="flex:1;height:36px;background:{_Q_COLORS[q]};opacity:{opacity};
                        border-radius:8px;display:flex;align-items:center;
                        justify-content:center;font-weight:800;color:#0d1117;{border}">
                Q{q}
            </div>"""

    q_label = _Q_LABELS.get(ep_q_int, "Unknown") if ep_q_int else "Not ranked"
    q_color = _Q_COLORS.get(ep_q_int, COLORS["gold"]) if ep_q_int else COLORS["gold"]

    st.markdown(f"""
    <div style="margin:8px 0 12px 0;">
        <div style="display:flex;gap:6px;margin-bottom:6px;">{segs_html}</div>
        <div style="text-align:center;font-size:0.72rem;color:{COLORS['text_muted']};">
            Q1 = Top 20% Economic Profit Earners &nbsp;·&nbsp;
            Q2/Q3 = Hockey-Stick Zone &nbsp;·&nbsp;
            Q5 = Bottom 20% Capital Destroyers
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── EP Metrics Strip ─────────────────────────────────────────────────
    vel_sign = "+" if ep_vel >= 0 else ""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Economic Profit",
        f"₹{ep_val:,.0f} Cr",
        "EP Positive ✅" if ep_positive else "EP Negative ❌",
    )
    c2.metric(
        "EP Velocity (YoY)",
        f"{vel_sign}₹{ep_vel:,.0f} Cr",
        "Ascending ↑" if ep_vel > 0 else "Descending ↓",
    )
    c3.metric(
        "Quintile Position",
        f"Q{ep_q_int}" if ep_q_int else "N/A",
        q_label,
    )
    c4.metric("EP Trajectory", _esc(ep_curve))


# ═══════════════════════════════════════════════════════════════
# BRUISED BLUE CHIP BADGE — 29th WCS
# ═══════════════════════════════════════════════════════════════

def render_bruised_blue_chip_badge(stock: pd.Series):
    """
    Bruised Blue Chip premium franchise badge (29th WCS).
    Renders ONLY when bruised_blue_chip_29 (Agent 9 large-cap elite) or
    bruised_blue_chip (fallen-quality signal) is triggered.
    No-op when neither is active — keeps the tearsheet clean.
    """
    bbc29  = int(_g(stock, "bruised_blue_chip_29", 0))
    bbc_og = int(_g(stock, "bruised_blue_chip",    0))

    if not bbc29 and not bbc_og:
        return

    # ── Agent 9 variant: Large-Cap Elite ROCE + P/B ≤ 2.0 (primary badge) ──
    if bbc29:
        mcap     = _g(stock, "market_cap",    0)
        roce_10y = _g(stock, "roce_med_10y",  0)
        pb       = _g(stock, "pb_ratio",      0)
        pe_disc  = _g(stock, "pe_discount",   0)

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#091624,#0c2040);
                    border:2px solid {COLORS['blue']};border-radius:16px;
                    padding:18px 22px;margin:10px 0;
                    box-shadow:0 0 26px rgba(88,166,255,0.22);">
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
                <div style="font-size:1.8rem;">💙</div>
                <div>
                    <div style="font-size:1.05rem;font-weight:900;color:{COLORS['blue']};">
                        BRUISED BLUE CHIP — 29th WCS
                    </div>
                    <div style="font-size:0.75rem;color:{COLORS['text_muted']};">
                        Elite franchise at a historical valuation floor
                    </div>
                </div>
            </div>
            <div style="display:flex;gap:14px;flex-wrap:wrap;">
                <span style="font-size:0.78rem;color:{COLORS['green']};">
                    ✅ MCap ≥ ₹20,000 Cr &nbsp;(₹{mcap:,.0f} Cr)
                </span>
                <span style="font-size:0.78rem;color:{COLORS['green']};">
                    ✅ ROCE 10Y ≥ 20% &nbsp;({roce_10y:.1f}%)
                </span>
                <span style="font-size:0.78rem;color:{COLORS['green']};">
                    ✅ P/B ≤ 2.0× &nbsp;({pb:.2f}×)
                </span>
                <span style="font-size:0.78rem;color:{COLORS['green']};">
                    ✅ Sector Tailwind
                </span>
            </div>
            <div style="margin-top:10px;font-size:0.75rem;color:{COLORS['text_secondary']};
                        border-top:1px solid rgba(88,166,255,0.2);padding-top:8px;">
                Capital-efficient compounder with a decade of sustained value creation
                ({roce_10y:.1f}% 10Y ROCE), now trading at P/B {pb:.2f}× —
                asymmetric risk/reward per MOSL 29th Annual Wealth Creation Study.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Original fallen-quality signal (secondary) ──
    if bbc_og:
        dist_52wh = _g(stock, "dist_52wh",         0)
        d32       = _g(stock, "d32_pe_vs_median",   0)
        roce_5y   = _g(stock, "roce_med_5y",        0)
        st.markdown(f"""
        <div style="background:rgba(88,166,255,0.07);
                    border:1px solid rgba(88,166,255,0.3);
                    border-radius:10px;padding:10px 16px;margin:6px 0;">
            <span style="font-weight:700;color:{COLORS['blue']};">💙 Fallen Quality Signal:</span>
            <span style="font-size:0.8rem;color:{COLORS['text_secondary']};">
                {dist_52wh:.0f}% off 52W high ·
                {abs(d32):.0f}% below 10Y median PE ·
                ROCE 5Y: {roce_5y:.1f}%
            </span>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# MULTI-TRILLION TIPPING POINT CARD — 30th WCS
# ═══════════════════════════════════════════════════════════════

def render_multitrillioncap_card(stock: pd.Series):
    """
    Multi-Trillion Economy Compounding Tipping Point card (30th WCS).
    Renders the full signal grid when multitrillioncap_tipping_point == 1.
    Renders a lightweight 'Sunrise Sector' note for sector_tailwind == 1 only.
    Silent when neither applies.
    """
    mtp        = int(_g(stock, "multitrillioncap_tipping_point", 0))
    sector_tw  = int(_g(stock, "sector_tailwind",                0))

    if not mtp and not sector_tw:
        return

    # ── In sunrise sector but not at tipping point yet ──
    if not mtp and sector_tw:
        sector_nm = _esc(stock.get("sector", ""))
        st.markdown(f"""
        <div style="background:rgba(139,92,246,0.07);
                    border:1px dashed rgba(139,92,246,0.4);
                    border-radius:10px;padding:10px 14px;margin:6px 0;">
            <span style="font-size:0.8rem;color:{COLORS['purple']};">🌐 Sunrise Sector</span>
            <span style="font-size:0.75rem;color:{COLORS['text_muted']};margin-left:8px;">
                {sector_nm} — structural tailwind sector (30th WCS Multi-Trillion opportunity).
                Tipping point signals not yet fully triggered.
            </span>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Full tipping point — render the complete signal grid ──
    vol_ratio  = _g(stock, "vol_ratio",  0)
    pat_gr_3y  = _g(stock, "pat_gr_3y",  0)
    q_pat_yoy  = _g(stock, "q_pat_yoy",  0)
    dist_52wh  = _g(stock, "dist_52wh",  999)
    sector_nm  = _esc(stock.get("sector",   ""))
    industry_nm= _esc(stock.get("industry", ""))

    vol_ok   = vol_ratio >= 1.5
    earn_ok  = q_pat_yoy > 25 or pat_gr_3y > 25
    break_ok = dist_52wh <= 15

    def _sig_card(label: str, fired: bool, detail: str) -> str:
        col = COLORS["green"] if fired else COLORS["text_muted"]
        bdr = "rgba(139,92,246,0.40)" if fired else "rgba(139,92,246,0.15)"
        ico = "✅" if fired else "⬜"
        return f"""
        <div style="flex:1;min-width:130px;background:rgba(139,92,246,0.09);
                    border:1px solid {bdr};border-radius:8px;
                    padding:8px 12px;text-align:center;">
            <div style="font-size:1.1rem;">{ico}</div>
            <div style="font-size:0.72rem;font-weight:700;color:{col};margin-top:2px;">{label}</div>
            <div style="font-size:0.66rem;color:{COLORS['text_muted']};">{detail}</div>
        </div>"""

    sigs_html = (
        _sig_card("Volume Surge",          vol_ok,   f"{vol_ratio:.1f}× 20D SMA") +
        _sig_card("Earnings Acceleration", earn_ok,  f"PAT 3Y CAGR {pat_gr_3y:.0f}%") +
        _sig_card("Near Breakout",         break_ok, f"{dist_52wh:.0f}% from 52W high")
    )

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#110830,#1c0d46);
                border:2px solid {COLORS['purple']};border-radius:16px;
                padding:18px 22px;margin:10px 0;
                box-shadow:0 0 26px rgba(139,92,246,0.28);">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">
            <div style="font-size:1.6rem;">🌐</div>
            <div>
                <div style="font-size:1.0rem;font-weight:900;color:{COLORS['purple']};">
                    MULTI-TRILLION COMPOUNDING TIPPING POINT — 30th WCS
                </div>
                <div style="font-size:0.74rem;color:{COLORS['text_muted']};">
                    {sector_nm} · {industry_nm} · Structural tailwind sector at critical velocity
                </div>
            </div>
        </div>
        <div style="display:flex;gap:8px;flex-wrap:wrap;">{sigs_html}</div>
        <div style="margin-top:10px;font-size:0.74rem;color:{COLORS['text_muted']};
                    border-top:1px solid rgba(139,92,246,0.2);padding-top:8px;">
            India's Financial Services and Consumer sectors are on track to 3× their combined
            market cap by 2030 per MOSL 30th Study. Stocks reaching volume + earnings + breakout
            confluence are the structural compounders at compounding tipping velocity.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# FORENSIC FRAUD PERIMETER — WCS 24 / 25-Flag Cascade
# ═══════════════════════════════════════════════════════════════

def _get_flag_context(stock: pd.Series, rf_col: str) -> str:
    """Return stock-specific metric values for a fired forensic flag.
    Matches design: 'cfo_to_pat: 54.2%  ·  threshold: ≥70%' beneath each flag title.
    Returns empty string when data is unavailable (flag row still renders without sub).
    """
    def _v(col, fmt):
        raw = stock.get(col)
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            return None
        try:
            return fmt.format(float(raw))
        except Exception:
            return None

    if rf_col == "rf_low_cfo_pat":
        v = _v("cfo_to_pat", "{:.1f}%")
        return f"cfo_to_pat: {v}  ·  threshold: ≥70%" if v else ""
    if rf_col == "rf_high_receivables":
        v = _v("days_receivable", "{:.0f}d")
        return f"DSO: {v}  ·  threshold: >75d (products) / >120d (services)" if v else ""
    if rf_col == "rf_inventory_bloat":
        rg = _v("rev_gr_yoy", "{:.1f}%")
        return f"rev_gr: {rg}  ·  inventory grew faster than revenue" if rg else ""
    if rf_col == "rf_rising_debt":
        de = _v("debt_to_equity", "{:.2f}")
        de1 = _v("debt_to_equity_1yb", "{:.2f}")
        parts = []
        if de:  parts.append(f"D/E current: {de}")
        if de1: parts.append(f"prior year: {de1}")
        return "  ·  ".join(parts)
    if rf_col == "rf_pledge_elevated":
        v = _v("pledged_percentage", "{:.1f}%")
        return f"pledge: {v}  ·  threshold: >10%" if v else ""
    if rf_col == "rf_negative_fcf":
        ocf = _v("operating_cash_flow", "₹{:,.0f} Cr")
        fcf = _v("free_cash_flow",      "₹{:,.0f} Cr")
        parts = []
        if ocf: parts.append(f"OCF: {ocf}")
        if fcf: parts.append(f"FCF: {fcf}")
        return "  ·  ".join(parts)
    if rf_col == "rf_margin_squeeze":
        rg = _v("rev_gr_yoy", "{:.1f}%")
        pg = _v("pat_gr_yoy", "{:.1f}%")
        parts = []
        if rg: parts.append(f"rev_gr: +{rg}")
        if pg: parts.append(f"pat_gr: {pg}")
        return "  ·  ".join(parts)
    if rf_col == "rf_ssgr_deficit":
        gr   = _v("pat_gr_yoy", "{:.1f}%")
        ssgr = _v("ssgr",       "{:.1f}%")
        parts = []
        if gr:   parts.append(f"actual_gr: {gr}")
        if ssgr: parts.append(f"SSGR: {ssgr}")
        return "  ·  ".join(parts)
    if rf_col == "rf_opm_volatile":
        opm  = _v("opm",       "{:.1f}%")
        opm5 = _v("opm_med_5y", "{:.1f}%")
        parts = []
        if opm:  parts.append(f"OPM: {opm}")
        if opm5: parts.append(f"5Y median: {opm5}")
        return "  ·  ".join(parts)
    if rf_col == "rf_nfat_very_low":
        v = _v("nfat", "{:.2f}×")
        return f"NFAT: {v}  ·  threshold: <1.5×" if v else ""
    if rf_col == "rf_debt_ebitda_high":
        v = _v("debt_to_ebitda", "{:.1f}×")
        return f"Debt/EBITDA: {v}  ·  threshold: >5×" if v else ""
    if rf_col == "rf_tax_panic":
        v = _v("tax_rate_est", "{:.1f}%")
        return f"tax_rate: {v}  ·  threshold: <10% despite positive PAT" if v else ""
    if rf_col == "rf_fcf_to_cfo_low":
        v = _v("fcf_to_cfo_pct", "{:.1f}%")
        return f"FCF/CFO: {v}  ·  threshold: <15%" if v else ""
    if rf_col == "rf_low_fcf_ebitda":
        v = _v("fcf_to_ebitda_pct", "{:.1f}%")
        return f"FCF/EBITDA: {v}  ·  threshold: <30%" if v else ""
    if rf_col == "rf_high_cash_debt":
        de = _v("debt_to_equity", "{:.2f}")
        return f"D/E: {de}  ·  high cash + high debt simultaneously" if de else ""
    if rf_col == "rf_receivables_bloat":
        v = _v("days_receivable", "{:.0f}d")
        return f"DSO: {v}  ·  sector-relative expansion >20d" if v else ""
    if rf_col == "rf_high_accruals":
        v = _v("accruals_to_assets", "{:.1f}%")
        return f"accruals/assets: {v}  ·  threshold: >5%" if v else ""
    if rf_col == "rf_ccc_worsening":
        v = _v("ccc", "{:.0f}d")
        return f"CCC: {v}  ·  worsened >10 days YoY" if v else ""
    if rf_col == "rf_expense_rising":
        v = _v("expense_ratio", "{:.1f}%")
        return f"expense_ratio: {v}  ·  rose >3pp" if v else ""
    if rf_col == "rf_dilution":
        dil = _v("dilution_flag", "{:.0f}")
        shares_gr = _v("shares_gr_yoy", "{:.1f}%")
        return f"share count grew: {shares_gr}  ·  Tier 2+ dilution (>3%)" if shares_gr else ""
    if rf_col == "rf_itr_declining":
        itr = _v("inventory_turnover", "{:.2f}×")
        itr1 = _v("inventory_turnover_1yb", "{:.2f}×")
        parts = []
        if itr:  parts.append(f"ITR current: {itr}")
        if itr1: parts.append(f"prior year: {itr1}")
        return "  ·  ".join(parts) if parts else ""
    if rf_col == "rf_cwip_bloat":
        cwip = _v("cwip_to_assets", "{:.1f}%")
        return f"CWIP/assets: {cwip}  ·  grew >50% YoY — balance-sheet parking risk" if cwip else ""
    if rf_col == "rf_capex_mirage":
        rg = _v("rev_gr_yoy", "{:.1f}%")
        dep = _v("depreciation", "₹{:,.0f} Cr")
        parts = []
        if rg: parts.append(f"rev_gr: +{rg}")
        if dep: parts.append(f"dep: {dep}  ·  capex <0.5× dep")
        return "  ·  ".join(parts) if parts else ""
    if rf_col == "rf_psu_value_destruction":
        roce = _v("roce", "{:.1f}%")
        de = _v("debt_to_equity", "{:.2f}")
        parts = []
        if roce: parts.append(f"ROCE: {roce}")
        if de:   parts.append(f"D/E: {de}")
        return "  ·  ".join(parts) + "  ·  PSU capital spread < cost of capital" if parts else ""
    if rf_col == "rf_lease_inflation":
        opm = _v("opm", "{:.1f}%")
        return f"EBITDA-level OPM: {opm}  ·  Ind AS 116 RoU removes operating lease costs from EBITDA" if opm else ""
    return ""


def render_forensic_perimeter(stock: pd.Series):
    """
    Vectorized Fraud Perimeter Display.
    Outputs structured, named red-flag badges (not just a count) for every fired
    forensic signal. Connects directly to the cascading forensic filter multiplier.
    """
    flag_count     = int(_g(stock, "red_flag_count",         0))
    forensic_score = _g(stock,  "forensic_score",            100)
    forensic_label = stock.get("forensic_label", "🟢 Clean") or "🟢 Clean"
    f_mult         = _g(stock,  "forensic_multiplier",       1.0)
    piotroski      = int(_g(stock, "piotroski_fscore",        0))
    pio_label      = stock.get("piotroski_label",  "") or ""
    mgmt_int       = int(_g(stock, "management_integrity_score", 0))

    mult_color = (COLORS["green"]  if f_mult == 1.0 else
                  COLORS["gold"]   if f_mult >= 0.90 else
                  COLORS["orange"] if f_mult >= 0.75 else
                  COLORS["red"])

    flag_color = ("#3fb950" if flag_count == 0 else
                  "#d29922" if flag_count <= 2 else
                  "#ff6b35" if flag_count <= 4 else
                  "#f85149")

    # ── KPI strip ────────────────────────────────────────────────────────
    pio_clr  = (COLORS["green"] if piotroski >= 7 else
                COLORS["gold"]  if piotroski >= 5 else COLORS["red"])
    fsc_clr  = (COLORS["green"] if forensic_score >= 80 else
                COLORS["gold"]  if forensic_score >= 60 else COLORS["red"])

    st.markdown(f"""
    <div class="ts-kpi-strip">
      <div class="ts-kpi-cell" style="border-top:3px solid {flag_color};">
        <div class="ts-kpi-val" style="color:{flag_color};">{flag_count}</div>
        <div class="ts-kpi-lbl">Red Flags / 25</div>
      </div>
      <div class="ts-kpi-cell" style="border-top:3px solid {fsc_clr};">
        <div class="ts-kpi-val" style="color:{fsc_clr};">{forensic_score:.0f}</div>
        <div class="ts-kpi-lbl">Forensic Score</div>
      </div>
      <div class="ts-kpi-cell" style="border-top:3px solid {mult_color};">
        <div class="ts-kpi-val" style="color:{mult_color};">{f_mult:.0%}</div>
        <div class="ts-kpi-lbl">Score Multiplier</div>
      </div>
      <div class="ts-kpi-cell" style="border-top:3px solid {pio_clr};">
        <div class="ts-kpi-val" style="color:{pio_clr};">{piotroski}/9</div>
        <div class="ts-kpi-lbl">Piotroski F-Score</div>
      </div>
      <div class="ts-kpi-cell" style="border-top:3px solid {COLORS['purple']};">
        <div class="ts-kpi-val" style="color:{COLORS['purple']};">{mgmt_int}/3</div>
        <div class="ts-kpi-lbl">Mgmt Integrity</div>
      </div>
    </div>
    <div style="font-size:0.72rem;color:{COLORS['text_muted']};margin-bottom:12px;">
      Status: <strong style="color:{COLORS['text_primary']};">{_esc(forensic_label)}</strong>
      &nbsp;·&nbsp; Piotroski: <strong style="color:{pio_clr};">{_esc(pio_label)}</strong>
    </div>
    """, unsafe_allow_html=True)

    if flag_count == 0:
        st.markdown(f"""
        <div style="background:rgba(63,185,80,0.08);border:1px solid rgba(63,185,80,0.35);
                    border-radius:10px;padding:14px 18px;text-align:center;">
          <div style="font-size:1.2rem;margin-bottom:4px;">✅</div>
          <div style="font-size:0.85rem;font-weight:700;color:{COLORS['green']};">
            Clean Bill of Health
          </div>
          <div style="font-size:0.72rem;color:{COLORS['text_muted']};margin-top:4px;">
            Zero forensic red flags across all 25 accounting checks —
            17 Schilit/Malik shenanigans + 8 WCS 24 defensive protocols.
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    if f_mult < 1.0:
        st.markdown(f"""
        <div style="background:rgba(255,107,53,0.07);border:1px solid rgba(255,107,53,0.4);
                    border-radius:8px;padding:10px 14px;margin-bottom:10px;font-size:0.78rem;">
          ⚠️ <strong style="color:{COLORS['orange']};">Cascading Forensic Filter active:</strong>
          <span style="color:{COLORS['text_secondary']};">
            composite score × {f_mult:.0%} ({flag_count} flags fired).
            Engine multiplier preserves rank ordering while proportionally penalising risk.
          </span>
        </div>
        """, unsafe_allow_html=True)

    # ── Flags grouped by severity ─────────────────────────────────────────
    _SEV_ORDER = ["🔴", "🟠", "🟡"]
    _SEV_META  = {
        "🔴": (COLORS["red"],    "rgba(248,81,73,0.08)",   "rgba(248,81,73,0.5)",  "Critical"),
        "🟠": (COLORS["orange"], "rgba(255,107,53,0.08)",  "rgba(255,107,53,0.5)", "High"),
        "🟡": (COLORS["gold"],   "rgba(228,179,65,0.08)",  "rgba(228,179,65,0.5)", "Medium"),
    }

    for sev in _SEV_ORDER:
        sev_flags = [
            (rf_col, desc)
            for rf_col, (desc, s) in _FLAG_DISPLAY.items()
            if s == sev and int(_g(stock, rf_col, 0)) == 1
        ]
        if not sev_flags:
            continue

        clr, bg, bdr, label = _SEV_META[sev]
        st.markdown(
            f'<div style="font-size:0.65rem;font-weight:800;color:{clr};'
            f'text-transform:uppercase;letter-spacing:1px;margin:14px 0 6px 0;">'
            f'{sev} {label} — {len(sev_flags)} flag{"s" if len(sev_flags)>1 else ""}</div>',
            unsafe_allow_html=True,
        )
        for rf_col, desc in sev_flags:
            parts    = desc.split(" — ", 1)
            title    = parts[0].strip()
            fallback = parts[1].strip() if len(parts) > 1 else ""
            val_ctx  = _get_flag_context(stock, rf_col)
            sub_text = val_ctx if val_ctx else fallback
            st.markdown(
                f'<div class="ts-flag-row" style="background:{bg};border-left-color:{bdr};">'
                f'<div class="ts-flag-sev">{sev}</div>'
                f'<div>'
                f'<div class="ts-flag-title" style="color:{clr};">{_esc(title)}</div>'
                f'{"<div class=ts-flag-sub>" + _esc(sub_text) + "</div>" if sub_text else ""}'
                f'</div></div>',
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════
# GURU FRAMEWORK CHECKLIST — Coffee Can / QGLP / WCS 28-30
# ═══════════════════════════════════════════════════════════════

def render_guru_frameworks(stock: pd.Series):
    """
    Displays which institutional Guru frameworks the stock passes.
    Reads pre-computed framework flags from scoring_engine — no re-computation.
    """
    fw_str = stock.get("frameworks_passed", "None") or "None"
    fw_list = [f.strip() for f in fw_str.split(",") if f.strip() and f.strip() != "None"]

    _FW_META = {
        # Names must match exactly what scoring_engine writes into frameworks_passed column
        "Coffee Can":              (COLORS["gold"],   "☕", "ROCE ≥15% for 10Y + Rev CAGR ≥10% — Mukherjea"),
        "QGLP":                    (COLORS["purple"], "🏆", "Quality + Growth + Longevity + Price — Raamdeo"),
        "Lynch Dream":             (COLORS["green"],  "📈", "PEG ≤1.0 + Rev outpacing costs — Peter Lynch"),
        "Magic Formula":           (COLORS["gold"],   "🧮", "High ROCE + High Earnings Yield — Joel Greenblatt"),
        "SMILE":                   (COLORS["green"],  "😊", "Small cap + Moat + Institutional + Longevity + Earnings"),
        "CAN SLIM":                (COLORS["blue"],   "📊", "O'Neil: EPS + Revenue + Institutional + Near High"),
        "Fallen Quality":          (COLORS["cyan"],   "🩹", "All-cap fallen quality: ROCE≥15% + PAT CAGR≥10%, >40% off 52WH, cheap vs own 10Y PE"),
        "EP Improver":             (COLORS["green"],  "📈", "Economic Profit expanding — moving up Power Curve"),
        "Peaceful Investing":      (COLORS["gold"],   "🕊️", "Vijay Malik: NFAT + self-funded growth + clean accounts"),
        "Unusual Billionaires":    (COLORS["purple"], "💰", "Saurabh Mukherjea: promoter-run compounders"),
        "Fisher Quality":          (COLORS["green"],  "🎣", "Phil Fisher 15-point scuttlebutt quality check"),
        "Fisher Scalability":      (COLORS["purple"], "📡", "Fisher operating leverage inflection — Rev runway + OpLev + Pricing + Anti-dilution"),
        "100-Bagger":              (COLORS["gold"],   "💯", "Mayer: owner-operator + small + high ROCE + low payout"),
        "Diamond":                 (COLORS["cyan"],   "💎", "Deep value: Earnings Yield ≥ G-Sec + clean accounts"),
        "Wide Moat":               (COLORS["purple"], "🏰", "Pat Dorsey: structural moat with ROCE expanding"),
        "Outsider CEO":            (COLORS["orange"], "🎯", "Thorndike: buybacks + decentralised capital allocation"),
        "Quality Compounder":      (COLORS["green"],  "⭐", "ROCE ≥ 20% + PAT CAGR ≥ 15% for 10Y — proven compounder"),
        "Dhandho Asymmetry":       (COLORS["gold"],   "🎲", "Pabrai: Heads I win, tails I don't lose much"),
        "Parikh Contrarian":       (COLORS["orange"], "🔄", "Rajeev Parikh: contrarian with forensic clean bill"),
        "Baid Compounder":         (COLORS["green"],  "📚", "Gautam Baid: 7Y ROCE ≥ 15% + 10Y Rev CAGR ≥ 12% + no-stumble consistency"),
        "Long Game Quality":       (COLORS["purple"], "⏳", "10Y consistent PAT CAGR ≥ 15% + low volatility"),
        "SEPA Momentum":           (COLORS["blue"],   "⚡", "Mark Minervini: Stage 2 + RS + Earnings acceleration"),
        "Basant 30% Club":         (COLORS["gold"],   "🏅", "Basant Maheshwari: PAT CAGR ≥ 30% for 5Y + promoter"),
        "Quality Momentum":        (COLORS["green"],  "🚀", "High quality fundamentals + price momentum confluence"),
        "MOSL Wealth Creator":     (COLORS["gold"],   "🌟", "Raamdeo's Wealth Creator criteria from annual WCS"),
        "Economic Moat":           (COLORS["purple"], "🏰", "Morningstar wide-moat: ROCE > WACC sustained 10Y+"),
        "Blue Chip Quality":       (COLORS["blue"],   "💎", "Large-cap ≥ ₹20,000 Cr + ROCE ≥ 15% + clean accounts"),
        "SQGLP Century Stock":     (COLORS["gold"],   "💯", "Epoch 4 SQGLP: all 5 pillars + Mid/Small cap entry"),
        "CAP-GAP Compounder":      (COLORS["green"],  "📐", "Capital efficiency gap: ROCE expanding vs sector peers"),
        "Consistent in Volatile":  (COLORS["orange"], "🌊", "27th WCS: consistent compounder in volatile sector — 19% CAGR"),
        "EP Hockey Stick":         (COLORS["green"],  "🚀", "Q2/Q3 ascending EP Power Curve — 28th WCS"),
        "Bruised Blue Chip 29":    (COLORS["blue"],   "💙", "Elite ROCE + large-cap at P/B ≤2× — 29th WCS"),
        "Multi-Trillion Cap":      (COLORS["purple"], "🌐", "Sunrise sector at compounding velocity — 30th WCS"),
        "100x Candidate":          (COLORS["gold"],   "🚀", "17th WCS Mouse-to-Elephant: PAT CAGR ≥20% + ROCE ≥20% + mcap ≤₹15k Cr + D/E <0.5 + ROE ≥15%"),
        "Expectations Matrix":     (COLORS["purple"], "🔮", "Mauboussin PIE: implied CAP realistic + treadmill safe + operating leverage intact"),
        "Financial Shenanigans":   (COLORS["red"],    "🕵️", "Schilit clean bill — passes accounting-manipulation forensic perimeter"),
        "Marks Cycle Shield":      (COLORS["cyan"],   "🛡️", "Howard Marks: not at cyclical-peak margins; mean-reversion risk low"),
    }

    if not fw_list:
        st.info("No institutional Guru frameworks fully met in current market configuration.")
        return

    total_fw = len(_FW_META)
    passed_n = len(fw_list)
    pct = int(passed_n / total_fw * 100)
    bar_clr = COLORS["green"] if pct >= 30 else COLORS["gold"] if pct >= 10 else COLORS["orange"]

    grid_cards = ""
    for fw in fw_list:
        color, icon, desc = _FW_META.get(fw, (COLORS["text_muted"], "✅", fw))
        grid_cards += (
            f'<div class="ts-fw-card" style="background:{color}10;border-color:{color}40;">'
            f'<div class="ts-fw-card-head">'
            f'<span style="font-size:1.1rem;">{icon}</span>'
            f'<span class="ts-fw-card-name" style="color:{color};">{_esc(fw)}</span>'
            f'</div>'
            f'<div class="ts-fw-card-desc">{_esc(desc)}</div>'
            f'</div>'
        )

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:14px;margin-bottom:14px;
                background:{COLORS['bg_secondary']};border:1px solid {COLORS['border']};
                border-radius:10px;padding:12px 16px;">
      <div style="font-size:1.8rem;font-weight:900;color:{bar_clr};">{passed_n}</div>
      <div style="flex:1;">
        <div style="font-size:0.75rem;font-weight:700;color:{COLORS['text_primary']};">
          of {total_fw} Guru Frameworks Passed
        </div>
        <div style="height:6px;background:{COLORS['bg_tertiary']};border-radius:3px;
                    margin-top:6px;overflow:hidden;">
          <div style="width:{pct}%;height:6px;background:{bar_clr};border-radius:3px;"></div>
        </div>
      </div>
      <div style="font-size:0.7rem;color:{COLORS['text_muted']};">{pct}%</div>
    </div>
    <div class="ts-fw-grid">{grid_cards}</div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SYSTEMATIC FISHER PROXY — 100% Automated from CSV
# ═══════════════════════════════════════════════════════════════

def render_fisher_module(stock: pd.Series):
    """
    Translates Philip Fisher's 15 qualitative principles into strict quantitative
    proxies using ONLY pre-derived CSV columns. Zero manual input; zero re-computation.
    Columns read directly from the pre-computed stock row (data_engine outputs).
    """
    # ── Fisher Lifecycle Quadrant Banner ─────────────────────────────────────
    # Materialised by scoring_engine fw_fisher_scalability + fw_fisher dual-engine.
    # Placed at the TOP of the Fisher module so the strategic classification is
    # the first thing a user reads before the P1-P15 proxy detail below it.
    quadrant = stock.get("fisher_lifecycle_quadrant", "⚪ Laggard") or "⚪ Laggard"
    f_score  = int(float(_g(stock, "fisher_score", 0)))

    _q_colors = {
        "👑 Apex Winner":       "#bc8cff",   # purple  — quality + scalability firing
        "🐢 Steady Compounder": "#58a6ff",   # blue    — quality proven, no inflection
        "⚡ Catalyst Play":     "#d29922",   # gold    — inflection without structural quality
        "⚪ Laggard":           "#8b949e",   # grey    — neither gate passing
    }
    _q_descriptions = {
        "👑 Apex Winner":       "Elite quality business AT its operating leverage peak — prime entry signal",
        "🐢 Steady Compounder": "Structural quality proven; no current scalability inflection — steady long hold",
        "⚡ Catalyst Play":     "Inflection firing but structural quality absent — trading candidate, cap position size",
        "⚪ Laggard":           "Fails both Fisher Quality and Scalability gates — structural irrelevance",
    }
    q_clr  = _q_colors.get(quadrant, "#8b949e")
    q_desc = _q_descriptions.get(quadrant, "")

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d1117,#161b22);
                border:1px solid {q_clr}44;
                border-left:4px solid {q_clr};
                border-radius:12px;
                padding:14px 20px;margin-bottom:16px;
                box-shadow:0 2px 12px {q_clr}22;">
      <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;">
        <div style="flex:1;">
          <div style="font-size:0.62rem;font-weight:800;color:{q_clr};
                      text-transform:uppercase;letter-spacing:1.5px;margin-bottom:4px;">
            Fisher Lifecycle Quadrant
          </div>
          <div style="font-size:1.05rem;font-weight:900;color:{q_clr};">
            {_esc(quadrant)}
          </div>
          <div style="font-size:0.71rem;color:#8b949e;margin-top:4px;">
            {_esc(q_desc)}
          </div>
        </div>
        <div style="text-align:center;flex-shrink:0;">
          <div style="font-size:1.5rem;font-weight:900;color:{q_clr};line-height:1;">
            {f_score}/4
          </div>
          <div style="font-size:0.6rem;color:#8b949e;text-transform:uppercase;
                      letter-spacing:0.8px;margin-top:2px;">
            Scalability Gates
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    proxies = []

    # P1 — Market Potential: Revenue growth → rev_gr_5y (pre-computed 5Y CAGR)
    rev_gr = _g(stock, "rev_gr_5y", 0)
    proxies.append(("P1: Market Potential (Sales Growth >15%)", rev_gr >= 15, f"{rev_gr:.1f}%"))

    # P4 — Sales Org Efficiency: Profit CAGR > Revenue CAGR → operating_leverage binary
    p4_pass = int(_g(stock, "operating_leverage", 0)) == 1
    proxies.append(("P4: Sales Org Efficiency (Profit Gr > Sales Gr)",
                    p4_pass, "Passed" if p4_pass else "Failed"))

    # P5 — Worthwhile Margins: npm from pre-computed data_engine column
    npm = _g(stock, "npm", 0)
    proxies.append(("P5: Worthwhile Margins (NPM >10%)", npm >= 10, f"{npm:.1f}%"))

    # P6 — Maintaining Margins: current npm vs npm_1yb (1-year-back column)
    npm_1yb = _g(stock, "npm_1yb", 0)
    p6_pass = npm >= npm_1yb and npm > 0
    proxies.append(("P6: Margin Trajectory (NPM ≥ Last Year)",
                    p6_pass, "Improving" if p6_pass else "Declining"))

    # P10 — Accounting Controls: cfo_to_pat is PERCENTAGE in CSV (73.04 = 73%)
    # Threshold must be 70 (not 0.7). Confirmed unit: data_engine stores raw CSV value.
    cfo_pat = _g(stock, "cfo_to_pat", 0)
    proxies.append(("P10: Accounting Controls (CFO/PAT ≥70%)", cfo_pat >= 70, f"{cfo_pat:.1f}%"))

    # P13 — No Equity Dilution: dilution_flag = 0 means clean (no meaningful dilution)
    p13_pass = int(_g(stock, "dilution_flag", 1)) == 0
    proxies.append(("P13: No Equity Dilution (Share Count Stable)",
                    p13_pass, "Clean" if p13_pass else "Diluted"))

    # P15 — Management Integrity: forensic_label pre-classified by forensic_engine
    forensic_label = stock.get("forensic_label", "") or ""
    p15_pass = forensic_label == "🟢 Clean"
    proxies.append(("P15: Accounting Integrity (Clean/Watch)",
                    p15_pass, _esc(forensic_label)))

    passed     = sum(1 for _, is_pass, _ in proxies if is_pass)
    total      = len(proxies)
    score_pct  = (passed / total) * 100
    gauge_color = (COLORS["green"] if score_pct >= 80 else
                   COLORS["gold"]  if score_pct >= 50 else
                   COLORS["red"])

    # ── Score summary bar ─────────────────────────────────────────────────
    verdict = ("🟢 High Quality Alignment" if score_pct >= 80 else
               "🟡 Moderate Alignment"     if score_pct >= 50 else
               "🔴 Low Alignment — Review Carefully")
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:16px;
                background:{COLORS['bg_secondary']};border:1px solid {COLORS['border']};
                border-radius:10px;padding:12px 16px;margin-bottom:14px;">
      <div style="font-size:2.2rem;font-weight:900;color:{gauge_color};">{passed}/{total}</div>
      <div style="flex:1;">
        <div style="font-size:0.75rem;font-weight:700;color:{COLORS['text_primary']};">{verdict}</div>
        <div style="height:6px;background:{COLORS['bg_tertiary']};border-radius:3px;
                    margin-top:6px;overflow:hidden;">
          <div style="width:{score_pct:.0f}%;height:6px;background:{gauge_color};
                      border-radius:3px;"></div>
        </div>
      </div>
      <div style="font-size:0.7rem;color:{COLORS['text_muted']};">Fisher Score</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Card grid ─────────────────────────────────────────────────────────
    cards_html = ""
    for desc, is_pass, val in proxies:
        clr  = COLORS["green"] if is_pass else COLORS["red"]
        bg   = f"{clr}0d"
        bdr  = f"{clr}40"
        ico  = "✅" if is_pass else "❌"
        # Extract short key (e.g. "P1: Market Potential" → "P1")
        short = desc.split(":")[0].strip() if ":" in desc else desc[:4]
        long  = desc.split(":", 1)[1].strip() if ":" in desc else desc
        cards_html += (
            f'<div class="ts-fisher-card" style="background:{bg};border-color:{bdr};">'
            f'<div class="ts-fisher-head">'
            f'<span style="font-size:0.85rem;">{ico}</span>'
            f'<span class="ts-fisher-key" style="color:{clr};">{_esc(short)}</span>'
            f'<span style="font-size:0.66rem;color:{COLORS["text_muted"]};">'
            f'{_esc(long)}</span>'
            f'</div>'
            f'<div class="ts-fisher-val" style="color:{clr};">{_esc(val)}</div>'
            f'</div>'
        )

    st.markdown(f'<div class="ts-fisher-grid">{cards_html}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# FINANCIAL INSIGHTS PANEL — Translated Business Analysis
# ═══════════════════════════════════════════════════════════════

def render_financial_insights(stock: pd.Series):
    """
    Translates raw CSV metrics into human-language grouped verdicts.
    4 cards: Business Quality · Cash & Debt · Valuation · Ownership.
    Replaces the raw metric grid rows in the Tear-Sheet main view.
    """

    def _row(label: str, passed, value_str: str, context: str = "") -> str:
        if passed is True:
            ico, clr = "✅", COLORS["green"]
        elif passed is False:
            ico, clr = "❌", COLORS["red"]
        else:
            ico, clr = "⚪", COLORS["text_muted"]
        c_sec = COLORS["text_secondary"]
        c_mut = COLORS["text_muted"]
        # Context truncates with ellipsis — value NEVER wraps (white-space:nowrap;flex-shrink:0)
        ctx = (
            f'<span style="color:{c_mut};font-size:0.68rem;flex:1;min-width:0;'
            f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'
            f'margin-left:6px;">{_esc(context)}</span>'
        ) if context else ""
        return (
            f'<div style="display:flex;align-items:center;gap:6px;padding:5px 0;'
            f'border-bottom:1px solid rgba(255,255,255,0.04);">'
            f'<span style="font-size:0.85rem;width:18px;flex-shrink:0;">{ico}</span>'
            f'<span style="font-size:0.76rem;color:{c_sec};width:155px;flex-shrink:0;">'
            f'{_esc(label)}</span>'
            f'<span style="font-size:0.80rem;font-weight:700;color:{clr};'
            f'white-space:nowrap;flex-shrink:0;">'
            f'{_esc(value_str)}</span>'
            f'{ctx}</div>'
        )

    def _card(title: str, icon: str, rows_html: str, border: str) -> str:
        bg = COLORS["bg_secondary"]
        bdr = COLORS["border"]
        return (
            f'<div style="background:{bg};border:1px solid {bdr};'
            f'border-left:3px solid {border};border-radius:10px;'
            f'padding:14px 16px;margin-bottom:12px;">'
            f'<div style="font-size:0.68rem;font-weight:800;color:{border};'
            f'text-transform:uppercase;letter-spacing:1.2px;margin-bottom:9px;">'
            f'{icon}&nbsp; {_esc(title)}</div>'
            f'{rows_html}'
            f'</div>'
        )

    # ── Business Quality ──
    _r10y_raw = _g(stock, "roce_med_10y", None)   # None = missing; 0 = real zero
    roce_10y  = _r10y_raw if _r10y_raw is not None else _g(stock, "roce", 0)
    roce_curr = _g(stock, "roce", 0)
    pat_5y    = _g(stock, "pat_gr_5y", 0)
    rev_5y    = _g(stock, "rev_gr_5y", 0)
    op_lev    = int(_g(stock, "operating_leverage", 0))
    npm       = _g(stock, "npm", 0)
    npm_5y    = _g(stock, "npm_med_5y", npm)

    bq = ""
    bq += _row(
        "ROCE — 10Y Median",
        roce_10y >= 15,
        f"{roce_10y:.1f}%",
        f"Current {roce_curr:.1f}% · {'Accelerating ↑' if roce_curr >= roce_10y else 'Decelerating ↓'}",
    )
    bq += _row(
        "Profit CAGR — 5 Years",
        pat_5y >= 15,
        f"{pat_5y:.1f}% p.a.",
        f"vs Revenue {rev_5y:.1f}% · {'Expanding margin ✅' if pat_5y > rev_5y else 'Margin pressure ⚠️'}",
    )
    bq += _row(
        "Sales→Profit Conversion",
        op_lev == 1,
        "Positive ✅" if op_lev else "Negative",
        "PAT CAGR > Revenue CAGR → scalable cost structure",
    )
    bq += _row(
        "Net Margin — 5Y Median",
        npm_5y >= 10,
        f"{npm_5y:.1f}%",
        f"Current {npm:.1f}% · {'Stable/Improving' if npm >= npm_5y * 0.95 else 'Declining'}",
    )

    # ── Cash & Debt Quality ──
    cfo_pat = _g(stock, "cfo_to_pat", 0)
    ssgr    = _g(stock, "ssgr", 0)
    ssgr_c  = _g(stock, "ssgr_cushion", 0)
    ssgr_sf = int(_g(stock, "ssgr_self_funded", 0))
    de      = _g(stock, "debt_to_equity", 0)
    tax     = _g(stock, "tax_rate_est", 0)

    cd = ""
    cd += _row(
        "Cash Earnings (CFO/PAT)",
        cfo_pat >= 70,
        f"{cfo_pat:.1f}%",
        "≥70%: real cash  |  50–70%: watch  |  <50%: accrual risk",
    )
    ssgr_txt = (
        f"Self-Funded — SSGR {ssgr:.1f}% covers growth ({ssgr_c:.1f}% cushion)"
        if ssgr_sf else
        f"External Capital — actual growth exceeds SSGR {ssgr:.1f}%"
    )
    cd += _row("Growth Funding (SSGR)", ssgr_sf == 1, ssgr_txt, "")
    # Distinguish truly debt-free (int_cov data absent AND D/E near zero) from
    # missing coverage data (company has debt but interest_coverage not in CSV).
    _int_raw = _g(stock, "interest_coverage", None)  # None = NaN/missing
    if _int_raw is None:
        int_pass = de < 0.05
        int_str  = "Debt-free" if de < 0.05 else "Coverage N/A"
    elif _int_raw > 0.01:
        int_pass = _int_raw >= 3
        int_str  = f"{_int_raw:.1f}×"
    else:
        int_pass = de < 0.05
        int_str  = "Debt-free" if de < 0.05 else "Coverage 0×"
    cd += _row(
        "Debt Safety",
        int_pass,
        int_str,
        f"D/E {de:.2f}  |  Safe: Int.Cov ≥3× or near-zero debt",
    )
    tax_ok = (30 <= tax <= 55) if tax > 5 else None
    cd += _row(
        "Tax Rate — Malik P3 proxy",
        tax_ok,
        f"{tax:.1f}%",
        "Normal band 30–55%  |  <10%: Sharp Practices flag",
    )

    # ── Valuation ──
    pe_disc  = _g(stock, "pe_discount", 0)
    fcf_y    = _g(stock, "fcf_yield", 0)
    ey       = _g(stock, "earnings_yield", 0)
    peg      = _g(stock, "peg", 0)
    peg_zone = str(stock.get("peg_zone", "") or "")   # _row() escapes it

    vl = ""
    if pe_disc > 1:
        vl += _row("P/E vs 10Y Average", pe_disc >= 15,
                   f"{pe_disc:.1f}% below avg", "≥20%: historically cheap  |  0–20%: fair")
    elif pe_disc < -1:
        vl += _row("P/E vs 10Y Average", False,
                   f"{abs(pe_disc):.1f}% above avg", "Trading at premium to historical mean")
    else:
        vl += _row("P/E vs 10Y Average", None, "At median", "Fair value territory")

    vl += _row(
        "PEG Ratio",
        (0 < peg <= 1.0) if peg > 0 else None,
        f"{peg:.2f}×" if peg > 0 else "N/A",
        f"{peg_zone}  |  Lynch rule: ≤1.0 = growth at bargain price",
    )
    vl += _row(
        "Earnings Yield",
        ey >= 4,
        f"{ey:.1f}%",
        "Bond-equity benchmark: ≥4% justifies equity risk",
    )
    if fcf_y > 0:
        vl += _row(
            "FCF Yield",
            fcf_y >= 3,
            f"{fcf_y:.1f}%",
            "≥4%: excellent  |  2–4%: reasonable  |  <2%: low",
        )

    # ── Ownership Alignment ──
    prom    = _g(stock, "promoter_holdings", 0)
    pledge  = _g(stock, "pledged_percentage", 0)
    fii     = _g(stock, "fii_holdings", 0)
    dii     = _g(stock, "dii_holdings", 0)
    ch_prom = _g(stock, "change_promoter_lq", 0)
    smart   = str(stock.get("smart_money_flow", "⚪ Neutral") or "⚪ Neutral")  # _row() escapes it

    _dir = "↑" if ch_prom > 0.05 else ("↓" if ch_prom < -0.05 else "→")
    prom_ctx = (
        f"{_dir} {abs(ch_prom):.1f}% last Q  |  "
        f"{'Dynasty ≥60%' if prom >= 60 else ('Well-aligned ≥50%' if prom >= 50 else 'Below ideal <50%')}"
    )
    ow = ""
    ow += _row("Promoter Holding", prom >= 50, f"{prom:.1f}%", prom_ctx)
    ow += _row("Promoter Pledge",  pledge <= 10, f"{pledge:.1f}%",
               "0%: clean  |  <5%: low risk  |  >10%: red flag")
    ow += _row("FII + DII Holdings", fii >= 5 or dii >= 5,
               f"FII {fii:.1f}%  ·  DII {dii:.1f}%",
               f"Smart Money: {smart}")

    # ── Render in 2-column layout ──
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(_card("Business Quality", "🏭", bq, COLORS["purple"]),
                    unsafe_allow_html=True)
        st.markdown(_card("Valuation", "💰", vl, COLORS["gold"]),
                    unsafe_allow_html=True)
    with col2:
        st.markdown(_card("Cash & Debt Quality", "💵", cd, COLORS["green"]),
                    unsafe_allow_html=True)
        st.markdown(_card("Ownership Alignment", "👥", ow, COLORS["blue"]),
                    unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# STOCK HERO HEADER — Premium identity card
# ═══════════════════════════════════════════════════════════════

def render_stock_hero(stock: pd.Series, regime: str = "SIDEWAYS", tier_colors: dict = None):
    """
    Full-width premium hero header. Displays stock identity, composite score ring,
    tier badge, moat quad, forensic status, regime, and all active pills in one card.
    Pure display — reads pre-computed columns only.
    """
    tier_num   = int(_g(stock, "conviction_tier", 5))
    tc         = (tier_colors or TIER_COLORS).get(tier_num, TIER_COLORS[5])
    tcfg       = next((t for t in CONVICTION_TIERS if t["tier"] == tier_num), CONVICTION_TIERS[-1])
    comp       = float(_g(stock, "composite_score", 0))
    name       = _esc(stock.get("name", "N/A") or "N/A")
    rank       = int(_g(stock, "rank", 0))
    sector     = _esc(stock.get("sector", "") or "")
    industry   = _esc(stock.get("industry", "") or "")
    mcap       = _g(stock, "market_cap", 0)
    mcat       = _esc(stock.get("market_category", "") or "")
    mg_quad    = _esc(stock.get("moat_growth_quad", "") or "")
    f_lbl      = _esc(stock.get("forensic_label", "🟢 Clean") or "🟢 Clean")
    reg_map    = {"BULL": ("🟢 Bull", COLORS["green"]), "BEAR": ("🔴 Bear", COLORS["red"])}
    reg_txt, reg_clr = reg_map.get(regime, ("🟡 Sideways", COLORS["gold"]))

    # Score ring color — matches tier
    ring_clr = tc["text"]
    ring_bdr = tc["border"]

    # ── Active pills (catalysts + frameworks + special signals) ──
    pill_items = []
    _CAT_PILLS = [
        ("cat_capacity",        COLORS["blue"],   "🔥 Capacity Explosion"),
        ("cat_oplev",           COLORS["green"],  "🔥 OpLev Inflection"),
        ("cat_inst_discovery",  COLORS["purple"], "🔥 Inst Discovery"),
        ("cat_deleveraging",    COLORS["gold"],   "🔥 Deleveraging"),
        ("cat_lynch_dream",     COLORS["green"],  "🔥 Lynch Dream"),
    ]
    for col, clr, lbl in _CAT_PILLS:
        if int(_g(stock, col, 0)) == 1:
            pill_items.append((lbl, clr))

    if int(_g(stock, "tsunami_signal", 0)) == 1:
        pill_items.append(("🌊 Tsunami", COLORS["purple"]))
    if int(_g(stock, "net_debt_negative", 0)) == 1:
        pill_items.append(("💰 Net Cash", COLORS["green"]))
    if int(_g(stock, "bruised_blue_chip_29", 0)) == 1:
        pill_items.append(("💙 Bruised Blue Chip", COLORS["blue"]))

    fw_str  = stock.get("frameworks_passed", "None") or "None"
    fw_list = [f.strip() for f in fw_str.split(",") if f.strip() and f.strip() != "None"]
    for fw in fw_list[:8]:  # cap pills at 8 frameworks to avoid overflow
        pill_items.append((f"🏛️ {fw}", COLORS["text_secondary"]))

    pills_html = "".join(
        f'<span style="display:inline-block;padding:3px 10px;border-radius:20px;'
        f'font-size:0.68rem;font-weight:600;margin:2px 3px;'
        f'background:{clr}18;border:1px solid {clr}55;color:{clr};">{lbl}</span>'
        for lbl, clr in pill_items
    )

    # ── Tier / status badges ──
    def _badge(txt, clr):
        return (f'<span style="display:inline-flex;align-items:center;'
                f'padding:4px 12px;border-radius:20px;font-size:0.72rem;font-weight:700;'
                f'background:{clr}18;border:1px solid {clr}55;color:{clr};margin:2px 3px;">'
                f'{txt}</span>')

    _mg_badge = (
        _badge(mg_quad, COLORS["green"] if "Wealth Creator" in mg_quad else
                        COLORS["gold"]  if "Quality Trap"   in mg_quad else
                        COLORS["blue"]  if "Growth Trap"    in mg_quad else COLORS["red"])
    ) if mg_quad else ""
    badges_html = (
        _badge(f"{tcfg['emoji']} {tcfg['label']}", ring_clr) +
        _mg_badge +
        _badge(f_lbl,   COLORS["green"]  if "Clean"   in f_lbl else
                        COLORS["gold"]   if "Watch"   in f_lbl else
                        COLORS["orange"] if "Caution" in f_lbl else COLORS["red"]) +
        _badge(reg_txt, reg_clr)
    )

    st.markdown(f"""
    <div class="ts-hero">
      <div style="display:flex;align-items:center;justify-content:space-between;gap:20px;position:relative;">
        <!-- Identity column -->
        <div style="flex:1;min-width:0;">
          <div style="font-size:0.65rem;color:{COLORS['text_muted']};letter-spacing:2px;
                      text-transform:uppercase;margin-bottom:5px;">
            #{rank} &nbsp;·&nbsp; {sector} &nbsp;·&nbsp; {mcat}
          </div>
          <div style="font-size:2.1rem;font-weight:900;color:{COLORS['text_primary']};
                      line-height:1.1;word-break:break-word;">{name}</div>
          <div style="font-size:0.78rem;color:{COLORS['text_muted']};margin-top:4px;">
            {industry} &nbsp;·&nbsp; ₹{mcap:,.0f} Cr
          </div>
          <div style="margin-top:12px;">{badges_html}</div>
        </div>
        <!-- Score ring -->
        <div class="ts-score-ring" style="border-color:{ring_bdr};
             box-shadow:0 0 28px {ring_bdr},inset 0 0 20px rgba(0,0,0,0.4);">
          <div class="ts-score-val" style="color:{ring_clr};">{comp:.0f}</div>
          <div class="ts-score-lbl" style="color:{ring_clr};">/ 100</div>
        </div>
      </div>
      <!-- Pills row -->
      <div style="margin-top:16px;border-top:1px solid {COLORS['border']};
                  padding-top:12px;line-height:2.2;">
        {pills_html if pills_html else
         f'<span style="font-size:0.7rem;color:{COLORS["text_muted"]};">No active catalyst or framework signals</span>'}
      </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SCORE STRIP — 5-score visual overview
# ═══════════════════════════════════════════════════════════════

def render_score_strip(stock: pd.Series):
    """
    Horizontal 5-cell strip: Moat · Growth · Cash · Momentum · Governance.
    Each cell has big number + colored mini progress bar.
    """
    moat  = float(_g(stock, "moat_score",       0))
    grow  = float(_g(stock, "growth_score",      0))
    cash  = float(_g(stock, "cash_score",        0))
    mom   = float(_g(stock, "momentum_score",    0))
    gov   = float(_g(stock, "governance_bonus",  0))

    def _cell(label: str, icon: str, val: float, color: str) -> str:
        w   = max(0.0, min(100.0, val))
        neg = val < 0
        disp = f"{val:+.0f}" if neg else f"{val:.0f}"
        zone = ("Strong" if val >= 70 else "Average" if val >= 40 else "Weak") if not neg else "Penalty"
        zone_clr = (COLORS["green"] if val >= 70 else
                    COLORS["gold"]  if val >= 40 else COLORS["red"])
        return (
            f'<div class="ts-score-cell" style="border-top:3px solid {color};">'
            f'<div class="ts-score-cell-lbl">{icon} {label}</div>'
            f'<div class="ts-score-cell-val" style="color:{color};">{disp}</div>'
            f'<div class="ts-score-bar-bg"><div class="ts-score-bar-fill" '
            f'style="width:{w:.1f}%;background:{color};"></div></div>'
            f'<div style="font-size:0.52rem;color:{zone_clr};margin-top:4px;'
            f'text-transform:uppercase;letter-spacing:0.6px;font-weight:700;">{zone}</div>'
            f'</div>'
        )

    cells = (
        _cell("Moat",       "🏰", moat, COLORS["purple"]) +
        _cell("Growth",     "🌱", grow, COLORS["green"])  +
        _cell("Cash",       "💵", cash, COLORS["blue"])   +
        _cell("Momentum",   "⚡", mom,  COLORS["orange"]) +
        _cell("Governance", "👑", gov,  COLORS["gold"])
    )
    st.markdown(f'<div class="ts-score-strip">{cells}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SELL ALERTS PANEL — Prominent multi-alert display
# ═══════════════════════════════════════════════════════════════

def render_sell_alerts_panel(stock: pd.Series):
    """
    Renders sell alert banners prominently.
    Shows a green confirmation when no alerts are active — silence was confusing users.
    Shows each fired alert as a distinct colored banner with explanation.
    """
    has_any = int(_g(stock, "sell_alert_any", 0)) == 1
    if not has_any:
        st.markdown(f"""
        <div style="background:rgba(63,185,80,0.07);border:1px solid rgba(63,185,80,0.3);
                    border-radius:10px;padding:11px 18px;margin:4px 0 12px 0;
                    display:flex;align-items:center;gap:10px;">
          <span style="font-size:1rem;">✅</span>
          <div>
            <div style="font-size:0.78rem;font-weight:700;color:{COLORS['green']};">
              No Exit Signals Active
            </div>
            <div style="font-size:0.68rem;color:{COLORS['text_muted']};margin-top:1px;">
              All 6 Baid/Howard Marks/Mauboussin sell triggers checked — none fired.
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    _ALERTS = [
        ("sell_alert_thesis_broken",    COLORS["red"],    "🔴",
         "Investment Thesis Broken",
         "ROCE trajectory is declining structurally — the business moat is eroding. "
         "Quality score is automatically penalized. Re-evaluate the competitive position."),
        ("sell_alert_mgmt_deteriorated", COLORS["orange"], "🟠",
         "Management Deterioration",
         "Pledge rising + promoter selling + D/E rising simultaneously — "
         "insider confidence is collapsing. Three independent signals, all firing together."),
        ("sell_alert_cash_collapse",     COLORS["red"],    "🔴",
         "Cash Quality Collapse",
         "CFO/PAT dropped below 50% — reported profits are no longer backed by cash. "
         "Baid's #1 red flag: earnings may be fictional."),
        ("sell_alert_overvalued",        COLORS["gold"],   "🟡",
         "Price Excess (Howard Marks)",
         "PEG > 2.5 or P/E > 30% above own 10Y median. Even great businesses are terrible "
         "investments at extreme prices. Howard Marks' extreme caution zone."),
        ("sell_alert_treadmill",         COLORS["orange"], "🟠",
         "Growth Treadmill (Mauboussin)",
         "P/E > 50× implies 15-20 year CAP assumption, but growth is decelerating AND ROCE "
         "is declining. The machine is priced for perfection and visibly slipping."),
        ("sell_alert_sequential_decline", COLORS["red"],   "🔴",
         "Sequential Revenue Collapse",
         "Current year revenue negative + 3Y CAGR also negative + PAT declining. "
         "Not a one-bad-year blip — this is structural multi-year collapse. Exit signal."),
    ]

    banners_html = ""
    for col, clr, sev, title, body in _ALERTS:
        if int(_g(stock, col, 0)) != 1:
            continue
        banners_html += (
            f'<div class="ts-sell-banner" style="background:{clr}0d;border-color:{clr}55;">'
            f'<div class="ts-sell-icon">{sev}</div>'
            f'<div>'
            f'<div class="ts-sell-title" style="color:{clr};">{title}</div>'
            f'<div class="ts-sell-body" style="color:{COLORS["text_secondary"]};">{body}</div>'
            f'</div>'
            f'</div>'
        )

    st.markdown(f"""
    <div style="background:rgba(248,81,73,0.06);border:1px solid rgba(248,81,73,0.4);
                border-radius:14px;padding:14px 18px;margin:8px 0 16px 0;">
      <div style="font-size:0.72rem;font-weight:800;letter-spacing:1.5px;color:{COLORS['red']};
                  text-transform:uppercase;margin-bottom:12px;">
        🚨 &nbsp;Sell Alert(s) Active — Review Before Holding
      </div>
      {banners_html}
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# RAW SIGNALS PANEL — Structured metric grid (replaces expander)
# ═══════════════════════════════════════════════════════════════

def render_raw_signals(stock: pd.Series):
    """
    Renders all raw numeric signals in a clean labeled grid.
    Grouped into logical clusters. Used inside the 'All Data' inner tab.
    """
    def _cell(label: str, val, fmt: str = "") -> str:
        if isinstance(val, float) and np.isnan(val):
            disp = "N/A"
        elif fmt:
            try:
                disp = fmt.format(val)
            except Exception:
                disp = str(val)
        else:
            disp = str(val) if val is not None else "N/A"
        return (
            f'<div class="ts-raw-cell">'
            f'<div class="ts-raw-lbl">{_esc(label)}</div>'
            f'<div class="ts-raw-val">{_esc(disp)}</div>'
            f'</div>'
        )

    def _section(title: str, color: str, cells_html: str):
        st.markdown(
            f'<div style="font-size:0.7rem;font-weight:800;color:{color};'
            f'text-transform:uppercase;letter-spacing:1px;margin:18px 0 8px 0;">'
            f'{title}</div>'
            f'<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));'
            f'gap:6px;">{cells_html}</div>',
            unsafe_allow_html=True,
        )

    g = lambda k, d=0: _g(stock, k, d)

    # Quality
    _section("🏭 Business Quality", COLORS["purple"],
        _cell("ROCE Current",  g("roce"),           "{:.1f}%") +
        _cell("ROCE 10Y Med",  g("roce_med_10y"),   "{:.1f}%") +
        _cell("ROCE 5Y Med",   g("roce_med_5y"),    "{:.1f}%") +
        _cell("ROE Current",   g("roe"),            "{:.1f}%") +
        _cell("ROE 10Y Med",   g("roe_med_10y"),    "{:.1f}%") +
        _cell("NPM",           g("npm"),            "{:.1f}%") +
        _cell("NPM 5Y Med",    g("npm_med_5y"),     "{:.1f}%") +
        _cell("OPM",           g("opm"),            "{:.1f}%") +
        _cell("Malik Score",   g("malik_score"),    "{:.0f}/5") +
        _cell("Malik Pass",    "Yes ✅" if g("malik_pass") == 1 else "No", "") +
        _cell("Lynch Score",   g("lynch_score"),    "{:.0f}/4") +
        _cell("Lynch Pass",    "Yes ✅" if g("lynch_pass") == 1 else "No", "") +
        _cell("Piotroski",     g("piotroski_fscore"),"{:.0f}/9") +
        _cell("Fisher Scal. Score", g("fisher_score"),   "{:.0f}/4") +
        _cell("Fisher Quadrant",    stock.get("fisher_lifecycle_quadrant", "⚪ Laggard") or "⚪ Laggard", "")
    )

    # Growth
    _section("🌱 Growth", COLORS["green"],
        _cell("PAT 5Y CAGR",   g("pat_gr_5y"),      "{:.1f}%") +
        _cell("PAT 3Y CAGR",   g("pat_gr_3y"),      "{:.1f}%") +
        _cell("PAT YoY",       g("pat_gr_yoy"),     "{:.1f}%") +
        _cell("Rev 10Y CAGR",  g("rev_gr_10y"),     "{:.1f}%") +
        _cell("Rev 5Y CAGR",   g("rev_gr_5y"),      "{:.1f}%") +
        _cell("Rev YoY",       g("rev_gr_yoy"),     "{:.1f}%") +
        _cell("EPS 5Y CAGR",   g("eps_gr_5y"),      "{:.1f}%") +
        _cell("EPS YoY",       g("eps_gr_yoy"),     "{:.1f}%") +
        _cell("Q PAT YoY",     g("q_pat_yoy"),      "{:.1f}%") +
        _cell("Op Leverage",   "Yes" if g("operating_leverage") == 1 else "No", "")
    )

    # Cash & Debt
    _section("💵 Cash & Debt", COLORS["blue"],
        _cell("CFO/PAT",       g("cfo_to_pat"),      "{:.1f}%") +
        _cell("FCF Yield",     g("fcf_yield"),       "{:.1f}%") +
        _cell("FCF/CFO",       g("fcf_to_cfo_pct"),  "{:.1f}%") +
        _cell("FCF/PAT",       g("d28_fcf_to_pat_pct"), "{:.1f}%") +
        _cell("SSGR",          g("ssgr"),            "{:.1f}%") +
        _cell("SSGR Cushion",  g("ssgr_cushion"),    "{:.1f}%") +
        _cell("D/E Ratio",     g("debt_to_equity"),  "{:.2f}") +
        _cell("Int Coverage",  g("interest_coverage"),"{:.1f}×") +
        _cell("Current Ratio", g("current_ratio"),   "{:.2f}") +
        _cell("Tax Rate Est",  g("tax_rate_est"),    "{:.1f}%")
    )

    # Valuation
    _section("💰 Valuation", COLORS["gold"],
        _cell("PE",            g("pe"),              "{:.1f}×") +
        _cell("PEG",           g("peg"),             "{:.2f}") +
        _cell("PEG Zone",      stock.get("peg_zone","") or "", "") +
        _cell("Earnings Yield",g("earnings_yield"),  "{:.1f}%") +
        _cell("PE vs 10Y Med", g("pe_discount"),     "{:.1f}%") +
        _cell("EV/EBITDA Dir", g("ev_ebitda_direction"), "{:.2f}") +
        _cell("Payback Ratio", g("payback_ratio"),   "{:.1f}y") +
        _cell("P/E vs ROE MoS",g("pe_vs_roe_mos"),  "{:.1f}") +
        _cell("Valuation Scr", g("valuation_score"), "{:.0f}/100") +
        _cell("Buy Zone",      stock.get("buy_zone_label","") or "", "")
    )

    # Ownership
    _section("👥 Ownership & Governance", COLORS["orange"],
        _cell("Promoter %",    g("promoter_holdings"),    "{:.1f}%") +
        _cell("Pledge %",      g("pledged_percentage"),   "{:.1f}%") +
        _cell("FII %",         g("fii_holdings"),         "{:.1f}%") +
        _cell("DII %",         g("dii_holdings"),         "{:.1f}%") +
        _cell("Promoter Chg", g("change_promoter_lq"),   "{:+.1f}%") +
        _cell("FII Chg",       g("change_fii_lq"),        "{:+.1f}%") +
        _cell("Smart Money",   stock.get("smart_money_flow","") or "","") +
        _cell("Gov Bonus",     g("governance_bonus"),     "{:.0f}") +
        _cell("Mgmt Integrity",g("management_integrity_score"),"{:.0f}/3") +
        _cell("Dilution Flag", "Yes ⚠️" if g("dilution_flag") == 1 else "Clean ✅","")
    )

    # Technical
    _section("⚡ Technical & Momentum", COLORS["cyan"],
        _cell("CRS 50D",       g("crs_50d"),         "{:.0f}") +
        _cell("CRS 26W",       g("crs_26w"),         "{:.0f}") +
        _cell("CRS 52W",       g("crs_52w"),         "{:.0f}") +
        _cell("RS Composite",  g("d47_rs_composite"),"{:.1f}") +
        _cell("RSI 14D",       g("rsi_14d"),         "{:.1f}") +
        _cell("Vol Ratio",     g("vol_ratio"),       "{:.2f}×") +
        _cell("Dist 52WH",     g("dist_52wh"),       "{:.1f}%") +
        _cell("VSTOP Green",   "Yes ✅" if g("vstop_green") == 1 else "No","") +
        _cell("Breakout Scr",  g("breakout_score"),  "{:.0f}") +
        _cell("Momentum Scr",  g("momentum_score"),  "{:.0f}/100")
    )

    # Forensic flags summary
    _section("🔬 Forensic Summary", COLORS["red"],
        _cell("Red Flags",     g("red_flag_count"),      "{:.0f}/25") +
        _cell("Forensic Scr",  g("forensic_score"),      "{:.0f}/100") +
        _cell("Forensic Mult", g("forensic_multiplier"), "{:.0%}") +
        _cell("Piotroski",     g("piotroski_fscore"),    "{:.0f}/9") +
        _cell("Econ Profit",   g("economic_profit"),     "₹{:,.0f} Cr") +
        _cell("EP Quintile",   stock.get("ep_quintile","") or "","") +
        _cell("QGLP Score",    g("qglp_score"),          "{:.0f}/100") +
        _cell("QGLP Pass",     "Yes ✅" if g("qglp_pass") == 1 else "No","") +
        _cell("Composite Scr", g("composite_score"),     "{:.0f}/100") +
        _cell("Conviction Tier",g("conviction_tier"),    "Tier {:.0f}")
    )

    # MOSL Wealth Creation signals (9 Annual Wealth Creation Studies extracted into the engine)
    _yn = lambda k: "Yes ✅" if g(k) == 1 else "No"
    _section("🏛️ MOSL Wealth Creation Signals", COLORS["gold"],
        # 13th — Great/Good/Gruesome taxonomy
        _cell("Corporate Class", stock.get("corporate_class","") or "N/A", "") +
        # 17th — Economic Moat persistence (sector-relative ROE across 5 timeframes)
        _cell("EMC Sector-Beat",  g("emc_sector_beat_count"), "{:.0f}/5") +
        _cell("EMC Flag",         _yn("emc_flag"), "") +
        # 22nd — CAP & GAP longevity (duration above cost of capital / 15% growth)
        _cell("CAP Years",        g("cap_years_proxy"), "{:.0f}/5") +
        _cell("GAP Years",        g("gap_years_proxy"), "{:.0f}/3") +
        _cell("CAP-GAP Score",    g("cap_gap_score"),   "{:.0f}/4") +
        # 27th — Consistents vs Volatiles
        _cell("Consistency Champ", _yn("consistency_champion"), "") +
        _cell("PAT Falls >10%",   g("pat_decline_count_5y"), "{:.0f}/5") +
        _cell("Volatile Flag",    _yn("mosl_volatile_flag"), "") +
        # 28th — EP Power Curve quintile (1=highest EP)
        _cell("EP Quintile",      stock.get("ep_quintile","") or "N/A", "") +
        _cell("EP Top Q1/Q2",     _yn("ep_top_quintile_flag"), "") +
        # 14th — Winner Category (sector tailwind) × Category Winner (leader)
        _cell("Winner Category",  _yn("winner_category_flag"), "") +
        _cell("Winning Invest.",  _yn("category_winner_in_winner_cat"), "") +
        # 17th — 100x candidate (Mouse-to-Elephant)
        _cell("100x Candidate",   _yn("mosl_100x_candidate"), "") +
        # 29th — Bruised Blue Chip (P/B < 2x quality fallen)
        _cell("Bruised Blue Chip", _yn("bruised_blue_chip_29"), "") +
        # 23rd — Growth-Value trap (growth + ROE < cost of equity)
        _cell("Growth-Value Trap", _yn("growth_value_trap"), "") +
        # 9th — Cyclical peak trap (commodity at peak-cycle deceptive low P/E)
        _cell("Cyclical Peak Trap", _yn("cyclical_peak_trap"), "") +
        # 26th — Atoms vs Bits business design
        _cell("Atoms/Bits",       stock.get("atoms_to_bits_label","") or "N/A", "")
    )

    # Dorsey Wide Moat pillar cells
    _section("🏰 Dorsey Wide Moat Pillars", COLORS["purple"],
        _cell("M — Moat Level",   "✅" if g("dorsey_moat_level")    == 1 else "❌", "") +
        _cell("D — Direction",    "✅" if g("dorsey_moat_direction") == 1 else "❌", "") +
        _cell("V — FCF Yield",    "✅" if g("dorsey_fcf_valuation")  == 1 else "❌", "") +
        _cell("Q — Cash Quality", "✅" if g("dorsey_cash_quality")   == 1 else "❌", "") +
        _cell("C — Cap Structure","✅" if g("dorsey_cap_structure")  == 1 else "❌", "") +
        _cell("Dorsey Score",     g("dorsey_score"),  "{:.0f}/5") +
        _cell("Dorsey Pass",      "Yes ✅" if g("dorsey_pass") == 1 else "No", "")
    )

    # Outsider CEO capital allocation pillar cells
    _section("🏆 Outsider CEO Pillars", "#f0a500",
        _cell("S — Share Retire", "✅" if g("outsider_share_retirement") == 1 else "❌", "") +
        _cell("D — Debt Discip",  "✅" if g("outsider_debt_discipline")  == 1 else "❌", "") +
        _cell("C — Cash Gen",     "✅" if g("outsider_cash_generation")  == 1 else "❌", "") +
        _cell("R — Capital Ret",  "✅" if g("outsider_capital_returns")  == 1 else "❌", "") +
        _cell("Outsider Score",   g("outsider_score"), "{:.0f}/4") +
        _cell("Outsider Pass",    "Yes ✅" if g("outsider_pass") == 1 else "No", "")
    )

    # Howard Marks Cycle & Risk Defensive Shield pillar cells
    _section("🛡️ Marks Cycle Shield Pillars", "#00CED1",
        _cell("M — Margin",       "✅" if g("marks_margin_spike")   == 1 else "❌", "") +
        _cell("P — Price/Value",  "✅" if g("marks_price_value")    == 1 else "❌", "") +
        _cell("L — Leverage",     "✅" if g("marks_leverage_trap")  == 1 else "❌", "") +
        _cell("D — Defensive",    "✅" if g("marks_defensive_base") == 1 else "❌", "") +
        _cell("Marks Score",      g("marks_score"), "{:.0f}/4") +
        _cell("Marks Pass",       "Yes ✅" if g("marks_pass") == 1 else "No", "")
    )

    _section("🚀 Lynch Fast Grower Pillars", "#e74c3c",
        _cell("V — Growth Vel",   "✅" if g("lynch_growth_velocity") == 1 else "❌", "") +
        _cell("P — PEG Spot",     "✅" if g("lynch_valuation_peg")   == 1 else "❌", "") +
        _cell("D — Pre-Discov",   "✅" if g("lynch_pre_discovery")   == 1 else "❌", "") +
        _cell("F — Fortress Own", "✅" if g("lynch_fortress_owner")  == 1 else "❌", "") +
        _cell("Lynch Score",      g("lynch_score"), "{:.0f}/4") +
        _cell("Lynch Pass",       "Yes ✅" if g("lynch_pass") == 1 else "No", "")
    )

    _section("🔮 Mauboussin Expectations Pillars", "#8b5cf6",
        _cell("T — Treadmill",    "✅" if g("mauboussin_treadmill_breach") == 1 else "❌", "") +
        _cell("O — Op Leverage",  "✅" if g("mauboussin_oplev_drift")      == 1 else "❌", "") +
        _cell("C — CAP Trap",     "✅ Clear" if g("mauboussin_cap_trap") == 0 else "❌ Trap", "") +
        _cell("Implied CAP",      g("mauboussin_implied_cap"), "{:.2f}") +
        _cell("Mauboussin Score", g("mauboussin_score"), "{:.0f}/3") +
        _cell("Mauboussin Pass",  "Yes ✅" if g("mauboussin_pass") == 1 else "No", "")
    )

    # CAN SLIM pillar cells
    _section("📊 CAN SLIM® Pillars", COLORS["blue"],
        _cell("C — Qtr EPS",      "✅" if g("can_slim_c") == 1 else "❌", "") +
        _cell("A — Annual Gr",    "✅" if g("can_slim_a") == 1 else "❌", "") +
        _cell("N — Near High",    "✅" if g("can_slim_n") == 1 else "❌", "") +
        _cell("S — Volume",       "✅" if g("can_slim_s") == 1 else "❌", "") +
        _cell("L — RS Leader",    "✅" if g("can_slim_l") == 1 else "❌", "") +
        _cell("I — Institution",  "✅" if g("can_slim_i") == 1 else "❌", "") +
        _cell("M — Market",       "✅" if g("can_slim_m") == 1 else "❌", "") +
        _cell("VCP Dryup",        "✅" if g("can_slim_vcp") == 1 else "—", "") +
        _cell("RS Uptrend",       "✅" if g("can_slim_rs_trend") == 1 else "—", "") +
        _cell("CAN SLIM Score",   g("can_slim_score"), "{:.0f}/17") +
        _cell("Market Regime",    stock.get("market_regime", "—") or "—", "")
    )


# ═══════════════════════════════════════════════════════════════
# CAN SLIM® TACTICAL MOMENTUM RADAR — O'Neil
# ═══════════════════════════════════════════════════════════════

def render_canslim_radar(stock: pd.Series):
    """
    Renders William O'Neil's 7-pillar CAN SLIM tactical momentum radar panel.
    PURE DISPLAY — Reads pre-materialized binary pillar columns from scoring_engine.py.
    Zero threshold re-computation; zero scoring logic; immune to parameter drift.
    """
    st.markdown("<div class='sec-head'>📊 CAN SLIM® Tactical Momentum Radar</div>",
                unsafe_allow_html=True)

    cs_pass  = int(_g(stock, "can_slim_pass", 0))
    cs_score = int(_g(stock, "can_slim_score", 0))
    regime   = str(stock.get("market_regime", "SIDEWAYS") or "SIDEWAYS").upper()

    pillars = [
        ("C", "Current Earnings", int(_g(stock, "can_slim_c", 0)) == 1,
         "Quarterly EPS & Sales Growth ≥ 25% YoY"),
        ("A", "Annual Growth",    int(_g(stock, "can_slim_a", 0)) == 1,
         "5Y EPS CAGR ≥ 25% · ROE ≥ 17% · 3Y Unbroken Step"),
        ("N", "New Breakout",     int(_g(stock, "can_slim_n", 0)) == 1,
         "Price Within 15% of 52-Week High"),
        ("S", "Supply & Demand",  int(_g(stock, "can_slim_s", 0)) == 1,
         "Breakout Session Volume Surge ≥ 1.5×"),
        ("L", "Leader / Laggard", int(_g(stock, "can_slim_l", 0)) == 1,
         "IBD RS Composite Percentile Rank ≥ 80"),
        ("I", "Institutional",    int(_g(stock, "can_slim_i", 0)) == 1,
         "Active Smart-Money Inflow (FII or DII +)"),
        ("M", "Market Direction", int(_g(stock, "can_slim_m", 0)) == 1,
         f"Regime: {_esc(regime)} — PAUSED if BEAR"),
    ]

    vcp_active = int(_g(stock, "can_slim_vcp",      0)) == 1
    rs_active  = int(_g(stock, "can_slim_rs_trend", 0)) == 1

    hdr_color  = COLORS["blue"] if cs_pass else COLORS["text_muted"]
    status_msg = "🟢 PASSED BREAKOUT GATE" if cs_pass else "⚪ Tactical Sieve Hold"

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d1117 0%,#161b22 100%);
                border:1px solid {COLORS['border']};border-top:3px solid {hdr_color};
                border-radius:12px;padding:14px 18px;margin-bottom:12px;">
      <div style="display:flex;align-items:center;justify-content:space-between;">
        <div>
          <div style="font-size:0.95rem;font-weight:800;color:#f0f6fc;">
            O'Neil Momentum Compliance Profile
          </div>
          <div style="font-size:0.72rem;color:#8b949e;margin-top:2px;">
            Status: <strong style="color:{hdr_color};">{status_msg}</strong>
          </div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:1.5rem;font-weight:900;color:{hdr_color};line-height:1.0;">
            {cs_score}
            <span style="font-size:0.85rem;color:#8b949e;font-weight:400;">&thinsp;/ 17</span>
          </div>
          <div style="font-size:0.6rem;color:#8b949e;text-transform:uppercase;
                      letter-spacing:0.5px;margin-top:2px;">Total Tactical Components</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    grid_html = ""
    for letter, title, passed, baseline in pillars:
        # Critical pillars (C, A, M) get red on failure; others get muted grey
        clr = COLORS["green"] if passed else ("#f85149" if letter in ("C", "A", "M") else "#8b949e")
        bg_opacity = "15" if passed else "08"
        ico = "✅" if passed else "❌"

        desc = baseline
        if letter == "S" and vcp_active:
            desc += " <span style='color:#bc8cff;'>(🔥 VCP Dryup)</span>"
        if letter == "L" and rs_active:
            desc += " <span style='color:#58a6ff;'>(🔥 RS Uptrend)</span>"

        grid_html += (
            f"<div style='background:{clr}{bg_opacity};border:1px solid {clr}40;"
            f"border-radius:8px;padding:10px;text-align:center;min-width:110px;flex:1;'>"
            f"<div style='font-size:1.6rem;font-weight:900;color:{clr};line-height:1.1;'>{letter}</div>"
            f"<div style='font-size:0.68rem;font-weight:700;color:#f0f6fc;margin-top:4px;"
            f"white-space:nowrap;'>{_esc(title)}</div>"
            f"<div style='font-size:0.58rem;color:#8b949e;margin-top:2px;line-height:1.2;'>{desc}</div>"
            f"<div style='font-size:1.0rem;margin-top:4px;'>{ico}</div>"
            f"</div>"
        )

    st.markdown(
        f"<div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px;'>{grid_html}</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
# SCHILIT FORENSIC SHIELD — Accounting Shenanigans Audit
# ═══════════════════════════════════════════════════════════════

def _get_schilit_context(stock: pd.Series, checker_col: str) -> str:
    """Returns a compact metric string to show beneath an active Schilit flag."""
    if checker_col == "schilit_ems_flag":
        aw  = int(_g(stock, "accruals_warning", 0))
        igp = _g(stock, "inv_vs_rev_gap", 0)
        return f"accruals_warning: {aw}  ·  inv_gap: {igp:.1f}%"
    if checker_col == "schilit_cfs_flag":
        pg  = _g(stock, "pat_gr_yoy",  0)
        ocf = _g(stock, "ocf_growth",  0)
        return f"pat_gr_yoy: {pg:.1f}%  ·  ocf_growth: {ocf:.1f}%"
    if checker_col == "schilit_kms_lev_flag":
        hcd = int(_g(stock, "high_cash_high_debt", 0))
        return f"high_cash_high_debt: {hcd}"
    if checker_col == "schilit_kms_bloat_flag":
        dso = _g(stock, "dso_delta_3y",           0)
        idc = _g(stock, "inventory_days_change",  0)
        return f"dso_delta_3y: {dso:.0f}d  ·  inventory_days_change: {idc:.0f}d"
    return ""


def render_schilit_shield(stock: pd.Series):
    """
    Renders Howard Schilit's Financial Shenanigans Accounting Audit Shield.
    PURE DISPLAY — Reads pre-materialized boolean flags from forensic_engine.py.
    All 6 column names verified against forensic_engine.py lines 820-826.
    """
    st.markdown("<div class='sec-head'>🛡️ Schilit Accounting Anomaly Shield</div>",
                unsafe_allow_html=True)

    f_score = _g(stock, "schilit_forensic_score", 100.0)
    f_pass  = int(_g(stock, "schilit_pass", 1))

    checkers = [
        ("EMS Anomaly Gimmick",
         int(_g(stock, "schilit_ems_flag",       0)) == 1,
         "schilit_ems_flag",
         "Revenue Recognition / Expense Capitalization Metrics"),
        ("CFS Cash Flow Trap",
         int(_g(stock, "schilit_cfs_flag",       0)) == 1,
         "schilit_cfs_flag",
         "Operating Cash Divergence / Paper Profit Shifts"),
        ("KMS Leverage Mirage",
         int(_g(stock, "schilit_kms_lev_flag",   0)) == 1,
         "schilit_kms_lev_flag",
         "Off-Balance Sheet Guarantees & Pledged Cash Mismatches"),
        ("KMS Operational Bloat",
         int(_g(stock, "schilit_kms_bloat_flag", 0)) == 1,
         "schilit_kms_bloat_flag",
         "Channel Stuffing / Asset Aging Accumulation"),
    ]

    shield_clr  = COLORS["green"] if f_pass else COLORS["red"]
    status_txt  = ("🛡️ PERIMETER SECURE — CLEAN AUDIT"
                   if f_pass else
                   "🚨 COGNITIVE RISK CAPTURED — SHENANIGAN ALERT")

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d1117 0%,#161b22 100%);
                border:2px solid {shield_clr}40;border-left:6px solid {shield_clr};
                border-radius:12px;padding:14px 20px;margin-bottom:14px;
                box-shadow:0 4px 20px rgba(0,0,0,0.2);">
      <div style="display:flex;align-items:center;justify-content:space-between;gap:15px;">
        <div>
          <div style="font-size:0.65rem;color:#8b949e;letter-spacing:1.5px;
                      text-transform:uppercase;">Accounting Security Shield</div>
          <div style="font-size:1.15rem;font-weight:900;color:{shield_clr};margin-top:2px;">
            {status_txt}
          </div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:1.6rem;font-weight:900;color:{shield_clr};line-height:1.0;">
            {f_score:.0f}
            <span style="font-size:0.85rem;color:#8b949e;font-weight:400;">&thinsp;/ 100</span>
          </div>
          <div style="font-size:0.6rem;color:#8b949e;text-transform:uppercase;
                      letter-spacing:0.5px;margin-top:2px;">Forensic Credibility</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    cells_html = ""
    for title, triggered, col_name, narrative in checkers:
        clr = COLORS["red"] if triggered else COLORS["green"]
        ico = "⚠️ Triggered" if triggered else "🎯 Clear"
        bg  = f"{clr}0d"
        bdr = f"{clr}30"

        ctx_val = _get_schilit_context(stock, col_name) if triggered else ""
        sub_desc = (
            f"<div style='font-size:0.58rem;color:{clr};font-weight:700;margin-top:2px;'>"
            f"{_esc(ctx_val)}</div>"
            if ctx_val else
            f"<div style='font-size:0.58rem;color:#8b949e;margin-top:2px;'>"
            f"{_esc(narrative)}</div>"
        )

        cells_html += (
            f"<div style='background:{bg};border:1px solid {bdr};border-radius:8px;"
            f"padding:10px;flex:1;min-width:220px;'>"
            f"<div style='display:flex;align-items:center;justify-content:space-between;'>"
            f"<span style='font-size:0.78rem;font-weight:800;color:#f0f6fc;'>"
            f"{_esc(title)}</span>"
            f"<span style='font-size:0.68rem;font-weight:700;color:{clr};"
            f"background:{clr}15;padding:2px 8px;border-radius:12px;'>{ico}</span>"
            f"</div>"
            f"{sub_desc}"
            f"</div>"
        )

    st.markdown(
        f"<div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px;'>{cells_html}</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
# PAT DORSEY WIDE MOAT RADAR — 5-Pillar Economic Moat Audit
# ═══════════════════════════════════════════════════════════════

def render_dorsey_radar(stock: pd.Series):
    """
    Renders Pat Dorsey's Wide Moat 5-pillar economic moat audit card.
    PURE DISPLAY — Reads pre-materialized binary pillar columns from scoring_engine.py.
    Zero threshold re-computation; zero scoring logic; immune to parameter drift.
    Source: docs/dorsey_moat_specs.json
    """
    st.markdown("<div class='sec-head'>🏰 Dorsey Wide Moat Radar</div>",
                unsafe_allow_html=True)

    d_pass  = int(_g(stock, "dorsey_pass",  0))
    d_score = int(_g(stock, "dorsey_score", 0))

    pillars = [
        ("M", "Moat Return Level",
         int(_g(stock, "dorsey_moat_level",    0)) == 1,
         "Wide Moat Return Hurdle: 10Y & 5Y ROCE ≥ 20%"),
        ("D", "Moat Trajectory",
         int(_g(stock, "dorsey_moat_direction", 0)) == 1,
         "Advantage Direction: Stable or Widening Trajectory"),
        ("V", "FCF Valuation Yield",
         int(_g(stock, "dorsey_fcf_valuation",  0)) == 1,
         "Margin of Safety Floor: Free Cash Flow Yield ≥ 5%"),
        ("Q", "Cash Realization Quality",
         int(_g(stock, "dorsey_cash_quality",   0)) == 1,
         "Earnings Conversion Base: CFO/PAT Conversion ≥ 80%"),
        ("C", "Capital Structure Cushion",
         int(_g(stock, "dorsey_cap_structure",  0)) == 1,
         "Leverage Cushion Guard: D/E < 1.0 / Financial Exempt"),
    ]

    hdr_color  = COLORS["purple"] if d_pass else COLORS["text_muted"]
    status_msg = "🏰 CONFIRMED WIDE MOAT" if d_pass else "⚪ Moat Unconfirmed"

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d1117 0%,#161b22 100%);
                border:1px solid {COLORS['border']};border-top:3px solid {hdr_color};
                border-radius:12px;padding:14px 18px;margin-bottom:12px;">
      <div style="display:flex;align-items:center;justify-content:space-between;">
        <div>
          <div style="font-size:0.95rem;font-weight:800;color:#f0f6fc;">
            Pat Dorsey Economic Moat Compliance Profile
          </div>
          <div style="font-size:0.72rem;color:#8b949e;margin-top:2px;">
            Status: <strong style="color:{hdr_color};">{status_msg}</strong>
          </div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:1.5rem;font-weight:900;color:{hdr_color};line-height:1.0;">
            {d_score}
            <span style="font-size:0.85rem;color:#8b949e;font-weight:400;">&thinsp;/ 5</span>
          </div>
          <div style="font-size:0.6rem;color:#8b949e;text-transform:uppercase;
                      letter-spacing:0.5px;margin-top:2px;">Moat Gates Cleared</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    grid_html = ""
    for letter, title, passed, baseline in pillars:
        # All 5 pillars are equal weight — red on failure (each is a hard gate)
        clr = COLORS["purple"] if passed else "#f85149"
        bg_opacity = "15" if passed else "08"
        ico = "✅" if passed else "❌"

        grid_html += (
            f"<div style='background:{clr}{bg_opacity};border:1px solid {clr}40;"
            f"border-radius:8px;padding:10px;text-align:center;min-width:110px;flex:1;'>"
            f"<div style='font-size:1.6rem;font-weight:900;color:{clr};line-height:1.1;'>{letter}</div>"
            f"<div style='font-size:0.68rem;font-weight:700;color:#f0f6fc;margin-top:4px;"
            f"white-space:nowrap;'>{_esc(title)}</div>"
            f"<div style='font-size:0.58rem;color:#8b949e;margin-top:2px;line-height:1.2;'>"
            f"{_esc(baseline)}</div>"
            f"<div style='font-size:1.0rem;margin-top:4px;'>{ico}</div>"
            f"</div>"
        )

    st.markdown(
        f"<div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px;'>{grid_html}</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
# WILLIAM THORNDIKE OUTSIDER CEO RADAR — 4-Pillar Capital Allocation Audit
# ═══════════════════════════════════════════════════════════════

def render_outsider_radar(stock: pd.Series):
    """
    Renders William Thorndike Outsider CEO 4-pillar capital allocation audit card.
    PURE DISPLAY — Reads pre-materialized binary pillar columns from scoring_engine.py.
    Zero threshold re-computation; zero scoring logic; immune to parameter drift.
    Source: docs/outsider_specs.json
    """
    st.markdown("<div class='sec-head'>🏆 Thorndike Outsider CEO Radar</div>",
                unsafe_allow_html=True)

    o_pass  = int(_g(stock, "outsider_pass",  0))
    o_score = int(_g(stock, "outsider_score", 0))

    pillars = [
        ("S", "Share Retirement",
         int(_g(stock, "outsider_share_retirement", 0)) == 1,
         "Anti-Dilution Shield: share count stable or active repurchases"),
        ("D", "Debt Discipline",
         int(_g(stock, "outsider_debt_discipline",  0)) == 1,
         "Deleveraging Trend: 3Y D/E slope declining or stable"),
        ("C", "Cash Generation",
         int(_g(stock, "outsider_cash_generation",  0)) == 1,
         "Realization Floor: CFO/PAT conversion ≥ 85% (strictest base)"),
        ("R", "Capital Efficiency",
         int(_g(stock, "outsider_capital_returns",  0)) == 1,
         "Full-Cycle Returns: 10-Year ROCE Median ≥ 15%"),
    ]

    _OUTSIDER_GOLD = "#f0a500"
    hdr_color  = _OUTSIDER_GOLD if o_pass else COLORS["text_muted"]
    status_msg = "🏆 CONFIRMED OUTSIDER CEO" if o_pass else "⚪ Outsider Standard Not Met"

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d1117 0%,#1a1400 100%);
                border:1px solid {COLORS['border']};border-top:3px solid {hdr_color};
                border-radius:12px;padding:14px 18px;margin-bottom:12px;">
      <div style="display:flex;align-items:center;justify-content:space-between;">
        <div>
          <div style="font-size:0.95rem;font-weight:800;color:#f0f6fc;">
            Thorndike Capital Allocation Compliance Profile
          </div>
          <div style="font-size:0.72rem;color:#8b949e;margin-top:2px;">
            Status: <strong style="color:{hdr_color};">{status_msg}</strong>
          </div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:1.5rem;font-weight:900;color:{hdr_color};line-height:1.0;">
            {o_score}
            <span style="font-size:0.85rem;color:#8b949e;font-weight:400;">&thinsp;/ 4</span>
          </div>
          <div style="font-size:0.6rem;color:#8b949e;text-transform:uppercase;
                      letter-spacing:0.5px;margin-top:2px;">Capital Gates Cleared</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    grid_html = ""
    for letter, title, passed, baseline in pillars:
        clr        = _OUTSIDER_GOLD if passed else "#f85149"
        bg_opacity = "18" if passed else "08"
        ico        = "✅" if passed else "❌"

        grid_html += (
            f"<div style='background:{clr}{bg_opacity};border:1px solid {clr}40;"
            f"border-radius:8px;padding:10px;text-align:center;min-width:110px;flex:1;'>"
            f"<div style='font-size:1.6rem;font-weight:900;color:{clr};line-height:1.1;'>{letter}</div>"
            f"<div style='font-size:0.68rem;font-weight:700;color:#f0f6fc;margin-top:4px;"
            f"white-space:nowrap;'>{_esc(title)}</div>"
            f"<div style='font-size:0.58rem;color:#8b949e;margin-top:2px;line-height:1.2;'>"
            f"{_esc(baseline)}</div>"
            f"<div style='font-size:1.0rem;margin-top:4px;'>{ico}</div>"
            f"</div>"
        )

    st.markdown(
        f"<div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px;'>{grid_html}</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
# MARKS MARKET CYCLE & RISK DEFENSIVE SHIELD — Howard Marks
# ═══════════════════════════════════════════════════════════════

def render_marks_radar(stock: pd.Series):
    """
    Renders Howard Marks Market Cycle & Risk Defensive 4-pillar shield card.
    PURE DISPLAY — Reads pre-materialized binary pillar columns from scoring_engine.py.
    Zero threshold re-computation; zero scoring logic; immune to parameter drift.
    Source: docs/marks_cycle_specs.json v1.1-marks-cycle-codex-india-calibrated
    Thresholds (v1.1 India-calibrated): see docs/marks_cycle_specs.json for exact values.
    Companion Ch.9: D/E and CFO/PAT gates tightened for India defensive floor.
    """
    st.markdown("<div class='sec-head'>🛡️ Howard Marks Cycle & Risk Defensive Radar</div>",
                unsafe_allow_html=True)

    m_pass  = int(_g(stock, "marks_pass",  0))
    m_score = int(_g(stock, "marks_score", 0))

    pillars = [
        ("M", "Margin Extreme",
         int(_g(stock, "marks_margin_spike",   0)) == 1,
         "Pendulum Spike Guard: margins sit within sustainable historical limits"),
        ("P", "Price vs Value",
         int(_g(stock, "marks_price_value",    0)) == 1,
         "Asymmetry Margin: asset trades within a disciplined entry buy zone"),
        ("L", "Leverage Cushion",
         int(_g(stock, "marks_leverage_trap",  0)) == 1,
         "Risk Avoidance Line: balance sheet debt stays safely below caps"),
        ("D", "Defensive Cushion",
         int(_g(stock, "marks_defensive_base", 0)) == 1,
         "Margin for Error Base: CFO/PAT cash generation clears 70% floor"),
    ]

    _MARKS_CYAN = "#00CED1"
    hdr_color  = _MARKS_CYAN if m_pass else COLORS["text_muted"]
    status_msg = "🛡️ MARKS CYCLE SHIELD CONFIRMED" if m_pass else "⚪ Cycle Shield Not Cleared"

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d1117 0%,#001a1a 100%);
                border:1px solid {COLORS['border']};border-top:3px solid {hdr_color};
                border-radius:12px;padding:14px 18px;margin-bottom:12px;">
      <div style="display:flex;align-items:center;justify-content:space-between;">
        <div>
          <div style="font-size:0.95rem;font-weight:800;color:#f0f6fc;">
            Howard Marks Cycle Defence Compliance Profile
          </div>
          <div style="font-size:0.72rem;color:#8b949e;margin-top:2px;">
            Status: <strong style="color:{hdr_color};">{status_msg}</strong>
          </div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:1.5rem;font-weight:900;color:{hdr_color};line-height:1.0;">
            {m_score}
            <span style="font-size:0.85rem;color:#8b949e;font-weight:400;">&thinsp;/ 4</span>
          </div>
          <div style="font-size:0.6rem;color:#8b949e;text-transform:uppercase;
                      letter-spacing:0.5px;margin-top:2px;">Cycle Gates Cleared</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    grid_html = ""
    for letter, title, passed, baseline in pillars:
        clr        = _MARKS_CYAN if passed else "#f85149"
        bg_opacity = "18" if passed else "08"
        ico        = "✅" if passed else "❌"

        grid_html += (
            f"<div style='background:{clr}{bg_opacity};border:1px solid {clr}40;"
            f"border-radius:8px;padding:10px;text-align:center;min-width:110px;flex:1;'>"
            f"<div style='font-size:1.6rem;font-weight:900;color:{clr};line-height:1.1;'>{letter}</div>"
            f"<div style='font-size:0.68rem;font-weight:700;color:#f0f6fc;margin-top:4px;"
            f"white-space:nowrap;'>{_esc(title)}</div>"
            f"<div style='font-size:0.58rem;color:#8b949e;margin-top:2px;line-height:1.2;'>"
            f"{_esc(baseline)}</div>"
            f"<div style='font-size:1.0rem;margin-top:4px;'>{ico}</div>"
            f"</div>"
        )

    st.markdown(
        f"<div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px;'>{grid_html}</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
# DR. VIJAY MALIK PEACEFUL INVESTING RADAR — 5-Pillar Financial Quality Audit
# ═══════════════════════════════════════════════════════════════

def render_malik_radar(stock: pd.Series):
    """
    Renders Dr. Vijay Malik's Peaceful Investing 5-pillar financial quality audit card.
    PURE DISPLAY — Reads pre-materialized binary pillar columns from scoring_engine.py.
    Zero threshold re-computation; zero scoring logic; immune to parameter drift.
    Source: docs/malik_peaceful_specs.json v1.0-malik-peaceful-codex (all thresholds there).
    Pillars (G/P/F/C/S):
      G — Growth Runway:    malik_growth_runway    (Sales CAGR gate; see pillar_g_growth_runway)
      P — Margin Stability: malik_profit_stability (NPM stable gate; see pillar_p_profit_stability)
      F — Debt Fortress:    malik_debt_fortress    (ICR + D/E + CR; fin exempt; see pillar_f_debt_fortress)
      C — Cash Realization: malik_cash_generation  (CFO/PAT PERCENTAGE gate; see pillar_c_cash_generation)
      S — Self-Funded:      malik_self_funded      (SSGR binary flag; see pillar_s_self_funded)
    """
    st.markdown("<div class='sec-head'>🕊️ Dr. Vijay Malik — Peaceful Investing Radar</div>",
                unsafe_allow_html=True)

    m_pass  = int(_g(stock, "malik_pass",  0))
    m_score = int(_g(stock, "malik_score", 0))

    pillars = [
        ("G", "Growth Runway",
         int(_g(stock, "malik_growth_runway",    0)) == 1,
         "Sales Growth Hurdle: 10Y/5Y revenue CAGR clears the self-funding floor"),
        ("P", "Margin Stability",
         int(_g(stock, "malik_profit_stability", 0)) == 1,
         "Pricing Power Shield: current NPM stable and prior year not deteriorating"),
        ("F", "Debt Fortress",
         int(_g(stock, "malik_debt_fortress",    0)) == 1,
         "Leverage Cushion Gate: ICR, D/E, and Current Ratio all within safe range"),
        ("C", "Cash Realization",
         int(_g(stock, "malik_cash_generation",  0)) == 1,
         "Audited Reality Floor: operating cash flow substantively backs reported profit"),
        ("S", "Self-Funded Growth",
         int(_g(stock, "malik_self_funded",      0)) == 1,
         "Sustainable Growth Core: SSGR covers actual sales growth without new debt"),
    ]

    _MALIK_GREEN = "#2ecc71"
    hdr_color  = _MALIK_GREEN if m_pass else COLORS["text_muted"]
    status_msg = "PEACEFUL INVESTING CERTIFIED" if m_pass else "Peaceful Investing Criteria Not Met"

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d1117 0%,#0b1a0f 100%);
                border:1px solid {COLORS['border']};border-top:3px solid {hdr_color};
                border-radius:12px;padding:14px 18px;margin-bottom:12px;">
      <div style="display:flex;align-items:center;justify-content:space-between;">
        <div>
          <div style="font-size:0.95rem;font-weight:800;color:#f0f6fc;">
            Vijay Malik Financial Quality Compliance Profile
          </div>
          <div style="font-size:0.72rem;color:#8b949e;margin-top:2px;">
            Status: <strong style="color:{hdr_color};">{_esc(status_msg)}</strong>
          </div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:1.5rem;font-weight:900;color:{hdr_color};line-height:1.0;">
            {m_score}
            <span style="font-size:0.85rem;color:#8b949e;font-weight:400;">&thinsp;/ 5</span>
          </div>
          <div style="font-size:0.6rem;color:#8b949e;text-transform:uppercase;
                      letter-spacing:0.5px;margin-top:2px;">Peaceful Parameters Cleared</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    _mk_grid = ""
    for letter, title, passed, baseline in pillars:
        clr_mk     = _MALIK_GREEN if passed else "#f85149"
        bg_mk      = "18" if passed else "08"
        ico_mk     = "✅" if passed else "❌"

        _mk_grid += (
            f"<div style='background:{clr_mk}{bg_mk};border:1px solid {clr_mk}40;"
            f"border-radius:8px;padding:10px;text-align:center;min-width:110px;flex:1;'>"
            f"<div style='font-size:1.6rem;font-weight:900;color:{clr_mk};line-height:1.1;'>"
            f"{_esc(letter)}</div>"
            f"<div style='font-size:0.68rem;font-weight:700;color:#f0f6fc;margin-top:4px;"
            f"white-space:nowrap;'>{_esc(title)}</div>"
            f"<div style='font-size:0.58rem;color:#8b949e;margin-top:2px;line-height:1.2;'>"
            f"{_esc(baseline)}</div>"
            f"<div style='font-size:1.0rem;margin-top:4px;'>{ico_mk}</div>"
            f"</div>"
        )

    st.markdown(
        f"<div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px;'>{_mk_grid}</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PETER LYNCH FAST GROWER RADAR — 4-Pillar Tenbagger Discovery Audit
# ══════════════════════════════════════════════════════════════════════════════

def render_lynch_radar(stock: pd.Series):
    """
    Renders Peter Lynch's Fast Grower 4-pillar tenbagger discovery audit card.
    PURE DISPLAY — Reads pre-materialized binary pillar columns from scoring_engine.py.
    Zero threshold re-computation; zero scoring logic; immune to parameter drift.
    Source: docs/lynch_growth_specs.json v1.1-india-calibrated-fastgrower (all thresholds there).
    Pillars (V/P/D/F):
      V — Growth Velocity: lynch_growth_velocity (Rev 5Y CAGR + EPS per share + FCF cash gate)
      P — Valuation PEG:   lynch_valuation_peg   (positive PEG in sweet spot corridor)
      D — Pre-Discovery:   lynch_pre_discovery   (FII+DII combined below institutional threshold)
      F — Fortress Owner:  lynch_fortress_owner  (D/E balance sheet + promoter level OR active buying)
    """
    st.markdown("<div class='sec-head'>🚀 Peter Lynch — Fast Grower Tenbagger Radar</div>",
                unsafe_allow_html=True)

    l_pass  = int(_g(stock, "lynch_pass",  0))
    l_score = int(_g(stock, "lynch_score", 0))

    pillars = [
        ("V", "Growth Velocity",
         int(_g(stock, "lynch_growth_velocity", 0)) == 1,
         "Hyper-Growth Runway: revenue speed confirmed by EPS per share acceleration and positive free cash flow"),
        ("P", "PEG Sweet Spot",
         int(_g(stock, "lynch_valuation_peg",   0)) == 1,
         "GARP Entry Corridor: growth significantly outpaces the price paid for it"),
        ("D", "Pre-Discovery",
         int(_g(stock, "lynch_pre_discovery",   0)) == 1,
         "Combined Institutional Weight: FII plus DII below the pre-discovery threshold"),
        ("F", "Fortress Owner",
         int(_g(stock, "lynch_fortress_owner",  0)) == 1,
         "Skin In The Game Shield: conservative balance sheet and owner conviction by level or active buying"),
    ]

    _LYNCH_RED  = "#e74c3c"
    hdr_color   = _LYNCH_RED if l_pass else COLORS["text_muted"]
    status_msg  = "LYNCH TENBAGGER CERTIFIED" if l_pass else "Lynch Fast Grower Criteria Not Met"

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d1117 0%,#1a0a0a 100%);
                border:1px solid {COLORS['border']};border-top:3px solid {hdr_color};
                border-radius:12px;padding:14px 18px;margin-bottom:12px;">
      <div style="display:flex;align-items:center;justify-content:space-between;">
        <div>
          <div style="font-size:0.95rem;font-weight:800;color:#f0f6fc;">
            Peter Lynch Fast Grower Tenbagger Discovery Profile
          </div>
          <div style="font-size:0.72rem;color:#8b949e;margin-top:2px;">
            Status: <strong style="color:{hdr_color};">{_esc(status_msg)}</strong>
          </div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:1.5rem;font-weight:900;color:{hdr_color};line-height:1.0;">
            {l_score}
            <span style="font-size:0.85rem;color:#8b949e;font-weight:400;">&thinsp;/ 4</span>
          </div>
          <div style="font-size:0.6rem;color:#8b949e;text-transform:uppercase;
                      letter-spacing:0.5px;margin-top:2px;">Lynch Gates Cleared</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    _ly_grid = ""
    for letter, title, passed, baseline in pillars:
        clr_ly  = _LYNCH_RED if passed else COLORS["text_muted"]
        bg_ly   = "18" if passed else "08"
        ico_ly  = "✅" if passed else "❌"

        _ly_grid += (
            f"<div style='background:{clr_ly}{bg_ly};border:1px solid {clr_ly}40;"
            f"border-radius:8px;padding:10px;text-align:center;min-width:110px;flex:1;'>"
            f"<div style='font-size:1.6rem;font-weight:900;color:{clr_ly};line-height:1.1;'>"
            f"{_esc(letter)}</div>"
            f"<div style='font-size:0.68rem;font-weight:700;color:#f0f6fc;margin-top:4px;"
            f"white-space:nowrap;'>{_esc(title)}</div>"
            f"<div style='font-size:0.58rem;color:#8b949e;margin-top:2px;line-height:1.2;'>"
            f"{_esc(baseline)}</div>"
            f"<div style='font-size:1.0rem;margin-top:4px;'>{ico_ly}</div>"
            f"</div>"
        )

    st.markdown(
        f"<div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px;'>{_ly_grid}</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
# MAUBOUSSIN EXPECTATIONS INVESTING RADAR — Framework 34
# ═══════════════════════════════════════════════════════════════

def render_mauboussin_radar(stock: pd.Series):
    """
    Renders Mauboussin & Rappaport's Expectations Investing 3-layer audit card.
    PURE DISPLAY — Reads pre-materialized binary pillar columns from scoring_engine.py.
    Zero threshold re-computation; zero scoring logic; immune to parameter drift.
    Source: docs/mauboussin_expectations_specs.json v1.1-fixed-nopat-precision.
    Pillars (T/O/C):
      T — Treadmill Safety:      mauboussin_treadmill_breach (sell_alert_treadmill gate)
      O — OpLev Integrity:       mauboussin_oplev_drift      (operating_leverage gate)
      C — CAP Trap Clear:        mauboussin_cap_trap==0      (implied_cap > 15 + ROCE 3Y slope < -1)
    Layer 3: Interactive Reverse DCF Expected Value Calculator (on-demand, single-stock only).
    """
    _MAUB_COLOR = "#8b5cf6"

    st.markdown("<div class='sec-head'>🔮 Mauboussin — Expectations Investing Radar</div>",
                unsafe_allow_html=True)

    m_pass   = int(_g(stock, "mauboussin_pass",         0))
    m_score  = int(_g(stock, "mauboussin_score",        0))
    m_cap    = _g(stock, "mauboussin_implied_cap",      0.0)
    _nopat_raw = _g(stock, "mauboussin_nopat_margin",   None)
    m_nopat_str = f"{_nopat_raw:.1f}%" if _nopat_raw is not None and _nopat_raw == _nopat_raw else "—"

    pillars = [
        ("T", "Treadmill Safety",
         int(_g(stock, "mauboussin_treadmill_breach", 0)) == 1,
         "Expectations Treadmill Safe: stock not priced for indefinite perfection requiring continuous positive surprises"),
        ("O", "OpLev Integrity",
         int(_g(stock, "mauboussin_oplev_drift",      0)) == 1,
         "Operating Leverage Intact: incremental revenue converting efficiently to profit — economic engine healthy"),
        ("C", "CAP Trap Clear",
         int(_g(stock, "mauboussin_cap_trap",         0)) == 0,
         "Competitive Advantage Period Realistic: no high-CAP expectations paired with ROCE deceleration"),
    ]

    hdr_color  = _MAUB_COLOR if m_pass else COLORS["text_muted"]
    status_msg = "EXPECTATIONS MATRIX CERTIFIED" if m_pass else "Expectations Investing Criteria Not Met"

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d1117 0%,#120a1a 100%);
                border:1px solid {COLORS['border']};border-top:3px solid {hdr_color};
                border-radius:12px;padding:14px 18px;margin-bottom:12px;">
      <div style="display:flex;align-items:center;justify-content:space-between;">
        <div>
          <div style="font-size:0.95rem;font-weight:800;color:#f0f6fc;">
            Mauboussin Price-Implied Expectations (PIE) Audit
          </div>
          <div style="font-size:0.72rem;color:#8b949e;margin-top:2px;">
            Implied CAP Proxy: <strong style="color:{hdr_color};">{m_cap:.2f}</strong>
            &nbsp;·&nbsp;
            NOPAT Margin: <strong style="color:{hdr_color};">{m_nopat_str}</strong>
            &nbsp;·&nbsp;
            Status: <strong style="color:{hdr_color};">{_esc(status_msg)}</strong>
          </div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:1.5rem;font-weight:900;color:{hdr_color};line-height:1.0;">
            {m_score}
            <span style="font-size:0.85rem;color:#8b949e;font-weight:400;">&thinsp;/ 3</span>
          </div>
          <div style="font-size:0.6rem;color:#8b949e;text-transform:uppercase;
                      letter-spacing:0.5px;margin-top:2px;">Expectations Gates Cleared</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    _mb_grid = ""
    for letter, title, passed, baseline in pillars:
        clr_mb = _MAUB_COLOR if passed else COLORS["text_muted"]
        bg_mb  = "18" if passed else "08"
        ico_mb = "✅" if passed else "❌"

        _mb_grid += (
            f"<div style='background:{clr_mb}{bg_mb};border:1px solid {clr_mb}40;"
            f"border-radius:8px;padding:10px;text-align:center;min-width:110px;flex:1;'>"
            f"<div style='font-size:1.6rem;font-weight:900;color:{clr_mb};line-height:1.1;'>"
            f"{_esc(letter)}</div>"
            f"<div style='font-size:0.68rem;font-weight:700;color:#f0f6fc;margin-top:4px;"
            f"white-space:nowrap;'>{_esc(title)}</div>"
            f"<div style='font-size:0.58rem;color:#8b949e;margin-top:2px;line-height:1.2;'>"
            f"{_esc(baseline)}</div>"
            f"<div style='font-size:1.0rem;margin-top:4px;'>{ico_mb}</div>"
            f"</div>"
        )

    st.markdown(
        f"<div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px;'>{_mb_grid}</div>",
        unsafe_allow_html=True,
    )

    # ── Layer 3: Interactive Reverse DCF Expected Value Calculator ───────────
    st.markdown(
        f"<div style='font-size:0.7rem;font-weight:800;color:{_MAUB_COLOR};"
        f"text-transform:uppercase;letter-spacing:1px;margin:10px 0 8px 0;'>"
        f"🧮 Reverse DCF — Expected Value Calculator</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-size:0.65rem;color:#8b949e;margin-bottom:8px;'>"
        "Mauboussin Expected Value = P(Upside) × Upside% + P(Downside) × Downside%"
        "</div>",
        unsafe_allow_html=True,
    )

    _c1, _c2, _c3, _c4 = st.columns(4)
    with _c1:
        p_up  = st.number_input("P(Upside) %",   min_value=0.0, max_value=100.0,
                                value=60.0, step=5.0, key="maub_p_up")
    with _c2:
        up_pct = st.number_input("Upside %",      min_value=0.0, max_value=500.0,
                                 value=40.0, step=5.0, key="maub_up_pct")
    with _c3:
        p_dn  = st.number_input("P(Downside) %", min_value=0.0, max_value=100.0,
                                value=40.0, step=5.0, key="maub_p_dn")
    with _c4:
        dn_pct = st.number_input("Downside %",    min_value=0.0, max_value=100.0,
                                 value=20.0, step=5.0, key="maub_dn_pct")

    ev = (p_up / 100.0 * up_pct) - (p_dn / 100.0 * dn_pct)
    ev_color = _MAUB_COLOR if ev > 0 else "#e74c3c"
    ev_label = "POSITIVE EV — Asymmetric Opportunity" if ev > 0 else "NEGATIVE EV — Unfavorable Odds"

    st.markdown(
        f"<div style='background:#0d1117;border:1px solid {ev_color}40;"
        f"border-radius:8px;padding:10px 14px;margin-top:8px;'>"
        f"<span style='font-size:0.7rem;color:#8b949e;'>Expected Value: </span>"
        f"<strong style='font-size:1.1rem;color:{ev_color};'>{ev:+.1f}%</strong>"
        f"<span style='font-size:0.65rem;color:{ev_color};margin-left:8px;'>{_esc(ev_label)}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
# MOSL WEALTH CREATION MATRIX — single-card summary of the 30-study signals
# ═══════════════════════════════════════════════════════════════

def render_mosl_wealth_matrix(stock: pd.Series):
    """
    Renders the MOSL Wealth Creation Matrix — a compact, space-optimised card summarising
    the highest-conviction valuation signals from the 30 Annual Wealth Creation Studies.
    PURE DISPLAY — reads pre-materialized columns only; zero math, zero sorting, zero apply().
    Mounted beneath the Mauboussin radar. Defensive: all values via _g()/_esc(), float-guarded.
    Shows: 5-Yr Payback · P/E-vs-RoE margin of safety · Absolute Economic Profit · Atoms/Bits design.
    """
    def _num(key, default=0.0):
        """Float-safe numeric extraction — guards None / NaN / inf into a clean float."""
        v = stock.get(key, default)
        try:
            f = float(v)
            if f != f or f in (float("inf"), float("-inf")):   # NaN or inf
                return default
            return f
        except (TypeError, ValueError):
            return default

    # ── Pull pre-computed metrics defensively ───────────────────────────────
    payback   = _num("payback_trailing_5y", default=float("nan"))
    payback_g = _num("payback_ratio", default=float("nan"))       # growth-adjusted fallback
    pay_show  = payback if payback == payback else payback_g       # prefer trailing; else growth
    pe_to_roe = _num("pe_to_roe_ratio", default=float("nan"))     # PE / sustainable ROE; <1 = MoS
    pe_below  = int(_num("pe_below_roe", 0))
    ep_abs    = _num("economic_profit", default=float("nan"))     # ₹ Cr/yr (Net Worth × (RoE−CoE))
    ep_pos    = int(_num("economic_profit_positive", 0))
    atb       = str(_g(stock, "atoms_to_bits_label", "Hybrid") or "Hybrid")

    _GOLD, _GREEN, _RED, _BLUE, _MUTE = (
        COLORS["gold"], COLORS["green"], COLORS["red"], COLORS["blue"], COLORS["text_muted"]
    )

    # ── Tile 1: 5-Year Payback ──────────────────────────────────────────────
    if pay_show == pay_show and pay_show > 0:
        pay_clr   = _GREEN if pay_show < 1.0 else (_GOLD if pay_show < 2.0 else _MUTE)
        pay_val   = f"{pay_show:.2f}x"
        pay_badge = "5-YR PAYBACK CLEAR" if pay_show < 1.0 else (
            "Attractive (<2x)" if pay_show < 2.0 else "Full valuation")
    else:
        pay_clr, pay_val, pay_badge = _MUTE, "—", "No earnings basis"

    # ── Tile 2: P/E vs Sustainable RoE (Motilal's original margin of safety) ─
    if pe_to_roe == pe_to_roe and pe_to_roe > 0:
        roe_disc  = (1.0 - pe_to_roe) * 100.0   # positive = PE below ROE = margin of safety
        mos_clr   = _GREEN if pe_below == 1 else _RED
        mos_val   = f"{roe_disc:+.0f}%"
        mos_badge = "P/E BELOW SUSTAINABLE RoE" if pe_below == 1 else "Premium to RoE"
    else:
        mos_clr, mos_val, mos_badge = _MUTE, "—", "RoE basis unavailable"

    # ── Tile 3: Absolute Economic Profit (28th WCS, exact book-value math) ───
    if ep_abs == ep_abs:
        ep_clr   = _GREEN if ep_pos == 1 else _RED
        ep_val   = f"₹{ep_abs:,.0f} Cr"
        ep_badge = "VALUE CREATOR (RoE > CoE)" if ep_pos == 1 else "Value Destroyer (RoE < CoE)"
    else:
        ep_clr, ep_val, ep_badge = _MUTE, "—", "Net worth unavailable"

    # ── Tile 4: Business Design (26th WCS Atoms→Bits) ───────────────────────
    _atb_map = {
        "Bits":   (_BLUE,  "💡", "Asset-light · network-effect scale"),
        "Atoms":  (_GOLD,  "🏭", "Capital-intensive · linear scale"),
        "Hybrid": (_MUTE,  "⚙️", "Mixed physical + digital model"),
    }
    atb_clr, atb_icon, atb_desc = _atb_map.get(atb, (_MUTE, "⚙️", "Mixed model"))

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown(
        "<div class='sec-head'>🏛️ MOSL Wealth Creation Matrix — 30-Study Signal Summary</div>",
        unsafe_allow_html=True,
    )

    def _tile(color, value, label, sub):
        return (
            f"<div style='flex:1;flex-shrink:0;min-width:150px;white-space:nowrap;"
            f"background:{color}12;border:1px solid {color}40;border-radius:10px;padding:12px 14px;'>"
            f"<div style='font-size:0.6rem;font-weight:700;color:{_MUTE};text-transform:uppercase;"
            f"letter-spacing:0.6px;white-space:nowrap;'>{_esc(label)}</div>"
            f"<div style='font-size:1.35rem;font-weight:900;color:{color};line-height:1.2;"
            f"margin-top:3px;white-space:nowrap;'>{_esc(value)}</div>"
            f"<div style='font-size:0.62rem;color:{color};margin-top:2px;white-space:nowrap;'>"
            f"{_esc(sub)}</div>"
            f"</div>"
        )

    tiles = (
        _tile(pay_clr, pay_val, "⏳ 5-Yr Payback", pay_badge) +
        _tile(mos_clr, mos_val, "📉 P/E vs RoE", mos_badge) +
        _tile(ep_clr, ep_val, "📊 Economic Profit / yr", ep_badge) +
        _tile(atb_clr, f"{atb_icon} {atb}", "🌐 Business Design", atb_desc)
    )

    st.markdown(
        f"<div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px;'>{tiles}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='font-size:0.6rem;color:{_MUTE};margin-bottom:14px;'>"
        f"Payback = MktCap ÷ trailing-5Y PAT · P/E-vs-RoE = Motilal 1st-Study margin of safety · "
        f"Economic Profit = Net Worth × (RoE − {12.0:.0f}% cost of equity), exact book value</div>",
        unsafe_allow_html=True,
    )
