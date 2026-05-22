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
from config import COLORS


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
    "rf_receivables_bloat": ("DSO expansion >20 days above sector median — relative receivables manipulation", "🟡"),
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
    x_max = min(float(plot_df["Growth_X"].max()) * 1.05, 300)

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
# FORENSIC FRAUD PERIMETER — WCS 24 / 23-Flag Cascade
# ═══════════════════════════════════════════════════════════════

def render_forensic_perimeter(stock: pd.Series):
    """
    Vectorized Fraud Perimeter Display.
    Outputs structured, named red-flag badges (not just a count) for every fired
    forensic signal. Connects directly to the cascading forensic filter multiplier.
    """
    st.markdown("<div class='sec-head'>🔬 Forensic Fraud Perimeter (23-Flag Cascade)</div>",
                unsafe_allow_html=True)

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

    # ── Header KPI strip ─────────────────────────────────────────────────
    st.markdown(f"""
    <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px;">
        <div style="background:{COLORS['bg_secondary']};border:1px solid {COLORS['border']};
                    border-radius:10px;padding:10px 16px;min-width:120px;text-align:center;">
            <div style="font-size:1.5rem;font-weight:900;color:{flag_color};">{flag_count}</div>
            <div style="font-size:0.63rem;color:{COLORS['text_muted']};text-transform:uppercase;margin-top:2px;">Red Flags</div>
        </div>
        <div style="background:{COLORS['bg_secondary']};border:1px solid {COLORS['border']};
                    border-radius:10px;padding:10px 16px;min-width:120px;text-align:center;">
            <div style="font-size:1.5rem;font-weight:900;color:{COLORS['green']};">{forensic_score:.0f}</div>
            <div style="font-size:0.63rem;color:{COLORS['text_muted']};text-transform:uppercase;margin-top:2px;">Forensic Score</div>
        </div>
        <div style="background:{COLORS['bg_secondary']};border:1px solid {COLORS['border']};
                    border-radius:10px;padding:10px 16px;min-width:120px;text-align:center;">
            <div style="font-size:1.5rem;font-weight:900;color:{mult_color};">{f_mult:.0%}</div>
            <div style="font-size:0.63rem;color:{COLORS['text_muted']};text-transform:uppercase;margin-top:2px;">Score Multiplier</div>
        </div>
        <div style="background:{COLORS['bg_secondary']};border:1px solid {COLORS['border']};
                    border-radius:10px;padding:10px 16px;min-width:120px;text-align:center;">
            <div style="font-size:1.5rem;font-weight:900;color:{COLORS['blue']};">{piotroski}/9</div>
            <div style="font-size:0.63rem;color:{COLORS['text_muted']};text-transform:uppercase;margin-top:2px;">Piotroski</div>
        </div>
        <div style="background:{COLORS['bg_secondary']};border:1px solid {COLORS['border']};
                    border-radius:10px;padding:10px 16px;min-width:120px;text-align:center;">
            <div style="font-size:1.5rem;font-weight:900;color:{COLORS['purple']};">{mgmt_int}/3</div>
            <div style="font-size:0.63rem;color:{COLORS['text_muted']};text-transform:uppercase;margin-top:2px;">Mgmt Integrity</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.caption(f"**Status:** {_esc(forensic_label)} · **Piotroski:** {_esc(pio_label)}")

    if flag_count == 0:
        st.success(
            "✅ **Clean Bill of Health** — No forensic red flags across all 23 accounting "
            "checks (17 Schilit/Malik shenanigans + 6 WCS 24 defensive protocols)."
        )
        return

    if f_mult < 1.0:
        st.warning(
            f"⚠️ **Cascading Forensic Filter active:** composite score × {f_mult:.0%} "
            f"({flag_count} flags fired). Engine multiplier preserves rank ordering while "
            "proportionally penalising serial accounting risk."
        )

    st.markdown(f"**Active Red Flags — {flag_count} triggered:**")

    for rf_col, (desc, sev) in _FLAG_DISPLAY.items():
        if int(_g(stock, rf_col, 0)) != 1:
            continue
        bg  = ("rgba(248,81,73,0.09)"   if sev == "🔴" else
               "rgba(255,107,53,0.09)"  if sev == "🟠" else
               "rgba(228,179,65,0.09)")
        brd = ("rgba(248,81,73,0.40)"   if sev == "🔴" else
               "rgba(255,107,53,0.40)"  if sev == "🟠" else
               "rgba(228,179,65,0.40)")
        st.markdown(
            f'<div style="background:{bg};border-left:3px solid {brd};'
            f'border-radius:4px;padding:6px 14px;margin:3px 0;font-size:0.8rem;">'
            f'{sev} <strong>{_esc(desc)}</strong></div>',
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
    st.markdown("<div class='sec-head'>🏛️ Guru Framework Alignment</div>",
                unsafe_allow_html=True)

    fw_str = stock.get("frameworks_passed", "None") or "None"
    fw_list = [f.strip() for f in fw_str.split(",") if f.strip() and f.strip() != "None"]

    _FW_META = {
        "Coffee Can":          (COLORS["gold"],   "☕", "ROCE ≥15% for 10Y + Rev CAGR ≥10% — Mukherjea"),
        "QGLP":                (COLORS["purple"], "🏆", "Quality + Growth + Longevity + Price — Raamdeo"),
        "Lynch PEG Dream":     (COLORS["green"],  "📈", "PEG ≤1.0 + Rev outpacing costs — Peter Lynch"),
        "EP Hockey Stick":     (COLORS["green"],  "🚀", "Q2/Q3 ascending EP curve — 28th WCS"),
        "Bruised Blue Chip 29":(COLORS["blue"],   "💙", "Elite ROCE + large-cap at P/B ≤2× — 29th WCS"),
        "Multi-Trillion Cap":  (COLORS["purple"], "🌐", "Sunrise sector at compounding velocity — 30th WCS"),
    }

    if not fw_list:
        st.info("No institutional Guru frameworks fully met in current market configuration.")
        return

    for fw in fw_list:
        color, icon, desc = _FW_META.get(fw, (COLORS["text_muted"], "✅", fw))
        st.markdown(
            f'<div style="background:{color}14;border:1px solid {color}44;'
            f'border-radius:8px;padding:8px 14px;margin:4px 0;display:flex;'
            f'align-items:center;gap:10px;">'
            f'<span style="font-size:1.1rem;">{icon}</span>'
            f'<div><span style="font-weight:700;color:{color};">{_esc(fw)}</span>'
            f'<span style="font-size:0.75rem;color:{COLORS["text_muted"]};margin-left:8px;">'
            f'{_esc(desc)}</span></div></div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════
# SYSTEMATIC FISHER PROXY — 100% Automated from CSV
# ═══════════════════════════════════════════════════════════════

def render_fisher_module(stock: pd.Series):
    """
    Translates Philip Fisher's 15 qualitative principles into strict quantitative
    proxies using ONLY pre-derived CSV columns. Zero manual input; zero re-computation.
    Columns read directly from the pre-computed stock row (data_engine outputs).
    """
    st.markdown(f"""
    <div style="background:{COLORS['bg_secondary']};border-left:4px solid {COLORS['gold']};
                padding:10px 15px;margin-bottom:15px;border-radius:4px;">
        <h3 style="margin:0;font-size:1.1rem;color:{COLORS['gold']};">🧠 Systematic Fisher Proxy</h3>
        <p style="margin:4px 0 0 0;font-size:0.8rem;color:{COLORS['text_muted']};">
            100% Automated. Fisher&#x27;s 15 qualitative points translated into vectorized CSV signals.
        </p>
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
    p15_pass = forensic_label in ["🟢 Clean", "🟡 Watch"]
    proxies.append(("P15: Accounting Integrity (Clean/Watch)",
                    p15_pass, _esc(forensic_label)))

    passed = sum(1 for _, is_pass, _ in proxies if is_pass)
    total  = len(proxies)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("**Automated Proxy Checks**")
        for desc, is_pass, val in proxies:
            icon = "✅" if is_pass else "❌"
            st.markdown(f"{icon} **{_esc(desc)}**: `{_esc(val)}`")

    with c2:
        score_pct  = (passed / total) * 100
        gauge_color = (COLORS["green"] if score_pct >= 80 else
                       COLORS["gold"]  if score_pct >= 50 else
                       COLORS["red"])
        st.markdown(f"""
        <div style="background:{COLORS['bg_tertiary']};border:1px solid {COLORS['border']};
                    border-radius:12px;padding:20px;text-align:center;">
            <div style="font-size:0.85rem;color:{COLORS['text_muted']};text-transform:uppercase;">
                Fisher Quant Score
            </div>
            <div style="font-size:3rem;font-weight:900;color:{gauge_color};margin:10px 0;">
                {passed}/{total}
            </div>
            <div style="font-size:0.9rem;color:{COLORS['text_primary']};">
                {"🟢 High Alignment" if score_pct >= 80 else
                 "🟡 Moderate" if score_pct >= 50 else
                 "🔴 Low Alignment"}
            </div>
        </div>
        """, unsafe_allow_html=True)


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
        ctx = (
            f'<span style="color:{c_mut};font-size:0.69rem;margin-left:4px;">{_esc(context)}</span>'
        ) if context else ""
        return (
            f'<div style="display:flex;align-items:baseline;gap:6px;padding:5px 0;'
            f'border-bottom:1px solid rgba(255,255,255,0.04);">'
            f'<span style="font-size:0.85rem;width:18px;flex-shrink:0;">{ico}</span>'
            f'<span style="font-size:0.76rem;color:{c_sec};flex:0 0 170px;min-width:0;">'
            f'{_esc(label)}</span>'
            f'<span style="font-size:0.80rem;font-weight:700;color:{clr};flex:1;min-width:0;">'
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
    st.markdown("<div class='sec-head'>📊 Business & Financial Analysis</div>",
                unsafe_allow_html=True)
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
