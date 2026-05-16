import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
import os
from config import COLORS

# ═══════════════════════════════════════════════════════════════
# DATA VISUALIZATION: MOAT-GROWTH MATRIX
# ═══════════════════════════════════════════════════════════════

def render_moat_growth_matrix(df: pd.DataFrame, highlight_stock: str = None):
    """
    Renders an elite 2D Scatter plot mapping ROCE (Moat) vs Profit Growth (Growth).
    Highlights the 4 quadrants from the 22nd Wealth Creation Study.
    """
    st.markdown("<div class='sec-head'>🧭 Moat-Growth Matrix (22nd WCS)</div>", unsafe_allow_html=True)
    
    # We need valid data for both axes
    plot_df = df.copy()
    plot_df["Moat_Y"] = plot_df["roce_med_5y"].fillna(plot_df["roce"]).fillna(0)
    plot_df["Growth_X"] = plot_df["pat_gr_5y"].fillna(plot_df["pat_gr_3y"]).fillna(0)
    
    # Filter out extreme outliers for better visualization
    plot_df = plot_df[(plot_df["Moat_Y"] > -20) & (plot_df["Moat_Y"] < 100)]
    plot_df = plot_df[(plot_df["Growth_X"] > -40) & (plot_df["Growth_X"] < 150)]
    
    if len(plot_df) == 0:
        st.warning("Not enough valid data to plot the matrix.")
        return
        
    fig = px.scatter(
        plot_df, x="Growth_X", y="Moat_Y", 
        color="moat_growth_quad",
        color_discrete_map={
            "⭐ Wealth Creator": COLORS["green"],
            "🛡️ Quality Trap": COLORS["gold"],
            "⚡ Growth Trap": COLORS["blue"],
            "💀 Wealth Destroyer": COLORS["red"]
        },
        hover_name="name",
        hover_data={"Growth_X": ':.1f', "Moat_Y": ':.1f', "moat_growth_quad": False},
        labels={"Growth_X": "Growth (PAT CAGR %)", "Moat_Y": "Moat (ROCE %)"}
    )
    
    # Add Quadrant Lines (15% Growth, 15% ROCE)
    fig.add_vline(x=15, line_width=1, line_dash="dash", line_color=COLORS["border"])
    fig.add_hline(y=15, line_width=1, line_dash="dash", line_color=COLORS["border"])
    
    # Quadrant Annotations
    fig.add_annotation(x=80, y=80, text="⭐ Wealth Creators", showarrow=False, font=dict(color=COLORS["green"], size=16), opacity=0.3)
    fig.add_annotation(x=-20, y=80, text="🛡️ Quality Traps", showarrow=False, font=dict(color=COLORS["gold"], size=16), opacity=0.3)
    fig.add_annotation(x=80, y=-10, text="⚡ Growth Traps", showarrow=False, font=dict(color=COLORS["blue"], size=16), opacity=0.3)
    fig.add_annotation(x=-20, y=-10, text="💀 Destroyers", showarrow=False, font=dict(color=COLORS["red"], size=16), opacity=0.3)
    
    # Highlight specific stock if requested
    if highlight_stock:
        highlight_data = plot_df[plot_df["name"] == highlight_stock]
        if not highlight_data.empty:
            fig.add_trace(go.Scatter(
                x=highlight_data["Growth_X"],
                y=highlight_data["Moat_Y"],
                mode='markers+text',
                marker=dict(color='white', size=15, line=dict(color='black', width=2)),
                text=["🎯 " + highlight_stock],
                textposition="top center",
                name="Selected Stock",
                showlegend=False
            ))

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS["text_primary"]),
        margin=dict(l=0, r=0, t=30, b=0),
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=COLORS["border"], zeroline=True, zerolinewidth=2, zerolinecolor=COLORS["border"])
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=COLORS["border"], zeroline=True, zerolinewidth=2, zerolinecolor=COLORS["border"])
    
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# SYSTEMATIC FISHER PROXY (100% AUTOMATED FROM CSV)
# ═══════════════════════════════════════════════════════════════

def render_fisher_module(stock: pd.Series):
    """
    Translates Philip Fisher's qualitative principles into strict quantitative 
    proxies using ONLY the derived CSV data. Zero manual input required.
    """
    st.markdown(f"""
    <div style="background:{COLORS['bg_secondary']}; border-left:4px solid {COLORS['gold']}; padding:10px 15px; margin-bottom:15px; border-radius:4px;">
        <h3 style="margin:0; font-size:1.1rem; color:{COLORS['gold']};">🧠 Systematic Fisher Proxy</h3>
        <p style="margin:4px 0 0 0; font-size:0.8rem; color:{COLORS['text_muted']};">
            100% Automated. Translating Fisher's 15 Qualitative points into quantitative CSV realities.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate Automated Proxies
    proxies = []
    
    # 1. Growth Potential (Fisher P1) -> Rev Growth
    rev_gr = stock.get("rev_gr_5y", 0)
    p1_pass = rev_gr >= 15
    proxies.append(("P1: Market Potential (Sales Growth >15%)", p1_pass, f"{rev_gr:.1f}%"))
    
    # 2. Operating Leverage / Sales Org (Fisher P4) -> Profit growing faster than sales
    p4_pass = stock.get("operating_leverage", 0) == 1
    proxies.append(("P4: Sales Org Efficiency (Profit Gr > Sales Gr)", p4_pass, "Passed" if p4_pass else "Failed"))
    
    # 3. Worthwhile Profit Margin (Fisher P5) -> NPM
    npm = stock.get("npm", 0)
    p5_pass = npm >= 10
    proxies.append(("P5: Worthwhile Margins (NPM >10%)", p5_pass, f"{npm:.1f}%"))
    
    # 4. Maintaining Margins (Fisher P6) -> NPM vs 1YB
    npm_1yb = stock.get("npm_1yb", 0)
    p6_pass = npm >= npm_1yb and npm > 0
    proxies.append(("P6: Margin Trajectory (NPM ≥ Last Year)", p6_pass, "Improving" if p6_pass else "Declining"))
    
    # 5. Cost Controls (Fisher P10) -> Expense Ratio decreasing
    exp_ratio = stock.get("expense_ratio", 1)
    exp_ratio_1yb = stock.get("expense_ratio_1yb", 1)
    p10_pass = exp_ratio <= exp_ratio_1yb
    proxies.append(("P10: Cost Controls (Expense Ratio Falling)", p10_pass, "Tighter" if p10_pass else "Looser"))
    
    # 6. No Equity Dilution (Fisher P13)
    p13_pass = stock.get("dilution_flag", 1) == 0
    proxies.append(("P13: No Equity Dilution (Share Count Stable)", p13_pass, "Clean" if p13_pass else "Diluted"))
    
    # 7. Management Integrity (Fisher P15) -> Forensic Red Flags
    p15_pass = stock.get("forensic_label", "") == "🟢 Clean"
    proxies.append(("P15: Accounting Integrity (Zero Forensic Flags)", p15_pass, stock.get("forensic_label", "")))

    # Calculate Total Score
    passed = sum(1 for p in proxies if p[1])
    total = len(proxies)
    
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("**Automated Proxy Checks**")
        for desc, is_pass, val in proxies:
            icon = "✅" if is_pass else "❌"
            st.markdown(f"{icon} **{desc}**: `{val}`")
            
    with c2:
        score_pct = (passed / total) * 100
        gauge_color = COLORS['green'] if score_pct >= 80 else COLORS['gold'] if score_pct >= 50 else COLORS['red']
        
        st.markdown(f"""
        <div style="background:{COLORS['bg_tertiary']}; border:1px solid {COLORS['border']}; border-radius:12px; padding:20px; text-align:center;">
            <div style="font-size:0.85rem; color:{COLORS['text_muted']}; text-transform:uppercase;">Fisher Quant Score</div>
            <div style="font-size:3rem; font-weight:900; color:{gauge_color}; margin:10px 0;">{passed}/{total}</div>
            <div style="font-size:0.9rem; color:{COLORS['text_primary']};">
                {"🟢 High Alignment" if score_pct >= 80 else "🟡 Moderate" if score_pct >= 50 else "🔴 Low Alignment"}
            </div>
        </div>
        """, unsafe_allow_html=True)
