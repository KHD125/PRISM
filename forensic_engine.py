"""
Multibagger Discovery System — Forensic Engine
================================================
Financial Shenanigans detection + Piotroski F-Score.
Runs on stocks that pass hard gates to catch hidden risks
the binary gates don't surface.

Based on: Financial Shenanigans India Forensic Edition
         + Schilit's 17 Shenanigans adapted for Indian markets
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from config import FORENSIC, PIOTROSKI


# ═══════════════════════════════════════════════════════════════
# PIOTROSKI F-SCORE (0–9)
# ═══════════════════════════════════════════════════════════════

def compute_piotroski_fscore(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Piotroski F-Score (0-9) for every stock. Pure vectorized."""
    df = df.copy()

    # 1. ROA positive (net income / total assets > 0)
    df["f_roa_positive"] = (df["roa"] > 0).astype(int)

    # 2. Operating cash flow positive
    df["f_ocf_positive"] = (df["operating_cash_flow"] > 0).astype(int)

    # 3. ROA improving (current ROE > last year ROE as proxy)
    df["f_roa_improving"] = np.where(
        df["roe"].notna() & df["roe_1yb"].notna(),
        (df["roe"] > df["roe_1yb"]).astype(int),
        0
    )

    # 4. Accrual quality: OCF > PAT (cash confirms earnings)
    df["f_accrual_quality"] = np.where(
        df["operating_cash_flow"].notna() & df["pat"].notna(),
        (df["operating_cash_flow"] > df["pat"]).astype(int),
        0
    )

    # 5. Leverage declining: D/E decreasing
    df["f_leverage_declining"] = np.where(
        df["debt_to_equity"].notna() & df["debt_to_equity_1yb"].notna(),
        (df["debt_to_equity"] < df["debt_to_equity_1yb"]).astype(int),
        0
    )

    # 6. Liquidity improving: current ratio increasing
    df["f_liquidity_improving"] = np.where(
        df["current_ratio"].notna() & df["current_ratio_1yb"].notna(),
        (df["current_ratio"] > df["current_ratio_1yb"]).astype(int),
        0
    )

    # 7. No dilution: shares not increased
    df["f_no_dilution"] = np.where(
        df["equity_shares"].notna() & df["equity_shares_1yb"].notna(),
        (df["equity_shares"] <= df["equity_shares_1yb"]).astype(int),
        1  # benefit of doubt if data missing
    )

    # 8. Gross margin improving (OPM latest quarter > 1 year back as proxy)
    df["f_margin_improving"] = np.where(
        df["opm_latest_q"].notna() & df["opm_1yb"].notna(),
        (df["opm_latest_q"] > df["opm_1yb"]).astype(int),
        0
    )

    # 9. Asset turnover improving (if available)
    # We don't have asset_turnover_1yb directly, so we check ROCE direction as proxy
    df["f_efficiency_improving"] = np.where(
        df["roce"].notna() & df["roce_1yb"].notna(),
        (df["roce"] > df["roce_1yb"]).astype(int),
        0
    )

    # Sum all 9 components
    f_cols = [c for c in df.columns if c.startswith("f_")]
    df["piotroski_fscore"] = df[f_cols].sum(axis=1)

    # F-Score classification
    df["piotroski_label"] = np.select(
        [
            df["piotroski_fscore"] >= PIOTROSKI["strong"],
            df["piotroski_fscore"] >= PIOTROSKI["moderate"],
        ],
        ["🟢 Strong", "🟡 Moderate"],
        default="🔴 Weak"
    )

    print(f"\n🔬 Piotroski F-Score Distribution:")
    print(f"   Strong (≥7): {(df['piotroski_fscore'] >= 7).sum()}")
    print(f"   Moderate (5-6): {((df['piotroski_fscore'] >= 5) & (df['piotroski_fscore'] < 7)).sum()}")
    print(f"   Weak (≤4): {(df['piotroski_fscore'] <= 4).sum()}")

    return df


# ═══════════════════════════════════════════════════════════════
# RED FLAG TRIAGE — 10 Forensic Checks
# ═══════════════════════════════════════════════════════════════

def compute_red_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Run 10 forensic red flag checks on every stock. Vectorized."""
    df = df.copy()

    # 1. CFO/PAT below threshold
    df["rf_low_cfo_pat"] = np.where(
        df["cfo_to_pat"].notna(),
        (df["cfo_to_pat"] < FORENSIC["cfo_pat_alert"]).astype(int),
        0
    )

    # 2. Receivables quality: Days Receivable very high (> 90 days)
    df["rf_high_receivables"] = np.where(
        df["days_receivable"].notna(),
        (df["days_receivable"] > 90).astype(int),
        0
    )

    # 3. Inventory growing faster than revenue
    df["rf_inventory_bloat"] = np.where(
        df["inv_vs_rev_gap"].notna(),
        (df["inv_vs_rev_gap"] > 10).astype(int),  # 10pp gap
        0
    )

    # 4. D/E direction: rising debt
    df["rf_rising_debt"] = np.where(
        df["debt_to_equity"].notna() & df["debt_to_equity_1yb"].notna(),
        (df["debt_to_equity"] > df["debt_to_equity_1yb"]).astype(int),
        0
    )

    # 5. Cash conversion cycle increasing
    df["rf_ccc_worsening"] = np.where(
        df["ccc"].notna() & df["ccc_1yb"].notna(),
        (df["ccc"] > df["ccc_1yb"] + 10).astype(int),  # worsened by 10+ days
        0
    )

    # 6. Expense ratio rising (operational deterioration)
    df["rf_expense_rising"] = np.where(
        df["expense_ratio"].notna() & df["expense_ratio_1yb"].notna(),
        (df["expense_ratio"] > df["expense_ratio_1yb"]).astype(int),
        0
    )

    # 7. Pledge level elevated
    df["rf_pledge_elevated"] = np.where(
        df["pledged_percentage"].notna(),
        (df["pledged_percentage"] > FORENSIC["pledge_watch"]).astype(int),
        0
    )

    # 8. Share dilution
    df["rf_dilution"] = df.get("dilution_flag", 0).fillna(0).astype(int)

    # 9. Negative free cash flow (cash burn)
    df["rf_negative_fcf"] = np.where(
        df["free_cash_flow"].notna(),
        (df["free_cash_flow"] < 0).astype(int),
        0
    )

    # 10. Revenue growing but PAT declining (margin compression)
    df["rf_margin_squeeze"] = np.where(
        df["rev_gr_yoy"].notna() & df["pat_gr_yoy"].notna(),
        ((df["rev_gr_yoy"] > 5) & (df["pat_gr_yoy"] < 0)).astype(int),
        0
    )

    # Sum all red flags
    rf_cols = [c for c in df.columns if c.startswith("rf_")]
    df["red_flag_count"] = df[rf_cols].sum(axis=1)

    # Forensic score: 100 = clean, 0 = maximum flags
    max_flags = len(rf_cols)
    df["forensic_score"] = ((max_flags - df["red_flag_count"]) / max_flags * 100).clip(0, 100)

    # Risk classification
    df["forensic_label"] = np.select(
        [
            df["red_flag_count"] == 0,
            df["red_flag_count"] <= 2,
            df["red_flag_count"] <= 4,
        ],
        ["🟢 Clean", "🟡 Watch", "🟠 Caution"],
        default="🔴 High Risk"
    )

    # Human-readable flag list
    flag_descriptions = {
        "rf_low_cfo_pat": "Low CFO/PAT (<70%)",
        "rf_high_receivables": "High receivables (>90 days)",
        "rf_inventory_bloat": "Inventory growing faster than revenue",
        "rf_rising_debt": "Debt-to-equity rising",
        "rf_ccc_worsening": "Cash conversion cycle worsening",
        "rf_expense_rising": "Expense ratio rising",
        "rf_pledge_elevated": "Pledge > 10%",
        "rf_dilution": "Share dilution detected",
        "rf_negative_fcf": "Negative free cash flow",
        "rf_margin_squeeze": "Revenue up but profit down",
    }

    def _build_flag_list(row):
        flags = []
        for col, desc in flag_descriptions.items():
            if row.get(col, 0) == 1:
                flags.append(desc)
        return " | ".join(flags) if flags else "Clean ✅"

    df["red_flag_list"] = df.apply(_build_flag_list, axis=1)

    print(f"\n🚨 Red Flag Distribution:")
    print(f"   Clean (0 flags): {(df['red_flag_count'] == 0).sum()}")
    print(f"   Watch (1-2): {((df['red_flag_count'] >= 1) & (df['red_flag_count'] <= 2)).sum()}")
    print(f"   Caution (3-4): {((df['red_flag_count'] >= 3) & (df['red_flag_count'] <= 4)).sum()}")
    print(f"   High Risk (5+): {(df['red_flag_count'] >= 5).sum()}")

    return df


# ═══════════════════════════════════════════════════════════════
# CASHFLOW QUALITY TRIANGLE
# ═══════════════════════════════════════════════════════════════

def compute_cashflow_triangle(df: pd.DataFrame) -> pd.DataFrame:
    """Classify each stock's cashflow pattern into the Quality Triangle."""
    df = df.copy()

    ocf_pos = df["operating_cash_flow"] > 0
    icf_neg = df["investing_cash_flow"] < 0
    fcf_neg = df["financing_cash_flow"] < 0
    fcf_pos = df["financing_cash_flow"] > 0

    df["cf_triangle"] = np.select(
        [
            ocf_pos & icf_neg & fcf_neg,    # Perfect: self-funding + investing + deleveraging
            ocf_pos & icf_neg & fcf_pos,    # Growth: OCF positive but borrowing to grow
            ~ocf_pos & icf_neg & fcf_pos,   # Danger: cash burn + still spending + borrowing
        ],
        ["✅ Perfect — Buy Zone", "⚠️ Growth Phase — Watch D/E", "🚨 Debt Trap — Avoid"],
        default="⚪ Mixed Pattern"
    )

    print(f"\n💰 Cashflow Triangle Distribution:")
    print(df["cf_triangle"].value_counts().to_string())

    return df


# ═══════════════════════════════════════════════════════════════
# MASTER FORENSIC PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_forensic_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Execute the complete forensic analysis pipeline."""
    print("\n" + "="*60)
    print("🔬 FORENSIC ENGINE — Risk Intelligence")
    print("="*60)

    df = compute_piotroski_fscore(df)
    df = compute_red_flags(df)
    df = compute_cashflow_triangle(df)

    return df
