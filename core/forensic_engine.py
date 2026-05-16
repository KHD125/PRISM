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
    # BUG FIX: cfo_to_pat in CSV is a PERCENTAGE (e.g. 73.04), not a ratio (0.73)
    # FORENSIC["cfo_pat_alert"] = 0.7 was comparing percentage to ratio → always False
    # Fix: multiply threshold by 100 to match the CSV unit
    cfo_pat_threshold_pct = FORENSIC["cfo_pat_alert"] * 100  # 0.7 → 70.0
    df["rf_low_cfo_pat"] = np.where(
        df["cfo_to_pat"].notna(),
        (df["cfo_to_pat"] < cfo_pat_threshold_pct).astype(int),
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
    # BUG FIX: Any tiny rise (e.g. 0.01 → 0.02) was flagging clean companies.
    # Fix: Only flag if D/E rose by >10% relative (material rise) AND is above 0.3
    # This prevents penalizing essentially debt-free companies for rounding noise.
    de_rose_materially = (
        (df["debt_to_equity"] > df["debt_to_equity_1yb"] * 1.10) &  # >10% relative rise
        (df["debt_to_equity"] > 0.30)                                 # AND D/E is meaningful
    )
    df["rf_rising_debt"] = np.where(
        df["debt_to_equity"].notna() & df["debt_to_equity_1yb"].notna(),
        de_rose_materially.astype(int),
        0
    )

    # 5. Cash conversion cycle increasing
    df["rf_ccc_worsening"] = np.where(
        df["ccc"].notna() & df["ccc_1yb"].notna(),
        (df["ccc"] > df["ccc_1yb"] + 10).astype(int),  # worsened by 10+ days
        0
    )

    # 6. Expense ratio rising (operational deterioration)
    # BUG FIX: Any marginal rise (even 0.001) was flagging healthy companies.
    # Fix: Only flag if expense ratio rose by more than 3 percentage points.
    # Normal quarterly/annual noise is within 1-2pp. 3pp+ signals real deterioration.
    df["rf_expense_rising"] = np.where(
        df["expense_ratio"].notna() & df["expense_ratio_1yb"].notna(),
        (df["expense_ratio"] > df["expense_ratio_1yb"] + 0.03).astype(int),  # 3pp threshold
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
    # BUG FIX: Growth companies investing in capex naturally have negative FCF.
    # A company with CWIP converting to fixed assets + positive OCF is NOT in danger.
    # Fix: Only flag negative FCF when OCF itself is also negative (true cash burn).
    # If OCF > 0 but FCF < 0, it means they are investing capex — a HEALTHY signal.
    df["rf_negative_fcf"] = np.where(
        df["free_cash_flow"].notna() & df["operating_cash_flow"].notna(),
        ((df["free_cash_flow"] < 0) & (df["operating_cash_flow"] < 0)).astype(int),
        np.where(
            df["free_cash_flow"].notna(),
            (df["free_cash_flow"] < 0).astype(int),  # if OCF not available, use FCF alone
            0
        )
    )

    # 10. Revenue growing but PAT declining (margin compression)
    df["rf_margin_squeeze"] = np.where(
        df["rev_gr_yoy"].notna() & df["pat_gr_yoy"].notna(),
        ((df["rev_gr_yoy"] > 5) & (df["pat_gr_yoy"] < 0)).astype(int),
        0
    )

    # 11. High Cash + High Debt simultaneously (Malik Shenanigan 4)
    df["rf_high_cash_debt"] = df.get("high_cash_high_debt", pd.Series(0, index=df.index)).fillna(0).astype(int)

    # 12. Declining Inventory Turnover (Malik Shenanigan 3)
    df["rf_itr_declining"] = np.where(
        df["inventory_turnover"].notna() & df["inventory_turnover_1yb"].notna(),
        (df["inventory_turnover"] < df["inventory_turnover_1yb"] * 0.9).astype(int),  # 10%+ decline
        0
    )

    # 13. SSGR < actual growth (debt-dependent growth — Malik Ch.2)
    df["rf_ssgr_deficit"] = np.where(
        df.get("ssgr_cushion", pd.Series(np.nan, index=df.index)).notna(),
        (df.get("ssgr_cushion", pd.Series(0, index=df.index)) < -10).astype(int),  # SSGR trails by 10%+
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
        "rf_high_cash_debt": "High cash + high debt (Malik S4)",
        "rf_itr_declining": "Inventory turnover declining (Malik S3)",
        "rf_ssgr_deficit": "Growth exceeds SSGR (debt-dependent)",
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
