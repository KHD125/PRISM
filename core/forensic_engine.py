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

# Industries with long billing cycles — DSO > 90 days is normal, not a red flag
_SERVICE_INDUSTRIES = frozenset({
    "Information Technology", "IT - Software", "Computers - Software",
    "IT Services & Consulting", "BPO/KPO",
    "Pharmaceuticals", "Pharmaceuticals - Indian - Bulk Drugs",
    "Healthcare", "Biotechnology",
    "Financial Services", "Banking", "NBFC", "Insurance",
})


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

    # 2. Receivables quality: sector-aware DSO threshold
    # Services/IT/Pharma/Financials have 60-90 day billing cycles — 120d is the alert level.
    # Products/FMCG/Manufacturing: >75d signals collection risk.
    if "industry" in df.columns:
        is_service = df["industry"].isin(_SERVICE_INDUSTRIES)
        dso_threshold = np.where(is_service, 120, 75)
    else:
        dso_threshold = 90  # fallback if industry column unavailable
    df["rf_high_receivables"] = np.where(
        df["days_receivable"].notna(),
        (df["days_receivable"] > dso_threshold).astype(int),
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

    # 8. Share dilution — uses the new 4-tier materiality system from data_engine
    # dilution_flag: 0=Clean, 1=ESOP-level(<3%), 2=Meaningful(3-10%), 3=Predatory QIP(>10%)
    # Forensic flag only activates for Tier 2+ (>3% meaningful dilution).
    # Tier 1 (tiny ESOPs) is NOT a forensic red flag — it is normal corporate practice.
    df["rf_dilution"] = np.where(
        df.get("dilution_flag", pd.Series(0, index=df.index)).fillna(0) >= 2,
        1,  # Flag: meaningful or predatory dilution detected
        0   # Clean: no dilution, or minor ESOP-level (<3%)
    )

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
        (df.get("ssgr_cushion", pd.Series(0, index=df.index)) < -5).astype(int),  # SSGR trails by 5%+
        0
    )

    # 14. Accrual Ratio: (PAT - OCF) / Avg_Total_Assets > 5% (Beneish TATA — highest single fraud predictor)
    # High accruals mean reported earnings are not backed by cash. Coefficient in Beneish = 4.679 (largest).
    # Use AVERAGE total assets (current + 1YB)/2 — Gemini G-audit fix: point-in-time denominator can be
    # artificially inflated by late-year acquisitions, masking accrual manipulation.
    avg_ta = (df["total_assets"].fillna(0) + df["total_assets_1yb"].fillna(df["total_assets"].fillna(0))) / 2.0
    df["rf_high_accruals"] = np.where(
        df["pat"].notna() & df["operating_cash_flow"].notna() & (avg_ta > 0),
        (((df["pat"] - df["operating_cash_flow"]) / avg_ta) > 0.05).astype(int),
        0
    )

    # 15. FCF/EBITDA below threshold (Malik Shenanigan #5 — EBITDA misleads)
    # When FCF/EBITDA < 30%, EBITDA is significantly overstating true cash earnings.
    # FCF and EBITDA are both in Crores — ratio is dimensionless. Only flag when EBITDA > 0.
    df["rf_low_fcf_ebitda"] = np.where(
        df["free_cash_flow"].notna() & df["ebitda"].notna() & (df["ebitda"] > 0),
        ((df["free_cash_flow"] / df["ebitda"]) < 0.30).astype(int),
        0
    )

    # 16. FCF/CFO conversion low (Vijay Malik Vol 3: Bharat Rasayan / PIX Transmissions pattern)
    # When OCF is positive yet FCF is <15% of OCF, the company is a capital trap:
    # capex consumes nearly all operating cash, leaving nothing for debt repayment, dividends, or growth.
    # Distinct from rf_negative_fcf (which catches OCF < 0). This catches the "false abundance" case.
    df["rf_fcf_to_cfo_low"] = np.where(
        df["free_cash_flow"].notna() & df["operating_cash_flow"].notna() &
        (df["operating_cash_flow"] > 0),
        (df["free_cash_flow"].fillna(0) / df["operating_cash_flow"] < 0.15).astype(int),
        0
    )

    # 17. OPM highly volatile vs 5Y median (commodity trap / no pricing power)
    # Vijay Malik Vol 3: Maithan Alloys — OPM swings 3% → 21% = pure price taker.
    # Finolex Cables OPM stays in 7-16% band = pricing power + structural improvement.
    # Threshold: >30% deviation from 5Y median OPM = unreliable margins.
    # G7 FIX: removed opm_latest_q fallback — quarterly OPM is seasonally distorted.
    # Indian businesses have intense seasonal cycles (festivals, monsoon, govt spending).
    # Comparing a single quarter against a 5Y annual median generates false positives.
    # Only compare annual OPM (opm_1yb) against annual median; if annual missing, skip the flag.
    opm_annual_compare = df["opm_1yb"].fillna(df["opm_med_5y"])
    df["rf_opm_volatile"] = np.where(
        df["opm_med_5y"].notna() & (df["opm_med_5y"] > 0),
        (
            (opm_annual_compare - df["opm_med_5y"]).abs() / df["opm_med_5y"] > 0.30
        ).astype(int),
        0
    )

    # 18. NFAT very low (extreme capital intensity — growth destroys shareholder value)
    # Vijay Malik Vol 3: PIX Transmissions — NFAT < 1.5 = every rupee of growth needs heavy capex.
    # SSGR turns negative, FCF turns negative, debt rises. Growth becomes value destruction.
    # Only flag when nfat column exists and is populated.
    df["rf_nfat_very_low"] = np.where(
        df.get("nfat", pd.Series(np.nan, index=df.index)).notna(),
        (df.get("nfat", pd.Series(np.nan, index=df.index)) < 1.5).astype(int),
        0
    )

    # 19. Capitalizing Expenses (WorldCom / Satyam pattern)
    # Gross CapEx > 3× Depreciation without proportional revenue growth = hiding expenses on balance sheet.
    # WorldCom inflated profits by capitalizing ordinary line-costs as capital assets.
    # BUG FIX: Previous version used NET FA change (= gross capex - depreciation) as the capex estimate.
    # Threshold of 3× on net change was equivalent to requiring gross capex/dep > 4× — under-sensitive.
    # Correct: gross capex ≈ net FA increase + estimated depreciation. Then ratio vs dep = intended 3×.
    depr_est_wc        = df["fixed_assets_1yb"].fillna(0) * 0.10  # conservative flat 10% dep rate
    capex_net_wc       = (df["fixed_assets"].fillna(0) - df["fixed_assets_1yb"].fillna(0)).clip(lower=0)
    capex_gross_est_wc = capex_net_wc + depr_est_wc               # gross capex = net change + dep
    df["rf_capitalizing_expenses"] = np.where(
        (depr_est_wc > 0) & df["rev_gr_yoy"].notna(),
        (
            (capex_gross_est_wc / depr_est_wc > FORENSIC.get("capex_depr_ratio_max", 3.0)) &
            (df["rev_gr_yoy"] < 15)
        ).astype(int),
        0
    )

    # 20. Debt/EBITDA > 5× (Amtek Auto / infrastructure sector danger signal)
    # Financial Shenanigans India Ch.6 Case 8: Amtek Auto collapsed when Debt/EBITDA exceeded 10×.
    # Ch.7 Infrastructure section: any D/EBITDA > 5× in a non-financial company = critical warning.
    # Excludes financial sector stocks where debt is a product, not a risk factor.
    df["rf_debt_ebitda_high"] = np.where(
        df["debt"].notna() & df["ebitda"].notna() & (df["ebitda"] > 0) &
        (~df.get("is_financial", pd.Series(False, index=df.index)).fillna(False)),
        (df["debt"] / df["ebitda"] > 5.0).astype(int),
        0
    )

    # 21. CWIP bloating without asset conversion (IL&FS / infrastructure capitalization trap)
    # Financial Shenanigans Ch.2 EMS#4: CWIP as % of total assets growing YoY signals expenses
    # being parked on the balance sheet to avoid hitting the P&L.
    # IL&FS used CWIP inflation across 347 entities to hide losses for years.
    # RUNTIME FIX: np.where evaluates both branches before condition check — dividing by
    # cwip_pct_1yb=0 caused RuntimeWarning even when guarded. Use safe denominator instead.
    cwip_pct_current = np.where(
        df["total_assets"].fillna(0) > 0,
        df["cwip"].fillna(0) / df["total_assets"],
        np.nan
    )
    cwip_pct_1yb = np.where(
        df["total_assets_1yb"].fillna(0) > 0,
        df["cwip_1yb"].fillna(0) / df["total_assets_1yb"],
        np.nan
    )
    safe_cwip_1yb = np.where(cwip_pct_1yb > 0, cwip_pct_1yb, 1.0)  # safe denominator: 1.0 when 0 or NaN
    df["rf_cwip_bloat"] = np.where(
        pd.notna(cwip_pct_current) & pd.notna(cwip_pct_1yb) & (cwip_pct_1yb > 0),
        ((cwip_pct_current / safe_cwip_1yb) > 1.5).astype(int),  # CWIP share of assets grew >50% YoY
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
        "rf_high_receivables": "High receivables (>120d services / >75d products)",
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
        "rf_ssgr_deficit": "Growth exceeds SSGR by 5%+ (debt-dependent)",
        "rf_high_accruals":   "High accruals — PAT not backed by cash (Beneish TATA >5%)",
        "rf_low_fcf_ebitda":  "FCF/EBITDA <30% — EBITDA overstates real earnings (Malik S5)",
        "rf_fcf_to_cfo_low":           "FCF/CFO <15% — capital trap, capex consuming all operating cash",
        "rf_opm_volatile":             "OPM >30% off 5Y median — commodity trap, no pricing power",
        "rf_nfat_very_low":            "NFAT <1.5 — extreme capital intensity, growth destroys value",
        "rf_capitalizing_expenses":    "CapEx >3× Depreciation without rev growth — WorldCom expense-hiding pattern",
        "rf_debt_ebitda_high":         "Debt/EBITDA >5× — Amtek Auto collapse pattern, critical for infra/real estate",
        "rf_cwip_bloat":               "CWIP share of assets grew >50% YoY — IL&FS-style balance sheet parking",
    }


    # Vectorized flag list — no apply(axis=1)
    red_flag_combined = pd.Series("", index=df.index)
    for col, desc in flag_descriptions.items():
        if col in df.columns:
            red_flag_combined = red_flag_combined + np.where(df[col] == 1, desc + " | ", "")
    stripped = red_flag_combined.str.rstrip(" | ")
    df["red_flag_list"] = np.where(stripped == "", "Clean ✅", stripped)

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
