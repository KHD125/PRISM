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
from config import FORENSIC, PIOTROSKI, FORENSIC_MAX_FLAGS, CONVICTION_TIERS

# Industries with long billing cycles — DSO > 90 days is normal, not a red flag
# Sector-based classification for DSO threshold and capex-mirage exclusion.
# Uses the SECTOR column (81 values) — auto-adapts when Screener.in adds new sub-industries
# without requiring any code change. Verified: these 4 sectors contain all 40+ relevant
# IT / Pharma / Healthcare industries in the live CSV.
# Financial stocks are excluded separately via is_financial (days_receivable is NaN-out there).
_HIGH_DSO_SECTORS = frozenset({
    "IT - Software",      # 11 industries: ER&D, BPO, Data Centre, IT Product, Geospatial…
    "IT - Hardware",      # 5 industries: Computer Hardware, Peripherals, Networking…
    "Healthcare",         # 9 industries: Hospitals, Diagnostics, Medical Equipment…
    "Pharmaceuticals",    # 15 industries: all Pharma sub-types, Ayurvedic, Biosimilars…
})


# ═══════════════════════════════════════════════════════════════
# PIOTROSKI F-SCORE (0–9)
# ═══════════════════════════════════════════════════════════════

def compute_piotroski_fscore(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Piotroski F-Score (0-9) for every stock. Pure vectorized.

    All column access uses df.get() with NaN fallback — safe even if a CSV sheet is
    missing from the upload or has a column-name mismatch. Missing columns default to
    NaN, which the np.where guards convert to 0 (conservative / no credit).
    """
    df = df.copy()
    _nan = pd.Series(np.nan, index=df.index, dtype=float)

    # Safe column aliases — no KeyError if any sheet was not loaded
    _roa     = df.get("roa",                 _nan)
    _roa_1yb = df.get("roa_1yb",             _nan)
    _roe     = df.get("roe",                 _nan)
    _roe_1yb = df.get("roe_1yb",             _nan)
    _ocf     = df.get("operating_cash_flow", _nan)
    _pat     = df.get("pat",                 _nan)
    _de      = df.get("debt_to_equity",      _nan)
    _de_1yb  = df.get("debt_to_equity_1yb",  _nan)
    _cr      = df.get("current_ratio",       _nan)
    _cr_1yb  = df.get("current_ratio_1yb",   _nan)
    _eq      = df.get("equity_shares",       _nan)
    _eq_1yb  = df.get("equity_shares_1yb",   _nan)
    _opm_q   = df.get("opm_latest_q",        _nan)
    _opm_1yb = df.get("opm_1yb",             _nan)
    _roce    = df.get("roce",                _nan)
    _roce_1yb= df.get("roce_1yb",            _nan)

    # F1: ROA positive
    df["f_roa_positive"] = np.where(_roa.notna(), (_roa > 0).astype(int), 0)

    # F2: Operating cash flow positive
    df["f_ocf_positive"] = np.where(_ocf.notna(), (_ocf > 0).astype(int), 0)

    # F3: ROA improving — use actual roa_1yb (available in RATIO_COLS).
    # Fall back to ROE proxy only when roa_1yb is unavailable.
    df["f_roa_improving"] = np.where(
        _roa.notna() & _roa_1yb.notna(),
        (_roa > _roa_1yb).astype(int),
        np.where(_roe.notna() & _roe_1yb.notna(), (_roe > _roe_1yb).astype(int), 0),
    )

    # F4: Accrual quality — OCF > PAT confirms earnings are cash-backed
    df["f_accrual_quality"] = np.where(
        _ocf.notna() & _pat.notna(), (_ocf > _pat).astype(int), 0
    )

    # F5: Leverage declining — D/E decreasing YoY
    df["f_leverage_declining"] = np.where(
        _de.notna() & _de_1yb.notna(), (_de < _de_1yb).astype(int), 0
    )

    # F6: Liquidity improving — current ratio increasing YoY
    # Guard: identical current and 1YB values indicate a data copy error in the source;
    # treat as "no change detected" → 0 (conservative, not penalised beyond no-credit).
    df["f_liquidity_improving"] = np.where(
        _cr.notna() & _cr_1yb.notna() & (_cr != _cr_1yb),
        (_cr > _cr_1yb).astype(int),
        0,
    )

    # F7: No dilution — shares not increased YoY (benefit of doubt if data missing)
    df["f_no_dilution"] = np.where(
        _eq.notna() & _eq_1yb.notna(),
        (_eq <= _eq_1yb).astype(int),
        1,
    )

    # F8: Gross margin improving — OPM latest quarter vs 1 year back
    df["f_margin_improving"] = np.where(
        _opm_q.notna() & _opm_1yb.notna(), (_opm_q > _opm_1yb).astype(int), 0
    )

    # F9: Asset turnover improving — ROCE direction used as proxy
    df["f_efficiency_improving"] = np.where(
        _roce.notna() & _roce_1yb.notna(), (_roce > _roce_1yb).astype(int), 0
    )

    # Sum exactly the 9 authentic Piotroski components — hard-coded list of actual column names
    # created above. Prevents any other f_* columns (fcf_yield, fcf_to_cfo_pct, etc.) from
    # being accidentally included if those columns happen to be present in df.
    _PIOTROSKI_COLS = [
        "f_roa_positive", "f_ocf_positive", "f_roa_improving",
        "f_accrual_quality", "f_leverage_declining", "f_liquidity_improving",
        "f_no_dilution", "f_margin_improving", "f_efficiency_improving",
    ]
    f_cols = [c for c in _PIOTROSKI_COLS if c in df.columns]
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

    # 1. CFO/PAT below threshold (cfo_to_pat is PERCENTAGE in CSV: 73.04 = 73%)
    df["rf_low_cfo_pat"] = np.where(
        df["cfo_to_pat"].notna(),
        (df["cfo_to_pat"] < FORENSIC["cfo_pat_alert"]).astype(int),
        0
    )

    # 2. Receivables quality: sector-aware DSO threshold
    # IT / Pharma / Healthcare have 60-90d normal billing cycles → alert at 120d.
    # Products / FMCG / Manufacturing → alert at 75d.
    # Financial stocks: days_receivable is NaN-out in data_engine → always 0 here.
    is_service = df.get("sector", pd.Series("", index=df.index)).fillna("").isin(_HIGH_DSO_SECTORS)
    dso_threshold = np.where(is_service, 120, 75)
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
    _ta_1yb = df.get("total_assets_1yb", pd.Series(np.nan, index=df.index)).fillna(df["total_assets"].fillna(0))
    avg_ta = (df["total_assets"].fillna(0) + _ta_1yb) / 2.0
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
    # Excluded when cwip_conversion > 0: CWIP actively converting to fixed assets = deliberate
    # capacity expansion capex, not inefficiency. High-quality compounders in build phase fire this flag.
    _cwip_conv_active = df.get("cwip_conversion", pd.Series(0.0, index=df.index)).fillna(0) > 0
    df["rf_fcf_to_cfo_low"] = np.where(
        df["free_cash_flow"].notna() & df["operating_cash_flow"].notna() &
        (df["operating_cash_flow"] > 0) & ~_cwip_conv_active,
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
    # Financial sector excluded: banks/NBFCs have near-zero fixed assets relative to revenue —
    # NFAT is structurally meaningless for them and would generate false flags.
    _nfat_vals = df.get("nfat", pd.Series(np.nan, index=df.index))
    _is_fin_nfat = df.get("is_financial", pd.Series(False, index=df.index)).fillna(False)
    df["rf_nfat_very_low"] = np.where(
        _nfat_vals.notna() & ~_is_fin_nfat,
        (_nfat_vals < 1.5).astype(int),
        0
    )

    # 19. rf_capitalizing_expenses — RETIRED (superseded by rf_capex_mirage, flag #22)
    # The prior logic penalized high capex (>3× dep without rev growth) — an inverted diagnostic:
    # it flagged companies actively reinvesting in their asset base, not those neglecting it.
    # rf_capex_mirage (flag #22) correctly identifies the deferred-maintenance time bomb:
    # rapid revenue growth (>20%) paired with capex/dep < 0.5 — under-investment, not over-investment.
    # Column deliberately omitted so the max_flags denominator in forensic_score stays accurate.

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
    _ta_1yb_cwip = df.get("total_assets_1yb", pd.Series(np.nan, index=df.index)).fillna(0)
    cwip_pct_1yb = np.where(
        _ta_1yb_cwip > 0,
        df["cwip_1yb"].fillna(0) / _ta_1yb_cwip.replace(0, np.nan),
        np.nan
    )
    safe_cwip_1yb = np.where(cwip_pct_1yb > 0, cwip_pct_1yb, 1.0)  # safe denominator: 1.0 when 0 or NaN
    df["rf_cwip_bloat"] = np.where(
        pd.notna(cwip_pct_current) & pd.notna(cwip_pct_1yb) & (cwip_pct_1yb > 0),
        ((cwip_pct_current / safe_cwip_1yb) > 1.5).astype(int),  # CWIP share of assets grew >50% YoY
        0
    )

    # 22. Capex Velocity Mismatch — Capitalization Mirage (WCS Quantamental Agent 8)
    # A capital-intensive company showing rapid revenue growth (>20% YoY) but net capex/depreciation
    # ratio < 0.5 is NOT investing enough to sustain that growth. This reveals a deferred-maintenance
    # time bomb: the asset base is silently aging while revenues appear to scale.
    #
    # Formula: capex_net_ratio = max(0, FA - FA_1YB) / (FA_1YB × 0.10)
    # - Numerator: actual net new capital deployed (0 if FA flat or declining)
    # - Denominator: estimated annual depreciation of existing asset base
    # - Ratio < 0.5 means less than half of depreciated assets are being replaced
    #
    # EXCLUSION: IT/Software, Pharma, Healthcare, and Financial companies are asset-light —
    # high revenue with low capex is their COMPETITIVE ADVANTAGE, not a red flag.
    # Only flag capital-intensive businesses (Manufacturing, Metals, Infra, Auto, etc.)
    _is_svc_cm = df.get("sector", pd.Series("", index=df.index)).fillna("").isin(_HIGH_DSO_SECTORS)
    _is_fin_cm = df.get("is_financial", pd.Series(False, index=df.index)).fillna(False)
    _is_capital_intensive = ~_is_svc_cm & ~_is_fin_cm

    _depr_est_cm   = df["fixed_assets_1yb"].fillna(0) * 0.10
    _capex_net_cm  = (df["fixed_assets"].fillna(0) - df["fixed_assets_1yb"].fillna(0)).clip(lower=0)
    _capex_ratio_cm = np.where(
        _depr_est_cm > 0,
        _capex_net_cm / _depr_est_cm,
        1.0  # safe default: unknown depreciation base → neutral (1.0 = replacing all)
    )
    df["rf_capex_mirage"] = np.where(
        _is_capital_intensive & df["rev_gr_yoy"].notna() & (_depr_est_cm > 0),
        (
            (df["rev_gr_yoy"] > 20) &
            (_capex_ratio_cm < 0.5)
        ).astype(int),
        0
    )

    # 23. Effective Tax Rate Panic (WCS 24: Sharp Practices)
    # Companies with PAT > 0 paying an effective tax rate < 10% raise governance concerns:
    # deferred tax asset exhaustion, tax holiday abuse, or opaque structured avoidance.
    # GUARD: Bypass entirely for loss-makers (PAT ≤ 0) — they inherently pay no tax.
    # GUARD: Bypass when PBT ≤ 0 (pre-tax loss) — negative PBT makes the ratio meaningless.
    #
    # Formula: effective_tax_rate = (PBT − PAT) / PBT × 100
    # Uses real PBT column from Income Statement CSV — no estimation needed.
    _pbt_tp = df.get("pbt", pd.Series(np.nan, index=df.index)).fillna(np.nan)
    _eff_tax_tp = np.where(
        _pbt_tp > 0,
        ((_pbt_tp - df["pat"].fillna(0)).clip(lower=0) / _pbt_tp) * 100,
        np.nan
    )
    df["rf_tax_panic"] = np.where(
        df["pat"].notna() & (df["pat"] > 0) & pd.notna(_eff_tax_tp),
        (_eff_tax_tp < 10).astype(int),
        0
    )

    # 24. Sector-Relative Receivables Expansion (WCS 24: Receivables Manipulation)
    # rf_high_receivables (flag #2) catches absolute DSO level violations.
    # This flag catches EXPANSION velocity beyond sector norms: a company stretching
    # receivables faster than peers, even when absolute DSO is within acceptable range.
    # Trigger: (days_receivable − days_receivable_1yb) > sector median expansion + 20 days.
    if "days_receivable" in df.columns and "days_receivable_1yb" in df.columns:
        _dso_exp_raw = df["days_receivable"] - df["days_receivable_1yb"]
        if "sector" in df.columns:
            _sector_g = df["sector"].fillna("Unknown")
            _sector_med_exp = _dso_exp_raw.groupby(_sector_g).transform("median")
        else:
            _sector_med_exp = pd.Series(_dso_exp_raw.median(), index=df.index)
        _dso_exp = _dso_exp_raw.fillna(0)
        df["rf_receivables_bloat"] = np.where(
            df["days_receivable"].notna() & df["days_receivable_1yb"].notna(),
            (_dso_exp > _sector_med_exp.fillna(0) + 20).astype(int),
            0
        )
    else:
        df["rf_receivables_bloat"] = 0

    # 25. PSU Value-Destruction Loop (Epoch 3)
    df["rf_psu_value_destruction"] = df.get("psu_value_destruction_flag", pd.Series(0, index=df.index)).fillna(0).astype(int)

    # 26. CFO/EBITDA Below Coffee Can Clean Accounts Floor
    # Saurabh Mukherjea "Coffee Can Investing" Ch.3: "CFO/EBITDA must be above 0.9
    # for every one of the last 10 years." Below 0.9 (= 90%) for any year demands
    # investigation; below 0.8 is a disqualifying signal — cash is not backing profits.
    # Distinct from rf_low_cfo_pat (which checks OCF vs PAT, not vs EBITDA):
    #   CFO/PAT ≥ 70% checks that profit converts to cash.
    #   CFO/EBITDA ≥ 90% checks that EBITDA itself is not artificially inflated
    #   via receivables stuffing, aggressive revenue recognition, or RPT revenue.
    # cfo_to_ebitda stored as percentage in CSV (e.g. 92.4 = 92.4%). Threshold = 90.0.
    df["rf_low_cfo_ebitda"] = np.where(
        df.get("cfo_to_ebitda", pd.Series(np.nan, index=df.index)).notna(),
        (df.get("cfo_to_ebitda", pd.Series(100.0, index=df.index)) < FORENSIC["cfo_ebitda_clean_threshold"]).astype(int),
        0
    )

    # 27. Ind AS 116 Lease Inflation — D4 forensic shield
    # Ind AS 116 (effective FY2020) forces companies to capitalise operating leases as Right-of-Use
    # (RoU) assets. Lease rentals move out of "Other Expenses" into Depreciation + Finance Costs.
    # Net effect: EBITDA looks inflated (rentals no longer in opex), OCF looks depressed (lease
    # payments shift to Financing Cash Flow). QSR chains, retail malls, airlines with large lease
    # bases appear as high-EBITDA, cash-generative businesses even when unit economics are weak.
    # Detection: EBITDA materially exceeds Operating Cash Flow in lease-intensive sectors.
    # Using exact sector names from the production CSV — verified against 81-sector master list.
    _lease_sectors = [
        "Retail",
        "Quick Service Restaurant",
        "Air Transport Service",
        "Hotels & Restaurants",
        "Logistics",
    ]
    _in_lease_sector = df.get("sector", pd.Series("", index=df.index)).fillna("").isin(_lease_sectors)
    _ebitda_vals     = df.get("ebitda", pd.Series(np.nan, index=df.index)).fillna(0)
    _ocf_vals        = df.get("operating_cash_flow", pd.Series(np.nan, index=df.index)).fillna(0)
    _ebitda_cfo_gap  = _ebitda_vals - _ocf_vals
    df["rf_lease_inflation"] = np.where(
        _in_lease_sector
        & df.get("ebitda", pd.Series(np.nan, index=df.index)).notna()
        & df.get("operating_cash_flow", pd.Series(np.nan, index=df.index)).notna()
        & (_ebitda_vals > 0)
        & (_ebitda_cfo_gap > _ebitda_vals * 0.30),
        1,
        0,
    ).astype(int)

    # Sum all red flags
    rf_cols = [c for c in df.columns if c.startswith("rf_")]
    df["red_flag_count"] = df[rf_cols].sum(axis=1)

    # ── Diamond-Specific Forensic Subset ───────────────────────────────────────
    # fw_diamond uses dm_forensic_flag_count == 0 instead of red_flag_count == 0.
    # Only 6 of the 27 flags are directly relevant to Mukherjea's Three-Lens Framework
    # (Diamonds in the Dust). The other 21 flags belong to Malik, Coffee Can, WCS24,
    # Ind AS 116, CWIP-specific, and other frameworks — applying them to Diamond would
    # penalise companies for failing standards that the Diamond book never checks.
    #
    # FLAG CLASSIFICATION:
    #   Lens 1 (Clean Accounts — what Diamonds explicitly checks):
    #     rf_low_cfo_pat        — CFO/PAT < 70%: core cash quality signal (Diamond uses CFO/PAT)
    #     rf_high_accruals      — Beneish TATA > 5%: strongest single fraud predictor; PAT not cash-backed
    #     rf_high_receivables   — Absolute DSO > threshold: channel-stuffing detection
    #     rf_receivables_bloat  — DSO expanding faster than sector peers: systematic manipulation
    #   Gate Zero extension (governance trajectory, beyond the static direct gates):
    #     rf_rising_debt        — D/E trend rising: balance sheet deterioration even when level < 0.5
    #     rf_dilution           — Share dilution ≥ 3%: promoter misalignment via equity erosion
    #
    #   NOT included:
    #     rf_low_cfo_ebitda  — Coffee Can Investing standard (CFO/EBITDA); Diamond uses CFO/PAT
    #     rf_pledge_elevated — Redundant: pledge < 10 is already a direct gate in fw_diamond
    #     rf_negative_fcf    — Redundant: FCF/CFO ≥ 25% direct gate already excludes these
    #     rf_fcf_to_cfo_low  — Redundant: FCF/CFO ≥ 25% direct gate already stricter
    #     All Malik flags    — Vijay Malik's framework; not cited in Diamonds in the Dust
    #     rf_opm_volatile, rf_nfat_very_low, rf_ssgr_deficit — Bakshi/Malik-specific
    #     rf_cwip_bloat, rf_capex_mirage, rf_lease_inflation — Sector/Ind AS specific
    #     rf_psu_value_destruction, rf_tax_panic — WCS24/PSU specific
    _DIAMOND_FLAGS = [
        "rf_low_cfo_pat",        # Lens 1: cash earnings quality
        "rf_high_accruals",      # Lens 1: Beneish TATA — highest fraud coefficient
        "rf_high_receivables",   # Lens 1: absolute DSO channel-stuffing guard
        "rf_receivables_bloat",  # Lens 1: sector-relative DSO expansion
        "rf_rising_debt",        # Gate Zero: D/E trajectory (level gate already in fw_diamond)
        "rf_dilution",           # Gate Zero: promoter dilution signal
    ]
    dm_flag_cols = [c for c in _DIAMOND_FLAGS if c in df.columns]
    df["dm_forensic_flag_count"] = df[dm_flag_cols].sum(axis=1).astype(int)

    # Forensic score: 100 = clean, 0 = maximum flags.
    # Uses FORENSIC_MAX_FLAGS (config.py) — fixed denominator so scores don't shift when
    # new flags are added. Update FORENSIC_MAX_FLAGS whenever a new rf_ column is added.
    df["forensic_score"] = ((FORENSIC_MAX_FLAGS - df["red_flag_count"]) / FORENSIC_MAX_FLAGS * 100).clip(0, 100)

    # ── Study 24 (2019): Management Integrity Score (0-3) ──
    # "32% of stocks listed in 2014 fell 70%+ by 2019 — Sharp Practices destroyed value."
    # Integrity = Clean Accounts + Low Pledge + Promoter Ownership Alignment.
    # 3 = highest integrity; 0 = all three red flags present.
    df["management_integrity_score"] = (
        (df["red_flag_count"] == 0).astype(int) +
        (df.get("pledged_percentage", pd.Series(100, index=df.index)).fillna(100) < 5).astype(int) +
        (df.get("promoter_holdings", pd.Series(0, index=df.index)).fillna(0) >= 50).astype(int)
    )

    # ── TWO SEPARATE FORENSIC SYSTEMS — both shown in the UI ──
    # forensic_score (0-100): continuous score across ALL 25 red flags. Higher = cleaner.
    #   Used by: fw_diamond (requires == 0), fw_dhandho (requires == 0), forensic_score column.
    # forensic_label (text): WCS 24 hard-gate classification using 4 specific conditions:
    #   CFO/PAT ≥ 80% + pledge < 10% + no dilution + zero red flags → "🟢 Clean"
    #   Any one fails → "🚨 Sharp Practices Detected"
    #   A stock CAN have forensic_score=95 (1 flag) but label="🚨 Sharp Practices Detected".
    #   This is intentional: the label is a hard binary for the SQGLP integrity gate.
    # ──────────────────────────────────────────────────────────────

    # Risk classification - WCS 24 Forensic Hard Gates Absolute Gatekeeper Model
    cfo_pat_valid = (df["cfo_to_pat"].fillna(0.0) >= 80.0)
    pledge_valid  = (df.get("pledged_percentage", pd.Series(0.0, index=df.index)).fillna(0.0) < 10.0)
    no_dilution   = (
        df.get("equity_shares",     pd.Series(np.nan, index=df.index)).fillna(0.0) <=
        df.get("equity_shares_1yb", pd.Series(np.nan, index=df.index)).fillna(0.0)
    )
    no_red_flags  = (df["red_flag_count"] == 0)

    integrity_pass = cfo_pat_valid & pledge_valid & no_dilution & no_red_flags
    df["forensic_label"] = np.where(integrity_pass, "🟢 Clean", "🚨 Sharp Practices Detected")

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
        "rf_debt_ebitda_high":         "Debt/EBITDA >5× — Amtek Auto collapse pattern, critical for infra/real estate",
        "rf_cwip_bloat":               "CWIP share of assets grew >50% YoY — IL&FS-style balance sheet parking",
        "rf_capex_mirage":             "Rapid rev growth (>20%) but capex <0.5× dep — deferred-maintenance time bomb",
        "rf_tax_panic":                "Estimated effective tax rate <10% despite PAT>0 — Sharp Practices alert (WCS 24)",
        "rf_receivables_bloat":        "DSO expansion >20 days above sector median — sector-relative receivables manipulation",
        "rf_psu_value_destruction":    "PSU Value-Destruction Loop (low spread, high payout, CWIP delays)",
        "rf_lease_inflation":          "Ind AS 116 lease mirage — EBITDA inflated by RoU capitalisation (QSR/Retail/Aviation)",
        "rf_low_cfo_ebitda":           "CFO/EBITDA <90% — Coffee Can clean accounts failure (Mukherjea master signal)",
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
    print(f"🛡️ WCS 24 Forensic Label Distribution:")
    print(f"   🟢 Clean: {(df['forensic_label'] == '🟢 Clean').sum()}")
    print(f"   🚨 Sharp Practices Detected: {(df['forensic_label'] == '🚨 Sharp Practices Detected').sum()}")

    return df


# ═══════════════════════════════════════════════════════════════
# SCHILIT FINANCIAL SHENANIGANS — 4-CHECKER FORENSIC OVERLAY
# ═══════════════════════════════════════════════════════════════

def compute_schilit_forensic_score(df: pd.DataFrame) -> pd.DataFrame:
    """Howard Schilit Financial Shenanigans — 4 independent vectorized gimmick checkers.

    Source: Financial Shenanigans (Schilit/Perler/Engelhart) + India Forensic Edition.
    Spec:   docs/schilit_forensic_specs.json

    Scoring:
        schilit_forensic_score = 100.0 − (15.0 × number_of_checkers_fired)
        schilit_pass           = 1 if schilit_forensic_score >= 70.0 (≤ 2 checkers fired)

    4 Checkers (one per Schilit category):
        1. ems_revenue_expense_manipulation  — EMS #1/3/4/7: accruals, inv/rev gap, OPM volatility,
                                               NPM−OPM > 5pp (non-op income), AT↓+OPM↑ (expense cap)
        2. cfs_cash_flow_distortion          — CFS #1-3: earnings-cash divergence, Paper Profits label,
                                               cfo_to_ebitda < 60% (OCF/EBITDA weakness)
        3. kms_balance_sheet_leverage_trap   — KMS #6, EMS #5: hidden obligations, high-cash-high-debt
        4. kms_inventory_receivables_bloat   — EMS #1, KMS #3: DSO expansion, inventory bloat,
                                               absolute DSO > 90 days (Schilit Ch.3/4/14)
           (Checker 4 excludes _HIGH_DSO_SECTORS and financial stocks — see forensic_engine.py header)

    Architecture: fully vectorized Pandas/NumPy — zero apply(), zero iterrows().
    All column access via df.get() with safe NaN / zero defaults.
    """
    df = df.copy()
    _nan_f    = pd.Series(np.nan, index=df.index, dtype=float)
    _zero_i   = pd.Series(0,      index=df.index, dtype=int)
    _str_e    = pd.Series("",     index=df.index, dtype=object)
    # Shared financial-sector exclusion mask — used by Checker 2 (CFS signal 3) and Checker 4
    _is_fin_g = df.get("is_financial", pd.Series(False, index=df.index)).fillna(False)

    # ── Checker 1: EMS — Revenue & Expense Manipulation ────────────────────────
    # EMS #1 (revenue too early), EMS #3 (one-time income), EMS #4 (expense cap), EMS #7 (big bath)
    # Signal 1: accruals_ratio > 5% of total assets (Beneish TATA, highest fraud coefficient 4.679)
    # Signal 2: inventory growing >20pp faster than revenue (channel stuffing / demand concealment)
    # Signal 3: OPM >30% off 5Y median (volatile margins = commodity trap or manipulation)
    # Signal 4: NPM − OPM > 5pp (non-operating income inflating PAT — EMS #3, Schilit Ch.5)
    # Signal 5: asset turnover ↓ 0.10+ while OPM ↑ 2pp+ (expense capitalisation — EMS #4, Schilit Ch.6)
    _accruals_warn  = df.get("accruals_warning",  _zero_i).fillna(0).astype(bool)
    _inv_rev_gap    = df.get("inv_vs_rev_gap",    _nan_f).fillna(0)
    _opm_stab       = df.get("opm_stability",     _nan_f).fillna(0)

    # Signal 4 (EMS #3): Non-operating income inflating PAT — Schilit Ch.5
    # When NPM exceeds OPM by 5+ percentage points, the company is relying heavily on
    # non-operating income (investment gains, asset sales, insurance settlements) to
    # boost reported PAT above what operations generate.
    # Book cases: Intel (option-income boosting EPS, Ch.5), Marvell (tax-rate engineering),
    #             Sunbeam (insurance settlement as recurring operating income, Ch.5)
    # NaN subtraction → NaN > 5.0 → False in pandas (missing data never fires — safe by design)
    _npm_vals    = df.get("npm", _nan_f)           # current annual Net Profit Margin (%)
    _opm_vals    = df.get("opm", _nan_f)           # current annual Operating Profit Margin (%)
    _other_income_inflate = (_npm_vals - _opm_vals) > 5.0

    # Signal 5 (EMS #4): Expense capitalisation proxy — AOL pattern (Schilit Ch.6, Ch.9)
    # AOL capitalised subscriber acquisition costs → OPM improved dramatically while asset
    # base grew (asset turnover fell). Pattern: asset turnover declining by 0.10+ units
    # AND OPM simultaneously improving by 2pp+ = operating costs being moved off the P&L
    # onto the balance sheet as "assets" to suppress reported expense ratios.
    # WorldCom case (Ch.9 EMS#7): $3.8B capex reclassification depressed asset turnover
    # while showing inflated EBITDA margins — the same dual-signal fingerprint.
    # NaN in any column → False (safe; only fires when both data points are available)
    _at_curr     = df.get("asset_turnover",     _nan_f)   # current asset turnover ratio
    _at_1yb_val  = df.get("asset_turnover_1yb", _nan_f)   # asset turnover 1 year back
    _opm_1yb_val = df.get("opm_1yb",            _nan_f)   # OPM 1 year back (%)
    _expense_cap_proxy = (
        ((_at_1yb_val - _at_curr) > 0.10) &     # asset turnover declined by 0.10+ units
        ((_opm_vals   - _opm_1yb_val) > 2.0)    # AND OPM simultaneously improved by 2pp+
    )

    # Signal 6 (EMS #4): Depreciation Rate Manipulation — Schilit Ch.6
    # When fixed assets grow (company is actively investing) but the D&A / Fixed Assets ratio
    # simultaneously falls, management has extended the accounting useful lives of assets to
    # reduce D&A expense and artificially inflate EBIT and PAT.
    #
    # Book cases (Schilit Ch.6 EMS #4 / Ch.9 EMS #7):
    #   Qwest Communications: extended cable asset lives from 14 → 40 years → reduced D&A by
    #     ~$1 billion/year. FA grew; dep/FA fell from ~7.1% to ~2.5%. SEC settled for $100M.
    #   WorldCom: reclassified $3.8B of operating line costs as "capital assets", then depreciated
    #     over 10-40 years instead of expensing immediately. FA grew; dep_rate fell. Led to
    #     bankruptcy and $11B restatement — largest accounting fraud in US history at the time.
    #   Sunbeam (Ch.5 EMS#3+EMS#4): extended intangible and fixed asset lives alongside aggressive
    #     restructuring charges to boost subsequent years' earnings.
    #
    # Data source: dep_rate = D&A / Fixed_Assets (%) — pre-computed in data_engine.py.
    #   D&A = EBITDA − EBIT (exact accounting identity; EBIT now mapped from Income Statement CSV).
    #
    # Threshold logic:
    #   FA grew >5%  → confirms company is investing in assets (eliminates declining-FA / retiring-
    #                  assets cases where dep/FA naturally rises as denominator shrinks)
    #   dep/FA fell >20% → effective useful life extended by at least 25% (10yr asset → 12.5yr min)
    #                      This catches the Qwest-scale manipulations (14→40yr = 65% dep/FA fall)
    #
    # Financial stocks excluded via _is_fin_g: banks have near-zero fixed assets relative to
    # their balance sheet — dep/FA is structurally tiny and unstable, not a manipulation signal.
    # NaN in dep_rate or dep_rate_1yb → NaN comparison → False (safe, does not fire on missing data).
    _fa_scht     = df.get("fixed_assets",     _nan_f)   # current gross fixed assets (Cr)
    _fa_1yb_scht = df.get("fixed_assets_1yb", _nan_f)   # fixed assets 1 year back (Cr)
    _dep_rt_curr = df.get("dep_rate",         _nan_f)   # D&A / FA %, current year
    _dep_rt_1yb  = df.get("dep_rate_1yb",     _nan_f)   # D&A / FA %, 1 year back
    _dep_rate_manip = (
        (_fa_scht    > _fa_1yb_scht * 1.05) &   # FA grew >5% (material — not rounding noise)
        (_dep_rt_curr < _dep_rt_1yb  * 0.80) &  # dep/FA fell >20% (life extension fingerprint)
        ~_is_fin_g                               # financial stocks excluded (no FA base)
    )

    ems_revenue_expense_manipulation = (
        _accruals_warn                  |   # Signal 1: accruals_warning == 1
        (_inv_rev_gap > 20.0)           |   # Signal 2: inv_vs_rev_gap > 20.0 pp
        (_opm_stab    > 30.0)           |   # Signal 3: opm_stability > 30.0 % deviation
        _other_income_inflate           |   # Signal 4: NPM − OPM > 5pp (EMS #3: non-op income)
        _expense_cap_proxy              |   # Signal 5: AT ↓ + OPM ↑ (EMS #4: expense cap proxy)
        _dep_rate_manip                     # Signal 6: FA ↑ but dep_rate ↓ (EMS #4: life extension)
    )

    # ── Checker 2: CFS — Cash Flow Distortion ──────────────────────────────────
    # CFS #1 (financing→operating reclassification), CFS #2 (unsustainable OCF boosts),
    # CFS #3 (boosting CFFO via working capital / AR sales / DPO stretching)
    # Signal 1: PAT growing >15% while OCF simultaneously shrinks >15% (accrual divergence)
    # Signal 2: cash_machine_label == "📄 Paper Profits" (cfo_to_pat <= 80%)
    # Signal 3: cfo_to_ebitda < 60% (OCF converts <60% of EBITDA — Schilit Ch.12 CFS #3)
    _pat_gr   = df.get("pat_gr_yoy",        _nan_f).fillna(0)
    _ocf_gr   = df.get("ocf_growth",        _nan_f).fillna(0)
    _cash_lbl = df.get("cash_machine_label", _str_e).fillna("")

    # Signal 3 (CFS #3): CFO/EBITDA weakness — OCF converts <60% of EBITDA (Schilit Ch.12)
    # Book cases used to anchor the 60% threshold:
    #   Cardinal Health: sold $800M of AR to a bank → CFFO jumped $971M in one quarter; OCF/EBITDA
    #     spiked temporarily, then collapsed — the structural level was far below 60% (Ch.12 CFS#2)
    #   Tesla Model 3: customer deposits $350M = 88% of total OCF improvement (Ch.12 CFS#3)
    #   Home Depot: DPO stretched 23→34 days = $3B unsustainable CFFO inflation (Ch.11 CFS#2)
    #   Sun Microsystems: one-time settlements inflating CFFO (Ch.12)
    # These cases share a pattern: EBITDA is nominal but OCF is structurally far below EBITDA
    # because working capital is burning cash (payables stretching, AR inflation, deposit games).
    # Threshold: 60.0% (stored as percentage — e.g. 73.04 means 73%). Financial stocks excluded
    # because for banks/NBFCs operating cash flow vs EBITDA is structurally incomparable.
    # NaN → NaN < 60.0 = False (pandas/numpy semantics — safe, does not fire on missing data)
    _cfo_ebitda      = df.get("cfo_to_ebitda", _nan_f)   # percentage form (e.g. 73.04)
    _cfo_ebitda_weak = (_cfo_ebitda < 60.0) & ~_is_fin_g

    cfs_cash_flow_distortion = (
        ((_pat_gr > 15.0) & (_ocf_gr < -15.0))     |   # Signal 1: earnings-cash divergence
        (_cash_lbl == "📄 Paper Profits")           |   # Signal 2: cfo_to_pat <= 80%
        _cfo_ebitda_weak                                # Signal 3: CFS #3 — OCF < 60% of EBITDA
    )

    # ── Checker 3: KMS — Balance Sheet Leverage Trap ───────────────────────────
    # KMS #6 (balance sheet distortion), EMS #5 (hidden obligations / IL&FS pattern)
    # Signal A: total liabilities growing faster than debt = hidden off-BS obligations
    # Signal B: high cash AND high debt simultaneously = optically strong, structurally risky
    _hid_oblig   = df.get("hidden_obligation_growth", _zero_i).fillna(0).astype(bool)
    _hi_cash_dbt = df.get("high_cash_high_debt",      _zero_i).fillna(0).astype(bool)

    kms_balance_sheet_leverage_trap = (
        _hid_oblig      |   # hidden_obligation_growth == 1
        _hi_cash_dbt        # high_cash_high_debt == 1
    )

    # ── Checker 4: KMS — Inventory & Receivables Bloat ─────────────────────────
    # EMS #1 (DSO rising = revenue manipulation), KMS #2 (distorted Balance Sheet metrics)
    # Excluded: IT/Pharma/Healthcare (_HIGH_DSO_SECTORS) + financial stocks
    #   Rationale: 60-90d billing cycles are contractual/structural — not manipulation signals.
    #
    # Signal 3 (absolute DSO) sourced directly from Schilit 4th Ed. book chapters:
    #   Ch.3  (EMS #1): Computer Associates — DSO hit 247 days; 20-day YoY rise triggered detection
    #   Ch.4  (EMS #2): Hanergy Solar — DSO ballooned to 500 days; 57% trade receivables past due
    #   Ch.14 (KMS #2): Symbol Technologies — DSO 94→119→90 days (cosmetically lowered via AR→notes conversion)
    #   Ch.14 (KMS #2): UTStarcom — DSO distorted by reclassifying AR as "bank notes" in cash section
    #   Threshold 90d: for non-IT/Pharma/Healthcare, >90 days indicates collection failure or revenue manipulation.
    _sector      = df.get("sector",      pd.Series("", index=df.index)).fillna("")
    _is_high_dso = _sector.isin(_HIGH_DSO_SECTORS)
    _eligible    = ~_is_high_dso & ~_is_fin_g   # _is_fin_g computed at function top (shared)

    _dso_delta3y = df.get("dso_delta_3y",          _nan_f).fillna(0)
    _inv_day_chg = df.get("inventory_days_change",  _nan_f).fillna(0)
    _days_recv   = df.get("days_receivable",         _nan_f).fillna(0)   # absolute DSO level

    kms_inventory_receivables_bloat = (
        _eligible &
        (
            (_dso_delta3y > 15.0)   |   # dso_delta_3y > 15.0 days over 3 years (EMS #1, trend signal)
            (_inv_day_chg > 20.0)   |   # inventory_days_change > 20.0 days YoY (channel stuffing)
            (_days_recv   > 90.0)       # ABSOLUTE DSO > 90 days (Schilit Ch.3/4/14: CA=247d, Hanergy=500d)
        )
    )

    # ── Score: 100.0 − 15.0 per fired checker ──────────────────────────────────
    # Each checker deducts 15.0 points. 4 checkers = max 60 deduction → min score 40.
    # clip(0, 100) ensures no score escapes valid range from unexpected edge cases.
    _score = (
        100.0
        - ems_revenue_expense_manipulation.astype(float) * 15.0
        - cfs_cash_flow_distortion.astype(float)         * 15.0
        - kms_balance_sheet_leverage_trap.astype(float)  * 15.0
        - kms_inventory_receivables_bloat.astype(float)  * 15.0
    ).clip(lower=0.0, upper=100.0)

    # ── Output columns ──────────────────────────────────────────────────────────
    df["schilit_ems_flag"]         = ems_revenue_expense_manipulation.astype(int)
    df["schilit_cfs_flag"]         = cfs_cash_flow_distortion.astype(int)
    df["schilit_kms_lev_flag"]     = kms_balance_sheet_leverage_trap.astype(int)
    df["schilit_kms_bloat_flag"]   = kms_inventory_receivables_bloat.astype(int)
    df["schilit_forensic_score"]   = _score
    # schilit_pass = 1 if score >= 70.0 (at most 2 of 4 checkers fired)
    df["schilit_pass"]             = (_score >= 70.0).astype(int)

    return df


# ═══════════════════════════════════════════════════════════════
# CASCADING FORENSIC FILTER
# ═══════════════════════════════════════════════════════════════

def compute_cascading_forensic_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Apply a multiplier to composite_score based on forensic red flag count.

    Separates binary forensic flags (gating) from continuous quality scoring.
    Red flags act as a multiplier layer — a single isolated flag does not collapse
    an otherwise excellent company's rank, but 5+ flags cut the score in half.

    Multiplier tiers (based on red_flag_count):
        0 flags   → 1.00 (full pass-through — clean company)
        1–2 flags → 0.90 (Watch — 10% discount)
        3–4 flags → 0.75 (Caution — 25% discount)
        5+ flags  → 0.50 (High Risk — 50% discount)
    """
    df = df.copy()

    flag_count = df.get("red_flag_count", pd.Series(0, index=df.index)).fillna(0)

    df["forensic_multiplier"] = np.select(
        [flag_count == 0, flag_count <= 2, flag_count <= 4],
        [1.0, 0.9, 0.75],
        default=0.5
    )

    if "composite_score" in df.columns:
        df["composite_score"] = (
            df["composite_score"] * df["forensic_multiplier"]
        ).clip(0, 100)

        # Reassign conviction_tier after forensic multiplier may have reduced composite_score.
        # Without this, a Crown Jewel (Tier 1, score 90) with 5 flags keeps Tier 1 label
        # even though its score is now 45 — label and score become completely inconsistent.
        conditions = []
        choices = []
        for tier in CONVICTION_TIERS:
            conditions.append(df["composite_score"] >= tier["min"])
            choices.append(tier["tier"])
        df["conviction_tier"] = np.select(conditions, choices, default=5)
        df["tier_label"] = df["conviction_tier"].map(
            {t["tier"]: f"{t['emoji']} {t['label']}" for t in CONVICTION_TIERS}
        )

    return df


# ═══════════════════════════════════════════════════════════════
# CASHFLOW QUALITY TRIANGLE
# ═══════════════════════════════════════════════════════════════

def compute_cashflow_triangle(df: pd.DataFrame) -> pd.DataFrame:
    """Classify each stock's cashflow pattern into the Quality Triangle."""
    df = df.copy()

    _ocf = df.get("operating_cash_flow",  pd.Series(0.0, index=df.index)).fillna(0)
    _icf = df.get("investing_cash_flow",  pd.Series(0.0, index=df.index)).fillna(0)
    _fcf = df.get("financing_cash_flow",  pd.Series(0.0, index=df.index)).fillna(0)
    ocf_pos = _ocf > 0
    icf_neg = _icf < 0
    fcf_neg = _fcf < 0
    fcf_pos = _fcf > 0

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
    df = compute_schilit_forensic_score(df)      # Schilit 4-checker overlay
    df = compute_cashflow_triangle(df)
    df = compute_cascading_forensic_filter(df)

    return df
