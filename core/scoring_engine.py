"""
Multibagger Discovery System — Scoring Engine
===============================================
4-Layer scoring architecture:
  Layer 1: Hard Gates (binary pass/fail)
  Layer 2: Quality Score (0–100)
  Layer 3: Momentum Score (0–100)
  Layer 4: Conviction Tier assignment

Pure vectorized Pandas. No loops over rows.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from config import (
    HARD_GATES, QUALITY_WEIGHTS, MOMENTUM_WEIGHTS, COMPOSITE_WEIGHTS,
    MOAT_SIGNALS, GROWTH_SIGNALS, CASH_SIGNALS, MARGIN_SIGNALS,
    BALANCE_SHEET_SIGNALS, RS_SIGNALS, TREND_SIGNALS, BREAKOUT_SIGNALS,
    SECTOR_SIGNALS, GOVERNANCE_BONUS, CONVICTION_TIERS, RSI_ZONES, HIGH_AGE_ZONES,
    MCAP_TIERS, MCAP_MIN_FLOOR, GOVERNANCE_RISK_MULTIPLIERS,
    VALUATION_SIGNALS, PEG_ZONES, PAYBACK_ZONES, MEAN_REVERSION, BAID_SELL_TRIGGERS,
    DEFAULT_CYCLE_TEMPERATURE, MARKS_CYCLE,
    MASTER_PROFILES, ANALYSIS_MODES, WAVE_DETECTION,
    REGIME_ADJUSTMENTS, get_adaptive_weights,
    EPOCH2_REINVESTMENT, EPOCH3_TAXONOMY, EPOCH4_SQGLP, EPOCH5_MODERN, EPOCH35_UNUSUAL_BILLIONAIRES,
    COST_OF_EQUITY,
)


# ═══════════════════════════════════════════════════════════════
# SCORE CONFIDENCE: the continuous, NaN-propagating inputs behind
# quality/momentum percentile ranking. A missing input becomes a
# neutral 50 in ranking, so a data-starved stock compresses toward
# "average" while really being "unknown" — data_coverage_pct makes
# that visible. Binary flags are EXCLUDED: they fail closed (0)
# when data is missing and never masquerade as evidence.
# Display-only; must never feed scores or weights (re-weighting by
# coverage is an unvalidated model — deliberately not done).
# Guarded by tests/test_score_confidence.py: every name here must
# be genuinely ranked in this module.
# ═══════════════════════════════════════════════════════════════
CORE_SCORING_INPUTS = [
    # Moat (sector-relative ranks)
    "roce_med_10y", "roe_med_10y", "roce_trajectory", "roe_trajectory",
    "roce_current_vs_med",
    # Growth (winsorized ranks)
    "pat_gr_5y", "pat_gr_10y", "rev_gr_5y", "rev_gr_10y", "eps_gr_5y",
    "ebitda_gr_5y", "pat_acceleration", "rev_acceleration",
    "ebitda_acceleration", "q_pat_yoy", "q_rev_yoy",
    # Cash quality
    "cfo_to_pat", "cfo_to_ebitda", "fcf_yield", "capex_coverage",
    "fcf_to_cfo_pct", "cash_change",
    # Margins
    "npm_med_5y", "opm_med_5y", "gpm_med_5y", "npm_acceleration",
    "opm_acceleration",
    # Balance sheet & efficiency
    "debt_slope_3y", "reserves_growth", "cwip_conversion", "nfat", "roce",
    # Valuation (QGLP price layer)
    "pe_discount", "pe_discount_to_quality", "peg", "ev_compression",
    # Momentum (continuous technicals)
    "dist_52wh", "dist_13wh", "dist_ath", "golden_cross_days",
    "dist_52wh_days", "rsi_14d", "adx_14w", "vol_ratio",
]


# ═══════════════════════════════════════════════════════════════
# UTILITY: Percentile rank with NaN handling
# ═══════════════════════════════════════════════════════════════

def _pct_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    """Percentile rank (0–100) with NaN preserved. Inf coerced to NaN before ranking.
    ascending=True means higher values get higher rank.
    ascending=False means lower values get higher rank.
    Inf purge: any ratio computation producing +/-inf (e.g. divide by near-zero) would
    otherwise rank as the universe maximum/minimum, injecting a spurious signal.
    Coercing to NaN lets fillna(50) assign neutral instead of a false extreme rank.
    """
    return series.replace([np.inf, -np.inf], np.nan).rank(pct=True, ascending=ascending, na_option='keep') * 100


def _safe_clip(series: pd.Series, lo: float = 0, hi: float = 100) -> pd.Series:
    """Clip series to [lo, hi] range."""
    return series.clip(lower=lo, upper=hi)


def _zone_score(value: pd.Series, zones: dict) -> pd.Series:
    """Score based on value falling in predefined zones — fully vectorized with np.select.
    Boundaries: left-closed [min, max) for all but the last zone (open upper bound).
    Zones evaluated in reverse order so the lowest-min zone wins for any value ≥ its min."""
    zone_list = list(zones.values())
    conditions = []
    choices = []
    for i, z in enumerate(zone_list):
        if i < len(zone_list) - 1:
            conditions.append((value >= z["min"]) & (value < z["max"]))
        else:
            conditions.append(value >= z["min"])
        choices.append(z["score"])
    return pd.Series(np.select(conditions, choices, default=50.0), index=value.index, dtype=float)


def _sector_pct_rank(df: pd.DataFrame, col: str, sector_col: str = "sector",
                     ascending: bool = True, fillna_val: float = 50.0) -> pd.Series:
    """Sector-relative percentile rank (0-100) using groupby().rank() — fully vectorized.
    Falls back to universe rank when sector column is missing or col not in df."""
    if col not in df.columns:
        return pd.Series(fillna_val, index=df.index)
    if sector_col not in df.columns:
        return _pct_rank(df[col], ascending=ascending).fillna(fillna_val)
    sector_grp = df[sector_col].fillna("Unknown")
    return (
        df[col].replace([np.inf, -np.inf], np.nan)
        .groupby(sector_grp)
        .rank(pct=True, ascending=ascending, na_option="keep")
        .mul(100)
        .fillna(fillna_val)
    )


# ═══════════════════════════════════════════════════════════════
# LAYER 1: HARD GATES
# ═══════════════════════════════════════════════════════════════

def apply_hard_gates(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all hard gates. Returns df with gate_pass column and gate details."""
    df = df.copy()

    gate_results = {}
    _missing_gate_names: set = set()

    for gate_name, gate_cfg in HARD_GATES.items():
        col = gate_cfg["column"]
        op = gate_cfg["operator"]
        threshold = gate_cfg["threshold"]

        if col not in df.columns or df[col].isna().all():
            # Column missing OR entire column is NaN (CSV tab failed to load — schema guard
            # materialized it as all-NaN). Both cases: benefit of doubt, exclude from gate count.
            gate_results[gate_name] = pd.Series(True, index=df.index)
            _missing_gate_names.add(gate_name)
            continue

        series = df[col]

        if op == "<=":
            passed = series <= threshold
        elif op == ">=":
            passed = series >= threshold
        elif op == "==":
            passed = series == threshold
        elif op == ">":
            passed = series > threshold
        elif op == "<":
            passed = series < threshold
        else:
            passed = pd.Series(True, index=df.index)

        # NaN handling: if data is missing, we give benefit of doubt for non-critical gates
        # but for critical gates (pledge, debt), NaN = fail
        # critical_gates: NaN = fail (missing data is as suspicious as failing)
        # cash_quality: missing CFO/PAT data should not pass the "earnings are real cash" gate
        # positive_pat: unknown profitability should not pass the no-loss-makers gate
        critical_gates = {"pledge_safety", "pledge_direction", "positive_ocf", "cash_quality", "positive_pat"}
        if gate_name in critical_gates:
            passed = passed.fillna(False)
        else:
            passed = passed.fillna(True)

        gate_results[gate_name] = passed
        df[f"gate_{gate_name}"] = passed.astype(int)

    # Financial sector stocks: relax debt_safety and current_ratio gates
    if "is_financial" in df.columns:
        fin_mask = df["is_financial"] == True
        for relaxed_gate in ["debt_safety", "current_ratio", "cash_quality"]:
            if relaxed_gate in gate_results:
                gate_results[relaxed_gate] = gate_results[relaxed_gate] | fin_mask
                df[f"gate_{relaxed_gate}"] = gate_results[relaxed_gate].astype(int)

    # ── ALPHA VECTOR: TURNAROUND DELTA (Rate of Change Override) ──
    # Peter Lynch Turnarounds: Do not punish a company getting radically better just because it misses absolute thresholds.
    if "debt_safety" in gate_results and "de_slope_3y" in df.columns:
        # Override Debt Gate if they are aggressively deleveraging (D/E dropped by > 0.15)
        deleveraging_mask = df["de_slope_3y"].fillna(0) < -0.15
        gate_results["debt_safety"] = gate_results["debt_safety"] | deleveraging_mask
        df["gate_debt_safety"] = gate_results["debt_safety"].astype(int)

    # NOTE: A ROCE inflection override was intended here (Peter Lynch turnarounds: if current ROCE > 5Y median + 3%,
    # override the gate). It referenced key "return_on_capital" which never existed in HARD_GATES — dead code.
    # Removed. The deleveraging override above (de_slope_3y < -0.15) already covers the turnaround case.

    # Overall pass: must pass ALL gates (missing-column gates are True, don't block pass).
    # gates_passed / gates_total count only gates with an actual column — accurate display.
    all_gates = pd.DataFrame(gate_results)
    _present_gates = {k: v for k, v in gate_results.items() if k not in _missing_gate_names}
    df["gate_pass"] = all_gates.all(axis=1).astype(int)
    if _present_gates:
        df["gates_passed"] = pd.DataFrame(_present_gates).sum(axis=1).astype(int)
    else:
        df["gates_passed"] = pd.Series(0, index=df.index)
    df["gates_total"]  = len(_present_gates)
    df["gates_failed"] = df["gates_total"] - df["gates_passed"]

    # Build a human-readable failed gates string — fully vectorized (no apply/iterrows)
    # np.where per column avoids numpy string-multiply ufunc (broken in NumPy 2.x).
    gate_names = list(gate_results.keys())
    fail_df = pd.DataFrame(gate_results)
    failed_str = pd.Series("", index=df.index, dtype=object)
    for _gn in gate_names:
        failed_str = failed_str + pd.Series(
            np.where(~fail_df[_gn].values, _gn + ", ", ""),
            index=df.index, dtype=object,
        )
    failed_str = failed_str.str.rstrip(", ")
    df["failed_gates"] = np.where(failed_str != "", failed_str, "All passed ✅")

    passed_count = df["gate_pass"].sum()
    total = len(df)
    print(f"\n🚪 Hard Gates: {passed_count}/{total} stocks passed ({passed_count/total*100:.1f}%)")

    return df


# ═══════════════════════════════════════════════════════════════
# LAYER 2: QUALITY SCORE
# ═══════════════════════════════════════════════════════════════

def _compute_moat_score(df: pd.DataFrame) -> pd.Series:
    """Moat score: ROCE trajectory + ROE evaluated within industry/sector cohorts.
    Uses groupby-based _sector_pct_rank (pure sector-relative rank) for peer benchmarking.
    MEF (Moat Endurance Factor) modifier from 17th WCS boosts or penalises structural durability."""
    score = pd.Series(0.0, index=df.index)
    # Long-term signals require 10-year history to be meaningful.
    # fillna(0) for these two: a stock listed <10 years has NOT proven its moat — neutral credit is unearned.
    # Trajectory/current signals use fillna(50): short history doesn't invalidate recent performance trends.
    _long_term_signals = {"roce_med_10y", "roe_med_10y"}
    signals = {
        "roce_med_10y":        (_sector_pct_rank(df, "roce_med_10y", ascending=True), 0.35),
        "roce_trajectory":     (_sector_pct_rank(df, "roce_trajectory", ascending=True), 0.15),
        "roe_med_10y":         (_sector_pct_rank(df, "roe_med_10y", ascending=True), 0.25),
        "roe_trajectory":      (_sector_pct_rank(df, "roe_trajectory", ascending=True), 0.10),
        "roce_current_vs_med": (_sector_pct_rank(df, "roce_current_vs_med", ascending=True), 0.15),
    }

    for name, (ranked, weight) in signals.items():
        fill = 0 if name in _long_term_signals else 50
        score += ranked.fillna(fill) * weight

    # Moat Endurance Factor (17th WCS): structural moat durability modifier.
    # MEF = ROCE_current / ROCE_med_10y. Stable/expanding ≥ 1.0 = moat intact.
    # Boost expanding moats; penalize degrading ones where current ROCE trails long-run median.
    if "moat_endurance_factor" in df.columns:
        _mef = df["moat_endurance_factor"].fillna(1.0)
        _mef_adj = pd.Series(
            np.select(
                [_mef >= 1.2,  _mef >= 1.0,  _mef >= EPOCH3_TAXONOMY["mef_eroding_threshold"]],
                [5.0,           2.0,           0.0],
                default=-8.0   # MEF < 0.80: ROCE has degraded below 80% of 10Y median
            ),
            index=df.index
        )
        score = score + _mef_adj

    return _safe_clip(score)


def _compute_growth_score(df: pd.DataFrame) -> pd.Series:
    """Growth score: Revenue, PAT, EPS compounding + acceleration + quarterly freshness.
    Weights: long-term signals 0.94 + quarterly freshness 0.06 = 1.00"""
    score = pd.Series(0.0, index=df.index)

    def _pct_rank_w(col: str) -> pd.Series:
        """Winsorized percentile rank: clips at p01-p99 before ranking.
        Prevents extreme outliers (e.g. IOC PAT YoY +528%, COFORGE EPS +1068%)
        from compressing the remaining 2,100+ stocks' rank distribution.
        A stock with 25% growth would score near the median when a handful of
        stocks have >500% growth — winsorization restores discrimination power.
        Requires ≥ 20 non-null values to be meaningful; falls back to raw rank.
        Source: Coffee Can compendium D3 data quality protocol."""
        if col not in df.columns:
            return pd.Series(50.0, index=df.index)
        s = df[col]
        n_valid = s.notna().sum()
        if n_valid < 20:
            return _pct_rank(s, ascending=True).fillna(50)
        lo, hi = s.quantile(0.01), s.quantile(0.99)
        return _pct_rank(s.clip(lower=lo, upper=hi), ascending=True).fillna(50)

    signals = {
        "pat_gr_5y":           (True, 0.17),   # -0.03 to fund quarterly freshness layer
        "pat_gr_10y":          (True, 0.10),
        "rev_gr_5y":           (True, 0.17),   # -0.03 to fund quarterly freshness layer
        "rev_gr_10y":          (True, 0.10),
        "eps_gr_5y":           (True, 0.15),
        "ebitda_gr_5y":        (True, 0.10),
        "pat_acceleration":    (True, 0.06),
        "rev_acceleration":    (True, 0.05),
        "ebitda_acceleration": (True, 0.04),
    }
    # Signals sum: 0.17+0.10+0.17+0.10+0.15+0.10+0.06+0.05+0.04 = 0.94

    # Use winsorized rank for all growth signals to prevent outlier compression.
    # Acceleration signals (pat_acceleration, rev_acceleration, ebitda_acceleration)
    # are differences of CAGR values — far smaller magnitude, still benefit from winsorization.
    for col, (_ascending, weight) in signals.items():
        score += _pct_rank_w(col) * weight

    # Quarterly freshness (6%): latest-quarter YoY — 4-6 months fresher than annual data.
    # q_pat_yoy drives 60% (profit is the bottom line); q_rev_yoy drives 40% (top-line health).
    # Catches earnings inflections the annual score misses for 1-2 quarters post-result.
    _q_pat = _pct_rank(df.get("q_pat_yoy", pd.Series(np.nan, index=df.index)), ascending=True).fillna(50)
    _q_rev = _pct_rank(df.get("q_rev_yoy", pd.Series(np.nan, index=df.index)), ascending=True).fillna(50)
    score += (_q_pat * 0.60 + _q_rev * 0.40) * 0.06  # 0.94 + 0.06 = 1.00 ✅

    return _safe_clip(score)


def _compute_cash_score(df: pd.DataFrame) -> pd.Series:
    """Cash quality score: CFO ratios, FCF yield, FCF/CFO conversion strictly within sector cohorts."""
    score = pd.Series(0.0, index=df.index)

    # Sector-relative cohort cash quality signals
    score += _sector_pct_rank(df, "cfo_to_pat", ascending=True) * 0.20
    score += _sector_pct_rank(df, "cfo_to_ebitda", ascending=True) * 0.15
    score += _sector_pct_rank(df, "fcf_yield", ascending=True) * 0.15
    score += _sector_pct_rank(df, "capex_coverage", ascending=True) * 0.10

    # FCF/CFO: handled separately because negative-OCF companies must score 0, not neutral 50.
    ranked_fcf_cfo = _sector_pct_rank(df, "fcf_to_cfo_pct", ascending=True)
    if "operating_cash_flow" in df.columns:
        ranked_fcf_cfo = pd.Series(
            np.where(df["operating_cash_flow"].fillna(0) <= 0, 0.0, ranked_fcf_cfo),
            index=df.index
        )
    score += ranked_fcf_cfo * 0.15

    # Binary signals (0 or 100)
    binary = {
        "fcf_consistency": 0.15,   # FCF positive over time
        "self_funding":    0.10,   # SSGR ≥ actual growth (no external debt needed)
    }
    for col, weight in binary.items():
        if col in df.columns:
            score += df[col].fillna(0) * 100 * weight

    return _safe_clip(score)


def _compute_margin_score(df: pd.DataFrame) -> pd.Series:
    """Margin score: pricing power via NPM, OPM, GPM medians + acceleration + OPM stability.
    Vijay Malik: stable OPM = pricing power moat; volatile OPM = commodity trap.
    Weights: npm_med_5y=0.25, opm_med_5y=0.25, gpm_med_5y=0.15,
             npm_acceleration=0.15, opm_acceleration=0.10, opm_stable=0.10. Sum=1.00"""
    score = pd.Series(0.0, index=df.index)

    # Sector-normalized margin medians: commodity sectors (steel, cement) have 5% OPM norms;
    # branded FMCG sectors have 20%+. Blend 70% universe + 30% sector-relative removes this bias.
    for col, weight in (("npm_med_5y", 0.25), ("opm_med_5y", 0.25), ("gpm_med_5y", 0.15)):
        if col in df.columns:
            univ_r = _pct_rank(df[col], ascending=True).fillna(50)
            sect_r = _sector_pct_rank(df, col, ascending=True)
            score += (univ_r * 0.70 + sect_r * 0.30) * weight

    # Acceleration signals: improvement velocity is cross-sectorally comparable
    for col, weight in (("npm_acceleration", 0.15), ("opm_acceleration", 0.10)):
        if col in df.columns:
            score += _pct_rank(df[col], ascending=True).fillna(50) * weight

    # OPM stability (binary): stable OPM within ±20% of 5Y median = pricing power
    if "opm_stable" in df.columns:
        score += df["opm_stable"].fillna(0) * 100 * 0.10

    return _safe_clip(score)


def _compute_balance_sheet_score(df: pd.DataFrame) -> pd.Series:
    """Balance sheet score: fortress detection, deleveraging, CWIP conversion, capital efficiency.
    Vijay Malik: NFAT > 5 = capital-light moat (Finolex Cables). NFAT < 1.5 = capital trap.
    Weights: net_debt_negative=0.25, debt_slope=0.20, reserves_growth=0.15,
             cwip_conversion=0.15, cash_change=0.15, nfat=0.10. Sum=1.00"""
    score = pd.Series(0.0, index=df.index)

    # Net debt negative is a binary fortress signal
    if "net_debt_negative" in df.columns:
        score += df["net_debt_negative"].fillna(0) * 100 * 0.25

    # Debt slope: negative is good (deleveraging)
    if "debt_slope_3y" in df.columns:
        score += _pct_rank(df["debt_slope_3y"], ascending=False).fillna(50) * 0.20

    # Reserves growth: higher is better
    if "reserves_growth" in df.columns:
        score += _pct_rank(df["reserves_growth"], ascending=True).fillna(50) * 0.15

    # CWIP conversion: positive means capacity went live
    if "cwip_conversion" in df.columns:
        score += _pct_rank(df["cwip_conversion"], ascending=True).fillna(50) * 0.15

    # Cash change: positive is good
    if "cash_change" in df.columns:
        score += _pct_rank(df["cash_change"], ascending=True).fillna(50) * 0.15

    # NFAT: Net Fixed Asset Turnover — capital-light moat (Vijay Malik)
    # Higher NFAT = revenue per rupee of fixed assets = can grow without heavy capex
    if "nfat" in df.columns:
        score += _pct_rank(df["nfat"], ascending=True).fillna(50) * 0.10

    return _safe_clip(score)


def _compute_valuation_score(df: pd.DataFrame) -> pd.Series:
    """Valuation attractiveness: Marks + Baid entry price discipline strictly within sector cohorts.
    Lower valuations = higher score = better entry point."""
    score = pd.Series(0.0, index=df.index)

    # PE discount signal — blended: 40% trailing (vs own 10Y median) + 60% quality-adjusted (EVA Fair PE)
    # Trailing pe_discount rewards stocks cheap vs their own history (momentum of cheapness).
    # EVA pe_discount_to_quality rewards stocks cheap vs what their quality justifies (structural cheapness).
    # Blending captures both dimensions. Graceful fallback: if EVA column missing, uses trailing only.
    _pe_trail = _sector_pct_rank(df, "pe_discount", ascending=True)
    if "pe_discount_to_quality" in df.columns:
        _pe_qadj = _sector_pct_rank(df, "pe_discount_to_quality", ascending=True)
        _pe_sig  = (_pe_trail * 0.40 + _pe_qadj * 0.60).fillna(_pe_trail.fillna(50))
    else:
        _pe_sig  = _pe_trail
    score += _pe_sig * VALUATION_SIGNALS["pe_discount"]

    # PEG: lower is better (ascending=False)
    # Apply maximum penalty for negative/invalid PEG
    peg_score = _sector_pct_rank(df, "peg", ascending=False)
    peg_score = np.where(df["peg"].fillna(999.0) < 0, 5.0, peg_score)
    score += pd.Series(peg_score, index=df.index) * VALUATION_SIGNALS["peg_ratio"]

    # EV/EBITDA compression: positive ev_compression = getting cheaper = good
    score += _sector_pct_rank(df, "ev_compression", ascending=True) * VALUATION_SIGNALS["ev_compression"]

    # FCF yield: higher = more attractive (Marks: > 3% large-cap, > 4% mid-cap)
    score += _sector_pct_rank(df, "fcf_yield", ascending=True) * VALUATION_SIGNALS["fcf_yield_val"]

    # Baid's D/E < 0.5 fortress bonus (net cash companies score highest)
    if "debt_to_equity" in df.columns:
        fortress = np.where(df["debt_to_equity"] < 0.1, 100,
                  np.where(df["debt_to_equity"] < 0.3, 85,
                  np.where(df["debt_to_equity"] < 0.5, 70,
                  np.where(df["debt_to_equity"] < 1.0, 40, 10))))
        score += pd.Series(fortress, index=df.index, dtype=float) * VALUATION_SIGNALS["de_fortress"]

    # Payback Ratio: MOSL's most validated supernormal-return predictor (all 30 studies)
    # payback_ratio = market_cap / 5Y cumulative estimated PAT (growth-adjusted)
    # payback_ratio_proxy (15th WCS UU) = PE / PAT_growth_YoY — reactive crisis scanner.
    # When full payback_ratio is NaN, the proxy ensures all PE-bearing stocks score here.
    _payback_val = df.get("payback_ratio", pd.Series(np.nan, index=df.index)).copy()
    if "payback_ratio_proxy" in df.columns:
        _payback_val = _payback_val.fillna(df["payback_ratio_proxy"])
    if _payback_val.notna().sum() > 0:
        payback_score = _zone_score(_payback_val.clip(lower=0, upper=998), PAYBACK_ZONES)
        score += payback_score.fillna(50) * VALUATION_SIGNALS["payback_ratio"]

    # Agent 9 (1st WCS): Valuation Multiple Trap — PE > 35 AND ROE < 18%.
    # Market is pricing in returns the business cannot sustainably generate.
    # Slash valuation score by 40% for these stocks; the quality score still reflects fundamentals.
    if "valuation_multiple_trap" in df.columns:
        trap_mask = df["valuation_multiple_trap"].fillna(0) == 1
        score = pd.Series(
            np.where(trap_mask, score * 0.60, score),
            index=df.index
        )

    # ── 9th WCS: Cyclical Peak Trap — dampen the deceptive low-P/E reward ──
    # A commodity stock at cyclical peak shows a low P/E only because earnings are temporarily inflated.
    # pe_discount/payback above rewarded that "cheapness"; here we claw back 30% of the valuation score
    # so peak-cycle commodities don't rank as bargains. Dampen (×0.70), not destroy — it's a timing
    # signal, not a quality verdict. Fires only on the surgical triple-AND from data_engine.
    if "cyclical_peak_trap" in df.columns:
        _cyc_mask = df["cyclical_peak_trap"].fillna(0) == 1
        score = pd.Series(
            np.where(_cyc_mask, score * 0.70, score),
            index=df.index
        )

    # ── P/E Below Sustainable ROE — Soft Valuation Boost (1st–30th WCS, most consistent rule) ──
    # "Inherent margin of safety exists in buying at P/E substantially lower than sustainable ROE."
    # Confirmed in all 30 Annual Wealth Creation Studies — the most repeated single valuation rule.
    # +3 pts soft boost: meaningful signal but not dominant; _safe_clip enforces 100 ceiling.
    if "pe_below_roe" in df.columns:
        score = score + df["pe_below_roe"].fillna(0) * 3.0

    return _safe_clip(score)


def compute_quality_score(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the composite quality score (Layer 2).
    Integrates 6 sub-scores: Moat + Growth + Cash + Margin + Balance Sheet + Valuation.
    Applies Marks' Mean Reversion Risk penalty for cyclical peak margins.
    Detects Baid's Sell Triggers for existing holding alerts."""
    df = df.copy()

    df["moat_score"] = _compute_moat_score(df)
    df["growth_score"] = _compute_growth_score(df)
    df["cash_score"] = _compute_cash_score(df)
    df["margin_score"] = _compute_margin_score(df)
    df["balance_sheet_score"] = _compute_balance_sheet_score(df)
    df["valuation_score"] = _compute_valuation_score(df)

    # Weighted composite
    df["quality_score"] = (
        df["moat_score"] * QUALITY_WEIGHTS["moat"] +
        df["growth_score"] * QUALITY_WEIGHTS["growth"] +
        df["cash_score"] * QUALITY_WEIGHTS["cash"] +
        df["margin_score"] * QUALITY_WEIGHTS["margin"] +
        df["balance_sheet_score"] * QUALITY_WEIGHTS["balance_sheet"] +
        df["valuation_score"] * QUALITY_WEIGHTS["valuation"]
    )

    # ── MEAN REVERSION RISK (Marks: "Extremes revert toward average") ──
    # Flag stocks where current margins are way above 5Y medians = cyclical peak
    opm_spike = np.where(
        df["opm_med_5y"].notna() & (df["opm_med_5y"] > 0),
        df["opm_latest_q"] / df["opm_med_5y"],
        1.0
    )
    npm_spike = np.where(
        df["npm_med_5y"].notna() & (df["npm_med_5y"] > 0),
        df["npm_latest_q"] / df["npm_med_5y"],
        1.0
    )
    df["mean_reversion_risk"] = (
        (pd.Series(opm_spike, index=df.index) > MEAN_REVERSION["opm_spike_threshold"]) |
        (pd.Series(npm_spike, index=df.index) > MEAN_REVERSION["npm_spike_threshold"])
    ).astype(int)

    # Apply penalty to quality score for cyclical peak risk
    df["quality_score"] = np.where(
        df["mean_reversion_risk"] == 1,
        df["quality_score"] * MEAN_REVERSION["penalty_factor"],
        df["quality_score"]
    )

    # Agent 10 (6th WCS): Elite ROE Tier sub-factor — ROE ≥ 35% mapped into quality_score.
    # 6th Study (1996-2001): 12 companies with ROE > 35% created 50% of all wealth in that period.
    # Wired as 0.25-weight sub-factor within the moat bucket: contribution = 0.25 × 100 × moat_weight = 5.5 pts max.
    # Bounded by _safe_clip at step end — cannot exceed 100.
    if "roe_elite_flag" in df.columns:
        _roe_elite_contrib = df["roe_elite_flag"].fillna(0) * 100.0 * 0.25 * QUALITY_WEIGHTS["moat"]
        df["quality_score"] = df["quality_score"] + _roe_elite_contrib

    # ── EPOCH 2 (7th–12th WCS, 2002–2007): Self-Funding Scale Velocity ──
    # Agent 10 / 12th WCS: flag_epoch2_compounder — high-retention mid/small-cap ROCE leaders.
    # Three concurrent conditions (all must hold):
    #   (1) RR ≥ 60%: retaining ≥60% of earnings internally — no payout dilution of compounding base.
    #   (2) ROCE_10Y ≥ 20%: sustained capital efficiency — proven reinvestment returns over full cycle.
    #   (3) Scalable category: Mid/Small/Micro/Nano Cap — hasn't hit structural size ceiling yet.
    # 12th WCS empirical: this configuration produced the highest wealth-creation velocity in 2002-07.
    # Boost: +10 pts to quality_score (threshold-agnostic; final _safe_clip caps at 100).
    _epoch2_req_cols = ["reinvestment_rate", "fundamental_growth_capacity", "mcap_tier"]
    if all(c in df.columns for c in _epoch2_req_cols):
        df["flag_epoch2_compounder"] = (
            (df["reinvestment_rate"].fillna(0)     >= EPOCH2_REINVESTMENT["min_reinvestment_rate"]) &
            (df.get("roce_med_10y", pd.Series(0.0, index=df.index)).fillna(0)
             >= EPOCH2_REINVESTMENT["min_capital_efficiency"]) &
            df["mcap_tier"].isin(["Mid Cap", "Small Cap", "Micro Cap", "Nano Cap"])
        ).astype(int)
        df["quality_score"] = (
            df["quality_score"]
            + df["flag_epoch2_compounder"].fillna(0) * EPOCH2_REINVESTMENT["quality_boost_pts"]
        )
    else:
        df["flag_epoch2_compounder"] = 0

    # Capital Misallocation Risk Penalty (Agent 9 / 11th WCS): Buffett VCR test.
    # Companies retaining >50% of earnings but generating VCR < 1.0 are destroying minority value.
    # Apply 10% quality_score haircut — business is structurally misallocating the retained capital.
    if "capital_misallocation_risk" in df.columns:
        _cmal = df["capital_misallocation_risk"].fillna(0) == 1
        df["quality_score"] = pd.Series(
            np.where(_cmal,
                     df["quality_score"] * EPOCH2_REINVESTMENT["misallocation_penalty"],
                     df["quality_score"]),
            index=df.index
        )

    # ── 27th Study: Consistents Bonus / Volatiles Penalty ──
    # "Consistency is the source of outperformance; Volatility is the source of underperformance."
    # Bonus  +5 pts: consistency_champion == 1 (5Y crash-free, terminal > initial, CAGR > 0).
    # Penalty -10 pts: mosl_volatile_flag == 1 (severe recent crash OR 5Y contraction).
    # Both vectorized; _safe_clip at function end enforces [0, 100] ceiling.
    if "consistency_champion" in df.columns:
        df["quality_score"] = pd.Series(
            np.where(
                df["consistency_champion"].fillna(0) == 1,
                df["quality_score"] + 5.0,
                df["quality_score"]
            ),
            index=df.index
        )
    if "mosl_volatile_flag" in df.columns:
        df["quality_score"] = pd.Series(
            np.where(
                df["mosl_volatile_flag"].fillna(0) == 1,
                (df["quality_score"] - 10.0).clip(lower=0),
                df["quality_score"]
            ),
            index=df.index
        )

    # ── 23rd Study: Growth-Value Trap Penalty (Exhibit 8/9) ──
    # High growth + ROE < Cost of Equity = ACTIVE value destruction (negative firm value).
    # The high growth inflates growth_score; this -8 penalty offsets that deceptive lift so a
    # value-destroyer growing at 30% does not outrank a true compounder. Floor-clamped at 0.
    if "growth_value_trap" in df.columns:
        df["quality_score"] = pd.Series(
            np.where(
                df["growth_value_trap"].fillna(0) == 1,
                (df["quality_score"] - 8.0).clip(lower=0),
                df["quality_score"]
            ),
            index=df.index
        )

    # ── 28th Study: EP Power Curve Top-Quintile Bonus (Exhibit 10) ──
    # Companies in EP Quintile 1/2 (top-40% by absolute economic profit) delivered 24%/21% CAGR
    # vs 4-8% for Q4/Q5 — averaged across six overlapping 10-year periods. +3 modest reinforcement
    # (kept small to avoid over-rewarding size, since absolute EP correlates with market cap).
    if "ep_top_quintile_flag" in df.columns:
        df["quality_score"] = pd.Series(
            np.where(
                df["ep_top_quintile_flag"].fillna(0) == 1,
                df["quality_score"] + 3.0,
                df["quality_score"]
            ),
            index=df.index
        )

    # ── 14th Study: Winning Investment Bonus (Category Winner in a Winner Category) ──
    # The study's ultimate target = a sector LEADER operating inside a fast-growing (>1.5x GDP) sector.
    # A leader in a stagnant sector is less attractive than a leader riding a structural tailwind;
    # +3 rewards the combined signal (additive to category leadership, not double-counting).
    if "category_winner_in_winner_cat" in df.columns:
        df["quality_score"] = pd.Series(
            np.where(
                df["category_winner_in_winner_cat"].fillna(0) == 1,
                df["quality_score"] + 3.0,
                df["quality_score"]
            ),
            index=df.index
        )

    # ── BAID SELL TRIGGERS (alert flags for existing holdings) ──
    df["sell_alert_thesis_broken"] = (
        df.get("roce_trajectory", pd.Series(0, index=df.index)) < -3
    ).astype(int)
    df["sell_alert_mgmt_deteriorated"] = (
        (df.get("pledge_rising", pd.Series(0, index=df.index)) == 1) &
        (df.get("change_promoter_lq", pd.Series(0, index=df.index)) < 0) &
        (df.get("de_slope_3y", pd.Series(0, index=df.index)) > 0)
    ).astype(int)
    df["sell_alert_cash_collapse"] = (
        df.get("cfo_to_pat", pd.Series(100.0, index=df.index)) < 50  # PERCENTAGE: < 50% = poor cash quality (CFO < half of PAT)
    ).astype(int)

    # ── MARKS OVERVALUATION ALERT (Howard Marks — The Dalal Street Thinker) ──
    # Marks: "Even great businesses can be terrible investments at extreme prices."
    # Trigger 1: PEG > 2.5 — Marks' explicit "extreme caution" threshold (verbatim from codex)
    # Trigger 2: P/E > 30% above own 10Y median AND PEG > 2.0 — stock pricing in perfection vs history
    # d32_pe_vs_median: positive = expensive vs own decade history; fillna(0) = no history → neutral
    # peg fillna(999): missing PEG (no earnings) → treated as extreme (999 > 2.5 = alert fires)
    # This fills the gap: the other 3 alerts detect fundamental deterioration; this detects price excess.
    # fillna(0) for peg: NaN = no earnings / no valid PEG → unknown, not overvalued → don't alert
    # fillna(0) for pe_hist: NaN = no 10Y median history → neutral → don't alert
    _peg_sa  = df.get("peg",             pd.Series(0.0, index=df.index)).fillna(0)
    _pe_hist = df.get("d32_pe_vs_median", pd.Series(0.0, index=df.index)).fillna(0)
    df["sell_alert_overvalued"] = (
        (_peg_sa > 2.5) |
        ((_pe_hist > 30) & (_peg_sa > 2.0))
    ).astype(int)

    # ── EXPECTATIONS TREADMILL ALERT (Mauboussin/Rappaport — Expectations Investing Codex) ──
    # "A company priced for perfection must continuously EXCEED already-perfect expectations
    #  just to maintain its stock price." — Mauboussin
    # DISTINCT from sell_alert_overvalued (Marks: static overvaluation via PEG / PE vs history).
    # This alert is DYNAMIC: stock priced for a 15-20 year Competitive Advantage Period (CAP)
    # that is now visibly decelerating — the treadmill is slipping.
    # Example gap: PE 65× stock (Marks' PEG < 2.5, passes) but revenue growth collapsed from
    # 22% → 10% AND ROCE declining — Marks' alert misses it; Treadmill catches it.
    #
    # Three conditions required (conservative AND logic — all three must hold):
    #   1. pe > 50: book's CAP sensitivity table maps P/E 50× → 15-20 year above-WACC assumption.
    #      fillna(0): loss-making stocks (NaN PE) → 0 → not on the treadmill (correct).
    #   2. Growth deceleration vs own 3Y baseline:
    #      rev_gr_yoy < rev_gr_3y - 5: revenue growing 5+ ppts below its own 3Y CAGR.
    #      eps_gr_yoy < eps_gr_3y - 7: earnings growing 7+ ppts below 3Y CAGR (wider band for
    #      operating leverage noise). OR condition: either revenue or earnings must decelerate.
    #      Gap fillna(0): NaN gap (missing data) → 0 → 0 < -5 = False (won't fire on missing data).
    #   3. d35_roce_trend < 0: ROCE structural slope negative (2Y annualised) — moat not recovering.
    #      fillna(0): NaN → 0 → 0 < 0 = False (conservative: missing trend = neutral).
    #
    # False positive guard: a stock at PE 55× with ROCE expanding does NOT trigger (condition 3
    # saves it). Only the combination of premium + slowing + no margin recovery fires.
    _tm_nan      = pd.Series(np.nan, index=df.index)
    _pe_tm       = df.get("pe",            pd.Series(0.0, index=df.index)).fillna(0)
    _rev_yoy_tm  = df.get("rev_gr_yoy",   _tm_nan)
    _rev_3y_tm   = df.get("rev_gr_3y",    _tm_nan)
    _eps_yoy_tm  = df.get("eps_gr_yoy",   _tm_nan)
    _eps_3y_tm   = df.get("eps_gr_3y",    _tm_nan)
    _roce_dir_tm = df.get("d35_roce_trend", pd.Series(0.0, index=df.index)).fillna(0)
    # Deceleration gaps: negative = current growth below own historical baseline
    _rev_gap_tm  = (_rev_yoy_tm - _rev_3y_tm).fillna(0)   # fillna(0): NaN gap → not decelerating
    _eps_gap_tm  = (_eps_yoy_tm - _eps_3y_tm).fillna(0)
    df["sell_alert_treadmill"] = (
        (_pe_tm        > 50) &                          # Premium: 15-20Y CAP priced in
        ((_rev_gap_tm  < -5) | (_eps_gap_tm < -7)) &   # Slipping: growth falling behind own history
        (_roce_dir_tm  < 0)                             # No rescue: ROCE not expanding
    ).astype(int)

    # ── KHANDELWAL SEQUENTIAL DETERIORATION ALERT (Vishal Khandelwal — The Long Game) ──
    # Chapter 10: "Sell only on fundamental deterioration confirmed across 2+ consecutive years."
    # Core guard: one bad year ≠ thesis broken. A PATTERN of decline is required before selling.
    # This captures structural top-line collapse — revenue AND earnings both declining multi-year.
    # DISTINCT from sell_alert_thesis_broken (ROCE capital efficiency — capital allocation layer)
    # and sell_alert_cash_collapse (CFO quality — operating cash layer).
    # This alert fires when the REVENUE ENGINE itself is structurally shrinking.
    # rev_gr_3y < 0: 3-year CAGR negative → confirms decline extends beyond 1 year (multi-year).
    # pat_gr_yoy < 0: earnings also declining → no margin/mix offset rescuing the top-line fall.
    # All fillna(0): NaN → 0 → condition fails → no false alert on missing data (conservative).
    _rev_yoy_sd  = df.get("rev_gr_yoy", pd.Series(0.0, index=df.index)).fillna(0)
    _rev_3y_sd   = df.get("rev_gr_3y",  pd.Series(0.0, index=df.index)).fillna(0)
    _pat_yoy_sd  = df.get("pat_gr_yoy", pd.Series(0.0, index=df.index)).fillna(0)
    df["sell_alert_sequential_decline"] = (
        (_rev_yoy_sd < 0) &    # Current year revenue contracting
        (_rev_3y_sd  < 0) &    # 3Y CAGR also negative — multi-year revenue decline confirmed
        (_pat_yoy_sd < 0)      # Earnings also declining — no margin offset to revenue fall
    ).astype(int)

    df["sell_alert_any"] = (
        (df["sell_alert_thesis_broken"]       == 1) |
        (df["sell_alert_mgmt_deteriorated"]   == 1) |
        (df["sell_alert_cash_collapse"]       == 1) |
        (df["sell_alert_overvalued"]          == 1) |
        (df["sell_alert_treadmill"]           == 1) |
        (df["sell_alert_sequential_decline"]  == 1)
    ).astype(int)

    # ══════════════════════════════════════════════════════════════
    # MOD 3: Cobb-Douglas Geometric Quality Overlay
    # Geometric blending (weighted log-sum) ensures a single pillar near zero collapses
    # the combined rating — additive blending masks single-pillar failures.
    # All three pillars operate in [0, 100] space; fillna(50) = neutral midpoint for missing.
    # Clip floor at 5 prevents log(0) blowup. Companion column quality_score_geometric —
    # the legacy quality_score is preserved intact so no existing tests or composite weights break.
    # ══════════════════════════════════════════════════════════════
    _p1 = df.get("malik_checklist_score", pd.Series(50.0, index=df.index)).fillna(50).clip(5, 100).values / 100.0
    _p2 = df.get("ibas_moat_score", pd.Series(50.0, index=df.index)).fillna(50).clip(5, 100).values / 100.0
    _p3 = df.get("vqs_score",       pd.Series(50.0, index=df.index)).fillna(50).clip(5, 100).values / 100.0
    df["quality_score_geometric"] = (
        100.0 * np.exp(
            (0.40 * np.log(_p1)) +
            (0.40 * np.log(_p2)) +
            (0.20 * np.log(_p3))
        )
    ).round(2)

    df["quality_score"] = _safe_clip(df["quality_score"])
    mean_rev_count = int(df["mean_reversion_risk"].sum())
    sell_alerts = int(df["sell_alert_any"].sum())
    print(f"\U0001f4ca Quality Score: mean={df['quality_score'].mean():.1f}, "
          f"median={df['quality_score'].median():.1f}, "
          f"top 10%\u2265{df['quality_score'].quantile(0.9):.1f}")
    if mean_rev_count > 0:
        print(f"  \u26a0\ufe0f Mean Reversion Risk: {mean_rev_count} stocks at cyclical peak margins")
    if sell_alerts > 0:
        print(f"  \U0001f6a8 Baid Sell Triggers: {sell_alerts} stocks with active sell alerts")

    return df


# ═══════════════════════════════════════════════════════════════
# LAYER 3: MOMENTUM SCORE
# ═══════════════════════════════════════════════════════════════

def _compute_rs_score(df: pd.DataFrame) -> pd.Series:
    """Relative strength score across 3 timeframes."""
    score = pd.Series(0.0, index=df.index)
    for col, weight in RS_SIGNALS.items():
        if col in df.columns:
            score += _pct_rank(df[col], ascending=True).fillna(50) * weight
    return _safe_clip(score)


def _compute_trend_score(df: pd.DataFrame) -> pd.Series:
    """Trend quality: SMA200 direction, VSTOP, ADX, RSI zone, golden cross.

    above_sma200 was a hard gate (binary eliminate). Now a continuous signal:
    stocks below 200D SMA score 0/20 on this component instead of being eliminated.
    A quality stock in a correction still surfaces — the human decides on timing.
    """
    score = pd.Series(0.0, index=df.index)

    # SMA200 direction — replaces hard gate with a 20-point continuous penalty
    if "above_sma200" in df.columns:
        score += df["above_sma200"].fillna(0) * 100 * TREND_SIGNALS["above_sma200"]

    # VSTOP green (binary)
    if "vstop_green" in df.columns:
        score += df["vstop_green"].fillna(0) * 100 * TREND_SIGNALS["vstop_green"]

    # VSTOP fresh (binary)
    if "vstop_fresh" in df.columns:
        score += df["vstop_fresh"].fillna(0) * 100 * TREND_SIGNALS["vstop_fresh"]

    # ADX strength (> 25 is strong trend)
    if "adx_14w" in df.columns:
        adx_score = np.where(df["adx_14w"] >= 25, 100,
                   np.where(df["adx_14w"] >= 20, 70,
                   np.where(df["adx_14w"] >= 15, 40, 10)))
        score += pd.Series(adx_score, index=df.index).fillna(50) * TREND_SIGNALS["adx_strong"]

    # RSI zone scoring
    if "rsi_14d" in df.columns:
        rsi_score = _zone_score(df["rsi_14d"], RSI_ZONES)
        score += rsi_score.fillna(50) * TREND_SIGNALS["rsi_zone"]

    # Golden cross recency (lower days = better) — trend recovery signal
    if "golden_cross_days" in df.columns:
        gc_rank = _pct_rank(df["golden_cross_days"], ascending=False).fillna(50)
        score += gc_rank * TREND_SIGNALS["golden_cross"]

    return _safe_clip(score)


def _compute_breakout_score(df: pd.DataFrame) -> pd.Series:
    """Breakout proximity: nearness to highs and breakout windows."""
    score = pd.Series(0.0, index=df.index)

    # 52WH distance (lower % = closer to breakout = better)
    if "dist_52wh" in df.columns:
        score += _pct_rank(df["dist_52wh"], ascending=False).fillna(50) * BREAKOUT_SIGNALS["52wh_distance"]

    # 52WH age: how many days since the 52-week high was set (fewer = fresher momentum)
    # Zone-based (not percentile) because absolute thresholds matter regardless of market state:
    # a 250-day old high is stale whether peers are fresh or not.
    if "dist_52wh_days" in df.columns:
        age_score = _zone_score(df["dist_52wh_days"].fillna(9999), HIGH_AGE_ZONES)
        score += age_score.fillna(50) * BREAKOUT_SIGNALS["52wh_days"]

    # 13WH distance
    if "dist_13wh" in df.columns:
        score += _pct_rank(df["dist_13wh"], ascending=False).fillna(50) * BREAKOUT_SIGNALS["13wh_distance"]

    # Breakout window (binary)
    if "breakout_window" in df.columns:
        bw = df["breakout_window"].notna() & (df["breakout_window"] > 0)
        score += bw.astype(float) * 100 * BREAKOUT_SIGNALS["breakout_window"]

    # ATH distance (lower = better)
    if "dist_ath" in df.columns:
        score += _pct_rank(df["dist_ath"], ascending=False).fillna(50) * BREAKOUT_SIGNALS["ath_distance"]

    return _safe_clip(score)


def _compute_volume_score(df: pd.DataFrame) -> pd.Series:
    """Volume confirmation: institutional entry detection."""
    score = pd.Series(50.0, index=df.index)

    if "vol_ratio" in df.columns:
        # Vol ratio > 2 = institutional surge. NaN vol_ratio → neutral 50 (no data ≠ low volume).
        vol_score = np.where(df["vol_ratio"].isna(), 50,
                   np.where(df["vol_ratio"] >= 2.0, 100,
                   np.where(df["vol_ratio"] >= 1.5, 80,
                   np.where(df["vol_ratio"] >= 1.0, 60,
                   np.where(df["vol_ratio"] >= 0.7, 40, 20)))))
        score = pd.Series(vol_score, index=df.index, dtype=float)

    return _safe_clip(score)


def _compute_sector_leader_score(df: pd.DataFrame) -> pd.Series:
    """Sector leadership: outperformance vs industry peers.
    Ranked within sector so the score captures who outperformed their peers most,
    not which sector had the best absolute returns."""
    score = pd.Series(0.0, index=df.index)

    for col, weight in SECTOR_SIGNALS.items():
        if col in df.columns:
            # Sector-relative ranking: ensures a top IT stock isn't penalized in a tech downturn
            # and a mediocre bank isn't rewarded when banking sector outperformed.
            score += _sector_pct_rank(df, col, ascending=True) * weight

    return _safe_clip(score)


def compute_momentum_score(df: pd.DataFrame) -> pd.DataFrame:
    """Compute composite momentum score (Layer 3)."""
    df = df.copy()

    df["rs_score"] = _compute_rs_score(df)
    df["trend_score"] = _compute_trend_score(df)
    df["breakout_score"] = _compute_breakout_score(df)
    df["volume_score"] = _compute_volume_score(df)
    df["sector_leader_score"] = _compute_sector_leader_score(df)

    df["momentum_score"] = (
        df["rs_score"] * MOMENTUM_WEIGHTS["relative_strength"] +
        df["trend_score"] * MOMENTUM_WEIGHTS["trend_quality"] +
        df["breakout_score"] * MOMENTUM_WEIGHTS["breakout_proximity"] +
        df["volume_score"] * MOMENTUM_WEIGHTS["volume_confirm"] +
        df["sector_leader_score"] * MOMENTUM_WEIGHTS["sector_leadership"]
    )

    df["momentum_score"] = _safe_clip(df["momentum_score"])
    print(f"🚀 Momentum Score: mean={df['momentum_score'].mean():.1f}, "
          f"median={df['momentum_score'].median():.1f}, "
          f"top 10%≥{df['momentum_score'].quantile(0.9):.1f}")

    return df


# ═══════════════════════════════════════════════════════════════
# GOVERNANCE BONUS
# ═══════════════════════════════════════════════════════════════

def compute_governance_bonus(df: pd.DataFrame) -> pd.DataFrame:
    """Asymmetric governance: positive ownership signals → additive bonus (engine);
    the four hard risk signals → governance_risk_multiplier on composite (shield).
    Negative signals predict disasters far better than positive signals predict winners,
    so risk acts as a multiplier (scales with conviction) instead of flat point deductions.
    Outputs: governance_bonus, gov_risk_count, governance_risk_multiplier."""
    df = df.copy()
    bonus = pd.Series(0.0, index=df.index)

    # Promoter buying this quarter
    if "promoter_buying" in df.columns:
        bonus += df["promoter_buying"].fillna(0) * GOVERNANCE_BONUS["promoter_buying"]

    # FII accumulating
    if "change_fii_lq" in df.columns:
        bonus += (df["change_fii_lq"] > 0).astype(float) * GOVERNANCE_BONUS["fii_accumulating"]

    # DII accumulating
    if "change_dii_lq" in df.columns:
        bonus += (df["change_dii_lq"] > 0).astype(float) * GOVERNANCE_BONUS["dii_accumulating"]

    # Institutional convergence
    if "inst_convergence" in df.columns:
        bonus += df["inst_convergence"].fillna(0) * GOVERNANCE_BONUS["inst_convergence"]

    # Insider trading — reward buying directors only, not sellers
    if "insider_trading" in df.columns:
        _insider_bought = df["insider_trading"].fillna("").astype(str).str.contains("Bought", case=False)
        bonus += _insider_bought.astype(float) * GOVERNANCE_BONUS["insider_trading_present"]

    # Pledge falling over 1 year
    if "pledge_falling_1y" in df.columns:
        bonus += (df["pledge_falling_1y"] > 0).astype(float) * GOVERNANCE_BONUS["pledge_falling_1y"]

    # Promoter holding alignment (Mayer 100-Bagger: 10/10 Indian 100-baggers had ≥40%+ promoter)
    # Rewards baseline alignment LEVEL — distinct from promoter_buying which rewards quarterly activity.
    # promoter_holdings = numeric percentage (e.g. 55.3 = 55.3%)
    promo_pct = df.get("promoter_holdings",   pd.Series(np.nan, index=df.index)).fillna(0)
    promo_1y  = df.get("change_promoter_1y",  pd.Series(0.0, index=df.index)).fillna(0)
    promo_2y  = df.get("change_promoter_2y",  pd.Series(0.0, index=df.index)).fillna(0)
    promo_3y  = df.get("change_promoter_3y",  pd.Series(0.0, index=df.index)).fillna(0)
    if "promoter_holdings" in df.columns:
        bonus += (promo_pct >= 60).astype(float) * GOVERNANCE_BONUS["promoter_high_alignment"]
        bonus += ((promo_pct >= 50) & (promo_pct < 60)).astype(float) * GOVERNANCE_BONUS["promoter_good_alignment"]
        # 3-year trend signals — single-quarter buys/sells are noise; 3Y patterns are decisions.
        # Accumulation: net buying >3% over 3 years = sustained conviction, dynasty building.
        bonus += (promo_3y > 3).astype(float) * GOVERNANCE_BONUS["promoter_3y_accumulation"]

    # Undiscovered alpha: low FII + Tier C
    if "fii_holdings" in df.columns and "market_cap" in df.columns:
        undiscovered = (df["fii_holdings"] < 5) & (df["market_cap"] < 5000)
        bonus += undiscovered.astype(float) * GOVERNANCE_BONUS["undiscovered_alpha"]

    # Dilution: Tier 3 (>10%) is hard-gated and never reaches here.
    # Tier 1 (<3% ESOP) = -5 additive (routine noise). Tier 2 (3-10%) is a HARD RISK
    # SIGNAL — handled below via the governance risk multiplier, not additive points.
    dilution = df.get("dilution_flag", pd.Series(0, index=df.index)).fillna(0)
    if "dilution_flag" in df.columns:
        bonus += (dilution == 1).astype(float) * GOVERNANCE_BONUS["dilution_tier1_minor"]

    # G3 FIX heritage: penalties must never be silently erased. The additive bonus keeps the
    # clip(lower=-50) floor for the tier-1 deduction; the four HARD risk signals now act
    # through the composite multiplier below — stronger than flat points and impossible to clip away.
    df["governance_bonus"] = bonus.clip(lower=-50, upper=100)

    # ── Asymmetric Governance Risk Shield ──
    # Four hard ownership risk signals → composite multiplier (GOVERNANCE_RISK_MULTIPLIERS).
    # Negative signals predict disasters far better than positive signals predict winners,
    # so risk scales the WHOLE conviction instead of deducting flat points: a 90-composite
    # stock loses more absolute points than a 20-composite stock — the risk threatens more.
    _r_dilution_t2   = (dilution == 2)                              # 3-10% share dilution
    _r_3y_exit       = (promo_3y < -5)                              # systematic 3Y exit
    _r_2y_exit       = (promo_2y < -3) & (promo_3y >= -5)           # early warning (excl. of 3Y)
    _r_low_declining = (promo_pct < 40) & (promo_1y < 0)            # low base AND falling
    df["gov_risk_count"] = (
        _r_dilution_t2.astype(int) + _r_3y_exit.astype(int) +
        _r_2y_exit.astype(int) + _r_low_declining.astype(int)
    )
    _max_tier = max(GOVERNANCE_RISK_MULTIPLIERS)
    df["governance_risk_multiplier"] = (
        df["gov_risk_count"].clip(upper=_max_tier).map(GOVERNANCE_RISK_MULTIPLIERS)
    )
    return df


# ═══════════════════════════════════════════════════════════════
# LAYER 4: COMPOSITE + CONVICTION TIER
# ═══════════════════════════════════════════════════════════════

def compute_composite_score(
    df: pd.DataFrame,
    fundamental_w: float = 0.70,
    momentum_w: float = 0.30
) -> pd.DataFrame:
    """Final composite score and conviction tier assignment.
    
    Args:
        fundamental_w: Weight for quality/fundamental score (Analysis Mode)
        momentum_w: Weight for momentum score (Analysis Mode)
    """
    df = df.copy()

    # ══════════════════════════════════════════════════════════════
    # MOD 1: Cross-Sectional OLS Valuation Residual
    # Regress ln(P/B) against fundamentals (ROE, Rev growth, ROCE, ln(MCap)) to isolate
    # the component of P/B NOT explained by fundamentals — the signed pricing error.
    # Negative residual = cheapness given fundamentals (underpriced); positive = premium.
    # Uses np.linalg.lstsq — pure NumPy, no row iteration. Computed once on the full
    # cross-section before blending so every composite score can reflect this signal.
    # ══════════════════════════════════════════════════════════════
    _n_stocks = len(df)
    if _n_stocks > 10:
        _target_y = np.nan_to_num(
            np.log(
                df["pb_ratio"].fillna(df["pb_ratio"].replace(0, np.nan).median()).clip(lower=0.01)
            ).values,
            nan=0.0
        )
        _A_matrix = np.nan_to_num(
            np.column_stack([
                df["roe"].fillna(df["roe"].median()).values,
                df["rev_gr_5y"].fillna(df["rev_gr_5y"].median()).values,
                df["roce_med_5y"].fillna(df["roce_med_5y"].median()).values,
                np.log(df["market_cap"].fillna(df["market_cap"].median()).clip(lower=1.0)).values,
                np.ones(_n_stocks)
            ]),
            nan=0.0
        )
        try:
            _beta_coeffs, _, _, _ = np.linalg.lstsq(_A_matrix, _target_y, rcond=None)
            df["valuation_residual"] = _target_y - (_A_matrix @ _beta_coeffs)
        except np.linalg.LinAlgError:
            df["valuation_residual"] = 0.0
    else:
        df["valuation_residual"] = 0.0

    # MOD 2: Outlier Normalization Plane
    # Convert fat-tailed raw residual + expectations_gap to bounded [0,100] cross-sectional
    # percentile coordinates — insulates downstream scoring from division blowups.
    # expectations_gap comes from data_engine Pass 1; use .get() so tests with synthetic
    # frames (no data_engine columns) don't KeyError — missing → uniform NaN rank.
    df["valuation_residual_rank"] = df["valuation_residual"].rank(pct=True) * 100.0
    _eg = df.get("expectations_gap", pd.Series(np.nan, index=df.index))
    _eg_med = _eg.median()
    df["expectations_gap_rank"] = (
        _eg.fillna(_eg_med if pd.notna(_eg_med) else 0.0)
        .rank(pct=True) * 100.0
    )

    # Governance weight is fixed regardless of analysis mode — set via COMPOSITE_WEIGHTS["governance"] (currently 15%)
    gov_w = COMPOSITE_WEIGHTS.get("governance", 0.15)
    # Normalize fundamental + momentum to fill remaining 90%
    scale = 1.0 - gov_w
    fund_scaled = fundamental_w * scale
    mom_scaled  = momentum_w  * scale

    df["composite_score"] = (
        df["quality_score"]   * fund_scaled +
        df["momentum_score"]  * mom_scaled +
        df["governance_bonus"] * gov_w
    )

    # ── SQGLP 100x Engine Integration (Epoch 4 — 19th & 24th WCS) ──
    # Applied FIRST: most stringent multi-condition signal (+15 pts) earns priority before
    # smaller boosts compete to fill the 100-point cap.
    # Raamdeo Agrawal's 5-pillar SQGLP framework: Size + Quality + Growth + Longevity + Price.
    _mcat   = df.get("market_category", df.get("mcap_tier", pd.Series("", index=df.index)))
    _roce10y = df.get("roce_med_10y", pd.Series(0.0, index=df.index)).fillna(0.0)
    _peg_v   = df.get("peg",          pd.Series(999.0, index=df.index))
    _cfopat  = df.get("cfo_to_pat",   pd.Series(0.0, index=df.index)).fillna(0.0)
    df["flag_sqglp_engine"] = (
        _mcat.isin(["Mid Cap", "Small Cap", "Micro Cap"]) &                                    # S: Scale entry
        (_roce10y >= 25.0) &                                                                    # Q: Capital quality
        (_cfopat >= EPOCH4_SQGLP["min_cfo_to_pat_ratio"]) &                                   # Q: ECV >= 80% (percentage unit — matches cfo_to_pat CSV column, e.g. 73.04)
        (_peg_v.fillna(0.0) > 0) & (_peg_v.fillna(999.0) <= EPOCH4_SQGLP["max_peg_ratio"]) & # P: PEG <= 1.5 (config)
        (df.get("red_flag_count", pd.Series(99, index=df.index)).fillna(99) <= 2)              # Integrity gate: Watch or better (≤2 red flags)
    ).astype(int)

    df["composite_score"] = np.where(df["flag_sqglp_engine"] == 1, df["composite_score"] + 15.0, df["composite_score"])

    # EP Hockey-Stick Breakout boost (28th WCS): structural alpha inflection — emerging
    # value generator (Q2/Q3) ascending the Power Curve with institutional volume confirmation.
    _hs_boost = df.get("ep_hockey_stick_breakout", pd.Series(0, index=df.index)).fillna(0)
    df["composite_score"] = df["composite_score"] + _hs_boost * 5.0

    # Agent 8 (1st WCS): Mid-Cap Velocity Compounder boost.
    # Mid/Small/Micro-Cap stocks with sustained ROCE ≥ 20% compound dramatically faster than large caps.
    _mcv_boost = df.get("mcap_velocity_compounder", pd.Series(0, index=df.index)).fillna(0)
    df["composite_score"] = df["composite_score"] + _mcv_boost * 3.0

    df["composite_score"] = _safe_clip(df["composite_score"])

    # ── Value Migration Boost (20th WCS — Structural Sector Value Rotation) ──
    # Companies capturing structural market share from weaker sector peers: +4 composite pts.
    # Distinct from mcap_velocity_compounder (+3 pts): value_migration specifically rewards
    # top-quartile sector revenue captors with expanding ROCE.
    _vm_boost = df.get("value_migration_flag", pd.Series(0, index=df.index)).fillna(0)
    df["composite_score"] = _safe_clip(df["composite_score"] + _vm_boost * 4.0)

    # ── Bruised Blue Chip 29th WCS Entry Matrix (29th WCS) ──
    # Large-cap elite franchise (ROCE_10Y ≥ 20%) at deep valuation floor (P/B ≤ 2.0x).
    # Thresholds from EPOCH5_MODERN. +12 pts = asymmetric payoff at franchise valuation trough.
    _bbc29_boost = df.get("bruised_blue_chip_29", pd.Series(0, index=df.index)).fillna(0)
    df["composite_score"] = _safe_clip(df["composite_score"] + _bbc29_boost * 12.0)

    # ── Macro Tipping Point Velocity Boost (30th WCS — India Multi-Trillion Engine) ──
    # Financials/Consumer stocks hitting top-15% of tipping point velocity signal.
    # Uses positive-only quantile so non-tipping-sector zeros don't dilute the threshold.
    if "tipping_point_velocity" in df.columns:
        _ttp_vals = df["tipping_point_velocity"].fillna(0)
        _ttp_pos  = _ttp_vals[_ttp_vals > 0]
        if len(_ttp_pos) >= 10:
            _ttp_p85 = _ttp_pos.quantile(0.85)
            df["composite_score"] = _safe_clip(
                df["composite_score"] + (_ttp_vals > _ttp_p85).astype(float) * 10.0
            )

    # ── Consistency-in-Volatile Sector Premium (27th WCS) ──
    # Consistent PAT compounder inside a structurally volatile sector = highest-alpha combination.
    # 19% avg CAGR (31 companies) vs 16% for Consistents in Consistent sectors (27th WCS data).
    _civ_boost = df.get("consistent_in_volatile_flag", pd.Series(0, index=df.index)).fillna(0)
    df["composite_score"] = _safe_clip(df["composite_score"] + _civ_boost * 5.0)

    # ── Asymmetric Governance Risk Shield (applied LAST, before tier assignment) ──
    # Hard ownership risk signals (tier-2 dilution, promoter 3Y/2Y exit, low+declining
    # holdings) multiply the composite down. Applied after all boosts so the penalty
    # scales the WHOLE conviction (a boosted stock with promoter exodus is still risky).
    # .get() fallback: synthetic test frames without the column are not penalized.
    _gov_mult = df.get("governance_risk_multiplier", pd.Series(1.0, index=df.index)).fillna(1.0)
    df["composite_score"] = _safe_clip(df["composite_score"] * _gov_mult)

    # Assign conviction tiers
    conditions = []
    choices = []
    for tier in CONVICTION_TIERS:
        conditions.append(df["composite_score"] >= tier["min"])
        choices.append(tier["tier"])

    df["conviction_tier"] = np.select(conditions, choices, default=5)
    df["tier_label"] = df["conviction_tier"].map(
        {t["tier"]: f"{t['emoji']} {t['label']}" for t in CONVICTION_TIERS}
    )
    df["tier_emoji"] = df["conviction_tier"].map(
        {t["tier"]: t["emoji"] for t in CONVICTION_TIERS}
    )

    # ══════════════════════════════════════════════════════════════
    # MOD 5: Expected Returns & Vectorized Kelly-Minervini Sizing
    # All data_engine-computed columns accessed via .get() so tests with synthetic frames
    # that lack those columns don't KeyError — missing inputs propagate NaN (semantic truth).
    # ══════════════════════════════════════════════════════════════
    _pe_m5       = df.get("pe",               pd.Series(np.nan, index=df.index))
    _fair_pe_m5  = df.get("fair_pe_qglp",     pd.Series(np.nan, index=df.index))
    _g_star_m5   = df.get("g_star",           pd.Series(np.nan, index=df.index))
    _fcfy_m5     = df.get("fcf_yield",        pd.Series(np.nan, index=df.index))
    _sigma_g_m5  = df.get("sigma_g",          pd.Series(np.nan, index=df.index))
    _traj_m5     = df.get("trajectory_score", pd.Series(np.nan, index=df.index))
    _close_m5    = df.get("close_price",      pd.Series(np.nan, index=df.index))
    _vstop_m5    = df.get("vstop_value",      pd.Series(np.nan, index=df.index))

    # fillna(0.0): loss-makers have no PE → no re-rating estimate → neutral 0 term.
    # Without it, NaN poisons the whole identity even when g* and FCF yield are known.
    _re_rating_drift = (
        np.log(
            _fair_pe_m5.fillna(_pe_m5).clip(lower=1)
            / _pe_m5.clip(lower=1)
        ) / 5.0
    ).fillna(0.0)
    # σ²/2 Ito variance drag: converts % σ to decimal, applies geometric-vs-arithmetic correction.
    # E[log CAGR] = μ − σ²/2. At σ=20%: drag=2%; at σ=40%: drag=8%. Mathematically correct.
    _variance_drag = ((_sigma_g_m5.fillna(0.0) / 100.0) ** 2) / 2.0 * 100.0
    df["expected_cagr_engine"] = (
        _g_star_m5.fillna(0)
        + _fcfy_m5.fillna(0)
        + (_re_rating_drift * 100.0)
        - _variance_drag
    )

    _total_equity_rupees = 1_000_000.0
    _traj_norm = (_traj_m5.fillna(0) + 1.0) / 2.0
    df["win_rate_proxy"] = (0.35 + (_traj_norm * 0.30)).clip(0.35, 0.65)
    # Inner fillna(1.5): a loss-maker (NaN PE) with a negative residual would otherwise
    # produce NaN payoff → NaN Kelly → NaN weight. Neutral 1.5 matches the else branch.
    df["payoff_ratio_proxy"] = np.where(
        df["valuation_residual"] < 0,
        (_fair_pe_m5.fillna(1.0) / _pe_m5.clip(lower=1.0)).clip(1.0, 4.0).fillna(1.5),
        1.5
    )
    _raw_kelly = df["win_rate_proxy"] - ((1.0 - df["win_rate_proxy"]) / df["payoff_ratio_proxy"])
    _kelly_f_weight = (_raw_kelly * 0.25).clip(lower=0.0)

    # Minervini 1%-equity-risk cap — three regimes (vectorized, no clip(lower=1) hack):
    #   price > stop : risk = close − vstop → max shares = 1% equity / risk → weight cap
    #   price ≤ stop : trend broken, entry would be instantly stopped out → weight 0
    #                  (the old clip(lower=1) bug shrank risk to ₹1, exploded the share
    #                   cap, and silently handed breakdown stocks full Kelly weight)
    #   stop is NaN  : no stop computable (missing data or implausible VSTOP nullified
    #                  by the data_engine scale guard) → technical cap undefined →
    #                  Kelly-only weight; NaN must never reach the weight/allocation.
    _per_share_rupee_risk = _close_m5 - _vstop_m5
    _minervini_max_weight_pct = pd.Series(
        np.where(
            _per_share_rupee_risk > 0,
            ((_total_equity_rupees * 0.01) / _per_share_rupee_risk.clip(lower=0.01))
            * _close_m5 / _total_equity_rupees * 100.0,
            0.0                                  # price at/below stop → no position
        ),
        index=df.index
    )
    _minervini_max_weight_pct = _minervini_max_weight_pct.where(
        _vstop_m5.notna() & _close_m5.notna(),
        _kelly_f_weight * 100.0                  # stop unavailable → Kelly-only
    )

    df["optimal_portfolio_weight_pct"] = np.minimum(
        _kelly_f_weight * 100.0, _minervini_max_weight_pct
    ).clip(upper=20.0)
    df["rupee_capital_allocation"] = (
        _total_equity_rupees * (df["optimal_portfolio_weight_pct"] / 100.0)
    )

    # ── Mauboussin Ch.13 Payoff Framework: per-stock Expected Excess Return ──
    # EV = P(upside) × Upside% − P(downside) × Downside%   (book's exact identity)
    #   P(upside)  = win_rate_proxy (trajectory-tau calibrated, 0.35–0.65)
    #   Upside%    = re-rating gap to quality-justified fair PE (fair_pe_qglp / pe − 1),
    #                clipped [0%, 100%]; no PE (loss-maker) → no re-rating estimate → 0
    #   Downside%  = distance to the volatility stop (the defined-risk exit), clipped
    #                [5%, 50%]; stop missing or already breached → neutral 20% assumption
    # Book threshold: EV < 5% = insufficient compensation → no position (UI renders verdict).
    _ev_up_m5 = (
        ((_fair_pe_m5 / _pe_m5.clip(lower=1.0)) - 1.0).clip(0.0, 1.0) * 100.0
    ).fillna(0.0)
    _ev_dn_raw = pd.Series(
        np.where(
            _close_m5.fillna(0) > 0,
            (_close_m5 - _vstop_m5) / _close_m5 * 100.0,
            np.nan
        ),
        index=df.index
    ).clip(5.0, 50.0)
    _ev_dn_m5 = _ev_dn_raw.where(
        _vstop_m5.notna() & _close_m5.notna() & (_close_m5 > _vstop_m5),
        20.0
    )
    df["mauboussin_ev_upside_pct"]   = _ev_up_m5
    df["mauboussin_ev_downside_pct"] = _ev_dn_m5
    df["expected_excess_return"] = (
        df["win_rate_proxy"] * _ev_up_m5 - (1.0 - df["win_rate_proxy"]) * _ev_dn_m5
    )
    # Book's Ch.13 sizing table, materialized here so the UI stays pure display:
    # >15% high (8-12%) · 10-15% mod-high (5-8%) · 5-10% moderate (3-5%) · <5% none
    _eer = df["expected_excess_return"]
    df["mauboussin_ev_verdict"] = np.select(
        [_eer >= 15.0, _eer >= 10.0, _eer >= 5.0],
        ["High Conviction · 8–12% position",
         "Moderate-High · 5–8% position",
         "Moderate · 3–5% position"],
        default="Insufficient Edge · No position (< 5% min)"
    )

    # ── Score Confidence (display-only): share of ranked inputs actually present ──
    # reindex() returns exactly the CORE_SCORING_INPUTS columns, with an all-NaN
    # column for any input absent from the frame — absent counts as missing, by design.
    # Label is materialized HERE so the UI stays pure display (EV-verdict precedent).
    _evidence = df.reindex(columns=CORE_SCORING_INPUTS).notna()
    df["data_coverage_pct"] = _evidence.mean(axis=1) * 100.0
    df["data_coverage_label"] = (
        _evidence.sum(axis=1).astype(int).astype(str)
        + f"/{len(CORE_SCORING_INPUTS)} inputs"
    )

    print(f"\n🏆 Composite Score: mean={df['composite_score'].mean():.1f}, "
          f"median={df['composite_score'].median():.1f}")
    print("\nConviction Tier Distribution:")
    for tier in CONVICTION_TIERS:
        count = (df["conviction_tier"] == tier["tier"]).sum()
        print(f"  {tier['emoji']} Tier {tier['tier']} ({tier['label']}): {count} stocks")

    return df



# ═══════════════════════════════════════════════════════════════
# TSUNAMI SIGNAL & CATALYST MATRIX DETECTION
# ═══════════════════════════════════════════════════════════════

def detect_catalysts_and_tsunami(df: pd.DataFrame) -> pd.DataFrame:
    """Detect the highest-conviction setups and explicit catalyst triggers."""
    df = df.copy()

    # ── 1. Tsunami Signal ──
    # SMA200 is no longer a hard gate, but Tsunami is the rarest/highest-conviction
    # signal and STILL requires full technical confirmation including above_sma200.
    # (A stock in a correction with great fundamentals can score well — but to be
    # called a TSUNAMI setup, everything must align: gates + quality + technicals.)
    # Quality bar 65 (recalibrated 2026-06-12): the original 70 was set before the
    # GRUESOME ×0.50 haircut and trap penalties compressed quality scores (live
    # median ~31). On real data the 7 technical/governance conditions leave ~12
    # candidates and quality ≥ 70 killed ALL of them — the signal was DEAD (0 of
    # 2107, permanently). At 65 (still ~top 15% of post-penalty quality) Tsunami
    # fires for ~2 stocks: rarest-signal design intent, alive.
    tsunami_conditions = (
        (df["gate_pass"] == 1) &
        (df.get("above_sma200", pd.Series(0, index=df.index)) == 1) &
        (df["vstop_green"] == 1) &
        (df["vstop_fresh"] == 1) &
        (df["promoter_buying"] == 1) &
        (df.get("change_fii_lq", pd.Series(0, index=df.index)) > 0) &
        (df["quality_score"] >= 65) &
        (df.get("crs_aligned", pd.Series(0, index=df.index)) == 1)
    )

    df["tsunami_signal"] = tsunami_conditions.astype(int)

    # Tsunami with Tier C (undiscovered) is the ultimate signal
    df["tsunami_undiscovered"] = (
        tsunami_conditions & (df.get("market_cap", pd.Series(0, index=df.index)) < 5000)
    ).astype(int)

    # ── 2. Catalyst Matrix (The 'God Screen' Upgrade) ──

    # CAPACITY EXPLOSION: CWIP going live + FA growing >15% CAGR (D19 > 0 AND D20 > 15%)
    df["cat_capacity"] = (
        (df.get("d19_cwip_conversion", pd.Series(0, index=df.index)) > 0) &
        (df.get("d20_fa_cagr_3y", pd.Series(0, index=df.index)) > 15)
    ).astype(int)

    # OPERATING LEVERAGE INFLECTION: Revenue outpacing costs AND earnings accelerating
    # Handbook spec: D05 > 10 AND D06 > 25% (exact GOD Screen catalyst formula)
    df["cat_oplev"] = (
        (df.get("d05_rev_minus_exp_gr", pd.Series(0, index=df.index)) > 10) &
        (df.get("q_pat_yoy", pd.Series(0, index=df.index)) > 25)
    ).astype(int)

    # INSTITUTIONAL DISCOVERY: Smart money finding undercovered gem
    # D38 > 0.5 AND D46 > 2.0 (handbook spec) + FII room to grow
    df["cat_inst_discovery"] = (
        (df.get("d38_smart_money", pd.Series(0, index=df.index)) > 0.5) &
        (df.get("vol_ratio", pd.Series(0, index=df.index)) >= 2.0) &
        (df.get("fii_holdings", pd.Series(100, index=df.index)) < 15)
    ).astype(int)

    # DEBT DELEVERAGING CYCLE: Meaningful debt reduction underway
    df["cat_deleveraging"] = (
        (df.get("debt_slope_3y", pd.Series(0, index=df.index)) < 0) &
        (df.get("debt_to_equity_1yb", pd.Series(0, index=df.index)) > 0.5) &
        (df.get("debt_to_equity", pd.Series(1, index=df.index)) <= 0.5)
    ).astype(int)

    # LYNCH DREAM: PEG < 1 + operating leverage + earnings acceleration + ROCE improving
    # Handbook: PEG<1 AND D05>10 AND D06>30% AND D35>0
    df["cat_lynch_dream"] = (
        (df.get("peg", pd.Series(999, index=df.index)).fillna(999) > 0) &
        (df.get("peg", pd.Series(999, index=df.index)).fillna(999) < 1.0) &
        (df.get("d05_rev_minus_exp_gr", pd.Series(0, index=df.index)) > 10) &
        (df.get("q_pat_yoy", pd.Series(0, index=df.index)) > 30) &
        (df.get("d35_roce_trend", pd.Series(0, index=df.index)) > 0)
    ).astype(int)

    # Count total active catalysts (now 5 types)
    df["catalyst_count"] = (
        df["cat_capacity"] + df["cat_oplev"] + df["cat_inst_discovery"] +
        df["cat_deleveraging"] + df["cat_lynch_dream"]
    )

    count = df["tsunami_signal"].sum()
    undiscovered = df["tsunami_undiscovered"].sum()
    cat_count = (df["catalyst_count"] > 0).sum()
    print(f"\n🌊 Tsunami Signals: {count} stocks ({undiscovered} undiscovered Tier C)")
    print(f"🔥 Active Catalysts: {cat_count} stocks have at least 1 catalyst.")

    return df


# ═══════════════════════════════════════════════════════════════
# 8-FRAMEWORK GURU CLASSIFICATION (God Screen)
# ═══════════════════════════════════════════════════════════════

def compute_qglp_score(df: pd.DataFrame, profile: dict = None) -> pd.DataFrame:
    """Motilal Oswal QGLP Framework — weights driven by selected Scoring Profile."""
    df = df.copy()
    if profile is None:
        profile = MASTER_PROFILES["Balanced"]

    # Q: Quality (ROCE rank + management quality)
    q_score = _pct_rank(df.get("roce", pd.Series(0, index=df.index)), ascending=True).fillna(50) * 0.7
    if "promoter_buying" in df.columns:
        q_score += df.get("promoter_buying", pd.Series(0, index=df.index)) * 10
    if "pledge_rising" in df.columns:
        q_score -= df.get("pledge_rising", pd.Series(0, index=df.index)) * 10
    q_score = _safe_clip(q_score)

    # G: Growth (PAT + EPS CAGR)
    g_score = _pct_rank(df.get("pat_gr_5y", pd.Series(0, index=df.index)), ascending=True).fillna(50) * 0.5 + \
              _pct_rank(df.get("eps_gr_5y", pd.Series(0, index=df.index)), ascending=True).fillna(50) * 0.5
    g_score = _safe_clip(g_score)

    # L: Longevity (ROE Consistency 10Y)
    l_score = _pct_rank(df.get("roe_med_10y", pd.Series(0, index=df.index)), ascending=True).fillna(50)

    # P: Price (PEG zone score)
    if "peg" in df.columns:
        raw_peg_q = df["peg"].fillna(999)
        p_score = _zone_score(raw_peg_q.clip(lower=0, upper=998), PEG_ZONES).fillna(50)
        p_score = pd.Series(np.where(raw_peg_q < 0, 5.0, p_score), index=df.index)  # G2 FIX: negative PEG = max penalty
    else:
        p_score = pd.Series(50.0, index=df.index)

    df["qglp_quality"] = q_score
    df["qglp_growth"] = g_score
    df["qglp_longevity"] = l_score
    df["qglp_price"] = p_score

    # Apply profile-driven QGLP weights
    df["qglp_score"] = _safe_clip(
        q_score * profile["quality_w"] +
        g_score * profile["growth_w"] +
        l_score * profile["longevity_w"] +
        p_score * profile["price_w"]
    )

    # Profile-driven hard gates for QGLP pass
    roce_gate  = profile.get("roce_gate", 15.0)
    growth_gate = profile.get("growth_gate", 15.0)
    peg_gate   = profile.get("peg_gate", 1.5)

    df["qglp_pass"] = (
        (df.get("roce", pd.Series(0, index=df.index)).fillna(0) >= roce_gate) &
        (df.get("pat_gr_5y", pd.Series(0, index=df.index)).fillna(0) >= growth_gate) &
        (df.get("peg", pd.Series(999, index=df.index)).fillna(999) <= peg_gate) &
        (df.get("peg", pd.Series(-1, index=df.index)).fillna(-1) >= 0)
    ).astype(int)

    # ── God Screen: Frame Tagging (fully vectorized — no df.apply) ──
    # 1. QGLP (Raamdeo Agrawal)
    fw_qglp = df.get("qglp_pass", pd.Series(0, index=df.index)).fillna(0) == 1

    # 2. Coffee Can (Saurabh Mukherjea) — Mukherjea's exact Twin Filters + Clean Accounts:
    #    (a) Capital efficiency — Mukherjea Ch.2: "for capital-heavy businesses, use ROCE
    #        above 20% instead of ROE. High ROE via leverage is disqualified."
    #        Non-financial: ROCE 10Y/5Y ≥ 15% — removes leverage from the equation.
    #        Financial:     ROE  10Y/5Y ≥ 15% — banks structurally use leverage; ROE is correct.
    #        The D/E < 1.0 gate below is a secondary guard, but ROE itself is a flawed metric
    #        for non-financials: a manufacturer at D/E=0.9 can post ROE=18% on ROCE=9%.
    #    (b) Revenue growth CONSISTENCY across 3 CAGR windows — Mukherjea requires EVERY
    #        year to show ≥ 10% growth. Best proxy with available data:
    #        10Y CAGR ≥ 10% (sustained), 5Y CAGR ≥ 8% (recent, 8% absorbs one COVID year),
    #        YoY ≥ 0 (not currently contracting).
    #    (c) CFO/EBITDA ≥ 90% — Clean Accounts master signal (Ch.3). Stored as percentage.
    #    (d) D/E < 1.0 for non-financials (Mukherjea: high-leverage ROE is disqualified).
    #    (e) Pledge < 10% (governance pre-filter: Ch.5, ">10% requires investigation").
    _cc_nan = pd.Series(np.nan, index=df.index)
    is_fin_cc     = df.get("is_financial", pd.Series(False, index=df.index)).fillna(False)
    # Capital efficiency: ROCE for non-financials, ROE for financials
    _cc_roce_10y  = df.get("roce_med_10y", _cc_nan)
    _cc_roce_5y   = df.get("roce_med_5y",  _cc_nan)
    _cc_roe_10y   = df.get("roe_med_10y",  _cc_nan).fillna(df.get("roe_med_5y", _cc_nan))
    _cc_roe_rec   = df.get("roe_med_3y",   _cc_nan).fillna(df.get("roe_med_5y", _cc_nan)).fillna(df.get("roe", _cc_nan))
    _cc_cap_eff_nonfin = (
        (_cc_roce_10y.fillna(0) >= 15) &  # ROCE 10Y ≥ 15%: decade-long un-leveraged efficiency
        (_cc_roce_5y.fillna(0)  >= 15)    # ROCE 5Y  ≥ 15%: recent consistency — no collapse
    )
    _cc_cap_eff_fin = (
        (_cc_roe_10y.fillna(0)  >= 15) &  # ROE 10Y ≥ 15%: bank/NBFC capital returns
        (_cc_roe_rec.fillna(0)  >= 15)    # ROE recent ≥ 15%: book specifies 15% (not 12%)
    )
    _cc_efficiency = pd.Series(
        np.where(is_fin_cc, _cc_cap_eff_fin, _cc_cap_eff_nonfin),
        index=df.index
    )
    rev_10y_cc    = df.get("rev_gr_10y",  _cc_nan)
    rev_5y_cc     = df.get("rev_gr_5y",   _cc_nan)
    rev_yoy_cc    = df.get("rev_gr_yoy",  _cc_nan)
    cfo_ebitda_cc = df.get("cfo_to_ebitda", _cc_nan)
    de_cc         = df.get("debt_to_equity", _cc_nan)
    pledge_cc     = df.get("pledged_percentage", _cc_nan)
    # Individual year revenue growth checks (years 2-5 back).
    # Book: "revenue growth of 10% every year for ten consecutive years" (Ch.2, p.48).
    # StockScan provides Revenue 2-5YB → we verify years 2-5 individually; years 6-10 via CAGR.
    # fillna(5): missing year data doesn't penalize — treated as meeting the threshold.
    # Floor relaxed from 10% → 5% to absorb the COVID-19 pandemic anomaly year.
    _cc_y2 = df.get("rev_gr_y2", _cc_nan).fillna(5)
    _cc_y3 = df.get("rev_gr_y3", _cc_nan).fillna(5)
    _cc_y4 = df.get("rev_gr_y4", _cc_nan).fillna(5)
    _cc_y5 = df.get("rev_gr_y5", _cc_nan).fillna(5)
    _cc_each_year = (
        (_cc_y2 >= 5) & (_cc_y3 >= 5) & (_cc_y4 >= 5) & (_cc_y5 >= 5)
    )
    fw_coffee_can = (
        _cc_efficiency                            &  # ROCE (non-fin) / ROE (fin) hurdle
        (rev_10y_cc.fillna(0)    >= 10)           &  # 10Y revenue CAGR ≥ 10%
        (rev_5y_cc.fillna(0)     >= 8)            &  # 5Y revenue CAGR ≥ 8%
        (rev_yoy_cc.fillna(-1)   >= 0)            &  # not currently contracting (year 1)
        _cc_each_year                             &  # years 2-5: each ≥ 5% (pandemic-adjusted floor)
        (cfo_ebitda_cc.fillna(0) >= 90)           &  # CFO/EBITDA ≥ 90% (Clean Accounts)
        (is_fin_cc | (de_cc.fillna(999) < 1.0))   &  # D/E < 1 for non-financials
        (pledge_cc.fillna(0)     < 10)               # pledge < 10% governance gate
    )
    df["coffee_can_pass"] = fw_coffee_can.astype(int)  # overwrites data_engine simple version with full-logic

    # 3. Magic Formula (Joel Greenblatt) — high Earnings Yield + high ROCE
    ey_mf   = df.get("earnings_yield", pd.Series(np.nan, index=df.index)).fillna(0)
    roce_mf = df.get("roce", pd.Series(np.nan, index=df.index)).fillna(0)
    fw_magic_formula = (ey_mf >= 8) & (roce_mf >= 20)

    # 4. SMILE (Maheshwari) — Small/mid cap + high growth + ROCE
    mcap_sm  = df.get("market_cap", pd.Series(np.nan, index=df.index)).fillna(0)
    pat_gr_sm = df.get("pat_gr_5y", pd.Series(np.nan, index=df.index)).fillna(0)
    fw_smile = (mcap_sm < 15000) & (pat_gr_sm >= 20) & (roce_mf >= 20)

    # 5. Lynch Fast Grower v1.1 — One Up on Wall Street (Ch7+Ch10+Ch13+Ch15) + India Calibration
    # v1.1 book-grounded improvements over v1.0:
    #   EPS per share (Ch15) replaces PAT total; FCF cash gate (Ch13 p196) added to Pillar V
    #   FII+DII combined < 20% (Ch9/Ch15: "institutional ownership") replaces FII < 10%
    #   Promoter buying OR level (Ch15 p208+213: "insider buying positive sign") in Pillar F
    #   Inventory surge disqualifier (Ch13 p197: "inventories grow faster than sales = red flag")
    # fii/dii fillna(50): if data missing, assume already discovered → conservative gate failure
    # Spec: docs/lynch_growth_specs.json v1.1-india-calibrated-fastgrower
    _ly_nan      = pd.Series(np.nan, index=df.index)
    peg_ly       = df.get("peg",                pd.Series(999.0, index=df.index)).fillna(999)
    rev_ly       = df.get("rev_gr_5y",          _ly_nan)
    # V1.1: EPS per share — 5Y preferred, 3Y fallback (Lynch Ch15: "earnings growth rate")
    eps5y_ly     = df.get("eps_gr_5y",          _ly_nan)
    eps3y_ly     = df.get("eps_gr_3y",          _ly_nan)
    eps_ly       = eps5y_ly.fillna(eps3y_ly)    # 5Y preferred; 3Y fallback
    # V1.1: FCF verification — Lynch Ch13 p196: "make sure it's free cash flow"
    fcf_ly       = df.get("free_cash_flow",     _ly_nan)
    fcfy_ly      = df.get("fcf_yield",          _ly_nan)
    debt_ly      = df.get("debt_to_equity",     _ly_nan)
    fii_ly       = df.get("fii_holdings",       pd.Series(50.0, index=df.index)).fillna(50)
    # V1.1: DII added — Lynch Ch9/Ch15: "institutional ownership" = ALL institutions
    dii_ly       = df.get("dii_holdings",       pd.Series(50.0, index=df.index)).fillna(50)
    promo_ly     = df.get("promoter_holdings",  _ly_nan)
    # V1.1: Promoter buying — Lynch Ch15 p208+213: "insider buying is a positive sign"
    chg_promo_ly = df.get("change_promoter_1y", _ly_nan)
    is_fin_ly    = df.get("is_financial",       pd.Series(False, index=df.index)).fillna(False)
    # V1.1: Inventory surge — Lynch Ch13 p197: "inventories growing faster than sales = red flag"
    inv_gr_ly    = df.get("inv_growth",         _ly_nan)   # pre-computed in data_engine.py

    # ── Pillar V: Growth Velocity — Revenue speed + EPS per share + positive FCF ─
    _cash_ok_ly  = (fcf_ly.fillna(-1) > 0) | (fcfy_ly.fillna(-1) > 0)
    df["lynch_growth_velocity"] = (
        (rev_ly.fillna(0)  >= 20.0) &    # V1: Revenue 5Y CAGR ≥ 20% — Fast Grower definition
        (eps_ly.fillna(0)  >= 15.0) &    # V2: EPS per share ≥ 15% — dilution-adjusted earnings
        _cash_ok_ly                      # V3: FCF > 0 or FCF yield > 0 — cash verification
    ).astype(int)

    # ── Pillar P: PEG Sweet Spot ────────────────────────────────────────────────
    df["lynch_valuation_peg"] = (
        (peg_ly > 0) &                   # PEG must be positive (no earnings = not a Fast Grower yet)
        (peg_ly <= 0.75)                 # Lynch sweet spot: price trades at ≤ 0.75× growth rate
    ).astype(int)

    # ── Pillar D: Pre-Discovery — FII+DII combined < 20% ───────────────────────
    df["lynch_pre_discovery"] = (
        (fii_ly + dii_ly) < 20.0        # Combined institutional weight < 20% pre-discovery phase
    ).astype(int)

    # ── Pillar F: Fortress Balance Sheet + Owner Conviction (level OR active buying)
    _promo_ok_ly = (promo_ly.fillna(0) >= 45.0) | (chg_promo_ly.fillna(-1) > 0)
    df["lynch_fortress_owner"] = (
        (is_fin_ly | (debt_ly.fillna(999) < 0.5)) &   # Balance sheet: D/E < 0.5 (fin. sector exempt)
        _promo_ok_ly                                   # Owner conviction: ≥ 45% OR actively buying
    ).astype(int)

    # ── Inventory surge disqualifier — vetos lynch_pass but NOT lynch_score ────
    # Lynch Ch13 p197: "when inventories grow faster than sales — red flag"
    # A company with all 4 pillars green but ballooning inventory gets pass=0 while score=4
    _inv_surge_disq = (
        inv_gr_ly.notna() &
        (inv_gr_ly > (rev_ly.fillna(0) + 20.0))
    )

    fw_lynch = (
        (df["lynch_growth_velocity"] == 1) &
        (df["lynch_valuation_peg"]   == 1) &
        (df["lynch_pre_discovery"]   == 1) &
        (df["lynch_fortress_owner"]  == 1) &
        (~_inv_surge_disq)
    )
    df["lynch_pass"]  = fw_lynch.astype(int)
    df["lynch_score"] = (               # 0-4: pillar count only; inventory surge excluded from score
        df["lynch_growth_velocity"] +
        df["lynch_valuation_peg"]   +
        df["lynch_pre_discovery"]   +
        df["lynch_fortress_owner"]
    )

    # 6. CAN SLIM (William O'Neill) — earnings acceleration + technical leadership
    #    C: Quarterly EPS growth ≥ 25% YoY AND quarterly sales ≥ 25% YoY (Ch.3: top-line confirmation
    #       required — EPS spikes without revenue growth are value traps)
    #    A: Annual EPS 5Y CAGR ≥ 25% + ROE ≥ 17% + 3-year consistency (eps_gr_3y ≥ 0 AND
    #       eps_gr_yoy ≥ 0) — O'Neill specifies unbroken annual increases (Ch.4)
    #    N: Near 52W high — within 15% (book: "10-15% of 52-week high")
    #    S: Supply/Demand — vol_ratio ≥ 1.5 (book: "40-50% above average" = 1.4–1.5x)
    #       S-bonus: equity_shares <= equity_shares_1yb (float retraction / share buyback active)
    #    L: Leader — RS percentile ≥ 80 (book: "RS Rating 80+, average of winners = 87")
    #       IBD-weighted via d47_rs_composite: 50D×40% + 26W×30% + 52W×30% (data_engine.py D47)
    #       _pct_rank(d47_rs_composite, ascending=True) → percentile rank across all 2108 stocks
    #    I: Institutional sponsorship — FII or DII buying (best proxy for A/B A-D rating)
    #    M: Market direction — detect_market_regime(df) breadth consensus already computed in
    #       run_full_scoring() and stored in df.attrs["detected_market_regime"] before this function
    #       is called. BEAR regime blocks ALL new CAN SLIM entries (O'Neil Ch.9: 3/4 stocks follow
    #       market direction — buying in a bear market leads to immediate capital drawdown).
    #    Sources: CAN SLIM Mastery Guide Chapter 1 (C,A), Chapter 2 (N,S), Chapter 3 (L,I), Chapter 6 (M)
    #    Spec: docs/canslim_technical_specs.json
    # ── C criterion: use precomputed q_eps_yoy (EPS per share YoY) from data_engine.py.
    # O'Neil Ch.3 specifies EPS (per share), not total PAT. q_eps_yoy = (eps_lq - eps_pyq) /
    # abs(eps_pyq) * 100, with zero-base guard applied at source (data_engine.py:583).
    # fillna(0) → missing EPS history conservatively fails (0 < 25). No inline ratio
    # computation needed here — clean precomputed series only (noise-exclusion principle).
    q_eps_cs     = df.get("q_eps_yoy",       pd.Series(np.nan, index=df.index)).fillna(0)    # C: EPS per share gate
    q_rev_cs     = df.get("q_rev_yoy",       pd.Series(np.nan, index=df.index)).fillna(0)    # C: sales gate
    eps_gr_cs    = df.get("eps_gr_5y",        pd.Series(np.nan, index=df.index)).fillna(0)
    eps_gr_3y_cs = df.get("eps_gr_3y",        pd.Series(np.nan, index=df.index)).fillna(-1)  # A: consistency
    eps_yoy_cs   = df.get("eps_gr_yoy",       pd.Series(np.nan, index=df.index)).fillna(-1)  # A: consistency
    roe_cs       = df.get("roe",              pd.Series(np.nan, index=df.index)).fillna(0)    # A: ROE ≥ 17%
    dist_wh_cs   = df.get("dist_52wh",        pd.Series(999.0,  index=df.index)).fillna(999)
    # S baseline = 50-DAY average (O'Neil playbook, stated 4×: Ch.2 S-criterion, Ch.4
    # Rule 3, Ch.12 sacred table: "breakout volume 40–50%+ above the 50-day average").
    # vol_ratio (20D) is the fallback only when the 50D SMA is missing from the CSV.
    vol_r_cs     = (
        df.get("vol_ratio_50d",  pd.Series(np.nan, index=df.index))
        .fillna(df.get("vol_ratio", pd.Series(np.nan, index=df.index)))
        .fillna(1.0)
    )
    rs_comp_cs   = df.get("d47_rs_composite", pd.Series(np.nan, index=df.index)).fillna(0)
    fii_cs       = df.get("change_fii_lq",    pd.Series(0.0,    index=df.index)).fillna(0)
    dii_cs       = df.get("change_dii_lq",    pd.Series(0.0,    index=df.index)).fillna(0)
    # A: PAT step-growth — 3 consecutive years of unbroken profit increase (O'Neil Ch.4)
    # NaN comparisons in pandas return False → missing back-year data conservatively fails gate.
    # Note: uses PAT (total profit). O'Neil specifies EPS per share; PAT does not account for
    # dilution. Combined with eps_gr_3y_cs >= 0 and eps_yoy_cs >= 0 (EPS-based proxies above),
    # this adds direct year-by-year step verification as a complementary hard gate.
    pat_cs      = df.get("pat",     pd.Series(np.nan, index=df.index))
    pat_1yb_cs  = df.get("pat_1yb", pd.Series(np.nan, index=df.index))
    pat_2yb_cs  = df.get("pat_2yb", pd.Series(np.nan, index=df.index))
    pat_3yb_cs  = df.get("pat_3yb", pd.Series(np.nan, index=df.index))
    pat_step_ok = (
        (pat_cs     > pat_1yb_cs) &   # current year > 1 year ago
        (pat_1yb_cs > pat_2yb_cs) &   # 1 year ago > 2 years ago
        (pat_2yb_cs > pat_3yb_cs)     # 2 years ago > 3 years ago
    )
    # S-bonus: float retraction — equity_shares <= equity_shares_1yb (O'Neil Ch.6: supply constraint)
    # Share buybacks reduce outstanding float → structural supply-demand tilt toward price appreciation.
    # Added to can_slim_score only (not fw_can_slim hard gate) — O'Neil's primary S gate is volume surge.
    # Both values must be present; missing data = False (conservative: no buyback assumed).
    eq_shr_cs     = df.get("equity_shares",     pd.Series(np.nan, index=df.index))
    eq_shr_1yb_cs = df.get("equity_shares_1yb", pd.Series(np.nan, index=df.index))
    buyback_cs = (
        eq_shr_cs.notna() & eq_shr_1yb_cs.notna() &
        (eq_shr_cs <= eq_shr_1yb_cs)
    )
    # C+A bonus: EPS annual acceleration — eps_gr_yoy > eps_gr_3y means the MOST RECENT annual
    # EPS growth rate is HIGHER than the 3-year CAGR baseline, confirming the business is in
    # an acceleration phase (not a decelerating compounder riding prior-year momentum).
    # O'Neil Ch.3+Ch.4: "The rate of earnings increase should be getting larger in each of the
    # past few quarters or years — look for two or three periods of accelerating earnings growth."
    # Score bonus only (not fw_can_slim hard gate) — a great business can temporarily dip below
    # its 3Y CAGR while still growing 30%+ YoY (one-off high-base effect). Both series use the
    # same fillna strategy already applied above: eps_yoy_cs = fillna(-1), eps_gr_3y_cs = fillna(-1).
    eps_accel_cs = (eps_yoy_cs > eps_gr_3y_cs)   # C+A: current YoY EPS > 3Y CAGR = accelerating
    # A bonus: OPM expansion — O'Neil Ch.4: "Look for annual pre-tax profit margins at new peak
    # levels. This confirms pricing power and cost discipline, not just topline revenue growth."
    # Uses annual OPM (operating profit margin %) vs its prior year level.
    # fillna(0) neutral — missing margin data neither awarded nor penalized (not a hard gate).
    opm_cs      = df.get("opm",     pd.Series(np.nan, index=df.index)).fillna(0)
    opm_1yb_cs  = df.get("opm_1yb", pd.Series(np.nan, index=df.index)).fillna(0)
    opm_expand_cs = (opm_cs > opm_1yb_cs)         # A: OPM improving YoY → margin at new peak
    # S3-bonus: Minervini Volume Contraction Pattern (VCP) — volume drying up recently, then
    # surging on breakout. O'Neil Ch.6 + Minervini SEPA: the highest-conviction institutional
    # entries follow a period of volume contraction (supply absorbed quietly) then explosive
    # expansion on the pivot day. vol_sma_5d < vol_sma_20d confirms recent drying-up; vol_r_cs
    # >= 1.5 confirms the breakout surge is already firing. Both SMA series must be present and
    # positive — missing data fails conservatively (no dryup assumed).
    # Score bonus only — not a fw_can_slim hard gate (VCP is a setup quality enhancer).
    _vcp_5d  = df.get("vol_sma_5d",  pd.Series(np.nan, index=df.index))
    _vcp_20d = df.get("vol_sma_20d", pd.Series(np.nan, index=df.index))
    vcp_vol_cs = (
        _vcp_5d.notna() & _vcp_20d.notna() & (_vcp_20d > 0) &
        (_vcp_5d  < _vcp_20d) &   # S3: 5D avg < 20D avg → volume contracting in recent days
        (vol_r_cs >= 1.5)          # AND current session is surging ≥ 1.5× 20D baseline
    )
    # L2-bonus: RS line uptrend across all three CRS timeframes — crs_50d > crs_26w > crs_52w.
    # Static rs_pctrank_cs >= 80 measures WHERE the RS line is. This measures DIRECTION — whether
    # the RS line is rising. O'Neil: the strongest setups show RS lines making new highs BEFORE
    # price breaks out. crs_50d > crs_26w > crs_52w = short-term outperformance exceeds medium
    # which exceeds long-term → full ascending RS line. All three must be present; any missing
    # value fails conservatively (direction unconfirmed = no bonus).
    # Score bonus only — complements rs_pctrank_cs >= 80 hard gate (level + direction together).
    _rs_50d = df.get("crs_50d", pd.Series(np.nan, index=df.index))
    _rs_26w = df.get("crs_26w", pd.Series(np.nan, index=df.index))
    _rs_52w = df.get("crs_52w", pd.Series(np.nan, index=df.index))
    rs_uptrend_cs = (
        _rs_50d.notna() & _rs_26w.notna() & _rs_52w.notna() &
        (_rs_50d > _rs_26w) &   # L2: short-term RS > medium-term RS
        (_rs_26w > _rs_52w)     # medium-term RS > long-term RS → ascending RS line confirmed
    )
    # L: Percentile rank of IBD-weighted RS composite (50D×40% + 26W×30% + 52W×30%)
    # ascending=True: higher RS composite → higher rank → top 20% = RS Rating ≥ 80
    rs_pctrank_cs = _pct_rank(rs_comp_cs, ascending=True).fillna(50)
    # M: Market direction gate — stored in df.attrs by run_full_scoring() before this call.
    # detect_market_regime() is called before compute_qglp_score() in run_full_scoring().
    # Fallback to SIDEWAYS (pass) when called outside run_full_scoring() (e.g. unit tests).
    _regime_fallback = df["_detected_market_regime"].iloc[0] if "_detected_market_regime" in df.columns and len(df) > 0 else "SIDEWAYS"
    _regime_cs = df.attrs.get("detected_market_regime", _regime_fallback)  # "SIDEWAYS" when called outside run_full_scoring
    market_ok_cs = (_regime_cs != "BEAR")   # scalar bool — broadcasts across all rows
    fw_can_slim = (
        (q_eps_cs         >= 25.0)  &   # C: quarterly EPS per share +25%+ YoY (O'Neil Ch.3: EPS not PAT)
        (q_rev_cs         >= 25.0)  &   # C: quarterly sales +25%+ YoY (top-line validation)
        (eps_gr_cs        >= 25)    &   # A: annual EPS 5Y CAGR ≥ 25%
        (roe_cs           >= 17)    &   # A: ROE ≥ 17% (O'Neill's explicit A-criterion)
        (eps_gr_3y_cs     >= 0)     &   # A: 3Y EPS CAGR not negative — net positive trajectory
        (eps_yoy_cs       >= 0)     &   # A: not currently declining YoY — no recent contraction
        pat_step_ok                 &   # A: PAT grew each year for 3 consecutive years (O'Neil Ch.4)
        (dist_wh_cs       <= 15)    &   # N: within 15% of 52W high
        (vol_r_cs         >= 1.5)   &   # S: volume ≥ 1.5× avg (40-50%+ above average)
        (rs_pctrank_cs    >= 80)    &   # L: IBD-weighted RS percentile ≥ 80 (top 20% of universe)
        ((fii_cs > 0) | (dii_cs > 0)) & # I: institutional buying confirmed
        market_ok_cs                    # M: not in BEAR regime (breadth consensus)
    )
    df["can_slim_pass"] = fw_can_slim.astype(int)
    # CAN SLIM criteria count (0-17 components): useful for partial-pass display and ranking.
    # C: 2 hard + 1 bonus | A: 5 hard + 1 bonus | N: 1 | S: 1 hard + 2 bonus | L: 1 hard + 1 bonus | I: 1 | M: 1
    df["can_slim_score"] = (
        (q_eps_cs   >= 25.0).astype(int)  +                       # C1: quarterly EPS per share
        (q_rev_cs   >= 25.0).astype(int)  +                       # C2: quarterly sales
        eps_accel_cs.astype(int)          +                       # C+A bonus: EPS acceleration (YoY > 3Y CAGR)
        (eps_gr_cs  >= 25).astype(int)    +                       # A1: EPS 5Y CAGR
        (roe_cs     >= 17).astype(int)    +                       # A2: ROE
        (eps_gr_3y_cs >= 0).astype(int)   +                       # A3: 3Y EPS consistency
        (eps_yoy_cs >= 0).astype(int)     +                       # A4: YoY not declining
        pat_step_ok.astype(int)           +                       # A5: PAT step-growth (3 consecutive years)
        opm_expand_cs.astype(int)         +                       # A bonus: OPM expanding YoY (margin at peak)
        (dist_wh_cs <= 15).astype(int)    +                       # N
        (vol_r_cs   >= 1.5).astype(int)   +                       # S1: volume surge hard gate
        buyback_cs.astype(int)            +                       # S2-bonus: float retraction / share buyback
        vcp_vol_cs.astype(int)            +                       # S3-bonus: Minervini VCP (dryup + surge)
        (rs_pctrank_cs >= 80).astype(int) +                       # L1: IBD-weighted RS percentile hard gate
        rs_uptrend_cs.astype(int)         +                       # L2-bonus: RS line ascending (50D>26W>52W)
        ((fii_cs > 0) | (dii_cs > 0)).astype(int) +              # I
        pd.Series(int(market_ok_cs), index=df.index)              # M
    )

    # ── CAN SLIM Pillar Materialization ────────────────────────────────────────
    # Materialize individual pillar pass/fail flags as named DataFrame columns.
    # Single source of truth: all scoring logic lives here; ui_tearsheet.py reads
    # these flat 0/1 integers (pure display, zero threshold re-computation in UI).
    # M pillar broadcasts the scalar market_ok_cs across all rows via pd.Series.
    # market_regime stores the string regime tag for display (e.g., "BULL"/"BEAR").
    df["can_slim_c"]        = ((q_eps_cs >= 25.0) & (q_rev_cs >= 25.0)).astype(int)
    df["can_slim_a"]        = (
        (eps_gr_cs    >= 25)  &   # A1: EPS 5Y CAGR
        (roe_cs       >= 17)  &   # A2: ROE
        (eps_gr_3y_cs >= 0)   &   # A3: 3Y EPS consistency
        (eps_yoy_cs   >= 0)   &   # A4: YoY not declining
        pat_step_ok               # A5: PAT step-growth 3 consecutive years
    ).astype(int)
    df["can_slim_n"]        = (dist_wh_cs  <= 15).astype(int)
    df["can_slim_s"]        = (vol_r_cs    >= 1.5).astype(int)
    df["can_slim_l"]        = (rs_pctrank_cs >= 80).astype(int)
    df["can_slim_i"]        = ((fii_cs > 0) | (dii_cs > 0)).astype(int)
    df["can_slim_m"]        = pd.Series(int(market_ok_cs), index=df.index)
    df["market_regime"]     = pd.Series(str(_regime_cs).upper(), index=df.index)
    df["can_slim_vcp"]      = vcp_vol_cs.astype(int)       # S-bonus: Minervini VCP dryup
    df["can_slim_rs_trend"] = rs_uptrend_cs.astype(int)    # L-bonus: RS line ascending

    # 7. Bruised Blue Chip (29th MOSL Study) — quality company fallen hard + cheap vs history
    #    Criteria: ROCE ≥ 15% sustained + PAT CAGR ≥ 10% + fallen >40% from 52W high
    #              + current PE ≥ 25% below own 10Y median PE
    fw_bruised_bb = df.get("bruised_blue_chip", pd.Series(0, index=df.index)).fillna(0) == 1

    # 8. Economic Profit Improver (28th MOSL Study — TEM Hockey-Stick Setup)
    #    Companies moving UP the Economic Profit Power Curve:
    #    ROE improving + above cost of equity (10%) + Economic Profit is positive
    fw_ep_improver = (
        (df.get("eco_profit_improving", pd.Series(0, index=df.index)).fillna(0) == 1) &
        (df.get("economic_profit_positive", pd.Series(0, index=df.index)).fillna(0) == 1) &
        (df.get("d35_roce_trend", pd.Series(0, index=df.index)).fillna(0) > 0)  # ROCE also rising
    )

    # 9. Peaceful Investing (Vijay Malik) — India's most systematic forensic quality filter
    #    Malik's 8-parameter system is entirely derived from audited financials — the most
    #    India-specific framework in this system. 3 qualitative shenanigans (serial M&A,
    #    earnings smoothing, accounting policy changes) require data not in the CSV and are
    #    intentionally excluded. Financial sector is exempt from IC, D/E, and CR checks
    #    (structurally inapplicable, same as Coffee Can rationale).
    #    Unit note: cfo_to_pat is a PERCENTAGE in this CSV (73.04 = 73%). Threshold = 70, NOT 0.7.
    _mk_nan    = pd.Series(np.nan, index=df.index)
    rev_gr_mk  = df.get("rev_gr_10y", _mk_nan).fillna(df.get("rev_gr_5y", _mk_nan))
    npm_mk     = df.get("npm",              _mk_nan)
    npm_1yb_mk = df.get("npm_1yb",          _mk_nan)
    ic_mk      = df.get("interest_coverage", _mk_nan)
    de_mk      = df.get("debt_to_equity",    _mk_nan)
    cr_mk      = df.get("current_ratio",     _mk_nan)
    cfo_pat_mk = df.get("cfo_to_pat",        _mk_nan)   # PERCENTAGE: 73.04 = 73%
    ssgr_mk    = df.get("ssgr_self_funded",  pd.Series(0, index=df.index)).fillna(0)
    is_fin_mk  = df.get("is_financial",      pd.Series(False, index=df.index)).fillna(False)
    # NPM stability: current ≥ 8% AND (prior year ≥ 6% OR prior year data unavailable)
    # Guards against one-year NPM spike that doesn't reflect the business's true earning power.
    npm_stable_mk = (npm_mk.fillna(0) >= 8) & (npm_1yb_mk.isna() | (npm_1yb_mk.fillna(0) >= 6))
    # ── Pillar flags — 5 independent materialized binary columns ─────────────
    # Each pillar maps to one of Malik's 8 financial parameters or the SSGR signature.
    # Financial sector (is_fin_mk) is fully exempt from the Debt Fortress sub-gates.
    # Spec: docs/malik_peaceful_specs.json

    df["malik_growth_runway"] = (
        rev_gr_mk.fillna(0) >= 10.0               # P1: Sales CAGR ≥ 10% (10Y primary, 5Y fallback)
    ).astype(int)

    df["malik_profit_stability"] = (
        npm_stable_mk                              # P2: NPM ≥ 8% AND prior ≥ 6% (or unavailable)
    ).astype(int)

    df["malik_debt_fortress"] = (
        is_fin_mk | (                              # Financial sector exempt
            (ic_mk.fillna(0)   >= 3.0) &          # P4: Interest coverage ≥ 3×
            (de_mk.fillna(999) <= 0.5) &           # P5: D/E ≤ 0.5 — Malik's stricter standard
            (cr_mk.fillna(0)   >= 1.25)            # P6: Current ratio ≥ 1.25
        )
    ).astype(int)

    df["malik_cash_generation"] = (
        cfo_pat_mk.fillna(0) >= 70.0               # P8: CFO/PAT ≥ 70% — PERCENTAGE (70.0 not 0.70)
    ).astype(int)

    df["malik_self_funded"] = (
        ssgr_mk == 1                               # SSGR: growth self-funded — Malik's signature signal
    ).astype(int)

    fw_malik_peaceful = (
        (df["malik_growth_runway"]   == 1) &
        (df["malik_profit_stability"] == 1) &
        (df["malik_debt_fortress"]    == 1) &
        (df["malik_cash_generation"]  == 1) &
        (df["malik_self_funded"]      == 1)
    )
    df["malik_pass"]  = fw_malik_peaceful.astype(int)
    df["malik_score"] = (                          # 0-5: count of Malik pillars cleared
        df["malik_growth_runway"]    +
        df["malik_profit_stability"] +
        df["malik_debt_fortress"]    +
        df["malik_cash_generation"]  +
        df["malik_self_funded"]
    )

    # 10. Unusual Billionaires (Saurabh Mukherjea) — The Greatness Formula
    #    DELIBERATELY DISTINCT from Coffee Can (also Mukherjea):
    #      Coffee Can: ROE ≥ 15% + CFO/EBITDA ≥ 90% (cash quality first)
    #      Unusual Billionaires: ROCE ≥ 15% (capital efficiency first, no CFO/EBITDA gate)
    #    A high-leverage company can pass Coffee Can (ROE via D/E 0.8) but fail here (ROCE lower).
    #    A capex-heavy ROCE leader can fail Coffee Can's 90% CFO/EBITDA gate but pass here.
    #    The book's actual requirement is EVERY year for 10 years — not implementable without
    #    annual data. Best proxy: BOTH the 10Y and 5Y ROCE medians clear 15% (two overlapping
    #    windows, harder to fake than a single average). Same logic for revenue.
    _ub_nan     = pd.Series(np.nan, index=df.index)
    is_fin_ub   = df.get("is_financial",       pd.Series(False, index=df.index)).fillna(False)
    # Capital efficiency gate: ROCE for non-financials, ROE for financials.
    # Banks/NBFCs have no traditional "capital employed" → roce_med_10y is NaN or near-zero.
    # fillna(0) on NaN ROCE → 0 >= 15 = False → every elite bank silently fails without this routing.
    # Book Ch.1: "for BFSI companies: ROE ≥ 15% and loan growth ≥ 15% every year."
    roce_10y_ub = df.get("roce_med_10y",       _ub_nan)
    roce_5y_ub  = df.get("roce_med_5y",        _ub_nan)
    roe_10y_ub  = df.get("roe_med_10y",        _ub_nan).fillna(df.get("roe_med_5y", _ub_nan))
    roe_5y_ub   = df.get("roe_med_3y",         _ub_nan).fillna(df.get("roe_med_5y", _ub_nan)).fillna(df.get("roe", _ub_nan))
    ub_cap_eff_nonfin = (
        (roce_10y_ub.fillna(0) >= 15) &  # ROCE 10Y ≥ 15%: decade-long capital efficiency proven
        (roce_5y_ub.fillna(0)  >= 15)    # ROCE 5Y  ≥ 15%: moat still intact in recent window
    )
    ub_cap_eff_fin = (
        (roe_10y_ub.fillna(0)  >= 15) &  # ROE 10Y ≥ 15%: bank/NBFC capital returns over cycle
        (roe_5y_ub.fillna(0)   >= 15)    # ROE 5Y  ≥ 15%: returns sustained (not just historical)
    )
    ub_efficiency_gate = pd.Series(
        np.where(is_fin_ub, ub_cap_eff_fin, ub_cap_eff_nonfin),
        index=df.index
    )
    rev_10y_ub  = df.get("rev_gr_10y",         _ub_nan)
    rev_5y_ub   = df.get("rev_gr_5y",          _ub_nan)
    de_ub       = df.get("debt_to_equity",     _ub_nan)
    pledge_ub   = df.get("pledged_percentage", _ub_nan)
    promo_ub    = df.get("promoter_holdings",  _ub_nan)
    opm_st_ub   = df.get("opm_stable",         pd.Series(0, index=df.index)).fillna(0)
    # Sector-routed Greatness Formula growth hurdle (UB/Coffee Can unified research base):
    # financial companies (banks/NBFCs) must show 15% expansion; industrials need 10%.
    ub_growth_hurdle = pd.Series(np.where(is_fin_ub, 15.0, 10.0), index=df.index)
    fw_unusual_billionaires = (
        ub_efficiency_gate             &                     # ROCE (non-fin) / ROE (fin) ≥ 15% — book exact
        (rev_10y_ub.fillna(0)  >= ub_growth_hurdle) &       # Greatness: sector-routed 10/15% growth hurdle
        (rev_5y_ub.fillna(0)   >= 8)  &                     # Greatness: no recent sharp deceleration
        _cc_each_year                  &                     # Year-by-year: each of years 2-5 ≥ 10% growth
        (opm_st_ub == 1)               &                     # Moat proxy: OPM stable through cycles
        (is_fin_ub | (de_ub.fillna(999) < 1.0)) &           # Capital discipline: D/E < 1 for non-financials
        (pledge_ub.fillna(0)   < 10)  &                     # Governance pillar: pledge < 10%
        (promo_ub.fillna(0)    >= EPOCH35_UNUSUAL_BILLIONAIRES["min_promoter_stake"])  # Skin-in-game: promoter ≥ 45%
    )
    df["ub_pass"] = fw_unusual_billionaires.astype(int)

    # 11. Fisher Quality (Philip Fisher) — Systematic quantitative proxies for Fisher's key measurable criteria.
    #    Fisher's framework is 90% qualitative (scuttlebutt, channel checks, management DNA) —
    #    none of which is in the CSV. Only 6 of his 15 points have reliable quantitative proxies.
    #    These 6 already power the tearsheet's "Systematic Fisher Proxy" module. This framework
    #    tag makes those same checks filterable in the main scan — stocks passing all 6 earn the badge.
    #    P15 (integrity) is Fisher's MASTER FILTER — any forensic flag fails it unconditionally.
    #    Unit note: cfo_to_pat is PERCENTAGE (73.04 = 73%). Threshold = 70, not 0.7.
    _fi_nan     = pd.Series(np.nan, index=df.index)
    rev_gr_fi   = df.get("rev_gr_5y",          _fi_nan)
    npm_fi      = df.get("npm",                _fi_nan)
    npm_1yb_fi  = df.get("npm_1yb",            _fi_nan)
    cfo_pat_fi  = df.get("cfo_to_pat",         _fi_nan)   # PERCENTAGE: 73.04 = 73%
    dilut_fi    = df.get("dilution_flag",      pd.Series(1, index=df.index)).fillna(1)
    oplev_fi    = df.get("operating_leverage", pd.Series(0, index=df.index)).fillna(0)
    fscore_fi   = df.get("forensic_score",     pd.Series(999, index=df.index)).fillna(999)
    fw_fisher = (
        (fscore_fi   >= 90)                    &   # P15: Integrity — clean or watch rating (<= 2 flags)
        (rev_gr_fi.fillna(0)  >= 15)          &   # P1:  Market growth ≥ 15% revenue CAGR
        (npm_fi.fillna(0)     >= 10)          &   # P5:  Worthwhile margin ≥ 10% NPM
        (npm_fi.fillna(0) >= npm_1yb_fi.fillna(0)) &  # P6:  Margins not declining vs prior year
        (cfo_pat_fi.fillna(0) >= 70)          &   # P10: Accounting controls — CFO/PAT ≥ 70%
        (dilut_fi == 0)                       &   # P13: Zero equity dilution
        (oplev_fi == 1)                           # P6 proxy: Operating leverage — profit growing
                                                  #   faster than sales = the "margin improvement
                                                  #   drive" (book P6: "What is the company doing to
                                                  #   maintain or improve profit margins?").
                                                  #   NOT P4 — book P4 is the sales ORGANIZATION
                                                  #   (qualitative, no CSV proxy). Mislabel fixed
                                                  #   2026-06-12 against the converted book text.
    )

    # 12. 100-Bagger Hunter (Christopher Mayer — Hybrid Spec) — Early-stage Indian compounder
    #    Hybrid design: long-cycle quality (7Y ROCE) + early-stage growth momentum (3Y PAT)
    #    + governance gates. Spec ledger: docs/hundred_bagger_specs.json.
    #    S (Size): ₹200–₹3,000 Cr. ₹3k Cr × 100 = ₹3 Lakh Cr — historically achievable runway
    #      (Eicher Motors: ₹800 Cr → ₹1.1 Lakh Cr; Bajaj Finance: ₹1,200 Cr → ₹4.7 Lakh Cr).
    #    Q (Quality): ROCE 7Y median ≥ 15% — full business-cycle capital efficiency, not a 5Y spike.
    #    G (Growth): Revenue 5Y CAGR ≥ 15% (data-available for full universe) AND
    #                PAT 3Y CAGR ≥ 20% — the "engine-fire" signal that the compounding is active NOW.
    #    B (Balance sheet): D/E < 0.5 — fortress balance sheet; financial sector exempted (banks use
    #      leverage structurally — ROCE/ROE gates already capture quality for them).
    #    O (Owner-Operator): promoter ≥ 50% — majority control; founder thinks in decades, not quarters.
    #    C (Cash): CFO/PAT > 0 (PERCENTAGE) — real cash backs earnings; anti-fraud gate, not strict ≥80%.
    #      Small-caps in expansion phase can have CFO/PAT 50–70% (working capital absorption) — that
    #      is normal. What we reject is CFO/PAT ≤ 0: negative OCF on positive PAT = accounting concern.
    #    P (Price): PEG 0–2.0 — valuation discipline. Premium permitted for explosive growth but not
    #      unbounded. No annual stumble check — Mayer explicitly studies companies with messy early
    #      years; the annual stumble filter belongs to Baid Compounder, not this framework.
    #    Unit note: market_cap in Crores; promoter_holdings and cfo_to_pat as % (50.0 = 50%).
    _hb_nan     = pd.Series(np.nan,  index=df.index)
    _hb_peg_nan = pd.Series(999.0,   index=df.index)
    mcap_hb     = df.get("market_cap",        _hb_nan)
    roce_7y_hb  = df.get("roce_med_7y",       _hb_nan)
    rev_5y_hb   = df.get("rev_gr_5y",         _hb_nan)
    pat_3y_hb   = df.get("pat_gr_3y",         _hb_nan)
    de_hb       = df.get("debt_to_equity",    _hb_nan)
    promo_hb    = df.get("promoter_holdings", _hb_nan)   # % e.g. 55.3 = 55.3%
    pledge_hb   = df.get("pledged_percentage", _hb_nan)  # % pledged; missing in CSV = none reported
    cfo_pat_hb  = df.get("cfo_to_pat",        _hb_nan)   # PERCENTAGE: 73.0 = 73%, not 0.73
    is_fin_hb   = df.get("is_financial",      pd.Series(False, index=df.index)).fillna(False)
    peg_hb      = df.get("peg",               _hb_peg_nan)
    fw_100_bagger = (
        (mcap_hb.fillna(0)     >= 200.0)  &       # S: ₹200 Cr floor — proven model, liquid stock
        (mcap_hb.fillna(9999)  <= 3000.0) &       # S: ₹3,000 Cr ceiling — 100× math achievable
        (roce_7y_hb.fillna(0)  >= 15.0)   &       # Q: ROCE 7Y ≥ 15% — cycle-proof capital efficiency
        (rev_5y_hb.fillna(0)   >= 15.0)   &       # G: Revenue 5Y CAGR ≥ 15% — sustained top-line demand
        (pat_3y_hb.fillna(0)   >= 20.0)   &       # G: PAT 3Y CAGR ≥ 20% — earnings engine firing now
        (is_fin_hb | (de_hb.fillna(999) < 0.5)) & # B: D/E < 0.5 — fortress balance sheet
        (promo_hb.fillna(0)    >= 50.0)   &       # O: Promoter ≥ 50% — majority control, skin in game
        (pledge_hb.fillna(0)   < 20.0)    &       # O: Pledge < 20% — playbook safety line; a pledged
                                                  #    promoter faces margin-call selling pressure that
                                                  #    contradicts decades-horizon owner alignment
        (cfo_pat_hb.fillna(-1) >  0.0)    &       # C: CFO/PAT > 0 — real cash, not fraudulent earnings
        (peg_hb.fillna(999)    >  0.0)    &       # P: positive earnings required (no loss-maker)
        (peg_hb.fillna(999)    <= 2.0)            # P: PEG ≤ 2.0 — disciplined valuation entry
    )
    df["hundred_bagger_pass"] = fw_100_bagger.astype(int)

    # 13. Diamond Field Guide (Saurabh Mukherjea) — forensic-verified compounders
    #    Three-lens framework: Stage 1 Screen → Gate Zero → Lens 1 (Accounts) → Lens 2 (Moat) → Lens 3 (Capex)
    #    Spec ledger: docs/diamonds_financial_specs.json
    #    Key distinctions from other Mukherjea frameworks:
    #      - D/E < 0.5: STRICTEST of all three Mukherjea books (Coffee Can < 1.0, Unusual Billionaires < 1.0)
    #      - CFO/PAT ≥ 80%: Lens 1 cash earnings quality (book: 0.8 ratio; Coffee Can uses CFO/EBITDA)
    #      - DSO delta ≤ 15 days: Lens 1 channel-stuffing guard (1Y proxy — no 3YB receivable data in CSV)
    #      - FCF/CFO ≥ 25%: Lens 3 capital allocation surplus — new signal absent in all prior frameworks
    #      - forensic_score == 0: mandatory clean accounts — most frameworks don't hard-require this
    #      - Market cap ≥ ₹500 Cr: quality size floor (not in Coffee Can or Unusual Billionaires)
    #    NOT implementable (no CSV data): year-by-year CFO/PAT 10Y series, depreciation consistency,
    #    RPT ratios, auditor quality, GNPA/CASA (banks), moat durability scoring
    _dm_nan     = pd.Series(np.nan, index=df.index)
    roce_10y_dm = df.get("roce_med_10y",          _dm_nan)
    roce_5y_dm  = df.get("roce_med_5y",           _dm_nan)
    rev_10y_dm  = df.get("rev_gr_10y",            _dm_nan)
    rev_5y_dm   = df.get("rev_gr_5y",             _dm_nan)
    de_dm       = df.get("debt_to_equity",        _dm_nan)
    cfo_pat_dm  = df.get("cfo_to_pat",            _dm_nan)   # PERCENTAGE: 80.0 = 80%, not 0.80
    fcf_cfo_dm  = df.get("cumulative_fcf_to_ccfo", _dm_nan)  # PERCENTAGE proxy: 25.0 = 25%
    dso_delta_dm = df.get("dso_delta_3y",         _dm_nan)   # DAYS: 1Y delta proxy for 3Y window
    mcap_dm     = df.get("market_cap",            _dm_nan)   # Crores
    promo_dm    = df.get("promoter_holdings",     _dm_nan)   # % e.g. 40.0 = 40%
    pledge_dm   = df.get("pledged_percentage",    pd.Series(100.0, index=df.index)).fillna(100)
    # Diamond-specific forensic gate: only the 6 flags directly mapped to Mukherjea's
    # Three-Lens Framework. rf_low_cfo_ebitda (Coffee Can) and all Malik/WCS24/sector
    # flags are excluded — they belong to other frameworks, not Diamonds in the Dust.
    fscore_dm   = df.get("dm_forensic_flag_count", pd.Series(999, index=df.index)).fillna(999)
    is_fin_dm   = df.get("is_financial",          pd.Series(False, index=df.index)).fillna(False)
    fw_diamond = (
        (roce_10y_dm.fillna(0)   >= 15)   &   # Lens 2: ROCE > 15% 10Y — moat proven over full cycle
        (roce_5y_dm.fillna(0)    >= 15)   &   # Lens 2: ROCE > 15% 5Y — moat sustained recently
        (rev_10y_dm.fillna(0)    >= 10)   &   # Stage 1: Revenue growth 10Y > 10%
        (rev_5y_dm.fillna(0)     >=  8)   &   # Stage 1: Recent growth not decelerating sharply
        (is_fin_dm | (de_dm.fillna(999) < 0.5)) &  # Stage 1: D/E < 0.5 — strictest Mukherjea filter
        (cfo_pat_dm.fillna(0)    >= 80.0) &   # Lens 1: CFO/PAT ≥ 80% cash earnings quality (book: 0.8)
        (dso_delta_dm.fillna(999) <= 15.0) &  # Lens 1: DSO rise ≤ 15 days over 3Y (channel-stuffing guard)
        (fcf_cfo_dm.fillna(0)    >= 25)   &   # Lens 3: FCF/CFO ≥ 25% capital allocation surplus
        (mcap_dm.fillna(0)       >= 500)  &   # Stage 1: ≥ ₹500 Cr proven business scale
        (promo_dm.fillna(0)      >= 40)   &   # Gate Zero: Promoter ≥ 40% alignment
        (pledge_dm               <  10)   &   # Gate Zero: Pledge < 10%
        (fscore_dm               ==  0)       # Lens 1: Zero Diamond-specific forensic flags (dm_forensic_flag_count)
    )
    df["diamonds_pass"] = fw_diamond.astype(int)

    # 14. Dorsey Wide Moat (Pat Dorsey — The Five Rules for Successful Stock Investing)
    #    Confirmed Wide Moat at an attractive free-cash-flow price.
    #    THRESHOLD PROVENANCE (book-verified 2026-06-12 against the converted text; see
    #    docs/dorsey_moat_specs.json — the book is DIRECTIONAL, most numbers are proxies):
    #      1. ROCE ≥ 20% (both windows): BOOK-ANCHORED LEVEL — "consistent ROEs over 20
    #         percent, there's a good chance you're really on to something" (book, ROE
    #         benchmarks; ≥10% = worth investigating). ROCE-for-India + dual 10Y/5Y window
    #         construction = engineering proxy.
    #      2. FCF yield ≥ 5%: ENGINEERING PROXY — the book's 5% rule is FCF/SALES ("free
    #         cash flow as a percentage of sales is around 5 percent" = cash machine), and
    #         its valuation yield metric is "cash return" = FCF/Enterprise Value with NO
    #         fixed threshold. This gate blends the two: book metric family, transplanted
    #         number, market-cap denominator simplification. Do NOT quote as "Dorsey says".
    #      3. d35_roce_trend ≥ 0: moat DIRECTION — book-true concept (returns eroding
    #         toward cost of capital = moat narrowing); the 2Y-slope metric is a proxy.
    #    CFO/PAT ≥ 80%: ENGINEERING PROXY — no 0.8 threshold exists in the book; the
    #    cash-backs-earnings concept is book-true (shared bar with Diamond/Marks).
    #    D/E < 1.0: BOOK-EXACT investigate-line — "a debt-to-equity ratio over 1.0 — ask
    #    yourself the following [questions]" (book, financial-health chapter); financials
    #    exempt per the book's own banking-leverage caveat (ROE bar 12% for financials).
    #    NOT implementable: moat source classification (brand/network/switching — qualitative),
    #    margin-of-safety vs intrinsic value (requires DCF), CASA/GNPA banking metrics (not in CSV)
    _dw_nan      = pd.Series(np.nan, index=df.index)
    roce_10y_dw  = df.get("roce_med_10y",    _dw_nan)
    roce_5y_dw   = df.get("roce_med_5y",     _dw_nan)
    cfo_pat_dw   = df.get("cfo_to_pat",      _dw_nan)   # PERCENTAGE: 80.0 = 80%
    fcf_yield_dw = df.get("fcf_yield",       _dw_nan)   # PERCENTAGE: 5.0 = 5%
    roce_dir_dw  = df.get("d35_roce_trend",  _dw_nan)   # positive = ROCE structural slope rising (2Y annualised)
    de_dw        = df.get("debt_to_equity",  _dw_nan)
    is_fin_dw    = df.get("is_financial",    pd.Series(False, index=df.index)).fillna(False)

    # ── Dorsey Pillar Materialization ──────────────────────────────────────────
    # Single source of truth: all scoring logic lives here; ui_tearsheet reads
    # these flat 0/1 integers (pure display — zero threshold re-computation in UI).
    # M — Wide Moat Return Level: both 10Y AND 5Y ROCE ≥ 20% (book-anchored level: ROE>20%
    #     consistent = "really on to something"; ROCE/dual-window = proxy construction)
    df["dorsey_moat_level"]    = ((roce_10y_dw.fillna(0) >= 20.0) & (roce_5y_dw.fillna(0) >= 20.0)).astype(int)
    # D — Moat Direction: ROCE trajectory stable or widening (book-true concept; 2Y-slope metric = proxy)
    df["dorsey_moat_direction"] = (roce_dir_dw.fillna(-1) >= 0).astype(int)
    # V — FCF Valuation Yield ≥ 5% (PROXY: book's 5% is FCF/SALES; book's yield metric is
    #     cash return FCF/EV with no threshold — see provenance block above)
    df["dorsey_fcf_valuation"]  = (fcf_yield_dw.fillna(0) >= 5.0).astype(int)
    # Q — Cash Realization Quality: CFO/PAT ≥ 80% (PROXY: no 0.8 in the book; concept book-true)
    df["dorsey_cash_quality"]   = (cfo_pat_dw.fillna(0) >= 80.0).astype(int)
    # C — Capital Structure Cushion: D/E < 1.0 (BOOK-EXACT investigate-line); financials exempt
    df["dorsey_cap_structure"]  = (is_fin_dw | (de_dw.fillna(999) < 1.0)).astype(int)

    # Combine 5 materialized pillars into the pass flag and score
    fw_dorsey = (
        (df["dorsey_moat_level"]    == 1) &
        (df["dorsey_moat_direction"] == 1) &
        (df["dorsey_fcf_valuation"]  == 1) &
        (df["dorsey_cash_quality"]   == 1) &
        (df["dorsey_cap_structure"]  == 1)
    )
    df["dorsey_pass"]  = fw_dorsey.astype(int)
    df["dorsey_score"] = (
        df["dorsey_moat_level"]    +
        df["dorsey_moat_direction"] +
        df["dorsey_fcf_valuation"]  +
        df["dorsey_cash_quality"]   +
        df["dorsey_cap_structure"]
    )  # 0-5 sub-gate count; enables partial-pass ranking

    # 15. Outsiders on Dalal Street — Capital Allocation Excellence (Thorndike)
    #    4-Pillar Materialized Architecture — each pillar is an independent observable column.
    #    Source: docs/outsider_specs.json
    #
    #    Pillar S (Share Retirement): dilution_flag == 0
    #      Thorndike’s core finding: the best CEOs never diluted per-share value. Singleton
    #      repurchased 90% of Teledyne at 8× earnings; Graham/Washington Post reduced share count
    #      by 43%. fillna(1) = missing data → diluted assumed → conservative exclusion.
    #
    #    Pillar D (Debt Discipline): de_slope_3y <= 0
    #      3-year D/E slope flat or declining = management is deleveraging from cash generation,
    #      not leveraging up for empire-building. The ONLY framework rewarding deleveraging
    #      trajectory (all others use static D/E level). fillna(999) = trend unconfirmed → excluded.
    #
    #    Pillar C (Cash Generation): cfo_to_pat >= 85% (HIGHEST threshold in the entire system)
    #      Diamond=75%, Dorsey=80%, Outsiders=85%. True Outsider CEOs manage for cash, never
    #      reported earnings. cfo_to_pat is PERCENTAGE in CSV (85.0 = 85%). fillna(0) → excluded.
    #
    #    Pillar R (Capital Returns): roce_med_10y >= 15%
    #      Full-cycle hurdle rate: 10Y median ROCE above India WACC (≈12-13%). fillna(0) → excluded.
    #
    #    NOT implementable: HQ cost < 0.5% FCF (not in CSV), CEO communication quality,
    #    decentralisation structure, acquisition ROIC vs hurdle (no M&A data in CSV).

    # ── Pillar Materialization (vectorized, zero loops) ────────────────────────
    df["outsider_share_retirement"] = (
        df.get("dilution_flag", pd.Series(1, index=df.index)).fillna(1) == 0
    ).astype(int)

    df["outsider_debt_discipline"] = (
        df.get("de_slope_3y", pd.Series(999, index=df.index)).fillna(999) <= 0.0
    ).astype(int)

    df["outsider_cash_generation"] = (
        df.get("cfo_to_pat", pd.Series(0, index=df.index)).fillna(0) >= 85.0
    ).astype(int)

    df["outsider_capital_returns"] = (
        df.get("roce_med_10y", pd.Series(0, index=df.index)).fillna(0) >= 15.0
    ).astype(int)

    # ── Pass flag: AND of all 4 pillars ────────────────────────────────────
    fw_outsider = (
        (df["outsider_share_retirement"] == 1) &
        (df["outsider_debt_discipline"]  == 1) &
        (df["outsider_cash_generation"]  == 1) &
        (df["outsider_capital_returns"]  == 1)
    )
    df["outsider_pass"]  = fw_outsider.astype(int)
    df["outsider_score"] = (
        df["outsider_share_retirement"] +
        df["outsider_debt_discipline"]  +
        df["outsider_cash_generation"]  +
        df["outsider_capital_returns"]
    )  # 0-4 sub-gate count; enables partial-pass ranking (3/4 = one gate blocking)

    # 16. Quality Investing Codex (AKO Capital) — Three-Circle Quality Compounder
    #    Three-Circle Framework: Business Quality + Management Quality + Growth Quality must ALL pass.
    #    Two signals unique across all 16 frameworks:
    #      1. nfat > 4: capital intensity < 25% (Net FA Turnover > 4 = revenue is 4× net fixed assets).
    #         The ONLY framework explicitly gating on asset-lightness. Compounders like Asian Paints
    #         (NFAT~10), Pidilite (~8), Page Industries (~12) pass; capital-heavy businesses fail even
    #         with good ROCE — because high ROCE in capital-heavy industries is cyclical, not structural.
    #      2. fcf_yield >= 2: "fair value" FCF floor — NOT Dorsey's "cheap" 5%. The Codex says 2-3%
    #         is buy-and-hold for confirmed quality. Asian Paints at 2.5% FCF yield: PASSES here,
    #         FAILS fw_dorsey (needs 5%). This is the key insight: you can pay fair price for quality.
    #    CFO/PAT ≥ 80% (PERCENTAGE): Three-Circle Business Quality minimum (70-90% = good)
    #    nfat fillna(0): missing NFAT data → 0 < 4 → excluded (cannot confirm asset-light)
    #    fcf_yield fillna(0): negative/missing FCF → 0 < 2 → excluded (no cash generation = no quality)
    #    NOT implementable: ROIIC per decision (no M&A data), recurring revenue > 60% (no split in CSV),
    #    pricing power > CPI inflation (no revenue/unit data), management quality score 17/25 (qualitative)
    _qi_nan      = pd.Series(np.nan, index=df.index)
    roce_10y_qi  = df.get("roce_med_10y",    _qi_nan)
    roce_5y_qi   = df.get("roce_med_5y",     _qi_nan)
    rev_10y_qi   = df.get("rev_gr_10y",      _qi_nan).fillna(df.get("rev_gr_5y", _qi_nan))
    cfo_pat_qi   = df.get("cfo_to_pat",      _qi_nan)   # PERCENTAGE: 80.0 = 80%, not 0.80
    nfat_qi      = df.get("nfat",            _qi_nan)   # Revenue / Net Fixed Assets (turnover)
    fcf_yield_qi = df.get("fcf_yield",       _qi_nan)   # PERCENTAGE: 2.0 = 2% FCF yield
    de_qi        = df.get("debt_to_equity",  _qi_nan)
    is_fin_qi    = df.get("is_financial",    pd.Series(False, index=df.index)).fillna(False)
    fw_quality = (
        (roce_10y_qi.fillna(0)  >= 15) &             # Business Quality: ROIC > 15% 10Y (Circle 1)
        (roce_5y_qi.fillna(0)   >= 15) &             # Business Quality: ROIC sustained recently
        (rev_10y_qi.fillna(0)   >= 10) &             # Growth Quality: revenue compounding 10Y (Circle 3)
        (cfo_pat_qi.fillna(0)   >= 80) &             # Business Quality: FCF/PAT ≥ 80% (Three-Circle gate)
        (nfat_qi.fillna(0)      >   4) &             # UNIQUE: asset-light moat — capital intensity < 25%
        (fcf_yield_qi.fillna(0) >=  2) &             # UNIQUE: fair value FCF floor — not cheap, just rational
        (is_fin_qi | (de_qi.fillna(999) < 0.5))      # Balance sheet: D/E < 0.5 (book Stage 1 threshold)
    )

    # 17. Dhandho Asymmetry (Mohnish Pabrai — The Dhandho Investors Codex)
    #    "Heads I win, tails I don't lose much." — Pabrai's framework identifies situations where
    #    the stock price implies catastrophe but fundamentals confirm the business is intact.
    #    The two-part test: HIGH UNCERTAINTY (price signal) + LOW ACTUAL RISK (quality signal).
    #    Three signals unique across all 17 frameworks when combined:
    #      1. dist_52wh >= 30: fallen 30%+ from 52W high — the UNCERTAINTY proxy. Market has
    #         priced in distress. Bruised Blue Chip uses >40%; Dhandho's 30% threshold is
    #         deliberately wider — catches earlier-stage dislocations before they become obvious.
    #         Combined with FCF gate, this is NOT a pure distress play.
    #      2. fcf_yield >= 8%: the LOW ACTUAL RISK proof. A fallen stock still generating ≥8%
    #         FCF yield means payback ≤ 12.5 years — Pabrai's "bet" pays off even in zero-growth
    #         scenario. No other framework pairs a distress signal with an absolute FCF yield floor.
    #      3. forensic_score == 0: Pabrai's "accounting integrity" conviction — he only bets on
    #         companies where the financials are completely clean. Diamond also requires this,
    #         but Diamond has NO price-distress requirement (it's a buy-at-any-price quality filter).
    #    ROCE ≥ 15% (5Y median): moat-intact proof — business was quality pre-fall, not a value trap.
    #    CFO/PAT ≥ 70%: earnings must be real (PERCENTAGE: 70 = 70%, not 0.70).
    #    D/E < 0.5: Pabrai avoids leveraged businesses entirely — distress + leverage = actual risk.
    #    dist_52wh fillna(0): NaN → 0% fall → 0 < 30 → excluded (cannot confirm dislocation).
    #    fcf_yield fillna(0): NaN/negative FCF → excluded (no cash = no asymmetry, just value trap).
    #    NOT implementable: qualitative moat assessment (Pabrai reads annual reports personally),
    #    owner-operator check (no proxy for Pabrai's promoter-quality assessment beyond pledge),
    #    comparable transaction valuation (no private deal comps in CSV)
    _dh_nan      = pd.Series(np.nan, index=df.index)
    dist_wh_dh   = df.get("dist_52wh",       pd.Series(0.0, index=df.index)).fillna(0)
    fcf_yield_dh = df.get("fcf_yield",       _dh_nan)   # PERCENTAGE: 8.0 = 8% FCF yield
    roce_5y_dh   = df.get("roce_med_5y",     _dh_nan)   # 5Y ROCE median — moat-intact proxy
    cfo_pat_dh   = df.get("cfo_to_pat",      _dh_nan)   # PERCENTAGE: 70.0 = 70%, not 0.70
    de_dh        = df.get("debt_to_equity",  _dh_nan)
    _rfcount_dh  = df.get("red_flag_count",  pd.Series(999, index=df.index)).fillna(999)
    is_fin_dh    = df.get("is_financial",    pd.Series(False, index=df.index)).fillna(False)
    fw_dhandho = (
        (dist_wh_dh        >= 30) &              # HIGH UNCERTAINTY: fallen 30%+ from 52W high
        (fcf_yield_dh.fillna(0) >= 8) &          # LOW ACTUAL RISK: FCF yield ≥ 8% (payback ≤ 12.5Y)
        (roce_5y_dh.fillna(0)  >= 15) &          # Moat intact: ROIC > 15% over 5Y — not a value trap
        (cfo_pat_dh.fillna(0)  >= 70) &          # Earnings real: CFO/PAT ≥ 70% (Pabrai's cash test)
        (is_fin_dh | (de_dh.fillna(999) < 0.5)) & # Balance sheet: D/E < 0.5 — no leverage amplifying risk
        (_rfcount_dh           == 0)              # Accounting integrity: zero forensic red flags
    )

    # 18. Parikh Contrarian (Parag Parikh — Value Investing and Behavioral Finance)
    #    Graham's quantitative floor + Parag's quality filter + anti-herd behavioral overlay.
    #    PROVENANCE (book audited 2026-06-13): the real book is behavioral/qualitative essays —
    #    it contains NO numbered screen. The "Four-Stage Screen" and every threshold below come
    #    from the companion CODEX (Graham-anchored India-practice constructions), so all numbers
    #    are ENGINEERING PROXIES, not book quotes. Book-verified anchors that DO hold:
    #      - current_ratio > 1.5 is BOOK-EXACT to Graham's enterprising criteria (Intelligent
    #        Investor Ch.15, verified 3rd ed. 2026-06-13) — Parikh channels Graham.
    #      - CFO/PAT gate is directionally the book's own principle ("it is the net cash
    #        generated by the business that determines the returns", VIBF Ch.6) — number is proxy.
    #      - dist_52wh >= 30 proxies the book's contrarian condition "price reflects consensus,
    #        not reality"; the book's actual empirical method was Sensex lowest-PE-decile
    #        portfolios (avg PE 14, Ch.4.3) — relative, not absolute gates.
    #    Stage 2 of the codex screen (intrinsic value via DCF/EPV/Graham Number) is NOT
    #    implemented — requires WACC/maintenance-capex modeling, not automatable from the CSV.
    #    Three signals unique across all 18 frameworks when combined:
    #      1. pe < 20: Graham's absolute PE ceiling — no other framework gates on PE < 20.
    #         Parikh's core: "quality at a FAIR price." Dorsey, Quality Compounder, Diamond all have
    #         no PE ceiling. Magic Formula uses earnings yield but not a direct PE < 20 hard gate.
    #      2. current_ratio > 1.5: Graham's liquidity minimum — stricter than Malik's 1.25.
    #         No other framework in this system uses current_ratio > 1.5 as a hard entry condition.
    #      3. roce_med_5y >= 12: THE ONLY FRAMEWORK IN THIS SYSTEM WITH A 12% ROCE THRESHOLD.
    #         All 17 other frameworks use >= 15% (or require higher). Parikh doesn't require elite
    #         moats — a sustained ROCE of 12-14% at PE < 20 is the Graham "fair business at fair price"
    #         archetype. A stock with ROCE 13% FAILS every single other framework; PASSES here.
    #    de_slope_3y < 0: Parikh Stage 3 "D/E falling trend" — balance sheet actively strengthening.
    #         Outsider CEO also uses this, but combined with CFO/PAT >= 85% and dilution_flag == 0
    #         (creating an entirely different quality profile). Here it pairs with cheap PE < 20.
    #    dist_52wh >= 30: Stage 4 anti-herd overlay — contrarian entry, market is wrong/fearful.
    #         Dhandho also uses >= 30, but requires FCF yield >= 8% and ROCE >= 15% (higher bar).
    #         Parikh's version catches ROCE 12-14% quality companies Dhandho excludes entirely.
    #    promoter >= 35%: Parikh's skin-in-game threshold (lower than Diamond's 40%, unique level).
    #    pledge < 20%: Proxy for "no high/rising pledge" — Parikh requires clean promoter governance.
    #    pe fillna(999): NaN PE = no earnings = loss-maker → 999 < 20 = False (excluded — Graham
    #         explicitly requires positive earnings; loss-makers don't qualify).
    #    de_slope_3y fillna(1): NaN = no trend data/Ind AS restatement → 1 > 0 → excluded
    #         (cannot verify falling D/E = won't award the Parikh badge — conservative).
    #    NOT implemented — P/B < 3 and P/E × P/B < 22.5: COMPUTABLE (raw CSV column
    #    price_to_book is mapped with 100% coverage; derived pb_ratio agrees, median ratio
    #    1.0) but deliberately omitted — those are Intelligent Investor Ch.14/15
    #    criteria (verified against the 3rd ed. 2026-06-13: defensive PE≤15/PB≤1.5/PE×PB≤22.5,
    #    enterprising CR≥1.5); grafting them onto Parikh's screen would break Parikh book
    #    fidelity, and a standalone fw_graham is banned (substrate ruling, Security Analysis
    #    audit). Parikh's Stage 1 numbers descend from Graham's ENTERPRISING criteria, loosened.
    #    NOT implementable: intrinsic value discount 20% (requires DCF/EPV modeling — not automatable),
    #    analyst coverage < 3 (external data not in CSV), sector out-of-favour (qualitative),
    #    Mr. Market Temperature Score (macro data: SIP flows, demat accounts, IPO volumes)
    _pk_nan      = pd.Series(np.nan, index=df.index)
    pe_pk        = df.get("pe",                 pd.Series(999.0, index=df.index)).fillna(999)
    eps5y_pk     = df.get("eps_gr_5y",          _pk_nan)
    cr_pk        = df.get("current_ratio",      _pk_nan)
    mcap_pk      = df.get("market_cap",         _pk_nan)
    roce_5y_pk   = df.get("roce_med_5y",        _pk_nan)
    cfo_pat_pk   = df.get("cfo_to_pat",         _pk_nan)   # PERCENTAGE: 75.0 = 75%, not 0.75
    de_pk        = df.get("debt_to_equity",     _pk_nan)
    de_slope_pk  = df.get("de_slope_3y",        pd.Series(1.0, index=df.index)).fillna(1)
    promo_pk     = df.get("promoter_holdings",  _pk_nan)
    pledge_pk    = df.get("pledged_percentage",  pd.Series(100.0, index=df.index)).fillna(100)
    dist_wh_pk   = df.get("dist_52wh",           pd.Series(0.0, index=df.index)).fillna(0)
    is_fin_pk    = df.get("is_financial",        pd.Series(False, index=df.index)).fillna(False)
    fw_parikh = (
        (pe_pk                      <  20) &              # Graham Stage 1: absolute PE ceiling < 20
        (eps5y_pk.fillna(0)         >   8) &              # Graham Stage 1: EPS 5Y CAGR > 8% (earning power)
        (is_fin_pk | (cr_pk.fillna(0) > 1.5)) &           # Graham Stage 1: current ratio > 1.5 (fin exempt)
        (mcap_pk.fillna(0)          >= 500) &             # Graham Stage 1: ≥ ₹500 Cr proven business
        (roce_5y_pk.fillna(0)       >= 12) &              # Parag Stage 3: ROCE ≥ 12% sustained (UNIQUE threshold)
        (cfo_pat_pk.fillna(0)       >= 75) &              # Parag Stage 3: CFO/PAT ≥ 75% — earnings quality
        (is_fin_pk | (de_pk.fillna(999) <= 0.5)) &        # Graham Stage 1: D/E ≤ 0.5 (fin exempt)
        (de_slope_pk                <   0) &              # Parag Stage 3: D/E falling — balance sheet strengthening
        (promo_pk.fillna(0)         >=  35) &             # Parag Stage 3: promoter ≥ 35% — skin in game
        (pledge_pk                  <  20) &              # Parag Stage 3: pledge < 20% — no high/rising pledge
        (dist_wh_pk                 >=  30)               # Parikh Stage 4: fallen 30%+ — contrarian anti-herd entry
    )

    # 19. Baid Compounder (Gautam Baid — The Joys of Compounding, 2020)
    #    Baid's "Nirvana" = long-term ownership of competitively advantaged businesses with
    #    significant reinvestment potential (p.298, Ch.23 "The Market Is Efficient Most of the Time").
    #    All thresholds are engineering proxies from Ch.14 qualitative checklist (pp.180-193)
    #    — the book provides directional guidance ("higher is better"), NOT explicit percentages.
    #    See docs/baid_financial_specs.json for full audit and Gemini fabrication log.
    #
    #    Engineering proxy rationale:
    #    roce_med_7y >= 15: 7Y window spans full business cycle (more robust than prior 5Y)
    #    rev_gr_10y >= 12:  10Y CAGR aligns with long-horizon compounding philosophy
    #    _bd_each_year:     y2-y5 each >= 5% — no-stumble consistency (mirrors _cc_each_year at 5%)
    #    fcf_yield hurdle:  size-aware (3% large / 4% mid-small) — risk-adjusted margin of safety
    #    PEG 0–1.5:         unique entry corridor — no other framework uses exactly 1.5
    #    cfo_to_pat >= 80:  PERCENTAGE (80.0 = 80%); earnings quality gate
    _bd_nan      = pd.Series(np.nan, index=df.index)
    roce_7y_bd   = df.get("roce_med_7y",     _bd_nan)   # 7Y median — full business cycle window
    rev_10y_bd   = df.get("rev_gr_10y",      _bd_nan)   # 10Y revenue CAGR — long-horizon compounding
    rev_yoy_bd   = df.get("rev_gr_yoy",      _bd_nan)   # current-year floor
    fcf_yield_bd = df.get("fcf_yield",       _bd_nan)   # PERCENTAGE: 3.0 = 3%
    cfo_pat_bd   = df.get("cfo_to_pat",      _bd_nan)   # PERCENTAGE: 80.0 = 80%
    de_bd        = df.get("debt_to_equity",  _bd_nan)
    mcap_bd      = df.get("market_cap",      _bd_nan)
    peg_bd       = df.get("peg",             _bd_nan)
    is_fin_bd    = df.get("is_financial",    pd.Series(False, index=df.index)).fillna(False)
    mcap_tier_bd = df.get("mcap_tier",       pd.Series("", index=df.index)).fillna("")
    # Size-aware FCF yield hurdle: 3% for Mega/Large Cap, 4% for Mid/Small/Micro/Nano
    is_large_bd         = mcap_tier_bd.isin(["Mega Cap", "Large Cap"])
    baid_fcf_yield_hurdle = pd.Series(
        np.where(is_large_bd, 3.0, 4.0), index=df.index
    )
    # Anti-contraction annual guard: each visible back-year >= 0% (no negative growth allowed)
    # Relaxed from >= 5% to >= 0% — pandemic anomaly accommodation; absolute floor is no contraction.
    _bd_y2 = df.get("rev_gr_y2", _bd_nan).fillna(0)
    _bd_y3 = df.get("rev_gr_y3", _bd_nan).fillna(0)
    _bd_y4 = df.get("rev_gr_y4", _bd_nan).fillna(0)
    _bd_y5 = df.get("rev_gr_y5", _bd_nan).fillna(0)
    _bd_each_year = (
        (_bd_y2 >= 0) & (_bd_y3 >= 0) & (_bd_y4 >= 0) & (_bd_y5 >= 0)
    )
    fw_baid = (
        (roce_7y_bd.fillna(0)   >= 15) &              # 7Y ROCE ≥ 15% — sustained capital efficiency over full cycle
        (rev_10y_bd.fillna(0)   >= 12) &              # 10Y revenue CAGR ≥ 12% — long-horizon compounding velocity
        (rev_yoy_bd.fillna(0)   >=  5) &              # current year ≥ 5% — not actively decelerating
        _bd_each_year                  &              # years 2-5 each ≥ 0% — anti-contraction standard
        (fcf_yield_bd.fillna(0) >= baid_fcf_yield_hurdle) &  # size-aware FCF yield (3% large / 4% mid-small)
        (cfo_pat_bd.fillna(0)   >= 80) &              # CFO/PAT ≥ 80% — earnings are real cash (PERCENTAGE)
        (is_fin_bd | (de_bd.fillna(999) < 0.5)) &    # D/E < 0.5 fortress (financials exempt)
        (mcap_bd.fillna(0)      >= 500) &             # ≥ 500 Cr — proven size, avoids micro-cap noise
        (peg_bd.fillna(999)     >   0) &              # PEG > 0 — positive earnings required
        (peg_bd.fillna(999)     <= 2.0)               # PEG ≤ 2.0 — expanded GARP valuation regime
    )
    df["baid_pass"] = fw_baid.astype(int)

    # 20. Long Game Quality (Vishal Khandelwal — The Long Game)
    #    Khandelwal's "fortress compounder": a business that generates REAL free cash after ALL reinvestment.
    #    Two signals unique across all 19 existing frameworks:
    #    1. interest_coverage >= 5 — P1 "fort-like balance sheet": strictest ICR in system (Malik uses 3×)
    #    2. d28_fcf_to_pat_pct >= 60 — FCF AFTER capex as % of PAT; different from CFO/PAT (before capex)
    #       A capital-guzzler can have CFO/PAT=90% yet FCF/PAT=30% (heavy reinvestment bleeds free cash).
    #       No other framework tests FCF/PAT — they test either CFO/PAT or FCF/CFO (Mukherjea).
    #    Differentiation by framework:
    #    - vs Quality Compounder: no NFAT gate (catches capital-intensive but efficient compounders)
    #    - vs Diamond: no FCF/CFO gate; adds ICR≥5 + FCF/PAT≥60 — different ratio structure
    #    - vs Outsider CEO: no dilution gate + adds FCF/PAT≥60 + ICR≥5 instead of de_slope≤0
    #    Sources: Chapter 2 (5P Performance gates), Chapter 5 (valuation toolkit),
    #             Chapter 10 (IPS People checklist — 5 disqualifying conditions).
    _lg_nan       = pd.Series(np.nan, index=df.index)
    roce_10y_lg   = df.get("roce_med_10y",        _lg_nan)
    roce_5y_lg    = df.get("roce_med_5y",         _lg_nan)
    rev_10y_lg    = df.get("rev_gr_10y",          _lg_nan)
    rev_5y_lg     = df.get("rev_gr_5y",           _lg_nan)
    cfo_pat_lg    = df.get("cfo_to_pat",          _lg_nan)   # PERCENTAGE: 80.0 = 80%
    icr_lg        = df.get("interest_coverage",   _lg_nan)   # UNIQUE: fortress ICR ≥ 5× (fin exempt)
    fcf_pat_lg    = df.get("d28_fcf_to_pat_pct",  _lg_nan)   # UNIQUE: FCF/PAT % after capex ≥ 60%
    de_lg         = df.get("debt_to_equity",      _lg_nan)
    promo_lg      = df.get("promoter_holdings",   _lg_nan)
    pledge_lg     = df.get("pledged_percentage",  pd.Series(100.0, index=df.index)).fillna(100)
    mcap_lg       = df.get("market_cap",          _lg_nan)
    is_fin_lg     = df.get("is_financial",        pd.Series(False, index=df.index)).fillna(False)
    _rev_lg       = rev_10y_lg.fillna(rev_5y_lg)   # 10Y primary, 5Y fallback — same as Diamond/UB pattern
    # BUG FIX: Debt-free companies (D/E ≈ 0) have no interest expense → CSV reports NaN coverage.
    # fillna(0) would give 0 >= 5 = False → they fail the ICR gate despite having the STRONGEST
    # possible balance sheet (no debt = infinite interest coverage). Fix: exempt zero-debt companies.
    # de_lg.fillna(1.0): NaN D/E → 1.0 (assume some debt) → NOT zero → not exempt (conservative).
    _no_debt_lg   = de_lg.fillna(1.0) <= 0.01   # True only if D/E is essentially zero (debt-free)
    fw_long_game = (
        (roce_10y_lg.fillna(0)    >= 15) &               # P1: ROCE ≥ 15% over full 10Y economic cycle
        (roce_5y_lg.fillna(0)     >= 15) &               # P1: ROCE not declining in recent window
        (_rev_lg.fillna(0)        >= 10) &               # P1: Revenue CAGR ≥ 10% (10Y primary / 5Y fallback)
        (cfo_pat_lg.fillna(0)     >= 80) &               # P1: CFO/PAT ≥ 80% (PERCENTAGE) — cash earnings standard
        (is_fin_lg | _no_debt_lg | (icr_lg.fillna(0) >= 5)) &  # P1: UNIQUE — ICR≥5 (fin+zero-debt exempt)
        (fcf_pat_lg.fillna(0)     >= 60) &               # P1: UNIQUE — FCF/PAT ≥ 60% after all capex reinvestment
        (is_fin_lg | (de_lg.fillna(999) <= 0.5)) &       # P1: D/E ≤ 0.5 fortress balance sheet (fin exempt)
        (promo_lg.fillna(0)       >= 40) &               # P5: People checklist — promoter ≥ 40% skin in game
        (pledge_lg                <  10) &               # P5: People checklist — pledge < 10% disqualifier
        (mcap_lg.fillna(0)        >= 500)                # Screen: ₹500 Cr proven business scale
    )

    # ── Framework 21: fw_sepa — SEPA Momentum (Mark Minervini) ─────────────────
    # Specific Entry Point Analysis: the ONLY pure technical-momentum framework.
    # Decoupled into 7 observable pillars matching the system's standard pattern.
    # Source: Trade Like a Stock Market Wizard Ch.2 (Trend Template), Ch.4 (Fundamentals),
    #         Ch.5 (VCP). SEPA Trading Codex (India Edition).
    # Spec: docs/sepa_momentum_specs.json
    #
    # sepa_pass  = 6 hard-gate pillars AND mcap >= 500 (stock qualifies for SEPA analysis)
    # sepa_score = 7 pillars (sepa_vcp_dryup is score bonus — NOT hard gate, same as can_slim_vcp)
    # A stock with sepa_score=7 has both the quality AND the active VCP setup firing.
    # A stock with sepa_score=6 qualifies but the VCP isn't formed yet — add to watchlist.
    _sp_nan = pd.Series(np.nan, index=df.index)

    # Pillar T — Trend Template (Criteria 1+2+3+5 + VSTOP, 5-point score ≥ 4)
    _trend_sp = df.get("d45_trend_structure", pd.Series(0, index=df.index)).fillna(0)
    df["sepa_trend_template"]  = (_trend_sp >= 4).astype(int)

    # Pillar A — ADX Confirmed (trend strength gate; removed from d45, explicit here)
    _adx_sp = df.get("adx_14w", pd.Series(0, index=df.index)).fillna(0)
    df["sepa_adx_confirmed"]   = (_adx_sp >= 20).astype(int)

    # Pillar L — Low Base: Criterion 6 — price ≥ 30% above 52-week low
    _52wl_sp = df.get("dist_52wl", pd.Series(0, index=df.index)).fillna(0)
    df["sepa_low_base"]        = (_52wl_sp >= 30).astype(int)

    # Pillar R — RS Aligned: Criterion 8 — all 3 CRS timeframes positive (50D + 26W + 52W)
    df["sepa_rs_confirmed"]    = df.get("crs_aligned", pd.Series(0, index=df.index)).fillna(0).astype(int)

    # Pillar E — Earnings Fuel: REQ 1 (EPS ≥25%) + REQ 2 (Rev ≥20%) + REQ 4 (ROE ≥17%)
    _eps_sp = df.get("eps_gr_yoy", _sp_nan).fillna(0)
    _rev_sp = df.get("rev_gr_yoy", _sp_nan).fillna(0)
    _roe_sp = df.get("roe",        _sp_nan).fillna(0)
    df["sepa_earnings_fuel"]   = ((_eps_sp >= 25) & (_rev_sp >= 20) & (_roe_sp >= 17)).astype(int)

    # Pillar I — Institutional Sponsorship: REQ 7 — FII or DII stake increasing QoQ
    _fii_sp = df.get("change_fii_lq", pd.Series(0, index=df.index)).fillna(0)
    _dii_sp = df.get("change_dii_lq", pd.Series(0, index=df.index)).fillna(0)
    df["sepa_institutional"]   = ((_fii_sp > 0) | (_dii_sp > 0)).astype(int)

    # Pillar V — VCP Volume Dryup (SCORE BONUS ONLY — not in hard gate, same as can_slim_vcp)
    # Source: SEPA Codex Ch.5: "Average volume 10 days < Average volume 50 days"
    # Score 7 = quality + active VCP setup. Score 6 = quality, watchlist for next VCP.
    df["sepa_vcp_dryup"]       = df.get("vcp_volume_dryup", pd.Series(0, index=df.index)).fillna(0).astype(int)

    # Score: 7 pillars including VCP bonus
    df["sepa_score"] = (
        df["sepa_trend_template"] + df["sepa_adx_confirmed"] + df["sepa_low_base"] +
        df["sepa_rs_confirmed"]   + df["sepa_earnings_fuel"] + df["sepa_institutional"] +
        df["sepa_vcp_dryup"]
    )

    # Pass: 6 hard gates (VCP is bonus only) + India market cap floor
    _mcap_sp = df.get("market_cap", _sp_nan).fillna(0)
    # Trend Template Criterion 7 (added 2026-06-12 SEPA Codex audit): price within 25%
    # of the 52-week high (Close ≥ 0.75 × 52WH). The codex mandates ALL 8 criteria —
    # "almost passing = FAIL" — and C7 was the one criterion missing from this gate
    # (d45 covers C1/C2/C3/C5, low_base covers C6, crs_aligned proxies C8; the d45
    # comment claimed C7 was 'checked elsewhere' but nothing in fw_sepa gated it —
    # a stock 40% below its high could pass). fillna(999): missing 52WH distance →
    # C7 unverifiable → conservative fail. Non-pillar hard gate, same as the mcap floor.
    _dist_wh_sp = df.get("dist_52wh", _sp_nan).fillna(999)
    fw_sepa = (
        (df["sepa_trend_template"] == 1) &
        (df["sepa_adx_confirmed"]  == 1) &
        (df["sepa_low_base"]       == 1) &
        (df["sepa_rs_confirmed"]   == 1) &
        (df["sepa_earnings_fuel"]  == 1) &
        (df["sepa_institutional"]  == 1) &
        (_dist_wh_sp <= 25) &
        (_mcap_sp >= 500)
    )
    df["sepa_pass"] = fw_sepa.astype(int)

    # ── Framework 22: fw_basant — Basant 30% Club (Basant Maheshwari, "The Thoughtful Investor") ──
    # Maheshwari's "30% Club" thesis: companies sustaining 30% EPS CAGR re-rate massively as
    # the market prices in sustained growth. Three signals unique in the system:
    #   1. eps_gr_5y >= 30: no framework requires 30% EPS CAGR (SMILE=20%, Lynch=20% YoY only)
    #   2. promoter_holdings >= 55: strictest promoter threshold (100-Bagger=50%, Lynch=45%)
    #   3. market_cap <= 5000: max-cap sweet spot — Basant's "reasonable size" for multibaggers
    # Sources: Chapter 3 (30% Club criteria), Chapter 5 (3P Framework), Chapter 7 (Screener filters).
    _bs_nan      = pd.Series(np.nan, index=df.index)
    eps_5y_bs    = df.get("eps_gr_5y",           _bs_nan)   # UNIQUE: 30% EPS CAGR — highest bar in system
    eps_yoy_bs   = df.get("eps_gr_yoy",          _bs_nan)   # No-stumble proxy: current year still ≥ 20%
    cfo_pat_bs   = df.get("cfo_to_pat",          _bs_nan)   # PERCENTAGE: 75.0 = 75%
    roce_bs      = df.get("roce",                _bs_nan)   # Return quality gate
    de_bs        = df.get("debt_to_equity",      _bs_nan)
    is_fin_bs    = df.get("is_financial",        pd.Series(False, index=df.index)).fillna(False)
    promo_bs     = df.get("promoter_holdings",   _bs_nan)   # UNIQUE: ≥ 55% strictest in system
    peg_bs       = df.get("peg",                 _bs_nan)
    mcap_bs      = df.get("market_cap",          _bs_nan)
    fw_basant = (
        (eps_5y_bs.fillna(0)    >= 30) &             # UNIQUE: 30% Club EPS CAGR — highest growth bar in system
        (eps_yoy_bs.fillna(0)   >= 20) &             # No-stumble proxy: current year must still hit ≥ 20%
        (cfo_pat_bs.fillna(0)   >= 75) &             # Cash earnings quality: CFO ≥ 75% of PAT (PERCENTAGE)
        (roce_bs.fillna(0)      >= 20) &             # Return quality: ROCE ≥ 20%
        (is_fin_bs | (de_bs.fillna(999) <= 0.5)) &  # Debt discipline (financial sector exempt)
        (promo_bs.fillna(0)     >= 55) &             # UNIQUE: highest promoter conviction threshold in system
        (peg_bs.fillna(999)     >    0) &            # PEG must be positive (real growth, not distorted)
        (peg_bs.fillna(999)     <= 1.5) &            # Zone 1-2: growth still reasonably priced
        (mcap_bs.fillna(0)      >= 500) &            # Minimum size for institutional liquidity
        (mcap_bs.fillna(0)      <= 5000)             # UNIQUE: max cap sweet spot — multibagger range
    )

    # ── Framework 23: fw_qmom — Quality Momentum (Gray & Vogel QMOM, India Edition) ──
    # The ONLY pure-momentum-first framework in the system. All 22 prior frameworks are
    # fundamentals-led (quality/value/growth required, momentum is secondary confirmation).
    # QMOM flips the hierarchy: top-decile RS rank is the PRIMARY gate; quality is the filter
    # to EXCLUDE junk-momentum and crash risk — NOT a growth prerequisite.
    # Two signals unique in the system:
    #   1. rs_pctrank_qm >= 80 as PRIMARY gate: RS composite in top 20% of all 2108 stocks
    #      (CAN SLIM also uses rs_pctrank >= 80, but CAN SLIM requires EPS 25%+, quarterly
    #       growth, near-52WH, volume surge, ROE 17% — fundamentals-first. Here RS is the door.)
    #   2. pledged_percentage <= 30: ENGINEERING ADAPTATION (India), NOT from Gray's book.
    #      Gray (US-focused) has no pledge concept; this defends against operator-driven discrete
    #      spikes that fake momentum. The companion India codex carried it as "Ch.7" — FALSE:
    #      Gray's Ch.7 is Seasonality. Threshold is a defensible India proxy, not a book quote.
    # BOOK PROVENANCE (audited 2026-06-13 against the real Gray & Vogel text):
    #   - Gray's exact 5-step QMOM recipe (Ch.8): universe → top 10% generic momentum (100 of
    #     1000) → rank those on FIP, keep smoothest top 50% (~50 stocks) → equal-weight →
    #     rebalance end of Feb/May/Aug/Nov. Net selectivity ≈ 5% of universe.
    #   - OUR gate uses top-QUINTILE RS (rs_pctrank >= 80 = top 20%, LOOSER than Gray's decile)
    #     but tightens with the ROCE/CFO/D-E/pledge quality filters to 76 passers ≈ 3.6% — close
    #     to Gray's final 5% by a different path (quality filter substitutes for the FIP cut).
    #   - Generic momentum = "2-12 momentum" (cumulative 12-mo return SKIPPING the most recent
    #     month to dodge short-term reversal), top decile — Gray Ch.5 (NOT Ch.2).
    #   - "Frog-in-the-pan" path quality = Information Discreteness ID = sign(PRET)x(%neg-%pos)
    #     over daily returns; CONTINUOUS (smooth, many small up-days) beats DISCRETE (jumpy) by
    #     ~10.8% 3-factor alpha (Da-Gurun-Warachka, Table 6.3) — Gray Ch.6 (NOT Ch.3).
    #   - Quality is NOT a Gray prerequisite: Ch.4 argues value & momentum belong in SEPARATE
    #     sleeves combined at portfolio level. Our ROCE/CFO/D-E filter is an India junk-momentum
    #     defense (engineering choice), faithful to QMOM's spirit of excluding crash-prone movers.
    # What CANNOT be implemented (genuine data gaps, CSV has no monthly/daily return series):
    #   - 2-12 skip-month momentum (needs monthly price history) → crs_52w/RS composite is the proxy
    #   - FIP / Information Discreteness score (needs ~252 daily returns for %pos vs %neg) — same
    #     daily-OHLCV gap as the Weis/Wyckoff audit; d51_qmom_quality_score is a FUNDAMENTAL
    #     quality composite (ROCE+D/E+CFO ranks), NOT a path-quality/FIP proxy — display only.
    #   - Seasonality timing & monthly rebalancing (Ch.7) — a static screener can't time months.
    _qm_nan       = pd.Series(np.nan, index=df.index)
    rs_comp_qm    = df.get("d47_rs_composite",   pd.Series(0.0, index=df.index)).fillna(0)
    crs_ali_qm    = df.get("crs_aligned",         pd.Series(0, index=df.index)).fillna(0)
    roce_qm       = df.get("roce",                _qm_nan)
    de_qm         = df.get("debt_to_equity",      _qm_nan)
    is_fin_qm     = df.get("is_financial",        pd.Series(False, index=df.index)).fillna(False)
    cfo_pat_qm    = df.get("cfo_to_pat",          _qm_nan)   # PERCENTAGE: 70.0 = 70%
    pledge_qm     = df.get("pledged_percentage",  pd.Series(100.0, index=df.index)).fillna(100.0)
    mcap_qm       = df.get("market_cap",          _qm_nan)
    # RS percentile rank: ascending=True → higher RS composite → higher rank → >= 80 = top 20%
    rs_pctrank_qm = _pct_rank(rs_comp_qm, ascending=True).fillna(50)
    fw_qmom = (
        (rs_pctrank_qm          >= 80) &         # UNIQUE PRIMARY GATE: top 20% RS momentum leadership
        (crs_ali_qm             == 1) &          # All 3 CRS timeframes positive (smooth persistence proxy)
        (roce_qm.fillna(0)      >= 15) &         # Quality: ROCE ≥ 15% (proxy for ROIC > 15%)
        (is_fin_qm | (de_qm.fillna(999) < 0.5)) & # Balance sheet: D/E < 0.5 (financials exempt)
        (cfo_pat_qm.fillna(0)   >= 70) &         # Earnings quality: CFO/PAT ≥ 70% (PERCENTAGE)
        (pledge_qm              <= 30) &          # UNIQUE: India operator manipulation threshold
        (mcap_qm.fillna(0)      >= 500)          # Minimum liquidity: ₹500 Cr
    )

    # ── Framework 24: fw_mosl_wealth_creator — MOSL Wealth Creator (30 Annual Wealth Creation Studies) ──
    # Synthesises the 30 years of MOSL research into a single screen.
    # Three signals genuinely unique in the system:
    #   1. compound_growth_power_flag: PAT CAGR consistent across ALL 3 timeframes (3Y≥15%, 5Y≥12%, 10Y≥10%)
    #      — no other framework simultaneously checks 3 PAT timeframes. Proxy for MOSL's "Consistent" category.
    #   2. payback_lt2: payback ratio < 2 as a hard gate — 5th Study: payback 1-2 band = 37.4% annual returns
    #      — no other framework gates on payback_lt2 specifically.
    #   3. economic_profit_spread >= 10: ROCE ≥ 20% above cost of equity (CoE = 10%) = substantial moat
    #      — all other ROCE thresholds are absolute; this uses the margin-above-CoE framing.
    # These three together catch sustained, cash-backed compounders at reasonable prices — the MOSL DNA.
    _mw_nan = pd.Series(np.nan, index=df.index)
    cgp_mw      = df.get("compound_growth_power_flag", pd.Series(0, index=df.index)).fillna(0)
    paylt2_mw   = df.get("payback_lt2",               pd.Series(0, index=df.index)).fillna(0)
    ep_mw       = df.get("economic_profit_spread",     _mw_nan)   # ROCE - 10 (cost of equity)
    cfo_pat_mw  = df.get("cfo_to_pat",                _mw_nan)   # PERCENTAGE: 70.0 = 70%
    de_mw       = df.get("debt_to_equity",             _mw_nan)
    is_fin_mw   = df.get("is_financial",               pd.Series(False, index=df.index)).fillna(False)
    mcap_mw     = df.get("market_cap",                 _mw_nan)
    fw_mosl_wealth_creator = (
        (cgp_mw              == 1) &              # UNIQUE: PAT consistent across 3Y + 5Y + 10Y simultaneously
        (paylt2_mw           == 1) &              # UNIQUE: payback ratio < 2 as gate (MOSL 5th Study)
        (ep_mw.fillna(-99)   >= 10) &            # UNIQUE: ROCE ≥ 20% above CoE (substantial competitive moat)
        (cfo_pat_mw.fillna(0) >= 70) &           # Cash quality: CFO/PAT ≥ 70% (PERCENTAGE unit)
        (is_fin_mw | (de_mw.fillna(999) < 1.0)) & # Balance sheet: D/E < 1.0 (financials exempt)
        (mcap_mw.fillna(0)   >= 500)             # Scale: ₹500 Cr minimum
    )

    # ── Framework 25: fw_emc — Economic Moat Company (MOSL 17th Study, 2012 theme) ──
    # Study 17 backtested 1995-2012: EMC portfolio → 25% CAGR vs 12% for non-EMCs (Alpha +7%/yr).
    # EMC = ROE consistently above sector average (proxy: above sector median for both current + 5yr).
    # Adds sector-relative moat signal not present in any other framework.
    fw_emc = df.get("emc_flag", pd.Series(0, index=df.index)).fillna(0) == 1

    # ── Framework 26: fw_blue_chip — Blue Chip Quality (MOSL 16th Study, 2011 theme) ──
    # "Blue chips offer as much investment growth potential as lesser quality, with far less risk."
    # Blue chip criterion: consistent dividend + decade ROE > 20% + PAT no-crash consistency.
    fw_blue_chip = df.get("blue_chip_quality_flag", pd.Series(0, index=df.index)).fillna(0) == 1

    # ── Framework 27: fw_sqglp — SQGLP Century Stock (MOSL 19th Study, 2014 theme) ──
    # "100x requires vision to see, courage to buy, and patience to hold." — Thomas Phelps
    # SQGLP = Size (small) + Quality + Growth + Longevity + Price (favorable).
    # Score >= 4/5: highest-probability 100x candidate profile.
    fw_sqglp = df.get("century_stock_flag", pd.Series(0, index=df.index)).fillna(0) == 1

    # ── Framework 35: fw_mosl_100x — 100x Candidate (MOSL 17th Study, 2012 theme) ──
    # "Mouse to Elephant": all 5 mandatory conditions from the 17th Study must hold.
    # (1) PAT CAGR 5Y ≥ 20%  (2) ROCE ≥ 20%  (3) Market Cap ≤ ₹15,000 Cr
    # (4) D/E < 0.5  (5) ROE ≥ 15% — clean, unlevered compounding engine at early stage.
    fw_mosl_100x = df.get("mosl_100x_candidate", pd.Series(0, index=df.index)).fillna(0) == 1

    # ── Framework 28: fw_cap_gap — CAP + GAP Longevity Compounder (MOSL 22nd Study, 2017 theme) ──
    # Competitive Advantage Period (CAP) + Growth Advantage Period (GAP) both extended.
    # Proves DURATION of competitive advantage — not just current level.
    # Most frameworks test current ROCE/growth; CAP-GAP tests whether it has LASTED.
    # Backtested signal: companies with both extended CAP + GAP show the deepest compounding.
    fw_cap_gap = (
        (df.get("cap_extended_flag", pd.Series(0, index=df.index)).fillna(0) == 1) &
        (df.get("gap_extended_flag", pd.Series(0, index=df.index)).fillna(0) == 1)
    )

    # ── Framework 29: fw_consistent_volatile — Consistent in Volatile Sector (MOSL 27th Study, 2022) ──
    # Study 27 backtested 697 companies over 2007-2022 (15yr). Consistent companies in Volatile sectors
    # delivered 19% avg CAGR — the single highest return combination in the entire study.
    # Signal: deepest moat when a company sustains earnings consistency despite adverse sector dynamics.
    fw_consistent_volatile = (
        df.get("consistent_in_volatile_flag", pd.Series(0, index=df.index)).fillna(0) == 1
    )

    # ── Framework 30: fw_ep_hockey_stick — EP Hockey Stick (28th WCS, 2023) ──
    fw_ep_hockey_stick = (
        df.get("ep_hockey_stick", pd.Series(0, index=df.index)).fillna(0) == 1
    )

    # ── Framework 31: fw_bruised_bb_29 — Bruised Blue Chip WCS 29 (Large-Cap Edition, 2024) ──
    fw_bruised_bb_29 = (
        df.get("bruised_blue_chip_29", pd.Series(0, index=df.index)).fillna(0) == 1
    )

    # ── Framework 32: fw_multitrillioncap — Multi-Trillion Compounding Engine (30th WCS, 2025) ──
    fw_multitrillioncap = (
        df.get("multitrillioncap_tipping_point", pd.Series(0, index=df.index)).fillna(0) == 1
    )

    # ── Framework 33: fw_fisher_scalability — Fisher Operating Leverage & Scalability (Philip Fisher) ──
    # Vectorized proxies for Fisher's 4 most measurable scalability criteria from
    # "Common Stocks and Uncommon Profits." Fisher's framework is 90% qualitative
    # (scuttlebutt, management DNA) — these are the 4 points with reliable CSV proxies.
    # Distinct from fw_fisher (Fisher Quality, Framework 11):
    #   Fisher Quality: forensic integrity + NPM + CFO/PAT + zero dilution_flag (7 gates)
    #   Fisher Scalability: operating leverage inflection + revenue runway + pricing power + capital discipline (4 gates)
    # Output columns: fisher_pass (1/0) and fisher_score (0-4, one per sub-gate).
    # Spec ledger: docs/fisher_quality_specs.json
    # NaN strategy: revenue/oplev fillna(0) → gate fails if data absent; dilution fillna(999) → conservative exclusion.
    _fs_nan      = pd.Series(np.nan, index=df.index)
    rev_3y_fs    = df.get("rev_gr_3y",            _fs_nan)   # Revenue Growth 3Y CAGR (%)
    rev_yoy_fs   = df.get("rev_gr_yoy",           _fs_nan)   # Revenue Growth YoY (%)
    oplev_fs     = df.get("d05_rev_minus_exp_gr",  _fs_nan)   # Rev − Expense growth delta (%)
    opm_acc_fs   = df.get("opm_acceleration",      _fs_nan)   # OPM vs 1Y back (margin pts)
    npm_acc_fs   = df.get("npm_acceleration",      _fs_nan)   # NPM vs 1Y back (margin pts)
    dilut_pct_fs = df.get("dilution_pct",          _fs_nan)   # Share count growth YoY (%)
    # 4 sub-gate boolean masks (each 0/1 → fisher_score 0-4)
    _fs_rev_runway  = (rev_3y_fs.fillna(0) >= 12.0) & (rev_yoy_fs.fillna(0) >= 10.0)  # Points 1 & 2: multi-year + current demand runway
    _fs_op_lev      = oplev_fs.fillna(0) >= 2.0                                         # Points 5 & 6: revenue outpacing overhead (2pp threshold)
    _fs_pricing     = (opm_acc_fs.fillna(0) >= 0) | (npm_acc_fs.fillna(0) >= 0)         # Point 4: OR logic — either margin not declining
    _fs_anti_dilut  = dilut_pct_fs.fillna(999) <= 1.0                                   # Point 13: ≤1% annual share dilution (999 = missing = excluded)
    fw_fisher_scalability = _fs_rev_runway & _fs_op_lev & _fs_pricing & _fs_anti_dilut
    df["fisher_pass"]  = fw_fisher_scalability.astype(int)
    df["fisher_score"] = (
        _fs_rev_runway.astype(int) +
        _fs_op_lev.astype(int)     +
        _fs_pricing.astype(int)    +
        _fs_anti_dilut.astype(int)
    )
    # Fisher Quality pass column — mirrors fisher_pass symmetry for cross-strategy filtering.
    # fw_fisher is computed earlier (Framework 11); safe to reference here.
    # Pattern: consistent with ub_pass, baid_pass, diamonds_pass, hundred_bagger_pass.
    df["fisher_quality_pass"] = fw_fisher.astype(int)
    # 4-Quadrant Fisher Lifecycle Matrix — fully vectorized np.select, zero loops.
    # Classifies every stock by its combined Fisher dual-engine state.
    # ⚪ Laggard:          neither quality nor scalability gate clears
    # ⚡ Catalyst Play:    scalability inflection firing but structural quality absent (trading candidate)
    # 🐢 Steady Compounder: quality proven, no current inflection (steady-state long-term hold)
    # 👑 Apex Winner:      elite quality business at its operating leverage peak (prime entry)
    _lc_conds = [
        (df["fisher_quality_pass"] == 0) & (df["fisher_pass"] == 0),
        (df["fisher_quality_pass"] == 0) & (df["fisher_pass"] == 1),
        (df["fisher_quality_pass"] == 1) & (df["fisher_pass"] == 0),
        (df["fisher_quality_pass"] == 1) & (df["fisher_pass"] == 1),
    ]
    _lc_labels = ["⚪ Laggard", "⚡ Catalyst Play", "🐢 Steady Compounder", "👑 Apex Winner"]
    df["fisher_lifecycle_quadrant"] = np.select(_lc_conds, _lc_labels, default="⚪ Laggard")

    # Schilit Financial Shenanigans clean-pass flag (pre-computed in forensic_engine)
    fw_schilit = df.get("schilit_pass", pd.Series(0, index=df.index)).fillna(0).astype(bool)

    # ── Marks Market Cycle & Risk Defensive Shield (Howard Marks) ─────────────
    # 4-Pillar Materialized Architecture — each pillar is an independent observable column.
    # Source: docs/marks_cycle_specs.json
    #
    #    Pillar M (Margin Spike Guard): mean_reversion_risk == 0
    #      Marks: "Extremes revert toward average." (Ch.4 MEAN_REVERSION config)
    #      OPM or NPM >130% of 5Y median = cyclical peak risk (pendulum at greed extreme).
    #      fillna(0) = missing spike data → 0 → no spike confirmed → gate passes.
    #
    #    Pillar P (Price vs Value): buy_zone_label == "🟢 Perfect Entry (Low Risk)"
    #      dist_to_vstop <= 5% = maximum asymmetric risk/reward entry zone.
    #      Marks: "The ONLY time to act decisively is when price is below intrinsic value."
    #      fillna('') = missing label → fails (cannot confirm in buy zone).
    #
    #    Pillar L (Leverage Discipline): debt_to_equity < 0.5
    #      Marks: "Borrowed money amplifies ALL other risks." (Ch.2 Risk Type 4)
    #      Companion Ch.9 Pillar 1: "D/E < 0.5" — India-adapted Marks defensive floor.
    #      fillna(999) = missing D/E → 999 ≥ 0.5 → fails (leverage unconfirmed).
    #
    #    Pillar D (Defensive Cash): cfo_to_pat >= 80.0
    #      Marks: "Only companies generating real cash survive cycle downturns." (Ch.9 Pillar 1)
    #      Companion Ch.9 Pillar 1: "CFO/PAT > 0.8" = 80% in percentage-unit CSV data.
    #      cfo_to_pat is PERCENTAGE in CSV (73.04 = 73%). Threshold 80.0 not 0.80.
    #      fillna(0) = missing CFO/PAT → 0 < 80 → fails.
    #
    #    NOT implementable: 5-Dimension Market Temperature Score (macro data not in CSV),
    #    Second-Level Thinking score, psychological enemy detection, concentration risk,
    #    liquidity risk, contrarian opportunity score — see docs/marks_cycle_specs.json.

    # ── Pillar Materialization (vectorized, zero loops) ────────────────────────
    df["marks_margin_spike"] = (
        df.get("mean_reversion_risk", pd.Series(0, index=df.index)).fillna(0) == 0
    ).astype(int)

    df["marks_price_value"] = (
        df.get("buy_zone_label", pd.Series("", index=df.index)).fillna("") == "🟢 Perfect Entry (Low Risk)"
    ).astype(int)

    df["marks_leverage_trap"] = (
        df.get("debt_to_equity", pd.Series(999, index=df.index)).fillna(999) < 0.5
    ).astype(int)

    df["marks_defensive_base"] = (
        df.get("cfo_to_pat", pd.Series(0, index=df.index)).fillna(0) >= 80.0
    ).astype(int)

    # ── Pass flag: AND of all 4 pillars ────────────────────────────────────────
    fw_marks_cycle = (
        (df["marks_margin_spike"]   == 1) &
        (df["marks_price_value"]    == 1) &
        (df["marks_leverage_trap"]  == 1) &
        (df["marks_defensive_base"] == 1)
    )
    df["marks_pass"]  = fw_marks_cycle.astype(int)
    df["marks_score"] = (
        df["marks_margin_spike"]   +
        df["marks_price_value"]    +
        df["marks_leverage_trap"]  +
        df["marks_defensive_base"]
    )  # 0-4 sub-gate count; enables partial-pass ranking (3/4 = one cycle gate blocking)

    # 34. Mauboussin Expectations Investing Framework
    # Spec Reference: docs/mauboussin_expectations_specs.json v1.1-fixed-nopat-precision
    # PIE Framework: read what current price implies about competitive advantage duration (CAP).
    # Alpha = identifying where implied CAP is too optimistic vs business fundamentals.
    _pe_ly   = df.get("pe",  pd.Series(0.0, index=df.index)).fillna(0.0)
    _rr_ly   = df.get("reinvestment_rate", pd.Series(0.0, index=df.index)).fillna(0.0)
    # reinvestment_rate is decimal [0,1] — no /100 (v1.0 bug: was dividing again, killing the signal)

    # NOPAT Margin: EBIT × (1 − eff_tax) / Revenue × 100 — capital-structure-neutral profitability
    _pbt_v   = df.get("pbt",     pd.Series(np.nan, index=df.index))
    _pat_v   = df.get("pat",     pd.Series(0.0,    index=df.index)).fillna(0.0)
    _ebit_v  = df.get("ebit",    pd.Series(0.0,    index=df.index)).fillna(0.0)
    _rev_v   = df.get("revenue", pd.Series(np.nan, index=df.index))
    _eff_tax = ((_pbt_v.fillna(0) - _pat_v) / _pbt_v.replace(0, np.nan)).clip(0, 0.50).fillna(0.25)
    df["mauboussin_nopat_margin"] = np.where(
        _rev_v.fillna(0) > 0,
        _ebit_v * (1 - _eff_tax) / _rev_v * 100,
        np.nan
    )
    _nopat_m_ly = df["mauboussin_nopat_margin"].fillna(0.0) / 100.0

    # Layer 1: Implied CAP (legacy discriminant — empirically calibrated, not a true identity)
    df["mauboussin_implied_cap"] = _pe_ly * _nopat_m_ly * _rr_ly

    # FGV% (Stern-Stewart): fraction of market cap that is a future-growth promise.
    # SSV = NOPAT / CoE — steady-state perpetuity value assuming zero growth.
    # FGV% < 0: priced below no-growth value (Dhandho deep value).
    # FGV% > 0.6: 60%+ of market cap is a growth promise — high expectations risk.
    _rev_fgv   = df.get("revenue",    pd.Series(np.nan, index=df.index))
    _mcap_fgv  = df.get("market_cap", pd.Series(np.nan, index=df.index))
    _nopat_abs = _nopat_m_ly * _rev_fgv
    _ssv = pd.Series(
        np.where(
            _nopat_abs.notna() & (_nopat_abs > 0),
            _nopat_abs / (COST_OF_EQUITY / 100.0),
            np.nan
        ),
        index=df.index
    )
    df["fgv_pct"] = np.where(
        _mcap_fgv.notna() & (_mcap_fgv > 0) & _ssv.notna(),
        1.0 - _ssv / _mcap_fgv,
        np.nan
    )

    # Pillar T: Treadmill breach (1 = safe; treadmill sell alert not firing)
    df["mauboussin_treadmill_breach"] = np.where(
        df.get("sell_alert_treadmill", pd.Series(0, index=df.index)).fillna(0) == 1, 0, 1
    )

    # Pillar O: Operating leverage drift (1 = revenue converting efficiently to profit)
    df["mauboussin_oplev_drift"] = np.where(
        df.get("operating_leverage", pd.Series(1, index=df.index)).fillna(1) == 0, 0, 1
    )

    # Layer 2: CAP trap — structural 3-year slope replaces volatile 1-year delta (v1.0 noise bug)
    # Fallback chain: roce_med_3y (3Y median, smooth) → roce_2yb (point 2Y ago) → roce (slope=0)
    # roce_med_3y is preferred but not yet in CSV export; roce_2yb activates the signal immediately.
    _roce_v      = df.get("roce",        pd.Series(np.nan, index=df.index)).fillna(0.0)
    _roce_med3y  = (
        df.get("roce_med_3y", pd.Series(np.nan, index=df.index))
        .fillna(df.get("roce_2yb", pd.Series(np.nan, index=df.index)))
        .fillna(_roce_v)
    )
    _roce_slope_3y = (_roce_v - _roce_med3y) / 2.0
    df["mauboussin_cap_trap"] = (
        (df["mauboussin_implied_cap"] > 15.0) & (_roce_slope_3y < -1.0)
    ).astype(int)

    # Strategy pass flag: both operational gates clear AND not caught in expectations trap
    fw_mauboussin = (
        (df["mauboussin_treadmill_breach"] == 1) &
        (df["mauboussin_oplev_drift"]      == 1) &
        (df["mauboussin_cap_trap"]         == 0)
    )
    df["mauboussin_pass"]  = fw_mauboussin.astype(int)
    df["mauboussin_score"] = (
        df["mauboussin_treadmill_breach"] +
        df["mauboussin_oplev_drift"] +
        (df["mauboussin_cap_trap"] == 0).astype(int)
    )  # 0-3; score==3 ↔ pass==1 (bidirectional — no asymmetric disqualifier)

    # Build comma-separated framework string — fully vectorized, zero apply
    fw_str = (
        np.where(fw_qglp,                   "QGLP|",                       "") +
        np.where(fw_coffee_can,             "Coffee Can|",                 "") +
        np.where(fw_magic_formula,          "Magic Formula|",              "") +
        np.where(fw_smile,                  "SMILE|",                      "") +
        np.where(fw_lynch,                  "Lynch Dream|",                "") +
        np.where(fw_can_slim,               "CAN SLIM|",                   "") +
        np.where(fw_bruised_bb,             "Fallen Quality|",             "") +
        np.where(fw_ep_improver,            "EP Improver|",                "") +
        np.where(fw_malik_peaceful,         "Peaceful Investing|",         "") +
        np.where(fw_unusual_billionaires,   "Unusual Billionaires|",       "") +
        np.where(fw_fisher,                 "Fisher Quality|",             "") +
        np.where(fw_100_bagger,             "100-Bagger|",                 "") +
        np.where(fw_diamond,                "Diamond|",                    "") +
        np.where(fw_dorsey,                 "Wide Moat|",                  "") +
        np.where(fw_outsider,               "Outsider CEO|",               "") +
        np.where(fw_quality,                "Quality Compounder|",         "") +
        np.where(fw_dhandho,                "Dhandho Asymmetry|",          "") +
        np.where(fw_parikh,                 "Parikh Contrarian|",          "") +
        np.where(fw_baid,                   "Baid Compounder|",            "") +
        np.where(fw_long_game,              "Long Game Quality|",          "") +
        np.where(fw_sepa,                   "SEPA Momentum|",              "") +
        np.where(fw_basant,                 "Basant 30% Club|",            "") +
        np.where(fw_qmom,                   "Quality Momentum|",           "") +
        np.where(fw_mosl_wealth_creator,    "MOSL Wealth Creator|",        "") +
        np.where(fw_emc,                    "Economic Moat|",              "") +
        np.where(fw_blue_chip,              "Blue Chip Quality|",          "") +
        np.where(fw_sqglp,                  "SQGLP Century Stock|",        "") +
        np.where(fw_cap_gap,                "CAP-GAP Compounder|",         "") +
        np.where(fw_consistent_volatile,    "Consistent in Volatile|",     "") +
        np.where(fw_ep_hockey_stick,        "EP Hockey Stick|",            "") +
        np.where(fw_bruised_bb_29,          "Bruised Blue Chip 29|",       "") +
        np.where(fw_multitrillioncap,       "Multi-Trillion Cap|",         "") +
        np.where(fw_fisher_scalability,    "Fisher Scalability|",          "") +
        np.where(fw_schilit,               "Financial Shenanigans|",       "") +
        np.where(fw_marks_cycle,           "Marks Cycle Shield|",           "") +
        np.where(fw_mauboussin,            "Expectations Matrix|",          "") +
        np.where(fw_mosl_100x,             "100x Candidate|",               "")
    )
    df["frameworks_passed"] = (
        pd.Series(fw_str, index=df.index)
        .str.rstrip("|")
        .str.replace("|", ", ", regex=False)
    )
    df["frameworks_passed"] = np.where(df["frameworks_passed"] == "", "None", df["frameworks_passed"])
    # corporate_class is computed in run_full_scoring (before quality_score penalties are applied).
    # Do NOT recompute here — that would overwrite the label without reapplying penalties.

    return df


# ═══════════════════════════════════════════════════════════════
# 28th WCS: ECONOMIC PROFIT POWER CURVE (MOSL 2023)
# ═══════════════════════════════════════════════════════════════

def compute_ep_power_curve(df: pd.DataFrame) -> pd.DataFrame:
    """28th WCS Power Curve: cross-sectional EP quintile matrix + Hockey-Stick breakout flag.

    Quintile 1 = Top 20% absolute value creators (highest EP).
    Quintile 5 = Bottom 20% absolute value destroyers (lowest EP).

    Hockey-Stick Breakout (ep_hockey_stick_breakout = 1) requires all three concurrent:
      1. EP quintile ∈ {2, 3} — emerging value generator, not yet priced as elite
      2. economic_profit_velocity > 0 — ascending the Power Curve (improving EP YoY)
      3. vol_ratio > 1.0 — volume above 20D SMA, confirming institutional accumulation
    """
    df = df.copy()

    # ── Cross-Sectional EP Quintile Matrix ──
    # Preserve data_engine's rank-based ep_quintile if already computed; only recompute
    # when running on synthetic frames (e.g., unit tests) that bypassed data_engine.
    _ep_quintile_set = "ep_quintile" in df.columns and not df["ep_quintile"].isna().all()
    if not _ep_quintile_set:
        df["ep_quintile"] = pd.Series(np.nan, index=df.index)
    if not _ep_quintile_set and "economic_profit" in df.columns:
        ep_valid = df["economic_profit"].notna()
        if ep_valid.sum() >= 10:
            try:
                quintile_labels = pd.qcut(
                    df.loc[ep_valid, "economic_profit"],
                    q=5,
                    labels=[5, 4, 3, 2, 1],  # 1=top value creators, 5=bottom destroyers
                    duplicates="drop"
                )
                df.loc[ep_valid, "ep_quintile"] = quintile_labels.astype(float)
            except Exception:
                # Fallback: rank-based quintile if qcut fails (e.g. many ties)
                ep_rank_pct = df["economic_profit"].rank(pct=True, na_option="keep")
                df["ep_quintile"] = pd.cut(
                    ep_rank_pct,
                    bins=[0, 0.20, 0.40, 0.60, 0.80, 1.0],
                    labels=[5, 4, 3, 2, 1],
                    include_lowest=True
                ).astype(float)

    # ── Hockey-Stick Breakout: emerging EP generators + velocity + volume confirmation ──
    _ep_q   = df["ep_quintile"].fillna(99)
    _ep_vel = df.get("economic_profit_velocity", pd.Series(0.0, index=df.index)).fillna(0)
    _vol_r  = df.get("vol_ratio", pd.Series(0.0, index=df.index)).fillna(0)

    df["ep_hockey_stick_breakout"] = (
        (_ep_q.isin([2.0, 3.0])) &   # Quintile 2 or 3: emerging value generators
        (_ep_vel > 0) &               # Velocity positive: ascending the Power Curve
        (_vol_r  > 1.0)              # Volume above 20D SMA: institutional accumulation
    ).astype(int)

    breakout_count = int(df["ep_hockey_stick_breakout"].sum())
    print(f"\n📈 EP Power Curve: {int((df['ep_quintile'] == 1.0).sum())} Q1 value creators | "
          f"{breakout_count} Hockey-Stick breakout setups")

    return df


# ═══════════════════════════════════════════════════════════════
# WAVE DETECTION: MARKET REGIME AWARENESS
# ═══════════════════════════════════════════════════════════════

def detect_market_regime(df: pd.DataFrame) -> str:
    """Auto-detect market regime from multi-signal breadth consensus (2/3 required).
    Single CRS-50D can flip regime on one session's noise — consensus prevents whipsawing."""
    bull_signals = 0
    bear_signals = 0

    if "crs_50d" in df.columns:
        b50 = (df["crs_50d"] > 0).mean()
        if b50 > 0.60: bull_signals += 1
        elif b50 < 0.40: bear_signals += 1

    if "crs_26w" in df.columns:
        b26 = (df["crs_26w"] > 0).mean()
        if b26 > 0.55: bull_signals += 1
        elif b26 < 0.45: bear_signals += 1

    if "above_sma200" in df.columns:
        b200 = df["above_sma200"].fillna(0).mean()
        if b200 > 0.55: bull_signals += 1
        elif b200 < 0.45: bear_signals += 1

    if bull_signals >= 2:
        return "BULL"
    if bear_signals >= 2:
        return "BEAR"
    return "SIDEWAYS"


def run_full_scoring(
    df: pd.DataFrame,
    analysis_mode: str = "Hybrid",
    scoring_profile: str = "Balanced"
) -> pd.DataFrame:
    """Execute the complete 4-layer adaptive scoring pipeline.
    
    Architecture:
      1. Hard Gates (binary pass/fail)
      2. Quality + Momentum sub-scores (0-100)
      3. Regime detection → get_adaptive_weights(profile, regime)
      4. Composite blend using Analysis Mode + regime-adjusted momentum boost
    """
    df = df.copy()
    # Architecture (3-step contract in app.get_scored_data):
    #   1. compute_forensic_signals()  — must run BEFORE this function so that forensic_score,
    #      forensic_label, red_flag_count, schilit_pass are available when compute_qglp_score()
    #      and compute_composite_score() read them (flag_sqglp_engine, Diamond, Dhandho, etc.)
    #   2. run_full_scoring()          — this function
    #   3. apply_forensic_penalty()    — must run AFTER composite_score is assigned here
    # Do NOT call compute_forensic_signals() or run_forensic_analysis() inside this function —
    # forensic signals are already on df from step 1; calling them again would double-compute.

    mode = ANALYSIS_MODES.get(analysis_mode, ANALYSIS_MODES["Hybrid"])

    # ── Step 0: Detect market regime from the data ──
    regime = detect_market_regime(df)
    df.attrs["detected_market_regime"] = regime          # primary: fast scalar lookup
    df["_detected_market_regime"] = regime               # defensive: survives merge/concat

    # ── Step 1: Get regime-adaptive weights ──
    adaptive = get_adaptive_weights(scoring_profile, regime)

    print("\n" + "="*60)
    print(f"🏗️  SCORING ENGINE")
    print(f"   Mode:    {analysis_mode}")
    print(f"   Profile: {adaptive.get('profile_name')} | Regime: {adaptive.get('regime_label')}")
    print(f"   Weights: Q={adaptive['quality_w']:.0%} G={adaptive['growth_w']:.0%} "
          f"L={adaptive['longevity_w']:.0%} P={adaptive['price_w']:.0%}")
    print(f"   Gates:   ROCE≥{adaptive['roce_gate']:.0f}% | Growth≥{adaptive['growth_gate']:.0f}% | PEG≤{adaptive['peg_gate']:.1f}")
    print("="*60)

    # ── Layer 1: Hard Gates ──
    df = apply_hard_gates(df)

    # ── Layer 2: Quality Score ──
    df = compute_quality_score(df)

    # ── Epoch 3: Great/Good/Gruesome Taxonomy (13th WCS, vectorized, zero apply) ──
    # Study (13th WCS p.935): GREAT = "Very high AND RISING RoE"; GRUESOME = "Low / falling RoE".
    # The "rising / still-high-now" dimension matters: a company with a stellar 10Y median but whose
    # CURRENT return has collapsed has an ERODING moat — the study would NOT call it Great. So GREAT
    # adds a current-ROCE floor (>=15%, still genuinely high today) on top of the 10Y-median + asset-
    # light tests. Without it, a name like Crompton (10Y med 21% but current ROCE -1.2%) wrongly got
    # tagged GREAT and handed a +10% boost. Demoting such eroded moats to GOOD is faithful to the study.
    df["corporate_class"] = np.select(
        [
            (df["roce_med_10y"].fillna(0) >= 20.0) &
            (df["fcf_to_ocf_velocity"].fillna(0) >= 0.60) &
            (df["roce"].fillna(0) >= 15.0),                    # 13th WCS: still high NOW (not eroding)
            (df["roce_med_10y"].fillna(0) >= 12.0) &
            (df["fcf_to_ocf_velocity"].fillna(0) < 0.60),
            (df["roce_med_10y"].fillna(0) < 12.0),
        ],
        ["🏆 GREAT", "👍 GOOD", "💀 GRUESOME"],
        default="👍 GOOD"
    )

    # ── Epoch 3: Gruesome quality penalty (50% haircut) ──
    _gruesome_mask = df["corporate_class"] == "💀 GRUESOME"
    df.loc[_gruesome_mask, "quality_score"] = (
        df.loc[_gruesome_mask, "quality_score"] * EPOCH3_TAXONOMY["gruesome_quality_penalty"]
    )
    # ── Epoch 3: Great quality boost (10%) ──
    _great_mask = df["corporate_class"] == "🏆 GREAT"
    df.loc[_great_mask, "quality_score"] = (
        df.loc[_great_mask, "quality_score"] * EPOCH3_TAXONOMY["great_quality_boost"]
    ).clip(upper=100)

    # ── Layer 3: Momentum Score (apply regime momentum boost) ──
    df = compute_momentum_score(df)
    momentum_boost = adaptive.get("momentum_boost", 1.0)
    if momentum_boost != 1.0 and "momentum_score" in df.columns:
        df["momentum_score"] = _safe_clip(df["momentum_score"] * momentum_boost)
        print(f"   🌊 Regime momentum boost: {momentum_boost:.2f}x")

    # ── Governance Bonus ──
    df = compute_governance_bonus(df)

    # ── Profile-adaptive QGLP Framework ──
    df = compute_qglp_score(df, profile=adaptive)

    # ── 28th WCS EP Power Curve: quintile matrix + Hockey-Stick breakout ──
    df = compute_ep_power_curve(df)

    # ── Layer 4: Composite — blend per Analysis Mode ──
    fundamental_w = mode["fundamental_w"]
    momentum_w    = mode["momentum_w"]
    df = compute_composite_score(df, fundamental_w=fundamental_w, momentum_w=momentum_w)

    # ── Tsunami & Catalyst Detection ──
    df = detect_catalysts_and_tsunami(df)

    # ── Final sort ──
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    # Store adaptive weights in df for UI display
    df.attrs["adaptive_weights"] = adaptive

    print(f"\n✅ Scoring complete. Top 5:")
    top5 = df.head(5)[["rank", "name", "composite_score", "quality_score",
                        "momentum_score", "governance_bonus", "tier_label", "gate_pass"]]
    print(top5.to_string(index=False))

    return df
