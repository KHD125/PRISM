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
    MCAP_TIERS, MCAP_MIN_FLOOR,
    VALUATION_SIGNALS, PEG_ZONES, PAYBACK_ZONES, MEAN_REVERSION, BAID_SELL_TRIGGERS,
    DEFAULT_CYCLE_TEMPERATURE, MARKS_CYCLE,
    MASTER_PROFILES, ANALYSIS_MODES, WAVE_DETECTION,
    REGIME_ADJUSTMENTS, get_adaptive_weights,
    EPOCH2_REINVESTMENT, EPOCH3_TAXONOMY, EPOCH4_SQGLP, EPOCH5_MODERN,
)


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

    for gate_name, gate_cfg in HARD_GATES.items():
        col = gate_cfg["column"]
        op = gate_cfg["operator"]
        threshold = gate_cfg["threshold"]

        if col not in df.columns:
            # Gate column doesn't exist — skip but mark as N/A
            gate_results[gate_name] = pd.Series(True, index=df.index)
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

    # Overall pass: must pass ALL gates
    all_gates = pd.DataFrame(gate_results)
    df["gate_pass"] = all_gates.all(axis=1).astype(int)
    df["gates_passed"] = all_gates.sum(axis=1).astype(int)
    df["gates_total"] = len(gate_results)
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
    Uses groupby-based _sector_pct_rank (70% universe + 30% sector blend) for peer benchmarking.
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
    #   3. d35_roce_trend < 0: ROCE declining vs 1Y ago — no margin expansion to rescue decel.
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
    """Compute governance bonus from shareholding signals (0-100)."""
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
    if "promoter_holdings" in df.columns:
        promo_pct = df["promoter_holdings"].fillna(0)
        promo_1y  = df.get("change_promoter_1y", pd.Series(0.0, index=df.index)).fillna(0)
        promo_2y  = df.get("change_promoter_2y", pd.Series(0.0, index=df.index)).fillna(0)
        promo_3y  = df.get("change_promoter_3y", pd.Series(0.0, index=df.index)).fillna(0)
        bonus += (promo_pct >= 60).astype(float) * GOVERNANCE_BONUS["promoter_high_alignment"]
        bonus += ((promo_pct >= 50) & (promo_pct < 60)).astype(float) * GOVERNANCE_BONUS["promoter_good_alignment"]
        bonus += ((promo_pct < 40) & (promo_1y < 0)).astype(float) * GOVERNANCE_BONUS["promoter_low_declining"]
        # 3-year trend signals — single-quarter buys/sells are noise; 3Y patterns are decisions.
        # Accumulation: net buying >3% over 3 years = sustained conviction, dynasty building.
        bonus += (promo_3y > 3).astype(float) * GOVERNANCE_BONUS["promoter_3y_accumulation"]
        # Systematic exit: net selling >5% over 3 years = structural concern, promoter leaving.
        bonus += (promo_3y < -5).astype(float) * GOVERNANCE_BONUS["promoter_3y_exit"]
        # Early warning: 2Y selling >3% but 3Y hasn't crossed threshold yet = trend forming.
        _recent_exit = (promo_2y < -3) & (promo_3y >= -5)
        bonus += _recent_exit.astype(float) * GOVERNANCE_BONUS["promoter_2y_recent_exit"]

    # Undiscovered alpha: low FII + Tier C
    if "fii_holdings" in df.columns and "market_cap" in df.columns:
        undiscovered = (df["fii_holdings"] < 5) & (df["market_cap"] < 5000)
        bonus += undiscovered.astype(float) * GOVERNANCE_BONUS["undiscovered_alpha"]

    # Dilution penalty: Tier 3 (>10%) is hard-gated and never reaches here.
    # Tier 2 (3-10%) = -25 pts governance; Tier 1 (<3% ESOP) = -5 pts.
    if "dilution_flag" in df.columns:
        dilution = df["dilution_flag"].fillna(0)
        bonus += pd.Series(
            np.where(
                dilution == 2, GOVERNANCE_BONUS["dilution_tier2_penalty"],
                np.where(dilution == 1, GOVERNANCE_BONUS["dilution_tier1_minor"], 0)
            ),
            index=df.index
        )

    # G3 FIX: _safe_clip([0,100]) was erasing dilution penalties for companies starting at 0 governance.
    # A company with 0 base + dilution_flag=2 → bonus=-25 → clipped to 0 (penalty vanished).
    # Allow negative governance to drag down the composite score for serial diluters.
    df["governance_bonus"] = bonus.clip(lower=-50, upper=100)
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
        (_peg_v.fillna(0.0) > 0) & (_peg_v.fillna(999.0) <= EPOCH4_SQGLP["max_peg_ratio"]) & # P: PEG <= 1.0
        (df.get("forensic_label", pd.Series("", index=df.index)) == "🟢 Clean")                # Integrity gate
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
    tsunami_conditions = (
        (df["gate_pass"] == 1) &
        (df.get("above_sma200", pd.Series(0, index=df.index)) == 1) &
        (df["vstop_green"] == 1) &
        (df["vstop_fresh"] == 1) &
        (df["promoter_buying"] == 1) &
        (df.get("change_fii_lq", pd.Series(0, index=df.index)) > 0) &
        (df["quality_score"] >= 70) &
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
    _cc_roe_rec   = df.get("roe_med_5y",   _cc_nan).fillna(df.get("roe",        _cc_nan))
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
    # fillna(10): missing year data doesn't penalize — treated as meeting the threshold.
    _cc_y2 = df.get("rev_gr_y2", _cc_nan).fillna(10)
    _cc_y3 = df.get("rev_gr_y3", _cc_nan).fillna(10)
    _cc_y4 = df.get("rev_gr_y4", _cc_nan).fillna(10)
    _cc_y5 = df.get("rev_gr_y5", _cc_nan).fillna(10)
    _cc_each_year = (
        (_cc_y2 >= 10) & (_cc_y3 >= 10) & (_cc_y4 >= 10) & (_cc_y5 >= 10)
    )
    fw_coffee_can = (
        _cc_efficiency                            &  # ROCE (non-fin) / ROE (fin) hurdle
        (rev_10y_cc.fillna(0)    >= 10)           &  # 10Y revenue CAGR ≥ 10%
        (rev_5y_cc.fillna(0)     >= 8)            &  # 5Y revenue CAGR ≥ 8%
        (rev_yoy_cc.fillna(-1)   >= 0)            &  # not currently contracting (year 1)
        _cc_each_year                             &  # years 2-5: each ≥ 10% (book requirement)
        (cfo_ebitda_cc.fillna(0) >= 90)           &  # CFO/EBITDA ≥ 90% (Clean Accounts)
        (is_fin_cc | (de_cc.fillna(999) < 1.0))   &  # D/E < 1 for non-financials
        (pledge_cc.fillna(0)     < 10)               # pledge < 10% governance gate
    )

    # 3. Magic Formula (Joel Greenblatt) — high Earnings Yield + high ROCE
    ey_mf   = df.get("earnings_yield", pd.Series(np.nan, index=df.index)).fillna(0)
    roce_mf = df.get("roce", pd.Series(np.nan, index=df.index)).fillna(0)
    fw_magic_formula = (ey_mf >= 8) & (roce_mf >= 20)

    # 4. SMILE (Maheshwari) — Small/mid cap + high growth + ROCE
    mcap_sm  = df.get("market_cap", pd.Series(np.nan, index=df.index)).fillna(0)
    pat_gr_sm = df.get("pat_gr_5y", pd.Series(np.nan, index=df.index)).fillna(0)
    fw_smile = (mcap_sm < 15000) & (pat_gr_sm >= 20) & (roce_mf >= 20)

    # 5. Lynch Fast Grower (Peter Lynch — One Up on Dalal Street)
    # Pattern DNA from 20 Indian tenbagger case files in the book:
    # Rev CAGR > 20% + PEG < 0.75 (Lynch's preferred sweet spot, not just fair value at 1.0)
    # + pre-institutional discovery (FII < 10%) + owner-operator promoter ≥ 45%
    # fii_holdings fillna(50): if data missing, assume already discovered → exclude
    _ly_nan   = pd.Series(np.nan, index=df.index)
    peg_ly    = df.get("peg",               pd.Series(999.0, index=df.index)).fillna(999)
    rev_ly    = df.get("rev_gr_5y",         _ly_nan)
    pat3y_ly  = df.get("pat_gr_3y",         _ly_nan)
    debt_ly   = df.get("debt_to_equity",    _ly_nan)
    fii_ly    = df.get("fii_holdings",      pd.Series(50.0, index=df.index)).fillna(50)
    promo_ly  = df.get("promoter_holdings", _ly_nan)
    is_fin_ly = df.get("is_financial",      pd.Series(False, index=df.index)).fillna(False)
    fw_lynch = (
        (rev_ly.fillna(0)    >= 20) &                    # Fast Grower: 20%+ revenue CAGR
        (peg_ly > 0)                &                    # PEG must be positive (real growth)
        (peg_ly <= 0.75)            &                    # Lynch sweet spot: price ≤ 0.75× growth rate
        (pat3y_ly.fillna(0)  >= 15) &                    # Earnings confirming the revenue story
        (is_fin_ly | (debt_ly.fillna(999) < 0.5)) &      # Clean balance sheet
        (fii_ly < 10)               &                    # Pre-discovery: institutions < 10%
        (promo_ly.fillna(0)  >= 45)                      # Owner-operator conviction
    )

    # 6. CAN SLIM (William O'Neill) — earnings acceleration + technical leadership
    #    C: Quarterly EPS growth ≥ 25% YoY (book: "at least 25%", best are 40-100%+)
    #    A: Annual EPS 5Y CAGR ≥ 25% (FIXED: was 20%; book specifies 25%) + ROE ≥ 17%
    #    N: Near 52W high — within 15% (book: "10-15% of 52-week high")
    #    S: Supply/Demand — vol_ratio ≥ 1.5 (book: "40-50% above average" = 1.4–1.5x)
    #    L: Leader — RS percentile ≥ 80 (book: "RS Rating 80+, average of winners = 87")
    #       Approximated via _pct_rank(d47_rs_composite) — percentile rank across all 2108 stocks
    #    I: Institutional sponsorship — FII or DII buying (best proxy for A/B A-D rating)
    #    M: Market direction — NOT IMPLEMENTABLE (requires market index data absent from stock CSV)
    #    Sources: CAN SLIM Mastery Guide Chapter 1 (C,A), Chapter 2 (N,S), Chapter 3 (L,I), Chapter 6 (M)
    pat_lq_cs    = df.get("pat_lq",          pd.Series(np.nan, index=df.index)).fillna(np.nan)
    pat_pyq_cs   = df.get("pat_pyq",         pd.Series(np.nan, index=df.index)).fillna(np.nan)
    eps_gr_cs    = df.get("eps_gr_5y",        pd.Series(np.nan, index=df.index)).fillna(0)
    roe_cs       = df.get("roe",              pd.Series(np.nan, index=df.index)).fillna(0)    # A: ROE ≥ 17%
    dist_wh_cs   = df.get("dist_52wh",        pd.Series(999.0,  index=df.index)).fillna(999)
    vol_r_cs     = df.get("vol_ratio",        pd.Series(np.nan, index=df.index)).fillna(1.0)
    rs_comp_cs   = df.get("d47_rs_composite", pd.Series(np.nan, index=df.index)).fillna(0)
    fii_cs       = df.get("change_fii_lq",    pd.Series(0.0,    index=df.index)).fillna(0)
    dii_cs       = df.get("change_dii_lq",    pd.Series(0.0,    index=df.index)).fillna(0)
    # L: Percentile rank of composite RS (50D+26W+52W avg) across universe — approximates IBD RS Rating
    # ascending=True: higher RS composite → higher rank → top 20% = RS Rating ≥ 80
    rs_pctrank_cs = _pct_rank(rs_comp_cs, ascending=True).fillna(50)
    # C: Quarterly EPS growth: (pat_lq / pat_pyq - 1) >= 0.25, guarded against zero/negative base
    qtr_growth_ok = np.where(
        pat_lq_cs.notna() & pat_pyq_cs.notna() & (pat_pyq_cs > 0),
        ((pat_lq_cs / pat_pyq_cs - 1) >= 0.25),
        False
    )
    fw_can_slim = (
        qtr_growth_ok &                    # C: quarterly earnings +25%+ YoY
        (eps_gr_cs        >= 25) &         # A: annual EPS 5Y CAGR ≥ 25% (FIXED: was 20%)
        (roe_cs           >= 17) &         # A: ROE ≥ 17% (O'Neill's explicit A-criterion requirement)
        (dist_wh_cs       <= 15) &         # N: within 15% of 52W high
        (vol_r_cs         >= 1.5) &        # S: volume ≥ 1.5× avg (40-50%+ above average)
        (rs_pctrank_cs    >= 80) &         # L: RS Rating ≥ 80 (top 20% of universe by RS composite)
        ((fii_cs > 0) | (dii_cs > 0))     # I: institutional buying confirmed
    )
    # CAN SLIM criteria count (0-7): useful for partial-pass display and ranking within non-full-pass stocks
    df["can_slim_score"] = (
        pd.Series(qtr_growth_ok, index=df.index).astype(int) +   # C
        (eps_gr_cs >= 25).astype(int) +                           # A1: EPS CAGR
        (roe_cs    >= 17).astype(int) +                           # A2: ROE
        (dist_wh_cs <= 15).astype(int) +                          # N
        (vol_r_cs   >= 1.5).astype(int) +                         # S
        (rs_pctrank_cs >= 80).astype(int) +                       # L
        ((fii_cs > 0) | (dii_cs > 0)).astype(int)                # I
    )

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
    fw_malik_peaceful = (
        (rev_gr_mk.fillna(0)  >= 10) &                   # P1: Sales CAGR ≥ 10% (10Y primary, 5Y fallback)
        npm_stable_mk &                                   # P2: NPM ≥ 8% stable, not a one-year spike
        (is_fin_mk | (ic_mk.fillna(0)   >= 3)) &         # P4: Interest coverage ≥ 3× (fin exempt)
        (is_fin_mk | (de_mk.fillna(999) <= 0.5)) &       # P5: D/E ≤ 0.5 — Malik's stricter standard
        (is_fin_mk | (cr_mk.fillna(0)   >= 1.25)) &      # P6: Current ratio ≥ 1.25 (fin exempt)
        (cfo_pat_mk.fillna(0) >= 70) &                   # P8: CFO/PAT ≥ 70% — cash backs earnings
        (ssgr_mk == 1)                                    # SSGR: growth is self-funded (Malik's signature)
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
    roe_5y_ub   = df.get("roe_med_5y",         _ub_nan).fillna(df.get("roe",        _ub_nan))
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
        (pledge_ub.fillna(0)   < 10)                         # Governance pillar: pledge < 10%
    )

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
        (oplev_fi == 1)                           # P4:  Operating leverage — profit growing faster than sales
    )

    # 12. 100-Bagger Candidate (Christopher Mayer / SQGLP) — Small-cap owner-operator compounder
    #    The SQGLP framework: Small size + Quality + Growth + Longevity + Price.
    #    Key distinctions vs existing frameworks:
    #      - SMILE uses market cap < ₹15,000 Cr — far too wide for 100× math
    #      - QGLP has no size filter at all — a ₹50,000 Cr company passes
    #      - THIS framework: ₹200–₹3,000 Cr + promoter holding ≥ 50% — BOTH are new signals
    #    The S (Small size ₹200–₹3,000 Cr): 100× math is mathematically strained above ₹3,000 Cr.
    #    Below ₹200 Cr: too early-stage, unproven model, illiquid stock (Mayer explicitly excludes).
    #    Promoter holding ≥ 50%: the #1 owner-operator alignment signal in Indian markets.
    #    Founder with 50%+ stake thinks in decades, not quarters — the promoter IS the moat.
    #    CFO/PAT > 0: earnings must be backed by real cash (positive OCF proxy).
    #    Unit note: market_cap in Crores; promoter_holdings as percentage (55.3 = 55.3%).
    _hb_nan    = pd.Series(np.nan, index=df.index)
    mcap_hb    = df.get("market_cap",        _hb_nan)
    roce_hb    = df.get("roce_med_5y",       _hb_nan)
    rev_3y_hb  = df.get("rev_gr_3y",         _hb_nan)
    pat_3y_hb  = df.get("pat_gr_3y",         _hb_nan)
    de_hb      = df.get("debt_to_equity",    _hb_nan)
    promo_hb   = df.get("promoter_holdings", _hb_nan)  # % e.g. 55.3 = 55.3%
    cfo_pat_hb = df.get("cfo_to_pat",        _hb_nan)  # PERCENTAGE
    is_fin_hb  = df.get("is_financial",      pd.Series(False, index=df.index)).fillna(False)
    fw_100_bagger = (
        (mcap_hb.fillna(0)    >= 200)  &           # S: not pre-revenue micro-cap (too early-stage)
        (mcap_hb.fillna(9999) <= 3000) &           # S: 100× math strained above ₹3,000 Cr
        (roce_hb.fillna(0)    >= 15)   &           # Q: ROCE 5Y median ≥ 15% — capital efficiency proven
        (rev_3y_hb.fillna(0)  >= 18)   &           # G: Revenue 3Y CAGR ≥ 18% — accelerating growth
        (pat_3y_hb.fillna(0)  >= 20)   &           # G: Earnings 3Y CAGR ≥ 20% — the compounding engine
        (is_fin_hb | (de_hb.fillna(999) < 0.5)) &  # Q: D/E < 0.5 — clean balance sheet
        (promo_hb.fillna(0)   >= 50)   &           # Q: Promoter ≥ 50% — owner-operator, skin in game
        (cfo_pat_hb.fillna(-1) > 0)                # Q: CFO positive — earnings backed by real cash
    )

    # 13. Diamond Field Guide (Saurabh Mukherjea) — forensic-verified compounders
    #    Three-lens framework: Stage 1 Screen → Gate Zero → Lens 1 (Accounts) → Lens 2 (Moat) → Lens 3 (Capex)
    #    Key distinctions from other Mukherjea frameworks:
    #      - D/E < 0.5: STRICTEST of all three Mukherjea books (Coffee Can < 1.0, Unusual Billionaires < 1.0)
    #      - CFO/PAT ≥ 75%: Lens 1 cash earnings quality (Coffee Can uses CFO/EBITDA — different denominator)
    #      - FCF/CFO ≥ 25%: Lens 3 capital allocation surplus — new signal absent in all prior frameworks
    #      - forensic_score == 0: mandatory clean accounts — most frameworks don't hard-require this
    #      - Market cap ≥ ₹500 Cr: quality size floor (not in Coffee Can or Unusual Billionaires)
    #    NOT implementable (no CSV data): year-by-year CFO/PAT, DSO 3Y trend, contingent liabilities,
    #    depreciation consistency, auditor quality, RPT ratios, GNPA/CASA (banks), moat durability scoring
    _dm_nan    = pd.Series(np.nan, index=df.index)
    roce_10y_dm = df.get("roce_med_10y",       _dm_nan)
    roce_5y_dm  = df.get("roce_med_5y",        _dm_nan)
    rev_10y_dm  = df.get("rev_gr_10y",         _dm_nan)
    rev_5y_dm   = df.get("rev_gr_5y",          _dm_nan)
    de_dm       = df.get("debt_to_equity",     _dm_nan)
    cfo_pat_dm  = df.get("cfo_to_pat",         _dm_nan)   # PERCENTAGE: 75.0 = 75%, not 0.75
    fcf_cfo_dm  = df.get("fcf_to_cfo_pct",    _dm_nan)   # PERCENTAGE: 25.0 = 25%
    mcap_dm     = df.get("market_cap",         _dm_nan)   # Crores
    promo_dm    = df.get("promoter_holdings",  _dm_nan)   # % e.g. 40.0 = 40%
    pledge_dm   = df.get("pledged_percentage", pd.Series(100.0, index=df.index)).fillna(100)
    fscore_dm   = df.get("forensic_score",     pd.Series(999, index=df.index)).fillna(999)
    is_fin_dm   = df.get("is_financial",       pd.Series(False, index=df.index)).fillna(False)
    fw_diamond = (
        (roce_10y_dm.fillna(0) >= 15) &                     # Lens 2: ROCE > 15% 10Y — moat proven over full cycle
        (roce_5y_dm.fillna(0)  >= 15) &                     # Lens 2: ROCE > 15% 5Y — moat sustained recently
        (rev_10y_dm.fillna(0)  >= 10) &                     # Stage 1: Revenue growth 10Y > 10%
        (rev_5y_dm.fillna(0)   >=  8) &                     # Stage 1: Recent growth not decelerating sharply
        (is_fin_dm | (de_dm.fillna(999) < 0.5)) &           # Stage 1: D/E < 0.5 — strictest Mukherjea filter
        (cfo_pat_dm.fillna(0)  >= 75) &                     # Lens 1: CFO/PAT ≥ 75% cash earnings quality
        (fcf_cfo_dm.fillna(0)  >= 25) &                     # Lens 3: FCF/CFO ≥ 25% capital allocation surplus
        (mcap_dm.fillna(0)     >= 500) &                    # Stage 1: ≥ ₹500 Cr proven business scale
        (promo_dm.fillna(0)    >= 40) &                     # Gate Zero: Promoter ≥ 40% alignment
        (pledge_dm             <  10) &                     # Gate Zero: Pledge < 10%
        (fscore_dm             ==  0)                       # Lens 1: Zero forensic red flags
    )

    # 14. Dorsey Wide Moat (Pat Dorsey — The Moat Investor's Codex)
    #    Confirmed Wide Moat at an attractive free-cash-flow price.
    #    Three signals unique to this framework — none of the 13 existing tags use them together:
    #      1. ROCE ≥ 20%: "confirmed moat" (all other frameworks gate at 15%; Dorsey explicitly says
    #         "above 15% = likely moated; above 20% = confirmed wide moat" — a materially different bar)
    #      2. FCF yield ≥ 5%: Dorsey's primary valuation gate for wide moat stocks — no other framework
    #         in this system uses absolute FCF yield as a hard filter
    #      3. d35_roce_trend ≥ 0: moat DIRECTION — rising/stable ROIC = moat widening or intact;
    #         all existing frameworks are point-in-time snapshots, none check trajectory direction
    #    CFO/PAT ≥ 80%: stricter than Diamond's 75%, matches Dorsey: "< 70% = investigate, > 80% = genuine"
    #    D/E < 1.0 (not 0.5): Dorsey explicitly allows up to 1.0 for capital-intensive moat businesses
    #    (asset-heavy moats with switching costs pass where Diamond's 0.5 gate would reject them)
    #    NOT implementable: moat source classification (brand/network/switching — qualitative),
    #    margin-of-safety vs intrinsic value (requires DCF), CASA/GNPA banking metrics (not in CSV)
    _dw_nan      = pd.Series(np.nan, index=df.index)
    roce_10y_dw  = df.get("roce_med_10y",    _dw_nan)
    roce_5y_dw   = df.get("roce_med_5y",     _dw_nan)
    rev_5y_dw    = df.get("rev_gr_5y",       _dw_nan)
    cfo_pat_dw   = df.get("cfo_to_pat",      _dw_nan)   # PERCENTAGE: 80.0 = 80%
    fcf_yield_dw = df.get("fcf_yield",       _dw_nan)   # PERCENTAGE: 5.0 = 5%
    roce_dir_dw  = df.get("d35_roce_trend",  _dw_nan)   # positive = ROCE improving vs 1Y ago
    de_dw        = df.get("debt_to_equity",  _dw_nan)
    is_fin_dw    = df.get("is_financial",    pd.Series(False, index=df.index)).fillna(False)
    fw_dorsey = (
        (roce_10y_dw.fillna(0) >= 20) &              # Confirmed Wide Moat: ROIC > 20% over full cycle
        (roce_5y_dw.fillna(0)  >= 20) &              # Confirmed Wide Moat: sustained in recent window
        (rev_5y_dw.fillna(0)   >= 10) &              # Moat enables durable revenue compounding
        (cfo_pat_dw.fillna(0)  >= 80) &              # Earnings quality: CFO/PAT ≥ 80% (genuine profits)
        (fcf_yield_dw.fillna(0) >= 5) &              # Wide moat at attractive price: FCF yield ≥ 5%
        (roce_dir_dw.fillna(-1) >= 0) &              # Moat direction: ROCE not eroding vs 1Y ago
        (is_fin_dw | (de_dw.fillna(999) < 1.0))      # Capital discipline: < 1.0 (allows asset-heavy moats)
    )

    # 15. Outsiders on Dalal Street — Capital Allocation Excellence
    #    The Outsider CEO fingerprint: deleveraging + zero dilution + high cash conversion.
    #    Three signals unique across all 15 frameworks:
    #      1. de_slope_3y <= 0: D/E actively declining over 3Y — the ONLY framework to reward
    #         deleveraging as a positive quality signal (others use D/E as static threshold)
    #      2. dilution_flag == 0 as PRIMARY hard gate — Outsider DNA = per-share value creation,
    #         not empire building. Fisher also requires it, but among 7 other conditions.
    #      3. CFO/PAT ≥ 85%: highest cash quality threshold in the entire system
    #         (Diamond=75%, Dorsey=80%, Outsiders=85%)
    #    D/E < 0.75: book's explicit Stage 1 threshold — unique value (between Diamond's 0.5 and
    #         Coffee Can's 1.0; a deleveraging company at 0.7 passes here, fails Diamond)
    #    Market cap ≥ ₹1,000 Cr: book's explicit size floor (larger than Diamond's 500 Cr)
    #    de_slope_3y fillna(1): if D/E trend data missing/restatement suspected → exclude
    #         (cannot verify deleveraging = won't award the Outsider badge)
    #    NOT implementable: HQ cost < 0.5% FCF (not in CSV), CEO communication quality,
    #         decentralisation structure, acquisition ROIC vs hurdle (no M&A data)
    _os_nan      = pd.Series(np.nan, index=df.index)
    roce_10y_os  = df.get("roce_med_10y",    _os_nan)
    roce_5y_os   = df.get("roce_med_5y",     _os_nan)
    rev_10y_os   = df.get("rev_gr_10y",      _os_nan).fillna(df.get("rev_gr_5y", _os_nan))
    cfo_pat_os   = df.get("cfo_to_pat",      _os_nan)   # PERCENTAGE: 85.0 = 85%, not 0.85
    dilut_os     = df.get("dilution_flag",   pd.Series(1, index=df.index)).fillna(1)
    de_os        = df.get("debt_to_equity",  _os_nan)
    de_slope_os  = df.get("de_slope_3y",     pd.Series(1.0, index=df.index)).fillna(1)
    mcap_os      = df.get("market_cap",      _os_nan)   # Crores
    is_fin_os    = df.get("is_financial",    pd.Series(False, index=df.index)).fillna(False)
    fw_outsider = (
        (roce_10y_os.fillna(0) >= 15) &             # Quality business: ROIC above cost of capital 10Y
        (roce_5y_os.fillna(0)  >= 15) &             # Quality business: ROIC sustained recently
        (rev_10y_os.fillna(0)  >=  8) &             # Business still compounding revenue
        (cfo_pat_os.fillna(0)  >= 85) &             # Cash conversion: CFO/PAT ≥ 85% (highest bar in system)
        (dilut_os              ==  0) &             # Zero dilution: per-share value, not empire building
        (de_slope_os           <=  0) &             # Deleveraging: D/E declining/stable over 3Y (Outsider DNA)
        (is_fin_os | (de_os.fillna(999) < 0.75)) &  # Stage 1: D/E < 0.75 (book's explicit threshold)
        (mcap_os.fillna(0)     >= 1000)              # Stage 1: ≥ ₹1,000 Cr established business
    )

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
    fscore_dh    = df.get("forensic_score",  pd.Series(999, index=df.index)).fillna(999)
    is_fin_dh    = df.get("is_financial",    pd.Series(False, index=df.index)).fillna(False)
    fw_dhandho = (
        (dist_wh_dh        >= 30) &              # HIGH UNCERTAINTY: fallen 30%+ from 52W high
        (fcf_yield_dh.fillna(0) >= 8) &          # LOW ACTUAL RISK: FCF yield ≥ 8% (payback ≤ 12.5Y)
        (roce_5y_dh.fillna(0)  >= 15) &          # Moat intact: ROIC > 15% over 5Y — not a value trap
        (cfo_pat_dh.fillna(0)  >= 70) &          # Earnings real: CFO/PAT ≥ 70% (Pabrai's cash test)
        (is_fin_dh | (de_dh.fillna(999) < 0.5)) & # Balance sheet: D/E < 0.5 — no leverage amplifying risk
        (fscore_dh             == 0)              # Accounting integrity: zero forensic red flags
    )

    # 18. Parikh Contrarian (Parag Parikh — Value Investing and Behavioral Finance)
    #    Graham's quantitative floor + Parag's quality filter + anti-herd behavioral overlay.
    #    Implements Stages 1, 3, and 4 of the book's Four-Stage Screen. Stage 2 (intrinsic value
    #    via DCF/EPV/Graham Number) is NOT automatable — requires BVPS and WACC modeling absent
    #    from the CSV.
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
    #    NOT implementable: P/B < 3 (no price-to-book column in CSV), P/E × P/B < 22.5 (same),
    #    intrinsic value discount 20% (requires DCF/EPV/Graham Number with BVPS not in CSV),
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

    # 19. Baid Compounder (Gautam Baid — The Compounding Codex)
    #    Baid's "Nirvana" framework: long-duration compounders with ZERO revenue shortfalls.
    #    The defining uniqueness: rev_gr_yoy >= 5 enforces the "no single year below 5%" rule —
    #    Baid explicitly rejects stocks that stumble even once, as it signals moat fragility.
    #    Chapter 4 (Identifying Compounders), Chapter 6 (Valuation Discipline), Chapter 15 (Sell).
    #    PEG 0–1.5 = Baid's reasonable entry ("between fair and cheap") — no other framework uses 1.5.
    #    cfo_to_pat >= 80 = Baid's "FCF-to-PAT above 0.8" (PERCENTAGE: 80.0 = 80%).
    #    fcf_yield >= 3 = PERCENTAGE: unique between Quality Compounder's 2% and Dorsey's 5%.
    _bd_nan      = pd.Series(np.nan, index=df.index)
    roce_5y_bd   = df.get("roce_med_5y",     _bd_nan)
    rev_5y_bd    = df.get("rev_gr_5y",       _bd_nan)   # 5Y revenue CAGR
    rev_yoy_bd   = df.get("rev_gr_yoy",      _bd_nan)   # UNIQUE: current-year revenue floor (no stumble allowed)
    fcf_yield_bd = df.get("fcf_yield",       _bd_nan)   # PERCENTAGE: 3.0 = 3%
    cfo_pat_bd   = df.get("cfo_to_pat",      _bd_nan)   # PERCENTAGE: 80.0 = 80%
    de_bd        = df.get("debt_to_equity",  _bd_nan)
    mcap_bd      = df.get("market_cap",      _bd_nan)
    peg_bd       = df.get("peg",             _bd_nan)
    is_fin_bd    = df.get("is_financial",    pd.Series(False, index=df.index)).fillna(False)
    fw_baid = (
        (roce_5y_bd.fillna(0)   >= 15) &              # Chapter 4: ROCE > 15% sustained for 5Y — capital allocation proof
        (rev_5y_bd.fillna(0)    >= 12) &              # Chapter 4: revenue CAGR ≥ 12% over 5Y — compounding velocity
        (rev_yoy_bd.fillna(0)   >=  5) &              # UNIQUE — Chapter 4: current year ≥ 5%; no single year shortfall allowed
        (fcf_yield_bd.fillna(0) >=  3) &              # Chapter 6: FCF yield ≥ 3% — Baid's cash payback discipline
        (cfo_pat_bd.fillna(0)   >= 80) &              # Chapter 4: CFO/PAT ≥ 80% — Baid's "earnings quality" threshold
        (is_fin_bd | (de_bd.fillna(999) < 0.5)) &    # Chapter 4: D/E < 0.5 fortress balance sheet (fin exempt)
        (mcap_bd.fillna(0)      >= 500) &             # Chapter 4: proven size filter — avoids micro-cap noise
        (peg_bd.fillna(999)     >   0) &              # Chapter 6: PEG > 0 — must have positive earnings
        (peg_bd.fillna(999)     <= 1.5)               # Chapter 6: PEG ≤ 1.5 — Baid's "reasonable" entry (UNIQUE threshold)
    )

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

    # 21. SEPA Momentum (Mark Minervini — SEPA Trading Codex)
    #    Specific Entry Point Analysis: the ONLY pure technical-momentum framework in the system.
    #    All 20 prior frameworks are static quality gates (ROCE, CFO/PAT, D/E, etc.).
    #    SEPA adds DYNAMIC momentum gates — the stock must be winning RIGHT NOW, not just quality.
    #    Three signals unique across all 20 prior frameworks:
    #    1. d45_trend_structure >= 2: SMA alignment proxy (above_sma200 + VSTOP + ADX ≥ 2/3)
    #       No other framework tests MA structure. This is the "right stacking" filter.
    #    2. crs_aligned == 1: all THREE relative strength timeframes positive (50D, 26W, 52W)
    #       CAN SLIM uses crs_50d > 0 only (one timeframe). SEPA's RS≥70 requires sustained
    #       multi-timeframe outperformance — stocks coasting on prior RS get filtered out.
    #    3. roe >= 17: Minervini's specific ROE threshold (Coffee Can uses ≥15%; 17% is unique)
    #    Key differentiation from CAN SLIM (Framework 6):
    #    - CAN SLIM: quarterly PAT growth + 5Y EPS CAGR + within 15% of 52WH + vol≥1.5 + crs_50d
    #    - SEPA: annual EPS/rev YoY + ROE≥17 + SMA alignment + all-3-timeframe CRS + within 25% 52WH
    #    A stock with crs_50d>0 but crs_26w<0 passes CAN SLIM L, FAILS SEPA.
    #    A stock in Stage 3 (MAs flattening, crs_aligned=0) passes quality frameworks, FAILS SEPA.
    #    Sources: Chapter 2 (Trend Template, 8 criteria), Chapter 4 (7 fundamental requirements),
    #             Chapter 11 (India SEPA scanner, ₹500 Cr filter, RS ≥ 70).
    _sp_nan       = pd.Series(np.nan, index=df.index)
    trend_str_sp  = df.get("d45_trend_structure",  pd.Series(0, index=df.index)).fillna(0)
    dist_wh_sp    = df.get("dist_52wh",            pd.Series(999.0, index=df.index)).fillna(999)
    crs_ali_sp    = df.get("crs_aligned",          pd.Series(0, index=df.index)).fillna(0)
    eps_yoy_sp    = df.get("eps_gr_yoy",           _sp_nan)   # REQ 1: EPS ≥ 25% YoY acceleration
    rev_yoy_sp    = df.get("rev_gr_yoy",           _sp_nan)   # REQ 2: Revenue ≥ 20% YoY
    roe_sp        = df.get("roe",                  _sp_nan)   # REQ 4: ROE > 17% (UNIQUE threshold)
    fii_sp        = df.get("change_fii_lq",        pd.Series(0.0, index=df.index)).fillna(0)
    dii_sp        = df.get("change_dii_lq",        pd.Series(0.0, index=df.index)).fillna(0)
    mcap_sp       = df.get("market_cap",           _sp_nan)
    fw_sepa = (
        (trend_str_sp           >= 2) &              # Trend Template proxy: SMA alignment ≥ 2/3 criteria
        (dist_wh_sp             <= 25) &             # Within 25% of 52WH — Stage 2 uptrend near highs
        (crs_ali_sp             == 1) &              # UNIQUE: RS confirmed across all 3 timeframes (50D+26W+52W)
        (eps_yoy_sp.fillna(0)   >= 25) &             # REQ 1: EPS growth ≥ 25% YoY — acceleration threshold
        (rev_yoy_sp.fillna(0)   >= 20) &             # REQ 2: Revenue ≥ 20% YoY — sales growth gate
        (roe_sp.fillna(0)       >= 17) &             # REQ 4: UNIQUE — ROE ≥ 17% (Minervini's explicit level)
        ((fii_sp > 0) | (dii_sp > 0)) &             # REQ 7: Institutional sponsorship growing (FII or DII)
        (mcap_sp.fillna(0)      >= 500)              # India screen: ₹500 Cr minimum (Minervini's explicit filter)
    )

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
    #   2. pledged_percentage <= 30: India QMOM operator-manipulation defense threshold
    #      (all 22 prior frameworks use <= 10% or <= 20% for governance quality; 30% is the
    #       specific momentum-strategy threshold per handbook Ch.7 — excludes operator-driven
    #       discrete spikes that create false momentum signals)
    # What CANNOT be implemented (missing data):
    #   - 12-1 skip-month momentum (requires monthly price history not in CSV)
    #   - FIP score (requires 11 months of monthly returns not in CSV)
    #   - Regime filter (requires Nifty 500 index time-series not in stock CSV)
    # Best available proxies used: crs_aligned (all-timeframe RS positive = smooth persistence),
    #   d47_rs_composite percentile rank (top-decile proxy), d51_qmom_quality_score available.
    # Sources: Chapters 2 (12-1 momentum), 3 (FIP), 4 (quality overlay), 7 (India operator filter).
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

    # Build comma-separated framework string — fully vectorized, zero apply
    fw_str = (
        np.where(fw_qglp,                   "QGLP|",                       "") +
        np.where(fw_coffee_can,             "Coffee Can|",                 "") +
        np.where(fw_magic_formula,          "Magic Formula|",              "") +
        np.where(fw_smile,                  "SMILE|",                      "") +
        np.where(fw_lynch,                  "Lynch Dream|",                "") +
        np.where(fw_can_slim,               "CAN SLIM|",                   "") +
        np.where(fw_bruised_bb,             "Bruised Blue Chip|",          "") +
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
        np.where(fw_multitrillioncap,       "Multi-Trillion Cap|",         "")
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
    df["ep_quintile"] = pd.Series(np.nan, index=df.index)
    if "economic_profit" in df.columns:
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
    # Ensure forensic data is present for framework checks (fw_diamond, fw_dhandho need forensic_score).
    # Guard: skip if run_forensic_analysis already ran (avoids double computation on every scoring run).
    if "forensic_score" not in df.columns:
        from core.forensic_engine import compute_piotroski_fscore, compute_red_flags
        df = compute_piotroski_fscore(df)
        df = compute_red_flags(df)

    mode = ANALYSIS_MODES.get(analysis_mode, ANALYSIS_MODES["Hybrid"])

    # ── Step 0: Detect market regime from the data ──
    regime = detect_market_regime(df)
    df.attrs["detected_market_regime"] = regime

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

    # ── Epoch 3: Great/Good/Gruesome Taxonomy (vectorized, zero apply) ──
    df["corporate_class"] = np.select(
        [
            (df["roce_med_10y"].fillna(0) >= 20.0) &
            (df["fcf_to_ocf_velocity"].fillna(0) >= 0.60),
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
