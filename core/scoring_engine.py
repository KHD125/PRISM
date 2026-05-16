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
    SECTOR_SIGNALS, GOVERNANCE_BONUS, CONVICTION_TIERS, RSI_ZONES,
    MCAP_TIERS, MCAP_MIN_FLOOR,
    VALUATION_SIGNALS, PEG_ZONES, MEAN_REVERSION, BAID_SELL_TRIGGERS,
    DEFAULT_CYCLE_TEMPERATURE, MARKS_CYCLE,
    MASTER_PROFILES, ANALYSIS_MODES, WAVE_DETECTION,
    REGIME_ADJUSTMENTS, get_adaptive_weights,
)


# ═══════════════════════════════════════════════════════════════
# UTILITY: Percentile rank with NaN handling
# ═══════════════════════════════════════════════════════════════

def _pct_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    """Percentile rank (0–100) with NaN preserved.
    ascending=True means higher values get higher rank.
    ascending=False means lower values get higher rank.
    """
    return series.rank(pct=True, ascending=ascending, na_option='keep') * 100


def _safe_clip(series: pd.Series, lo: float = 0, hi: float = 100) -> pd.Series:
    """Clip series to [lo, hi] range."""
    return series.clip(lower=lo, upper=hi)


def _zone_score(value: pd.Series, zones: dict) -> pd.Series:
    """Score based on value falling in predefined zones."""
    result = pd.Series(50.0, index=value.index)  # default neutral
    for zone_name, z in zones.items():
        mask = (value >= z["min"]) & (value < z["max"])
        result = np.where(mask, z["score"], result)
    return pd.Series(result, index=value.index)


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
        critical_gates = {"pledge_safety", "pledge_direction", "positive_ocf"}
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

    # Overall pass: must pass ALL gates
    all_gates = pd.DataFrame(gate_results)
    df["gate_pass"] = all_gates.all(axis=1).astype(int)
    df["gates_passed"] = all_gates.sum(axis=1).astype(int)
    df["gates_total"] = len(gate_results)
    df["gates_failed"] = df["gates_total"] - df["gates_passed"]

    # Build a human-readable failed gates string
    def _failed_gates_str(row):
        failed = []
        for gn in gate_results:
            if not row.get(f"gate_{gn}", True):
                failed.append(gn)
        return ", ".join(failed) if failed else "All passed ✅"

    df["failed_gates"] = df.apply(_failed_gates_str, axis=1)

    passed_count = df["gate_pass"].sum()
    total = len(df)
    print(f"\n🚪 Hard Gates: {passed_count}/{total} stocks passed ({passed_count/total*100:.1f}%)")

    return df


# ═══════════════════════════════════════════════════════════════
# LAYER 2: QUALITY SCORE
# ═══════════════════════════════════════════════════════════════

def _compute_moat_score(df: pd.DataFrame) -> pd.Series:
    """Moat score: ROCE trajectory + ROE. Higher = wider moat."""
    score = pd.Series(0.0, index=df.index)

    signals = {
        "roce_med_10y":        (_pct_rank(df.get("roce_med_10y", 0), ascending=True), 0.35),
        "roce_trajectory":     (_pct_rank(df.get("roce_trajectory", 0), ascending=True), 0.15),
        "roe_med_10y":         (_pct_rank(df.get("roe_med_10y", 0), ascending=True), 0.25),
        "roe_trajectory":      (_pct_rank(df.get("roe_trajectory", 0), ascending=True), 0.10),
        "roce_current_vs_med": (_pct_rank(df.get("roce_current_vs_med", 0), ascending=True), 0.15),
    }

    for name, (ranked, weight) in signals.items():
        score += ranked.fillna(50) * weight

    return _safe_clip(score)


def _compute_growth_score(df: pd.DataFrame) -> pd.Series:
    """Growth score: Revenue, PAT, EPS compounding + acceleration."""
    score = pd.Series(0.0, index=df.index)

    signals = {
        "pat_gr_5y":        (True, 0.20),
        "pat_gr_10y":       (True, 0.10),
        "rev_gr_5y":        (True, 0.20),
        "rev_gr_10y":       (True, 0.10),
        "eps_gr_5y":        (True, 0.15),
        "ebitda_gr_5y":     (True, 0.10),
        "pat_acceleration": (True, 0.08),
        "rev_acceleration": (True, 0.07),
    }

    for col, (ascending, weight) in signals.items():
        if col in df.columns:
            score += _pct_rank(df[col], ascending=ascending).fillna(50) * weight

    return _safe_clip(score)


def _compute_cash_score(df: pd.DataFrame) -> pd.Series:
    """Cash quality score: CFO ratios, FCF yield, self-funding."""
    score = pd.Series(0.0, index=df.index)

    # Continuous signals (percentile ranked)
    continuous = {
        "cfo_to_pat":    (True, 0.25),
        "cfo_to_ebitda": (True, 0.15),
        "fcf_yield":     (True, 0.20),
        "capex_coverage": (True, 0.10),
    }
    for col, (ascending, weight) in continuous.items():
        if col in df.columns:
            score += _pct_rank(df[col], ascending=ascending).fillna(50) * weight

    # Binary signals (0 or 100)
    binary = {
        "fcf_consistency": 0.15,
        "self_funding":    0.15,
    }
    for col, weight in binary.items():
        if col in df.columns:
            score += df[col].fillna(0) * 100 * weight

    return _safe_clip(score)


def _compute_margin_score(df: pd.DataFrame) -> pd.Series:
    """Margin score: pricing power via NPM, OPM, GPM medians + acceleration."""
    score = pd.Series(0.0, index=df.index)

    signals = {
        "npm_med_5y":       (True, 0.30),
        "opm_med_5y":       (True, 0.25),
        "gpm_med_5y":       (True, 0.15),
        "npm_acceleration": (True, 0.15),
        "opm_acceleration": (True, 0.15),
    }

    for col, (ascending, weight) in signals.items():
        if col in df.columns:
            score += _pct_rank(df[col], ascending=ascending).fillna(50) * weight

    return _safe_clip(score)


def _compute_balance_sheet_score(df: pd.DataFrame) -> pd.Series:
    """Balance sheet score: fortress detection, deleveraging, CWIP conversion."""
    score = pd.Series(0.0, index=df.index)

    # Net debt negative is a binary fortress signal
    if "net_debt_negative" in df.columns:
        score += df["net_debt_negative"].fillna(0) * 100 * 0.25

    # Debt slope: negative is good (deleveraging)
    if "debt_slope_3y" in df.columns:
        score += _pct_rank(df["debt_slope_3y"], ascending=False).fillna(50) * 0.20

    # Reserves growth: higher is better
    if "reserves_growth" in df.columns:
        score += _pct_rank(df["reserves_growth"], ascending=True).fillna(50) * 0.20

    # CWIP conversion: positive means capacity went live
    if "cwip_conversion" in df.columns:
        score += _pct_rank(df["cwip_conversion"], ascending=True).fillna(50) * 0.15

    # Cash change: positive is good
    if "cash_change" in df.columns:
        score += _pct_rank(df["cash_change"], ascending=True).fillna(50) * 0.20

    return _safe_clip(score)


def _compute_valuation_score(df: pd.DataFrame) -> pd.Series:
    """Valuation attractiveness: Marks + Baid entry price discipline.
    Uses PE discount vs 10Y median, PEG zone, EV/EBITDA compression,
    FCF yield, and Baid's D/E < 0.5 fortress bonus.
    Lower valuations = higher score = better entry point."""
    score = pd.Series(0.0, index=df.index)

    # PE discount vs 10Y median: positive = trading below historical median = good
    if "pe_discount" in df.columns:
        score += _pct_rank(df["pe_discount"], ascending=True).fillna(50) * VALUATION_SIGNALS["pe_discount"]

    # PEG zone scoring (Baid + Marks: PEG < 1.0 = cheap, > 2.5 = extreme)
    if "peg" in df.columns:
        peg_score = _zone_score(df["peg"].clip(lower=0), PEG_ZONES)
        score += peg_score.fillna(50) * VALUATION_SIGNALS["peg_ratio"]

    # EV/EBITDA compression: positive ev_compression = getting cheaper = good
    if "ev_compression" in df.columns:
        score += _pct_rank(df["ev_compression"], ascending=True).fillna(50) * VALUATION_SIGNALS["ev_compression"]

    # FCF yield: higher = more attractive (Marks: > 3% large-cap, > 4% mid-cap)
    if "fcf_yield" in df.columns:
        score += _pct_rank(df["fcf_yield"], ascending=True).fillna(50) * VALUATION_SIGNALS["fcf_yield_val"]

    # Baid's D/E < 0.5 fortress bonus (net cash companies score highest)
    if "debt_to_equity" in df.columns:
        fortress = np.where(df["debt_to_equity"] < 0.1, 100,
                  np.where(df["debt_to_equity"] < 0.3, 85,
                  np.where(df["debt_to_equity"] < 0.5, 70,
                  np.where(df["debt_to_equity"] < 1.0, 40, 10))))
        score += pd.Series(fortress, index=df.index, dtype=float) * VALUATION_SIGNALS["de_fortress"]

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
        df.get("cfo_to_pat", pd.Series(1.0, index=df.index)) < 0.5
    ).astype(int)
    df["sell_alert_any"] = (
        (df["sell_alert_thesis_broken"] == 1) |
        (df["sell_alert_mgmt_deteriorated"] == 1) |
        (df["sell_alert_cash_collapse"] == 1)
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
    """Trend quality: VSTOP, ADX, RSI zone, golden cross."""
    score = pd.Series(0.0, index=df.index)

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

    # Golden cross recency (lower days = better)
    if "golden_cross_days" in df.columns:
        gc_rank = _pct_rank(df["golden_cross_days"], ascending=False).fillna(50)
        score += gc_rank * TREND_SIGNALS["golden_cross"]

    return _safe_clip(score)


def _compute_breakout_score(df: pd.DataFrame) -> pd.Series:
    """Breakout proximity: nearness to highs and breakout windows."""
    score = pd.Series(0.0, index=df.index)

    # 52WH distance (lower = closer to breakout = better)
    if "dist_52wh" in df.columns:
        score += _pct_rank(df["dist_52wh"], ascending=False).fillna(50) * BREAKOUT_SIGNALS["52wh_distance"]

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
        # Vol ratio > 2 = institutional surge
        vol_score = np.where(df["vol_ratio"] >= 2.0, 100,
                   np.where(df["vol_ratio"] >= 1.5, 80,
                   np.where(df["vol_ratio"] >= 1.0, 60,
                   np.where(df["vol_ratio"] >= 0.7, 40, 20))))
        score = pd.Series(vol_score, index=df.index, dtype=float)

    return _safe_clip(score)


def _compute_sector_leader_score(df: pd.DataFrame) -> pd.Series:
    """Sector leadership: outperformance vs industry peers."""
    score = pd.Series(0.0, index=df.index)

    for col, weight in SECTOR_SIGNALS.items():
        if col in df.columns:
            score += _pct_rank(df[col], ascending=True).fillna(50) * weight

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

    # Insider trading present
    if "insider_trading" in df.columns:
        bonus += df["insider_trading"].notna().astype(float) * GOVERNANCE_BONUS["insider_trading_present"]

    # Pledge falling over 1 year
    if "pledge_falling_1y" in df.columns:
        bonus += (df["pledge_falling_1y"] > 0).astype(float) * GOVERNANCE_BONUS["pledge_falling_1y"]

    # Undiscovered alpha: low FII + Tier C
    if "fii_holdings" in df.columns and "market_cap" in df.columns:
        undiscovered = (df["fii_holdings"] < 5) & (df["market_cap"] < 5000)
        bonus += undiscovered.astype(float) * GOVERNANCE_BONUS["undiscovered_alpha"]

    df["governance_bonus"] = _safe_clip(bonus)
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

    # Governance is always blended at 10% regardless of mode
    gov_w = COMPOSITE_WEIGHTS.get("governance", 0.10)
    # Normalize fundamental + momentum to fill remaining 90%
    scale = 1.0 - gov_w
    fund_scaled = fundamental_w * scale
    mom_scaled  = momentum_w  * scale

    df["composite_score"] = (
        df["quality_score"]   * fund_scaled +
        df["momentum_score"]  * mom_scaled +
        df["governance_bonus"] * gov_w
    )
    df["composite_score"] = _safe_clip(df["composite_score"])

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
    tsunami_conditions = (
        (df["gate_pass"] == 1) &
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
    # Capacity Explosion (CWIP converting to fixed assets)
    df["cat_capacity"] = (
        (df.get("cwip_conversion", pd.Series(0, index=df.index)) > 0) &
        (df.get("fixed_assets", pd.Series(0, index=df.index)) > df.get("fixed_assets_1yb", pd.Series(0, index=df.index)))
    ).astype(int)

    # Operating Leverage Inflection (PAT growing faster than Revenue + margin acceleration)
    df["cat_oplev"] = (
        (df.get("sales_profit_conversion", pd.Series(0, index=df.index)) > 5) & 
        (df.get("opm_acceleration", pd.Series(0, index=df.index)) > 0)
    ).astype(int)

    # Institutional Discovery (Heavy volume + DII/FII buying early)
    df["cat_inst_discovery"] = (
        (df.get("vol_ratio", pd.Series(0, index=df.index)) >= 1.5) &
        (df.get("inst_convergence", pd.Series(0, index=df.index)) == 1) &
        (df.get("fii_holdings", pd.Series(100, index=df.index)) < 15)  # Room to grow
    ).astype(int)
    
    # Debt Deleveraging Cycle
    df["cat_deleveraging"] = (
        (df.get("debt_slope_3y", pd.Series(0, index=df.index)) < 0) &
        (df.get("debt_to_equity_1yb", pd.Series(0, index=df.index)) > 0.5) &
        (df.get("debt_to_equity", pd.Series(1, index=df.index)) <= 0.5)
    ).astype(int)

    # Count total active catalysts
    df["catalyst_count"] = df["cat_capacity"] + df["cat_oplev"] + df["cat_inst_discovery"] + df["cat_deleveraging"]

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
        p_score = _zone_score(df["peg"].clip(lower=0), PEG_ZONES).fillna(50)
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

    # ── God Screen: Frame Tagging ──
    def _build_frameworks(row):
        fw = []
        # 1. QGLP (Raamdeo)
        if row.get("qglp_pass", 0) == 1:
            fw.append("QGLP")
            
        # 2. Coffee Can (Saurabh) - High ROCE & Rev Growth consistency
        roce_med = row.get("roce_med_10y", row.get("roce_med_5y", 0))
        rev_gr = row.get("rev_gr_10y", row.get("rev_gr_5y", 0))
        if pd.notna(roce_med) and pd.notna(rev_gr) and roce_med >= 15 and rev_gr >= 10:
            fw.append("Coffee Can")
            
        # 3. Magic Formula (Greenblatt) - High Earnings Yield + High ROCE
        ey = row.get("earnings_yield", 0)
        roce = row.get("roce", 0)
        if pd.notna(ey) and pd.notna(roce) and ey >= 8 and roce >= 20:
            fw.append("Magic Formula")
            
        # 4. SMILE (Maheshwari) - Small, Medium Exp, Initiating, Large Asp, Extra Scale
        mcap = row.get("market_cap", 0)
        pat_gr = row.get("pat_gr_5y", 0)
        if pd.notna(mcap) and pd.notna(pat_gr) and mcap < 15000 and pat_gr >= 20 and roce >= 20:
            fw.append("SMILE")
            
        # 5. Lynch Dream (Peter Lynch) - PEG < 1, Fast growth, low debt
        peg = row.get("peg", 999)
        debt = row.get("debt_to_equity", 999)
        if pd.notna(peg) and pd.notna(debt) and 0 < peg <= 1.0 and debt < 0.5 and pat_gr >= 15:
            fw.append("Lynch Dream")
            
        return ", ".join(fw) if fw else "None"
        
    df["frameworks_passed"] = df.apply(_build_frameworks, axis=1)

    return df


# ═══════════════════════════════════════════════════════════════
# WAVE DETECTION: MARKET REGIME AWARENESS
# ═══════════════════════════════════════════════════════════════

def detect_market_regime(df: pd.DataFrame) -> str:
    """Auto-detect market regime from breadth of CRS data."""
    if "crs_50d" in df.columns:
        breadth = (df["crs_50d"] > 0).mean()
        if breadth > 0.60:
            return "BULL"
        elif breadth < 0.40:
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
