"""
Verdict Engine — display-only decision-synthesis layer.
=======================================================
Fuses the engine's already-computed sub-verdicts (conviction_tier, mauboussin_ev_verdict,
corporate_class, buy_zone_label) and the six quality/risk axes into ONE coherent top-line
verdict: a Direction, a Conviction strength, a Confidence, a plain-English narrative, and the
single most important risk.

PIPELINE POSITION — STEP 4 (after apply_forensic_penalty in app.get_scored_data).
  Reads the POST-penalty composite_score / conviction_tier (apply_forensic_penalty re-derives the
  tier from the penalized score, so these are consistent here — verified 2026-06-14).

INVARIANTS:
  * ZERO new scoring. Every output column is a label/string DERIVED from existing columns.
    composite_score and all scoring columns are read-only here — never reassigned.
  * Fully vectorized (np.select / np.where) — no .apply, no row loops. get_scored_data is
    UNcached (<0.5s/dropdown), so this must stay O(rows) array ops.
  * Defensive: every input via _col() so a missing column degrades gracefully (never KeyErrors).
  * Asymmetric vetoes (consistent with the engine's forensic-multiplier philosophy): a verdict can
    only be capped DOWNWARD by a hard risk, never upgraded. The forensic multiplier already lowered
    the tier, so the 🚨 veto is a SAFETY CAP for the rare high-score-despite-sharp-practices case —
    not a double penalty.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_verdict(df: pd.DataFrame) -> pd.DataFrame:
    """Materialize the verdict_* display columns. Pure synthesis of existing signals."""
    df = df.copy()
    idx = df.index

    def _col(name: str, default):
        """Defensive column access — absent column → constant Series (never KeyError)."""
        return df[name] if name in df.columns else pd.Series(default, index=idx)

    # ── Inputs (all verified present 2026-06-14; _col() keeps it crash-proof regardless) ──
    tier    = _col("conviction_tier", 5).fillna(5).astype(float)
    rflags  = _col("red_flag_count", 0).fillna(0).astype(float)
    cclass  = _col("corporate_class", "").astype(str)
    cov     = _col("data_coverage_pct", 100.0).fillna(0.0)
    q       = _col("quality_score", np.nan)      # overall quality (narrative only — see note)
    g       = _col("growth_score", np.nan)
    eer     = _col("expected_excess_return", np.nan)
    pe      = _col("pe", np.nan)
    fair_pe = _col("fair_pe_qglp", np.nan)
    bz      = _col("buy_zone_label", "").astype(str)
    govmult = _col("governance_risk_multiplier", 1.0).fillna(1.0)
    # Orthogonal scorecard axes — the engine's INDEPENDENT sub-scores, NOT the quality_score
    # composite (which already contains growth 22% + valuation 10% → showing Quality+Growth+Value
    # would triple-count). Polarity verified 2026-06-14: all high=good (valuation_score corr w/ PE
    # is −0.14 → high = cheap = good).
    moat    = _col("moat_score", np.nan)
    valn    = _col("valuation_score", np.nan)
    bal     = _col("balance_sheet_score", np.nan)

    # ── Hard-risk veto masks (cap downward only) ──
    # CALIBRATED 2026-06-14: forensic_label is "🚨" for 98.6% of the universe (only 29 "Clean") →
    # useless as a veto. red_flag_count median is ~4-5 → "several flags" is the NORM. So the AVOID
    # veto uses genuinely SELECTIVE severe signals; the forensic multiplier already discounted
    # moderate-flag stocks into lower tiers (the 32 Tier-1-2 stocks are already clean by construction).
    fscore        = _col("forensic_score", 100.0).fillna(100.0)
    schilit_fail  = _col("schilit_pass", 1).fillna(1).astype(float) == 0   # Schilit checker hard-fail
    forensic_veto = (fscore < 50.0) | (rflags >= 10)        # severe accounting floor / extreme flags
    gruesome_veto = cclass.str.contains("GRUESOME", case=False)
    gov_veto      = govmult < 0.85               # governance risk multiplier materially below 1.0
    timing_poor   = bz.str.contains("Below|Avoid|Overextend|Stop", case=False)

    # ── Direction: base from post-penalty tier, then asymmetric caps ──
    base = np.select([tier <= 2, tier == 3], ["BUY", "WATCH"], default="AVOID")
    is_buy = pd.Series(base == "BUY", index=idx)
    _soft  = (schilit_fail | gov_veto | timing_poor) & is_buy   # soft downgrade BUY → WATCH
    direction = np.where(
        forensic_veto | gruesome_veto, "AVOID",
        np.where(_soft, "WATCH", base),
    )
    df["verdict_direction"] = direction

    df["verdict_emoji"] = np.select(
        [direction == "BUY", direction == "WATCH"], ["🟢", "🟡"], default="🔴"
    )

    # ── Conviction strength (from tier) ──
    df["verdict_strength"] = pd.Series(tier, index=idx).map(
        {1.0: "HIGH CONVICTION", 2.0: "STRONG", 3.0: "EMERGING", 4.0: "SPECULATIVE", 5.0: "WEAK"}
    ).fillna("WEAK")

    # ── Confidence (kept SEPARATE from the score — a high score on thin data is flagged) ──
    df["verdict_confidence"] = np.select(
        [cov >= 80, cov >= 60, cov >= 40], ["High", "Medium", "Low"], default="Very Low"
    )

    # ── Six axis pills (compact display strings) ──
    def _band_num(s: pd.Series, hi: float, mid: float) -> np.ndarray:
        v = s.fillna(-1.0)
        return np.select([v < 0, v >= hi, v >= mid], ["⚪", "🟢", "🟡"], default="🔴")

    def _pill_num(label: str, s: pd.Series, hi: float, mid: float, fmt="{:.0f}"):
        dot = _band_num(s, hi, mid)
        val = s.map(lambda x: fmt.format(x) if pd.notna(x) else "N/A")
        return pd.Series([f"{label} {d} {vv}" for d, vv in zip(dot, val)], index=idx)

    # 4 numeric axes (0-100 sub-scores, 🟢≥60 🟡≥40) + 2 categorical (Governance/Forensics —
    # cleaner than a compressed bonus score). Orthogonal: Moat·Growth·Valuation·Balance·Govern·Forensics.
    df["verdict_axis_moat"]      = _pill_num("Moat",      moat, 60, 40)
    df["verdict_axis_growth"]    = _pill_num("Growth",    g,    60, 40)
    df["verdict_axis_valuation"] = _pill_num("Valuation", valn, 60, 40)
    df["verdict_axis_balance"]   = _pill_num("Balance",   bal,  55, 38)
    df["verdict_axis_forensics"] = np.where(
        forensic_veto | schilit_fail, "Forensics 🔴 Flagged",
        np.where(rflags >= 5, "Forensics 🟡 Watch", "Forensics 🟢 Clean"),
    )
    df["verdict_axis_governance"] = np.select(
        [govmult < 0.85, govmult < 1.0], ["Govern 🔴 Risk", "Govern 🟡 Caution"],
        default="Govern 🟢 Safe",
    )

    # ── Single most important risk (one line, only the top one) ──
    df["verdict_top_risk"] = np.select(
        [forensic_veto, gruesome_veto, schilit_fail, gov_veto, timing_poor, cov < 50],
        [
            "🚨 Severe forensic / accounting-quality flags",
            "💀 Value-destroying capital allocation",
            "🕵️ Schilit forensic checker flags",
            "⚠️ Governance risk (pledge/dilution)",
            "⏳ Poor entry timing — wait for a base",
            "🔍 Thin data — verdict tentative",
        ],
        default="",
    )

    # ── Narrative: deterministic np.select over decision archetypes ──
    # Unambiguous helper signals (no dependence on any score's polarity sign convention):
    cheap  = (pe.notna() & fair_pe.notna() & (pe < fair_pe))                       # price below quality-fair PE
    pricey = (pe.notna() & fair_pe.notna() & (pe > fair_pe * 1.30))               # >30% above fair PE
    q_hi   = q.fillna(0) >= 65
    g_hi   = g.fillna(0) >= 60
    clean  = ~(forensic_veto | gruesome_veto | schilit_fail)
    great  = cclass.str.contains("GREAT", case=False)
    eer_pos = eer.fillna(0) >= 5.0
    # DIRECTION-AWARE so the narrative can NEVER contradict the direction. The positive archetypes
    # describe a good FUNDAMENTAL setup (quality/growth/cheap), but the direction reads the post-
    # penalty composite — so a cheap-and-clean stock whose composite still lands in AVOID (momentum /
    # risk-adjusted quality dragging it — a classic value trap, e.g. Sarda Energy: q71 g66 cheap but
    # composite 42 / Stage-4) must NOT read "high-conviction core holding". AVOID is matched FIRST so
    # it can never fall through to a buy-toned line; the strongest BUY phrasing is reserved for BUYs.
    dir_avoid = pd.Series(direction == "AVOID", index=idx)
    dir_buy   = pd.Series(direction == "BUY",   index=idx)

    df["verdict_narrative"] = np.select(
        [
            forensic_veto,
            gruesome_veto,
            schilit_fail & (tier <= 2),
            # AVOID matched up-front (after the hard-reason vetoes) → never a buy-toned narrative
            dir_avoid & q_hi & (cheap | eer_pos),
            dir_avoid,
            q_hi & g_hi & pricey & clean,
            dir_buy & q_hi & g_hi & (cheap | eer_pos) & clean,   # strongest phrasing: BUY only
            q_hi & g_hi & (cheap | eer_pos) & clean,             # same setup, non-BUY → softer
            q_hi & g_hi & clean,
            (cheap | eer_pos) & ~q_hi & clean,
            q_hi & ~g_hi & clean,
            great & clean,
            cov < 50,
        ],
        [
            "Severe forensic / accounting-quality flags override the thesis — avoid until the accounts are clean.",
            "Capital allocation destroys value (Gruesome) — avoid regardless of how cheap it looks.",
            "Quality screens well, but Schilit forensic checkers flag it — verify the accounts before buying.",
            "Cheap with decent quality on the screens — but the overall score is weak enough to land in AVOID; treat it as a value-trap watch, not a buy.",
            "Overall score lands in AVOID — the weak axes outweigh the strengths; pass until they improve.",
            "Elite franchise, but priced for perfection — wonderful business, demanding price; wait for a pullback.",
            "Elite compounder at a reasonable price — high-conviction core holding.",
            "A high-quality compounder trading at a fair price — quality and value aligned.",
            "Strong quality and growth, fairly valued — a solid compounder.",
            "Statistically cheap, but quality is thin — verify it isn't a value trap before buying.",
            "Quality intact but growth is decelerating — watch the next few prints before adding.",
            "Disciplined capital allocator (Great) — durable compounding, watch the entry price.",
            "Thin data coverage — treat this verdict as tentative until more inputs report.",
        ],
        default="No decisive edge on any axis right now — pass and revisit on a re-rate or result.",
    )

    return df
