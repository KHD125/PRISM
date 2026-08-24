"""
test_entry_timing_calibration.py
================================
Contract for the 2026-06-12 O'Neil/Minervini entry-timing audit fixes.

FIX 1 — vcp_volume_dryup needs Minervini's "dramatic" contraction:
  Old: vol_sma_10d < vol_sma_50d — true ~half the time for any stock (coin flip,
  fired for 61% of the universe). Minervini (Trade Like a Stock Market Wizard,
  converted p.~9389): "volume dries up DRAMATICALLY, accompanied by tightness in
  price"; p.~8854: "volume dries up considerably". New: 10D average must be below
  70% of the 50D average — a real 30%+ contraction, not noise.

FIX 2 — Tsunami quality bar 70 -> 65:
  The 7-condition technical+governance alignment leaves ~12 candidates on live
  data; quality_score >= 70 then kills ALL of them (post-GRUESOME-penalty quality
  median is ~31, so 70 = top ~6% AND perfect alignment simultaneously -> dead
  signal, 0 of 2107 forever). At 65 the signal fires for ~2 elite stocks — the
  'rarest, highest-conviction' design intent, alive.

Run with: pytest tests/test_entry_timing_calibration.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np

from test_data_quality_fixes import _frame
from data_engine import compute_derived_signals
from scoring_engine import detect_catalysts_and_tsunami


# ═══════════════════════════════════════════════════════════════════════════
# FIX 1 — VCP volume dryup materiality
# ═══════════════════════════════════════════════════════════════════════════

def test_vcp_noise_dip_does_not_fire():
    """10D vol at 96% of 50D vol is random fluctuation — Minervini's VCP needs
    volume drying up 'dramatically', not a coin flip."""
    out = compute_derived_signals(_frame(vol_sma_10d=48.0, vol_sma_50d=50.0))
    assert (out["vcp_volume_dryup"] == 0).all()


def test_vcp_dramatic_contraction_fires():
    """10D vol at 60% of 50D vol = genuine supply exhaustion in the base."""
    out = compute_derived_signals(_frame(vol_sma_10d=30.0, vol_sma_50d=50.0))
    assert (out["vcp_volume_dryup"] == 1).all()


def test_vcp_missing_volume_no_dryup():
    out = compute_derived_signals(_frame(vol_sma_10d=np.nan, vol_sma_50d=50.0))
    assert (out["vcp_volume_dryup"] == 0).all()


# ═══════════════════════════════════════════════════════════════════════════
# FIX 2 — Tsunami quality bar
# ═══════════════════════════════════════════════════════════════════════════

def _tsunami_frame(quality: float) -> pd.DataFrame:
    """All 7 technical/governance conditions aligned; quality under test."""
    n = 10
    return pd.DataFrame({
        "gate_pass":        [1] * n,
        "above_sma200":     [1] * n,
        "vstop_green":      [1] * n,
        "vstop_fresh":      [1] * n,
        "promoter_buying":  [1] * n,
        "change_fii_lq":    [0.5] * n,
        "quality_score":    [quality] * n,
        "crs_aligned":      [1] * n,
        "market_cap":       [3000.0] * n,
    })


def test_tsunami_fires_at_quality_66():
    out = detect_catalysts_and_tsunami(_tsunami_frame(66.0))
    assert (out["tsunami_signal"] == 1).all(), (
        "Full 8-way alignment with quality 66 must fire — the old 70 bar made the "
        "signal DEAD (0 of 2107: post-penalty quality median ~31, 70 unreachable "
        "simultaneously with perfect technical alignment)"
    )


def test_tsunami_blocked_below_65():
    out = detect_catalysts_and_tsunami(_tsunami_frame(60.0))
    assert (out["tsunami_signal"] == 0).all(), (
        "Quality 60 must NOT fire — Tsunami stays the rarest, highest-conviction signal"
    )


def test_tsunami_still_requires_all_technicals():
    frame = _tsunami_frame(80.0)
    frame["vstop_green"] = 0
    out = detect_catalysts_and_tsunami(frame)
    assert (out["tsunami_signal"] == 0).all()


# ═══════════════════════════════════════════════════════════════════════════
# FIX 3 — buy_zone_label: below-stop stocks must never be "Perfect Entry"
# (Marks audit 2026-06-12: dist_to_vstop goes NEGATIVE below the stop, and the
# old `dist <= 5` branch labeled broken trends "🟢 Perfect Entry (Low Risk)" —
# the most dangerous technical state wearing the safest label. This also
# polluted the Marks Cycle Shield's Price-vs-Value pillar, which reads the label.)
# ═══════════════════════════════════════════════════════════════════════════

def test_below_stop_is_not_perfect_entry():
    """Price 20% BELOW the volatility stop = trend broken, maximum technical risk."""
    out = compute_derived_signals(_frame(close_price=80.0, vstop_value=100.0))
    assert (out["buy_zone_label"] == "🔻 Below Stop (Trend Broken)").all(), (
        f"Got: {out['buy_zone_label'].iloc[0]} — a stock below its stop must never "
        "be labeled Perfect Entry"
    )


def test_just_above_stop_is_perfect_entry():
    """Price 3% above the stop = the genuine asymmetric risk/reward zone."""
    out = compute_derived_signals(_frame(close_price=103.0, vstop_value=100.0))
    assert (out["buy_zone_label"] == "🟢 Perfect Entry (Low Risk)").all()


def test_extended_far_above_stop():
    out = compute_derived_signals(_frame(close_price=130.0, vstop_value=100.0))
    assert (out["buy_zone_label"] == "🔴 Extended (Wait for Pullback)").all()


def test_d45_missing_ma_data_scores_zero_not_free_points():
    """SEPA Trend Template components must fail conservatively on missing MA data.
    Old bug: fillna(0) inside the comparisons inverted conservatism — a stock with
    MISSING sma_30w got a free C2 point (close > 0 is always true), and missing
    sma_200d gave a free C3 point. 32 + 54 stocks were affected on live data."""
    out = compute_derived_signals(_frame(
        close_price=100.0, sma_200d=90.0, sma_50d=95.0, sma_30w=np.nan,
    ))
    # above_sma200=1; C2 (close>30W)=0 NaN; C3 (30W>200D)=0 NaN; C5=0 NaN; vstop=0
    assert (out["d45_trend_structure"] == 1).all(), (
        f"Got {out['d45_trend_structure'].iloc[0]} — missing sma_30w must zero "
        "C2/C3/C5, not award free points"
    )


def test_d45_full_right_stacking_scores_four():
    """All MAs present and right-stacked (50D > 150D > 200D, price above all)."""
    out = compute_derived_signals(_frame(
        close_price=100.0, sma_200d=90.0, sma_50d=95.0, sma_30w=92.0,
    ))
    # above_sma200=1 + C2=1 + C3=1 + C5=1 + vstop_green=0 (no vstop in frame)
    assert (out["d45_trend_structure"] == 4).all()


def test_marks_price_value_pillar_rejects_below_stop():
    """The Marks Shield P pillar reads the label — broken trends must fail it."""
    from scoring_engine import compute_qglp_score
    df = compute_derived_signals(_frame(close_price=80.0, vstop_value=100.0))
    out = compute_qglp_score(df)
    assert (out["marks_price_value"] == 0).all(), (
        "Marks 'price below value' pillar must not award its check to a stock "
        "trading below its volatility stop"
    )


# ═══════════════════════════════════════════════════════════════════════════
# FIX 4 (2026-08-24) — the 12–25% band gap: "⚪ Uncharted" was lying to 650 stocks
# The np.select arms were [<0, ≤5, ≤12, >25] — nothing covered 12–25%, so 30.7%
# of the live universe (650 stocks, every one with a VALID dist_to_vstop) fell to
# the default "⚪ Uncharted", which the Reference tab explicitly defines as
# "missing price/volatility data". A healthy in-trend cushion read as a data hole.
# The fix adds an explicit ≤25 arm — "🟠 Loose Entry Zone" — and reserves the
# default for genuine NaN, restoring the documented meaning of Uncharted.
# ═══════════════════════════════════════════════════════════════════════════

def test_12_to_25_band_is_loose_entry_not_uncharted():
    """18% above the stop: valid data, in-trend, wide risk-to-stop — a real verdict, not a hole."""
    out = compute_derived_signals(_frame(close_price=118.0, vstop_value=100.0))
    lbl = out["buy_zone_label"].iloc[0]
    assert "Uncharted" not in lbl, "a stock with a VALID stop distance must never read Uncharted"
    assert lbl == "🟠 Loose Entry Zone", f"12-25% band must be Loose Entry Zone, got {lbl!r}"


def test_uncharted_reserved_for_genuinely_missing_stop():
    """No volatility stop at all -> the default fires, and ONLY then (its documented meaning)."""
    out = compute_derived_signals(_frame(close_price=118.0, vstop_value=np.nan))
    assert (out["buy_zone_label"] == "⚪ Uncharted").all()


def test_band_boundaries_are_exhaustive_and_ordered():
    """Every band edge lands where the ladder says: 5/12/25 boundaries inclusive-below."""
    cases = [(104.9, "🟢 Perfect Entry (Low Risk)"), (105.0, "🟢 Perfect Entry (Low Risk)"),
             (112.0, "🟡 Standard Zone"), (112.1, "🟠 Loose Entry Zone"),
             (125.0, "🟠 Loose Entry Zone"), (125.1, "🔴 Extended (Wait for Pullback)")]
    for close, want in cases:
        out = compute_derived_signals(_frame(close_price=close, vstop_value=100.0))
        got = out["buy_zone_label"].iloc[0]
        assert got == want, f"close={close}: want {want!r}, got {got!r}"


def test_no_buy_zone_label_trips_the_verdict_timing_veto_except_below_stop():
    """LANDMINE GUARD: verdict_engine's timing_poor is a SUBSTRING match —
    bz.str.contains("Below|Avoid|Overextend|Stop") — so a label merely CONTAINING 'Stop'
    (e.g. a candidate name '🔵 Above Stop' once floated for this band) would silently
    soft-downgrade every BUY in that band to WATCH. Exactly ONE label may match: the
    genuine '🔻 Below Stop (Trend Broken)'. Pins every label in the data_engine ladder."""
    import re
    LABELS = ["🔻 Below Stop (Trend Broken)", "🟢 Perfect Entry (Low Risk)", "🟡 Standard Zone",
              "🟠 Loose Entry Zone", "🔴 Extended (Wait for Pullback)", "⚪ Uncharted"]
    src = open(os.path.join(os.path.dirname(__file__), "..", "core", "data_engine.py"),
               encoding="utf-8").read()
    for lbl in LABELS:
        assert lbl in src, f"ladder label missing from data_engine: {lbl!r}"
    veto = re.compile("Below|Avoid|Overextend|Stop", re.IGNORECASE)
    trippers = [l for l in LABELS if veto.search(l)]
    assert trippers == ["🔻 Below Stop (Trend Broken)"], (
        f"labels tripping the verdict timing veto: {trippers} — only Below Stop may. "
        "A new/renamed band label must avoid the substrings Below/Avoid/Overextend/Stop."
    )


# ═══════════════════════════════════════════════════════════════════════════
# Execution Strip substrate (2026-08-24): fair_value_qglp — engine-materialized
# (never derived in the display layer), and undefined for loss-makers.
# ═══════════════════════════════════════════════════════════════════════════

def test_fair_value_qglp_guards():
    """fair_value = fair_pe_qglp × EPS, ONLY when EPS is reported and positive — a loss-maker's
    earnings-multiple 'fair value' is undefined (a negative target is nonsense), so it propagates
    NaN and the tearsheet renders an honest em-dash."""
    out = compute_derived_signals(_frame(eps=[10.0, -4.0, np.nan] + [10.0] * 22,
                                         pat_gr_5y=20.0, roce_med_10y=30.0))
    fv, fpe = out["fair_value_qglp"], out["fair_pe_qglp"]
    assert np.isclose(fv.iloc[0], round(fpe.iloc[0] * 10.0, 2)), "positive EPS -> fair PE × EPS"
    assert np.isnan(fv.iloc[1]), "negative EPS (loss-maker) must propagate NaN, never a negative target"
    assert np.isnan(fv.iloc[2]), "missing EPS must propagate NaN"
