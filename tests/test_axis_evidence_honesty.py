"""Contract: the verdict scorecard's axis pills never claim a verdict built on ZERO evidence.

The rank helpers (_sector_pct_rank / _pct_rank_w) neutral-fill NaN to 50 INSIDE themselves, so a
stock with every moat/growth input missing still receives a real-looking score of exactly 50 —
16 moat-blind and 25 growth-blind rows on the 2026-08-23 live frame (Ather Energy: "Moat 🟡 50"
directly above an all-dash evidence line). The engine's verdict layer already had the honest
⚪ N/A path (_band_num) — it just never fired, because the fabricated 50 arrives as a real number.

The fix is DISPLAY-ONLY, in two parts, both pinned here:
  1. scoring_engine emits `moat_signals_available` / `growth_signals_available` — evidence counts
     derived from the SAME weight-dict constants the scorers iterate (single source, cannot drift).
     Zero score arithmetic touched (bit-identity proven by before/after pickle, 2026-08-23).
  2. verdict_engine masks the axis pill to ⚪ N/A when its count is 0, and brands forensics
     "⚪ Unverified" when zero flags fired on Very-Low coverage (the existing <40 band — zero
     flags there means UNEVALUABLE, not clean: CLAUDE.md §5 "unverifiable is not passed").
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from core.scoring_engine import (BALANCE_SIGNAL_COLS, GROWTH_Q_COLS, GROWTH_SIGNAL_WEIGHTS,
                                 MOAT_SIGNAL_WEIGHTS, VALUATION_SIGNAL_COLS, run_full_scoring)
from core.verdict_engine import compute_verdict

_DEFAULTS = {
    "conviction_tier": 1, "composite_score": 90.0,
    "forensic_score": 90.0, "schilit_pass": 1, "red_flag_count": 1,
    "corporate_class": "🏆 GREAT",
    "quality_score": 90.0, "growth_score": 80.0, "moat_score": 70.0,
    "expected_excess_return": 20.0, "pe": 20.0, "fair_pe_qglp": 30.0,
    "buy_zone_label": "Accumulate",
    "governance_risk_multiplier": 1.0, "data_coverage_pct": 90.0,
}


def _frame(**overrides):
    return pd.DataFrame([{**_DEFAULTS, **overrides}])


# ── 1. the single-source constants (anti-drift substrate) ─────────────────────
def test_signal_weight_constants_are_the_scorers_actual_inputs():
    """The availability count and the score must share ONE signal list. The constants' weights
    must sum to the scorers' documented budgets (moat 1.00; growth 0.94 + 0.06 quarterly), and
    the scorer bodies must iterate the constants — not a private literal that can drift."""
    import inspect

    from core import scoring_engine as se

    assert abs(sum(MOAT_SIGNAL_WEIGHTS.values()) - 1.00) < 1e-9
    assert abs(sum(GROWTH_SIGNAL_WEIGHTS.values()) - 0.94) < 1e-9
    assert GROWTH_Q_COLS == ("q_pat_yoy", "q_rev_yoy")
    assert "MOAT_SIGNAL_WEIGHTS" in inspect.getsource(se._compute_moat_score)
    assert "GROWTH_SIGNAL_WEIGHTS" in inspect.getsource(se._compute_growth_score)


def test_availability_counts_are_emitted_and_correct():
    """run_full_scoring must emit both counts, equal to the notna tally over the constant lists —
    asserted on the OUTPUT frame (the counts' own contract), through the codified synthetic-frame
    path (§6: _frame → compute_derived_signals → compute_forensic_signals → run_full_scoring).
    Row 0 stays fully NaN (the _frame default) = the Ather shape: every derived moat/growth signal
    absent, yet the neutral-filled scores still real-looking numbers — count must read 0."""
    from data_engine import compute_derived_signals
    from forensic_engine import compute_forensic_signals
    from test_data_quality_fixes import _frame

    n = 8
    f = _frame(
        n=n,
        # give rows 1+ real long-history inputs so derivation produces live moat/growth signals
        roce=[np.nan] + [18.0 + i for i in range(n - 1)],
        roce_med_10y=[np.nan] + [15.0 + i for i in range(n - 1)],
        roe_med_10y=[np.nan] + [14.0 + i for i in range(n - 1)],
        pat_gr_5y=[np.nan] + [12.0 + i for i in range(n - 1)],
        rev_gr_5y=[np.nan] + [10.0 + i for i in range(n - 1)],
    )
    # row 0's pat/pbt/ebitda defaults could feed derived growth signals — blank them so it is
    # genuinely evidence-free on BOTH axes end to end
    for c in ("pat", "pbt", "ebitda"):
        f.loc[0, c] = np.nan
    out = run_full_scoring(compute_forensic_signals(compute_derived_signals(f)))

    for col, srcs in [("moat_signals_available", list(MOAT_SIGNAL_WEIGHTS)),
                      ("growth_signals_available", [*GROWTH_SIGNAL_WEIGHTS, *GROWTH_Q_COLS])]:
        assert col in out.columns, f"{col} not emitted by run_full_scoring"
        expect = out.reindex(columns=srcs).notna().sum(axis=1).astype(int)
        assert list(out[col]) == list(expect), f"{col} != notna tally over its constant list"

    # Address rows BY NAME — run_full_scoring re-sorts by composite and resets the index, so
    # positional .loc[0] is the top-ranked stock, not the fixture's row 0. (That artifact briefly
    # masqueraded as a sector-median imputation bug during this test's own development.)
    byname = out.set_index("name")
    assert int(byname.loc["Test Co 0", "moat_signals_available"]) == 0, "all-NaN row must count 0"
    assert int(byname.loc["Test Co 0", "growth_signals_available"]) == 0, "all-NaN row must count 0"
    assert int(byname.loc["Test Co 3", "moat_signals_available"]) > 0, "evidenced row must count > 0"

    # end-to-end: the same all-NaN row's PILLS read ⚪ N/A after compute_verdict
    v = compute_verdict(out).set_index("name")
    assert "N/A" in v.loc["Test Co 0", "verdict_axis_moat"]
    assert "N/A" in v.loc["Test Co 0", "verdict_axis_growth"]
    assert "N/A" not in v.loc["Test Co 3", "verdict_axis_moat"]


# ── 2. the pill masks to ⚪ N/A on zero evidence — and ONLY then ──────────────
def test_blind_axis_masks_to_na():
    df = compute_verdict(_frame(moat_score=50.0, moat_signals_available=0,
                                growth_score=50.0, growth_signals_available=0))
    assert "⚪" in df["verdict_axis_moat"].iloc[0]
    assert "N/A" in df["verdict_axis_moat"].iloc[0]
    assert "50" not in df["verdict_axis_moat"].iloc[0]
    assert "⚪" in df["verdict_axis_growth"].iloc[0]
    assert "N/A" in df["verdict_axis_growth"].iloc[0]


def test_partially_evidenced_axis_keeps_its_real_score():
    """54 of the 79 growth-blind-on-annual-lines stocks have OTHER live signals (quarterly, accel)
    — their scores are partially earned and must NOT be masked. One signal is enough to show."""
    df = compute_verdict(_frame(moat_score=62.0, moat_signals_available=1,
                                growth_score=71.0, growth_signals_available=2))
    assert "62" in df["verdict_axis_moat"].iloc[0]
    assert "🟢" in df["verdict_axis_moat"].iloc[0]
    assert "71" in df["verdict_axis_growth"].iloc[0]


def test_missing_availability_columns_keep_old_behavior():
    """Frames without the new columns (old snapshots, minimal test fixtures) must render exactly
    as before — the mask defaults OFF, never ON."""
    df = compute_verdict(_frame(moat_score=62.0))
    assert "62" in df["verdict_axis_moat"].iloc[0]
    assert "N/A" not in df["verdict_axis_moat"].iloc[0]


# ── 3. forensics: zero flags on Very-Low coverage is UNVERIFIED, not clean ────
def test_forensics_unverified_when_unevaluable():
    df = compute_verdict(_frame(red_flag_count=0, data_coverage_pct=25.0))
    assert "⚪" in df["verdict_axis_forensics"].iloc[0]
    assert "Unverified" in df["verdict_axis_forensics"].iloc[0]
    assert "Clean" not in df["verdict_axis_forensics"].iloc[0]


def test_forensics_clean_needs_coverage():
    """Zero flags WITH real coverage stays 🟢 Clean — the fix must not smear the 43 genuinely
    clean live rows."""
    df = compute_verdict(_frame(red_flag_count=0, data_coverage_pct=80.0))
    assert "🟢" in df["verdict_axis_forensics"].iloc[0]
    assert "Clean" in df["verdict_axis_forensics"].iloc[0]


def test_forensics_fired_flags_outrank_unverified():
    """Flags that DID fire are evidence — thin coverage must not soften a Watch/Flagged verdict."""
    watch = compute_verdict(_frame(red_flag_count=6, data_coverage_pct=25.0))
    assert "🟡" in watch["verdict_axis_forensics"].iloc[0]
    flagged = compute_verdict(_frame(red_flag_count=0, data_coverage_pct=25.0, forensic_score=40.0))
    assert "🔴" in flagged["verdict_axis_forensics"].iloc[0]


# ── 4. valuation + balance: the remaining two numeric axes (guarded 2026-08-23) ──────────
# Balance was the LARGEST fabrication cohort (87 live rows showing "Balance 🔴 36" — a red
# ACCUSATION built purely from neutral fills) and it hid behind two never-NaN sentinel columns
# (net_debt_negative, cwip_conversion) that made the naive all-NaN blindness test vacuous.


def test_valuation_balance_constants_name_real_scorer_inputs():
    """Anti-drift for the two sequential-block scorers (no weight dict to share): every column in
    the constant must appear in its scorer's source, so a renamed/removed signal fails here."""
    import inspect

    from core import scoring_engine as se

    val_src = inspect.getsource(se._compute_valuation_score)
    for c in VALUATION_SIGNAL_COLS:
        assert c in val_src, f"{c} not in _compute_valuation_score — constant drifted"
    bal_src = inspect.getsource(se._compute_balance_sheet_score)
    for c in BALANCE_SIGNAL_COLS:
        assert c in bal_src, f"{c} not in _compute_balance_sheet_score — constant drifted"
    # the sentinels must stay EXCLUDED (they would make the blind test vacuous)
    assert "net_debt_negative" not in BALANCE_SIGNAL_COLS
    assert "cwip_conversion" not in BALANCE_SIGNAL_COLS


def test_blind_valuation_and_balance_mask_to_na():
    df = compute_verdict(_frame(valuation_score=46.0, valuation_signals_available=0,
                                balance_sheet_score=36.1, balance_signals_available=0))
    assert "N/A" in df["verdict_axis_valuation"].iloc[0]
    assert "⚪" in df["verdict_axis_valuation"].iloc[0]
    assert "N/A" in df["verdict_axis_balance"].iloc[0]
    assert "36" not in df["verdict_axis_balance"].iloc[0]
    # one real signal keeps the pill — and absent count columns keep old behavior
    keep = compute_verdict(_frame(valuation_score=46.0, valuation_signals_available=1,
                                  balance_sheet_score=61.1, balance_signals_available=1))
    assert "46" in keep["verdict_axis_valuation"].iloc[0]
    assert "61" in keep["verdict_axis_balance"].iloc[0]
    old = compute_verdict(_frame(valuation_score=46.0, balance_sheet_score=36.1))
    assert "46" in old["verdict_axis_valuation"].iloc[0]
    assert "36" in old["verdict_axis_balance"].iloc[0]


def test_net_cash_counts_as_genuine_balance_evidence():
    """net_debt_negative == 1 requires real debt/cash figures, so it counts as evidence and keeps
    the pill; a 0 proves nothing (data_engine emits 0 for missing inputs) and must NOT count.
    Exercised through the real pipeline path so the ==1 term itself is under test."""
    import numpy as np

    from data_engine import compute_derived_signals
    from forensic_engine import compute_forensic_signals
    from test_data_quality_fixes import _frame as _dq_frame

    f = _dq_frame(n=6)
    out = run_full_scoring(compute_forensic_signals(compute_derived_signals(f))).set_index("name")
    ranked_nan = out.reindex(columns=list(BALANCE_SIGNAL_COLS)).notna().sum(axis=1)
    ndn = (out["net_debt_negative"] == 1).astype(int)
    assert list(out["balance_signals_available"]) == list((ranked_nan + ndn).astype(int)),         "balance count must be notna(ranked signals) + (net_debt_negative == 1)"


def test_narrative_untouched_by_masking():
    """The masked series feed ONLY the pills: g_hi uses fillna(0), and a fabricated 50 was already
    below the 60 bar, so blind rows keep the same narrative with and without the counts."""
    a = compute_verdict(_frame(growth_score=50.0, moat_score=50.0,
                               valuation_score=46.0, balance_sheet_score=36.1))
    b = compute_verdict(_frame(growth_score=50.0, moat_score=50.0,
                               valuation_score=46.0, balance_sheet_score=36.1,
                               growth_signals_available=0, moat_signals_available=0,
                               valuation_signals_available=0, balance_signals_available=0))
    assert a["verdict_narrative"].iloc[0] == b["verdict_narrative"].iloc[0]
    assert a["verdict_direction"].iloc[0] == b["verdict_direction"].iloc[0]
