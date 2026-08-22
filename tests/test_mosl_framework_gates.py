"""
test_mosl_framework_gates.py
============================
Contract for the three MOSL framework-gate inputs that had ZERO test coverage
(Phase-2 gate audit, 2026-08-22): the SQGLP five-factor screen behind
fw_sqglp (🏛️ Fwk 27), the 100x candidate screen behind fw_mosl_100x (🐘 Fwk 35),
and the CAP/GAP longevity flags behind fw_cap_gap (📐 Fwk 28).

The audit found the gate layer SOUND — every screen uses conservative `.fillna()`
at the final flag coordinate (missing data can only fail a criterion, never pass
it). These tests pin that property plus each screen's documented thresholds, so a
future edit cannot silently loosen a gate or flip a fallback's direction.

Threshold provenance (in code comments, book-verified 2026-06-13):
  SQGLP (19th WCS): S < ₹5,000 Cr (the USD 500M small base — the 10x unit bug that
    once let 92% of the universe pass S is pinned here) · Q = ROCE≥15 & ROE≥15 &
    CFO/PAT≥70 · G = PAT-5Y≥20 & Rev-5Y≥15 · L = 10Y growth≥12 · P = PE≤15;
    century_stock_flag = score ≥ 4.
  100x (19th WCS): ALL FIVE of PAT-5Y≥20, ROCE≥20, mcap≤₹15,000 Cr, D/E<0.5, ROE≥15.
  CAP/GAP (22nd WCS): CAP = 10Y & 5Y medians and CURRENT ROCE all ≥ COST_OF_EQUITY
    (the hurdle is the config constant, not a hardcoded 10); GAP = PAT growth ≥15%
    across ALL of 10Y/5Y/3Y.

Run with: pytest tests/test_mosl_framework_gates.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from config import COST_OF_EQUITY
from data_engine import compute_derived_signals
from test_data_quality_fixes import _frame


def _one(**over):
    """One-stock frame through the real derived-signal engine."""
    return compute_derived_signals(_frame(n=1, **over)).iloc[0]


# A stock that passes every SQGLP/100x/CAP-GAP criterion — the shared healthy base.
_PASS_ALL = dict(
    market_cap=[3_000.0], roce=[25.0], roe=[25.0], cfo_to_pat=[85.0],
    pat_gr_5y=[25.0], rev_gr_5y=[20.0], pat_gr_10y=[18.0], pat_gr_3y=[22.0],
    pe=[12.0], debt_to_equity=[0.2],
    roce_med_10y=[22.0], roce_med_7y=[22.0], roce_med_5y=[22.0], roce_1yb=[22.0],
)


# ══════════════════════════════════════════════════════════════════════
# SQGLP (century_stock_flag) — five factors, score ≥ 4
# ══════════════════════════════════════════════════════════════════════

def test_sqglp_all_five_letters_fire_on_the_full_profile():
    r = _one(**_PASS_ALL)
    assert (r["sqglp_s"], r["sqglp_q"], r["sqglp_g"], r["sqglp_l"], r["sqglp_p"]) == (1, 1, 1, 1, 1)
    assert r["sqglp_score"] == 5 and r["century_stock_flag"] == 1


def test_sqglp_size_gate_rejects_the_large_cap():
    """The 2026-06-13 unit-bug pin: S must fail at ₹40,000 Cr (the old <40k line let
    92% of the universe pass and made S dead weight)."""
    r = _one(**{**_PASS_ALL, "market_cap": [40_000.0]})
    assert r["sqglp_s"] == 0
    assert r["century_stock_flag"] == 1, "score 4/5 without S must still be a century candidate"


def test_sqglp_two_missing_letters_drop_below_the_bar():
    r = _one(**{**_PASS_ALL, "market_cap": [40_000.0], "pe": [30.0]})
    assert r["sqglp_score"] == 3 and r["century_stock_flag"] == 0


def test_sqglp_missing_data_fails_the_letter_never_passes_it():
    """Conservative-fallback contract: NaN quality inputs → Q = 0, not 1."""
    r = _one(**{**_PASS_ALL, "roce": [np.nan], "roe": [np.nan], "cfo_to_pat": [np.nan]})
    assert r["sqglp_q"] == 0


def test_sqglp_missing_pe_fails_the_price_letter():
    r = _one(**{**_PASS_ALL, "pe": [np.nan]})
    assert r["sqglp_p"] == 0


# ══════════════════════════════════════════════════════════════════════
# 100x candidate — all five gates mandatory (AND, deliberately rare)
# ══════════════════════════════════════════════════════════════════════

def test_100x_full_profile_passes():
    assert _one(**_PASS_ALL)["mosl_100x_candidate"] == 1


def test_100x_any_single_gate_failure_rejects():
    for knock in (dict(pat_gr_5y=[15.0]), dict(roce=[18.0]), dict(market_cap=[16_000.0]),
                  dict(debt_to_equity=[0.8]), dict(roe=[12.0])):
        r = _one(**{**_PASS_ALL, **knock})
        assert r["mosl_100x_candidate"] == 0, f"should reject on {list(knock)[0]}"


def test_100x_missing_leverage_data_rejects():
    """D/E unknown → fillna(999) → cannot pass the unlevered gate."""
    assert _one(**{**_PASS_ALL, "debt_to_equity": [np.nan]})["mosl_100x_candidate"] == 0


# ══════════════════════════════════════════════════════════════════════
# CAP / GAP longevity (22nd WCS) — hurdle is the config constant
# ══════════════════════════════════════════════════════════════════════

def test_cap_and_gap_extended_on_the_full_profile():
    r = _one(**_PASS_ALL)
    assert r["cap_extended_flag"] == 1 and r["gap_extended_flag"] == 1


def test_cap_hurdle_is_cost_of_equity_not_a_hardcoded_ten():
    """ROCE at CoE−1 everywhere must fail CAP: under a hardcoded 10 hurdle, 11%-ROCE
    (a value destroyer at CoE=12) would count as an extended advantage period."""
    below = COST_OF_EQUITY - 1.0
    r = _one(**{**_PASS_ALL, "roce": [below], "roce_1yb": [below],
                "roce_med_5y": [below], "roce_med_7y": [below], "roce_med_10y": [below]})
    assert r["cap_extended_flag"] == 0
    assert r["cap_years_proxy"] == 0


def test_gap_requires_all_three_growth_windows():
    r = _one(**{**_PASS_ALL, "pat_gr_10y": [10.0]})     # long window below 15
    assert r["gap_extended_flag"] == 0


def test_cap_missing_history_counts_zero_years_not_five():
    r = _one(**{**_PASS_ALL, "roce": [np.nan], "roce_1yb": [np.nan],
                "roce_med_5y": [np.nan], "roce_med_7y": [np.nan], "roce_med_10y": [np.nan]})
    assert r["cap_years_proxy"] == 0 and r["cap_extended_flag"] == 0
