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
from scoring_engine import compute_qglp_score
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
    roe_med_5y=[25.0], roe_med_10y=[25.0], roe_1yb=[25.0], roe_med_3y=[25.0],
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


def test_cap_is_defined_on_roe_not_roce():
    """22nd WCS, verbatim box: "CAP is RoE > Cost of Equity". A bank-like profile —
    strong RoE windows, ROCE low/meaningless — must pass CAP (the study's own framing
    includes financials; ROCE structurally punishes them)."""
    r = _one(**{**_PASS_ALL, "roce": [6.0], "roce_1yb": [6.0],
                "roce_med_5y": [6.0], "roce_med_7y": [6.0], "roce_med_10y": [6.0]})
    assert r["cap_extended_flag"] == 1


def test_cap_hurdle_is_the_studys_15_not_the_system_12():
    """22nd WCS: "We deem Cost of Equity in India to be about 15%" — RoE at 13.5
    (above the system-wide 12) must still FAIL the study's own bar. Same precedent as
    emerging_vc_flag, which uses the 18th study's 15% while economic_profit uses 12."""
    r = _one(**{**_PASS_ALL, "roe": [13.5], "roe_1yb": [13.5], "roe_med_3y": [13.5],
                "roe_med_5y": [13.5], "roe_med_10y": [13.5]})
    assert r["cap_extended_flag"] == 0
    assert r["cap_years_proxy"] == 0


def test_gap_requires_all_three_growth_windows():
    r = _one(**{**_PASS_ALL, "pat_gr_10y": [10.0]})     # long window below 15
    assert r["gap_extended_flag"] == 0


def test_cap_missing_roe_history_counts_zero_years_not_five():
    r = _one(**{**_PASS_ALL, "roe": [np.nan], "roe_1yb": [np.nan], "roe_med_3y": [np.nan],
                "roe_med_5y": [np.nan], "roe_med_10y": [np.nan]})
    assert r["cap_years_proxy"] == 0 and r["cap_extended_flag"] == 0


# ══════════════════════════════════════════════════════════════════════
# Economic Moat (17th WCS) — sector AVERAGE, and the book's peerless rule
# ══════════════════════════════════════════════════════════════════════
# Methodology p.35, verbatim: "we calculated the average sector RoE for each of the
# 8 years... A company was decided to be an EMC if for at least 6 of the 8 years, its
# RoE was higher than the industry average." And for companies without comparable
# peers: "given their high absolute RoEs, we deem them to enjoy Economic Moat" — the
# engine's proxy for "high absolute" is beating the UNIVERSE average. The old code
# compared against the sector MEDIAN (a ~50% coin-flip per window on a persistent
# metric) and made sole-listed companies permanently ineligible (self-median).

def _emc_frame(sectors, roes):
    over = {"sector": sectors}
    for c in ("roe", "roe_1yb", "roe_med_3y", "roe_med_5y", "roe_med_10y"):
        over[c] = roes                       # constant history per stock
    return compute_derived_signals(_frame(n=len(roes), **over))


def test_emc_compares_against_the_sector_average_not_the_median():
    """Skewed sector: RoEs [10,10,10,10,60,15] → median 10, mean 19.2. The 15%-RoE
    stock beats the median (coin-flip logic) but NOT the book's average — no moat."""
    df = _emc_frame(["Chemicals"] * 6, [10.0, 10.0, 10.0, 10.0, 60.0, 15.0])
    assert int(df["emc_flag"].iloc[5]) == 0


def test_emc_the_outlier_that_drags_the_average_up_still_qualifies():
    df = _emc_frame(["Chemicals"] * 6, [10.0, 10.0, 10.0, 10.0, 60.0, 15.0])
    assert int(df["emc_flag"].iloc[4]) == 1


def test_peerless_high_roe_company_is_eligible():
    """Sole-listed with RoE 30 vs a ~10% universe: the book deems it an EMC; the old
    self-median made it structurally impossible."""
    df = _emc_frame(["Chemicals"] * 5 + ["Lonely Sector"],
                    [8.0, 9.0, 10.0, 11.0, 12.0, 30.0])
    assert int(df["emc_flag"].iloc[5]) == 1


def test_peerless_low_roe_company_is_not_eligible():
    df = _emc_frame(["Chemicals"] * 5 + ["Lonely Sector"],
                    [18.0, 19.0, 20.0, 21.0, 22.0, 5.0])
    assert int(df["emc_flag"].iloc[5]) == 0


# ══════════════════════════════════════════════════════════════════════
# QGLP (25th WCS) — the Q-leg spans the full scale
# ══════════════════════════════════════════════════════════════════════
# The 25th's QGLP Checklist assigns NO numeric weights ("QGL is the Value component
# which is then juxtaposed with P"). The old ×0.7 on the Q percentile capped Quality
# at 80 (live median 35) while Price ran to 100 (live median 85) — inverting the
# book's Q-first, P-as-final-check structure in the displayed score and the Market
# Pulse QGLP ordering. Display/ordering only: qglp_pass gates are independent.

def _qglp_frame(**over):
    import pandas as pd
    base = {
        "company_id": [f"NSE:Q{i}" for i in range(10)],
        "name": [f"Q Co {i}" for i in range(10)],
        "roce": list(np.linspace(5.0, 40.0, 10)),
        "promoter_buying": [0] * 10,
        "pledge_rising":   [0] * 10,
    }
    base.update(over)
    return compute_qglp_score(pd.DataFrame(base))


def test_quality_leg_spans_the_full_scale():
    q = _qglp_frame()["qglp_quality"]
    assert q.max() >= 95.0, f"top-ROCE stock capped at {q.max():.0f} — the ×0.7 is back"


def test_quality_leg_promoter_adjustments_still_apply():
    df = _qglp_frame(pledge_rising=[0] * 9 + [1])
    assert df["qglp_quality"].iloc[9] <= 92.0     # top stock docked ~10 for rising pledge
    df2 = _qglp_frame(promoter_buying=[0] * 9 + [1])
    assert df2["qglp_quality"].iloc[9] >= df["qglp_quality"].iloc[9]
