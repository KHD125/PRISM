"""
test_dilution_corporate_action.py
=================================
Contract: a CORPORATE ACTION is not dilution.

`dilution_pct` is a raw share-count delta, so a 1:1 BONUS ISSUE (+100%) scored worse than a
predatory 15% QIP. 492 stocks failed the Tier-3 hard gate; 145 had a share multiple within 0.5%
of a clean bonus/split ratio — Nestle India 2.000x, Adani Power 5.000x, Trent 1.500x, Ashok
Leyland 2.000x, Pidilite 2.001x — every one penalised for REWARDING its shareholders. 31 lost a
forensic-multiplier tier to the spurious `rf_dilution` flag, a ~32% composite haircut, and 133
stocks failed the hard gate on this criterion ALONE.

THE RULE: share count >= 1.5x in one year is structural (bonus / split / rights / IPO re-basing),
not ordinary issuance — you cannot place more than half your own share capital with third parties.
Below 1.5x is the real placement zone and stays Tier 3.

The fixtures below are REAL COMPANIES whose corporate actions were verified against public record,
in both directions. They are the point of this file: a threshold with no ground truth behind it is
a guess, and two earlier approaches (clean-ratio matching; the accounting identity dNW - PAT) were
killed by exactly these cases —
  * Maruti Interior 4.000x is as "clean" a ratio as exists, and it is a RIGHTS ISSUE raising cash.
  * Vraj Iron (a confirmed IPO cash raise) reads +3.8% on the accounting identity, BELOW Shilchar's
    bonus at +4.2% — the classes overlap, so no threshold on that measure separates them.

KNOWN FALSE NEGATIVE (contained by design, pinned below): a distressed debt-to-equity conversion
can exceed 1.5x and IS real dilution — Sumeet Industries 6.70x, banks taking equity under an NCLT
plan. It is unreachable: every stock this arm reclassifies already failed `gate_pass` before the
change, and Sumeet is composite ~11, tier 5, 14 red flags, GRUESOME, AVOID.

Run with: pytest tests/test_dilution_corporate_action.py -v
"""

import contextlib
import io as _io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import pytest

from core import run_scoring_pipeline
from data_engine import (coerce_numeric_columns, compute_derived_signals, load_all_csvs,
                         merge_datasets)
from test_data_quality_fixes import _frame

# ── Verified against public record, 2026-08-25 ──────────────────────────────────────────
RECLASSIFY = {                       # pro-rata / structural → must NOT be Tier 3
    "BSE Ltd":                                 "2:1 bonus, record date 23 May 2025",
    "Indraprastha Gas Ltd":                    "1:1 bonus, record date 31 Jan 2025",
    "Shilchar Technologies Ltd":               "1:2 bonus, record date 6 Jun 2025",
    "Samvardhana Motherson International Ltd": "1:2 bonus, record date 18 Jul 2025",
    "Sigma Solve Ltd":                         "1:10 stock split, 2025",
    "New Delhi Television Ltd":                "3:4 rights issue, Sept-Oct 2025",
}
STAYS_DILUTION = {                   # real cash into the company at <1.5x → must STAY Tier 3
    # Vraj Iron RETIRED 2026-08-30: its Jun-2024 IPO aged out of the rolling YoY share-count
    # window on the 2026-08-28 data refresh (equity_shares == equity_shares_1yb == 32,982,619,
    # so dilution_flag = 0 is CORRECT — no new dilution in the trailing year). The case proved
    # the real-cash-raise direction while it was in-window; Saraswati carries it until it ages
    # out too, at which point a fresher raise from the live data must replace it, not a forced
    # re-flagging of an undiluted year.
    "Saraswati Saree Depot Ltd": "IPO Aug 2024",
}


@pytest.fixture(scope="module")
def live():
    with contextlib.redirect_stdout(_io.StringIO()):
        # full pipeline: the containment test reads red_flag_count, a forensic-engine column
        return run_scoring_pipeline(compute_derived_signals(
            coerce_numeric_columns(merge_datasets(load_all_csvs("local"))))).set_index("name")


# ── 1. Ground truth, both directions ────────────────────────────────────────────────────
@pytest.mark.parametrize("name,action", sorted(RECLASSIFY.items()))
def test_verified_corporate_actions_are_not_flagged_as_dilution(live, name, action):
    if name not in live.index:
        pytest.skip(f"{name} not in this universe snapshot")
    row = live.loc[name]
    assert row["dilution_is_corporate_action"] == 1, f"{name} ({action}) not recognised"
    assert row["dilution_flag"] < 3, (
        f"{name} is a {action} — pro-rata, every holder's stake unchanged — but it is still "
        f"flagged Tier-3 predatory dilution (dilution_pct={row['dilution_pct']:.1f}%)"
    )


@pytest.mark.parametrize("name,action", sorted(STAYS_DILUTION.items()))
def test_real_cash_raises_below_the_threshold_stay_flagged(live, name, action):
    if name not in live.index:
        pytest.skip(f"{name} not in this universe snapshot")
    row = live.loc[name]
    assert row["dilution_is_corporate_action"] == 0, f"{name} ({action}) wrongly exonerated"
    assert row["dilution_flag"] == 3, f"{name} raised real cash ({action}) and must stay Tier 3"


# ── 2. The threshold itself ─────────────────────────────────────────────────────────────
def _run(shares_1yb, shares):
    out = compute_derived_signals(_frame(n=2, equity_shares_1yb=[float(shares_1yb)] * 2,
                                         equity_shares=[float(shares)] * 2))
    return out.iloc[0]


@pytest.mark.parametrize("mult,expect_corp", [
    (1.11, False),   # 11% — an ordinary placement, the zone the gate exists for
    (1.30, False),
    (1.49, False),
    (1.50, True),    # inclusive: Shilchar and Samvardhana both land EXACTLY here
    (2.00, True),    # 1:1 bonus
    (10.0, True),    # 1:10 split
])
def test_threshold_is_one_and_a_half_times_inclusive(mult, expect_corp):
    row = _run(1_000_000, round(1_000_000 * mult))
    # bool() is load-bearing: numpy returns np.bool_, for which `x is True` evaluates FALSE —
    # the identity trap that silently greyed out 18 tearsheet elements before it was pinned.
    assert bool(row["dilution_is_corporate_action"]) is expect_corp
    assert bool(row["dilution_flag"] < 3) is expect_corp


def test_tier_one_not_tier_zero_so_nothing_is_certified_zero_dilution():
    """Fisher P13 and Outsider pillar S gate on `dilution_flag == 0`. A bonus issuer DID change
    its share count, so it must never satisfy a 'zero equity dilution' claim."""
    row = _run(1_000_000, 2_000_000)
    assert row["dilution_flag"] == 1, "must be Tier 1 — Tier 0 would falsely certify zero dilution"


def test_lower_tiers_are_untouched():
    for mult, tier in [(0.90, 0), (1.00, 0), (1.02, 1), (1.05, 2)]:
        row = _run(1_000_000, round(1_000_000 * mult))
        assert row["dilution_flag"] == tier, f"multiple {mult} should be Tier {tier}"


# ── 3. Missing data must never manufacture an exoneration ───────────────────────────────
@pytest.mark.parametrize("a,b", [(np.nan, 2_000_000), (1_000_000, np.nan), (0, 2_000_000)])
def test_unusable_share_data_is_never_called_a_corporate_action(a, b):
    """Pins the OUTCOME, not the mechanism — and mutation testing showed why that distinction
    matters here. Deleting the `shares_valid &` guard does NOT fail this test, because
    `dilution_pct` already carries a 0.0 sentinel on invalid rows, so `>= 50` is unreachable
    there anyway. The guard is belt-and-braces (it would start earning its keep the day that
    sentinel is corrected to NaN, per §5). What must never regress is this outcome: absent
    share data cannot produce a corporate-action claim."""
    row = _run(a, b)
    assert row["dilution_is_corporate_action"] == 0, (
        "a corporate action was asserted on share data that cannot support it"
    )


# ── 4. The Tier-1 ESOP deduction must not land on corporate actions ─────────────────────
def test_corporate_actions_are_exempt_from_the_tier1_esop_deduction(live):
    """GOVERNANCE_BONUS["dilution_tier1_minor"] is -5 for "<3% ESOP dilution". Reclassifying
    bonus issues INTO Tier 1 would hand them that deduction — re-applying, in miniature, the
    exact bug being fixed (penalising a company for rewarding its shareholders).

    This is load-bearing, not theoretical: measured on live data, without the exemption the
    reclassification made 244 stocks WORSE (a stock that moves 3->1 sheds the rf_dilution red
    flag, but that only pays if it crosses a forensic-multiplier tier boundary — while the -5
    applied unconditionally). With the exemption: 42 stocks change, all 42 improve, none regress.
    """
    from config import GOVERNANCE_BONUS
    from scoring_engine import compute_governance_bonus

    # Two rows identical except the share change: 2% (real ESOP dilution) vs 100% (a 1:1 bonus).
    # BOTH land in Tier 1 — that is the whole point — so any difference in governance_bonus is
    # attributable to the exemption alone.
    frame = compute_derived_signals(_frame(
        n=2,
        equity_shares_1yb=[1_000_000.0, 1_000_000.0],
        equity_shares=[1_020_000.0, 2_000_000.0],
    ))
    assert frame["dilution_flag"].tolist() == [1, 1], "both rows must be Tier 1 for this to isolate"
    assert frame["dilution_is_corporate_action"].tolist() == [0, 1]

    scored = compute_governance_bonus(frame)
    esop, corp = scored["governance_bonus"].iloc[0], scored["governance_bonus"].iloc[1]
    penalty = float(GOVERNANCE_BONUS["dilution_tier1_minor"])          # -5
    assert corp - esop == pytest.approx(abs(penalty)), (
        f"the ESOP deduction ({penalty}) is landing on corporate actions: bonus-issue row scored "
        f"{corp}, ESOP row {esop} — expected exactly {abs(penalty)} apart"
    )


# ── 5. Piotroski F7 must answer the same question the same way ──────────────────────────
def test_piotroski_f7_exempts_corporate_actions(live):
    """F7 is "the firm did not ISSUE common equity" — an OFFERING. A bonus issue sells nothing
    and raises nothing. Before this fix, `_eq <= _eq_1yb` docked a point from 286 companies for
    handing shareholders free stock, and the engine held two OPPOSITE answers to "did this
    company issue equity?" — dilution_flag saying no, F7 saying yes, on the same rows."""
    corp = live[live["dilution_is_corporate_action"] == 1]
    assert len(corp) > 50, "sanity: the corporate-action arm should be firing on live data"
    assert (corp["f_no_dilution"] == 1).all(), (
        f"{int((corp['f_no_dilution'] == 0).sum())} corporate actions still lose the Piotroski "
        f"F7 point — F7 and dilution_flag disagree about the same event"
    )


def test_piotroski_f7_still_penalises_real_issuance():
    """The exemption must not become a blanket pass: sub-1.5x issuance still fails F7."""
    out = compute_derived_signals(_frame(n=2, equity_shares_1yb=[1_000_000.0] * 2,
                                         equity_shares=[1_200_000.0] * 2))
    from forensic_engine import compute_forensic_signals
    with contextlib.redirect_stdout(_io.StringIO()):
        scored = compute_forensic_signals(out)
    assert scored["dilution_is_corporate_action"].iloc[0] == 0
    assert scored["f_no_dilution"].iloc[0] == 0, "a 20% placement must still fail F7"


def test_piotroski_f7_degrades_safely_without_the_column():
    """Old snapshots and minimal fixtures predate `dilution_is_corporate_action`; F7 must fall
    back to the original comparison rather than exempting everything."""
    from forensic_engine import compute_forensic_signals
    out = compute_derived_signals(_frame(n=2, equity_shares_1yb=[1_000_000.0] * 2,
                                         equity_shares=[2_000_000.0] * 2))
    out = out.drop(columns=["dilution_is_corporate_action"])
    with contextlib.redirect_stdout(_io.StringIO()):
        scored = compute_forensic_signals(out)
    assert scored["f_no_dilution"].iloc[0] == 0, (
        "without the column the exemption must NOT apply — absent evidence is not an exemption"
    )


# ── 6. The contained false negative — pinned so it stays contained ──────────────────────
def test_known_false_negative_remains_rejected_by_other_signals(live):
    """Sumeet Industries: NCLT debt-to-equity, 6.70x — real dilution this rule cannot see.
    It is acceptable ONLY because the engine rejects it on independent grounds."""
    if "Sumeet Industries Ltd" not in live.index:
        pytest.skip("not in this universe snapshot")
    row = live.loc["Sumeet Industries Ltd"]
    assert row["dilution_is_corporate_action"] == 1        # the rule does exonerate it
    assert row["red_flag_count"] >= 8, (
        "the ONLY reason the distressed-restructuring blind spot is tolerable is that these "
        "companies are caught by many other signals — that containment has weakened"
    )
