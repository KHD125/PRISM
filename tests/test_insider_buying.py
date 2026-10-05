"""
test_insider_buying.py
======================
The insider-buying signal, revived 2026-10-05.

PRISM was designed to reward insiders BUYING: GOVERNANCE_BONUS["insider_trading_present"] (15 points)
on the governance bonus, and +2 on D44. Both searched the vendor's "Insider Trading" column for the
TEXT "Bought" — but the vendor sends a SIGNED NUMBER, the net value of recent insider trades
(Ramkrishna Forgings +399.84, AXISCADES -860.01), and the loader keeps it as raw values. So neither
fired for any stock, on any vintage. Found while researching what the evidence says about this data:
insider purchase portfolios earned positive abnormal returns in Indian PIT disclosures (2007-2015).

MEASURED BEFORE REVIVING (four snapshots): the sign agrees with the promoter-holding change on 67-80%
of rows, so + is buying; buyers are not pledge-contaminated (pledge rising 3.9% against 4.8% for the
universe and 9.0% for sellers); from the June snapshot buyers beat the median by +3.3 / +3.5pp over
94-108 days (n=42, hit 57-60%) while sellers sat near 0 — mixed at 66 days, so it is weak evidence
that agrees with the design and the literature, not proof. December's 6-month window decides whether
the bonus stays (docs/december-2026-plan.md, rule 10), measured by tools/validate.py's cohorts.

WHAT IS PINNED
  1. ONE definition — data_engine.insider_net_buyer: a positive number is net buying; a seller, a
     zero, a blank and any TEXT are not (text would mean the vendor changed format — test 6).
  2. The engine derives insider_net_buyer from it, and both consumers read that column: the
     governance bonus pays a net buyer exactly the configured amount, D44 adds exactly 2.
  3. No engine code searches the column for "Bought" any more.
  4. tools/validate.py measures net buyers against net sellers on every snapshot, with the engine's
     own definition.
  5. The glossary says insider buying earns governance points — it now does.
  6. On live data the vendor column is numeric and two-signed: the data model the fix stands on.

Run with: pytest tests/test_insider_buying.py -v
"""

import ast
import contextlib
import io as _io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
import pandas as pd
import pytest

from config import GOVERNANCE_BONUS
from core.data_engine import compute_derived_signals, insider_net_buyer
from core.scoring_engine import compute_governance_bonus
from tests.test_data_quality_fixes import _frame


def _derived(**cols):
    f = _frame(len(next(iter(cols.values()))), **cols)
    with contextlib.redirect_stdout(_io.StringIO()):
        return compute_derived_signals(f)


# ── 1. One definition ───────────────────────────────────────────────────────────────────────────
def test_net_buying_is_a_positive_value_and_nothing_else_is():
    got = insider_net_buyer(pd.Series([399.84, -860.01, 0.0, np.nan, "12.5", "Bought"], dtype=object))
    assert got.tolist() == [1, 0, 0, 0, 1, 0], (
        "+ is net buying; a seller, a zero, a blank and text are not (text = the vendor changed format)")


# ── 2. Both consumers read it ───────────────────────────────────────────────────────────────────
def test_the_engine_derives_the_flag():
    out = _derived(insider_trading=[399.84, -860.01, np.nan, 0.0], market_cap=[1000.0] * 4)
    assert out["insider_net_buyer"].tolist() == [1, 0, 0, 0]


def test_d44_adds_two_for_a_net_buyer_and_nothing_for_a_seller():
    out = _derived(insider_trading=[50.0, -50.0, np.nan], market_cap=[1000.0] * 3)
    d44 = out["d44_smart_money_comp"]
    assert d44.iloc[0] - d44.iloc[1] == 2 and d44.iloc[1] == d44.iloc[2]


def test_the_governance_bonus_pays_a_net_buyer_exactly_the_configured_amount():
    out = _derived(insider_trading=[50.0, -50.0, np.nan], market_cap=[1000.0] * 3)
    g = compute_governance_bonus(out)["governance_bonus"]
    assert g.iloc[0] - g.iloc[1] == GOVERNANCE_BONUS["insider_trading_present"], (
        "a net buyer must earn exactly the configured insider bonus")
    assert g.iloc[1] == g.iloc[2], "a seller earns what a stock with no reported trade earns: nothing"


# ── 3. The text model is retired ────────────────────────────────────────────────────────────────
def test_no_engine_code_searches_the_column_for_text():
    for mod in ("data_engine.py", "scoring_engine.py"):
        tree = ast.parse((ROOT / "core" / mod).read_text(encoding="utf-8"))
        texts = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        assert "Bought" not in texts, f"{mod} still searches for the text 'Bought' — the column is a number"


# ── 4. December's instrument ────────────────────────────────────────────────────────────────────
def test_validate_measures_net_buyers_against_net_sellers():
    if not (ROOT / "tools" / "validate.py").exists():
        pytest.skip("tools/ is gitignored; the harness is local-only")
    with contextlib.redirect_stdout(_io.StringIO()):
        import validate
    df = pd.DataFrame({"insider_trading": [10.0, -10.0, np.nan, 3.0]})
    sets = dict(validate._registered_cohorts(df))
    title = next((t for t in sets if t.lower().startswith("insider")), None)
    assert title, "no insider cohort registered in tools/validate.py"
    masks = dict(sets[title])
    buyers = next(m for k, m in masks.items() if "buy" in k.lower())
    sellers = next(m for k, m in masks.items() if "sell" in k.lower())
    assert buyers.tolist() == [True, False, False, True] and sellers.tolist() == [False, True, False, False]


# ── 5. The words match the engine ───────────────────────────────────────────────────────────────
def test_the_glossary_says_insider_buying_earns_governance_points():
    from ui.ui_components import _RAW_GLOSSARY
    for key in ("Governance Score", "Gov Bonus"):
        assert "insider" in _RAW_GLOSSARY[key].lower(), f"{key!r} does not mention insider buying"


# ── 6. The data model the fix stands on ─────────────────────────────────────────────────────────
_DATA = ROOT / "Other Resources" / "CSV Data"


@pytest.mark.skipif(not _DATA.is_dir(), reason="local data not present (code-only checkout)")
def test_the_vendor_column_is_a_signed_number_and_the_flag_is_alive():
    from data_engine import coerce_numeric_columns, load_all_csvs, merge_datasets
    with contextlib.redirect_stdout(_io.StringIO()):
        live = compute_derived_signals(coerce_numeric_columns(merge_datasets(load_all_csvs("local"))))
    raw = live["insider_trading"]
    num = pd.to_numeric(raw, errors="coerce")
    assert num[raw.notna()].notna().all(), (
        "non-numeric insider values: the vendor changed format, and the insider bonus would go dead again")
    assert (num > 0).any() and (num < 0).any(), "the column is no longer two-signed — re-check its meaning"
    rate = live["insider_net_buyer"].mean()
    assert 0.003 <= rate <= 0.30, f"insider_net_buyer fires on {rate:.1%} — dead or saturated"
