"""
test_market_pulse_lens_order.py
===============================
Two ORDER decisions on the Market Pulse lens tabs (2026-10-06), both display-only.

1. 🏛️ QGLP LISTS ITS PASSERS IN PRISM'S RANK ORDER, NOT BY THE QGLP SCORE. QGLP decides WHO is
   listed (the pass gate); PRISM's score decides the ORDER. Measured among QGLP passers on the four
   saved snapshots (corporate actions excluded), forward rank-IC of the QGLP score was flat or
   negative in every window (-0.02 / -0.14 / +0.01 / -0.11 / -0.13) while PRISM's score was
   positive in every one (+0.08 / +0.06 / +0.05 / +0.12 / +0.03); the top 20 by QGLP score trailed
   the other passers by 5 to 14 points in three windows. Sorted by QGLP score, the tab opened on
   stocks ranked ~1,000-2,300 and 18 of its top 20 were FLAWED. The QGLP score itself and its
   rank correlation with composite (-0.04 among passers) are untouched — the column stays and
   sorts on a header click.

2. 🔭 MOSL LISTS A STOCK'S LENSES RAREST FIRST. The "Lenses cleared" cell is cut off at the
   table's edge on ~30% of rows, and in the list's own order the common lenses (Economic Moat,
   Consistent in Volatile, QGLP) came first while the rare ones that say the most (Blue Chip
   Quality, MOSL Wealth Creator) fell off the edge. The count column already says how many.

Both tests EXECUTE the app's own statements (read from app.py's AST) on a small frame, so they
test the shipped code rather than a copy of it.
"""

import ast
import io
import os
import re

import pandas as pd
import pytest

_APP = os.path.join(os.path.dirname(__file__), "..", "app.py")
_SRC = io.open(_APP, encoding="utf-8").read()
_TREE = ast.parse(_SRC, filename="app.py")


def _assign(target):
    """The single assignment statement whose target is `target` (a name, or a subscript source)."""
    hits = [n for n in ast.walk(_TREE) if isinstance(n, ast.Assign)
            and ast.get_source_segment(_SRC, n.targets[0]) == target]
    assert len(hits) == 1, f"expected one assignment to {target!r}, found {len(hits)}"
    return hits[0]


def _run(target, ns):
    exec(compile(ast.Module([_assign(target)], []), "app.py", "exec"), ns)
    return ns


# -- 1. QGLP: the pass gate decides who, PRISM's rank decides the order ---------------------
def test_qglp_passers_are_listed_in_prism_rank_order():
    frame = pd.DataFrame({
        "name":       ["High QGLP", "Mid", "Top PRISM", "Not a passer"],
        "qglp_pass":  [1, 1, 1, 0],
        "qglp_score": [99, 80, 60, 100],
        "rank":       [900, 40, 3, 1],
    })
    got = _run("_mp_qglp", {"_mp_df": frame})["_mp_qglp"]
    assert list(got["name"]) == ["Top PRISM", "Mid", "High QGLP"], (
        "QGLP passers must read in PRISM's rank order — the QGLP score did not predict returns "
        "among passers in any test window, PRISM's score did in every one"
    )


def test_the_rank_header_says_what_orders_the_table():
    """A reader who sees Rank 4, 6, 8 and a QGLP bar out of order must be told why."""
    i = _SRC.index("_q_sel = st.dataframe(")
    block = _SRC[i:_SRC.index("selection_mode=", i)]
    m = re.search(r'"rank":\s*st\.column_config\.NumberColumn\((.*?)\),\n', block, re.S)
    assert m, "the QGLP table's Rank column config is gone"
    help_txt = " ".join(re.findall(r'"([^"]*)"', m.group(1)))
    assert "order" in help_txt.lower() and "QGLP" in help_txt, (
        f"the Rank header must say it is the table's order and why QGLP's score is not: {help_txt!r}")


# -- 2. MOSL: rarest lens first --------------------------------------------------------------
def _mosl(frameworks):
    frame = pd.DataFrame({"frameworks_passed": frameworks})
    ns = {"pd": pd, "re": re, "_mp_df": frame}
    for t in ("_MOSL_LENSES", "_tok", "_mosl_order"):
        _run(t, ns)
    ns["_mosl"] = frame.copy()
    _run('_mosl["mosl_n"]', ns)
    _run('_mosl["mosl_hits"]', ns)
    return ns


def test_mosl_lenses_read_rarest_first():
    # Economic Moat on 3 stocks, QGLP on 2, Blue Chip Quality on 1: rarest first on every row.
    ns = _mosl([
        "QGLP, Economic Moat, Blue Chip Quality",
        "QGLP, Economic Moat",
        "Economic Moat, Some Other Framework",
    ])
    hits = list(ns["_mosl"]["mosl_hits"])
    assert hits[0] == "Blue Chip Quality · QGLP · Economic Moat", hits[0]
    assert hits[1] == "QGLP · Economic Moat", hits[1]
    assert hits[2] == "Economic Moat", hits[2]
    assert list(ns["_mosl"]["mosl_n"]) == [3, 2, 1], "the order must not change the count"


def test_mosl_ties_keep_the_list_order_and_tokens_stay_exact():
    # QGLP and Economic Moat both fire once: a tie keeps _MOSL_LENSES' own order (QGLP first).
    # "SQGLP Century Stock" must not count as QGLP — the substring trap.
    ns = _mosl(["Economic Moat, QGLP", "SQGLP Century Stock"])
    assert list(ns["_mosl"]["mosl_hits"]) == ["QGLP · Economic Moat", "SQGLP Century Stock"]
    order = ns["_mosl_order"]
    assert order.index("QGLP") < order.index("Economic Moat") < order.index("SQGLP Century Stock")
    assert sorted(ns["_mosl_order"]) == sorted(ns["_MOSL_LENSES"]), "the order must hold every lens once"
