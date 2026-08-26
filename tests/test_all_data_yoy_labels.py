"""
test_all_data_yoy_labels.py
===========================
Contract: no cell on the All Data tab may label a QUARTERLY figure as if it were annual.

THE BUG (fixed 2026-08-26, found by reading a live screenshot). The Growth block rendered four
YoY cells — "PAT YoY", "Rev YoY", "EPS YoY" and "Q PAT YoY". The "Q" prefix on the last told the
reader the others were annual. They were not: `pat_gr_yoy` / `rev_gr_yoy` / `eps_gr_yoy` are
quarterly under annual-sounding names, and each is IDENTICAL to its q_* twin on ~97% of rows:

    pat_gr_yoy vs q_pat_yoy : 2,029 of 2,085 (97.3%)
    eps_gr_yoy vs q_eps_yoy : 2,016 of 2,075 (97.2%)
    rev_gr_yoy vs q_rev_yoy : 2,016 of 2,072 (97.3%)

So "PAT YoY" and "Q PAT YoY" printed the same number twice, and the unprefixed one sat directly
beneath "PAT 5Y CAGR" and "PAT 3Y CAGR" — inviting exactly the wrong comparison. On the screenshot
that surfaced this: PAT 5Y −11.8%, PAT 3Y +39.0%, PAT YoY +108.5%, Q PAT YoY +108.5%. A reader
concludes the latest YEAR outgrew the 3-year trend. One QUARTER did.

THE FIX, and why it is not "delete all three": only `q_pat_yoy` is rendered on this tab —
`q_rev_yoy` and `q_eps_yoy` are not. Deleting "Rev YoY" and "EPS YoY" would have removed those
numbers from the app entirely. So the true duplicate was deleted and the other two were RELABELLED
to carry the "Q" prefix their basis has always deserved.

Run with: pytest tests/test_all_data_yoy_labels.py -v
"""

import contextlib
import io as _io
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import pytest

import ui.ui_tearsheet as T
from data_engine import (coerce_numeric_columns, compute_derived_signals, load_all_csvs,
                         merge_datasets)

# The columns that are quarterly despite an annual-sounding name.
QUARTERLY_UNDER_ANNUAL_NAME = ["pat_gr_yoy", "rev_gr_yoy", "eps_gr_yoy"]


@pytest.fixture(scope="module")
def live():
    with contextlib.redirect_stdout(_io.StringIO()):
        return compute_derived_signals(
            coerce_numeric_columns(merge_datasets(load_all_csvs("local"))))


def _render(row):
    out = []

    class _Rec:
        def markdown(self, *a, **k):
            if a:
                out.append(str(a[0]))

        def __getattr__(self, _n):
            return lambda *a, **k: None

    real = T.st
    try:
        T.st = _Rec()
        T.render_raw_signals(row)
    finally:
        T.st = real
    return " ".join(out)


def _yoy_labels(html):
    """Every cell label containing 'YoY', matched exactly — a substring test is useless here
    because "PAT YoY" occurs inside "Q PAT YoY"."""
    return sorted({m.strip() for m in re.findall(r">([A-Z][A-Za-z0-9 /%×Δ→&.\-]{1,22})<", html)
                   if "YoY" in m})


# ── 1. The premise: these columns really are quarterly ──────────────────────────────────
@pytest.mark.parametrize("annual_name,q_twin", [
    ("pat_gr_yoy", "q_pat_yoy"), ("rev_gr_yoy", "q_rev_yoy"), ("eps_gr_yoy", "q_eps_yoy"),
])
def test_the_annual_sounding_column_is_actually_its_quarterly_twin(live, annual_name, q_twin):
    both = live[annual_name].notna() & live[q_twin].notna()
    assert both.sum() > 1500, "not enough overlap to judge"
    same = ((live[annual_name][both] - live[q_twin][both]).abs() < 0.05).mean()
    assert same > 0.95, (
        f"{annual_name} agrees with {q_twin} on only {same:.1%} of rows — the premise of this "
        f"contract (that it is quarterly) no longer holds; re-audit before trusting the labels"
    )


# ── 2. No unprefixed YoY label may survive ──────────────────────────────────────────────
def test_no_yoy_cell_is_labelled_as_if_annual(live):
    labels = _yoy_labels(_render(live.iloc[0]))
    assert labels, "no YoY cells rendered at all — the Growth block changed shape"
    bad = [l for l in labels if not l.startswith("Q ")]
    assert not bad, (
        f"these YoY cells render a QUARTERLY column under an annual-sounding label: {bad}. "
        f"Beside 'PAT 5Y CAGR' and 'PAT 3Y CAGR' they invite an annual-to-annual comparison "
        f"that is not what the number means."
    )


def test_the_duplicate_pat_cell_is_gone(live):
    """"PAT YoY" and "Q PAT YoY" printed the identical number."""
    labels = _yoy_labels(_render(live.iloc[0]))
    assert "PAT YoY" not in labels
    assert "Q PAT YoY" in labels, "the honestly-named cell must survive — it carries the number"


def test_no_number_was_lost_in_the_dedup(live):
    """q_rev_yoy / q_eps_yoy are NOT rendered on this tab, so the rev and eps cells had to be
    relabelled rather than deleted. All three quarterly growth figures must still be present."""
    labels = _yoy_labels(_render(live.iloc[0]))
    for expected in ("Q PAT YoY", "Q Rev YoY", "Q EPS YoY"):
        assert expected in labels, f"{expected} vanished — a growth figure was dropped, not relabelled"


# ── 3. The quintile is a label, not a float ─────────────────────────────────────────────
def test_ep_quintile_renders_as_Q1_not_1_point_0(live):
    """ep_quintile is float64, so it printed "1.0" here while every other surface says "Q1"."""
    row = live[live["ep_quintile"].notna()].iloc[0]
    html = _render(row)
    seg = html[html.index("EP Quintile"):][:300]
    q = int(float(row["ep_quintile"]))
    assert f"Q{q}" in seg, f"expected Q{q} in the cell, got: {seg[:120]}"
    assert f">{float(row['ep_quintile'])}<" not in seg, "still rendering the raw float"


def test_ep_quintile_missing_renders_na(live):
    row = live.iloc[0].copy()
    row["ep_quintile"] = float("nan")
    html = _render(row)
    seg = html[html.index("EP Quintile"):][:300]
    assert "N/A" in seg, "a missing quintile must read N/A, never Q0 or a crash"
