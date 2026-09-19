"""
test_pyq_margin_basis.py
========================
Tripwire for five columns mapped on 2026-09-18 but NOT YET CARRIED BY THE SOURCE SHEET.

WHY A TRIPWIRE INSTEAD OF AN IMPLEMENTATION. Two of the five repair a SCORED defect, and the
source does not carry them yet (verified against the 2026-08-28 CSV drops and the 2026-09-09
workbook: absent from every tab). Rebasing a scored input without a live /census is precisely the
calibration-by-guesswork §5's signal-liveness discipline exists to stop, so the wiring is
deliberately deferred — and a deferral nobody is reminded of is just a defect with a comment.

THE DEFECT BEING HELD OPEN, in the codebase's own words (ui/ui_tearsheet.py ~L4396):

    npm/opm_acceleration        = latest QUARTER - the ANNUAL figure 1Y back
    gpm_acceleration            = latest QUARTER - the 5Y MEDIAN   (a third base again)

`npm_acceleration` carries weight 0.15 and `opm_acceleration` 0.10 inside the margin facet
(core/scoring_engine.py ~L400), so this reaches composite_score for the whole universe. A quarter
minus an annual number is not a delta (§5 cross-year basis rule), and it folds SEASONALITY into a
number read as structural: a festive-quarter margin measured against an annual base reads as
"accelerating" when nothing changed at all.

HOW THIS FILE BEHAVES:
  * structural pins run ALWAYS — the five vendor headers stay mapped, in the right tab, and the
    provenance comment that explains the deferral cannot be quietly deleted.
  * behavioural pins SKIP while the columns are absent and go RED the moment real data arrives,
    which is the signal to do the rebasing + census. They are supposed to fail then. That failure
    is the reminder, not a regression.

Run with: pytest tests/test_pyq_margin_basis.py -v
"""
from __future__ import annotations

import contextlib
import io as _io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import pytest

from core.data_engine import RATIO_COLS, TECHNICAL_COLS

_MIN_COVERAGE = 0.10        # below this the column is present but empty — not yet real data


@pytest.fixture(scope="module")
def live():
    from core import fetch_and_clean_data, run_scoring_pipeline
    with contextlib.redirect_stdout(_io.StringIO()):
        return run_scoring_pipeline(fetch_and_clean_data("local").copy())


def _present(df, col):
    """A column counts as ARRIVED only when it exists AND carries real values."""
    if col not in df.columns:
        return False
    return pd.to_numeric(df[col], errors="coerce").notna().mean() >= _MIN_COVERAGE


# ── 1. Structural: the sockets exist and sit in the right tab ────────────────────────────

@pytest.mark.parametrize("header,internal", [
    ("OPM Preceding Year Quarter", "opm_pyq"),
    ("NPM Preceding Year Quarter", "npm_pyq"),
    ("GPM Preceding Year Quarter", "gpm_pyq"),
])
def test_the_margin_pyq_headers_are_mapped_in_the_ratio_tab(header, internal):
    """Exact vendor strings — a typo here means the column silently never arrives."""
    assert RATIO_COLS.get(header) == internal, (
        f"{header!r} must map to {internal!r} in RATIO_COLS; the vendor groups the margins under "
        f"Ratios, and a mapping in the wrong tab dict is never consulted for that sheet."
    )


@pytest.mark.parametrize("header,internal", [
    ("Listed Days", "listed_days"),
    ("Returns Since Result", "returns_since_result"),
])
def test_the_technical_headers_are_mapped_in_the_technical_tab(header, internal):
    assert TECHNICAL_COLS.get(header) == internal, (
        f"{header!r} must map to {internal!r} in TECHNICAL_COLS (vendor category: Price & Technical)."
    )


def test_the_deferral_is_explained_where_the_mapping_lives():
    """A mapped-but-unwired column with no reason attached reads as an oversight to the next
    reader, who then either wires it blind or deletes it. The reason travels with the mapping."""
    src = _io.open(os.path.join(os.path.dirname(__file__), "..", "core", "data_engine.py"),
                   encoding="utf-8").read()
    assert "NOT YET WIRED" in src, "the deferral marker was removed from data_engine.py"
    assert "signal-liveness" in src or "/census" in src, (
        "the reason the wiring is deferred (a scored rebasing needs a live census) is gone"
    )


# ── 1b. The rebasing itself, once it shipped ─────────────────────────────────────────────

def _nan_frame(n=12, **overrides):
    """compute_derived_signals needs every mapped column present; build that from the mapping
    dicts themselves so the fixture cannot rot as columns are added."""
    import numpy as np
    from core import data_engine as de
    cols = set()
    for d in (de.COMMON_COLS, de.RATIO_COLS, de.INCOME_COLS, de.BALANCE_COLS,
              de.CASHFLOW_COLS, de.SHAREHOLDING_COLS, de.TECHNICAL_COLS):
        cols |= set(d.values())
    f = pd.DataFrame({c: [np.nan] * n for c in sorted(cols)})
    f["company_id"] = [f"NSE:T{i}" for i in range(n)]
    f["name"] = [f"Test Co {i}" for i in range(n)]
    for k, v in overrides.items():
        f[k] = v
    return f


@pytest.mark.parametrize("margin", ["opm", "npm", "gpm"])
def test_a_vintage_without_pyq_yields_nan_never_the_old_basis(margin):
    """THE REGRESSION THIS FIX COULD CAUSE, pinned. An archived vintage predating *_pyq must NOT
    silently fall back to the old formula: two bases in one column makes the number's meaning
    depend on coverage, which is the exact defect being repaired. NaN is the honest answer —
    _compute_margin_score ranks with .fillna(50), so it reads as "no information", not as bad news.
    """
    import numpy as np
    from core.data_engine import compute_derived_signals
    with contextlib.redirect_stdout(_io.StringIO()):
        out = compute_derived_signals(_nan_frame(**{
            f"{margin}_latest_q": 10.0,
            f"{margin}_pyq": np.nan,        # the archived-vintage case
            f"{margin}_1yb": 3.0,           # the OLD base — must NOT be used
            f"{margin}_med_5y": 2.0,        # the OLD gpm base — must NOT be used either
        }))
    got = pd.to_numeric(out[f"{margin}_acceleration"], errors="coerce")
    assert got.isna().all(), (
        f"{margin}_acceleration fell back to a second basis when {margin}_pyq was absent; it read "
        f"{got.dropna().unique()[:3]} where NaN is required (7.0 would be the retired 1yb form, "
        f"8.0 the retired med_5y form, 0.0 a fabricated 'flat')."
    )


@pytest.mark.parametrize("margin", ["opm", "npm", "gpm"])
def test_the_rebased_formula_is_latest_quarter_minus_the_same_quarter_last_year(margin):
    import numpy as np
    from core.data_engine import compute_derived_signals
    with contextlib.redirect_stdout(_io.StringIO()):
        out = compute_derived_signals(_nan_frame(**{
            f"{margin}_latest_q": 10.0, f"{margin}_pyq": 6.0,
            f"{margin}_1yb": 3.0, f"{margin}_med_5y": 2.0,
        }))
    got = pd.to_numeric(out[f"{margin}_acceleration"], errors="coerce")
    assert np.allclose(got.dropna(), 4.0), (
        f"expected 10.0 − 6.0 = 4.0 percentage points; got {got.dropna().unique()[:3]}"
    )


def test_the_tearsheet_base_label_cannot_drift_from_the_engine(live):
    """A stale base label beside a corrected number reads as authoritative and is worse than the
    original defect. The display table names its base column; that column must be the one the
    engine actually subtracted (the Fisher module/engine drift precedent, commit 7fff308)."""
    import numpy as np
    from ui.ui_tearsheet import _ACCEL_MARGIN
    assert _ACCEL_MARGIN, "the margin trajectory table vanished"
    for row in _ACCEL_MARGIN:
        label, accel_col, latest_col, base_col = row[0], row[1], row[2], row[3]
        for c in (accel_col, latest_col, base_col):
            assert c in live.columns, f"{label}: {c} is not in the scored frame"
        expected = pd.to_numeric(live[latest_col], errors="coerce") - \
            pd.to_numeric(live[base_col], errors="coerce")
        got = pd.to_numeric(live[accel_col], errors="coerce")
        both = expected.notna() & got.notna()
        assert both.sum() > 100, f"{label}: too few comparable rows"
        assert np.allclose(got[both], expected[both], atol=1e-6), (
            f"{label}: the tearsheet says the base is {base_col!r}, but {accel_col} was not "
            f"computed against it. The on-screen label is lying about the number beside it."
        )


# ── 2. Behavioural: dormant until the sheet carries the data, then RED ───────────────────

@pytest.mark.parametrize("margin", ["opm", "npm", "gpm"])
def test_once_pyq_arrives_the_acceleration_must_be_rebased_onto_it(live, margin):
    """THE POINT OF THIS FILE. Skips today; fails the day the data lands and nothing was rebased.

    A failure here is the to-do list, not a regression: rebase <margin>_acceleration onto
    <margin>_pyq, re-run /census, and measure the composite shift before committing.
    """
    pyq, latest, accel = f"{margin}_pyq", f"{margin}_latest_q", f"{margin}_acceleration"
    if not _present(live, pyq):
        pytest.skip(f"{pyq} not in the source sheet yet — tripwire dormant")
    assert accel in live.columns, f"{accel} vanished while {pyq} arrived"
    expected = pd.to_numeric(live[latest], errors="coerce") - pd.to_numeric(live[pyq], errors="coerce")
    got = pd.to_numeric(live[accel], errors="coerce")
    both = expected.notna() & got.notna()
    assert both.sum() > 100, f"too few comparable rows to judge {accel}"
    import numpy as np
    assert np.allclose(got[both], expected[both], atol=1e-6), (
        f"{pyq} IS NOW AVAILABLE but {accel} is still on its old base. Rebase it to "
        f"{latest} − {pyq} (one basis, seasonality cancelled), then run /census and measure the "
        f"composite_score shift — {accel} is SCORED, so this moves the whole universe."
    )


def test_once_listed_days_arrives_the_dilution_arm_must_consult_it(live):
    """docs/known-issues.md calls the missing listing date the reason the IPO tier cannot be built.
    Once it exists, that justification is stale and the arm has to be revisited."""
    if not _present(live, "listed_days"):
        pytest.skip("listed_days not in the source sheet yet — tripwire dormant")
    src = _io.open(os.path.join(os.path.dirname(__file__), "..", "core", "data_engine.py"),
                   encoding="utf-8").read()
    idx = src.find('_corp_action')
    block = src[max(0, idx - 4000): idx + 2000]
    assert "listed_days" in block, (
        "listed_days IS NOW AVAILABLE, so the dilution corporate-action arm's stated blocker "
        "('there is no listing-date column') is no longer true. Either consult it there, or "
        "rewrite the justification in docs/known-issues.md to say why it still is not used."
    )


@pytest.mark.xfail(strict=True, reason=(
    "PHASE 3, deliberately not this commit. returns_since_result ARRIVED (99.0% coverage) but a "
    "new signal ships only after a distribution census and an orthogonality check against the "
    "momentum family — display + sort first, scored only if a forward window earns it. "
    "strict=True: this XPASSes and FAILS the day it reaches a surface, so the marker cannot "
    "outlive the deferral it documents."))
def test_once_returns_since_result_arrives_it_must_reach_a_surface(live):
    """A signal nobody can see is orphan #416. PRISM already carries 400+ of those."""
    if not _present(live, "returns_since_result"):
        pytest.skip("returns_since_result not in the source sheet yet — tripwire dormant")
    surfaces = ""
    for rel in (("app.py",), ("ui", "ui_scanner.py"), ("ui", "ui_tearsheet.py")):
        p = os.path.join(os.path.dirname(__file__), "..", *rel)
        if os.path.exists(p):
            surfaces += _io.open(p, encoding="utf-8").read()
    assert "returns_since_result" in surfaces, (
        "returns_since_result IS NOW AVAILABLE but reaches no UI surface. Run a fire-rate census "
        "first, then give it a home (Deep Scanner preset column + header tooltip) — display and "
        "sort only until a forward window says it deserves to be scored."
    )
