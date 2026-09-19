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


def test_returns_since_result_stays_off_screen_it_is_a_50_day_return_renamed(live):
    """TOMBSTONE — Phase 3 was REJECTED by its own admission census on 2026-09-19. Pinned OUT so
    nobody re-proposes it (the dilution_vampire / cyclical_mirage / Value-Creation-Velocity
    pattern: a rejection is only durable if a test holds it).

    THE ARGUMENT FOR IT WAS WRONG, and measurement is what showed it. The case was "every momentum
    column here is fixed-calendar and straddles the earnings event arbitrarily, so post-earnings
    drift is invisible to 724 columns". That reasoning collapses because Indian reporting is
    SYNCHRONISED: 2,259 of 2,691 stocks (84%) sit at a result age of 30-60 days, so "since result"
    IS a fixed ~50-day window for almost the whole universe. Measured Spearman, 2026-09-18 vintage:

        crs_50d +0.784 · rsi_14d +0.761 · ret_vs_n500_3m +0.745 · rs_score +0.706
        momentum_score +0.689 · breakout_score +0.640 · dist_52wh -0.640

    Max |rho| 0.784 against a column already on screen, inside the 0.598-0.806 band this codebase
    has twice called redundant. It also carries a max of +12,181%, so it would need a degenerate
    guard before it could even be sorted on.

    WHAT WOULD REOPEN IT: reporting dates de-synchronising (check the 30-60d concentration), or a
    forward window showing it beats crs_50d on rank-IC despite the overlap. The column stays
    MAPPED — it is free, and the measurement should not have to be re-derived — it just stays off
    every surface.
    """
    if not _present(live, "returns_since_result"):
        pytest.skip("returns_since_result absent from this vintage")
    import numpy as np
    r = pd.to_numeric(live["returns_since_result"], errors="coerce")
    ok = r.notna() & pd.to_numeric(live["crs_50d"], errors="coerce").notna()
    rho = r[ok].rank().corr(pd.to_numeric(live["crs_50d"], errors="coerce")[ok].rank())
    assert rho > 0.60, (
        f"returns_since_result is no longer redundant with crs_50d (rho={rho:+.3f}, was +0.784). "
        f"The rejection rested on that overlap — re-run the admission census, because this may "
        f"now deserve a surface."
    )
    age = pd.to_numeric(live["result_age_days"], errors="coerce")
    conc = float(((age >= 30) & (age < 60)).mean())
    assert conc > 0.60, (
        f"only {100*conc:.0f}% of the universe now sits at a 30-60 day result age (was 84%). "
        f"Reporting has de-synchronised, so 'since result' may finally be event-time rather than "
        f"a fixed window — the rejection's premise is gone, re-measure it."
    )
    surfaces = ""
    for rel in (("app.py",), ("ui", "ui_scanner.py"), ("ui", "ui_tearsheet.py")):
        p = os.path.join(os.path.dirname(__file__), "..", *rel)
        if os.path.exists(p):
            surfaces += _io.open(p, encoding="utf-8").read()
    assert "returns_since_result" not in surfaces, (
        "returns_since_result reached a UI surface. It was rejected on 2026-09-19 as crs_50d "
        "renamed (rho +0.784); if that has changed, delete this tombstone with the new census "
        "attached rather than shipping past it."
    )
