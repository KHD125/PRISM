"""Contract: 'Mandate Fit' = safety floor ∩ mandate thesis — the mandate-responsive qualified count.

The mandate card surfaces two DISJOINT gate sets: gate_pass (the universal safety floor — D/E, pledge,
CFO/PAT, promoter, PAT>0 …, profile-INVARIANT) and the per-profile qglp_pass (the ROCE/Growth/PEG
thesis screen). Neither alone is the useful number a user wants:
  - gate_pass never moves when you switch mandates (always the same safe universe);
  - qglp_pass alone can EXCEED the safe universe (it includes names that fail the safety floor).
Mandate Fit = (gate_pass & qglp_pass) is the only count that is a STRICT subset of the safety floor.
These tests pin that semantic + that app.py surfaces it and no longer mislabels the QGLP screen as
"Hard Gates" (which collides with the global HARD_GATES the config tab owns).

UPDATED 2026-09-20 (§6). The second half of the original claim — "and moves per mandate" — was
removed with the eight-profile selector, because moving per mandate is exactly what made QGLP the
only one of 38 frameworks whose verdict was a user setting. The assertion is now its INVERSE: the
fit count must be invariant to the analysis mode. That is the behavioural proof that the round trip
Breakout -> Technical Only -> Breakout can no longer move the QGLP screen (it used to: 413 -> 631).
"""
import io
import os
import contextlib
from pathlib import Path

import pytest

# Real CSV data is gitignored (code-only repo); guard the slow real-data test PER-TEST so the static
# source-check test below still runs in a code-only CI clone. Mirrors test_ui_smoke.py.
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Other Resources", "CSV Data")


@pytest.mark.slow
@pytest.mark.skipif(not os.path.isdir(_DATA_DIR),
                    reason="local CSV data absent (code-only checkout) — needs real data")
def test_mandate_fit_is_a_strict_subset_and_is_invariant_to_the_analysis_mode():
    """Two properties, on real data.

    (1) STRICT SUBSET — the anti-overstatement guard this file was written for: the fit must sit
        inside the safety floor, and strictly inside it, or intersecting adds nothing.
    (2) MODE-INVARIANT — the guarantee the 2026-09-20 removal bought. The QGLP screen is now a
        constant, so re-scoring under a different analysis mode must leave the fit count and
        qglp_pass itself untouched. Before the removal each mode carried a permitted-profile list
        and "Technical Only" silently switched the screen to Momentum (413 -> 631 passers), which
        this assertion would have caught and no test then did.
    """
    from core import run_scoring_pipeline
    from core.data_engine import fetch_and_clean_data
    with contextlib.redirect_stdout(io.StringIO()):
        clean = fetch_and_clean_data("local")
        runs = {m: run_scoring_pipeline(clean.copy(), m) for m in ("Fundamental", "Technical")}

    fits, passes = {}, {}
    for name, df in runs.items():
        floor = (df["gate_pass"] == 1)
        thesis = (df.get("qglp_pass", df["gate_pass"] * 0) == 1)
        fit, n_floor = int((floor & thesis).sum()), int(floor.sum())
        assert fit <= n_floor, f"{name}: Mandate Fit must be <= the safety floor"
        assert fit < n_floor, (
            f"{name}: Mandate Fit ({fit}) equals the whole floor ({n_floor}) — the QGLP screen "
            f"is excluding nobody, so intersecting it with the floor is decorative"
        )
        fits[name], passes[name] = fit, int(thesis.sum())
    assert len(set(fits.values())) == 1, (
        f"the QGLP screen must not vary by analysis mode, got fits {fits} — a mode that moves it "
        f"is the silent-rewrite defect returning (see tests/test_qglp_profile_fixed.py)"
    )
    assert len(set(passes.values())) == 1, f"qglp_pass must not vary by analysis mode, got {passes}"


def test_app_surfaces_the_fit_count_beside_its_knobs():
    """After the Command Center removal (2026-08-24) the fit count lives in ⚙️ Config, under the
    Analysis Mode selectbox: _fit_cfg = gate_pass ∩ qglp_pass, labelled 'QGLP screen' (honest
    name — it IS the QGLP screen, not the engine's hard gates). The old front-page 'Mandate
    Screen' banner is gone with the mandates, and since 2026-09-20 the profile selectbox that
    used to sit beside this line is gone too — the gates it printed are now a stated fact."""
    src = (Path(__file__).resolve().parent.parent / "app.py").read_text(encoding="utf-8")
    assert "_fit_cfg" in src, "Config must compute the fit count (gate_pass ∩ qglp_pass)"
    assert 'df.get("qglp_pass"' in src and '(df["gate_pass"] == 1)' in src, (
        "the fit count must intersect the safety floor with the per-profile thesis screen"
    )
    assert "QGLP screen" in src, "the fit line must be honestly labelled 'QGLP screen'"
    assert "Mandate Screen" not in src, "the front-page mandate banner is removed — no stale label"
    assert "Hard Gates — ROCE" not in src, "the old mislabel must stay gone"
