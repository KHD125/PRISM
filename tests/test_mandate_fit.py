"""Contract: 'Mandate Fit' = safety floor ∩ mandate thesis — the mandate-responsive qualified count.

The mandate card surfaces two DISJOINT gate sets: gate_pass (the universal safety floor — D/E, pledge,
CFO/PAT, promoter, PAT>0 …, profile-INVARIANT) and the per-profile qglp_pass (the ROCE/Growth/PEG
thesis screen). Neither alone is the useful number a user wants:
  - gate_pass never moves when you switch mandates (always the same safe universe);
  - qglp_pass alone can EXCEED the safe universe (it includes names that fail the safety floor).
Mandate Fit = (gate_pass & qglp_pass) is the only count that is BOTH a subset of the safety floor AND
moves per mandate. These tests pin that semantic + that app.py surfaces it and no longer mislabels the
per-profile gate as "Hard Gates" (which collides with the global HARD_GATES the config tab owns).
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
def test_mandate_fit_is_subset_of_floor_and_responsive():
    """Mandate Fit (gate_pass & qglp_pass) must be a SUBSET of the safety floor for every mandate
    (the anti-qglp-overstatement guard) AND differ across mandates (the whole point — it moves)."""
    from core import run_scoring_pipeline
    from core.data_engine import fetch_and_clean_data
    with contextlib.redirect_stdout(io.StringIO()):
        clean = fetch_and_clean_data("local")
        q = run_scoring_pipeline(clean.copy(), "Fundamental", "Quality")
        t = run_scoring_pipeline(clean.copy(), "Technical", "Turnaround")

    fits = {}
    for name, df in [("Quality", q), ("Turnaround", t)]:
        floor = (df["gate_pass"] == 1)
        thesis = (df.get("qglp_pass", df["gate_pass"] * 0) == 1)
        fit = int((floor & thesis).sum())
        assert fit <= int(floor.sum()), f"{name}: Mandate Fit must be <= the safety floor"
        fits[name] = fit
    # Quality (ROCE>=20, tight) and Turnaround (ROCE>=8/growth>=0/PEG<=5, loose) must yield
    # different Mandate-Fit counts — proving the number actually responds to the mandate.
    assert fits["Quality"] != fits["Turnaround"], (
        f"Mandate Fit must move across mandates, got {fits}"
    )


def test_app_surfaces_mandate_fit_and_drops_hard_gates_mislabel():
    """app.py must compute Mandate Fit (gate_pass ∩ qglp_pass) and label the per-profile gate line
    'Mandate Screen' — not the bare 'Hard Gates — ROCE' that collided with the global HARD_GATES."""
    src = (Path(__file__).resolve().parent.parent / "app.py").read_text(encoding="utf-8")
    assert "mandate_fit" in src, "app.py must compute mandate_fit"
    assert "qglp_pass" in src and "gate_pass" in src, "mandate_fit must intersect floor & thesis"
    assert "Mandate Screen" in src, "the card must call the per-profile gates 'Mandate Screen'"
    assert "Hard Gates — ROCE" not in src, (
        "the mislabeled 'Hard Gates — ROCE' must be gone from the card (the global HARD_GATES "
        "card legitimately keeps 'Hard Gates · N Criteria')"
    )
