"""Contract: Deep-Cyclical/Commodity fair_pe_qglp is capped.

The EVA fair PE (growth × ROCE/CoC) is right for durable compounders, but for commodity/deep-cyclical
businesses pat_gr_5y and roce_med_10y are CYCLICAL-PEAK and mean-revert — so they were minting an
absurd compounder multiple (Coal India 9.4 trailing PE → 86 "fair" PE → +815% phantom upside, the
classic value-trap amplified into the Valuation axis). data_engine.compute_derived_signals now clips
their fair PE at the high end of what cyclicals sustain through-cycle, and ONLY that tier — genuine
compounders keep the full multiple.

Data-gated like test_output_consistency / test_ui_smoke: skips cleanly on a code-only checkout.
"""
import contextlib
import io
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_DATA = os.path.join(os.path.dirname(__file__), "..", "Other Resources", "CSV Data")
pytestmark = pytest.mark.skipif(not os.path.isdir(_DATA), reason="needs local CSV data")

_CAP = 18.0   # mirrors _CYCLICAL_FAIR_PE_CAP in data_engine.compute_derived_signals (update both)
_TIER = "Deep Cyclical / Commodity"


@pytest.fixture(scope="module")
def scored():
    with contextlib.redirect_stdout(io.StringIO()):
        from core import fetch_and_clean_data, run_scoring_pipeline
        return run_scoring_pipeline(fetch_and_clean_data("local"))


def test_deep_cyclical_fair_pe_is_capped(scored):
    dc = scored.loc[scored["cyclicality_tier"] == _TIER, "fair_pe_qglp"].dropna()
    assert len(dc) > 50, "expected a populated deep-cyclical cohort"
    assert (dc <= _CAP + 1e-6).all(), (
        f"deep-cyclical fair_pe_qglp exceeds the {_CAP} cap (max {dc.max()}) — the cyclical clip "
        f"regressed; commodity peak-earnings would re-inflate into phantom value-trap upside"
    )


def test_non_cyclical_fair_pe_untouched(scored):
    """Genuine compounders MUST keep a fair PE above the cyclical cap — proving the clip is scoped to
    the deep-commodity tier and didn't quietly cap the whole universe."""
    nc = scored.loc[scored["cyclicality_tier"] != _TIER, "fair_pe_qglp"].dropna()
    assert (nc > _CAP).sum() >= 50, (
        "no non-cyclical stock retains a fair PE above the cap — the clip is mis-scoped (it must "
        "touch ONLY Deep Cyclical / Commodity, not durable compounders)"
    )
