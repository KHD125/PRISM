"""Contract: 🔥 Capacity Explosion is described as what it IS — fixed-asset growth, not CWIP.

cat_capacity (core/scoring_engine.py) = d19_cwip_conversion > 0 AND d20_fa_cagr_3y > 15. Despite its
name, d19_cwip_conversion is fixed_assets − fixed_assets_1yb (data_engine D19, corrected from the old
double-counting ΔCWIP + ΔFA form), and d20 is the 3-year fixed-asset CAGR. CWIP is read by NEITHER.
The Reference entry nevertheless said "fixed assets/CWIP converting", so a reader picking the catalyst
believed they were screening for capacity going live out of construction — the signal PRISM actually
computes as cat_cwip_inflection, which is NOT on screen because it has shown no forward edge
(measured 2026-10-02: −0.6 / −0.2 / −0.1pp across the three snapshot windows; cwip_pre_inflection
−1.4 / −1.2 / −2.0pp).

The pins are tied to the ENGINE, not to a sentence: if CWIP is ever wired into cat_capacity, the first
two tests fail and force this description to be revisited, instead of the text silently becoming
true or staying false.
"""
import ast
from pathlib import Path

import numpy as np

from core.data_engine import compute_derived_signals
from tests.test_data_quality_fixes import _frame
from ui.ui_reference_data import CONCEPT_REFERENCE

_ROOT = Path(__file__).resolve().parent.parent


def _cat_capacity_columns():
    """Every string constant inside the expression assigned to df["cat_capacity"]."""
    src = (_ROOT / "core" / "scoring_engine.py").read_text(encoding="utf-8")
    found = []
    for n in ast.walk(ast.parse(src)):
        if (isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Subscript)
                and isinstance(n.targets[0].slice, ast.Constant)
                and n.targets[0].slice.value == "cat_capacity"):
            found.append({c.value for c in ast.walk(n.value)
                          if isinstance(c, ast.Constant) and isinstance(c.value, str)})
    assert len(found) == 1, f"expected exactly one cat_capacity assignment, found {len(found)}"
    return found[0]


def test_cat_capacity_reads_only_the_two_fixed_asset_inputs():
    assert _cat_capacity_columns() == {"d19_cwip_conversion", "d20_fa_cagr_3y"}


def test_neither_input_moves_when_only_cwip_changes():
    """Behavioural: two frames identical except a WILDLY different CWIP must produce identical
    d19/d20 — so the inputs to the catalyst are blind to CWIP whatever their names say."""
    fa = dict(fixed_assets=1300.0, fixed_assets_1yb=1100.0, fixed_assets_3yb=700.0)
    a = compute_derived_signals(_frame(4, cwip=5.0, cwip_1yb=900.0, **fa))
    b = compute_derived_signals(_frame(4, cwip=800.0, cwip_1yb=2.0, **fa))
    for col in ("d19_cwip_conversion", "d20_fa_cagr_3y"):
        assert np.allclose(a[col], b[col], equal_nan=True), col
    # teeth: the inputs are live on this frame, so 'identical' is not 'both NaN'
    assert (a["d19_cwip_conversion"] > 0).all() and (a["d20_fa_cagr_3y"] > 15).all()


def _entry(label):
    hits = [e for cat in CONCEPT_REFERENCE.values() for l, e in cat if l == label]
    assert len(hits) == 1, f"{label}: {len(hits)} entries"
    return hits[0]


def test_the_reference_describes_fixed_asset_growth_not_cwip():
    text = _entry("🔥 Capacity Explosion")
    low = text.lower()
    assert "fixed asset" in low and "15%" in text and "3 years" in low, text
    assert "not cwip" in low, "the entry must say plainly that CWIP is not read: " + text
    for false_claim in ("fixed assets/cwip", "cwip converting", "cwip going live"):
        assert false_claim not in low, f"claims {false_claim!r}: {text}"
