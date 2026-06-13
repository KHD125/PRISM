"""
test_corporate_class_taxonomy.py
================================
Contract for the Great/Good/Gruesome taxonomy (13th MOSL Wealth Creation Study,
2003-08, Annexure 2 p.31) as implemented in run_full_scoring -> corporate_class.

Book criteria: GREAT = avg RoE > 25% (the 15-25% band is explicitly GOOD, p.29);
GRUESOME = avg RoE < 10%. The engine uses ROCE (leverage-immune) and:
  • GREAT requires roce_med_10y >= 25  (book-aligned 2026-06-13; was 20 — which mislabeled
    the book's 20-25% GOOD band as GREAT).
  • GRUESOME at roce_med_10y < 12 is KEPT (not the book's RoE<10): 12 = COST_OF_EQUITY, so
    ROCE<12 = earning below cost of capital = destroying economic value (coherent with the
    economic_profit engine). Deliberate, justified deviation.

These assertions depend only on roce_med_10y (a mapped column that survives the pipeline),
so they are robust to recomputation of the asset-light FCF gate.

Run with: pytest tests/test_corporate_class_taxonomy.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
sys.path.insert(0, os.path.dirname(__file__))

from data_engine import compute_derived_signals
from forensic_engine import compute_forensic_signals
from scoring_engine import run_full_scoring
from test_data_quality_fixes import _frame


def _classify(roce_med_10y_vals):
    n = len(roce_med_10y_vals)
    f = _frame(n=n, roce_med_10y=roce_med_10y_vals, roce=roce_med_10y_vals)
    out = run_full_scoring(compute_forensic_signals(compute_derived_signals(f)))
    return list(out["corporate_class"])


def test_good_band_20_to_25_is_not_great():
    """The book puts the 15-25% band in GOOD (p.29). A 22% / 24.9% ROCE business must NOT be
    GREAT — this is the book-alignment fix (was wrongly GREAT at the old >=20 threshold)."""
    cls = _classify([22.0, 24.9])
    assert cls[0] != "🏆 GREAT", "22% ROCE is the book's GOOD band, not GREAT"
    assert cls[1] != "🏆 GREAT", "24.9% ROCE is below the book's >25% GREAT bar"


def test_below_cost_of_capital_is_gruesome():
    """ROCE < 12 (= COST_OF_EQUITY) = below cost of capital = GRUESOME. Locks the kept-12
    decision: an 11% ROCE business must stay GRUESOME (would flip to GOOD if aligned to book's 10)."""
    cls = _classify([8.0, 11.0])
    assert cls[0] == "💀 GRUESOME"
    assert cls[1] == "💀 GRUESOME", "11% ROCE is below the 12% cost-of-capital floor → GRUESOME"


def test_above_cost_of_capital_not_gruesome():
    """A 15% ROCE business (above the 12% floor) must NOT be GRUESOME."""
    cls = _classify([15.0])
    assert cls[0] != "💀 GRUESOME"
