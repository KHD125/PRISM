"""Contract: the concept reference — every CATEGORICAL VALUE-label PRISM shows is explained, and the
explanations are not shallow. This is the permanence guarantee for "everything explainable is explained":
a new label can't ship unexplained because these tests fail.

Three guards:
  1. COVERAGE — every user-facing FILTERABLE value (the ones on a Tear-Sheet cell you can also filter on)
     has a CONCEPT_REFERENCE entry. This is the gap the user named (Wealth Creator, Growth Trap, …).
  2. DRIFT — the _CANONICAL enumeration is pinned to the LIVE ui_discovery filters (every value must
     appear there), so it can't silently fall out of sync with the code it claims to cover.
  3. QUALITY — every explanation clears a 40-char floor (catches "there but not properly explained").
"""
from pathlib import Path

from ui.ui_reference_data import CONCEPT_REFERENCE

_ROOT = Path(__file__).resolve().parent.parent
_DISC = (_ROOT / "ui" / "ui_discovery.py").read_text(encoding="utf-8")

# Distinctive substring of every FILTERABLE categorical value (emoji/parens stripped). Cross-checked
# to ui_discovery below, so these are the live filter labels — not a free-floating wishlist.
_CANONICAL = {
    "Moat × Growth": ["Wealth Creator", "Quality Trap", "Growth Trap", "Wealth Destroyer"],
    "PEG Zone":      ["Deep Value", "Fair PEG", "Stretched", "Expensive", "Overpriced", "Declining"],
    "Buy Zone":      ["Perfect Entry", "Standard Zone", "Extended", "Below Stop", "Uncharted"],
    "Weinstein":     ["Stage 2 Advancing", "Stage 1 Basing", "Stage 3 Top", "Stage 4 Declining"],
    "Lynch":         ["Fast Grower", "Stalwart", "Slow Grower"],
    "Moat Endurance":["Expanding", "Intact", "Eroding", "Degrading"],
    "Cash-Flow Tri": ["Growth Phase", "Debt Trap", "Mixed Pattern"],
    "Smart-Money":   ["Elite Accumulation", "Strong Accumulation", "Moderate Interest", "Distribution"],
    "Corporate":     ["GREAT", "GOOD", "GRUESOME"],
    "Verdict":       ["BUY", "WATCH", "AVOID"],
    "Cyclicality":   ["Deep Cyclical", "Defensive", "Structural-Growth", "Financials", "Catch-all"],
    "Capital Phase": ["Hot Capital", "Capital Starved", "Neutral"],
}

_CONCEPT_LABELS = [lbl for entries in CONCEPT_REFERENCE.values() for lbl, _ in entries]


def test_every_filterable_value_is_explained():
    missing = [f"{cat}:{v}" for cat, vals in _CANONICAL.items() for v in vals
               if not any(v in lbl for lbl in _CONCEPT_LABELS)]
    assert not missing, f"value-labels with no CONCEPT_REFERENCE explanation: {missing}"


def test_canonical_is_pinned_to_the_live_filters():
    missing = [f"{cat}:{v}" for cat, vals in _CANONICAL.items() for v in vals if v not in _DISC]
    assert not missing, f"_CANONICAL drifted from ui_discovery (not found in the filter code): {missing}"


def test_no_shallow_explanation():
    thin = [(cat, lbl, len(exp.strip())) for cat, entries in CONCEPT_REFERENCE.items()
            for lbl, exp in entries if len(exp.strip()) < 40]
    assert not thin, f"explanations under the 40-char quality floor: {thin}"


def test_reference_data_is_pure():
    # accurate purity check: no streamlit IMPORT (the docstring may mention the word). The module is
    # pure data — a dict literal — so it imports nothing.
    src = (_ROOT / "ui" / "ui_reference_data.py").read_text(encoding="utf-8")
    assert "import streamlit" not in src and "\nimport " not in src, \
        "ui_reference_data.py must be pure data — no imports / no Streamlit"
