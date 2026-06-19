"""Contract: every config.HARD_GATES safety gate is explained in the Reference.

Hard gates are NOT np.select categorical labels — they're `config.HARD_GATES` dict entries, rendered
on the Config tab and as the Tear-Sheet's SYSTEM-REJECTED verdict reason. The categorical-label
coverage test (test_concept_reference) never sees them, so this is the dedicated, drift-proof guard:
a new hard gate can't ship without its plain-language Reference entry (it RED-fails listing the gate).
"""
import re

from config import HARD_GATES
from ui.ui_reference_data import CONCEPT_REFERENCE


def _norm(s: str) -> str:
    """snake_case key / display label → space-separated lowercase tokens for substring matching."""
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def test_every_hard_gate_has_a_reference_entry():
    """Each HARD_GATES key (e.g. 'debt_safety') must appear in some CONCEPT_REFERENCE label
    (e.g. 'Debt Safety (gate)') — so every gate a stock can be SYSTEM-REJECTED on is decodable
    in the 📖 Reference tab."""
    labels = [_norm(lbl) for cat in CONCEPT_REFERENCE.values() for (lbl, _exp) in cat]
    missing = sorted(g for g in HARD_GATES if not any(_norm(g) in lbl for lbl in labels))
    assert not missing, (
        f"{len(missing)} hard gate(s) with NO Reference explanation — add each to "
        f"CONCEPT_REFERENCE's Hard Gates category: {missing}"
    )
