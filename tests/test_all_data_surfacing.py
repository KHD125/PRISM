"""Contract: the All Data tab (render_raw_signals) surfaces the verified-alive orphans
that were previously computed-but-hidden, so its "Full Universe Output" promise stays honest.

Each column below was confirmed POPULATED and NON-DEGENERATE on the live 2107-stock universe
(2026-06-15) before being surfaced — adding a dead/constant column would re-create the
"0.0 means missing-or-zero?" ambiguity this tab exists to avoid. Rejected at that gate (and
therefore deliberately NOT asserted here): effective_tax_rate_pct (byte-identical to the shown
tax_rate_est), reinvestment_rate (≡1.0) and retention_rate (≡100.0) (DPR-broken constants),
and dividend_yield_ratio (4% populated).
"""
from pathlib import Path

import pytest

_UI_SRC = Path(__file__).resolve().parent.parent / "ui" / "ui_tearsheet.py"


def _raw_signals_block() -> str:
    src = _UI_SRC.read_text(encoding="utf-8")
    start = src.find("def render_raw_signals")
    assert start != -1, "render_raw_signals not found in ui_tearsheet.py"
    end = src.find("\ndef ", start + 1)
    return src[start:end if end != -1 else len(src)]


# (column, why it earns a cell)
SURFACED_ORPHANS = [
    ("ibas_architecture_score",     "IBAS moat pillar — the 4 average to the scorecard's IBAS aggregate"),
    ("ibas_innovation_score",       "IBAS moat pillar"),
    ("ibas_reputation_score",       "IBAS moat pillar"),
    ("ibas_strategic_assets_score", "IBAS moat pillar"),
    ("malik_label",                 "Malik verdict word (Score/Pass were already shown, not the label)"),
    ("lynch_category",              "Lynch archetype: Fast Grower / Stalwart / Slow Grower / Turnaround"),
    ("accruals_ratio",              "Sloan accruals — forensic earnings-quality signal"),
    ("economic_profit_spread",      "ROIC − WACC spread, complements the shown economic_profit"),
    ("fcf_imputed_flag",            "FCF provenance — the FCF shown is imputed, not raw"),
    ("fcf_reconstructed_flag",      "FCF provenance — the FCF shown is reconstructed, not raw"),
]


@pytest.mark.parametrize("col,reason", SURFACED_ORPHANS, ids=[c for c, _ in SURFACED_ORPHANS])
def test_verified_orphan_is_surfaced(col, reason):
    """Each verified-alive orphan must appear in render_raw_signals (kept honest, no drift)."""
    block = _raw_signals_block()
    assert col in block, f"render_raw_signals must surface {col} — {reason}"
