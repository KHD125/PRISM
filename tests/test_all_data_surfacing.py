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
    # ── pass 2: valuation lenses + competitive-position finals (verified alive 2026-06-15) ──
    ("ps_ratio",                    "Price-to-Sales — a standard valuation multiple that was absent"),
    ("fgv_pct",                     "Future Growth Value — share of price betting on future growth"),
    ("mef_label",                   "Moat Endurance verdict (widening / intact / eroding / degrading)"),
    ("sector_leader_score",         "leadership rank within the company's own sector"),
    ("ebit_vs_rev_spread_3y",       "numeric 3Y operating leverage (the Op-Leverage Yes/No was binary)"),
    # ── pass 3: verified-alive orphans surfaced 2026-06-17 (ui_coverage + live liveness re-gate) ──
    ("moat_endurance_factor",       "numeric moat-endurance (current ÷ 10y-median ROCE) behind the MEF label"),
    ("roe_elite_flag",              "ROE ≥ 35% — MOSL 6th-study elite-return flag"),
    ("roe_trend_rising_flag",       "ROE above its 5y AND 10y median — improving return trajectory"),
    ("pat_decline_1y_pct",          "one-year PAT change % (consistency / downside)"),
    ("value_migration_flag",        "top-quartile sector revenue growth — value migrating to the business"),
    ("cwip_ratio",                  "CWIP ÷ fixed assets — capacity build-out intensity"),
    ("ebitda_to_pat_gap_pct",       "(Dep+Interest+Tax) ÷ EBITDA — operating-to-net profit drag"),
    ("supplier_float_score",        "negative-CCC supplier-float moat score (0-100)"),
    ("negative_wc_flag",            "negative cash-conversion cycle — self-funded growth"),
    ("payoff_ratio_proxy",          "Mauboussin payoff multiple (fair PE ÷ PE for undervalued names)"),
    ("expectations_gap_rank",       "percentile rank of the market-implied expectations gap"),
    ("trend_score",                 "0-100 technical trend-quality composite"),
    ("dm_forensic_flag_count",      "count of Mukherjea 'Diamonds' forensic checks fired"),
    ("sqglp_score",                 "MOSL SQGLP 5-pillar score (0-5) — strictest QGLP variant"),
    ("vqs_score",                   "quantitative value-quality composite (Gray-style)"),
    ("sector_consistent_type",      "sector structural type: Consistent vs Volatile"),
    # ── pass 4: curated residual orphans (coverage audit 2026-06-17, verified-alive + non-redundant) ──
    ("mcap_tier",                   "market-cap tier label (Mega/Large/Mid/Small/Micro/Nano) - size context"),
    ("roe_turnaround_flag",         "ROE<15 but rising above its 5y trend - early-innings turnaround bargain"),
    ("category_winner_flag",        "14th-WCS sector leader: top-30% sector ROCE + above-market revenue growth"),
    ("enduring_vc_flag",            "18th-WCS enduring value creator: EP+ AND consistent AND decade of ROE>=15"),
    ("compound_growth_power_flag",  "growth >=15/12/10% across 3/5/10Y - compounding at every horizon"),
    ("consistent_in_volatile_flag", "consistent earner inside a structurally volatile sector"),
    ("d51_qmom_quality_score",      "Gray QMOM quality (0-1): ROCE/low-D-E/CFO/gross-profitability rank average"),
    ("eps_strong_acceleration",     "EPS growth itself accelerating (O'Neil/CAN SLIM core)"),
    ("uu_setup_flag",               "15th-WCS UU setup: small-cap + payback<1 + ROE turnaround"),
    ("fast_creator_setup",          "fastest-creator setup: mcap<4kCr + PE<10 + PAT CAGR>=35%"),
    ("cyclical_mirage_flag",        "revenue surge with weak 10Y ROCE - growth not backed by capital returns"),
    ("dilution_vampire_flag",       "rev>=30% but ROE<12% funded by equity dilution - value-eroding growth"),
    ("pledge_rerate_catalyst",      "promoter de-pledging (>10% -> -30% -> near-clean) - re-rating catalyst"),
]

# New cell label → its expected glossary key (the "?" tooltip must exist for each new term).
_PASS2_LABELS = ["P/S", "FGV", "Moat Endurance", "Sector Leader", "Op Lev (3Y)"]

# pass-3 cell labels (each must carry a "?" glossary tooltip).
_PASS3_LABELS = [
    "Moat Endur ×", "Elite ROE", "ROE Rising", "PAT 1Y Δ %", "Value Migration",
    "CWIP/FA %", "EBITDA→PAT Gap", "Supplier Float", "Negative WC", "Payoff Ratio",
    "Exp Gap Rank", "Trend Score", "Diamond Flags", "SQGLP Score", "QV Score", "Sector Type",
]


@pytest.mark.parametrize("col,reason", SURFACED_ORPHANS, ids=[c for c, _ in SURFACED_ORPHANS])
def test_verified_orphan_is_surfaced(col, reason):
    """Each verified-alive orphan must appear in render_raw_signals (kept honest, no drift)."""
    block = _raw_signals_block()
    assert col in block, f"render_raw_signals must surface {col} — {reason}"


# ── Plain-language "?" tooltip glossary (the Learner persona) ──
def test_glossary_entries_are_nonempty_plain_language():
    """Every glossary key must map to a real, non-empty plain-language sentence — this is the
    completeness net: a term with no explanation is a flag to write one or remove the cell."""
    from ui.ui_tearsheet import _RAW_GLOSSARY
    assert len(_RAW_GLOSSARY) >= 50, "glossary should cover the tab's jargon"
    for term, definition in _RAW_GLOSSARY.items():
        assert isinstance(definition, str), f"{term!r} definition must be a string"
        assert len(definition.strip()) >= 20, (
            f"{term!r} definition is too short to be a real plain-language explanation"
        )


def test_cell_helper_wires_the_tooltip():
    """The _cell helper must render the CSS '?' affordance via the shared help_chip() (no
    widget/state). The chip markup + glossary lookup now live in ONE place (help_chip), reused by
    the Layer-1 hero and Layer-2 scorecard too — see test_tooltip_coverage.py."""
    block = _raw_signals_block()
    assert "help_chip(" in block, "_cell must wire the tooltip through the shared help_chip()"
    # Single source of the .ts-help markup + glossary lookup now lives in help_chip, which was moved
    # to ui/ui_components.py (the module that owns the .ts-help CSS) and is re-imported by ui_tearsheet.
    comp_src = (_UI_SRC.parent / "ui_components.py").read_text(encoding="utf-8")
    chip = comp_src[comp_src.find("def help_chip"):comp_src.find("def render_stock_card")]
    assert 'class="ts-help"' in chip, "help_chip must render the .ts-help '?' affordance"
    assert "_RAW_GLOSSARY.get(label" in chip, "help_chip must auto-look-up the glossary by label"


@pytest.mark.parametrize("label", _PASS2_LABELS)
def test_new_labels_have_glossary_tooltip(label):
    """Every newly-surfaced cell must carry a plain-language '?' tooltip (no bare jargon)."""
    from ui.ui_tearsheet import _RAW_GLOSSARY
    assert label in _RAW_GLOSSARY, f"new cell {label!r} must have a _RAW_GLOSSARY entry"
    assert len(_RAW_GLOSSARY[label].strip()) >= 20, f"{label!r} tooltip too short"


@pytest.mark.parametrize("label", _PASS3_LABELS)
def test_pass3_labels_have_glossary_tooltip(label):
    """Every pass-3 surfaced cell must carry a plain-language '?' tooltip (no bare jargon)."""
    from ui.ui_tearsheet import _RAW_GLOSSARY
    assert label in _RAW_GLOSSARY, f"new cell {label!r} must have a _RAW_GLOSSARY entry"
    assert len(_RAW_GLOSSARY[label].strip()) >= 20, f"{label!r} tooltip too short"


# pass-4 cell labels (each must carry a "?" glossary tooltip).
_PASS4_LABELS = [
    "Mcap Tier", "ROE Turnaround", "Category Winner", "Enduring VC", "Compound Power",
    "Steady in Volatile", "QMOM Quality", "EPS Accelerating", "UU Setup", "Fast Creator",
    "Cyclical Mirage", "Dilution Vampire", "Pledge Re-rate",
]


@pytest.mark.parametrize("label", _PASS4_LABELS)
def test_pass4_labels_have_glossary_tooltip(label):
    """Every pass-4 surfaced cell must carry a plain-language '?' tooltip (no bare jargon)."""
    from ui.ui_tearsheet import _RAW_GLOSSARY
    assert label in _RAW_GLOSSARY, f"new cell {label!r} must have a _RAW_GLOSSARY entry"
    assert len(_RAW_GLOSSARY[label].strip()) >= 20, f"{label!r} tooltip too short"
