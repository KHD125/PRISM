"""Contract: the plain-language "?" help affordance is no longer trapped in the All Data tab.

The inverted-pyramid tearsheet leads with the verdict (Layer 1 hero) and the 6-axis scorecard
(Layer 2) — the surfaces EVERY user sees first and that carry the most cryptic jargon (WCS, VCR,
Terms-of-Trade, EPS-Accel ...). This contract guards that those Layer-1/2 surfaces wire the SAME
pure-CSS `.ts-help` chip (no widget, no state — CLAUDE.md §5) from the SAME single-source glossary,
so the help reaches the casual reader, not only the expert who drills to the deepest tab.

Same testing style as test_all_data_surfacing.py: static source-block inspection + glossary import.
"""
import ast
import re
from pathlib import Path

import pytest

_UI_SRC = Path(__file__).resolve().parent.parent / "ui" / "ui_tearsheet.py"


def _fn_block(name: str) -> str:
    """Return the source of a top-level function `def name(` up to the next top-level def."""
    src = _UI_SRC.read_text(encoding="utf-8")
    start = src.find(f"def {name}")
    assert start != -1, f"{name} not found in ui_tearsheet.py"
    end = src.find("\ndef ", start + 1)
    return src[start:end if end != -1 else len(src)]


# ── The shared chip helper (single source of the `.ts-help` markup) ──
def test_help_chip_renders_glossary_tooltip():
    """help_chip(label) renders the pure-CSS `.ts-help` chip from the glossary by label."""
    from ui.ui_tearsheet import help_chip

    out = help_chip("WCS")
    assert 'class="ts-help"' in out, "help_chip must render the .ts-help affordance"
    assert "data-tip=" in out, "help_chip must carry the tooltip text in data-tip"


def test_help_chip_explicit_tip_overrides_lookup():
    """An explicit tip wins over the glossary lookup (mirrors _cell's help= override)."""
    from ui.ui_tearsheet import help_chip

    out = help_chip("WCS", "a bespoke explanation that is plenty long")
    assert "a bespoke explanation that is plenty long" in out


def test_help_chip_returns_empty_when_no_explanation():
    """No glossary entry and no explicit tip → no chip (the completeness net, never a bare '?')."""
    from ui.ui_tearsheet import help_chip

    assert help_chip("__definitely_not_a_real_label__") == ""


def test_help_chip_escapes_quotes_in_tooltip():
    """The tooltip text lands in an HTML attribute — quotes must be escaped, not break markup."""
    from ui.ui_tearsheet import help_chip

    out = help_chip("", 'has a " double quote inside it for safety')
    assert '"' not in out.split("data-tip=")[1].split(">")[0].replace('data-tip="', "", 1)[:-1] \
        or "&quot;" in out or "&#34;" in out, "double quotes in the tip must be HTML-escaped"


# ── Layer-1 / Layer-2 surfaces must wire the chip ──
def test_scorecard_wires_help_chip():
    """render_verdict_scorecard (Layer 2) must render the `?` chip — its axes + Deep Signals +
    Entry Timing strips were previously bare jargon."""
    block = _fn_block("render_verdict_scorecard")
    assert "help_chip(" in block, "the 6-axis scorecard must wire help_chip for its cryptic terms"


def test_hero_wires_help_chip():
    """render_stock_hero (Layer 1) must explain its headline numbers (composite ring, tier, Evidence)."""
    block = _fn_block("render_stock_hero")
    assert "help_chip(" in block, "the hero band must wire help_chip for the composite/tier/Evidence"


# ── Every newly-surfaced cryptic term needs a real plain-language definition ──
_LAYER2_TERMS = [
    # Deep Signals strip
    "WCS", "Econ-Profit", "VCR", "Terms-of-Trade", "Cash-Machine",
    # Entry Timing strip
    "RS", "Traj", "EPS-Accel", "Vol",
    # 6 verdict axes
    "Moat Axis", "Growth Axis", "Valuation Axis", "Balance Axis",
    "Governance Axis", "Forensics Axis",
    # Hero headline numbers
    "Composite Score", "Evidence Coverage",
]


@pytest.mark.parametrize("term", _LAYER2_TERMS)
def test_layer2_term_has_plain_language_glossary(term):
    """Each Layer-1/2 chip term must map to a real, non-empty plain-language explanation that
    explains the TERM and never judges the value (CLAUDE.md: thresholds in UI = engine drift)."""
    from ui.ui_tearsheet import _RAW_GLOSSARY

    assert term in _RAW_GLOSSARY, f"Layer-2 term {term!r} must have a _RAW_GLOSSARY entry"
    assert len(_RAW_GLOSSARY[term].strip()) >= 20, f"{term!r} tooltip too short to be real"


# ── Priority 3: the scanner grid's machine-named headers get plain-language tooltips ──
_SCANNER_SRC = Path(__file__).resolve().parent.parent / "ui" / "ui_scanner.py"


def test_deep_scanner_wires_header_help_tooltips():
    """The LIVE Deep Scanner (app.py st.dataframe) must wire help= into its column_config from
    _SCANNER_HEADER_TIPS, so every machine-named header self-explains on hover. Replaces the dead
    AgGrid headerTooltip contract — render_scanner_grid was retired (it was never rendered)."""
    app_src = (Path(__file__).resolve().parent.parent / "app.py").read_text(encoding="utf-8")
    assert "from ui.ui_scanner import _SCANNER_HEADER_TIPS" in app_src, (
        "Deep Scanner must import the shared tooltip map"
    )
    assert "help=_SCANNER_HEADER_TIPS.get(" in app_src, (
        "Deep Scanner column_config must pass help= from _SCANNER_HEADER_TIPS"
    )


def test_scanner_tips_reuse_the_shared_glossary():
    """The scanner must pull its tips from the SAME glossary as the tearsheet — one definition,
    no drift between the grid header and the tearsheet '?' chip."""
    from ui.ui_scanner import _SCANNER_HEADER_TIPS
    from ui.ui_tearsheet import _RAW_GLOSSARY

    assert _SCANNER_HEADER_TIPS, "scanner header-tip map must be populated"
    assert _SCANNER_HEADER_TIPS["composite_score"] == _RAW_GLOSSARY["Composite Score"], (
        "scanner composite_score tip must be the exact glossary sentence (single source of truth)"
    )
    for col, tip in _SCANNER_HEADER_TIPS.items():
        assert isinstance(tip, str) and len(tip.strip()) >= 20, f"{col!r} header tip too short"


# ── Priority 4: every sidebar "Refine" filter explains itself (Streamlit native help=) ──
# The discovery filter cascade lives in ui/ui_discovery.py (stateful counterpart to ui_tearsheet).
_DISCOVERY_SRC = Path(__file__).resolve().parent.parent / "ui" / "ui_discovery.py"


@pytest.mark.parametrize("label", ["Gate-passed only", "Min Quality Score", "Min Composite Score"])
def test_refine_filter_has_help_text(label):
    """Each power-user Refine-group filter must carry a help= tooltip (the native '?' beside the
    widget) — Min Quality and Min Composite are a pre-/post-penalty pair and must both be explained."""
    src = _DISCOVERY_SRC.read_text(encoding="utf-8")
    i = src.find(f'"{label}"')
    assert i != -1, f"{label!r} filter not found in ui_discovery.py"
    rest = src[i:]
    # Bound the scan to THIS widget's call: stop at the start of the next st.* widget.
    m = re.search(r"st\.(slider|checkbox|selectbox|multiselect)\(", rest[1:])
    window = rest[: (m.start() + 1) if m else 400]
    assert "help=" in window, f"the {label!r} sidebar filter must have a help= tooltip"


# ── Framework cards: every card explains the framework's IDEA (handbook-sourced), not just its gate ──
def test_every_framework_card_has_an_idea_tooltip():
    """Coverage contract: every framework name in _FW_META must have a plain-language _FW_IDEA
    one-liner — so a future new framework can't ship a card without its beginner explainer
    (ties into CLAUDE.md §7 'ships complete')."""
    from ui.ui_tearsheet import _FW_IDEA, _FW_META

    names = set(_FW_META)   # _FW_META is now module-scope (importable) — read it directly, not via regex
    assert len(names) >= 35, f"expected ~37 framework names in _FW_META, found {len(names)}"
    missing = sorted(n for n in names if n not in _FW_IDEA)
    assert not missing, f"these framework cards have no plain-language idea tooltip: {missing}"
    for n in names:
        assert len(_FW_IDEA[n].strip()) >= 20, f"{n!r} idea tooltip too short to be real"


def test_framework_cards_wire_help_chip():
    """The framework card must render the '?' idea tooltip (the gate spec stays visible beneath)."""
    block = _fn_block("render_guru_frameworks")
    assert "help_chip(" in block, "framework cards must wire the help_chip '?' idea tooltip"
    assert "_FW_IDEA" in block, "framework cards must source the tooltip from _FW_IDEA"


# ── Score strip: each of the 5 sub-scores explains itself (distinct from the 6 verdict axes) ──
@pytest.mark.parametrize(
    "term", ["Moat Score", "Growth Score", "Cash Score", "Momentum Score", "Governance Score"]
)
def test_score_strip_subscore_has_glossary(term):
    from ui.ui_tearsheet import _RAW_GLOSSARY

    assert term in _RAW_GLOSSARY, f"score-strip sub-score {term!r} must have a _RAW_GLOSSARY entry"
    assert len(_RAW_GLOSSARY[term].strip()) >= 20, f"{term!r} tooltip too short to be real"


def test_score_strip_wires_help_chip():
    block = _fn_block("render_score_strip")
    assert "help_chip(" in block, "the 5-cell score strip must wire the '?' help chip"


# ── Round 5: ui_components is the SINGLE home of the chip; discovery-card parity + sidebar gaps ──
_COMPONENTS_SRC = Path(__file__).resolve().parent.parent / "ui" / "ui_components.py"


def test_chip_home_is_ui_components():
    """help_chip + _RAW_GLOSSARY now live in ui_components (which already owns the .ts-help CSS).
    Every glossary key must resolve (a dropped key would silently un-explain a term). The count grows
    only as new cells are surfaced with their tooltips (158 after the 2026-06-17 All-Data pass-3)."""
    from ui.ui_components import help_chip, _RAW_GLOSSARY

    assert 'class="ts-help"' in help_chip("WCS"), "help_chip must render the .ts-help affordance"
    assert len(_RAW_GLOSSARY) >= 158, "glossary must not lose entries (158 after the pass-3 surfacing)"


def test_help_chip_exported_from_ui_package_facade():
    """app.py imports names from the `ui` package (`from ui import ...`), so help_chip must be
    re-exported by ui/__init__.py too — otherwise the Discovery legend crashes the tab at runtime
    (a gap a static source check misses; caught here so the suite guards it, not just visual-check)."""
    from ui import help_chip as facade_help_chip
    import ui.ui_components as comp

    assert facade_help_chip is comp.help_chip, "ui package must re-export the SAME help_chip object"


def test_tearsheet_reexports_the_same_objects():
    """Backward-compat + single source of truth: ui_tearsheet must re-export the SAME objects
    (identity, not copies) so existing `from ui.ui_tearsheet import ...` keeps working and the two
    surfaces can never drift."""
    import ui.ui_components as comp
    import ui.ui_tearsheet as ts

    assert ts._RAW_GLOSSARY is comp._RAW_GLOSSARY, "glossary must be the SAME object (single source)"
    assert ts.help_chip is comp.help_chip, "help_chip must be the SAME object, re-exported"


def test_scanner_sources_glossary_from_components():
    """The scanner must import the glossary from its single home (ui_components), not the tearsheet."""
    src = _SCANNER_SRC.read_text(encoding="utf-8")
    assert "from ui.ui_components import" in src, "scanner must import from ui_components"
    assert "from ui.ui_tearsheet import" not in src, "scanner must no longer import from ui_tearsheet"


def test_stock_card_does_not_repeat_subscore_chip():
    """ANTI-CLUTTER contract: the per-card sub-score bars must NOT each carry a '?' — that repeated
    the same 5 identical tooltips on every card down the scan list (~100 chips). The sub-scores are
    explained ONCE by a legend above the ranked list instead (test_discovery_list_has_*_legend)."""
    src = _COMPONENTS_SRC.read_text(encoding="utf-8")
    start = src.find("def render_stock_card")
    assert start != -1, "render_stock_card not found"
    end = src.find("\ndef ", start + 1)
    block = src[start : end if end != -1 else len(src)]
    assert "help_chip(" not in block, "render_stock_card must NOT repeat the '?' chip on every card"


def test_discovery_list_has_one_time_subscore_legend():
    """The Discovery ranked list (app.py) explains the 5 sub-scores ONCE via a help_chip legend
    above the cards — not per card. app.py must both import help_chip and use it near 'Top Picks'."""
    app_src = (Path(__file__).resolve().parent.parent / "app.py").read_text(encoding="utf-8")
    assert app_src.count("help_chip") >= 2, "app.py must import help_chip AND use it in the legend"
    i = app_src.find("Top Picks")
    assert i != -1, "Top Picks list header not found in app.py"
    assert "help_chip(" in app_src[i : i + 1200], "discovery list must wire a one-time help_chip legend"


@pytest.mark.parametrize("label", ["Sector", "Moat"])
def test_discovery_filter_has_help_text(label):
    """The Sector + Moat discovery filters must carry native help= (parity with every sibling filter).
    Stateful sidebar → native Streamlit help=, NOT the .ts-help chip."""
    src = _DISCOVERY_SRC.read_text(encoding="utf-8")
    # Anchor to the WIDGET call whose first arg is the label — not any string. (The _CHIP_META
    # registry also lists "Sector"/"Moat" as chip labels, so a bare src.find would match that tuple
    # first and miss the real widget.) The assertion below is unchanged: the widget must carry help=.
    wm = re.search(r'(?:st\.selectbox|st\.multiselect|st\.slider|st\.checkbox|_ms_cascade)\(\s*"'
                   + re.escape(label) + r'"', src)
    assert wm is not None, f"{label!r} filter widget not found in ui_discovery.py"
    i = wm.start()
    rest = src[i:]
    # Bound to THIS widget's call: stop at the next widget (incl. the _ms_cascade helper, so a
    # following sibling's help= can't leak into this window and produce a false pass).
    nxt = re.search(r"(st\.selectbox|st\.slider|st\.checkbox|st\.multiselect|_ms_cascade)\(", rest[1:])
    window = rest[: (nxt.start() + 1) if nxt else 500]
    assert "help=" in window, f"the {label!r} discovery filter must have a help= tooltip"


# ── Deep Scanner: every column a view-preset surfaces must explain itself on hover ──
# The 5 column-view presets (app.py `_DS_VIEWS`) are what a user deliberately switches between, so
# every column they expose should carry an AgGrid headerTooltip. This pins the scanner tip map to the
# preset definition — a future preset column can't ship as a bare machine-name header.
_DS_IDENTITY_COLS = {"name", "sector", "market_category"}


def _ds_view_columns() -> set:
    """Every column string across app.py's inline `_DS_VIEWS` preset dict (AST — no execution)."""
    app_src = (Path(__file__).resolve().parent.parent / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(app_src, filename="app.py")
    cols: set = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "_DS_VIEWS" for t in node.targets)
            and isinstance(node.value, ast.Dict)
        ):
            for value in node.value.values:
                if isinstance(value, ast.List):
                    for el in value.elts:
                        if isinstance(el, ast.Constant) and isinstance(el.value, str):
                            cols.add(el.value)
    return cols


_DS_VIEW_COLS = _ds_view_columns()


def test_ds_views_parsed_nonempty():
    """Guard against a vacuous pass if the AST parse / _DS_VIEWS location ever changes."""
    assert len(_DS_VIEW_COLS) >= 20, (
        f"expected ~33 preset columns parsed from app.py _DS_VIEWS, got {len(_DS_VIEW_COLS)}"
    )


def test_every_scanner_preset_column_has_header_tip():
    """Every non-identity column the Deep Scanner's view presets surface must have a header tooltip —
    so switching to Quality / Valuation / Forensic / Technical never shows bare machine-name headers."""
    from ui.ui_scanner import _SCANNER_HEADER_TIPS

    missing = sorted(
        c for c in _DS_VIEW_COLS
        if c not in _DS_IDENTITY_COLS and c not in _SCANNER_HEADER_TIPS
    )
    assert not missing, f"Deep Scanner preset columns with no header tooltip: {missing}"


def _deep_scanner_block(tree):
    """The `with tabs[1]:` (Deep Scanner) With-node — scopes label extraction to this tab so it can't
    pick up a same-named key from an unrelated dict elsewhere in app.py."""
    for node in ast.walk(tree):
        if isinstance(node, ast.With):
            for item in node.items:
                ctx = item.context_expr
                if (isinstance(ctx, ast.Subscript) and isinstance(ctx.value, ast.Name)
                        and ctx.value.id == "tabs"
                        and isinstance(ctx.slice, ast.Constant) and ctx.slice.value == 1):
                    return node
    return None


def _ds_labeled_columns(block) -> set:
    """Columns that get an explicit friendly header — the keys of every str/tuple-valued dict literal
    in the Deep Scanner block (the score / number / boolean / text config label maps). _DS_VIEWS is
    list-valued so it's excluded: a column merely listed in a preset but never labeled is NOT here."""
    labeled: set = set()
    for node in ast.walk(block):
        if isinstance(node, ast.Dict) and node.keys:
            keys_str = all(isinstance(k, ast.Constant) and isinstance(k.value, str) for k in node.keys)
            vals_ok = all(isinstance(v, (ast.Constant, ast.Tuple)) for v in node.values)
            if keys_str and vals_ok:
                labeled.update(k.value for k in node.keys)
    return labeled


def test_deep_scanner_columns_have_friendly_headers():
    """Every column a _DS_VIEWS preset surfaces must get an explicit friendly header label (via the
    score / number / boolean / text config maps) — never fall through to a raw snake_case header.
    The 3 identity columns (name / sector / market_category) rendered raw until labeled (2026-06-20).
    Sibling of test_every_scanner_preset_column_has_header_tip: that pins TOOLTIPS (identity cols
    exempt); this pins HEADER LABELS (identity cols included — a bare 'sector' header looks unfinished)."""
    app_src = (Path(__file__).resolve().parent.parent / "app.py").read_text(encoding="utf-8")
    block = _deep_scanner_block(ast.parse(app_src, filename="app.py"))
    assert block is not None, "could not locate the `with tabs[1]:` Deep Scanner block in app.py"
    labeled = _ds_labeled_columns(block)
    raw = sorted(c for c in _DS_VIEW_COLS if c not in labeled)
    assert not raw, (
        f"Deep Scanner preset columns with no friendly header (render as raw snake_case): {raw}. "
        f"Add each to a column-config label map (score / number / boolean / text)."
    )


# ── Forensics tab: the 5 Forensic-Perimeter KPIs explain themselves (the headline forensic numbers) ──
# The Score Multiplier especially — it IS the penalty that cuts the composite, so a bare "75%" is
# meaningless without the chip. The red-flag rows / Schilit / Fisher already self-document inline.
_FORENSIC_KPI_KEYS = ["Red Flags", "Forensic Scr", "Forensic Mult", "Piotroski", "Mgmt Integrity"]


@pytest.mark.parametrize("key", _FORENSIC_KPI_KEYS)
def test_forensic_kpi_terms_have_glossary(key):
    from ui.ui_components import _RAW_GLOSSARY

    assert key in _RAW_GLOSSARY, f"forensic KPI term {key!r} must have a glossary entry"
    assert len(_RAW_GLOSSARY[key].strip()) >= 20, f"{key!r} tooltip too short to be real"


def test_forensic_perimeter_wires_kpi_help_chips():
    """The Forensic Perimeter's 5 headline KPIs (Red Flags / Forensic Score / Score Multiplier /
    Piotroski / Mgmt Integrity) must each carry a '?' sourced from the shared glossary."""
    block = _fn_block("render_forensic_perimeter")
    assert "help_chip(" in block, "the forensic KPI strip must wire help_chip"
    for k in _FORENSIC_KPI_KEYS:
        assert (f"help_chip('{k}')" in block) or (f'help_chip("{k}")' in block), (
            f"forensic KPI {k!r} is not wired with help_chip"
        )
