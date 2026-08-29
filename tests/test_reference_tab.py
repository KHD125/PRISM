import ast
import html
import os

from ui.ui_components import _RAW_GLOSSARY
from ui.ui_reference import render_reference

_REF = os.path.join(os.path.dirname(__file__), "..", "ui", "ui_reference.py")


def test_renders_every_glossary_term():
    """Coverage guard: an empty query renders ALL 173 terms (escaping-aware)."""
    out = render_reference(_RAW_GLOSSARY, "")
    missing = [t for t in _RAW_GLOSSARY if html.escape(t) not in out]
    assert not missing, f"Reference dropped {len(missing)} terms: {missing[:5]}"


def test_query_filters_to_strict_subset():
    """A distinctive query narrows to its term and excludes unrelated ones."""
    out = render_reference(_RAW_GLOSSARY, "Dilution Vampire")
    assert html.escape("Dilution Vampire") in out
    assert html.escape("ROCE Current") not in out          # unrelated term filtered out
    assert out != render_reference(_RAW_GLOSSARY, "")        # strictly fewer than all


def test_no_match_is_graceful():
    """A query that matches nothing returns a friendly message, not broken/empty HTML."""
    out = render_reference(_RAW_GLOSSARY, "zzzznotarealterm")
    assert "No terms match" in out


def test_special_chars_are_escaped():
    """Definitions contain '<' — they must be HTML-escaped, not injected raw."""
    out = render_reference({"X": "Nano (<₹100 Cr)"}, "")
    assert "&lt;₹100" in out and "(<₹100" not in out


def test_render_is_pure_no_streamlit_calls():
    """Structural purity: ui_reference.py makes ZERO st.* calls (widgets live in app.py)."""
    tree = ast.parse(open(_REF, encoding="utf-8").read(), filename=_REF)
    st_calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name) and n.func.value.id == "st"
    ]
    assert not st_calls, "ui_reference.py must be pure HTML — no st.* calls"
    assert isinstance(render_reference({"Foo": "bar"}, ""), str)


# ── Framework registry ON SCREEN (2026-08-28) ────────────────────────────────────────────────
def test_frameworks_are_searchable_on_screen():
    """Before 2026-08-28 the Markdown export documented all 37 frameworks while on-screen search
    could find only 6 — the answer to "what is Dhandho Asymmetry?" lived in the download but not
    the search box. The on-screen section renders from the SAME _FW_META-derived dict the export
    consumes, so the two can never disagree."""
    from ui.ui_reference import render_frameworks
    from ui.ui_tearsheet import _FW_META
    fw = {name: {"emoji": meta[1], "name": name, "desc": meta[2]} for name, meta in _FW_META.items()}
    html_all = render_frameworks(fw, "")
    for name in ("Dhandho Asymmetry", "QGLP", "100-Bagger"):
        assert name in html_all, f"{name} missing from the on-screen framework registry"
    assert render_frameworks(fw, "dhandho") != ""
    assert render_frameworks(fw, "zzz-no-such-framework") == ""


def test_app_renders_the_framework_registry():
    """Updated 2026-08-29 for the two-mode layout: the registry renders through _show_frameworks,
    which must be wired into BOTH modes (see test_reference_two_mode_layout for the full contract)."""
    import io as _io, os
    src = _io.open(os.path.join(os.path.dirname(__file__), "..", "app.py"), encoding="utf-8").read()
    assert "render_frameworks(_fw_md, q)" in src, (
        "app.py no longer renders the framework registry on screen — the export/search asymmetry is back"
    )
    assert "Framework Registry" in src


# ── Wealth Creation Studies corpus (2026-08-29) ──────────────────────────────────────────────
def test_wcs_studies_entries_are_complete_and_verified_shape():
    """The honesty contract: an entry ships only after a COMPLETE read of the study, and every
    entry carries the full schema (a study summarized without its 'In PRISM' mapping, or with an
    empty summary, is a half-entry that must not ship). Entries stay in study order."""
    from ui.ui_reference_data import WCS_STUDIES
    assert 1 <= len(WCS_STUDIES) <= 30
    for s in WCS_STUDIES:
        for key in ("study", "years", "pub", "theme", "says", "prism"):
            assert len(str(s.get(key, "")).strip()) > 0, f"{s.get('study')} missing {key}"
        assert len(s["says"]) >= 200, f"{s['study']}: summary under the depth floor — not 'best explained'"
        assert len(s["prism"]) >= 80, f"{s['study']}: the engine mapping is too thin"
    nums = [int(str(s["study"]).split()[0].rstrip("stndrh")) for s in WCS_STUDIES]
    assert nums == sorted(nums), "studies must stay in study-number order"


def test_wcs_studies_are_searchable_on_screen():
    """The corpus obeys the same search grammar as the other four: token-AND over every field,
    empty string when nothing matches, everything when the query is empty."""
    from ui.ui_reference import render_wcs_studies
    from ui.ui_reference_data import WCS_STUDIES
    html_all = render_wcs_studies(WCS_STUDIES, "")
    for s in WCS_STUDIES:
        assert s["study"] in html_all
    assert render_wcs_studies(WCS_STUDIES, "margin of safety") != ""
    assert render_wcs_studies(WCS_STUDIES, "zzz-no-such-study") == ""


def test_wcs_studies_ride_into_the_markdown_download():
    """One generator, one source: the download emits the SAME list the screen renders — the
    export/search asymmetry that bit the frameworks (2026-08-28) cannot recur here."""
    from ui.ui_reference import build_reference_markdown
    from ui.ui_reference_data import WCS_STUDIES
    md = build_reference_markdown({"T": "d"}, {"C": [("l", "e" * 45)]}, {}, studies=WCS_STUDIES)
    assert "## Wealth Creation Studies" in md
    for s in WCS_STUDIES:
        assert s["study"] in md and s["theme"] in md


def test_app_renders_the_wcs_studies_corpus():
    """Updated 2026-08-29 for the two-mode layout: the corpus renders through _show_wcs."""
    import io as _io, os
    src = _io.open(os.path.join(os.path.dirname(__file__), "..", "app.py"), encoding="utf-8").read()
    assert "render_wcs_studies(WCS_STUDIES, q)" in src, (
        "app.py no longer renders the Wealth Creation Studies on screen"
    )
    assert "studies=WCS_STUDIES" in src, "the download no longer carries the studies corpus"


# ── Two-mode Reference layout (2026-08-29) ───────────────────────────────────────────────────
def test_reference_two_mode_layout():
    """THE TWO-MODE CONTRACT. BROWSE (empty query) = five inner tabs, one corpus each — the tab
    was five stacked corpora (~70KB) and scrolling was the app's worst UX. SEARCH (any query) =
    the tab bar disappears and results come UNIFIED from all five corpora — one box searches
    everything, so a user never has to know which section holds the answer (the export/search
    asymmetry class, frameworks 2026-08-28, must not recur mode-wise). Every corpus must be wired
    into BOTH branches: dropping one from either mode turns this red."""
    import io as _io, os
    src = _io.open(os.path.join(os.path.dirname(__file__), "..", "app.py"), encoding="utf-8").read()
    helpers = ("_show_concepts", "_show_glossary", "_show_flags", "_show_frameworks", "_show_wcs")

    # SEARCH MODE: between the query conditional and the browse branch, all five filtered by _ref_q.
    search_branch = src[src.index("if _ref_q.strip():"):src.index("_ref_tabs = st.tabs")]
    for h in helpers:
        assert f"{h}(_ref_q)" in search_branch, f"search mode lost {h} — unified search broken"

    # BROWSE MODE: five inner tabs with the exact corpus labels, all five rendered unfiltered.
    browse = src[src.index("_ref_tabs = st.tabs"):]
    for label in ("🏷️ Labels & Verdicts", "📖 Glossary", "🚩 Red Flags",
                  "🏛️ Frameworks", "📚 WCS Studies"):
        assert label in browse, f"browse mode lost its {label} tab"
    for h in helpers:
        assert f'{h}("")' in browse, f"browse mode lost {h} — a corpus vanished from its tab"

    # The inner-tab variable must NOT be _mp_tabs (would confuse the Market Pulse tab extractor).
    assert "_ref_tabs" in src and "_mp_tabs = st.tabs" not in browse
