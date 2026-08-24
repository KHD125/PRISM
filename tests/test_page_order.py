"""Contract: the main page reads identity -> control -> context.

The compact PRISM brand strip must render at the TOP — above the mandate selector — and the
old tall gradient hero call (the bare ``render_hero_banner()`` that used to sit below the
controls) must be gone. This is a static source-position check (no data dir / no Streamlit
runtime needed), pinning the 2026-06-17 top-of-page IA reorder so it can't silently drift back.
"""
from pathlib import Path

_APP = Path(__file__).resolve().parent.parent / "app.py"


def test_banner_renders_above_the_tabs():
    """IA: identity first. The compact brand strip renders before the stats strip and the tabs.
    (The old anchor was the mandate selector — removed 2026-08-24 with the Command Center.)"""
    src = _APP.read_text(encoding="utf-8")
    banner = src.find("render_hero_banner(compact=True)")
    stats  = src.find("render_metric_strip(")
    tabs   = src.find("st.tabs(")
    assert banner != -1, "compact PRISM brand strip call is missing from app.py"
    assert stats != -1 and tabs != -1, "stats strip / tabs anchors not found in app.py"
    assert banner < stats < tabs, "page order must be brand strip -> stats strip -> tabs"


def test_old_big_hero_call_is_removed():
    src = _APP.read_text(encoding="utf-8")
    assert "\nrender_hero_banner()\n" not in src, (
        "the old tall hero call render_hero_banner() must be removed — the page top uses "
        "render_hero_banner(compact=True) only"
    )


def test_discovery_sorts_by_composite_explicitly():
    """LOAD-BEARING SORT (2026-08-24). The pipeline leaves the frame in STALE ROW ORDER:
    run_full_scoring sorts by the PRE-penalty composite and resets the index, then
    apply_forensic_penalty multiplies composite_score and re-derives `rank` WITHOUT re-sorting.
    Measured live: `rank` is NON-monotonic in frame order (though it matches the composite
    ranking exactly). So the Discovery tab MUST sort explicitly — relying on the frame's
    natural order would render cards contradicting their own #rank badge, a silent display bug
    no render test would catch (cards render fine; only the ORDER is wrong).

    This pins the sort itself, and that the removed 'Sort by' control stays removed (its
    Quality/Momentum/PEG options ordered cards by numbers the cards never display; the Deep
    Scanner sorts those AND shows them as columns)."""
    src = _APP.read_text(encoding="utf-8")
    assert 'filt.sort_values("composite_score", ascending=False)' in src, (
        "Discovery must sort explicitly by composite_score descending — the pipeline's row "
        "order is stale pre-penalty order, so the natural order contradicts the #rank badge."
    )
    for zombie in ["_disc_sort", "_sort_map", 'key="disc_sort"']:
        assert zombie not in src, (
            f"{zombie!r} is back — the Discovery 'Sort by' control was removed 2026-08-24; "
            "numeric comparison belongs in the Deep Scanner, which shows the sorted column."
        )


def test_pipeline_row_order_is_stale_by_design():
    """The root fact the contract above defends against, asserted on the ENGINE (not app.py):
    apply_forensic_penalty re-ranks but never re-sorts. If a future change makes the pipeline
    return composite-sorted rows, this test fails LOUDLY — at which point the Discovery sort
    becomes redundant rather than load-bearing, and this pair should be revisited together."""
    import ast

    fe = (_APP.parent / "core" / "forensic_engine.py").read_text(encoding="utf-8")
    tree = ast.parse(fe)
    # apply_forensic_penalty is a thin wrapper; the rank re-derivation lives in its delegate.
    # Find whichever function actually assigns df["rank"] — robust to that indirection.
    ranker = next((n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef)
                   and 'df["rank"]' in (ast.get_source_segment(fe, n) or "")), None)
    assert ranker is not None, (
        'no function in core/forensic_engine.py assigns df["rank"] — step 3 must re-derive rank '
        "from the penalized score (CLAUDE.md §5), else rank goes stale."
    )
    body = ast.get_source_segment(fe, ranker) or ""
    assert "sort_values" not in body, (
        f"{ranker.name} now sorts — the Discovery sort is no longer load-bearing. "
        "Re-read both contracts before changing either."
    )
