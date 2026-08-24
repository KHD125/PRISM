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
