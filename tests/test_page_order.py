"""Contract: the main page reads identity -> control -> context.

The compact PRISM brand strip must render at the TOP — above the mandate selector — and the
old tall gradient hero call (the bare ``render_hero_banner()`` that used to sit below the
controls) must be gone. This is a static source-position check (no data dir / no Streamlit
runtime needed), pinning the 2026-06-17 top-of-page IA reorder so it can't silently drift back.
"""
from pathlib import Path

_APP = Path(__file__).resolve().parent.parent / "app.py"


def test_banner_renders_above_mandate_selector():
    src = _APP.read_text(encoding="utf-8")
    banner = src.find("render_hero_banner(compact=True)")
    # Stable anchor = the mandate-selector section header (the old `if "sel_mandate" not in
    # st.session_state` anchor was removed when sel_mandate became derived from the active combo).
    selector = src.find("# ── Mandate Selector — button row")
    assert banner != -1, "compact PRISM brand strip call is missing from app.py"
    assert selector != -1, "mandate selector anchor not found in app.py"
    assert banner < selector, "compact banner must render above the mandate selector"


def test_old_big_hero_call_is_removed():
    src = _APP.read_text(encoding="utf-8")
    assert "\nrender_hero_banner()\n" not in src, (
        "the old tall hero call render_hero_banner() must be removed — the page top uses "
        "render_hero_banner(compact=True) only"
    )
