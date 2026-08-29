"""Contract: the sheet-link box survives a Community Cloud restart, and Clear Cache & Reload
visibly proves itself.

THE BUG THIS PINS AGAINST (reproduced live on the deployed app, 2026-08-29): Community Cloud
sleeps/restarts the server, wiping session_state while the BROWSER keeps showing the old page.
The sheet link then sits visibly in the box but the new server session never received it — the
first click (Clear Cache & Reload included) dropped to the welcome screen, and Enter on the
UNCHANGED text re-submitted nothing (the frontend considers it already committed). The fix is
?sheet=<id> query-param persistence: the URL itself carries whatever the user last typed, a fresh
session seeds the box from it (seed-before-instantiate — the widget-state law), and data
auto-reloads with zero retyping. Nothing is hardcoded.

These are static source pins (the same style as test_cache_contract.py): Streamlit's runtime
cannot be unit-tested here, but the source structure that drives it can.
"""
import re
from pathlib import Path

_APP = Path(__file__).resolve().parent.parent / "app.py"


def _src() -> str:
    return _APP.read_text(encoding="utf-8")


def test_sheet_box_is_keyed_and_seeded_before_instantiation():
    """The box must use a session key (no value= param) and the key must be seeded BEFORE the
    widget instantiates — the only pattern that keeps server and frontend in agreement after a
    restart. A value= default cannot recover: the frontend ghost-displays old text the server
    never received."""
    src = _src()
    m = re.search(r'st\.text_input\("Google Sheets URL or ID"([^)]*)\)', src)
    assert m, "the sheet-link text_input vanished"
    assert 'key="data_sheet_box"' in m.group(1), "sheet box lost its session key"
    assert "value=" not in m.group(1), (
        "value= and key= must not mix — seed the key instead (Streamlit ignores value= when the "
        "key exists and warns; the seed IS the default)")
    seed = src.index('st.session_state.data_sheet_box = _default_sheet')
    widget = src.index('key="data_sheet_box"')
    assert seed < widget, "the key must be seeded BEFORE the widget instantiates"


def test_sheet_box_key_is_outside_the_sb_namespace():
    """clear_all_filters() deletes EVERY sb_* key. If the sheet box ever moves into that
    namespace, clicking 'Clear all filters' would unload the user's data source."""
    m = re.search(r'st\.text_input\("Google Sheets URL or ID"[^)]*key="([^"]+)"', _src())
    assert m and not m.group(1).startswith("sb_"), (
        "sheet box key must NOT be sb_-prefixed — clear_all_filters wipes sb_*")


def test_query_param_round_trip_exists():
    """Read on boot (seeds the box after a restart) AND write after a successful entry (so the
    URL carries the id into the next restart). Both directions, extracted id only — never the
    raw pasted URL (URL-in-URL breaks bookmarking)."""
    src = _src()
    assert 'st.query_params.get("sheet", "")' in src, "boot no longer reads ?sheet= — restarts lose the link again"
    assert 'st.query_params["sheet"] = _sid' in src, "successful entry no longer writes ?sheet="
    assert "_sid = extract_spreadsheet_id(sheet_id)" in src, (
        "the param must store the EXTRACTED id, not the raw pasted URL")
    assert "from core.data_engine import extract_spreadsheet_id" in src


def test_env_var_still_outranks_the_query_param():
    """Dev autoload contract: PRISM_SHEET_ID/STOCKSCAN_SHEET_ID pre-fill wins over ?sheet= so the
    Playwright/visual-check loop keeps booting against the intended dev sheet."""
    src = _src()
    env = src.index('os.environ.get("PRISM_SHEET_ID")')
    qp = src.index('st.query_params.get("sheet", "")')
    assert env < qp and "if not _default_sheet:" in src[env:qp + 200], (
        "the query param must only fill in when the env default is empty")


def test_clear_cache_reload_confirms_itself():
    """User-reported 2026-08-29: 'I don't know if it clears cache or not.' The button sets a flag
    before st.rerun(); after the fresh load, a toast confirms with the measured reload time. The
    flag is POPPED so the toast fires exactly once."""
    src = _src()
    btn = src.index('st.button("🔄 Clear Cache & Reload"')
    rerun = src.index("st.rerun()", btn)
    assert 'st.session_state["_cache_cleared"] = True' in src[btn:rerun], (
        "the button no longer flags the reload for confirmation")
    assert 'st.session_state.pop("_cache_cleared", False)' in src, "the toast no longer pops the flag"
    assert "st.toast" in src.split('pop("_cache_cleared", False)')[1][:200], (
        "the popped flag must drive a st.toast confirmation")
