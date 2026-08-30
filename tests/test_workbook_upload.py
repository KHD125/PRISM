"""Contract: the two 2026-08-30 data-ingestion conveniences.

1. DATED CSV RESOLUTION — the CSV drops are now named after their data session
   ("PRISM 2026-08-28 Fri - Ratio.csv"): the date/day part changes every refresh while the
   "PRISM" prefix and "- <Tab>.csv" suffix are fixed. config._get_actual_path must resolve
   them (newest vintage when several coexist, exact legacy name still winning outright).

2. SINGLE-WORKBOOK UPLOAD — one .xlsx (e.g. "PRISM 2026-08-28 Fri.xlsx") carrying the six
   §0-contract tabs uploads through the SAME by-name parse path as the Google Sheets
   download, so the two sources can never diverge. Tab names are the locked SHEET_TAB_NAMES.
"""
import io
import os
import time
from pathlib import Path

import pandas as pd

_APP = Path(__file__).resolve().parent.parent / "app.py"


# ── 1. dated CSV resolution ──────────────────────────────────────────────────────────────────
def test_dated_export_names_resolve(tmp_path):
    from config import _get_actual_path
    d = tmp_path / "CSV Data"
    d.mkdir()
    (d / "PRISM 2026-08-28 Fri - Ratio.csv").write_text("x")
    got = _get_actual_path(str(tmp_path), "CSV Data", "Prism - Ratio.csv")
    assert os.path.basename(got) == "PRISM 2026-08-28 Fri - Ratio.csv"


def test_exact_legacy_name_still_wins(tmp_path):
    from config import _get_actual_path
    d = tmp_path / "CSV Data"
    d.mkdir()
    (d / "Prism - Ratio.csv").write_text("legacy")
    (d / "PRISM 2026-08-28 Fri - Ratio.csv").write_text("dated")
    got = _get_actual_path(str(tmp_path), "CSV Data", "Prism - Ratio.csv")
    assert os.path.basename(got) == "Prism - Ratio.csv"


def test_newest_vintage_wins_when_several_coexist(tmp_path):
    from config import _get_actual_path
    d = tmp_path / "CSV Data"
    d.mkdir()
    old = d / "PRISM 2026-07-31 Thu - Ratio.csv"
    new = d / "PRISM 2026-08-28 Fri - Ratio.csv"
    old.write_text("old")
    new.write_text("new")
    past = time.time() - 86400
    os.utime(old, (past, past))                      # old vintage: modified yesterday
    got = _get_actual_path(str(tmp_path), "CSV Data", "Prism - Ratio.csv")
    assert os.path.basename(got) == "PRISM 2026-08-28 Fri - Ratio.csv"


def test_wrong_tab_suffix_never_cross_matches(tmp_path):
    """'PRISM ... - Technicals.csv' must NEVER resolve a request for the Ratio file — a
    cross-matched tab would feed the wrong sheet into the wrong column mapping (the §0
    wrong-tab guard would catch it later, but resolution must not create the hazard)."""
    from config import _get_actual_path
    d = tmp_path / "CSV Data"
    d.mkdir()
    (d / "PRISM 2026-08-28 Fri - Technicals.csv").write_text("tech")
    got = _get_actual_path(str(tmp_path), "CSV Data", "Prism - Ratio.csv")
    assert os.path.basename(got) == "Prism - Ratio.csv"   # unresolved → falls through untouched


# ── 2. single-workbook upload ────────────────────────────────────────────────────────────────
def _mini_workbook() -> io.BytesIO:
    """An in-memory XLSX with all six contract tabs, each carrying the identity columns plus
    two real mapped headers (so the wrong-tab guard passes) and one recognizable value."""
    from config import SHEET_TAB_NAMES
    tab_cols = {
        "ratio":        ["ROCE Median 10 Years", "ROCE Median 7 Years"],
        "income":       ["PAT Growth 5 Years", "PAT Growth 10 Years"],
        "balance":      ["Debt", "Debt 1 Year Back"],
        "cashflow":     ["Operating Cash Flow", "Operating Cash Flow 1 Year Back"],
        "shareholding": ["Promoter Holdings", "FII Holdings"],
        "technical":    ["Market Capitalization", "Market Category"],
    }
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for key, tab in SHEET_TAB_NAMES.items():
            frame = pd.DataFrame({
                "companyId": [101, 102], "Name": ["Alpha Ltd", "Beta Ltd"],
                tab_cols[key][0]: [11.5, 22.5], tab_cols[key][1]: [1.0, 2.0],
            })
            frame.to_excel(xw, sheet_name=tab, index=False)
    buf.seek(0)
    return buf


def test_workbook_upload_loads_all_six_tabs_by_name():
    from core.data_engine import load_all_csvs
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        ds = load_all_csvs("upload", uploaded_files={"workbook": _mini_workbook()})
    assert sorted(ds) == ["balance", "cashflow", "income", "ratio", "shareholding", "technical"]
    for name, frame in sorted(ds.items()):
        assert "company_id" in frame.columns and len(frame) == 2, f"{name} tab mis-parsed"
    # a mapped value survives the by-name parse into its snake_case column
    assert float(ds["ratio"]["roce_med_10y"].iloc[0]) == 11.5


def test_workbook_with_missing_tab_fails_loud():
    """A workbook missing a contract tab must raise naming the tab — silently proceeding
    would score the whole universe on a part-empty merge (the flat-scores regression class)."""
    from config import SHEET_TAB_NAMES
    from core.data_engine import load_all_csvs
    import contextlib
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        # a VALID ratio tab (mapped columns present), but every other contract tab absent —
        # so the failure under test is the missing-tab check, not the wrong-tab guard
        pd.DataFrame({"companyId": [1], "Name": ["Solo Ltd"],
                      "ROCE Median 10 Years": [10.0]}
                     ).to_excel(xw, sheet_name=SHEET_TAB_NAMES["ratio"], index=False)
    buf.seek(0)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            load_all_csvs("upload", uploaded_files={"workbook": buf})
        raise AssertionError("a 1-tab workbook loaded without error")
    except Exception as e:
        assert "not found in the uploaded workbook" in str(e)


def test_app_uploader_accepts_xlsx_and_routes_the_workbook():
    src = _APP.read_text(encoding="utf-8")
    assert '"xlsx"' in src.split("file_uploader", 1)[1][:300], "the uploader no longer accepts .xlsx"
    assert '{"workbook": _xlsx_uploads[0]}' in src, "an uploaded workbook is no longer routed by key"
    # CSV fallback intact: the 6-slot matcher must still exist for CSV-only uploads
    assert '"shareholding" in fname' in src, "the 6-CSV matching path was lost"
