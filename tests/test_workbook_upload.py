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


# ── LOCAL WORKBOOK (2026-09-10) ──────────────────────────────────────────────────────────────
# The data folder now holds ONE workbook ("PRISM 2026-09-09 Wed.xlsx") where it used to hold six
# dated CSVs, and `load_all_csvs("local")` still looked only for `PRISM <date> - <Tab>.csv`. The
# whole suite went red — 184 errors, all FileNotFoundError — because every test that loads local
# data died. The deployed app was unaffected (it runs in sheet mode), which is exactly why this
# needs a contract test: the breakage is invisible from production.
def _write_mini_workbook(path):
    from config import SHEET_TAB_NAMES
    tabs = {"ratio": {"companyId": [1, 2], "Name": ["A Ltd", "B Ltd"], "ROCE Median 10 Years": [11.5, 9.0]},
            "income": {"companyId": [1, 2], "Name": ["A Ltd", "B Ltd"], "PAT": [10.0, 20.0]},
            "balance": {"companyId": [1, 2], "Name": ["A Ltd", "B Ltd"], "Debt": [1.0, 2.0]},
            "cashflow": {"companyId": [1, 2], "Name": ["A Ltd", "B Ltd"], "Operating Cash Flow": [5.0, 6.0]},
            "shareholding": {"companyId": [1, 2], "Name": ["A Ltd", "B Ltd"], "Promoter Holdings": [60.0, 40.0]},
            "technical": {"companyId": [1, 2], "Name": ["A Ltd", "B Ltd"], "Close Price": [100.0, 200.0]}}
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for key, cols in tabs.items():
            pd.DataFrame(cols).to_excel(xw, sheet_name=SHEET_TAB_NAMES[key], index=False)


def test_local_falls_back_to_the_newest_prism_workbook(tmp_path, monkeypatch):
    """When the dated CSVs are absent, `local` must resolve the newest PRISM workbook and parse it
    through the SAME by-name path the upload and sheet modes use — not a second parser that could
    drift from the §0 contract."""
    import contextlib

    import core.data_engine as de
    old = tmp_path / "PRISM 2026-09-01 Mon.xlsx"
    new = tmp_path / "PRISM 2026-09-09 Wed.xlsx"
    _write_mini_workbook(old)
    time.sleep(0.05)
    _write_mini_workbook(new)
    monkeypatch.setattr(de, "CSV_FILES", {k: str(tmp_path / f"Prism - {v}.csv")
                                          for k, v in {"ratio": "Ratio", "income": "Income Statement",
                                                       "balance": "Balance Sheet", "cashflow": "Cashflow",
                                                       "shareholding": "Shareholdings",
                                                       "technical": "Technicals"}.items()})
    assert de._resolve_local_workbook() == str(new), "must pick the NEWEST workbook"
    with contextlib.redirect_stdout(io.StringIO()):
        ds = de.load_all_csvs("local")
    assert sorted(ds) == ["balance", "cashflow", "income", "ratio", "shareholding", "technical"]
    assert float(ds["ratio"]["roce_med_10y"].iloc[0]) == 11.5, "the workbook was not really parsed"


def test_local_takes_the_newest_vintage_in_either_format(tmp_path, monkeypatch):
    """NEWEST WINS, ACROSS FORMATS — not a format preference. config.py already rules that "if
    several vintages coexist, take the newest by modification time"; a "CSVs always win" rule
    would be a second, conflicting one and would serve STALE data the day an old CSV set
    outranked a fresh workbook. Proven in BOTH directions, because a rule that only holds one
    way round is not a rule."""
    import contextlib

    import core.data_engine as de
    names = {"ratio": "Ratio", "income": "Income Statement", "balance": "Balance Sheet",
             "cashflow": "Cashflow", "shareholding": "Shareholdings", "technical": "Technicals"}
    rows = {"ratio": "companyId,Name,ROCE Median 10 Years\n1,A Ltd,77.7\n",
            "income": "companyId,Name,PAT\n1,A Ltd,10\n",
            "balance": "companyId,Name,Debt\n1,A Ltd,1\n",
            "cashflow": "companyId,Name,Operating Cash Flow\n1,A Ltd,5\n",
            "shareholding": "companyId,Name,Promoter Holdings\n1,A Ltd,60\n",
            "technical": "companyId,Name,Close Price\n1,A Ltd,100\n"}
    monkeypatch.setattr(de, "CSV_FILES", {k: str(tmp_path / f"Prism - {v}.csv") for k, v in names.items()})

    def load():
        with contextlib.redirect_stdout(io.StringIO()):
            return de.load_all_csvs("local")

    # CSVs written LAST -> the CSVs are the newest vintage -> 77.7
    _write_mini_workbook(tmp_path / "PRISM 2026-09-09 Wed.xlsx")
    time.sleep(0.05)
    for k, v in names.items():
        (tmp_path / f"Prism - {v}.csv").write_text(rows[k], encoding="utf-8")
    assert float(load()["ratio"]["roce_med_10y"].iloc[0]) == 77.7, "stale workbook beat fresher CSVs"

    # now the WORKBOOK is rewritten last -> it is the newest vintage -> 11.5
    time.sleep(0.05)
    _write_mini_workbook(tmp_path / "PRISM 2026-09-09 Wed.xlsx")
    assert float(load()["ratio"]["roce_med_10y"].iloc[0]) == 11.5, "stale CSVs beat a fresher workbook"


def test_local_with_neither_csvs_nor_workbook_fails_loud(tmp_path, monkeypatch):
    """An empty folder must raise the original FileNotFoundError, not a confusing workbook error."""
    import contextlib

    import core.data_engine as de
    monkeypatch.setattr(de, "CSV_FILES", {k: str(tmp_path / f"Prism - {k}.csv")
                                          for k in ["ratio", "income", "balance", "cashflow",
                                                    "shareholding", "technical"]})
    assert de._resolve_local_workbook() is None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            de.load_all_csvs("local")
        raise AssertionError("an empty data folder loaded without error")
    except FileNotFoundError:
        pass


def test_the_local_workbook_resolver_ignores_unrelated_xlsx(tmp_path, monkeypatch):
    """Only 'PRISM*.xlsx' qualifies — the folder also holds exports and scratch files, and picking
    one of those would parse a stranger's spreadsheet as the universe."""
    import core.data_engine as de
    (tmp_path / "Watchlist export.xlsx").write_bytes(b"not a workbook")
    (tmp_path / "prism_scored_2026-08-26.csv").write_text("x", encoding="utf-8")
    monkeypatch.setattr(de, "CSV_FILES", {"ratio": str(tmp_path / "Prism - Ratio.csv")})
    assert de._resolve_local_workbook() is None, "a non-PRISM xlsx was accepted as the universe"
    _write_mini_workbook(tmp_path / "PRISM 2026-09-09 Wed.xlsx")
    assert de._resolve_local_workbook().endswith("PRISM 2026-09-09 Wed.xlsx")


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
