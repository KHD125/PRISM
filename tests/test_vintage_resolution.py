"""
test_vintage_resolution.py
==========================
Contract: "NEWEST VINTAGE WINS" means the newest DATA, and the data's date is IN THE FILENAME.

THE DEFECT (2026-09-19). All three local resolvers — config._get_actual_path (dated CSVs),
data_engine._resolve_local_workbook (PRISM*.xlsx) and the cross-format choice in load_all_csvs —
ranked candidates by MODIFICATION TIME. The ingestion pipeline stamps every drop with its data
session ("PRISM 2026-09-18 Fri.xlsx", "PRISM 2026-08-28 Fri - Ratio.csv"), so the vintage is a
fact of the NAME; mtime is a fact of when the file was last COPIED. The user copied
"PRISM 2026-06-29.xlsx" into the drops folder at 16:46 and, from that moment, every local run —
tests, /verify, /census, snapshots, the dev server — silently scored JUNE's 2,107-stock universe
in place of September's 2,717, with the loader cheerfully printing "newest local vintage is a
workbook: PRISM 2026-06-29.xlsx". Nothing failed. The previous tests had PINNED the mtime rule
("take the newest by modification time"), which is how a design flaw survives: it was a contract.

THE RULE NOW. Sort by the date parsed from the basename (YYYY-MM-DD, validated), then mtime, then
name. mtime decides only between files of the SAME vintage date, or when a name carries no date at
all (the legacy "Prism - Ratio.csv" shape) — the one case where the clock is the only evidence.
Cross-format: when both the CSV set and the workbook are dated, the dates decide; otherwise mtime.
The exact legacy filename still wins outright in _get_actual_path (unchanged).

Run with: pytest tests/test_vintage_resolution.py -v
"""

import contextlib
import datetime
import glob
import io
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import pytest

from test_workbook_upload import _write_mini_workbook

_TABS = {"ratio": "Ratio", "income": "Income Statement", "balance": "Balance Sheet",
         "cashflow": "Cashflow", "shareholding": "Shareholdings", "technical": "Technicals"}
_ROWS = {"ratio": "companyId,Name,ROCE Median 10 Years\n1,A Ltd,77.7\n",
         "income": "companyId,Name,PAT\n1,A Ltd,10\n",
         "balance": "companyId,Name,Debt\n1,A Ltd,1\n",
         "cashflow": "companyId,Name,Operating Cash Flow\n1,A Ltd,5\n",
         "shareholding": "companyId,Name,Promoter Holdings\n1,A Ltd,60\n",
         "technical": "companyId,Name,Close Price\n1,A Ltd,100\n"}
PAST = time.time() - 30 * 86400          # "copied a month ago"


def _touch(path, when):
    os.utime(path, (when, when))


# ── 1. the date is parsed from the NAME, and only a real date counts ────────────────────────
@pytest.mark.parametrize("name,expect", [
    ("PRISM 2026-09-18 Fri.xlsx",            datetime.date(2026, 9, 18)),
    ("PRISM 2026-06-29.xlsx",                datetime.date(2026, 6, 29)),    # no weekday suffix
    ("PRISM 2026-08-28 Fri - Ratio.csv",     datetime.date(2026, 8, 28)),
    ("prism 2026-01-05 mon - technicals.csv", datetime.date(2026, 1, 5)),    # case-insensitive
    ("Prism - Ratio.csv",                    None),                          # legacy: no vintage
    ("PRISM 2026-13-45 Xyz.xlsx",            None),                          # not a date: never crash
    ("PRISM.xlsx",                           None),
])
def test_vintage_date_is_read_from_the_basename(name, expect):
    from config import vintage_date
    assert vintage_date(os.path.join("any", "folder", name)) == expect


# ── 2. THE REPRODUCTION: an older vintage copied in LATER must never win ────────────────────
def test_an_older_workbook_copied_in_later_never_beats_a_newer_vintage(tmp_path, monkeypatch):
    """Exactly what happened on 2026-09-19: PRISM 2026-06-29.xlsx arrived in the folder after
    PRISM 2026-09-18 Fri.xlsx and, by mtime, became 'newest'."""
    import core.data_engine as de
    sept = tmp_path / "PRISM 2026-09-18 Fri.xlsx"
    june = tmp_path / "PRISM 2026-06-29.xlsx"
    _write_mini_workbook(sept)
    _write_mini_workbook(june)
    _touch(sept, PAST)                    # September's data, copied a month ago
    _touch(june, time.time())             # June's data, copied just now
    monkeypatch.setattr(de, "CSV_FILES", {k: str(tmp_path / f"Prism - {v}.csv") for k, v in _TABS.items()})
    assert de._resolve_local_workbook() == str(sept), (
        "the resolver picked the file that was COPIED last instead of the DATA that is newest"
    )


def test_an_older_csv_drop_copied_in_later_never_beats_a_newer_vintage(tmp_path):
    """The CSV-side twin of the same defect, through config._get_actual_path."""
    from config import _get_actual_path
    d = tmp_path / "CSV Data"
    d.mkdir()
    new = d / "PRISM 2026-09-18 Fri - Ratio.csv"
    old = d / "PRISM 2026-07-31 Thu - Ratio.csv"
    new.write_text("new")
    old.write_text("old")
    _touch(new, PAST)
    _touch(old, time.time())
    got = _get_actual_path(str(tmp_path), "CSV Data", "Prism - Ratio.csv")
    assert os.path.basename(got) == "PRISM 2026-09-18 Fri - Ratio.csv"


# ── 3. cross-format: dates decide when both sides carry one — in BOTH directions ────────────
def _csv_set(folder, stamp, roce):
    paths = {}
    for k, v in _TABS.items():
        p = folder / f"PRISM {stamp} - {v}.csv"
        p.write_text(_ROWS[k].replace("77.7", roce), encoding="utf-8")
        paths[k] = str(p)
    return paths


def _load(de):
    with contextlib.redirect_stdout(io.StringIO()):
        return de.load_all_csvs("local")


def test_dated_csvs_copied_later_do_not_beat_a_newer_dated_workbook(tmp_path, monkeypatch):
    import core.data_engine as de
    csvs = _csv_set(tmp_path, "2026-08-28 Fri", "77.7")
    wb = tmp_path / "PRISM 2026-09-18 Fri.xlsx"
    _write_mini_workbook(wb)                                   # roce_med_10y = 11.5
    _touch(wb, PAST)
    for p in csvs.values():
        _touch(p, time.time())                                 # the OLDER data, copied just now
    monkeypatch.setattr(de, "CSV_FILES", csvs)
    assert float(_load(de)["ratio"]["roce_med_10y"].iloc[0]) == 11.5, "stale CSVs beat a newer workbook by mtime"


def test_a_dated_workbook_copied_later_does_not_beat_newer_dated_csvs(tmp_path, monkeypatch):
    import core.data_engine as de
    csvs = _csv_set(tmp_path, "2026-09-18 Fri", "77.7")
    wb = tmp_path / "PRISM 2026-06-29.xlsx"
    _write_mini_workbook(wb)
    for p in csvs.values():
        _touch(p, PAST)
    _touch(wb, time.time())                                    # June's workbook, copied just now
    monkeypatch.setattr(de, "CSV_FILES", csvs)
    assert float(_load(de)["ratio"]["roce_med_10y"].iloc[0]) == 77.7, "a stale workbook beat newer CSVs by mtime"


# ── 4. mtime still matters — but only where the date cannot decide ──────────────────────────
def test_same_vintage_date_falls_back_to_mtime(tmp_path, monkeypatch):
    """Two drops of the SAME session (a re-fetch): the later copy is the one to trust."""
    import core.data_engine as de
    a = tmp_path / "PRISM 2026-09-18 Fri.xlsx"
    b = tmp_path / "PRISM 2026-09-18 Fri (1).xlsx"
    _write_mini_workbook(a)
    _write_mini_workbook(b)
    _touch(a, PAST)
    _touch(b, time.time())
    monkeypatch.setattr(de, "CSV_FILES", {k: str(tmp_path / f"Prism - {v}.csv") for k, v in _TABS.items()})
    assert de._resolve_local_workbook() == str(b)
    _touch(a, time.time() + 5)
    assert de._resolve_local_workbook() == str(a)


def test_legacy_undated_csvs_still_compare_by_mtime_against_a_workbook(tmp_path, monkeypatch):
    """The one place the clock is the only evidence: a legacy 'Prism - Ratio.csv' set carries no
    vintage, so against a dated workbook the newer FILE wins — both ways, as before."""
    import core.data_engine as de
    csvs = {k: str(tmp_path / f"Prism - {v}.csv") for k, v in _TABS.items()}
    for k, p in csvs.items():
        open(p, "w", encoding="utf-8").write(_ROWS[k])
    wb = tmp_path / "PRISM 2026-09-09 Wed.xlsx"
    _write_mini_workbook(wb)
    monkeypatch.setattr(de, "CSV_FILES", csvs)
    _touch(wb, PAST)
    for p in csvs.values():
        _touch(p, time.time())
    assert float(_load(de)["ratio"]["roce_med_10y"].iloc[0]) == 77.7
    _touch(wb, time.time() + 5)
    assert float(_load(de)["ratio"]["roce_med_10y"].iloc[0]) == 11.5


# ── 5. live property: whatever is in the real folder, the resolver picks the max DATE ───────
def test_the_live_folder_resolves_to_its_newest_dated_workbook():
    from config import CSV_FILES, vintage_date
    import core.data_engine as de
    folder = os.path.dirname(CSV_FILES["ratio"])
    books = [p for p in glob.glob(os.path.join(folder, "*.xlsx"))
             if os.path.basename(p).lower().startswith("prism")]
    if len(books) < 2:
        pytest.skip("fewer than two PRISM workbooks in the live folder — nothing to disambiguate")
    dated = [p for p in books if vintage_date(p) is not None]
    want = max(vintage_date(p) for p in dated)
    got = de._resolve_local_workbook()
    assert vintage_date(got) == want, (
        f"resolved {os.path.basename(got)} but the newest vintage on disk is {want}"
    )
