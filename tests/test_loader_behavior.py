"""
test_loader_behavior.py
=======================
BEHAVIORAL contract for the locked Tier-1 CSV loaders in core/data_engine.py.

Phase-1 audit finding C4: load_all_csvs / merge_datasets / _apply_column_mapping were only
verified by `inspect.getsource()` substring assertions — which pass even if the behavior breaks
and break on innocent refactors. These tests EXECUTE the functions on synthetic frames (no data
files needed, so they run on a code-only checkout too) and pin the behavior that actually matters.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from core.data_engine import _apply_column_mapping, merge_datasets


# ── _apply_column_mapping ────────────────────────────────────────────────────

def test_wrong_tab_guard_raises_when_no_expected_columns_match():
    """The wrong-tab guard (CLAUDE.md §0: 'never weaken it') must raise when the source returns
    a different tab — i.e. none of the sheet-specific columns are present."""
    df = pd.DataFrame({"companyId": ["NSE:A"], "SomethingElse": [1]})
    col_map = {"ROCE Median 10 Years": "roce_med_10y"}  # none of these in df
    with pytest.raises(ValueError, match="wrong sheet/tab"):
        _apply_column_mapping(df, col_map, "ratio")


def test_promotes_companyid_header_row_when_columns_are_section_labels():
    """Local CSV/export layout: row 0 is an emoji/section label row, the real header (with
    'companyId') is row 1. The loader must auto-detect and promote it."""
    raw = pd.DataFrame(
        [["companyId", "ROCE Median 10 Years"],
         ["NSE:A", 25.0]],
        columns=["Details", "Unnamed: 1"],
    )
    out = _apply_column_mapping(raw, {"ROCE Median 10 Years": "roce_med_10y"}, "ratio")
    assert list(out.columns) == ["company_id", "roce_med_10y"]
    assert out.iloc[0]["company_id"] == "NSE:A"
    assert float(out.iloc[0]["roce_med_10y"]) == 25.0


def test_renames_to_snake_case_and_drops_unmapped_columns():
    df = pd.DataFrame({"companyId": ["NSE:A"], "ROCE Median 10 Years": [25.0], "JunkCol": [9]})
    out = _apply_column_mapping(df, {"ROCE Median 10 Years": "roce_med_10y"}, "ratio")
    assert "company_id" in out.columns and "roce_med_10y" in out.columns
    assert "JunkCol" not in out.columns
    assert "ROCE Median 10 Years" not in out.columns


def test_drops_phantom_nan_and_blank_company_id_rows():
    """A bloated source export (a tab whose used-range declares ~50k rows when only ~2.1k hold
    data) emits thousands of empty rows with a NaN/blank company_id. The left-join in
    merge_datasets matches NaN==NaN, so those phantoms cartesian-explode → millions of rows →
    OOM. openpyxl happens to trim trailing empties; calamine/others do NOT — so the loader must
    drop every row without a usable company_id ITSELF, engine-agnostically. A blank ("" / "  ")
    id is just as unusable as NaN and must go too (na_values runs at read time, upstream of this
    function, so a row can still arrive here with a literal empty-string id)."""
    raw = pd.DataFrame(
        {
            "companyId": ["NSE:A", np.nan, "NSE:B", "", "  "],
            "ROCE Median 10 Years": [25.0, np.nan, 30.0, np.nan, np.nan],
        }
    )
    out = _apply_column_mapping(raw, {"ROCE Median 10 Years": "roce_med_10y"}, "ratio")
    assert list(out["company_id"]) == ["NSE:A", "NSE:B"]   # NaN / "" / "  " phantoms dropped
    assert len(out) == 2


def test_keeps_all_rows_when_company_id_clean():
    """No-op on clean sources: when every company_id is present, no valid row is ever dropped
    (this is why the guard is behavior-preserving on local CSV / upload / openpyxl-trimmed sheets)."""
    raw = pd.DataFrame(
        {
            "companyId": ["NSE:A", "NSE:B", "NSE:C"],
            "ROCE Median 10 Years": [25.0, 30.0, 35.0],
        }
    )
    out = _apply_column_mapping(raw, {"ROCE Median 10 Years": "roce_med_10y"}, "ratio")
    assert len(out) == 3
    assert list(out["company_id"]) == ["NSE:A", "NSE:B", "NSE:C"]


# ── merge_datasets ───────────────────────────────────────────────────────────

def _datasets():
    return {
        "ratio":        pd.DataFrame({"company_id": ["A", "B", "C"], "name": ["a", "b", "c"], "roce": [1, 2, 3]}),
        "income":       pd.DataFrame({"company_id": ["A", "B"], "pat": [10, 20]}),          # C missing
        "balance":      pd.DataFrame({"company_id": ["A", "B", "C"], "debt": [5, 6, 7]}),
        "cashflow":     pd.DataFrame({"company_id": ["A"], "fcf": [100]}),                   # B, C missing
        "shareholding": pd.DataFrame({"company_id": ["A", "B", "C"], "promoter": [50, 60, 70]}),
        "technical":    pd.DataFrame({"company_id": ["A", "B", "C"], "name": ["a", "b", "c"], "vstop_value": [1, 2, 3]}),
    }


def test_merge_is_left_join_on_ratio_and_nan_fills_missing_stocks():
    """Ratio is the authority: every ratio stock survives; stocks absent from another sheet get
    NaN for that sheet's columns, never dropped."""
    master = merge_datasets(_datasets())
    assert len(master) == 3
    assert set(master["company_id"]) == {"A", "B", "C"}
    idx = master.set_index("company_id")
    assert pd.isna(idx.loc["C", "pat"])    # C absent from income → NaN
    assert pd.isna(idx.loc["B", "fcf"])    # B absent from cashflow → NaN
    assert idx.loc["C", "debt"] == 7       # present everywhere → real value


def test_merge_does_not_duplicate_columns_shared_across_sheets():
    """A common column ('name') present in both ratio and technical must not produce a suffixed
    duplicate ('name_technical') — the merge brings only columns not already in master."""
    master = merge_datasets(_datasets())
    assert list(master.columns).count("name") == 1
    assert not any(c.endswith("_technical") or c.endswith("_income") for c in master.columns)
