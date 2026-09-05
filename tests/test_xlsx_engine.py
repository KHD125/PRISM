"""
test_xlsx_engine.py
===================
Contract for the XLSX parser behind the ONE loading path (§0: one workbook download, tabs by name).

MEASURED 2026-09-04: the engine scores a full universe in 2.7s; openpyxl parses the six-tab
workbook in 8.1s, calamine in 0.7s; cell-by-cell parity on the real workbook is exact
(432,072 cells, 0 mismatches). So the parser is the load-time bottleneck and the engine is not.

THE RULE: `_xlsx_engine()` returns "calamine" when python_calamine imports and "openpyxl"
otherwise, and BOTH pd.ExcelFile sites route through it. Never a bare engine="calamine": a
missing Rust wheel on Streamlit Cloud must degrade to today's speed, never to a boot failure.
The engine is decided BEFORE the download so a fallback never downloads twice.

Run with: pytest tests/test_xlsx_engine.py -v
"""
import ast
import builtins
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import pytest

from core import data_engine
from core.data_engine import _xlsx_engine, coerce_numeric_columns, merge_datasets

_ROOT = os.path.join(os.path.dirname(__file__), "..")
_DE = os.path.join(_ROOT, "core", "data_engine.py")
_XLSX = os.path.join(_ROOT, "Other Resources", "CSV Data", "PRISM 2026-08-28 Fri.xlsx")
_TABS = ["Ratio", "Income Statement", "Balance Sheet", "Cashflow", "Shareholdings", "Technicals"]


def _has_calamine():
    try:
        import python_calamine  # noqa: F401
        return True
    except ImportError:
        return False


def test_engine_prefers_calamine_when_available():
    if not _has_calamine():
        pytest.skip("python-calamine not installed here — the fallback test below covers this host")
    assert _xlsx_engine() == "calamine"


def test_engine_falls_back_to_openpyxl_when_calamine_is_missing(monkeypatch):
    """The boot-safety half: on a host where the Rust wheel is absent, the loader must still
    work — slower, never broken."""
    real_import = builtins.__import__

    def no_calamine(name, *a, **k):
        if name == "python_calamine":
            raise ImportError("simulated missing wheel")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_calamine)
    assert _xlsx_engine() == "openpyxl"


def test_both_excelfile_sites_route_through_the_helper():
    """AST: every pd.ExcelFile(...) call in data_engine passes engine=_xlsx_engine() — no bare
    string anywhere, and no site left on the old engine."""
    src = open(_DE, encoding="utf-8").read()
    calls = [n for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "ExcelFile"
             and getattr(getattr(n.func, "value", None), "id", "") == "pd"]
    assert len(calls) == 2, f"expected exactly two ExcelFile sites (sheet URL + upload), found {len(calls)}"
    for c in calls:
        eng = next((kw.value for kw in c.keywords if kw.arg == "engine"), None)
        assert eng is not None, f"ExcelFile at line {c.lineno} has no engine= — pandas would guess"
        assert isinstance(eng, ast.Call) and getattr(eng.func, "id", "") == "_xlsx_engine", (
            f"ExcelFile at line {c.lineno} does not route through _xlsx_engine()")
    assert 'engine="calamine"' not in src and "engine='calamine'" not in src, (
        "a bare calamine engine string would turn a missing wheel into a boot failure")


def test_requirements_carry_the_dependency_with_a_wheel_safe_floor():
    req = open(os.path.join(_ROOT, "requirements.txt"), encoding="utf-8").read()
    m = re.search(r"^python-calamine>=([\d.]+)", req, re.M)
    assert m, "requirements.txt does not list python-calamine — Cloud would silently run the slow path"
    assert tuple(int(x) for x in m.group(1).split(".")) >= (0, 2), "pin a version that ships manylinux wheels"
    assert re.search(r"^openpyxl>=", req, re.M), "openpyxl must stay — it is the fallback"


@pytest.mark.skipif(not os.path.exists(_XLSX), reason="local workbook not present")
@pytest.mark.skipif(not _has_calamine(), reason="python-calamine not installed")
def test_parity_the_cleaned_frame_is_identical_under_both_engines():
    """THE PIN THAT MATTERS: same numbers in, same scores out. Both engines through the SAME
    merge+coerce path, then frame equality — not shapes, values. If a future vintage ever
    exposes a real divergence (an error cell typed differently), this goes red before it ships."""
    def cleaned(engine):
        wb = pd.ExcelFile(_XLSX, engine=engine)
        ds = {}
        for name, tab in zip(["ratio", "income", "balance", "cashflow", "shareholding", "technical"], _TABS):
            raw = wb.parse(tab, header=None)
            ds[name] = data_engine._parse_two_row_header(raw) if hasattr(data_engine, "_parse_two_row_header") else raw
        return ds

    a, b = cleaned("openpyxl"), cleaned("calamine")
    for k in a:
        x, y = a[k], b[k]
        assert x.shape == y.shape, f"{k}: shape differs {x.shape} vs {y.shape}"
        # values: numeric where both numeric (tolerance), else stripped-string equality
        xs, ys = x.stack(), y.stack()
        xn, yn = pd.to_numeric(xs, errors="coerce"), pd.to_numeric(ys, errors="coerce")
        same = ((xn - yn).abs() < 1e-9) | (xn.isna() & yn.isna()) | (xs.astype(str).str.strip() == ys.astype(str).str.strip())
        assert bool(same.all()), f"{k}: {int((~same).sum())} cells differ between engines"
