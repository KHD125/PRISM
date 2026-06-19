"""Contract for the sidebar full-universe CSV export (ui/ui_export.py). Tests the PURE core
(_to_csv_bytes) — Excel-safe BOM, every column preserved, NaN never leaks as a literal. Pins the
2026-06-19 export feature so the full-dump download can't silently regress (wrong encoding / dropped
columns / 'nan' text)."""
import io

import numpy as np
import pandas as pd

from ui.ui_export import _to_csv_bytes


def test_csv_round_trips_with_bom_and_emoji():
    df = pd.DataFrame({"name": ["HDFC", "Infosys"],
                       "tier_label": ["🏆 Crown Jewels", "❌ Not Ready"],
                       "composite_score": [88.5, 12.0]})
    raw = _to_csv_bytes(df)
    assert raw[:3] == b"\xef\xbb\xbf"                            # UTF-8 BOM (Excel-safe)
    back = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")    # round-trips, BOM stripped
    assert list(back.columns) == ["name", "tier_label", "composite_score"]
    assert len(back) == 2 and back["tier_label"].iloc[0] == "🏆 Crown Jewels"


def test_all_columns_preserved():
    df = pd.DataFrame({f"c{i}": [i] for i in range(60)})
    back = pd.read_csv(io.BytesIO(_to_csv_bytes(df)), encoding="utf-8-sig")
    assert list(back.columns) == [f"c{i}" for i in range(60)]    # nothing dropped


def test_missing_values_no_literal_nan_leak():
    raw = _to_csv_bytes(pd.DataFrame({"a": [1.0, np.nan], "b": ["x", None]}))
    text = raw.decode("utf-8-sig")
    assert "nan" not in text.lower() and "None" not in text      # missing → empty field, not literal
    back = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    assert back["a"].isna().iloc[1]                              # NaN round-trips as missing
