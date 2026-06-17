"""Contract: the Moat-Growth Matrix EXCLUDES data-missing stocks instead of planting them at 0.

render_moat_growth_matrix once applied `.fillna(0)` to both axes BEFORE the `notna()` filter — which
defeated the filter, so a stock with no ROCE/growth (after the real-value fallback) landed at
coordinate 0 (the origin → the 💀 Wealth Destroyer corner). That sentinel injection violated CLAUDE.md
§5 semantic-truth ("never inject sentinel 0 for missing data; propagate NaN") and contradicted the
function's own docstring. On the live 2,107-stock universe it mis-planted 154 stocks (17 at exact (0,0),
all engine-labelled Wealth Destroyer). The data-prep is now the pure helper `_moat_growth_plot_frame`;
these tests pin the exclusion behaviour so the sentinel can never return.
"""
import numpy as np
import pandas as pd

from ui.ui_tearsheet import _moat_growth_plot_frame

_COLS = ["name", "roce_med_5y", "roce", "pat_gr_5y", "pat_gr_3y", "moat_growth_quad"]


def _frame(rows):
    return pd.DataFrame(rows, columns=_COLS)


def test_present_stock_kept_with_real_coords():
    """A fully-populated stock survives, carrying its real ROCE / growth as the plot coordinates."""
    out = _moat_growth_plot_frame(_frame([["A", 25.0, 20.0, 30.0, 10.0, "⭐ Wealth Creator"]]))
    assert list(out["name"]) == ["A"]
    assert out.iloc[0]["Moat_Y"] == 25.0 and out.iloc[0]["Growth_X"] == 30.0


def test_real_value_fallback_used_not_sentinel():
    """5-yr source NaN → falls back to the real current/3-yr value, NOT to a sentinel 0."""
    out = _moat_growth_plot_frame(_frame([["B", np.nan, 18.0, np.nan, 12.0, ""]]))
    assert len(out) == 1
    assert out.iloc[0]["Moat_Y"] == 18.0 and out.iloc[0]["Growth_X"] == 12.0


def test_missing_axis_excluded_not_planted_at_zero():
    """The core regression guard: a stock missing an entire axis (both sources NaN) is EXCLUDED —
    never coerced to 0 and shown in the Wealth-Destroyer corner."""
    out = _moat_growth_plot_frame(_frame([
        ["OK",       15.0,   15.0,   15.0,   15.0,   ""],
        ["NoMoat",   np.nan, np.nan, 20.0,   20.0,   ""],   # both ROCE sources NaN
        ["NoGrowth", 20.0,   20.0,   np.nan, np.nan, ""],   # both PAT-CAGR sources NaN
    ]))
    assert set(out["name"]) == {"OK"}, "missing-axis stocks must be EXCLUDED, never planted at 0"
    assert (out["Moat_Y"] != 0).all() and (out["Growth_X"] != 0).all()
