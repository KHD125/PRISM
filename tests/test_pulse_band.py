"""Contract: the Market Pulse "Pulse band" — pure, vectorized market-state aggregates that degrade
gracefully on missing/NaN data and never leak a literal nan/None into the rendered HTML. Pins the
2026-06-18 Pulse-band stage so the breadth/regime/conviction surfacing can't silently regress.
"""
import inspect

import numpy as np
import pandas as pd

from ui.ui_components import _pulse_stats, _pulse_band_html


def _full_frame():
    """A 6-row frame exercising every Pulse column (emoji-prefixed strings as the engine produces)."""
    return pd.DataFrame({
        "weinstein_stage": ["📈 Stage 2 Advancing", "📈 Stage 2 Advancing", "🔄 Stage 1 Basing",
                            "⚠️ Stage 3 Top", "📉 Stage 4 Declining", "❔ Unknown"],
        "dist_52wh":       [10, 20, 30, 40, 50, np.nan],
        "conviction_tier": [1, 2, 3, 4, 5, 5],
        "sector_capital_phase": ["🔥 Hot Capital (caution)", "❄️ Capital Starved (opportunity)",
                                 "⚖️ Neutral", "⚖️ Neutral", "⚖️ Neutral", "⚖️ Neutral"],
        "composite_score": [80, 60, 40, 30, 20, 10],
        "sector_tailwind": [1, 1, 0, 0, 0, 0],
    })


def test_pulse_stats_full_frame():
    s = _pulse_stats(_full_frame())
    assert s["n"] == 6
    # Stage-N substring match; Unknown is the residual → five buckets sum to n (bar fills 100%)
    assert s["breadth"] == {"Advancing": 2, "Basing": 1, "Topping": 1, "Declining": 1, "Unknown": 1}
    assert sum(s["breadth"].values()) == s["n"]
    assert s["off_high"] == 30.0          # median([10,20,30,40,50]) — NaN skipped
    assert {t: c for t, _, _, c in s["ladder"]} == {1: 1, 2: 1, 3: 1, 4: 1, 5: 2}
    assert s["capital"] == {"hot": 1, "starved": 1}
    assert s["med_composite"] == 35.0     # median([10,20,30,40,60,80])
    assert round(s["tailwind_pct"]) == 33  # 2 of 6


def test_pulse_stats_missing_columns_is_graceful():
    s = _pulse_stats(pd.DataFrame({"name": ["A", "B"]}))   # zero pulse columns — must not raise
    assert s["breadth"] is None and s["ladder"] is None and s["capital"] is None
    assert s["off_high"] is None and s["med_composite"] is None and s["tailwind_pct"] is None
    assert s["regime"] == "SIDEWAYS"                        # df.attrs default
    html = _pulse_band_html(s)
    for bad in ("nan", "None", "NaN"):
        assert bad not in html


def test_pulse_stats_present_but_unclassified_breadth():
    df = _full_frame().copy()
    df["weinstein_stage"] = "❔ Unknown"                    # column present but nothing classified
    assert _pulse_stats(df)["breadth"] is None             # → treated as unavailable, not an all-grey bar


def test_pulse_band_html_no_nan_leak_on_all_nan_numerics():
    df = _full_frame().copy()
    df["dist_52wh"] = np.nan
    df["composite_score"] = np.nan
    df["sector_tailwind"] = np.nan
    html = _pulse_band_html(_pulse_stats(df))
    for bad in ("nan", "None", "NaN"):
        assert bad not in html
    assert "Adv" in html                                   # breadth still rendered (the dash is for numerics)
    assert "—" in html                                     # missing numerics shown as an em-dash


def test_pulse_band_regime_chip_reflects_attrs():
    df = _full_frame()
    df.attrs["detected_market_regime"] = "BULL"
    assert "BULL" in _pulse_band_html(_pulse_stats(df))


def test_pulse_helpers_are_pure():
    for fn in (_pulse_stats, _pulse_band_html):
        assert "st." not in inspect.getsource(fn), f"{fn.__name__} must make no st.* call"
