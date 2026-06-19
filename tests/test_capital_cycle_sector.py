"""
test_capital_cycle_sector.py
============================
Contract for the sector capital-cycle signal from Capital Returns (Chancellor):
asset growth matters at the SECTORAL level — capital flooding into a sector
pressures all its constituents' returns; capital-starved sectors set up recovery.

sector_asset_growth = sector-median asset_growth_yoy (guarded: sector size >= 5).
sector_capital_phase: 🔥 Hot Capital (>20%) / ❄️ Capital Starved (<5%) / ⚖️ Neutral.

Run with: pytest tests/test_capital_cycle_sector.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from data_engine import compute_derived_signals
from test_data_quality_fixes import _frame


def _sector_df(n=10, **overrides):
    """n stocks all in one sector (Chemicals) so the sector median is well-defined."""
    base = dict(n=n, total_assets=1000.0, total_assets_1yb=900.0)
    base.update(overrides)
    return compute_derived_signals(_frame(**base))


def test_hot_capital_sector():
    """Sector-wide ~66% asset growth (TA 1000 vs 600) -> Hot Capital (caution)."""
    out = _sector_df(total_assets_1yb=600.0)
    assert (out["sector_capital_phase"].str.contains("Hot Capital")).all()
    assert np.allclose(out["sector_asset_growth"], (1000.0 - 600.0) / 600.0 * 100.0)


def test_capital_starved_sector():
    """Sector-wide ~2% asset growth (TA 1000 vs 980) -> Capital Starved (opportunity)."""
    out = _sector_df(total_assets_1yb=980.0)
    assert (out["sector_capital_phase"].str.contains("Capital Starved")).all()


def test_neutral_sector():
    """Sector-wide ~11% asset growth (5-20 band) -> Neutral."""
    out = _sector_df(total_assets_1yb=900.0)
    assert (out["sector_capital_phase"].str.contains("Neutral")).all()


def test_hot_bar_recalibrated_to_20pct():
    """Pins the 2026-06-18 Hot 30%->20% recalibration: a ~25% sector is now Hot (was Neutral at 30),
    and a ~18% sector stays Neutral — brackets the new bar so it can't silently drift back. The old
    test_hot_capital_sector (66.7%) is Hot at both 30 and 20, so it does NOT pin the boundary."""
    hot = _sector_df(total_assets_1yb=800.0)    # (1000-800)/800 = 25.0% > 20 -> Hot
    assert (hot["sector_capital_phase"].str.contains("Hot Capital")).all()
    assert np.allclose(hot["sector_asset_growth"], 25.0)
    near = _sector_df(total_assets_1yb=850.0)    # (1000-850)/850 = 17.6% < 20 -> Neutral (was the gap)
    assert (near["sector_capital_phase"] == "⚖️ Neutral").all()


def test_small_sector_is_neutral_not_classified():
    """Sector with < 5 stocks -> median unstable -> NaN sector_asset_growth -> Neutral."""
    out = _sector_df(n=3, total_assets_1yb=600.0)  # would be Hot, but only 3 stocks
    assert out["sector_asset_growth"].isna().all()
    assert (out["sector_capital_phase"] == "⚖️ Neutral").all()
