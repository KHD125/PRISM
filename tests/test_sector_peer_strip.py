"""
test_sector_peer_strip.py
=========================
Unit contract for the Overview "vs Sector Peers" strip (ui_tearsheet._sector_peer_strip_html).

WHY this exists: the strip's whole reason to be is the VALUE-TRAP GUARD — a stock can score
high in absolute terms yet sit below its sector's ROCE median. The 6-axis scorecard is
absolute-only and structurally cannot show this; the strip surfaces the sector-relative lens
by reusing four already-computed-but-orphaned columns (sector_roce_pct_rank, emc_flag,
emc_sector_beat_count, sector_capital_phase). These tests pin the percentile→colour→label
mapping deterministically against synthetic rows (no CSV data needed).
"""

import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ui.ui_tearsheet import _sector_peer_strip_html
from config import COLORS


def _stock(**kw) -> pd.Series:
    base = dict(
        sector="Auto Ancillaries",
        sector_roce_pct_rank=0.50,
        emc_flag=0,
        emc_sector_beat_count=0,
        sector_capital_phase="⚖️ Neutral",
    )
    base.update(kw)
    return pd.Series(base)


def test_sector_leader_is_green():
    # A genuine leader (97th pctile) → green + the "sector ROCE leader" tag + a percentile-DERIVED
    # "Top 3%" (100-97). The old code hardcoded "Top 30% — sector ROCE leader" for the whole 70-100
    # band, contradicting the displayed value; that fixed string must be gone now.
    html = _sector_peer_strip_html(_stock(sector_roce_pct_rank=0.97))
    assert "97" in html
    assert COLORS["green"] in html
    assert "sector roce leader" in html.lower()
    assert "top 3%" in html.lower()
    assert "top 30%" not in html.lower()


def test_above_median_is_gold():
    html = _sector_peer_strip_html(_stock(sector_roce_pct_rank=0.60))
    assert "60" in html
    assert COLORS["gold"] in html
    assert "above sector median" in html.lower()


def test_below_median_flags_value_trap_red():
    """The core feature: below-sector-median ROCE renders RED and names the value-trap risk."""
    html = _sector_peer_strip_html(_stock(sector_roce_pct_rank=0.30))
    assert "30" in html
    assert COLORS["red"] in html
    assert "below sector median" in html.lower()
    assert "value-trap" in html.lower()


def test_emc_beat_count_and_label():
    html = _sector_peer_strip_html(_stock(emc_flag=1, emc_sector_beat_count=5))
    assert "5/5" in html
    assert "beats sector roe" in html.lower()


def test_emc_lagging_is_muted():
    html = _sector_peer_strip_html(_stock(emc_flag=0, emc_sector_beat_count=1))
    assert "1/5" in html
    assert "lags sector roe" in html.lower()


def test_sector_name_in_header():
    html = _sector_peer_strip_html(_stock(sector="Pharmaceuticals"))
    assert "Pharmaceuticals" in html


def test_capital_phase_hot_caution():
    html = _sector_peer_strip_html(_stock(sector_capital_phase="🔥 Hot Capital (caution)"))
    assert "Hot" in html
    assert "mean-reversion" in html.lower()


def test_capital_phase_starved_opportunity():
    html = _sector_peer_strip_html(
        _stock(sector_capital_phase="❄️ Capital Starved (opportunity)")
    )
    assert "Starved" in html
    assert "opportunity" in html.lower()


def test_missing_rank_is_graceful_dash():
    s = _stock().drop("sector_roce_pct_rank")
    html = _sector_peer_strip_html(s)
    assert "—" in html
    assert "no sector" in html.lower()


def test_no_nan_leaks_into_html():
    s = _stock(sector_roce_pct_rank=np.nan, emc_sector_beat_count=np.nan)
    html = _sector_peer_strip_html(s).lower()
    assert "nan" not in html
    assert not re.findall(r"[+\-₹ >]nan[%< ]", html)


def test_html_is_escaped_against_injection():
    html = _sector_peer_strip_html(_stock(sector="Foo <script>&"))
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


# ── 🏅 Rank in Sector tile (named-cohort position) ──────────────────────────
def test_sector_rank_tile_shows_position():
    html = _sector_peer_strip_html(_stock(sector_composite_rank=3, sector_peer_count=47))
    assert "#3 of 47" in html
    assert "rank in sector" in html.lower()


def test_sector_rank_top_quartile_is_green():
    # rank 2 of 20 = pos 0.10 (<=0.25). ROCE tile forced RED (0.30) so green is the rank tile's.
    html = _sector_peer_strip_html(
        _stock(sector_roce_pct_rank=0.30, sector_composite_rank=2, sector_peer_count=20)
    )
    assert "#2 of 20" in html
    assert "Top 10% by composite" in html
    assert COLORS["green"] in html


def test_sector_rank_mid_is_gold():
    # rank 10 of 20 = pos 0.50 (<=0.50). ROCE forced RED so the only gold is the rank tile.
    html = _sector_peer_strip_html(
        _stock(sector_roce_pct_rank=0.30, sector_composite_rank=10, sector_peer_count=20)
    )
    assert "#10 of 20" in html
    assert "Top 50% by composite" in html
    assert COLORS["gold"] in html


def test_sector_rank_bottom_is_muted():
    # rank 18 of 20 = pos 0.90 -> muted. ROCE forced RED, EMC/capital default muted -> no
    # green/gold anywhere proves the rank tile is muted.
    html = _sector_peer_strip_html(
        _stock(sector_roce_pct_rank=0.30, sector_composite_rank=18, sector_peer_count=20)
    )
    assert "#18 of 20" in html
    assert COLORS["green"] not in html
    assert COLORS["gold"] not in html


def test_sole_peer_has_no_cohort():
    html = _sector_peer_strip_html(_stock(sector_composite_rank=1, sector_peer_count=1))
    assert "no sector cohort" in html.lower()


def test_rank_tile_no_nan_leak_when_missing():
    # default _stock() has no rank columns -> graceful dash, never "nan"
    html = _sector_peer_strip_html(_stock()).lower()
    assert "rank in sector" in html
    assert "nan" not in html
