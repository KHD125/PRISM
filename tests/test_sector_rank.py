"""
test_sector_rank.py
===================
Contract for the sector-cohort rank columns added in apply_forensic_penalty (step 3) — the data
behind the Overview "🏅 Rank in Sector" tile.

sector_composite_rank = rank within the stock's OWN sector by the post-penalty composite_score
(method="min" → ties share a rank); sector_peer_count = sector group size. Both float + NaN-safe:
a NaN/absent sector propagates NaN (no sentinel). Computed alongside the post-penalty overall
`rank`, so both use the SAME composite basis and can never contradict each other.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

from core.forensic_engine import apply_forensic_penalty


def _run():
    # red_flag_count = 0 -> multiplier 1.0 -> composite unchanged, so ranks are predictable.
    df = pd.DataFrame({
        "sector":          ["A", "A", "A", "B", "B", None],
        "composite_score": [90.0, 70.0, 70.0, 50.0, 80.0, 60.0],
        "red_flag_count":  [0, 0, 0, 0, 0, 0],
    })
    return apply_forensic_penalty(df)


def test_sector_rank_within_group_min_method():
    r = _run()["sector_composite_rank"]
    # Sector A [90,70,70] -> [1,2,2] (ties share min); Sector B [50,80] -> [2,1]
    assert r.iloc[0] == 1
    assert r.iloc[1] == 2 and r.iloc[2] == 2
    assert r.iloc[3] == 2 and r.iloc[4] == 1


def test_sector_peer_count_is_group_size():
    c = _run()["sector_peer_count"]
    assert c.iloc[0] == 3 and c.iloc[1] == 3 and c.iloc[2] == 3
    assert c.iloc[3] == 2 and c.iloc[4] == 2


def test_rank_one_is_highest_composite_in_sector():
    out = _run()
    for sec in ["A", "B"]:
        grp = out[out["sector"] == sec]
        top = grp.loc[grp["sector_composite_rank"] == 1, "composite_score"].iloc[0]
        assert top == grp["composite_score"].max()


def test_nan_sector_propagates_nan_not_sentinel():
    out = _run()
    assert pd.isna(out["sector_composite_rank"].iloc[5])
    assert pd.isna(out["sector_peer_count"].iloc[5])


def test_no_sector_column_is_a_no_op():
    """Minimal forensic-only frame (no 'sector') must not crash and must not invent the columns."""
    out = apply_forensic_penalty(pd.DataFrame({
        "composite_score": [80.0, 60.0],
        "red_flag_count":  [0, 0],
    }))
    assert "sector_composite_rank" not in out.columns
    assert "sector_peer_count" not in out.columns
