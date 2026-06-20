"""Contract: the Overview's sector ROCE-percentile tile subtitle is DERIVED from the actual
percentile, so it can never contradict the displayed value.

The old code hardcoded "Top 30% — sector ROCE leader" for the ENTIRE 70-100 percentile band, so a
100th-percentile stock rendered "100 · Top 30% — sector ROCE leader" — self-contradictory (100 ≠
top-30%) AND it crowned every top-third stock the "leader." Fixed 2026-06-20 via the pure helper
_roce_sector_label; this pins it.
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ui.ui_tearsheet import _roce_sector_label


def test_leader_band_topX_matches_percentile_not_fixed_30():
    band, val, sub = _roce_sector_label(1.0)            # 100th pctile = the genuine leader
    assert band == "leader" and val == "100"
    assert "Top 1%" in sub and "sector ROCE leader" in sub
    assert "Top 30%" not in sub                          # the original bug must be gone

    band, val, sub = _roce_sector_label(0.71)           # 71st pctile = top 29%, NOT the leader
    assert band == "leader" and val == "71"
    assert "Top 29%" in sub and "sector ROCE leader" not in sub

    assert "Top 15%" in _roce_sector_label(0.85)[2]


def test_bands_and_nan_safe():
    assert _roce_sector_label(0.60)[0] == "above"
    assert _roce_sector_label(0.30)[0] == "below"
    assert _roce_sector_label(None) == ("none", "—", "No sector peer rank")
    assert _roce_sector_label(float("nan"))[0] == "none"     # NaN-safe (semantic-truth)


def test_subtitle_never_understates_the_value():
    """For every percentile in the leader band, the displayed 'Top X%' MUST equal 100 - percentile
    (floored at 1) — so the subtitle can never claim a worse rank than the tile's number shows."""
    for r in [0.70, 0.78, 0.91, 0.95, 1.0]:
        _, val, sub = _roce_sector_label(r)
        m = re.search(r"Top (\d+)%", sub)
        assert m, f"no 'Top X%' in subtitle: {sub!r}"
        assert int(m.group(1)) == max(1, round(100 - float(r) * 100)), (r, sub)
