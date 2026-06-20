"""Drift-proof contract: the Forensics tab's Fisher Quality module (render_fisher_module) must show the
SAME pass/fail as the engine's `fisher_quality_pass` for EVERY stock.

History: the module used to recompute a softer 7-proxy verdict — its P15 used the lenient forensic-
cascade "Clean" rule (forensic ≥80 & ≤3 flags) while the engine's fw_fisher gate uses the STRICTER
forensic_score ≥90. So Sarda Energy (forensic 89.29) showed "7/7 High Quality Alignment" in the module
yet had NO "Fisher Quality" pill in the Frameworks tab (and its own banner read "⚪ Laggard"). Option B
fixed it by routing the module through _fisher_quality_proxies, which mirrors fw_fisher EXACTLY. This
test pins that equivalence so the contradiction can never return.
"""
import contextlib
import io
from pathlib import Path

import pandas as pd
import pytest

from ui.ui_tearsheet import _fisher_quality_proxies

_DATA_DIR = Path(__file__).resolve().parent.parent / "Other Resources" / "CSV Data"


def test_helper_returns_seven_named_gates():
    """The helper returns exactly the 7 Fisher Quality proxy gates, each a (label, bool, value) tuple."""
    stock = pd.Series({"rev_gr_5y": 20.0, "npm": 18.0, "npm_1yb": 14.0, "cfo_to_pat": 160.0,
                       "dilution_flag": 0, "operating_leverage": 1, "forensic_score": 95.0})
    proxies = _fisher_quality_proxies(stock)
    assert len(proxies) == 7
    for label, is_pass, val in proxies:
        assert isinstance(label, str) and isinstance(is_pass, bool) and isinstance(val, str)
    assert all(p for _, p, _ in proxies), "a fully-qualifying stock must show all 7 gates green"


def test_p15_is_the_strict_forensic_90_bar():
    """P15 must be the engine's strict forensic ≥90 bar — the Sarda case: 6 gates pass, forensic 89 → the
    module must NOT show a full pass (mirrors the engine rejecting the Fisher Quality pill)."""
    base = {"rev_gr_5y": 21.0, "npm": 18.7, "npm_1yb": 14.7, "cfo_to_pat": 163.0,
            "dilution_flag": 0, "operating_leverage": 1}
    sarda = _fisher_quality_proxies(pd.Series({**base, "forensic_score": 89.29}))
    p15 = next(p for label, p, _ in sarda if label.startswith("P15"))
    assert p15 is False, "forensic 89.29 < 90 must FAIL Fisher's strict integrity gate"
    assert not all(p for _, p, _ in sarda), "Sarda must not read a full Fisher Quality pass"
    # One notch higher clears it.
    clears = _fisher_quality_proxies(pd.Series({**base, "forensic_score": 90.0}))
    assert all(p for _, p, _ in clears), "forensic 90 (all 6 others pass) must clear all 7 gates"


@pytest.mark.skipif(not _DATA_DIR.is_dir(),
                    reason="Local CSV data not present (code-only checkout) — needs the real scored frame")
def test_module_pass_equals_engine_for_every_stock():
    """Across the live ~2107-stock universe, the module's 7-proxy verdict (all gates pass) must equal the
    engine's fisher_quality_pass for EVERY row — the contradiction Option B eliminated, drift-pinned."""
    with contextlib.redirect_stdout(io.StringIO()):
        from core import fetch_and_clean_data, run_scoring_pipeline
        df = run_scoring_pipeline(fetch_and_clean_data("local"))
    assert "fisher_quality_pass" in df.columns
    mismatches = []
    for _, row in df.iterrows():
        shown_pass = all(p for _, p, _ in _fisher_quality_proxies(row))
        engine_pass = int(row.get("fisher_quality_pass", 0)) == 1
        if shown_pass != engine_pass:
            mismatches.append((row.get("name"), shown_pass, engine_pass))
    assert not mismatches, (
        f"{len(mismatches)} stocks where the Fisher module's displayed pass disagrees with the engine's "
        f"fisher_quality_pass (the Option-B divergence must stay fixed): {mismatches[:8]}")
