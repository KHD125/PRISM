"""
test_forensic_calibration.py
============================
Contract for the 2026-06-12 Schilit forensic audit fixes (book: Financial Shenanigans,
India Forensic Edition — converted at Other Resources/Financial Shenanigans (...)).

FIX 1 — rf_low_cfo_ebitda recalibrated 90% -> 50%:
  The <90% threshold was Mukherjea's Coffee Can ELITE-QUALITY gate misused as a fraud
  flag. CFO is AFTER tax+interest while EBITDA is BEFORE both, so ~75% is mathematical
  par for a clean 25%-tax company — the old flag fired for 54% of the universe
  (median CFO/EBITDA = 68.8%), flagging ordinary tax mathematics as shenanigans and
  inflating red_flag_count -> harsher forensic multipliers for half the market.
  Schilit's actual cash-quality detections are CFFO vs Net Income (rf_low_cfo_pat,
  <70%) and FCF/EBITDA < 0.3 (rf_low_fcf_ebitda) — both already implemented exactly.
  50% = cash conversion clearly below what taxes alone explain -> genuine anomaly.
  (The Coffee Can framework gate keeps its own >= 90% requirement, untouched.)

FIX 2 — rf_psu_value_destruction was DEAD (0 fires on 2107 stocks):
  The WCS24 "value-destruction loop" was a 4-way AND whose legs barely intersect
  (PSU+low_spread=6, PSU+low_velocity=5, PSU+cwip_delays=4 -> all four = 0).
  New: PSU + below-CoE returns (the core of value destruction) + AT LEAST ONE
  reinforcing leg (poor reinvestment velocity OR stuck CWIP).

Run with: pytest tests/test_forensic_calibration.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
sys.path.insert(0, os.path.dirname(__file__))   # enables cross-test helper import

import pandas as pd
import numpy as np

from config import FORENSIC
from forensic_engine import compute_forensic_signals
from test_data_quality_fixes import _frame   # full-merge synthetic frame helper


# ═══════════════════════════════════════════════════════════════════════════
# FIX 1 — CFO/EBITDA forensic threshold
# ═══════════════════════════════════════════════════════════════════════════

def test_config_cfo_ebitda_forensic_threshold_is_50():
    assert FORENSIC["cfo_ebitda_clean_threshold"] == 50.0, (
        "Forensic CFO/EBITDA threshold must be 50%. 90% is Coffee Can's elite-quality "
        "gate (kept in fw_coffee_can), NOT a Schilit fraud signal — CFO is after-tax, "
        "EBITDA pre-tax, so ~75% is par; <90% flagged the median Indian company."
    )


def _forensic_frame(n: int = 12, **overrides) -> pd.DataFrame:
    """Pipeline-faithful: derived signals first (forensic engine needs them), then
    override the column under test."""
    from data_engine import compute_derived_signals
    df = compute_derived_signals(_frame(n))
    for k, v in overrides.items():
        df[k] = [v] * n if not isinstance(v, (list, np.ndarray)) else v
    return df


def test_taxpaying_median_company_not_flagged():
    """CFO/EBITDA = 70% is normal after-tax conversion (median universe = 68.8%) —
    must NOT count as a forensic red flag."""
    out = compute_forensic_signals(_forensic_frame(cfo_to_ebitda=70.0))
    assert (out["rf_low_cfo_ebitda"] == 0).all()


def test_genuinely_weak_cash_conversion_flagged():
    """CFO/EBITDA = 40%: more than taxes is leaking — fires."""
    out = compute_forensic_signals(_forensic_frame(cfo_to_ebitda=40.0))
    assert (out["rf_low_cfo_ebitda"] == 1).all()


def test_missing_cfo_ebitda_not_flagged():
    out = compute_forensic_signals(_forensic_frame(cfo_to_ebitda=np.nan))
    assert (out["rf_low_cfo_ebitda"] == 0).all()


# ═══════════════════════════════════════════════════════════════════════════
# FIX 2 — PSU value-destruction loop must be alive
# ═══════════════════════════════════════════════════════════════════════════

def _psu_frame(**overrides):
    """A PSU (NTPC-named, govt majority, zero pledge) with below-CoE returns and
    stuck CWIP, but HEALTHY reinvestment velocity — fires under the new core+one
    logic, was impossible under the old 4-way AND."""
    from data_engine import compute_derived_signals
    base = dict(
        name=["NTPC Test Ltd"] * 25,
        promoter_holdings=51.0,
        pledged_percentage=0.0,
        roce=5.0,                      # capital_return_spread = 5 - 12 = -7 <= 0 (core)
        dividend_payout_ratio=0.0,     # reinvestment_rate = 1.0 (velocity HEALTHY)
        operating_cash_flow=100.0,
        free_cash_flow=80.0,           # fcf_to_ocf_velocity = 0.8 (velocity HEALTHY)
        cwip=100.0, cwip_1yb=100.0,    # CWIP present both years
        fixed_assets=500.0, fixed_assets_1yb=500.0,   # no capacity went live -> conversion <= 0
    )
    base.update(overrides)
    return compute_derived_signals(_frame(25, **base))


def test_psu_core_plus_cwip_delay_fires():
    out = _psu_frame()
    assert (out["psu_value_destruction_flag"] == 1).all(), (
        "PSU with below-CoE returns AND stuck CWIP must fire even when reinvestment "
        "velocity is healthy — the old 4-way AND made this flag dead code (0 of 2107)"
    )


def test_private_company_never_fires():
    out = _psu_frame(name=["Private Widgets Ltd"] * 25, sector=["Chemicals"] * 25)
    assert (out["psu_value_destruction_flag"] == 0).all()


def test_healthy_psu_not_flagged():
    """PSU earning ABOVE cost of equity: no value destruction, regardless of CWIP."""
    out = _psu_frame(roce=20.0)   # spread = +8
    assert (out["psu_value_destruction_flag"] == 0).all()
