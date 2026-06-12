"""
Compiler Contract Verification — rf_capex_mirage (Flag 22) refactor.
2026-05-31 | Multibagger Discovery System

Scenarios:
  A    Zero-denominator protection:  dep_rate_1yb=0 → no Inf/NaN in output
  A2   Zero FA_1yb protection:       fixed_assets_1yb=0 → no Inf
  B    Capital-intensive mask:        IT sector → rf=0 always
  B2   Financial exclusion:           is_financial=True → rf=0 always
  C    True-positive:                 fast rev + under-investment → rf=1
  C2   Borderline ratio 0.499:        → rf=1
  D    True-negative adequate capex:  ratio>=0.5 → rf=0
  D2   Revenue below threshold:       rev<=20% → rf=0
  E    Missing dep_rate_1yb (NaN):    → rf=0 (conservative fallback)
  E2   Missing rev_gr_yoy (NaN):      → rf=0
  F    2000-row vectorized:           all outputs strictly binary 0/1, no NaN/Inf
"""
import numpy as np
import pandas as pd


# ── Exact replica of the refactored rf_capex_mirage logic ──────────────────
_HIGH_DSO_SECTORS = frozenset({
    "IT - Software", "IT - Hardware", "Healthcare", "Pharmaceuticals"
})


def compute_rf_capex_mirage(df: pd.DataFrame) -> pd.Series:
    """Inline replication of forensic_engine.py Flag 22 post-refactor."""
    _is_svc_cm = (
        df.get("sector", pd.Series("", index=df.index))
        .fillna("").isin(_HIGH_DSO_SECTORS)
    )
    _is_fin_cm = (
        df.get("is_financial", pd.Series(False, index=df.index)).fillna(False)
    )
    _is_capital_intensive = ~_is_svc_cm & ~_is_fin_cm

    _dep_rate_1yb_col = df.get("dep_rate_1yb", pd.Series(np.nan, index=df.index))
    _depr_exact_cm = (
        (_dep_rate_1yb_col.fillna(0) / 100.0) * df["fixed_assets_1yb"].fillna(0)
    )
    _capex_net_cm = (
        (df["fixed_assets"].fillna(0) - df["fixed_assets_1yb"].fillna(0)).clip(lower=0)
    )

    # Zero-denominator protection: np.where(denom > 0, num/denom, np.nan)
    _capex_ratio_cm = np.where(
        _depr_exact_cm > 0.0,
        _capex_net_cm / _depr_exact_cm,
        np.nan,
    )

    return pd.Series(
        np.where(
            _is_capital_intensive
            & df["rev_gr_yoy"].notna()
            & (_depr_exact_cm > 0.0),
            (
                (df["rev_gr_yoy"] > 20)
                & (pd.Series(_capex_ratio_cm, index=df.index).fillna(1.0) < 0.5)
            ).astype(int),
            0,
        ),
        index=df.index,
    )


# ── Shared helper ─────────────────────────────────────────────────────────────

def _make_row(**overrides) -> pd.DataFrame:
    """Single-row DataFrame with manufacturing-sector defaults."""
    row = dict(
        sector="Manufacturing",
        is_financial=False,
        fixed_assets=1000.0,
        fixed_assets_1yb=800.0,
        dep_rate_1yb=10.0,   # 10% × 800 = 80 Cr depreciation
        rev_gr_yoy=25.0,     # >20% triggers revenue condition
    )
    row.update(overrides)
    return pd.DataFrame([row])


# ── Test A: Zero-denominator protection ──────────────────────────────────────

def test_A_zero_dep_rate_no_inf():
    df = _make_row(dep_rate_1yb=0.0)
    r = compute_rf_capex_mirage(df)
    assert r.iloc[0] == 0,           "dep_rate_1yb=0 → depr_exact=0 → flag must be 0"
    assert not np.isinf(r.iloc[0]), "must not produce Inf"
    assert not np.isnan(r.iloc[0]), "must not produce NaN"


def test_A2_zero_fa_1yb_no_inf():
    df = _make_row(fixed_assets_1yb=0.0, dep_rate_1yb=10.0)
    r = compute_rf_capex_mirage(df)
    assert r.iloc[0] == 0
    assert not np.isinf(r.iloc[0])
    assert not np.isnan(r.iloc[0])


# ── Test B: Asset-light / financial exclusion ─────────────────────────────────

def test_B_IT_sector_excluded():
    # Extreme under-investment + high revenue — must still be 0
    df = _make_row(
        sector="IT - Software",
        rev_gr_yoy=50.0,
        dep_rate_1yb=30.0,
        fixed_assets=100.0,     # massive FA drop
        fixed_assets_1yb=1000.0,
    )
    assert compute_rf_capex_mirage(df).iloc[0] == 0, "IT sector must never flag"


def test_B2_financial_excluded():
    df = _make_row(is_financial=True, rev_gr_yoy=35.0)
    assert compute_rf_capex_mirage(df).iloc[0] == 0, "financial sector must never flag"


def test_B3_pharma_excluded():
    df = _make_row(sector="Pharmaceuticals", rev_gr_yoy=40.0, dep_rate_1yb=18.0,
                   fixed_assets=200.0, fixed_assets_1yb=800.0)
    assert compute_rf_capex_mirage(df).iloc[0] == 0


# ── Test C: True-positive (deferred-maintenance time bomb) ────────────────────

def test_C_true_positive():
    # depr_exact = 10% × 800 = 80 Cr; capex_net = 830-800 = 30 Cr
    # ratio = 30/80 = 0.375 < 0.5 AND rev_gr=25 > 20 → FLAG
    df = _make_row(fixed_assets=830.0, fixed_assets_1yb=800.0,
                   dep_rate_1yb=10.0, rev_gr_yoy=25.0)
    assert compute_rf_capex_mirage(df).iloc[0] == 1, "deferred-maintenance must flag"


def test_C2_borderline_ratio_just_below_threshold():
    # depr_exact = 10% × 1000 = 100 Cr; capex_net = 49.9 Cr → ratio = 0.499 < 0.5
    df = _make_row(fixed_assets=1049.9, fixed_assets_1yb=1000.0,
                   dep_rate_1yb=10.0, rev_gr_yoy=21.0)
    assert compute_rf_capex_mirage(df).iloc[0] == 1


def test_C3_high_dep_rate_sector():
    # Telecom: dep_rate_1yb=18% × 500 = 90 Cr; capex_net = 530-500 = 30 Cr
    # ratio = 30/90 = 0.333 < 0.5 → FLAG
    df = _make_row(sector="Telecom-Equipment",   # not in _HIGH_DSO_SECTORS
                   dep_rate_1yb=18.0, fixed_assets=530.0, fixed_assets_1yb=500.0,
                   rev_gr_yoy=22.0)
    assert compute_rf_capex_mirage(df).iloc[0] == 1


# ── Test D: True-negative (adequate capex) ───────────────────────────────────

def test_D_adequate_capex_clean():
    # depr_exact = 10% × 800 = 80 Cr; capex_net = 1300-800 = 500 Cr
    # ratio = 500/80 = 6.25 > 0.5 → CLEAN
    df = _make_row(fixed_assets=1300.0, fixed_assets_1yb=800.0,
                   dep_rate_1yb=10.0, rev_gr_yoy=25.0)
    assert compute_rf_capex_mirage(df).iloc[0] == 0


def test_D2_ratio_exactly_half():
    # depr_exact = 10% × 1000 = 100; capex_net = 50 → ratio = 0.50 NOT < 0.5 → CLEAN
    df = _make_row(fixed_assets=1050.0, fixed_assets_1yb=1000.0,
                   dep_rate_1yb=10.0, rev_gr_yoy=25.0)
    assert compute_rf_capex_mirage(df).iloc[0] == 0, "ratio=0.5 is exactly at threshold, must not flag"


def test_D3_revenue_at_threshold_not_exceeded():
    # rev_gr_yoy=20.0 is NOT > 20 (strict >) → CLEAN
    df = _make_row(fixed_assets=830.0, fixed_assets_1yb=800.0,
                   dep_rate_1yb=10.0, rev_gr_yoy=20.0)
    assert compute_rf_capex_mirage(df).iloc[0] == 0, "rev_gr=20 exactly must not flag (strict >)"


# ── Test E: Missing data NaN safety ──────────────────────────────────────────

def test_E_missing_dep_rate_stays_zero():
    df = _make_row()
    df["dep_rate_1yb"] = np.nan
    r = compute_rf_capex_mirage(df)
    assert r.iloc[0] == 0, "NaN dep_rate_1yb → depr_exact=0 → flag=0"
    assert not np.isnan(r.iloc[0])


def test_E2_missing_rev_gr_stays_zero():
    df = _make_row(rev_gr_yoy=np.nan)
    r = compute_rf_capex_mirage(df)
    assert r.iloc[0] == 0, "NaN rev_gr_yoy → condition gate fails → flag=0"
    assert not np.isnan(r.iloc[0])


def test_E3_both_missing_stays_zero():
    df = _make_row(rev_gr_yoy=np.nan, dep_rate_1yb=np.nan)
    r = compute_rf_capex_mirage(df)
    assert r.iloc[0] == 0


# ── Test F: 2000-row vectorized binary output integrity ───────────────────────

def test_F_vectorized_binary_no_inf_no_nan():
    N = 2000
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "sector": rng.choice(
            ["Manufacturing", "Steel", "IT - Software", "Pharmaceuticals",
             "Cement", "Auto", "Telecom-Equipment", "FMCG"], N
        ),
        "is_financial": rng.choice([False, True], N, p=[0.85, 0.15]),
        "fixed_assets":     rng.uniform(50, 10_000, N),
        "fixed_assets_1yb": rng.uniform(50, 10_000, N),
        "dep_rate_1yb": rng.choice([np.nan, 4.0, 5.0, 8.0, 10.0, 18.0, 30.0], N),
        "rev_gr_yoy":   np.where(
            rng.random(N) < 0.1, np.nan, rng.uniform(-15, 60, N)
        ),
    })
    result = compute_rf_capex_mirage(df)
    assert not result.isna().any(),       "output must have no NaN"
    assert not np.isinf(result).any(),   "output must have no Inf"
    assert result.isin([0, 1]).all(),     "output must be strictly binary 0/1"
    print(f"\n  [F] Flagged: {result.sum()} / {N} ({result.mean():.1%})")


# ── Entry point (run directly without pytest) ─────────────────────────────────
if __name__ == "__main__":
    tests = [
        ("A  zero dep_rate no Inf",              test_A_zero_dep_rate_no_inf),
        ("A2 zero FA_1yb no Inf",                test_A2_zero_fa_1yb_no_inf),
        ("B  IT sector excluded",                test_B_IT_sector_excluded),
        ("B2 financial excluded",                test_B2_financial_excluded),
        ("B3 pharma excluded",                   test_B3_pharma_excluded),
        ("C  true positive deferred-maint",      test_C_true_positive),
        ("C2 borderline ratio 0.499",            test_C2_borderline_ratio_just_below_threshold),
        ("C3 high dep-rate sector",              test_C3_high_dep_rate_sector),
        ("D  adequate capex clean",              test_D_adequate_capex_clean),
        ("D2 ratio exactly 0.5 clean",           test_D2_ratio_exactly_half),
        ("D3 rev_gr=20 exactly clean",           test_D3_revenue_at_threshold_not_exceeded),
        ("E  NaN dep_rate_1yb safe",             test_E_missing_dep_rate_stays_zero),
        ("E2 NaN rev_gr_yoy safe",               test_E2_missing_rev_gr_stays_zero),
        ("E3 both NaN safe",                     test_E3_both_missing_stays_zero),
        ("F  2000-row vectorized binary",        test_F_vectorized_binary_no_inf_no_nan),
    ]
    passed = failed = 0
    for label, fn in tests:
        try:
            fn()
            print(f"  PASS  {label}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {label}  →  {e}")
            failed += 1
    print(f"\n{'='*55}")
    print(f"  Results: {passed} passed, {failed} failed out of {len(tests)} tests")
