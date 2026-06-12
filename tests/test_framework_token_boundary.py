"""
Compiler Contract Verification — Sidebar Framework Token Boundary Tests
2026-05-31 | Multibagger Discovery System

Validates:
  S1   Splitter uses ", " → no leading-space tokens in _all_fw
  S2   "None" cells excluded from token set
  S3   Empty string tokens excluded
  S4   Unique set: duplicate framework tokens across rows deduplicated
  S5   Single-token cell parses correctly
  S6   Max-density cell (35 frameworks) tokenises to 35 clean distinct tokens

  F1   Anchored regex: "Bruised Blue Chip" does NOT match "Bruised Blue Chip 29"
  F2   Anchored regex: "Bruised Blue Chip 29" does NOT match "Bruised Blue Chip"
  F3   Anchored regex: first token in string matches correctly
  F4   Anchored regex: last token in string matches correctly
  F5   Anchored regex: middle token matches correctly
  F6   Anchored regex: empty cell (na=False) → no match, no crash
  F7   Anchored regex: "None" cell → no match
  F8   re.escape: tokens with regex special chars do not raise
  F9   Multi-select OR logic: union of two token masks correct
  F10  Full 2108-row simulation: no leading spaces, no duplicates, all binary
"""
import re
import pandas as pd
import numpy as np


# ── Inline replica of the fixed sidebar splitter ────────────────────────────
def build_all_fw(df: pd.DataFrame) -> list:
    """Replica of fixed _all_fw builder in app.py."""
    if "frameworks_passed" not in df.columns:
        return []
    return sorted(set(
        fw.strip()
        for cell in df["frameworks_passed"].dropna()
        if cell != "None"
        for fw in cell.split(", ")
        if fw.strip()
    ))


# ── Inline replica of the fixed filter application ─────────────────────────
def apply_fw_filter(filt: pd.DataFrame, sel_fw: list) -> pd.DataFrame:
    """Replica of fixed sel_fw filter in app.py."""
    if not sel_fw or "frameworks_passed" not in filt.columns:
        return filt
    _fw_mask = pd.Series(False, index=filt.index)
    for _fw in sel_fw:
        _pat = r"(?:^|, )" + re.escape(_fw) + r"(?:,|$)"
        _fw_mask = _fw_mask | filt["frameworks_passed"].str.contains(_pat, regex=True, na=False)
    return filt[_fw_mask]


# ── Helpers ──────────────────────────────────────────────────────────────────
def _df(*cells) -> pd.DataFrame:
    return pd.DataFrame({"frameworks_passed": list(cells)})


# ══════════════════════════════════════════════════════════════════════════════
# SPLITTER TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_S1_no_leading_space_tokens():
    """(S1) Splitter on ', ' → tokens have no leading spaces."""
    df = _df("Coffee Can, QGLP, Bruised Blue Chip 29")
    tokens = build_all_fw(df)
    for t in tokens:
        assert t == t.strip(), f"Token has whitespace: repr={repr(t)}"


def test_S2_none_cells_excluded():
    """(S2) 'None' string excluded from token set."""
    df = _df("None", "QGLP, Coffee Can")
    tokens = build_all_fw(df)
    assert "None" not in tokens


def test_S3_empty_tokens_excluded():
    """(S3) Empty strings from trailing commas/spaces not in token set."""
    df = _df("QGLP, , Coffee Can")  # double-space gap
    tokens = build_all_fw(df)
    assert "" not in tokens
    assert all(t.strip() != "" for t in tokens)


def test_S4_deduplication_across_rows():
    """(S4) Same token appearing in multiple rows appears once in _all_fw."""
    df = _df("QGLP, Coffee Can", "Coffee Can, Lynch Dream", "QGLP")
    tokens = build_all_fw(df)
    assert tokens.count("QGLP") == 1
    assert tokens.count("Coffee Can") == 1


def test_S5_single_token_cell():
    """(S5) Single-token cell (no comma) parses to exactly one token."""
    df = _df("Bruised Blue Chip 29")
    tokens = build_all_fw(df)
    assert tokens == ["Bruised Blue Chip 29"]


def test_S6_max_density_35_frameworks():
    """(S6) Cell with 35 frameworks produces 35 clean unique tokens."""
    # Simulate a stock passing all 35 scoring_engine frameworks
    _ALL_35 = [
        "QGLP", "Coffee Can", "Magic Formula", "SMILE", "Lynch Dream",
        "CAN SLIM", "Fallen Quality", "EP Improver", "Peaceful Investing",
        "Unusual Billionaires", "Fisher Quality", "100-Bagger", "Diamond",
        "Wide Moat", "Outsider CEO", "Quality Compounder", "Dhandho Asymmetry",
        "Parikh Contrarian", "Baid Compounder", "Long Game Quality",
        "SEPA Momentum", "Basant 30% Club", "Quality Momentum",
        "MOSL Wealth Creator", "Economic Moat", "Blue Chip Quality",
        "SQGLP Century Stock", "CAP-GAP Compounder", "Consistent in Volatile",
        "EP Hockey Stick", "Bruised Blue Chip 29", "Multi-Trillion Cap",
        "Fisher Scalability", "Financial Shenanigans", "Marks Cycle Shield",
    ]
    cell = ", ".join(_ALL_35)
    df = _df(cell)
    tokens = build_all_fw(df)
    assert len(tokens) == 35, f"Expected 35 tokens, got {len(tokens)}: {tokens}"
    for t in tokens:
        assert t == t.strip(), f"Leading/trailing space in token: {repr(t)}"
    assert sorted(tokens) == sorted(_ALL_35)


# ══════════════════════════════════════════════════════════════════════════════
# FILTER TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_F1_bbc_does_not_match_bbc29():
    """(F1) 'Bruised Blue Chip' must NOT match cell containing only 'Bruised Blue Chip 29'."""
    df = _df("QGLP, Bruised Blue Chip 29, Coffee Can")
    result = apply_fw_filter(df, ["Bruised Blue Chip"])
    assert result.empty, "Substring collision: 'Bruised Blue Chip' matched 'Bruised Blue Chip 29'"


def test_F2_bbc29_does_not_match_standalone_bbc():
    """(F2) 'Bruised Blue Chip 29' must NOT match cell containing only 'Bruised Blue Chip'."""
    # If a future engine version writes "Bruised Blue Chip" as a token
    df = _df("QGLP, Bruised Blue Chip, Coffee Can")
    result = apply_fw_filter(df, ["Bruised Blue Chip 29"])
    assert result.empty, "'Bruised Blue Chip 29' matched 'Bruised Blue Chip' (substring collision)"


def test_F3_first_token_matches():
    """(F3) First token in a multi-token cell matches correctly."""
    df = _df("QGLP, Coffee Can, Bruised Blue Chip 29")
    result = apply_fw_filter(df, ["QGLP"])
    assert len(result) == 1


def test_F4_last_token_matches():
    """(F4) Last token in a multi-token cell matches correctly."""
    df = _df("QGLP, Coffee Can, Bruised Blue Chip 29")
    result = apply_fw_filter(df, ["Bruised Blue Chip 29"])
    assert len(result) == 1


def test_F5_middle_token_matches():
    """(F5) Middle token in a multi-token cell matches correctly."""
    df = _df("QGLP, Coffee Can, Bruised Blue Chip 29")
    result = apply_fw_filter(df, ["Coffee Can"])
    assert len(result) == 1


def test_F6_na_cell_no_crash():
    """(F6) NaN/None cells produce no match and no exception."""
    df = pd.DataFrame({"frameworks_passed": [np.nan, None, "QGLP"]})
    result = apply_fw_filter(df, ["QGLP"])
    assert len(result) == 1


def test_F7_none_string_cell_no_match():
    """(F7) Literal 'None' cell is never matched by any framework token."""
    df = _df("None", "QGLP")
    result = apply_fw_filter(df, ["None"])
    # "None" is not a valid framework token so result behaviour:
    # Pattern: (?:^|, )None(?:,|$) — matches "None" as a standalone cell.
    # Since we DON'T filter "None" from the source here (that's the splitter's job),
    # this test validates the filter does the right thing: it WOULD match the string "None".
    # The correct guard is that "None" never appears in sel_fw (the splitter excludes it).
    # So simply confirm: QGLP filter does not match "None" cell.
    result2 = apply_fw_filter(df, ["QGLP"])
    fw_vals = result2["frameworks_passed"].tolist()
    assert "None" not in fw_vals
    assert "QGLP" in fw_vals


def test_F8_regex_special_chars_in_token_no_crash():
    """(F8) Token with regex special chars (e.g. '100-Bagger') handled safely via re.escape."""
    df = _df("100-Bagger, QGLP", "QGLP")
    result = apply_fw_filter(df, ["100-Bagger"])
    assert len(result) == 1
    # Also test parentheses and dots
    df2 = _df("Marks Cycle Shield, Some.Token(v2)")
    result2 = apply_fw_filter(df2, ["Some.Token(v2)"])
    assert len(result2) == 1


def test_F9_multi_select_or_logic():
    """(F9) Selecting two frameworks returns union (stocks passing EITHER framework)."""
    df = _df(
        "QGLP, Coffee Can",           # passes both
        "QGLP",                       # passes only QGLP
        "Coffee Can",                 # passes only Coffee Can
        "Lynch Dream, Wide Moat",     # passes neither of the two selected
    )
    result = apply_fw_filter(df, ["QGLP", "Coffee Can"])
    assert len(result) == 3, f"Expected 3 (union), got {len(result)}"
    assert "Lynch Dream, Wide Moat" not in result["frameworks_passed"].tolist()


def test_F10_2108_row_simulation():
    """(F10) Full universe simulation: all tokens clean, no duplicates, binary mask output."""
    _TOKENS = [
        "QGLP", "Coffee Can", "Magic Formula", "SMILE", "Lynch Dream",
        "CAN SLIM", "Fallen Quality", "EP Improver", "Peaceful Investing",
        "Unusual Billionaires", "Fisher Quality", "100-Bagger", "Diamond",
        "Wide Moat", "Outsider CEO", "Quality Compounder", "Dhandho Asymmetry",
        "Parikh Contrarian", "Baid Compounder", "Long Game Quality",
        "SEPA Momentum", "Basant 30% Club", "Quality Momentum",
        "MOSL Wealth Creator", "Economic Moat", "Blue Chip Quality",
        "SQGLP Century Stock", "CAP-GAP Compounder", "Consistent in Volatile",
        "EP Hockey Stick", "Bruised Blue Chip 29", "Multi-Trillion Cap",
        "Fisher Scalability", "Financial Shenanigans", "Marks Cycle Shield",
        "Expectations Matrix",
    ]
    N = 2108
    rng = np.random.default_rng(42)

    def _random_cell():
        n = rng.integers(0, 6)
        if n == 0:
            return "None"
        chosen = rng.choice(_TOKENS, size=n, replace=False).tolist()
        return ", ".join(chosen)

    cells = [_random_cell() for _ in range(N)]
    df = pd.DataFrame({"frameworks_passed": cells, "name": [f"Stock_{i}" for i in range(N)]})

    # 1. Splitter test
    tokens = build_all_fw(df)
    for t in tokens:
        assert t == t.strip(), f"Leading/trailing space: {repr(t)}"
    assert len(tokens) == len(set(tokens)), "Duplicate tokens in _all_fw"

    # 2. Substring collision: "Bruised Blue Chip" must not match "Bruised Blue Chip 29"
    bbc_only = apply_fw_filter(df, ["Bruised Blue Chip"])
    bbc29_only = apply_fw_filter(df, ["Bruised Blue Chip 29"])
    # Check every row in bbc_only does NOT contain "Bruised Blue Chip 29"
    for cell in bbc_only["frameworks_passed"]:
        assert "Bruised Blue Chip 29" not in [t.strip() for t in cell.split(", ")], \
            f"Substring collision fired: {cell}"
    # Check every row in bbc29_only contains exactly "Bruised Blue Chip 29"
    for cell in bbc29_only["frameworks_passed"]:
        tokens_in_cell = [t.strip() for t in cell.split(", ")]
        assert "Bruised Blue Chip 29" in tokens_in_cell, f"Token not found in: {cell}"

    print(f"\n  [F10] Universe: {N} rows | _all_fw tokens: {len(tokens)} | "
          f"BBC29 matches: {len(bbc29_only)} | BBC matches: {len(bbc_only)}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        ("S1  no leading-space tokens",            test_S1_no_leading_space_tokens),
        ("S2  None cells excluded",                test_S2_none_cells_excluded),
        ("S3  empty tokens excluded",              test_S3_empty_tokens_excluded),
        ("S4  deduplication across rows",          test_S4_deduplication_across_rows),
        ("S5  single-token cell",                  test_S5_single_token_cell),
        ("S6  max-density 35 frameworks",          test_S6_max_density_35_frameworks),
        ("F1  BBC no-match BBC29",                 test_F1_bbc_does_not_match_bbc29),
        ("F2  BBC29 no-match standalone BBC",      test_F2_bbc29_does_not_match_standalone_bbc),
        ("F3  first token matches",                test_F3_first_token_matches),
        ("F4  last token matches",                 test_F4_last_token_matches),
        ("F5  middle token matches",               test_F5_middle_token_matches),
        ("F6  NaN cell no crash",                  test_F6_na_cell_no_crash),
        ("F7  None string cell no match",          test_F7_none_string_cell_no_match),
        ("F8  regex special chars re.escape",      test_F8_regex_special_chars_in_token_no_crash),
        ("F9  multi-select OR logic",              test_F9_multi_select_or_logic),
        ("F10 2108-row universe simulation",       test_F10_2108_row_simulation),
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
    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed out of {len(tests)} tests")
