"""Full-universe CSV export for the sidebar download button. The pure core (_to_csv_bytes) is
unit-tested; the cached entry (scored_universe_csv) is what app.py calls."""
import streamlit as st


def _to_csv_bytes(df) -> bytes:
    """Full frame → UTF-8-with-BOM CSV bytes. The BOM (utf-8-sig) makes Excel render the emoji/unicode
    column labels and Indian stock names correctly instead of mojibake. NaN → empty field (to_csv
    default na_rep=''), so no literal 'nan'/'None' leaks. Pure — no st.*; unit-tested."""
    return df.to_csv(index=False).encode("utf-8-sig")


@st.cache_data(show_spinner=False)
def scored_universe_csv(score_key: str, _df) -> bytes:
    """Cached on the cheap `score_key` (the existing scoring cache key — same key ⟺ same scored df),
    so to_csv runs ONCE per data/profile load, not on every sidebar rerun. `_df`'s leading underscore
    tells st.cache_data NOT to hash the 2107×~700 frame each call (Streamlit skips underscore args)."""
    return _to_csv_bytes(_df)
