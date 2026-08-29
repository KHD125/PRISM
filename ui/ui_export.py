"""Full-universe CSV export for the sidebar download button. The pure core (_to_csv_bytes) and the
cache-key digest (universe_signature) are unit-tested; the cached entry (scored_universe_csv) is
what app.py calls."""
import hashlib

import streamlit as st


def _to_csv_bytes(df) -> bytes:
    """Full frame → UTF-8-with-BOM CSV bytes. The BOM (utf-8-sig) makes Excel render the emoji/unicode
    column labels and Indian stock names correctly instead of mojibake. NaN → empty field (to_csv
    default na_rep=''), so no literal 'nan'/'None' leaks. Pure — no st.*; unit-tested."""
    return df.to_csv(index=False).encode("utf-8-sig")


def universe_signature(names) -> str:
    """Exact, order-sensitive identity digest of a filtered frame's row set — md5 over the joined
    stock names (unique + never-NaN in this universe; ~2 ms on 2,117 rows).

    WHY EXACT (2026-08-29 audit): the previous cache key hashed content as
    `len|composite_sum:.2f` — lossy. On live data 768 of 2,117 stocks share a 2dp-rounded
    composite with another stock, and two real one-stock filter states (Shilchar vs Kothari, both
    90.00) produced the identical key, so the second download silently served the FIRST stock's
    CSV. Different row sets must never share a cache entry; identical ones still hit it."""
    return hashlib.md5("|".join(names.astype(str)).encode("utf-8")).hexdigest()


@st.cache_data(show_spinner=False, max_entries=8)
def scored_universe_csv(score_key: str, _df) -> bytes:
    """Cached on `score_key` — now `<scoring key>|<universe_signature digest>` from app.py, which
    identifies the exact (data, mode, profile, row set): to_csv runs ONCE per distinct filter
    state, not on every sidebar rerun. `_df`'s leading underscore tells st.cache_data NOT to hash
    the 2117×~700 frame each call (Streamlit skips underscore args). max_entries=8 caps the cache:
    each full-universe entry is ~12.3 MB (measured), and without a bound every filter state ever
    visited stays resident for the life of the Cloud container."""
    return _to_csv_bytes(_df)
