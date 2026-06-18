"""
test_cache_contract.py
======================
Pins the Tier-1 data-cache key contract in app.py get_clean_data.

Streamlit EXCLUDES underscore-prefixed parameters from a @st.cache_data key
(docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_data: "prepend the parameter
name with an underscore ... it will not be used for caching"). So:
  • the file signature MUST be NON-underscored (`file_signature`) → hashed → a swapped upload busts
    the cache (else uploads return stale data until manual Clear Cache), and
  • the raw upload streams MUST stay underscored (`_uploaded_dict`) → skipped (they are unhashable;
    hashing them raises UnhashableParamError).
Streamlit's cache runtime can't be unit-tested in pytest, but the signature that DRIVES it can —
this static contract catches a regression in either direction.
"""
import os
import re


def _signature() -> str:
    path = os.path.join(os.path.dirname(__file__), "..", "app.py")
    with open(path, encoding="utf-8") as f:
        src = f.read()
    m = re.search(r"def get_clean_data\(([^)]*)\)", src)
    assert m, "get_clean_data signature not found in app.py"
    return m.group(1)


def test_upload_cache_key_param_is_hashable_not_underscored():
    """file_signature must NOT be underscored, or Streamlit drops it from the cache key and swapped
    uploads return stale data."""
    sig = _signature()
    assert "file_signature" in sig and "_file_signature" not in sig, (
        f"file_signature must be non-underscored so Streamlit hashes it (uploads go stale otherwise); "
        f"got signature: ({sig})"
    )


def test_uploaded_dict_stays_underscored_for_unhashable_streams():
    """Raw file streams are unhashable -> _uploaded_dict must keep its underscore (removing it would
    make Streamlit try to hash the streams -> UnhashableParamError crash)."""
    assert "_uploaded_dict" in _signature(), (
        "_uploaded_dict must stay underscored (raw upload streams are unhashable)"
    )
