"""
Core Execution Engines
======================
Exposes the primary data, scoring, and forensic engines.
"""

from .data_engine import fetch_and_clean_data
from .forensic_engine import run_forensic_analysis, compute_forensic_signals, apply_forensic_penalty
from .scoring_engine import run_full_scoring
from .verdict_engine import compute_verdict

__all__ = [
    "fetch_and_clean_data",
    "run_forensic_analysis",
    "compute_forensic_signals",
    "apply_forensic_penalty",
    "run_full_scoring",
    "compute_verdict",
]
