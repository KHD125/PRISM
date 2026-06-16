"""
test_url_parser.py
==================
Coverage for core.data_engine.extract_spreadsheet_id — the user-facing parser that turns a
pasted Google Sheets URL (or a bare ID) into the spreadsheet ID used to load the live data.
Phase-1 audit finding C7: this is on the data-entry hot path yet had zero tests. These pin its
documented behavior so a future refactor can't silently break sheet loading for every user.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.data_engine import extract_spreadsheet_id


def test_extracts_id_from_full_edit_url():
    url = "https://docs.google.com/spreadsheets/d/1AbC-dEf_GhI/edit#gid=0"
    assert extract_spreadsheet_id(url) == "1AbC-dEf_GhI"


def test_extracts_id_with_dashes_and_underscores():
    url = "https://docs.google.com/spreadsheets/d/1a-b_c-D2_3/edit?usp=sharing"
    assert extract_spreadsheet_id(url) == "1a-b_c-D2_3"


def test_bare_id_returned_unchanged():
    assert extract_spreadsheet_id("1AbC-dEf_GhI") == "1AbC-dEf_GhI"


def test_bare_id_is_stripped_of_whitespace():
    assert extract_spreadsheet_id("  1AbC_dEf  ") == "1AbC_dEf"


def test_empty_string_returns_empty():
    assert extract_spreadsheet_id("") == ""


def test_none_returns_empty():
    assert extract_spreadsheet_id(None) == ""


def test_url_without_spreadsheet_pattern_falls_back_to_input():
    """A URL that isn't a Sheets link is returned as-is (documented fallback)."""
    assert extract_spreadsheet_id("https://example.com/foo/bar") == "https://example.com/foo/bar"
