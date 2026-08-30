"""Contract tests for the data-freshness label (core/sheet_meta.py).

The label's whole job is to tell the truth about how old the sheet is. The
subtle part is WHICH session counts as current: the pipeline runs at 06:00 IST,
before the 15:30 close, so today's close cannot be in the sheet until tomorrow's
run. Grading against the last CLOSED session would show "behind" every weekday
afternoon during entirely normal operation -- the pins below fix that boundary.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "core")):
    if p not in sys.path:
        sys.path.insert(0, p)

from core import sheet_meta as sm  # noqa: E402

IST = sm.IST


def _at(y, m, d, hh=10, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=IST)


# ---------------------------------------------------------------- title parsing
@pytest.mark.parametrize("header,expected", [
    ('attachment; filename="PRISM 2026-08-28 Fri.xlsx"', "PRISM 2026-08-28 Fri"),
    ("attachment; filename*=UTF-8''PRISM%202026-08-28%20Fri.xlsx", "PRISM 2026-08-28 Fri"),
    # filename* wins when both are present -- it is the encoding-aware form
    ('attachment; filename="fallback.xlsx"; filename*=UTF-8\'\'PRISM%202026-08-28%20Fri.xlsx',
     "PRISM 2026-08-28 Fri"),
    ('attachment; filename="Prism.xlsx"', "Prism"),
    ("", None),
    ("attachment", None),
])
def test_title_parsed_from_content_disposition(header, expected):
    assert sm._parse_title(header) == expected


@pytest.mark.parametrize("title,expected", [
    ("PRISM 2026-08-28 Fri", date(2026, 8, 28)),
    ("PRISM 2026-01-01 Thu", date(2026, 1, 1)),
    ("anything 2026-12-31 whatever", date(2026, 12, 31)),
    ("Prism", None),
    ("PRISM 28-08-2026", None),      # not ISO -> refuse rather than guess
    ("PRISM 2026-13-45 Fri", None),  # syntactically ISO, not a real date
    (None, None),
    ("", None),
])
def test_data_date_parsed_from_title(title, expected):
    assert sm.parse_data_date(title) == expected


# ------------------------------------------------------------- session boundary
def test_expected_session_is_the_previous_weekday_after_the_run():
    # Tue 06:00 has passed -> the run captured Monday's close.
    assert sm.expected_session(_at(2026, 9, 1, 10)) == date(2026, 8, 31)


def test_afternoon_does_not_make_the_sheet_look_stale():
    """The regression this boundary exists to prevent.

    At 16:00 Tuesday, Tuesday's session HAS closed -- but the pipeline will not
    fetch it until Wednesday 06:00. Grading against the last closed session
    would show 'behind' here, every single weekday afternoon.
    """
    morning = sm.expected_session(_at(2026, 9, 1, 10))
    afternoon = sm.expected_session(_at(2026, 9, 1, 16))
    assert morning == afternoon == date(2026, 8, 31)


def test_before_the_run_still_points_at_the_older_session():
    # 05:00 Tuesday: today's 06:00 job has not run, so Friday is still the best available.
    assert sm.expected_session(_at(2026, 9, 1, 5)) == date(2026, 8, 28)


@pytest.mark.parametrize("when,expected", [
    (_at(2026, 8, 29, 10), date(2026, 8, 28)),   # Sat -> Friday
    (_at(2026, 8, 30, 10), date(2026, 8, 28)),   # Sun -> Friday
    (_at(2026, 8, 31, 10), date(2026, 8, 28)),   # Mon -> Friday (weekend skipped)
])
def test_weekends_resolve_back_to_friday(when, expected):
    assert sm.expected_session(when) == expected


@pytest.mark.parametrize("start,end,n", [
    (date(2026, 8, 28), date(2026, 8, 28), 0),
    (date(2026, 8, 28), date(2026, 8, 31), 1),   # Fri -> Mon is ONE session, not 3 days
    (date(2026, 8, 28), date(2026, 9, 1), 2),
    (date(2026, 8, 31), date(2026, 8, 28), 0),   # never negative
])
def test_sessions_counted_not_calendar_days(start, end, n):
    assert sm.sessions_between(start, end) == n


# --------------------------------------------------------------------- grading
def test_current_data_reads_green():
    f = sm.describe(None, "sheet", now=_at(2026, 9, 1, 10), title="PRISM 2026-08-31 Mon")
    assert (f.status, f.tone, f.day) == ("current", "green", "Mon")
    assert f.label == "Mon, 31 Aug 2026"
    assert f.is_known


def test_one_session_behind_reads_amber():
    f = sm.describe(None, "sheet", now=_at(2026, 9, 1, 10), title="PRISM 2026-08-28 Fri")
    assert (f.status, f.tone, f.sessions_behind) == ("1 session behind", "gold", 1)


def test_several_sessions_behind_reads_red():
    f = sm.describe(None, "sheet", now=_at(2026, 9, 4, 10), title="PRISM 2026-08-28 Fri")
    assert f.tone == "red" and f.sessions_behind >= 2
    assert "sessions behind" in f.status


def test_monday_morning_on_friday_data_is_current_not_three_days_old():
    """Calendar arithmetic would call this 3 days stale and paint it red on every
    Monday. It is the freshest data that exists."""
    f = sm.describe(None, "sheet", now=_at(2026, 8, 31, 8), title="PRISM 2026-08-28 Fri")
    assert (f.status, f.tone) == ("current", "green")


# ------------------------------------------------------------------- fail-soft
def test_unnamed_sheet_shows_its_real_name_rather_than_a_fake_date():
    f = sm.describe(None, "sheet", now=_at(2026, 9, 1), title="Prism")
    assert f.data_date is None and f.is_known is False
    assert f.label == "Prism" and f.tone == "muted"


def test_missing_title_never_raises():
    f = sm.describe(None, "sheet", now=_at(2026, 9, 1), title=None)
    assert f.data_date is None and f.tone == "muted"


def test_fetch_returns_none_instead_of_raising(monkeypatch):
    """A network failure must cost a label, never a render."""
    import requests

    def boom(*a, **k):
        raise requests.ConnectionError("no network")

    monkeypatch.setattr(requests, "get", boom)
    assert sm.fetch_sheet_title("someid") is None


def test_empty_sheet_id_makes_no_request(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must not call out with an empty id")

    import requests
    monkeypatch.setattr(requests, "get", boom)
    assert sm.fetch_sheet_title("") is None


def test_http_error_yields_none(monkeypatch):
    import requests

    class Resp:
        status_code = 404
        headers: dict = {}

        def close(self):
            pass

    monkeypatch.setattr(requests, "get", lambda *a, **k: Resp())
    assert sm.fetch_sheet_title("someid") is None
