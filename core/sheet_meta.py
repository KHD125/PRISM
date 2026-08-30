"""How fresh is the data? — read from the spreadsheet's own name.

The ingestion pipeline renames the sheet after every load to the trading session
its numbers came from, e.g. "PRISM 2026-08-28 Fri". Google returns that name in
the export endpoint's Content-Disposition header, so the app can report data
freshness without a Drive API, credentials, or an extra column in the sheet.

DELIBERATELY SEPARATE FROM data_engine.load_all_datasets. That loader is locked
(CLAUDE.md §0) because a past change to it silently nulled 5 of 6 tabs. A
cosmetic freshness label must never be able to break data loading, so this
module opens its own one-byte request and shares nothing with the load path.

Everything here fails soft: a network hiccup or an unexpected sheet name costs
you a label, never a render.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional
from urllib.parse import unquote

IST = timezone(timedelta(hours=5, minutes=30))

# The pipeline writes "<name> YYYY-MM-DD Ddd"; we only need the ISO date, which
# is why the pipeline uses ISO in the first place -- it is unambiguous to parse
# and to read.
_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")

# filename*=UTF-8''... (RFC 5987) is preferred when present; filename="..." is
# the fallback every server sends.
_FN_STAR_RE = re.compile(r"filename\*\s*=\s*[^']*''([^;]+)", re.I)
_FN_RE = re.compile(r'filename\s*=\s*"([^"]+)"', re.I)

# Market close, Asia/Kolkata. A session is only "available" once it has closed.
_CLOSE_HOUR, _CLOSE_MIN = 15, 30
# The pipeline's daily run time. Data can only be as fresh as the last run.
_RUN_HOUR = 6


@dataclass(frozen=True)
class Freshness:
    """What the UI needs to render, already decided."""
    title: Optional[str]          # raw sheet name, for the tooltip
    data_date: Optional[date]     # session the data represents
    label: str                    # "Fri, 28 Aug 2026" | "Prism" | "Local CSVs"
    day: str                      # "Fri" ("" when unknown)
    status: str                   # "current" | "1 session behind" | ... | "unknown"
    sessions_behind: Optional[int]
    tone: str                     # "green" | "gold" | "red" | "muted"

    @property
    def is_known(self) -> bool:
        return self.data_date is not None


def _parse_title(disposition: str) -> Optional[str]:
    """Sheet name out of a Content-Disposition header, minus the .xlsx."""
    if not disposition:
        return None
    m = _FN_STAR_RE.search(disposition) or _FN_RE.search(disposition)
    if not m:
        return None
    name = unquote(m.group(1)).strip()
    return re.sub(r"\.xlsx$", "", name, flags=re.I) or None


def fetch_sheet_title(sheet_id: str, timeout: float = 8.0) -> Optional[str]:
    """The spreadsheet's name, via a one-byte range request.

    Asking for bytes 0-0 means Google sends headers and essentially no body, so
    this costs a round trip rather than a workbook download. Returns None on any
    failure -- the caller renders without a date rather than showing an error.
    """
    if not sheet_id:
        return None
    try:
        import requests

        r = requests.get(
            f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx",
            headers={"Range": "bytes=0-0"},
            timeout=timeout,
            allow_redirects=True,
            stream=True,
        )
        try:
            if r.status_code >= 400:
                return None
            return _parse_title(r.headers.get("Content-Disposition", ""))
        finally:
            r.close()
    except Exception:
        return None


def parse_data_date(title: Optional[str]) -> Optional[date]:
    """The ISO date embedded in the sheet name, if there is one."""
    if not title:
        return None
    m = _DATE_RE.search(title)
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None            # e.g. 2026-13-45 in a hand-edited name


def _prev_weekday(d: date) -> date:
    d -= timedelta(days=1)
    while d.weekday() >= 5:    # 5=Sat, 6=Sun
        d -= timedelta(days=1)
    return d


def expected_session(now: Optional[datetime] = None) -> date:
    """The most recent session the pipeline could already have captured.

    Not simply "the last closed session": the job runs at 06:00 IST, before the
    15:30 close, so today's close cannot appear in the sheet until tomorrow's
    run. Comparing against the last CLOSED session would therefore paint the
    card amber every weekday afternoon during completely normal operation.

    So: take the most recent 06:00 boundary, and the last session that had
    closed before it.
    """
    now = now or datetime.now(IST)
    # The day whose 06:00 run is the most recent one to have happened.
    boundary = now.date() if now.hour >= _RUN_HOUR else now.date() - timedelta(days=1)
    # STRICTLY before that day: the 06:00 run precedes the 15:30 close, so the
    # boundary day's own session is not in the sheet yet.
    return _prev_weekday(boundary)


def sessions_between(start: date, end: date) -> int:
    """Trading sessions after `start` up to and including `end` (0 if not behind)."""
    if end <= start:
        return 0
    n, d = 0, start
    while d < end:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n += 1
    return n


def describe(sheet_id: Optional[str], data_source: str = "sheet",
             now: Optional[datetime] = None, title: Optional[str] = None) -> Freshness:
    """Everything the UI needs, with every failure already handled."""
    now = now or datetime.now(IST)

    if data_source != "sheet":
        d = _newest_csv_date()
        if d is None:
            return Freshness(None, None, "Local CSVs", "", "unknown", None, "muted")
        return _grade(None, d, now, fmt_source="file")

    if title is None:
        title = fetch_sheet_title(sheet_id or "")
    d = parse_data_date(title)
    if d is None:
        # A sheet that exists but is not named by the pipeline: show its real
        # name rather than inventing a date.
        return Freshness(title, None, title or "Unknown", "", "unknown", None, "muted")
    return _grade(title, d, now)


def _grade(title: Optional[str], d: date, now: datetime, fmt_source: str = "sheet") -> Freshness:
    behind = sessions_between(d, expected_session(now))
    if behind <= 0:
        status, tone = "current", "green"
    elif behind == 1:
        status, tone = "1 session behind", "gold"
    else:
        status, tone = f"{behind} sessions behind", "red"
    return Freshness(
        title=title,
        data_date=d,
        label=d.strftime("%a, %d %b %Y"),
        day=d.strftime("%a"),
        status=status,
        sessions_behind=behind,
        tone=tone,
    )


def _newest_csv_date() -> Optional[date]:
    """Local-CSV mode: the most recently written of the six source files."""
    try:
        from config import CSV_FILES

        times = [os.path.getmtime(p) for p in CSV_FILES.values() if os.path.exists(p)]
        if not times:
            return None
        return datetime.fromtimestamp(max(times), IST).date()
    except Exception:
        return None
