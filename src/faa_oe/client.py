from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, replace
from datetime import date, timedelta
from typing import Iterator, Protocol

import httpx

from faa_oe.parse import count_cases

log = logging.getLogger(__name__)

BASE_URL = "https://oeaaa.faa.gov/oeaaa/services"
USER_AGENT = "faa-oe-research/0.1 (academic alt-data research; python-httpx)"
ROW_CAP = 8000
CASE_TYPES = ("OE", "NRA")
# WTE/WTW are the wind-turbine "regions": wind cases get their own ASN series there, not under the geographic regions.
REGIONS = ("AAL", "ACE", "AEA", "AGL", "ANE", "ANM", "ASO", "ASW", "AWP", "WTE", "WTW")
STATES = (
    "AK AL AR AZ CA CO CT DC DE FL GA HI IA ID IL IN KS KY LA MA MD ME MI MN MO MS MT NC ND NE NH NJ NM NV NY "
    "OH OK OR PA RI SC SD TN TX UT VA VT WA WI WV WY PR VI GU AS MP"
).split()


@dataclass(frozen=True)
class Window:
    """Inclusive dateEntered range [start, end] for one ASN year (the URL path year).

    `asn_year` defaults to the dates' year; set it to look for cases whose ASN year differs from the year they
    were entered (rare, e.g. 2021-AAL-6-NRA created 2020-11-03).
    """

    case_type: str
    region: str | None
    start: date
    end: date
    state: str | None = None
    asn_year: int | None = None

    def __post_init__(self) -> None:
        if self.start.year != self.end.year:
            raise ValueError(f"window crosses a year boundary: {self.start}..{self.end}")
        if self.start > self.end:
            raise ValueError(f"empty window: {self.start}..{self.end}")

    @property
    def year(self) -> int:
        return self.asn_year if self.asn_year is not None else self.start.year

    def params(self) -> dict[str, str]:
        # The API compares dateEnteredEnd as an exclusive midnight timestamp: end=2025-01-31 drops every case
        # entered on the 31st (and NRA createdDate after 00:00:00). Ask for end + 1 day to get an inclusive end.
        p = {
            "dateEnteredStart": self.start.isoformat(),
            "dateEnteredEnd": (self.end + timedelta(days=1)).isoformat(),
        }
        if self.region:
            p["region"] = self.region
        if self.state:
            p["state"] = self.state
        return p

    def __str__(self) -> str:
        parts = [self.case_type, str(self.year), self.region or "ALL", f"{self.start}..{self.end}"]
        return " ".join(parts + ([self.state] if self.state else []))


class CaseListSource(Protocol):
    def case_list(self, window: Window) -> bytes: ...


class FAAClient:
    def __init__(
        self,
        base_url: str = BASE_URL,
        min_interval: float = 1.0,
        timeout: float = 90.0,
        max_retries: int = 5,
        backoff: float = 2.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._http = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT, "Accept": "application/xml"},
            transport=transport,
            follow_redirects=True,
        )
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.backoff = backoff
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> FAAClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _throttle(self) -> None:
        # Spaces out request *starts* across threads, so a few slow concurrent requests still average <= 1 req/s.
        with self._lock:
            now = time.monotonic()
            wait = self._next_slot - now
            self._next_slot = max(now, self._next_slot) + self.min_interval
        if wait > 0:
            time.sleep(wait)

    def get(self, path: str, params: dict[str, str] | None = None) -> bytes:
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                r = self._http.get(path, params=params)
                if r.status_code < 500 and r.status_code != 429:
                    r.raise_for_status()
                    return r.content
                err: Exception = httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                err = e
            if attempt == self.max_retries:
                raise err
            delay = self.backoff * 2**attempt
            log.warning("GET %s %s failed (%s); retry %d in %.0fs", path, params, err, attempt + 1, delay)
            time.sleep(delay)
        raise AssertionError("unreachable")

    def case_list(self, window: Window) -> bytes:
        return self.get(f"/caseList/{window.case_type}/{window.year}", window.params())

    def case(self, asn: str) -> bytes:
        return self.get(f"/case/{asn}")


@dataclass(frozen=True)
class Fetched:
    window: Window
    xml: bytes
    n_rows: int
    capped: bool = False


def fetch_adaptive(
    source: CaseListSource,
    window: Window,
    cap: int = ROW_CAP,
    states: tuple[str, ...] | list[str] = STATES,
) -> Iterator[Fetched]:
    """Fetch a window, bisecting by date and then by state whenever the response hits the row cap."""
    xml = source.case_list(window)
    n = count_cases(xml)
    if n < cap:
        yield Fetched(window, xml, n)
        return
    if window.start < window.end:
        mid = window.start + timedelta(days=(window.end - window.start).days // 2)
        log.info("%s returned %d rows (cap %d); bisecting by date", window, n, cap)
        yield from fetch_adaptive(source, replace(window, end=mid), cap, states)
        yield from fetch_adaptive(source, replace(window, start=mid + timedelta(days=1)), cap, states)
    elif window.state is None:
        # Gotcha: cases whose nearestState is missing or outside STATES are lost on this path.
        log.warning("%s still capped at a single day; splitting by state", window)
        for s in states:
            yield from fetch_adaptive(source, replace(window, state=s), cap, states)
    else:
        log.warning("%s still capped at %d rows after day+state split; data truncated", window, n)
        yield Fetched(window, xml, n, capped=True)


def month_windows(case_type: str, region: str | None, start: date, end: date) -> list[Window]:
    """Calendar-month windows covering [start, end]; never crosses a year boundary."""
    out: list[Window] = []
    cur = start
    while cur <= end:
        nxt = date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)
        out.append(Window(case_type, region, cur, min(end, nxt - timedelta(days=1))))
        cur = nxt
    return out
