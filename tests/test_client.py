from __future__ import annotations

import logging
from datetime import date

import httpx
import pytest

from conftest import FakeSource
from faa_oe.client import FAAClient, Window, fetch_adaptive, month_windows


def _window(start: date, end: date) -> Window:
    return Window("OE", "ASW", start, end)


def test_no_split_under_cap():
    src = FakeSource(per_day=3)
    out = list(fetch_adaptive(src, _window(date(2025, 1, 1), date(2025, 1, 31)), cap=100))
    assert len(out) == 1 and out[0].n_rows == 93 and not out[0].capped


def test_bisects_by_date_until_under_cap():
    src = FakeSource(per_day=10)
    w = _window(date(2025, 1, 1), date(2025, 1, 31))
    out = list(fetch_adaptive(src, w, cap=100))
    assert all(f.n_rows < 100 for f in out)
    assert sum(f.n_rows for f in out) == 310
    spans = sorted((f.window.start, f.window.end) for f in out)
    assert spans[0][0] == w.start and spans[-1][1] == w.end
    assert all(a[1].toordinal() + 1 == b[0].toordinal() for a, b in zip(spans, spans[1:]))


def test_single_day_splits_by_state():
    src = FakeSource(per_day=200, per_state=5)
    out = list(fetch_adaptive(src, _window(date(2025, 1, 1), date(2025, 1, 2)), cap=100, states=["TX", "OK"]))
    assert sorted((f.window.start.day, f.window.state) for f in out) == [(1, "OK"), (1, "TX"), (2, "OK"), (2, "TX")]
    assert sum(f.n_rows for f in out) == 20


def test_still_capped_logs_warning(caplog):
    src = FakeSource(per_day=200, per_state=150)
    with caplog.at_level(logging.WARNING):
        out = list(fetch_adaptive(src, _window(date(2025, 1, 1), date(2025, 1, 1)), cap=100, states=["TX"]))
    assert len(out) == 1 and out[0].capped and out[0].n_rows == 150
    assert any("still capped" in r.message and "truncated" in r.message for r in caplog.records)


def test_window_rejects_year_crossing():
    with pytest.raises(ValueError):
        Window("OE", "ASW", date(2024, 12, 20), date(2025, 1, 5))


def test_asn_year_overrides_path_year():
    w = Window("NRA", "ASO", date(2024, 1, 1), date(2024, 12, 31), asn_year=2025)
    assert w.year == 2025
    assert w.params()["dateEnteredEnd"] == "2025-01-01"


def test_month_windows_split_at_year_boundary():
    ws = month_windows("OE", "ASW", date(2024, 11, 15), date(2025, 2, 3))
    assert [(w.start, w.end) for w in ws] == [
        (date(2024, 11, 15), date(2024, 11, 30)),
        (date(2024, 12, 1), date(2024, 12, 31)),
        (date(2025, 1, 1), date(2025, 1, 31)),
        (date(2025, 2, 1), date(2025, 2, 3)),
    ]


def _client(handler) -> FAAClient:
    return FAAClient(min_interval=0, backoff=0, max_retries=3, transport=httpx.MockTransport(handler))


def test_retries_on_5xx_then_succeeds():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(503) if len(calls) < 3 else httpx.Response(200, content=b"<caseList/>")

    w = _window(date(2025, 1, 1), date(2025, 1, 31))
    assert _client(handler).case_list(w) == b"<caseList/>"
    assert len(calls) == 3
    assert calls[-1].url.path == "/oeaaa/services/caseList/OE/2025"
    assert calls[-1].url.params["region"] == "ASW"
    assert calls[-1].url.params["dateEnteredStart"] == "2025-01-01"
    assert calls[-1].url.params["dateEnteredEnd"] == "2025-02-01"  # API end bound is exclusive
    assert "faa-oe-research" in calls[-1].headers["user-agent"]


def test_retries_on_timeout_and_gives_up():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(httpx.ReadTimeout):
        _client(handler).get("/case/x")
    assert len(calls) == 4


def test_no_retry_on_4xx():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(404)

    with pytest.raises(httpx.HTTPStatusError):
        _client(handler).get("/case/x")
    assert len(calls) == 1
