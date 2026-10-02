from __future__ import annotations

from datetime import date

import pytest

from faa_oe.client import FAAClient, Window, fetch_adaptive


@pytest.mark.live
def test_live_tx_week_matches_probe():
    with FAAClient() as client:
        out = list(fetch_adaptive(client, Window("OE", None, date(2026, 9, 1), date(2026, 9, 7), state="TX")))
    n = sum(f.n_rows for f in out)
    assert 625 * 0.95 <= n <= 625 * 1.05
