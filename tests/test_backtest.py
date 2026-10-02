from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from faa_oe import backtest as bt


def _quarters(n: int, start: date = date(2010, 1, 1)) -> list[date]:
    return pl.date_range(start, date(start.year + n // 4 + 1, 1, 1), "1q", eager=True)[:n].to_list()


def _lagged(n: int = 48, lead: int = 3, noise: float = 0.2, seed: int = 1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = np.full(n, np.nan)
    y[lead:] = x[:-lead] + noise * rng.normal(size=n - lead)
    return pl.DataFrame({"period": _quarters(n), "x": x, "y": y}).with_columns(pl.col("y").fill_nan(None))


def test_lead_lag_recovers_known_lead():
    ll = bt.lead_lag(_lagged(), transform="yoy")
    best = ll.sort("corr", descending=True).row(0, named=True)
    assert best["lead"] == 3
    assert best["corr"] > 0.9 and best["ci_lo"] > 0.7
    other = ll.filter(pl.col("lead") != 3)
    assert (other["ci_lo"] < 0).all()


def test_nowcast_beats_naive_at_true_lead_only():
    d = _lagged()
    good = bt.score(bt.nowcast(d, 3))
    bad = bt.score(bt.nowcast(d, 1))
    assert good["n_oos"] >= 30
    assert good["ratio_vs_last"] < 0.4 and good["ratio_vs_last_year"] < 0.4
    assert bad["ratio_vs_last_year"] > 0.8


def test_nowcast_uses_no_future_kpi():
    d = _lagged()
    base = bt.nowcast(d, 3)
    shocked = bt.nowcast(d.with_columns(y=pl.when(pl.col("period") == d["period"][-1]).then(1e6).otherwise(pl.col("y"))), 3)
    cols = ["signal", "ar1", "ar1_signal"]
    assert base.head(base.height - 1).select(cols).equals(shocked.head(shocked.height - 1).select(cols))


def test_transforms():
    q = _quarters(9)
    sig = pl.DataFrame({"period": q, "value": [9.0, 9, 9, 9, 19, 19, 19, 19, 9]})
    kpi = pl.DataFrame({"period": q, "value": [5.0, 5, 5, 5, 6, 6, 6, 6, 6]})
    d = bt.aligned(sig, kpi, "pct")
    assert d["x"].to_list()[4] == np.log(20) - np.log(10)
    assert d["y"].to_list()[4:] == [1.0, 1.0, 1.0, 1.0, 0.0]
    d2 = bt.aligned(sig, kpi, "usd_m")
    assert abs(d2["y"][4] - np.log(6 / 5)) < 1e-12


def test_run_end_to_end_on_synthetic_panel():
    q = _quarters(40, date(2015, 1, 1))
    rng = np.random.default_rng(3)
    counts = np.exp(np.cumsum(rng.normal(0, 0.15, 40)) + 5)
    panel = pl.DataFrame({
        "period": q, "basis": "pit", "level": "group", "name": "private_and_carriers", "category": "all",
        "kind": "new_or_unknown", "n": counts.round().astype(np.uint32), "n_hw": counts, "n_projects": counts.astype(np.uint32),
        "complete": True,
    })
    kpi_vals = np.concatenate([[np.nan] * 4, counts[:-4] * 0.5])
    truth_df = pl.DataFrame({"period": q, "entity": "SBAC", "metric": "towers_built_us", "value": kpi_vals,
                             "unit": "towers", "source": "x"}).drop_nulls().filter(pl.col("value").is_not_nan())
    pair = bt.Pairing("t", "a", "private_and_carriers", "new_or_unknown", "SBAC", "towers_built_us", 4)
    res = bt.run(panel, truth_df, pairings=(pair, bt.Pairing("u", "a", "all", "new_or_unknown", "X", "missing", 1)))
    assert res.skipped and res.skipped[0].startswith("u:")
    row = res.oos.row(0, named=True)
    assert row["best_lead_insample"] == 4 and row["corr_at_lead"] > 0.95
    assert row["ratio_vs_last"] < 0.2
