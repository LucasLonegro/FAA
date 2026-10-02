"""Filing signals vs reported KPIs: lead-lag correlations and an expanding-window out-of-sample nowcast.

Samples are small (tens of quarters), so everything is kept simple and reported with n:

- Transforms. Dollar series: YoY log difference, log(x_t) - log(x_{t-4}); counts (filings, towers / sites
  built) use log(1 + x), since US build counts can be 0.
  KPIs that are already growth rates (unit pct): 4-quarter difference in percentage points. Levels (log signal,
  raw KPI) are reported alongside for reference only; trending levels correlate spuriously.
- Lead-lag: corr(signal_{t-k}, kpi_t) for k = 0..6, with a moving-block bootstrap 95% CI (blocks of 4
  quarters over the paired observations) when n >= 12. Below that the bootstrap has too few distinct
  resamples, so the CI is Fisher-z, which ignores autocorrelation and is too narrow (`ci_method` says which).
- Nowcast: for each quarter t, OLS kpi ~ signal_{t-L} fitted only on pairs whose KPI quarter is before t, at the
  lead L fixed in PAIRINGS before any out-of-sample result was looked at. Benchmarks on the same transformed
  scale: last value (y_{t-1}), same quarter last year in levels (no YoY change, i.e. 0), AR(1), and AR(1) plus
  the signal. All models are scored on the quarters where every model has a forecast.

Pre-specified leads (from the Stage 2 filing -> ASR construction lag, median 327 days, IQR 217-475):
  new-build KPIs (towers built, sites constructed): 4 quarters (a filing in mid-quarter t is built in t+3 or
  t+4, more often t+4); carrier capex: 3 quarters (spend precedes completion); modification filings vs organic
  growth / leasing revenue: 1 quarter (amendment filed, equipment installed and billed within 1-2 quarters).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from faa_oe import features, truth
from faa_oe.paths import DataPaths

LEADS = tuple(range(7))
BLOCK = 4
N_BOOT = 2000
MIN_CORR_N = 6
MIN_BOOT_N = 12
MIN_TRAIN = 8
MIN_OOS = 4


@dataclass(frozen=True)
class Pairing:
    key: str
    family: str
    signal: str
    kind: str
    kpi_entity: str
    kpi_metric: str
    lead: int
    note: str = ""


PAIRINGS: tuple[Pairing, ...] = (
    Pairing("a1", "a", "private_and_carriers", "new_or_unknown", "SBAC", "towers_built_us", 4,
            "SBAC builds few US towers; expect weak"),
    Pairing("a2", "a", "private_and_carriers", "new_or_unknown", "SBAC", "towers_built", 4,
            "total incl. international; FAA covers US only"),
    Pairing("a3", "a", "private_and_carriers", "new_or_unknown", "AMT", "sites_constructed_usc", 4,
            "US (& Canada) segment"),
    Pairing("a4", "a", "private_and_carriers", "new_or_unknown", "AMT", "sites_constructed_total", 4,
            "mostly international builds; expect weak"),
    Pairing("a5", "a", "private_and_carriers", "new_or_unknown", truth.CARRIERS_SUM, "capex_usd_m", 3,
            "VZ+T+TMUS cash capex"),
    Pairing("a6", "a", "all", "new_or_unknown", truth.CARRIERS_SUM, "capex_usd_m", 3,
            "all new_or_unknown tower filings"),
    Pairing("b1", "b", "American Tower", "modification", "AMT", "usc_organic_tenant_billings_growth_pct", 1),
    Pairing("b2", "b", "SBA Communications", "modification", "SBAC", "domestic_site_leasing_revenue", 1),
    Pairing("b3", "b", "Crown Castle", "modification", "CCI", "organic_site_rental_billings_growth_pct", 1),
    Pairing("b4", "b", "Crown Castle", "modification", "CCI", "towers_organic_contribution_usd_m", 1),
    Pairing("b5", "b", "listed_reits", "modification", "AMT", "usc_organic_tenant_billings_growth_pct", 1,
            "all three REITs' modifications"),
    Pairing("b6", "b", "Crown Castle", "modification", "CCI", "towers_site_rental_revenues_usd_m", 1),
    Pairing("c1", "c", "vz_incl_towers_jv", "new_or_unknown", "VZ", "capex_usd_m", 3, "VZ + The Towers JV"),
    Pairing("c2", "c", "Verizon", "new_or_unknown", "VZ", "capex_usd_m", 3, "VZ sponsor only"),
)


def parse_quarter(s: str | None) -> date | None:
    return truth.quarter_start(s) if s else None


def yoy_signal(v: pl.Expr) -> pl.Expr:
    return (v + 1).log() - (v + 1).log().shift(4)


COUNT_UNITS = ("towers", "sites")


def yoy_kpi(v: pl.Expr, unit: str) -> pl.Expr:
    if unit == "pct":
        return v - v.shift(4)
    if unit in COUNT_UNITS:
        return yoy_signal(v)
    return pl.when(v > 0).then(v.log()) - pl.when(v.shift(4) > 0).then(v.shift(4).log())


def aligned(signal: pl.DataFrame, kpi: pl.DataFrame, unit: str) -> pl.DataFrame:
    """Quarterly grid with raw and transformed signal (x, x_level) and KPI (y, y_level)."""
    lo = min(signal["period"].min(), kpi["period"].min())
    hi = max(signal["period"].max(), kpi["period"].max())
    grid = pl.DataFrame({"period": pl.date_range(lo, hi, "1q", eager=True)})
    d = (
        grid.join(signal.select("period", s=pl.col("value")), on="period", how="left")
        .join(kpi.select("period", k=pl.col("value")), on="period", how="left")
        .sort("period")
    )
    return d.with_columns(
        x=yoy_signal(pl.col("s")),
        y=yoy_kpi(pl.col("k"), unit),
        x_level=(pl.col("s") + 1).log(),
        y_level=pl.col("k"),
    )


def _pairs(d: pl.DataFrame, lead: int, x: str = "x", y: str = "y") -> tuple[np.ndarray, np.ndarray]:
    xs = d[x].shift(lead).to_numpy()
    ys = d[y].to_numpy()
    ok = ~(np.isnan(xs) | np.isnan(ys))
    return xs[ok].astype(float), ys[ok].astype(float)


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def block_bootstrap_ci(x: np.ndarray, y: np.ndarray, block: int = BLOCK, n_boot: int = N_BOOT, seed: int = 0,
                       alpha: float = 0.05) -> tuple[float, float]:
    n = len(x)
    if n < MIN_CORR_N:
        return math.nan, math.nan
    b = max(1, min(block, n // 2))
    rng = np.random.default_rng(seed)
    starts = np.arange(n - b + 1)
    k = math.ceil(n / b)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = (rng.choice(starts, k)[:, None] + np.arange(b)).ravel()[:n]
        stats[i] = _corr(x[idx], y[idx])
    stats = stats[~np.isnan(stats)]
    if len(stats) < n_boot // 2:
        return math.nan, math.nan
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def fisher_ci(r: float, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n <= 3 or abs(r) >= 1:
        return math.nan, math.nan
    h = z / math.sqrt(n - 3)
    return math.tanh(math.atanh(r) - h), math.tanh(math.atanh(r) + h)


def lead_lag(d: pl.DataFrame, leads=LEADS, transform: str = "yoy", seed: int = 0) -> pl.DataFrame:
    xc, yc = ("x", "y") if transform == "yoy" else ("x_level", "y_level")
    rows = []
    for k in leads:
        x, y = _pairs(d, k, xc, yc)
        r = _corr(x, y) if len(x) >= MIN_CORR_N else math.nan
        lo = hi = math.nan
        method = None
        if not math.isnan(r):
            if len(x) >= MIN_BOOT_N:
                (lo, hi), method = block_bootstrap_ci(x, y, seed=seed + k), "block_bootstrap"
            else:
                (lo, hi), method = fisher_ci(r, len(x)), "fisher_z"
        rows.append((k, len(x), r, lo, hi, method))
    return pl.DataFrame(rows, schema={"lead": pl.Int64, "n": pl.Int64, "corr": pl.Float64, "ci_lo": pl.Float64,
                                      "ci_hi": pl.Float64, "ci_method": pl.String},
                        orient="row").with_columns(pl.col(pl.Float64).fill_nan(None))


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(np.column_stack([np.ones(len(y)), X]), y, rcond=None)[0]


def nowcast(d: pl.DataFrame, lead: int, min_train: int = MIN_TRAIN) -> pl.DataFrame:
    """Expanding-window one-step forecasts of y_t; training uses only target quarters before t."""
    y = d["y"].to_numpy().astype(float)
    xl = d["x"].shift(lead).to_numpy().astype(float)
    y1 = d["y"].shift(1).to_numpy().astype(float)
    periods = d["period"].to_list()
    rows = []
    for t in range(len(y)):
        if np.isnan(y[t]) or np.isnan(xl[t]):
            continue
        past = np.arange(t)
        tr = past[~(np.isnan(y[past]) | np.isnan(xl[past]))]
        if len(tr) < min_train:
            continue
        b = _ols(xl[tr][:, None], y[tr])
        sig = b[0] + b[1] * xl[t]
        tr_ar = tr[~np.isnan(y1[tr])]
        ar1 = ar1x = math.nan
        if len(tr_ar) >= min_train and not np.isnan(y1[t]):
            a = _ols(y1[tr_ar][:, None], y[tr_ar])
            ar1 = a[0] + a[1] * y1[t]
            c = _ols(np.column_stack([y1[tr_ar], xl[tr_ar]]), y[tr_ar])
            ar1x = c[0] + c[1] * y1[t] + c[2] * xl[t]
        rows.append((periods[t], y[t], sig, y1[t], 0.0, ar1, ar1x, len(tr)))
    return pl.DataFrame(
        rows,
        schema={"period": pl.Date, "actual": pl.Float64, "signal": pl.Float64, "last": pl.Float64,
                "last_year": pl.Float64, "ar1": pl.Float64, "ar1_signal": pl.Float64, "n_train": pl.Int64},
        orient="row",
    ).with_columns(pl.col(pl.Float64).fill_nan(None))


MODELS = ("signal", "last", "last_year", "ar1", "ar1_signal")


def score(fc: pl.DataFrame) -> dict:
    common = fc.drop_nulls(subset=["actual", *MODELS])
    out: dict = {"n_oos": common.height}
    for m in MODELS:
        out[f"rmse_{m}"] = math.sqrt(((common[m] - common["actual"]) ** 2).mean()) if common.height else None
    def ratio(a: str, b: str):
        ra, rb = out[f"rmse_{a}"], out[f"rmse_{b}"]
        return ra / rb if ra is not None and rb and common.height >= MIN_OOS else None
    out |= {
        "ratio_vs_last": ratio("signal", "last"),
        "ratio_vs_last_year": ratio("signal", "last_year"),
        "ratio_ar1x_vs_ar1": ratio("ar1_signal", "ar1"),
    }
    return out


def kpi_unit(truth_df: pl.DataFrame, entity: str, metric: str) -> str | None:
    u = truth_df.filter((pl.col("entity") == entity) & (pl.col("metric") == metric))["unit"]
    return u[0] if len(u) else None


def signal_series(panel: pl.DataFrame, p: Pairing) -> pl.DataFrame:
    return features.series(panel, p.signal, kind=p.kind, category="all", basis="pit")


@dataclass
class Result:
    leadlag: pl.DataFrame
    oos: pl.DataFrame
    forecasts: pl.DataFrame
    data: pl.DataFrame
    skipped: list[str]


def run(
    panel: pl.DataFrame,
    truth_df: pl.DataFrame,
    start: date | None = None,
    end: date | None = None,
    pairings: tuple[Pairing, ...] = PAIRINGS,
    min_train: int = MIN_TRAIN,
) -> Result:
    """All pairings. `start` defaults to the first quarter of the last unbroken covered run of the FAA pull."""
    start = start or features.default_start(panel)
    ll, oos, fcs, data, skipped = [], [], [], [], []
    for p in pairings:
        unit = kpi_unit(truth_df, p.kpi_entity, p.kpi_metric)
        if unit is None:
            skipped.append(f"{p.key}: no KPI {p.kpi_entity} {p.kpi_metric}")
            continue
        sig = signal_series(panel, p)
        kpi = truth.kpi_series(truth_df, p.kpi_entity, p.kpi_metric)
        # KPI history before `start` stays: it only feeds the KPI's own YoY and the AR benchmarks.
        if start:
            sig = sig.filter(pl.col("period") >= start)
        if end:
            sig, kpi = sig.filter(pl.col("period") <= end), kpi.filter(pl.col("period") <= end)
        if sig.is_empty() or kpi.is_empty():
            skipped.append(f"{p.key}: no overlap")
            continue
        d = aligned(sig, kpi, unit).filter(pl.col("period") <= sig["period"].max())
        meta = {"key": p.key, "family": p.family, "signal": p.signal, "kind": p.kind, "kpi": f"{p.kpi_entity} {p.kpi_metric}"}
        for tr in ("yoy", "level"):
            ll.append(lead_lag(d, transform=tr).with_columns(transform=pl.lit(tr), **{k: pl.lit(v) for k, v in meta.items()}))
        fc = nowcast(d, p.lead, min_train)
        fcs.append(fc.with_columns(key=pl.lit(p.key)))
        best = ll[-2].filter(pl.col("corr").is_not_null()).sort(pl.col("corr").abs(), descending=True).head(1)
        oos.append(meta | {"lead": p.lead, "corr_at_lead": _at(ll[-2], p.lead),
                           "best_lead_insample": best["lead"][0] if best.height else None} | score(fc))
        data.append(d.with_columns(key=pl.lit(p.key)))
    oos_df = pl.DataFrame(oos) if oos else pl.DataFrame()
    return Result(
        leadlag=pl.concat(ll) if ll else pl.DataFrame(),
        oos=oos_df,
        forecasts=pl.concat(fcs) if fcs else pl.DataFrame(),
        data=pl.concat(data, how="diagonal_relaxed") if data else pl.DataFrame(),
        skipped=skipped,
    )


def _at(ll: pl.DataFrame, lead: int) -> float | None:
    r = ll.filter(pl.col("lead") == lead)["corr"]
    return r[0] if len(r) else None


def save(res: Result, data_dir: Path | str | None = None) -> Path:
    out = DataPaths.resolve(data_dir).parquet
    for name in ("leadlag", "oos", "forecasts", "data"):
        df = getattr(res, name)
        if not df.is_empty():
            df.write_parquet(out / f"backtest_{name}.parquet")
    return out


def load(data_dir: Path | str | None = None) -> Result:
    p = DataPaths.resolve(data_dir).parquet

    def read(name: str) -> pl.DataFrame:
        f = p / f"backtest_{name}.parquet"
        return pl.read_parquet(f) if f.exists() else pl.DataFrame()

    return Result(read("leadlag"), read("oos"), read("forecasts"), read("data"), [])


def wide_leadlag(ll: pl.DataFrame, transform: str = "yoy") -> pl.DataFrame:
    """key x lead table of 'corr [lo, hi] (n)' strings."""
    d = ll.filter(pl.col("transform") == transform).with_columns(
        cell=pl.when(pl.col("corr").is_null()).then(pl.format("- ({})", "n")).otherwise(
            pl.format("{} [{}, {}]{} ({})", pl.col("corr").round(2), pl.col("ci_lo").round(2), pl.col("ci_hi").round(2),
                      pl.when(pl.col("ci_method") == "fisher_z").then(pl.lit("*")).otherwise(pl.lit("")), "n")
        )
    )
    return d.pivot(on="lead", index=["key", "signal", "kind", "kpi"], values="cell", sort_columns=True).sort("key")
