"""Ground truth: hand-collected tower KPIs and carrier capex from SEC XBRL.

Tower KPIs live in ground_truth/tower_kpis.csv (quarter, entity, metric, value, unit, source_url, note), each row
taken from an earnings release or supplemental. Lines starting with '#' are comments.

Carrier capex comes from the SEC companyfacts API. Cash-flow items are reported year-to-date, so quarters are
differenced: Q1 = 3M, Q2 = 6M - 3M, Q3 = 9M - 6M, Q4 = FY - 9M. Each YTD value is the one first filed (10-Q /
10-K as originally reported, not later recasts), which is what was known at the time. Capex tags differ by
company and year, so each carrier has a priority list of concepts, optionally bounded to periods ending on or
after a date (Verizon's PaymentsToAcquireOtherProductiveAssets held an unrelated small item before 2017).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx
import polars as pl

from faa_oe.paths import PROJECT_ROOT, DataPaths

log = logging.getLogger(__name__)

KPI_CSV = PROJECT_ROOT / "ground_truth" / "tower_kpis.csv"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
DEFAULT_USER_AGENT = "faa-oe research pipeline"
FORMS = ("10-Q", "10-K", "10-Q/A", "10-K/A")
CARRIERS_SUM = "VZ+T+TMUS"


@dataclass(frozen=True)
class Concept:
    name: str
    min_end: date | None = None


CAPEX: dict[str, tuple[str, tuple[Concept, ...]]] = {
    "VZ": ("0000732712", (Concept("PaymentsToAcquirePropertyPlantAndEquipment"),
                          Concept("PaymentsToAcquireProductiveAssets"),
                          Concept("PaymentsToAcquireOtherProductiveAssets", date(2017, 1, 1)))),
    "T": ("0000732717", (Concept("PaymentsToAcquirePropertyPlantAndEquipment"),
                         Concept("PaymentsToAcquireProductiveAssets"))),
    "TMUS": ("0001283699", (Concept("PaymentsToAcquirePropertyPlantAndEquipment"),)),
}


class SecBlocked(RuntimeError):
    pass


def quarter_start(label: str) -> date:
    """'2019Q3' or '2019-Q3' -> date(2019, 7, 1)."""
    s = label.strip().upper().replace("-", "")
    year, q = s.split("Q")
    return date(int(year), 3 * (int(q) - 1) + 1, 1)


def quarter_label(d: date) -> str:
    return f"{d.year}Q{(d.month - 1) // 3 + 1}"


def load_kpis(path: Path | str | None = None) -> pl.DataFrame:
    df = pl.read_csv(path or KPI_CSV, comment_prefix="#", schema_overrides={"value": pl.Float64, "quarter": pl.String})
    return df.with_columns(
        period=pl.col("quarter").map_elements(quarter_start, return_dtype=pl.Date), source=pl.lit("kpi_csv")
    )


KPI_HEADER = (
    "# UNVERIFIED - awaiting spot-check. Hand-collected from earnings releases (SEC 8-K EX-99.1); each row's source_url\n"
    "# was opened when collected and `note` quotes the figure. Missing quarters are absent, never interpolated.\n"
)


def merge_raw_kpis(raw: list[Path], dest: Path | str | None = None) -> pl.DataFrame:
    """Concatenate per-company raw KPI CSVs into the curated file (sorted, de-duplicated, with the header)."""
    df = (
        pl.concat([pl.read_csv(p, schema_overrides={"value": pl.Float64, "quarter": pl.String}) for p in raw])
        .unique(["quarter", "entity", "metric"], keep="first")
        .sort("entity", "metric", "quarter")
    )
    dest = Path(dest or KPI_CSV)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8", newline="") as fh:
        fh.write(KPI_HEADER)
        df.write_csv(fh)
    return df


def user_agent() -> str:
    return os.environ.get("SEC_USER_AGENT") or DEFAULT_USER_AGENT


def fetch_companyfacts(cik: str, cache_dir: Path, refresh: bool = False, max_age_days: float = 7.0) -> dict:
    dest = cache_dir / f"companyfacts_CIK{cik}.json"
    fresh = dest.exists() and (date.today() - date.fromtimestamp(dest.stat().st_mtime)).days < max_age_days
    if fresh and not refresh:
        return json.loads(dest.read_text(encoding="utf-8"))
    r = httpx.get(COMPANYFACTS_URL.format(cik=cik), headers={"User-Agent": user_agent()}, timeout=60,
                  follow_redirects=True)
    if r.status_code in (403, 429):
        raise SecBlocked(f"SEC returned {r.status_code} for CIK {cik} with User-Agent {user_agent()!r}; set SEC_USER_AGENT")
    r.raise_for_status()
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    return r.json()


def ytd_facts(facts: dict, concepts: tuple[Concept, ...]) -> pl.DataFrame:
    """(start, end, value) as first filed, merged across concepts in priority order."""
    rows = []
    for rank, c in enumerate(concepts):
        for x in facts.get("facts", {}).get("us-gaap", {}).get(c.name, {}).get("units", {}).get("USD", []):
            if x.get("form") not in FORMS or "start" not in x:
                continue
            end = date.fromisoformat(x["end"])
            if c.min_end and end < c.min_end:
                continue
            rows.append((date.fromisoformat(x["start"]), end, float(x["val"]), x["filed"], rank, c.name))
    schema = {"start": pl.Date, "end": pl.Date, "value": pl.Float64, "filed": pl.String, "rank": pl.Int32, "concept": pl.String}
    df = pl.DataFrame(rows, schema=schema, orient="row")
    return (
        df.sort("start", "end", "rank", "filed")
        .unique(["start", "end"], keep="first", maintain_order=True)
        .with_columns(filed=pl.col("filed").str.to_date())
    )


def _months(start: pl.Expr, end: pl.Expr) -> pl.Expr:
    return (end.dt.year() - start.dt.year()) * 12 + end.dt.month() - start.dt.month() + 1


def quarterly_from_ytd(ytd: pl.DataFrame) -> pl.DataFrame:
    """Quarter values from year-to-date durations sharing a fiscal-year start; direct 3-month facts fill gaps.

    Returns period (quarter start), value, filed (latest filing date the value relies on), method.
    """
    d = ytd.filter(pl.col("start").dt.day() == 1).with_columns(months=_months(pl.col("start"), pl.col("end")))
    fy_starts = d.filter(pl.col("months").is_in([6, 9, 12])).select("start").unique()
    ytd_rows = d.filter(pl.col("months").is_in([3, 6, 9, 12])).join(fy_starts, on="start", how="semi").sort("start", "months")
    diffed = (
        ytd_rows.with_columns(
            prev_months=pl.col("months").shift(1).over("start"),
            prev_value=pl.col("value").shift(1).over("start"),
            prev_filed=pl.col("filed").shift(1).over("start"),
        )
        .filter((pl.col("months") == 3) | (pl.col("prev_months") == pl.col("months") - 3))
        .select(
            period=pl.col("end").dt.month_start().dt.offset_by("-2mo"),
            value=pl.when(pl.col("months") == 3).then(pl.col("value")).otherwise(pl.col("value") - pl.col("prev_value")),
            filed=pl.max_horizontal("filed", "prev_filed"),
            method=pl.when(pl.col("months") == 3).then(pl.lit("3M")).otherwise(pl.format("{}M-{}M", "months", "prev_months")),
        )
    )
    direct = d.filter(pl.col("months") == 3).select(
        period=pl.col("start"), value="value", filed="filed", method=pl.lit("3M direct")
    )
    return (
        pl.concat([diffed, direct.join(diffed, on="period", how="anti")])
        .filter(pl.col("period").dt.month().is_in([1, 4, 7, 10]))
        .sort("period")
    )


def carrier_capex(data_dir: Path | str | None = None, refresh: bool = False) -> pl.DataFrame:
    """Quarterly capex (USD m) for VZ, T, TMUS and their sum; cached under data/truth/."""
    cache = DataPaths.resolve(data_dir).root / "truth"
    frames = []
    for ticker, (cik, concepts) in CAPEX.items():
        q = quarterly_from_ytd(ytd_facts(fetch_companyfacts(cik, cache, refresh=refresh), concepts))
        frames.append(q.with_columns(entity=pl.lit(ticker)))
    df = pl.concat(frames).with_columns(
        value=(pl.col("value") / 1e6).round(1), metric=pl.lit("capex_usd_m"), unit=pl.lit("usd_m"),
        source=pl.lit("sec_xbrl"),
    )
    total = (
        df.group_by("period")
        .agg(value=pl.col("value").sum(), n=pl.len(), filed=pl.col("filed").max())
        .filter(pl.col("n") == len(CAPEX))
        .select("period", "value", "filed", method=pl.lit("sum"), entity=pl.lit(CARRIERS_SUM),
                metric=pl.lit("capex_usd_m"), unit=pl.lit("usd_m"), source=pl.lit("sec_xbrl"))
    )
    out = pl.concat([df, total.select(df.columns)]).sort("entity", "period")
    out.write_parquet(cache / "carrier_capex.parquet")
    return out


TRUTH_COLUMNS = ["period", "entity", "metric", "value", "unit", "source"]


def load_truth(data_dir: Path | str | None = None, kpi_path: Path | str | None = None, refresh: bool = False,
               capex: bool = True) -> pl.DataFrame:
    """KPIs + carrier capex, long: period, entity, metric, value, unit, source.

    Capex is read from the cache when present; if SEC refuses the request, capex is left out with a warning.
    """
    parts = [load_kpis(kpi_path).select(TRUTH_COLUMNS)] if Path(kpi_path or KPI_CSV).exists() else []
    if capex:
        cached = DataPaths.resolve(data_dir).root / "truth" / "carrier_capex.parquet"
        try:
            cx = pl.read_parquet(cached) if cached.exists() and not refresh else carrier_capex(data_dir, refresh)
            parts.append(cx.select(TRUTH_COLUMNS))
        except SecBlocked as e:
            log.warning("carrier capex skipped: %s", e)
    if not parts:
        return pl.DataFrame(schema={"period": pl.Date, "entity": pl.String, "metric": pl.String, "value": pl.Float64,
                                    "unit": pl.String, "source": pl.String})
    return pl.concat(parts, how="vertical_relaxed").sort("entity", "metric", "period")


def kpi_series(truth: pl.DataFrame, entity: str, metric: str) -> pl.DataFrame:
    return truth.filter((pl.col("entity") == entity) & (pl.col("metric") == metric)).select("period", "value", "unit").sort("period")


def coverage_table(truth: pl.DataFrame) -> pl.DataFrame:
    return (
        truth.group_by("entity", "metric", "unit", "source")
        .agg(n=pl.len(), first=pl.col("period").min(), last=pl.col("period").max())
        .with_columns(
            pl.col("first").map_elements(quarter_label, return_dtype=pl.String),
            pl.col("last").map_elements(quarter_label, return_dtype=pl.String),
        )
        .sort("entity", "metric")
    )
