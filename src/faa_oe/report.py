"""reports/summary.md: coverage, attribution, PIT vs hindsight build kinds, lags, backtest tables, caveats."""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import polars as pl

from faa_oe import backtest, coverage, enrich, features, truth
from faa_oe.paths import PROJECT_ROOT, DataPaths

REPORT = PROJECT_ROOT / "reports" / "summary.md"
REITS = {"AMT": "American Tower", "CCI": "Crown Castle", "SBAC": "SBA Communications"}


def md(df: pl.DataFrame, digits: int = 3) -> str:
    if df.is_empty():
        return "_(none)_\n"

    def cell(v) -> str:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return ""
        if isinstance(v, float):
            return f"{v:,.{digits}f}" if abs(v) < 1000 else f"{v:,.0f}"
        if isinstance(v, int) and not isinstance(v, bool) and abs(v) >= 10000:
            return f"{v:,}"
        return str(v).replace("|", "/")

    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * df.width]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.iter_rows()]
    return "\n".join(lines) + "\n"


def completeness_by_year(data_dir: Path | str | None = None) -> pl.DataFrame:
    cols = ["asn_region", "asn_year", "asn_type", "asn_sequence", "status_code"]
    c = coverage.completeness(pl.scan_parquet(DataPaths.resolve(data_dir).cases).select(cols).collect())
    return (
        c.group_by("asn_year", "asn_type")
        .agg(cases=pl.col("n_cases").sum(), seq_held=pl.col("n_distinct_seq").sum(), seq_issued=pl.col("max_seq").sum())
        .with_columns(completeness=(pl.col("seq_held") / pl.col("seq_issued")).round(3))
        .pivot(on="asn_type", index="asn_year", values=["cases", "completeness"])
        .sort("asn_year")
    )


def attribution_by_year(tw: pl.DataFrame) -> pl.DataFrame:
    return (
        tw.group_by(pl.col("date_entered").dt.year().alias("year"))
        .agg(
            filings=pl.len(),
            hindsight_any=pl.col("entity").is_not_null().mean().round(3),
            pit_sponsor=(pl.col("entity_source") == "sponsor").fill_null(False).mean().round(3),
            hindsight_listed=pl.col("ticker").is_not_null().mean().round(3),
        )
        .sort("year")
    )


TW_COLS = ["asn", "date_entered", "entity", "ticker", "entity_type", "entity_source", "asr_owner_entity",
           "asr_registration_number", "fcc_asr_number", "asr_owner", "build_kind", "filing_to_construction_days"]


def load_tower(data_dir: Path | str | None = None) -> pl.DataFrame:
    return (
        pl.scan_parquet(DataPaths.resolve(data_dir).enriched)
        .filter(pl.col("is_tower_signal"))
        .select(TW_COLS)
        .collect()
        .with_columns(reg=pl.coalesce("asr_registration_number",
                                      pl.col("fcc_asr_number").str.replace_all(r"\D", "").str.strip_chars_start("0")))
    )


def _half(c: str = "date_entered") -> pl.Expr:
    return (pl.col(c).dt.year().cast(pl.String) + "H" + ((pl.col(c).dt.month() > 6).cast(pl.Int8) + 1).cast(pl.String)).alias("half")


def reit_monthly(tw: pl.DataFrame, since: date) -> pl.DataFrame:
    names = list(REITS.values()) + ["Vertical Bridge", features.TOWERS_JV]
    return (
        tw.filter(pl.col("entity").is_in(names) & (pl.col("date_entered") >= since))
        .with_columns(month=pl.col("date_entered").dt.truncate("1mo"))
        .filter(pl.col("month").dt.month_end() <= tw["date_entered"].max())
        .group_by("month", "entity").len("n")
        .pivot(on="entity", index="month", values="n").fill_null(0).sort("month")
    )


def filer_shift(tw: pl.DataFrame, owner: str = "Crown Castle", since: date = date(2024, 7, 1)) -> pl.DataFrame:
    """Filings on structures whose current ASR owner is `owner`, by filing entity x half-year."""
    d = tw.filter((pl.col("asr_owner_entity") == owner) & (pl.col("date_entered") >= since)).with_columns(_half())
    return (
        d.group_by("half", pl.col("entity").fill_null("<unattributed>")).len("n")
        .pivot(on="half", index="entity", values="n", sort_columns=True).fill_null(0)
        .with_columns(total=pl.sum_horizontal(pl.exclude("entity"))).sort("total", descending=True).drop("total")
    )


def same_structure_followups(tw: pl.DataFrame, entity: str = "Crown Castle", base=(date(2024, 1, 1), date(2025, 12, 31)),
                             after: date = date(2026, 1, 1)) -> tuple[int, int, pl.DataFrame]:
    """Structures (ASR number) `entity` filed for in `base`; who filed on the same structures from `after` on."""
    regs = tw.filter((pl.col("entity") == entity) & pl.col("date_entered").is_between(*base) & pl.col("reg").is_not_null())["reg"].unique()
    later = tw.filter(pl.col("reg").is_in(regs.implode()) & (pl.col("date_entered") >= after))
    by = later.group_by(pl.col("entity").fill_null("<unattributed>")).len("n").sort("n", descending=True)
    return len(regs), later.height, by


def recent_owner_mix(tw: pl.DataFrame, entity: str = "Vertical Bridge", since: date = date(2026, 8, 1)) -> pl.DataFrame:
    d = tw.filter((pl.col("entity") == entity) & (pl.col("date_entered") >= since))
    return (
        d.group_by(pl.col("asr_owner_entity").fill_null("<no ASR match>").alias("asr_owner_entity"), "build_kind").len("n")
        .pivot(on="build_kind", index="asr_owner_entity", values="n").fill_null(0)
        .with_columns(total=pl.sum_horizontal(pl.exclude("asr_owner_entity"))).sort("total", descending=True).head(8)
    )


def _q(d: date | None) -> str:
    return truth.quarter_label(d) if d else "-"


def build(data_dir: Path | str | None = None, start: date | None = None, end: date | None = None) -> str:
    panel = features.load_panel("Q", data_dir)
    tdf = truth.load_truth(data_dir)
    start = start or features.default_start(panel)
    res = backtest.run(panel, tdf, start=start, end=end)
    backtest.save(res, data_dir)
    tw = load_tower(data_dir)
    pk = pl.read_parquet(DataPaths.resolve(data_dir).parquet / "pit_kinds.parquet")
    asof = tw["date_entered"].max()
    out: list[str] = []
    w = out.append

    w("# FAA tower filings vs reported tower / carrier KPIs\n")
    w(f"Data: {tw['date_entered'].min()} to {asof} ({tw.height:,} tower-signal filings). "
      f"Backtest signal window starts {_q(start)}"
      + (f", ends {_q(end)}" if end else "") + ". Generated by `py -m faa_oe report`.\n")

    w("\n## Coverage\n")
    w("Share of issued ASN sequence numbers the API still serves (holes = withdrawn / deleted / unpublished cases; "
      "a survivorship gap the signal cannot see), by ASN year and series.\n\n")
    w(md(completeness_by_year(data_dir)))
    cov = features.coverage_ok(panel).with_columns(pl.col("period").map_elements(truth.quarter_label, return_dtype=pl.String))
    bad = cov.filter(~pl.col("ok"))["period"].to_list()
    w(f"\nQuarters with fewer than half the median all-case count (treated as gaps in the pull): {', '.join(bad) or 'none'}.\n")

    w("\n## Attribution (tower-signal filings)\n")
    w("`pit_sponsor` is the point-in-time rate used by the features (sponsor rules only); `hindsight_any` adds the "
      "ASR-owner fallback, which relies on registrations made after many filings.\n\n")
    w(md(attribution_by_year(tw)))

    w("\n## Build kind: point-in-time vs hindsight\n")
    w("`pit_kind` uses only evidence dated before the filing; hindsight `build_kind` (Stage 2) also uses later ASR "
      "registrations / construction (`hs_new_build`). PIT `new_or_unknown` = hindsight new builds + unknowns + the few "
      "cases whose only evidence came later.\n\n")
    w(md(features.kind_counts(pk)))
    w("\nWhich evidence decided `modification` (share of filings per year; `none` = new_or_unknown):\n\n")
    w(md(features.pit_basis_shares(pk)))

    w("\n## Filing to construction (hindsight, new builds with an ASR construction date)\n\n")
    lags = tw.filter(pl.col("filing_to_construction_days") >= -30)
    w(md(enrich.lag_summary(lags, by=None), 0))
    w("\n" + md(enrich.lag_summary(lags, by="entity", min_n=40).head(10), 0))

    w("\n## KPI series\n\n")
    w(md(truth.coverage_table(tdf)) if not tdf.is_empty() else "_no ground truth loaded_\n")

    w("\n## Lead-lag: corr(signal_{t-k}, KPI_t), YoY transforms\n")
    w("Cell: corr [95% CI] (n); moving-block bootstrap CI, or Fisher-z marked * when n < 12 (too narrow). Signals are point-in-time filing counts; KPIs are log YoY "
      "(pp change YoY for % KPIs). Many pairings x 7 leads: expect some |r| > 0.5 by chance at these n.\n\n")
    if not res.leadlag.is_empty():
        w(md(backtest.wide_leadlag(res.leadlag, "yoy")))
        w("\nLevels (reference only; shared trends inflate these):\n\n")
        w(md(backtest.wide_leadlag(res.leadlag, "level")))

    w("\n## Out-of-sample nowcast at the pre-specified lead\n")
    w(f"Expanding window, at least {backtest.MIN_TRAIN} training pairs; ratios are left blank below "
      f"{backtest.MIN_OOS} out-of-sample quarters. Ratios < 1 favour the signal. "
      "`last` = last KPI value (YoY scale), `last_year` = same quarter last year in levels (no YoY change), "
      "`ar1x/ar1` = AR(1) with vs without the signal.\n\n")
    if not res.oos.is_empty():
        cols = ["key", "signal", "kind", "kpi", "lead", "corr_at_lead", "best_lead_insample", "n_oos",
                "ratio_vs_last", "ratio_vs_last_year", "ratio_ar1x_vs_ar1"]
        w(md(res.oos.select(cols), 2))
    if res.skipped:
        w("\nSkipped: " + "; ".join(res.skipped) + "\n")

    w("\n## Crown Castle / Vertical Bridge 2026 check\n")
    w("Monthly tower-signal filings (hindsight entity):\n\n")
    w(md(reit_monthly(tw, date(asof.year - 1, 1, 1)).with_columns(pl.col("month").dt.strftime("%Y-%m"))))
    n_regs, n_later, by = same_structure_followups(tw)
    w(f"\nStructures Crown Castle filed on in 2024-25 (by ASR number): {n_regs:,}. Filings on those same structures "
      f"since 2026-01-01: {n_later:,}, by filer:\n\n")
    w(md(by))
    w("\nAll filings on structures whose ASR owner is Crown Castle, by filer and half-year:\n\n")
    w(md(filer_shift(tw)))
    w("\nVertical Bridge filings since 2026-08-01 by ASR owner and hindsight build kind:\n\n")
    w(md(recent_owner_mix(tw)))

    w("\n## Caveats\n")
    w("- Filings are counted by `date_entered`; status and completion are never conditioned on. Cases the API no "
      "longer serves are missing (see Coverage). Completeness is lower for older ASN years, so counts as seen today "
      "understate older periods relative to what was on file at the time, which biases YoY filing growth upward "
      "wherever completeness rises; KPI correlations are affected only if that drift lines up with the KPI.\n"
      "- `pit_kind` still relies on the ASR file as downloaded now: registration dates are estimated from the "
      "registration number (hence a 365-day margin), and a registration's coordinates/height are current values.\n"
      "- Sponsor rules (config/sponsors.yaml) were written in 2026 with knowledge of later corporate structure "
      "(e.g. The Towers JV); the mapping of names is hindsight, the names themselves are not.\n"
      "- `prior_case` evidence grows with the history held; early quarters of the pull have less of it, so their "
      "new_or_unknown share is biased up.\n"
      "- Hand-collected KPIs await spot-check (ground_truth/tower_kpis.csv). Capex is cash capex as first filed; "
      "TMUS includes Sprint from 2020Q2. CCI organic billings growth from 2023 is the company's adjusted figure "
      "(ex Sprint cancellations, from 2026 also ex DISH); every KPI is as first reported, not restated.\n"
      "- With ~10-45 quarters per test, a single significant correlation among dozens is weak evidence; the OOS ratio "
      "at the pre-specified lead is the test that counts.\n")
    return "".join(out)


def run(data_dir: Path | str | None = None, start: date | None = None, end: date | None = None,
        path: Path = REPORT) -> Path:
    text = build(data_dir, start, end)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
