"""Weekly and quarterly filing panels for the backtest, built point-in-time.

Stage 2's `build_kind` and entity attribution use facts that only became known after a filing: an ASR
registration or construction after the filing marks a new build, and the ASR-owner fallback names the owner of
a structure registered later. Features therefore use point-in-time (`pit`) variants:

  entity      sponsor rules only (the sponsor name is on the filing); the ASR-owner fallback is dropped.
  pit_kind    `modification` if, on date_entered, evidence of an existing structure already existed:
                asr_registered  ASR registration (estimated date) more than `reg_margin_days` before filing
                asr_constructed ASR construction date more than `built_margin_days` before filing
                faa_built       FAA date_built more than `built_margin_days` before filing
                keyword         structure_description names work on an existing structure
                prior_case      an earlier tower / monopole / antenna-mount filing within 100 m, entered at
                                least 60 days before (optionally within `prior_lookback_days`)
              An ASR structure dismantled before the filing is not evidence. Otherwise `new_or_unknown`;
              nothing after the filing date is used.
  projects    multi-case projects re-clustered on the sponsor-based key; `n_projects` counts the first case
              of each project, so a project is counted once, in the period it started.

Filings are counted by date_entered only and never conditioned on current status. Cases the API no longer
serves (withdrawn, deleted) are simply absent: see coverage.completeness for how many.

The hindsight variant (`basis = "hindsight"`: entity with ASR-owner fallback, Stage 2 build_kind) is kept for
descriptive charts only.

Panels are long and sparse (only non-zero cells): period, basis, level (entity / group / coverage), name,
category (tower, monopole, all), kind (incl. all), n, n_hw, n_projects, complete. `n_hw` weights each filing by
AGL height / 200 ft (missing height counts 1). Use `series` to read one zero-filled series.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from faa_oe.build_kind import BUILD_KIND_CATEGORIES, EXISTING_KEYWORDS, PRIOR_CASE_CATEGORIES
from faa_oe.geo import pairs_within
from faa_oe.paths import DataPaths
from faa_oe.projects import assign_projects

log = logging.getLogger(__name__)

PIT_KINDS = ("modification", "new_or_unknown")
EVIDENCE = ("asr_registered", "asr_constructed", "faa_built", "keyword", "prior_case")
REG_MARGIN_DAYS = 365
BUILT_MARGIN_DAYS = 30
PRIOR_RADIUS_M = 100.0
PRIOR_MIN_DAYS = 60
HEIGHT_UNIT_FT = 200.0

TOWERS_JV = "The Towers (Vertical Bridge/Verizon JV)"
REITS = ("AMT", "CCI", "SBAC")
CARRIERS = ("VZ", "T", "TMUS")
ENTITY_TYPES = ("tower_reit", "towerco", "carrier", "private_towerco")


def _groups() -> dict[str, pl.Expr]:
    t, e, et = pl.col("ticker"), pl.col("entity"), pl.col("entity_type")
    carriers = t.is_in(CARRIERS)
    private = et == "private_towerco"
    return {
        "listed_reits": t.is_in(REITS),
        "carriers": carriers,
        "carriers_incl_towers_jv": carriers | (e == TOWERS_JV),
        "vz_incl_towers_jv": (t == "VZ") | (e == TOWERS_JV),
        "private_builders": private,
        "private_and_carriers": private | carriers,
        "all": pl.lit(True),
    }


GROUPS = tuple(_groups())

INPUT = [
    "asn", "date_entered", "category", "is_tower_signal", "sponsor_norm", "entity", "ticker", "entity_type",
    "entity_source", "structure_description", "date_built", "latitude", "longitude", "agl_structure_height",
    "asr_registration_date", "asr_date_constructed", "asr_date_dismantled", "build_kind", "project_id",
]


def pit_kind(
    cases: pl.DataFrame,
    reg_margin_days: int = REG_MARGIN_DAYS,
    built_margin_days: int = BUILT_MARGIN_DAYS,
    prior_radius_m: float = PRIOR_RADIUS_M,
    prior_min_days: int = PRIOR_MIN_DAYS,
    prior_lookback_days: int | None = None,
) -> pl.DataFrame:
    """asn -> pit_kind, pit_basis and one boolean per evidence (tower / monopole / COW rows only).

    `cases` needs asn, category, date_entered, date_built, structure_description, latitude, longitude,
    asr_registration_date, asr_date_constructed, asr_date_dismantled; rows of the prior-case categories are
    used as earlier filings.
    """
    de = pl.col("date_entered")
    standing = pl.col("asr_date_dismantled").is_null() | (pl.col("asr_date_dismantled") >= de)
    d = cases.filter(pl.col("category").is_in(BUILD_KIND_CATEGORIES))
    prior = _prior_cases(cases, prior_radius_m, prior_min_days, prior_lookback_days)
    d = d.join(prior, on="asn", how="left").select(
        "asn",
        asr_registered=(standing & (pl.col("asr_registration_date") < de - pl.duration(days=reg_margin_days))).fill_null(False),
        asr_constructed=(standing & (pl.col("asr_date_constructed") < de - pl.duration(days=built_margin_days))).fill_null(False),
        faa_built=(pl.col("date_built") < de - pl.duration(days=built_margin_days)).fill_null(False),
        keyword=pl.col("structure_description").fill_null("").str.to_uppercase().str.contains(EXISTING_KEYWORDS),
        prior_case=pl.col("n_prior_cases").fill_null(0) > 0,
    )
    basis = pl.coalesce([pl.when(pl.col(e)).then(pl.lit(e)) for e in EVIDENCE])
    return d.with_columns(
        pit_kind=pl.when(pl.any_horizontal(EVIDENCE)).then(pl.lit("modification")).otherwise(pl.lit("new_or_unknown")),
        pit_basis=basis.fill_null("none"),
    )


def _prior_cases(cases: pl.DataFrame, radius_m: float, min_days: int, lookback_days: int | None) -> pl.DataFrame:
    cols = ["asn", "latitude", "longitude", "date_entered"]
    left = cases.filter(pl.col("category").is_in(BUILD_KIND_CATEGORIES)).select(cols)
    right = cases.filter(pl.col("category").is_in(PRIOR_CASE_CATEGORIES)).select(cols)
    pairs = pairs_within(left, right, radius_m)
    cond = pl.col("date_entered_r") <= pl.col("date_entered") - pl.duration(days=min_days)
    if lookback_days is not None:
        cond = cond & (pl.col("date_entered_r") >= pl.col("date_entered") - pl.duration(days=lookback_days))
    return pairs.filter(cond).group_by("asn").agg(n_prior_cases=pl.len())


def load_inputs(data_dir: Path | str | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(tower-ish cases with the INPUT columns, all-case counts per day for coverage)."""
    lf = pl.scan_parquet(DataPaths.resolve(data_dir).enriched)
    cats = sorted(set(BUILD_KIND_CATEGORIES) | set(PRIOR_CASE_CATEGORIES))
    cases = lf.filter(pl.col("category").is_in(cats) & pl.col("date_entered").is_not_null()).select(INPUT).collect()
    daily = lf.filter(pl.col("date_entered").is_not_null()).group_by("date_entered").agg(n=pl.len()).collect()
    return cases, daily


def prepare(cases: pl.DataFrame, **pit_kw) -> pl.DataFrame:
    """Tower-signal filings with pit and hindsight entity / kind / project-start columns."""
    pk = pit_kind(cases, **pit_kw)
    tw = cases.filter(pl.col("is_tower_signal")).join(pk.select("asn", "pit_kind", "pit_basis"), on="asn", how="left")
    by_sponsor = pl.col("entity_source") == "sponsor"
    tw = tw.with_columns(
        entity_pit=pl.when(by_sponsor).then("entity"),
        ticker_pit=pl.when(by_sponsor).then("ticker"),
        entity_type_pit=pl.when(by_sponsor).then("entity_type"),
    )
    proj = assign_projects(
        tw.select("asn", "date_entered", "latitude", "longitude", project_key=pl.coalesce("entity_pit", "sponsor_norm"))
    )
    return tw.join(proj.select("asn", project_id_pit="project_id"), on="asn", how="left").with_columns(
        w=pl.col("agl_structure_height").fill_null(HEIGHT_UNIT_FT) / HEIGHT_UNIT_FT,
        project_start_pit=pl.col("project_id_pit") == pl.col("asn"),
        project_start_hindsight=(pl.col("project_id").fill_null(pl.col("asn")) == pl.col("asn")),
    )


def _basis_frame(tw: pl.DataFrame, basis: str) -> pl.DataFrame:
    if basis == "pit":
        cols = {"entity_pit": "entity", "ticker_pit": "ticker", "entity_type_pit": "entity_type",
                "pit_kind": "kind", "project_start_pit": "project_start"}
    else:
        cols = {"entity": "entity", "ticker": "ticker", "entity_type": "entity_type",
                "build_kind": "kind", "project_start_hindsight": "project_start"}
    return tw.select("date_entered", "category", "w", *[pl.col(k).alias(v) for k, v in cols.items()]).with_columns(
        basis=pl.lit(basis)
    )


def _expand(d: pl.DataFrame) -> pl.DataFrame:
    d = pl.concat([d, d.with_columns(category=pl.lit("all"))])
    return pl.concat([d, d.with_columns(kind=pl.lit("all"))])


def _period(freq: str) -> pl.Expr:
    return pl.col("date_entered").dt.truncate("1w" if freq == "W" else "1q").alias("period")


def _period_end(freq: str) -> pl.Expr:
    nxt = pl.col("period").dt.offset_by("1w" if freq == "W" else "1q")
    return nxt - pl.duration(days=1)


AGG = [pl.len().alias("n"), pl.col("w").sum().alias("n_hw"), pl.col("project_start").sum().alias("n_projects")]
KEYS = ["period", "basis", "level", "name", "category", "kind"]


def panel(tw: pl.DataFrame, daily: pl.DataFrame, freq: str, asof: date | None = None,
          start: date | None = None, end: date | None = None) -> pl.DataFrame:
    """Long sparse panel at `freq` ('W' or 'Q') from `prepare` output and daily all-case counts."""
    asof = asof or daily["date_entered"].max()
    frames = []
    for basis in ("pit", "hindsight"):
        d = _expand(_basis_frame(tw, basis)).with_columns(_period(freq))
        ent = d.filter(pl.col("entity").is_not_null() & pl.col("entity_type").is_in(ENTITY_TYPES))
        frames.append(ent.group_by("period", "basis", "entity", "category", "kind").agg(AGG)
                      .rename({"entity": "name"}).with_columns(level=pl.lit("entity")))
        for name, cond in _groups().items():
            g = d.filter(cond.fill_null(False))
            frames.append(g.group_by("period", "basis", "category", "kind").agg(AGG)
                          .with_columns(name=pl.lit(name), level=pl.lit("group")))
    cov = daily.with_columns(_period(freq)).group_by("period").agg(
        n=pl.col("n").sum().cast(pl.UInt32), n_hw=pl.col("n").sum().cast(pl.Float64), n_projects=pl.col("n").sum().cast(pl.UInt32)
    ).with_columns(basis=pl.lit("pit"), level=pl.lit("coverage"), name=pl.lit("all_cases"), category=pl.lit("any"), kind=pl.lit("all"))
    out = pl.concat([f.select(*KEYS, "n", "n_hw", "n_projects").with_columns(pl.col("n").cast(pl.UInt32),
                                                                              pl.col("n_projects").cast(pl.UInt32))
                     for f in [*frames, cov]])
    if start:
        out = out.filter(pl.col("period") >= start)
    if end:
        out = out.filter(pl.col("period") <= end)
    return out.with_columns(complete=_period_end(freq) <= asof).sort(KEYS)


def run(data_dir: Path | str | None = None, start: date | None = None, **pit_kw) -> dict[str, Path]:
    paths = DataPaths.resolve(data_dir)
    cases, daily = load_inputs(data_dir)
    tw = prepare(cases, **pit_kw)
    del cases
    asof = daily["date_entered"].max()
    out = {}
    for freq, name in (("W", "features_weekly"), ("Q", "features_quarterly")):
        p = paths.parquet / f"{name}.parquet"
        df = panel(tw, daily, freq, asof=asof, start=start)
        df.write_parquet(p)
        log.info("wrote %s (%s rows)", p, f"{df.height:,}")
        out[freq] = p
    kinds = tw.select("asn", "date_entered", "category", "entity_pit", "pit_kind", "pit_basis", "build_kind")
    kinds.write_parquet(paths.parquet / "pit_kinds.parquet")
    return out


def load_panel(freq: str = "Q", data_dir: Path | str | None = None) -> pl.DataFrame:
    name = "features_weekly" if freq == "W" else "features_quarterly"
    return pl.read_parquet(DataPaths.resolve(data_dir).parquet / f"{name}.parquet")


def series(
    p: pl.DataFrame,
    name: str,
    kind: str = "new_or_unknown",
    category: str = "all",
    basis: str = "pit",
    metric: str = "n",
    freq: str = "Q",
    complete_only: bool = True,
) -> pl.DataFrame:
    """One zero-filled series (period, value) over the panel's full period range."""
    grid_src = p.filter(pl.col("complete")) if complete_only else p
    if grid_src.is_empty():
        return pl.DataFrame(schema={"period": pl.Date, "value": pl.Float64})
    grid = pl.date_range(grid_src["period"].min(), grid_src["period"].max(), "1w" if freq == "W" else "1q", eager=True)
    s = p.filter((pl.col("name") == name) & (pl.col("kind") == kind) & (pl.col("category") == category)
                 & (pl.col("basis") == basis)).select("period", value=pl.col(metric).cast(pl.Float64))
    return pl.DataFrame({"period": grid}).join(s, on="period", how="left").with_columns(pl.col("value").fill_null(0.0))


def kind_counts(tw: pl.DataFrame, by: str = "year") -> pl.DataFrame:
    """PIT vs hindsight build-kind counts per year (or quarter) of date_entered."""
    period = pl.col("date_entered").dt.year().alias(by) if by == "year" else pl.col("date_entered").dt.truncate("1q").alias(by)
    pit = tw.group_by(period, "pit_kind").len("n").pivot(on="pit_kind", index=by, values="n").fill_null(0)
    hs = tw.group_by(period, "build_kind").len("n").pivot(on="build_kind", index=by, values="n").fill_null(0)
    hs = hs.rename({c: f"hs_{c}" for c in hs.columns if c != by})
    return pit.join(hs, on=by, how="full", coalesce=True).fill_null(0).sort(by)


def pit_basis_shares(tw: pl.DataFrame) -> pl.DataFrame:
    """Share of filings each evidence decided, per year: shows where prior-case reach is still short."""
    return (
        tw.group_by(pl.col("date_entered").dt.year().alias("year"), "pit_basis").len("n")
        .with_columns(share=(pl.col("n") / pl.col("n").sum().over("year")).round(3))
        .pivot(on="pit_basis", index="year", values="share").fill_null(0).sort("year")
    )


def coverage_ok(p: pl.DataFrame, min_ratio: float = 0.5) -> pl.DataFrame:
    """Per period: all-case count and whether it is at least `min_ratio` of the median complete period."""
    c = p.filter((pl.col("level") == "coverage") & pl.col("complete")).select("period", "n").sort("period")
    med = c["n"].median() or 0
    return c.with_columns(ok=pl.col("n") >= min_ratio * med)


def default_start(p: pl.DataFrame, min_ratio: float = 0.5, freq: str = "Q") -> date | None:
    """First period of the last unbroken run of covered periods (gaps in the pull break the run)."""
    c = coverage_ok(p, min_ratio)
    if c.is_empty():
        return None
    start = None
    prev = None
    step_days = 8 if freq == "W" else 95
    for period, ok in c.select("period", "ok").iter_rows():
        if not ok or (prev is not None and (period - prev) > timedelta(days=step_days)):
            start = None
        if ok and start is None:
            start = period
        prev = period
    return start
