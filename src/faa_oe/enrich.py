"""Stage 2 enrichment: cases + category + entity + build_kind + project + ASR -> cases_enriched.parquet."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import duckdb
import polars as pl

from faa_oe import asr, entities
from faa_oe.build_kind import build_kind
from faa_oe.classify import classify
from faa_oe.paths import DataPaths
from faa_oe.projects import assign_projects, project_key

log = logging.getLogger(__name__)

WORK_COLUMNS = [
    "asn", "sponsor", "structure_type", "structure_description", "date_entered", "date_built",
    "fcc_asr_number", "latitude", "longitude",
]
CASE_ASR_COLUMNS = {
    "registration_number": "asr_registration_number",
    "match_method": "asr_match_method",
    "match_distance_m": "asr_match_distance_m",
    "asr_status": "asr_status",
    "asr_structure_type": "asr_structure_type",
    "asr_registration_date": "asr_registration_date",
    "asr_date_constructed": "asr_date_constructed",
    "asr_date_dismantled": "asr_date_dismantled",
    "asr_height_agl_ft": "asr_height_agl_ft",
    "asr_owner": "asr_owner",
    "asr_owner_domain": "asr_owner_domain",
    "asr_owner_entity": "asr_owner_entity",
    "asr_owner_ticker": "asr_owner_ticker",
    "asr_owner_entity_type": "asr_owner_entity_type",
}
CASE_ASR_SCHEMA = {"asn": pl.String} | {
    c: pl.Date if "date" in c else pl.Float64 if c in ("match_distance_m", "asr_height_agl_ft") else pl.String
    for c in CASE_ASR_COLUMNS
}
VIEWS = {"cases_enriched": "enriched", "asr": "asr", "case_asr": "case_asr"}


def enrichment(cases: pl.DataFrame, case_asr: pl.DataFrame | None) -> pl.DataFrame:
    """asn-keyed Stage 2 columns for the projected `cases` (WORK_COLUMNS)."""
    if case_asr is None:
        case_asr = pl.DataFrame(schema=CASE_ASR_SCHEMA)
    else:
        # Re-derive owner entities so sponsors.yaml edits apply without re-running the ASR step.
        owner_cols = ["asr_owner_entity", "asr_owner_ticker", "asr_owner_entity_type"]
        case_asr = case_asr.drop(owner_cols, strict=False)
        case_asr = case_asr.join(entities.owner_table(case_asr), on=["asr_owner", "asr_owner_domain"], how="left", nulls_equal=True)
    df = entities.apply_owner_fallback(
        entities.attribute(classify(cases)).join(
            case_asr.select("asn", *CASE_ASR_COLUMNS).rename(CASE_ASR_COLUMNS), on="asn", how="left"
        )
    )
    bk = build_kind(df, case_asr.select("asn", "match_method", "asr_registration_date", "asr_date_constructed"))
    projects = assign_projects(df.with_columns(project_key()))
    df = df.join(bk, on="asn", how="left").join(projects, on="asn", how="left")
    return df.with_columns(
        filing_to_construction_days=pl.when(pl.col("build_kind") == "new_build")
        .then((pl.col("asr_date_constructed") - pl.col("date_entered")).dt.total_days())
        .cast(pl.Int32)
    ).drop([c for c in WORK_COLUMNS if c != "asn"])


def run(data_dir: Path | str | None = None, refresh_asr: bool = False, use_asr: bool = True) -> Path:
    paths = DataPaths.resolve(data_dir)
    case_asr = None
    if use_asr:
        _, case_asr = asr.run(data_dir, refresh=refresh_asr)
    elif paths.case_asr.exists():
        case_asr = pl.read_parquet(paths.case_asr)

    cases = pl.scan_parquet(paths.cases).select(WORK_COLUMNS).collect()
    extra = enrichment(cases, case_asr)
    del cases

    tmp = paths.enriched.with_suffix(".tmp.parquet")
    pl.scan_parquet(paths.cases).join(extra.lazy(), on="asn", how="left").sink_parquet(tmp)
    _replace(tmp, paths.enriched)
    register_views(paths)
    log.info("wrote %s (%s rows)", paths.enriched, f"{extra.height:,}")
    return paths.enriched


def _replace(tmp: Path, dest: Path) -> None:
    # A notebook or DuckDB holding the old file open makes os.replace fail transiently on Windows.
    for attempt in range(10):
        try:
            os.replace(tmp, dest)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(1 + attempt)


def register_views(paths: DataPaths) -> None:
    try:
        with duckdb.connect(str(paths.duckdb)) as con:
            for name, attr in VIEWS.items():
                p: Path = getattr(paths, attr)
                if p.exists():
                    con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{p.resolve().as_posix()}')")
    except duckdb.IOException as e:
        log.warning("could not update %s (open elsewhere?): %s", paths.duckdb, e)


TOWER_VIEW_COLUMNS = [
    "asn", "date_entered", "category", "is_tower_signal", "sponsor", "sponsor_norm", "entity", "ticker",
    "entity_type", "entity_source", "build_kind", "build_kind_basis", "project_id", "project_size",
    "asr_match_method", "asr_date_constructed", "asr_owner_entity", "filing_to_construction_days",
    "agl_structure_height", "nearest_state", "status_code",
]


def load_enriched(data_dir: Path | str | None = None, columns: list[str] | None = None, tower_only: bool = True) -> pl.DataFrame:
    lf = pl.scan_parquet(DataPaths.resolve(data_dir).enriched).select(columns or TOWER_VIEW_COLUMNS)
    return (lf.filter(pl.col("is_tower_signal")) if tower_only else lf).collect()


def entity_year_table(df: pl.DataFrame, listed_only: bool = True) -> pl.DataFrame:
    """Tower-signal filings per entity x year, split by build_kind (wide)."""
    d = df.filter(pl.col("is_tower_signal") & pl.col("entity").is_not_null())
    if listed_only:
        d = d.filter(pl.col("ticker").is_not_null() | (pl.col("entity_type") == "private_towerco"))
    return (
        d.with_columns(year=pl.col("date_entered").dt.year())
        .group_by("entity", "ticker", "year", "build_kind")
        .len("n")
        .pivot(on="build_kind", index=["entity", "ticker", "year"], values="n", sort_columns=True)
        .fill_null(0)
        .sort("entity", "year")
    )


def lag_summary(df: pl.DataFrame, by: str | None = "entity", min_n: int = 5) -> pl.DataFrame:
    """Filing -> ASR construction lag for new builds that have a construction date."""
    d = df.filter(pl.col("filing_to_construction_days").is_not_null())
    aggs = [
        pl.len().alias("n"),
        pl.col("filing_to_construction_days").median().alias("median_days"),
        pl.col("filing_to_construction_days").quantile(0.25).alias("p25_days"),
        pl.col("filing_to_construction_days").quantile(0.75).alias("p75_days"),
    ]
    if by is None:
        return d.select(aggs)
    return d.group_by(by).agg(aggs).filter(pl.col("n") >= min_n).sort("n", descending=True)


def print_summary(data_dir: Path | str | None = None) -> None:
    df = load_enriched(data_dir, tower_only=False, columns=TOWER_VIEW_COLUMNS + ["fcc_asr_number"])
    tw = df.filter(pl.col("is_tower_signal"))
    with pl.Config(tbl_rows=60, tbl_cols=-1, tbl_width_chars=200, tbl_hide_dataframe_shape=True, fmt_str_lengths=50):
        print(f"{df.height:,} cases; {tw.height:,} tower-signal filings")
        print("\nAttribution rate (tower signal)")
        print(entities.attribution_rate(df))
        print("\nbuild_kind x basis (tower signal)")
        print(tw.group_by("build_kind", "build_kind_basis").len("n").sort("n", descending=True))
        print("\nASR match (tower signal)")
        print(tw.group_by("asr_match_method").len("n").with_columns(share=(pl.col("n") / tw.height).round(3)).sort("n", descending=True))
        print("\nFiling -> construction lag, new builds (days)")
        print(lag_summary(tw, by=None))
        print(lag_summary(tw, by="entity").head(15))
        print(f"\nProjects: {tw['project_id'].n_unique():,} distinct among {tw.height:,} tower-signal filings")


BUILD_KINDS = ("modification", "new_build", "unknown")


def monthly_by_entity(df: pl.DataFrame, entities_: list[str]) -> pl.DataFrame:
    """Tower-signal filings per month x entity x build_kind (long, zero-filled over the months present)."""
    d = df.filter(pl.col("is_tower_signal") & pl.col("entity").is_in(entities_)).with_columns(
        month=pl.col("date_entered").dt.truncate("1mo")
    )
    counts = d.group_by("month", "entity", "build_kind").len("n")
    months = d.select(pl.col("month").unique())
    grid = months.join(pl.DataFrame({"entity": entities_}), how="cross").join(
        pl.DataFrame({"build_kind": list(BUILD_KINDS)}), how="cross"
    )
    return (
        grid.join(counts, on=["month", "entity", "build_kind"], how="left")
        .with_columns(pl.col("n").fill_null(0))
        .sort("entity", "month", "build_kind")
    )
