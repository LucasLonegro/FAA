from __future__ import annotations

from pathlib import Path

import polars as pl

from faa_oe.paths import DataPaths

KEYS = ["asn_region", "asn_year", "asn_type"]


def load_cases(data_dir: Path | str | None = None) -> pl.DataFrame:
    return pl.read_parquet(DataPaths.resolve(data_dir).cases)


def load_history(data_dir: Path | str | None = None) -> pl.DataFrame:
    return pl.read_parquet(DataPaths.resolve(data_dir).history)


def status_group() -> pl.Expr:
    return pl.col("status_code").str.split("-").list.first().fill_null("NONE").alias("status_group")


def structure_group() -> pl.Expr:
    """Coarse structure type: text before '$', with anything wind-related folded into WINDMILL."""
    st = pl.col("structure_type").fill_null("UNKNOWN").str.to_uppercase()
    return (
        pl.when(st.str.contains("WIND"))
        .then(pl.lit("WINDMILL"))
        .otherwise(st.str.split("$").list.first().str.strip_chars())
        .alias("structure_group")
    )


def completeness(cases: pl.DataFrame) -> pl.DataFrame:
    """Per ASN region x year x ASN series (OE/NRA suffix): share of the issued sequence numbers we hold.

    Sequences are issued consecutively per region/year/series. The NRA series is split across both endpoints
    (the OE endpoint serves on-airport NRA-series cases with caseType=OE), so group on the suffix. Holes are
    cases the API no longer serves (withdrawn, deleted or never published). The max sequence itself is a lower
    bound on cases issued.
    """
    base = cases.group_by(KEYS).agg(
        n_cases=pl.len(),
        n_distinct_seq=pl.col("asn_sequence").n_unique(),
        min_seq=pl.col("asn_sequence").min(),
        max_seq=pl.col("asn_sequence").max(),
    )
    status = (
        cases.with_columns(status_group())
        .group_by([*KEYS, "status_group"])
        .len()
        .pivot(on="status_group", index=KEYS, values="len")
        .fill_null(0)
    )
    status = status.rename({c: f"n_{c.lower()}" for c in status.columns if c not in KEYS})
    return (
        base.with_columns(completeness=(pl.col("n_distinct_seq") / pl.col("max_seq")).round(3))
        .join(status, on=KEYS, how="left")
        .sort(KEYS)
    )


def completeness_matrix(cases: pl.DataFrame, asn_type: str = "OE") -> pl.DataFrame:
    """Wide region x year view of completeness for one ASN series."""
    c = completeness(cases).filter(pl.col("asn_type") == asn_type)
    years = sorted(c["asn_year"].unique().to_list())
    wide = c.pivot(on="asn_year", index="asn_region", values="completeness", sort_columns=True)
    return wide.select("asn_region", *[str(y) for y in years]).sort("asn_region")


def weekly_counts(cases: pl.DataFrame, by: str | pl.Expr | None = "asn_region") -> pl.DataFrame:
    """Cases per ISO week (Monday start) of date_entered, long format."""
    week = pl.col("date_entered").dt.truncate("1w").alias("week")
    keys = [week] if by is None else [week, by]
    return cases.filter(pl.col("date_entered").is_not_null()).group_by(keys).len("n").sort(pl.all())


def status_counts(cases: pl.DataFrame) -> pl.DataFrame:
    return cases.group_by("asn_type", "status_code").len("n").sort("n", descending=True)


def print_report(data_dir: Path | str | None = None, long: bool = False) -> None:
    cases = load_cases(data_dir)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250, tbl_hide_dataframe_shape=True):
        print(f"{cases.height:,} cases, date_entered {cases['date_entered'].min()} .. {cases['date_entered'].max()}\n")
        if long:
            print(completeness(cases))
        for t in ("OE", "NRA"):
            print(f"\nCompleteness (distinct asnSequence / max asnSequence), {t}-series ASNs, region x year")
            print(completeness_matrix(cases, t))
            n = cases.filter(pl.col("asn_type") == t).pivot(
                on="asn_year", index="asn_region", values="asn", aggregate_function="len", sort_columns=True
            )
            print(f"\nCase counts, {t}-series ASNs")
            print(n.sort("asn_region"))
