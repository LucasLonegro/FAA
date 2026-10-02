"""FCC Antenna Structure Registration (ASR): download, parse, and join to FAA cases.

Source: the complete weekly registration file of the FCC ULS/ASR public-access downloads,
https://data.fcc.gov/download/pub/uls/complete/r_tower.zip (rebuilt every Sunday; the daily
r_tow_{mon..sat}.zip files are incremental and not needed for a full refresh). It holds pipe-delimited
record files, one registration per RA row:

  RA  registration: status, dates, heights, structure type, latest FAA study number
  CO  coordinates (coordinate_type T = the structure; A = antenna-array members)
  EN  entities (entity_type O = owner, R = contact), with name and e-mail
  HS  history events (one dated row per action; the earliest is taken as the registration date)
  RE, SC  remarks and special conditions (unused)

Field positions follow the FCC ASR public-access definitions (PUBACC_RA / _CO / _EN / _HS). The fcc.gov
documentation pages refuse scripted access, so the positions below were checked against live rows: RA rows
carry 49 fields, e.g. registration 1252613 -> constructed 06/30/2006, overall height 90.2 m AGL, FAA study
2024-ASO-5144-OE. Heights are metres; dates are MM/DD/YYYY. Only the owner's e-mail *domain* is kept.

Status codes (RA field 8): C constructed, G granted (not yet built), I dismantled, T terminated,
A cancelled (applications whose purpose is CA).

Case join, in priority order: the case's fcc_asr_number; the ASR's latest FAA study number equal to the
case ASN; nearest registration within 50 m whose overall AGL height is within 10% of the case's (tower-ish
categories only).
"""

from __future__ import annotations

import io
import logging
import os
import re
import time
import zipfile
from pathlib import Path

import httpx
import polars as pl

from faa_oe import entities
from faa_oe.classify import classify
from faa_oe.geo import haversine_m, pairs_within
from faa_oe.ingest import _write_atomic
from faa_oe.paths import DataPaths

log = logging.getLogger(__name__)

ASR_URL = "https://data.fcc.gov/download/pub/uls/complete/r_tower.zip"
ZIP_NAME = "r_tower.zip"
FT_PER_M = 3.28084
PROXIMITY_M = 50.0
HEIGHT_TOL = 0.10
PROXIMITY_CATEGORIES = ("tower", "monopole", "cow", "antenna_mount")

RA_FIELDS = {
    3: "registration_number",
    5: "application_purpose",
    8: "status_code",
    9: "date_entered",
    11: "date_issued",
    12: "date_constructed",
    13: "date_dismantled",
    14: "date_action",
    24: "city",
    25: "state",
    30: "overall_height_agl_m",
    31: "overall_height_amsl_m",
    32: "asr_structure_type",
    33: "faa_determination_date",
    34: "faa_study_number",
}
CO_FIELDS = {3: "registration_number", 5: "coordinate_type", 6: "lat_d", 7: "lat_m", 8: "lat_s", 9: "lat_dir",
             11: "lon_d", 12: "lon_m", 13: "lon_s", 14: "lon_dir"}
EN_FIELDS = {3: "registration_number", 5: "entity_type", 9: "entity_name", 16: "email"}
HS_FIELDS = {3: "registration_number", 5: "date", 6: "event"}
STATUS = {"C": "constructed", "G": "granted", "I": "dismantled", "T": "terminated", "A": "cancelled"}
DATE_COLS = ("date_entered", "date_issued", "date_constructed", "date_dismantled", "date_action", "faa_determination_date")


def download(paths: DataPaths, max_age_days: float = 6.0, force: bool = False, url: str = ASR_URL) -> Path:
    """Fetch r_tower.zip into data/asr/ unless a copy younger than `max_age_days` is already there."""
    dest = paths.asr_dir / ZIP_NAME
    if dest.exists() and not force and (time.time() - dest.stat().st_mtime) < max_age_days * 86400:
        log.info("using cached %s", dest)
        return dest
    paths.asr_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".zip.part")
    log.info("downloading %s", url)
    with httpx.stream("GET", url, timeout=httpx.Timeout(60.0, read=300.0), follow_redirects=True,
                      headers={"User-Agent": "faa-oe research pipeline"}) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    with zipfile.ZipFile(tmp) as z:
        missing = {"RA.dat", "CO.dat", "EN.dat", "HS.dat"} - set(z.namelist())
        if missing:
            raise ValueError(f"{url} lacks {sorted(missing)}")
    os.replace(tmp, dest)
    return dest


def _read(z: zipfile.ZipFile, name: str, fields: dict[int, str]) -> pl.DataFrame:
    df = pl.read_csv(
        io.BytesIO(z.read(name)),
        separator="|",
        has_header=False,
        quote_char=None,
        infer_schema=False,
        encoding="utf8-lossy",
        truncate_ragged_lines=True,
        columns=list(fields),
    )
    return df.rename(dict(zip(df.columns, fields.values()))).with_columns(pl.all().str.strip_chars().replace("", None))


def _dms(d: str, m: str, s: str, hemi: str, neg: str) -> pl.Expr:
    v = pl.col(d).cast(pl.Float64) + pl.col(m).cast(pl.Float64) / 60 + pl.col(s).cast(pl.Float64) / 3600
    return pl.when(pl.col(hemi) == neg).then(-v).otherwise(v)


def normalize_asr_number(col: str | pl.Expr) -> pl.Expr:
    """'1252613', 'A1252613', '01252613', '1252613.0' -> '1252613'."""
    c = pl.col(col) if isinstance(col, str) else col
    digits = c.str.replace(r"\.0+$", "").str.replace_all(r"\D", "").str.strip_chars_start("0")
    return pl.when(digits.str.len_chars() > 0).then(digits)


_STUDY = re.compile(r"^(\d{2}|\d{4})-([A-Z]+)-0*(\d+)-(OE|NRA)$")


def normalize_study_number(s: str | None) -> str | None:
    """FAA study number as an ASN: '96-ACE-0407-OE' -> '1996-ACE-407-OE'."""
    m = _STUDY.match((s or "").strip().upper())
    if not m:
        return None
    y = int(m.group(1))
    if y < 100:
        y += 2000 if y < 50 else 1900
    return f"{y}-{m.group(2)}-{int(m.group(3))}-{m.group(4)}"


def parse_registrations(zip_path: Path | str) -> pl.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        ra = _read(z, "RA.dat", RA_FIELDS)
        co = _read(z, "CO.dat", CO_FIELDS)
        en = _read(z, "EN.dat", EN_FIELDS)
        hs = _read(z, "HS.dat", HS_FIELDS)

    coords = (
        co.filter(pl.col("coordinate_type") == "T")
        .select(
            "registration_number",
            latitude=_dms("lat_d", "lat_m", "lat_s", "lat_dir", "S"),
            longitude=_dms("lon_d", "lon_m", "lon_s", "lon_dir", "W"),
        )
        .unique("registration_number", keep="first")
    )
    owners = (
        en.filter(pl.col("entity_type") == "O")
        .select(
            "registration_number",
            asr_owner=pl.col("entity_name"),
            asr_owner_domain=pl.col("email").str.extract(r"@([^@\s]+)$").str.to_lowercase(),
        )
        .unique("registration_number", keep="first")
    )
    first_seen = (
        hs.with_columns(pl.col("date").str.strptime(pl.Date, "%m/%d/%Y", strict=False))
        .group_by("registration_number")
        .agg(first_history_date=pl.col("date").min())
    )
    studies = ra["faa_study_number"].unique().to_list()
    study_map = {s: normalize_study_number(s) for s in studies if s}
    return (
        ra.with_columns(
            [pl.col(c).str.strptime(pl.Date, "%m/%d/%Y", strict=False) for c in DATE_COLS]
            + [pl.col(c).cast(pl.Float64, strict=False) for c in ("overall_height_agl_m", "overall_height_amsl_m")]
        )
        .with_columns(
            status=pl.col("status_code").replace_strict(STATUS, default="unknown"),
            overall_height_agl_ft=(pl.col("overall_height_agl_m") * FT_PER_M).round(1),
            overall_height_amsl_ft=(pl.col("overall_height_amsl_m") * FT_PER_M).round(1),
            faa_asn=pl.col("faa_study_number").replace_strict(study_map, default=None, return_dtype=pl.String),
        )
        .join(first_seen, on="registration_number", how="left")
        .pipe(_estimate_registration_date)
        .join(coords, on="registration_number", how="left")
        .join(owners, on="registration_number", how="left")
        .drop("overall_height_agl_m", "overall_height_amsl_m", "first_history_date")
        .sort("registration_number")
    )


def _estimate_registration_date(ra: pl.DataFrame, bin_size: int = 500) -> pl.DataFrame:
    """Registration date from the sequential registration number.

    RA only carries the latest application and HS history starts in 1999, so a structure's own earliest
    known date is often a recent modification. Numbers are issued in order, so take the 10th percentile of
    the earliest known date within each block of `bin_size` numbers, force it to be non-decreasing, and use
    the smaller of that and the registration's own earliest date.
    """
    ra = ra.with_columns(
        _n=pl.col("registration_number").cast(pl.Int64, strict=False),
        _own=pl.min_horizontal("first_history_date", "date_entered", "date_issued"),
    )
    bins = (
        ra.group_by(_bin=pl.col("_n") // bin_size)
        .agg(_est=pl.col("_own").quantile(0.1, "lower").cast(pl.Date))
        .sort("_bin")
        .with_columns(pl.col("_est").cum_max())
    )
    return (
        ra.with_columns(_bin=pl.col("_n") // bin_size)
        .join(bins, on="_bin", how="left")
        .with_columns(registration_date=pl.min_horizontal("_own", "_est"))
        .drop("_n", "_own", "_bin", "_est")
    )


CASE_INPUT = ["asn", "fcc_asr_number", "latitude", "longitude", "agl_structure_height", "structure_type"]
ASR_JOIN_COLS = {
    "status": "asr_status",
    "asr_structure_type": "asr_structure_type",
    "registration_date": "asr_registration_date",
    "date_constructed": "asr_date_constructed",
    "date_dismantled": "asr_date_dismantled",
    "overall_height_agl_ft": "asr_height_agl_ft",
    "faa_asn": "asr_faa_asn",
    "asr_owner": "asr_owner",
    "asr_owner_domain": "asr_owner_domain",
}


def match_cases(cases: pl.DataFrame | pl.LazyFrame, asr: pl.DataFrame) -> pl.DataFrame:
    """One row per matched case: asn -> registration_number, match_method, match_distance_m."""
    c = cases.lazy().select(CASE_INPUT).pipe(classify).with_columns(asr_key=normalize_asr_number("fcc_asr_number")).collect()
    a = asr.select("registration_number", "faa_asn", "latitude", "longitude", "overall_height_agl_ft")

    by_number = (
        c.filter(pl.col("asr_key").is_not_null())
        .join(a.select("registration_number", "latitude", "longitude"), left_on="asr_key", right_on="registration_number",
              how="inner", suffix="_r")
        .select("asn", registration_number="asr_key", match_method=pl.lit("asr_number"),
                match_distance_m=_dist("latitude", "longitude", "latitude_r", "longitude_r"))
    )
    done = set(by_number["asn"].to_list())

    by_study = (
        c.filter(~pl.col("asn").is_in(done))
        .join(a.filter(pl.col("faa_asn").is_not_null()).select("registration_number", "faa_asn", "latitude", "longitude"),
              left_on="asn", right_on="faa_asn", how="inner", suffix="_r")
        .select("asn", "registration_number", match_method=pl.lit("faa_study"),
                match_distance_m=_dist("latitude", "longitude", "latitude_r", "longitude_r"))
        .unique("asn", keep="first")
    )
    done |= set(by_study["asn"].to_list())

    near = c.filter(
        ~pl.col("asn").is_in(done)
        & pl.col("category").is_in(PROXIMITY_CATEGORIES)
        & pl.col("agl_structure_height").is_not_null()
    ).select("asn", "latitude", "longitude", "agl_structure_height")
    by_prox = (
        pairs_within(near, a.filter(pl.col("overall_height_agl_ft").is_not_null()), PROXIMITY_M)
        .filter(
            (pl.col("agl_structure_height") - pl.col("overall_height_agl_ft_r")).abs()
            <= HEIGHT_TOL * pl.max_horizontal("agl_structure_height", "overall_height_agl_ft_r")
        )
        .sort("asn", "distance_m", "registration_number_r", descending=[False, False, True])
        .unique("asn", keep="first")
        .select("asn", registration_number="registration_number_r", match_method=pl.lit("proximity"),
                match_distance_m="distance_m")
    )
    return pl.concat([by_number, by_study, by_prox], how="vertical_relaxed")


def _dist(lat1: str, lon1: str, lat2: str, lon2: str) -> pl.Expr:
    return haversine_m(pl.col(lat1), pl.col(lon1), pl.col(lat2), pl.col(lon2)).round(1)


def case_asr_table(matches: pl.DataFrame, asr: pl.DataFrame) -> pl.DataFrame:
    """Matches enriched with the registration's dates, status, height and owner (plus the owner's entity)."""
    out = matches.join(
        asr.select("registration_number", *ASR_JOIN_COLS).rename(ASR_JOIN_COLS), on="registration_number", how="left"
    )
    return out.join(entities.owner_table(out), on=["asr_owner", "asr_owner_domain"], how="left", nulls_equal=True)


def run(data_dir: Path | str | None = None, refresh: bool = False) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Download (cached), parse to asr.parquet, match cases to case_asr.parquet."""
    paths = DataPaths.resolve(data_dir)
    zip_path = download(paths, force=refresh)
    asr = parse_registrations(zip_path)
    _write_atomic(asr, paths.asr)
    matches = match_cases(pl.scan_parquet(paths.cases), asr)
    case_asr = case_asr_table(matches, asr)
    _write_atomic(case_asr, paths.case_asr)
    log.info("asr: %s registrations, %s cases matched", f"{asr.height:,}", f"{case_asr.height:,}")
    return asr, case_asr


def match_summary(cases: pl.DataFrame, case_asr: pl.DataFrame) -> pl.DataFrame:
    """Per category: share of cases matched to ASR, split by method."""
    d = cases.join(case_asr.select("asn", "match_method"), on="asn", how="left")
    return (
        d.group_by("category")
        .agg(
            n=pl.len(),
            has_asr_number=pl.col("fcc_asr_number").is_not_null().mean().round(3),
            by_number=(pl.col("match_method") == "asr_number").fill_null(False).mean().round(3),
            by_study=(pl.col("match_method") == "faa_study").fill_null(False).mean().round(3),
            by_proximity=(pl.col("match_method") == "proximity").fill_null(False).mean().round(3),
            matched=pl.col("match_method").is_not_null().mean().round(3),
        )
        .sort("n", descending=True)
    )
