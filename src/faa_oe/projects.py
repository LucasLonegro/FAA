"""Collapse multi-case projects: one sponsor filing several structures at one site in one go.

Greedy, per project key (entity, else normalised sponsor), in date_entered order: a case joins the nearest
open project whose first case lies within RADIUS_M and was entered at most WINDOW_DAYS earlier; otherwise
it starts a new project. Distances are to the project's first case, so projects cannot creep along a wind
farm or a transmission line. `project_id` is the ASN of the project's first case.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import date

import polars as pl

from faa_oe.geo import cell_size_deg, haversine_scalar

RADIUS_M = 500.0
WINDOW_DAYS = 30


def project_key() -> pl.Expr:
    return pl.coalesce("entity", "sponsor_norm").alias("project_key")


def _cluster(rows: list[tuple[str, date, float, float]], radius_m: float, window_days: int) -> dict[str, str]:
    dlat, dlon = cell_size_deg(radius_m)
    open_by_cell: dict[tuple[int, int], list[tuple[str, date, float, float]]] = defaultdict(list)
    out: dict[str, str] = {}
    for asn, d, lat, lon in rows:
        cy, cx = math.floor(lat / dlat), math.floor(lon / dlon)
        best, best_dist = None, radius_m
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                bucket = open_by_cell.get((cy + dy, cx + dx))
                if not bucket:
                    continue
                bucket[:] = [p for p in bucket if (d - p[1]).days <= window_days]
                for p in bucket:
                    dist = haversine_scalar(lat, lon, p[2], p[3])
                    if dist <= best_dist:
                        best, best_dist = p, dist
        if best is None:
            open_by_cell[(cy, cx)].append((asn, d, lat, lon))
            out[asn] = asn
        else:
            out[asn] = best[0]
    return out


def assign_projects(
    cases: pl.DataFrame, radius_m: float = RADIUS_M, window_days: int = WINDOW_DAYS
) -> pl.DataFrame:
    """asn -> project_id, project_size. Needs asn, project_key, date_entered, latitude, longitude."""
    usable = cases.filter(
        pl.col("project_key").is_not_null()
        & pl.col("date_entered").is_not_null()
        & pl.col("latitude").is_not_null()
        & pl.col("longitude").is_not_null()
    ).sort("project_key", "date_entered", "asn")
    mapping: dict[str, str] = {}
    for _, g in usable.group_by("project_key", maintain_order=True):
        rows = list(zip(g["asn"], g["date_entered"], g["latitude"], g["longitude"]))
        mapping.update(_cluster(rows, radius_m, window_days))
    ids = pl.DataFrame({"asn": list(mapping), "project_id": list(mapping.values())}, schema={"asn": pl.String, "project_id": pl.String})
    return (
        cases.select("asn")
        .join(ids, on="asn", how="left")
        .with_columns(pl.col("project_id").fill_null(pl.col("asn")))
        .with_columns(project_size=pl.len().over("project_id").cast(pl.Int32))
    )
