from __future__ import annotations

import math

import polars as pl

EARTH_RADIUS_M = 6_371_008.8
M_PER_DEG_LAT = 111_320.0
# Longitude cells are sized for this latitude so a cell is never narrower than the radius anywhere in the
# data (northern Alaska is ~71.4N); lower latitudes just get a few more candidate pairs.
_MAX_ABS_LAT = 75.0


def haversine_m(lat1: pl.Expr, lon1: pl.Expr, lat2: pl.Expr, lon2: pl.Expr) -> pl.Expr:
    p1, p2 = lat1.radians(), lat2.radians()
    dphi = p2 - p1
    dlmb = (lon2 - lon1).radians()
    a = (dphi / 2).sin().pow(2) + p1.cos() * p2.cos() * (dlmb / 2).sin().pow(2)
    return 2 * EARTH_RADIUS_M * a.sqrt().arcsin()


def haversine_scalar(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def cell_size_deg(radius_m: float) -> tuple[float, float]:
    dlat = radius_m / M_PER_DEG_LAT
    return dlat, radius_m / (M_PER_DEG_LAT * math.cos(math.radians(_MAX_ABS_LAT)))


def with_cells(df: pl.LazyFrame, radius_m: float, lat: str = "latitude", lon: str = "longitude") -> pl.LazyFrame:
    dlat, dlon = cell_size_deg(radius_m)
    return df.with_columns(
        _cy=(pl.col(lat) / dlat).floor().cast(pl.Int64),
        _cx=(pl.col(lon) / dlon).floor().cast(pl.Int64),
    )


def pairs_within(
    left: pl.DataFrame | pl.LazyFrame,
    right: pl.DataFrame | pl.LazyFrame,
    radius_m: float,
    right_suffix: str = "_r",
) -> pl.DataFrame:
    """All (left, right) row pairs within `radius_m`, via a grid join on 3x3 neighbouring cells.

    Both frames need `latitude` and `longitude`; every other column is carried through, right-hand columns
    get `right_suffix`. Adds `distance_m`.
    """
    lat, lon = "latitude", "longitude"
    rlat, rlon = f"{lat}{right_suffix}", f"{lon}{right_suffix}"
    l = with_cells(left.lazy().filter(pl.col(lat).is_not_null() & pl.col(lon).is_not_null()), radius_m)
    r = with_cells(right.lazy().filter(pl.col(lat).is_not_null() & pl.col(lon).is_not_null()), radius_m)
    offsets = pl.LazyFrame({"_dy": [-1, -1, -1, 0, 0, 0, 1, 1, 1], "_dx": [-1, 0, 1] * 3})
    l = l.join(offsets, how="cross").with_columns(_cy=pl.col("_cy") + pl.col("_dy"), _cx=pl.col("_cx") + pl.col("_dx"))
    r = r.rename({c: f"{c}{right_suffix}" for c in r.collect_schema().names() if c not in ("_cy", "_cx")})
    return (
        l.join(r, on=["_cy", "_cx"], how="inner")
        .with_columns(distance_m=haversine_m(pl.col(lat), pl.col(lon), pl.col(rlat), pl.col(rlon)))
        .filter(pl.col("distance_m") <= radius_m)
        .drop("_cy", "_cx", "_dy", "_dx")
        .collect()
    )
