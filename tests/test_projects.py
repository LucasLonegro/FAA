from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl

from faa_oe.geo import haversine_scalar, pairs_within
from faa_oe.projects import assign_projects

LAT, LON = 40.0, -75.0
M = 1 / 111_320.0


def _frame(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={"asn": pl.String, "project_key": pl.String, "date_entered": pl.Date, "latitude": pl.Float64, "longitude": pl.Float64},
        orient="row",
    )


def test_projects_cluster_by_key_distance_and_window():
    d0 = date(2025, 3, 1)
    df = _frame(
        [
            ("a1", "SBA", d0, LAT, LON),
            ("a2", "SBA", d0 + timedelta(days=5), LAT + 200 * M, LON),
            ("a3", "SBA", d0 + timedelta(days=30), LAT - 300 * M, LON),
            ("late", "SBA", d0 + timedelta(days=31), LAT, LON),
            ("far", "SBA", d0 + timedelta(days=1), LAT + 600 * M, LON),
            ("other_key", "AMT", d0, LAT, LON),
            ("no_key", None, d0, LAT, LON),
            ("no_coords", "SBA", d0, None, None),
        ]
    )
    out = dict(assign_projects(df).select("asn", "project_id").iter_rows())
    assert out["a1"] == out["a2"] == out["a3"] == "a1"
    assert out["late"] == "late"
    assert out["far"] == "far"
    assert out["other_key"] == "other_key"
    assert out["no_key"] == "no_key"
    assert out["no_coords"] == "no_coords"
    sizes = dict(assign_projects(df).select("asn", "project_size").iter_rows())
    assert sizes["a1"] == 3 and sizes["late"] == 1


def test_anchor_distance_prevents_chaining():
    d0 = date(2025, 3, 1)
    df = _frame([(f"t{i}", "WINDCO", d0, LAT + i * 400 * M, LON) for i in range(4)])
    out = assign_projects(df)
    assert out["project_id"].to_list() == ["t0", "t0", "t2", "t2"]


def test_pairs_within_across_cell_edges_and_high_latitude():
    for lat in (0.0, 40.0, 70.0):
        left = pl.DataFrame({"id": [1], "latitude": [lat], "longitude": [-150.0]})
        lon_m = M / math.cos(math.radians(lat))
        pts = [(lat + dy * M, -150.0 + dx * lon_m) for dy, dx in [(49, 0), (0, 49), (-35, -35), (51, 0), (0, 60)]]
        right = pl.DataFrame({"id": list(range(5)), "latitude": [p[0] for p in pts], "longitude": [p[1] for p in pts]})
        got = pairs_within(left, right, 50.0)
        expected = {i for i, p in enumerate(pts) if haversine_scalar(lat, -150.0, *p) <= 50.0}
        assert set(got["id_r"].to_list()) == expected
        assert expected >= {0, 2}
