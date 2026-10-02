from __future__ import annotations

import random
from datetime import date, timedelta

import polars as pl
import pytest

from faa_oe import features as f

FILED = date(2025, 6, 1)
LAT, LON = 35.0, -97.0
M_PER_DEG_LAT = 111_320.0
FAR = 10_000.0

SCHEMA = {
    "asn": pl.String, "date_entered": pl.Date, "category": pl.String, "is_tower_signal": pl.Boolean,
    "sponsor_norm": pl.String, "entity": pl.String, "ticker": pl.String, "entity_type": pl.String,
    "entity_source": pl.String, "structure_description": pl.String, "date_built": pl.Date, "latitude": pl.Float64,
    "longitude": pl.Float64, "agl_structure_height": pl.Float64, "asr_registration_date": pl.Date,
    "asr_date_constructed": pl.Date, "asr_date_dismantled": pl.Date, "build_kind": pl.String, "project_id": pl.String,
}


def _case(asn: str, i: float = 0.0, **kw) -> dict:
    row = dict.fromkeys(SCHEMA)
    row |= {
        "asn": asn, "date_entered": FILED, "category": "tower", "is_tower_signal": True, "sponsor_norm": "X",
        "structure_description": "Site name", "latitude": LAT + i * FAR / M_PER_DEG_LAT, "longitude": LON,
        "agl_structure_height": 200.0,
    }
    return row | kw


def _frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=SCHEMA)


def _kinds(rows: list[dict], **kw) -> dict[str, tuple[str, str]]:
    out = f.pit_kind(_frame(rows), **kw)
    return {a: (k, b) for a, k, b in out.select("asn", "pit_kind", "pit_basis").iter_rows()}


def test_pit_kind_uses_only_evidence_before_filing():
    d = lambda n: FILED + timedelta(days=n)
    k = _kinds([
        _case("asr_after", 0, asr_registration_date=d(60), asr_date_constructed=d(300)),
        _case("asr_long_before", 1, asr_registration_date=d(-800)),
        _case("asr_just_before", 2, asr_registration_date=d(-100)),
        _case("built_before", 3, asr_registration_date=d(-100), asr_date_constructed=d(-40)),
        _case("built_just_before", 4, asr_date_constructed=d(-10)),
        _case("dismantled_before", 5, asr_registration_date=d(-3000), asr_date_constructed=d(-2900),
              asr_date_dismantled=d(-20)),
        _case("dismantled_after", 6, asr_registration_date=d(-3000), asr_date_dismantled=d(200)),
        _case("faa_built_before", 7, date_built=d(-400)),
        _case("faa_built_after", 8, date_built=d(200)),
        _case("keyword", 9, structure_description="Smith Hill - COLO"),
        _case("prior", 10),
        _case("prior_r", 10 + 40 / FAR, date_entered=d(-90), category="antenna_mount", is_tower_signal=False),
        _case("prior_recent", 11),
        _case("prior_recent_r", 11 + 40 / FAR, date_entered=d(-30)),
        _case("prior_later", 12),
        _case("prior_later_r", 12 + 40 / FAR, date_entered=d(120)),
        _case("nothing", 13),
    ])
    assert k["asr_after"] == ("new_or_unknown", "none")
    assert k["asr_long_before"] == ("modification", "asr_registered")
    assert k["asr_just_before"] == ("new_or_unknown", "none")
    assert k["built_before"] == ("modification", "asr_constructed")
    assert k["built_just_before"][0] == "new_or_unknown"
    assert k["dismantled_before"][0] == "new_or_unknown"
    assert k["dismantled_after"] == ("modification", "asr_registered")
    assert k["faa_built_before"] == ("modification", "faa_built")
    assert k["faa_built_after"][0] == "new_or_unknown"
    assert k["keyword"] == ("modification", "keyword")
    assert k["prior"] == ("modification", "prior_case")
    assert k["prior_recent"][0] == "new_or_unknown"
    assert k["prior_later"][0] == "new_or_unknown"
    assert k["prior_later_r"] == ("modification", "prior_case")
    assert k["nothing"] == ("new_or_unknown", "none")
    assert "prior_r" not in k


def test_prior_lookback():
    rows = [_case("a"), _case("a_r", 40 / FAR, date_entered=FILED - timedelta(days=900))]
    assert _kinds(rows)["a"][0] == "modification"
    assert _kinds(rows, prior_lookback_days=730)["a"][0] == "new_or_unknown"


ENTITIES = [
    ("American Tower", "AMT", "tower_reit"), ("Crown Castle", "CCI", "tower_reit"), ("Verizon", "VZ", "carrier"),
    ("AT&T", "T", "carrier"), ("Vertical Bridge", None, "private_towerco"), (f.TOWERS_JV, None, "private_towerco"),
]


def _synthetic(n: int = 600, seed: int = 7) -> pl.DataFrame:
    rng = random.Random(seed)
    start = date(2023, 1, 1)
    rows = []
    for i in range(n):
        de = start + timedelta(days=rng.randrange(0, 720))
        ent = rng.choice(ENTITIES + [(None, None, None)])
        src = rng.choice(["sponsor", "sponsor", "asr_owner"]) if ent[0] else None
        reg = de + timedelta(days=rng.randrange(-4000, 600)) if rng.random() < 0.8 else None
        built = reg + timedelta(days=rng.randrange(0, 500)) if reg and rng.random() < 0.7 else None
        rows.append(_case(
            f"c{i}", 0,
            date_entered=de,
            category=rng.choice(["tower", "tower", "monopole", "antenna_mount"]),
            latitude=LAT + rng.randrange(0, 60) * 0.0004,
            longitude=LON + rng.randrange(0, 40) * 0.0004,
            entity=ent[0], ticker=ent[1], entity_type=ent[2], entity_source=src,
            sponsor_norm=ent[0] or rng.choice(["JOE", "ANN"]),
            asr_registration_date=reg, asr_date_constructed=built,
            asr_date_dismantled=built + timedelta(days=400) if built and rng.random() < 0.1 else None,
            date_built=de + timedelta(days=rng.randrange(-900, 400)) if rng.random() < 0.1 else None,
            agl_structure_height=float(rng.randrange(60, 400)),
            build_kind="new_build" if reg and reg > de else "modification",
        ))
    df = _frame(rows).with_columns(is_tower_signal=pl.col("category").is_in(["tower", "monopole"]))
    return df.with_columns(project_id=pl.col("asn"))


def _as_known_at(cases: pl.DataFrame, cutoff: date) -> pl.DataFrame:
    """The database as it stood on `cutoff`: later filings absent, later ASR / FAA events not yet recorded."""
    later = lambda c: pl.when(pl.col(c) <= cutoff).then(pl.col(c))
    reg_unknown = pl.col("asr_registration_date").is_null() | (pl.col("asr_registration_date") > cutoff)
    return cases.filter(pl.col("date_entered") <= cutoff).with_columns(
        later("asr_registration_date"), later("asr_date_constructed"), later("asr_date_dismantled"), later("date_built"),
        entity=pl.when((pl.col("entity_source") == "asr_owner") & reg_unknown).then(None).otherwise(pl.col("entity")),
    )


def _daily(cases: pl.DataFrame) -> pl.DataFrame:
    return cases.group_by("date_entered").agg(n=pl.len())


@pytest.mark.parametrize("cutoff", [date(2023, 6, 30), date(2024, 3, 31), date(2024, 9, 30)])
def test_quarterly_pit_features_do_not_depend_on_later_dates(cutoff):
    full = _synthetic()
    known = _as_known_at(full, cutoff)
    cols = ["period", "basis", "level", "name", "category", "kind", "n", "n_hw", "n_projects"]
    pick = lambda p: p.filter((pl.col("basis") == "pit") & (pl.col("period") <= cutoff)).select(cols).sort(cols[:6])

    a = pick(f.panel(f.prepare(full), _daily(full), "Q"))
    b = pick(f.panel(f.prepare(known), _daily(known), "Q"))
    assert a.height > 50
    assert a.equals(b)


def test_hindsight_basis_does_see_the_future():
    full = _synthetic()
    cutoff = date(2023, 12, 31)
    known = _as_known_at(full, cutoff).with_columns(
        build_kind=pl.when(pl.col("asr_registration_date").is_null()).then(pl.lit("unknown")).otherwise("build_kind")
    )
    pick = lambda p: p.filter((pl.col("basis") == "hindsight") & (pl.col("period") <= cutoff)).sort("period", "level", "name", "category", "kind")
    a = pick(f.panel(f.prepare(full), _daily(full), "Q"))
    b = pick(f.panel(f.prepare(known), _daily(known), "Q"))
    assert not a.equals(b)


def test_entity_pit_ignores_asr_owner_fallback_and_groups():
    rows = [
        _case("vz", 0, entity="Verizon", ticker="VZ", entity_type="carrier", entity_source="sponsor"),
        _case("jv", 1, entity=f.TOWERS_JV, entity_type="private_towerco", entity_source="sponsor"),
        _case("owner", 2, entity="Crown Castle", ticker="CCI", entity_type="tower_reit", entity_source="asr_owner"),
    ]
    cases = _frame(rows)
    p = f.panel(f.prepare(cases), _daily(cases), "Q")
    n = lambda name, basis: p.filter((pl.col("name") == name) & (pl.col("basis") == basis) & (pl.col("kind") == "all")
                                     & (pl.col("category") == "all"))["n"].sum()
    assert n("carriers", "pit") == 1
    assert n("vz_incl_towers_jv", "pit") == 2
    assert n("carriers_incl_towers_jv", "pit") == 2
    assert n("private_builders", "pit") == 1
    assert n("listed_reits", "pit") == 0
    assert n("listed_reits", "hindsight") == 1
    assert n("all", "pit") == 3


def test_series_zero_fills_and_complete_flag():
    rows = [_case("a", 0, date_entered=date(2024, 1, 5)), _case("b", 1, date_entered=date(2024, 7, 5))]
    cases = _frame(rows)
    daily = pl.DataFrame({"date_entered": [date(2024, 1, 5), date(2024, 7, 5), date(2024, 8, 1)], "n": [1, 1, 1]})
    p = f.panel(f.prepare(cases), daily, "Q")
    s = f.series(p, "all", kind="all", complete_only=False)
    assert s["period"].to_list() == [date(2024, 1, 1), date(2024, 4, 1), date(2024, 7, 1)]
    assert s["value"].to_list() == [1.0, 0.0, 1.0]
    assert f.series(p, "all", kind="all")["period"].to_list() == [date(2024, 1, 1)]


def test_default_start_skips_gap():
    q = [date(2022, 1, 1), date(2022, 4, 1), date(2022, 7, 1), date(2022, 10, 1), date(2023, 1, 1), date(2023, 4, 1)]
    p = pl.DataFrame({"period": q, "level": "coverage", "n": [100, 2, 100, 100, 100, 90], "complete": True})
    assert f.default_start(p) == date(2022, 7, 1)
    p = p.filter(pl.col("period") != date(2022, 10, 1))
    assert f.default_start(p) == date(2023, 1, 1)
