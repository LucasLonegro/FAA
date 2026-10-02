from __future__ import annotations

from datetime import date

import polars as pl

from faa_oe.build_kind import build_kind, signal_agreement

FILED = date(2025, 6, 1)
LAT, LON = 35.0, -97.0
M_PER_DEG_LAT = 111_320.0


def _case(asn: str, dlat_m: float = 0.0, **kw) -> dict:
    row = {
        "asn": asn,
        "category": "tower",
        "date_entered": FILED,
        "date_built": None,
        "structure_description": "Site name",
        "fcc_asr_number": None,
        "latitude": LAT + dlat_m / M_PER_DEG_LAT,
        "longitude": LON,
    }
    return row | kw


def _frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "asn": pl.String, "category": pl.String, "date_entered": pl.Date, "date_built": pl.Date,
            "structure_description": pl.String, "fcc_asr_number": pl.String, "latitude": pl.Float64,
            "longitude": pl.Float64,
        },
    )


def _asr(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={"asn": pl.String, "match_method": pl.String, "asr_registration_date": pl.Date, "asr_date_constructed": pl.Date},
        orient="row",
    )


def _kinds(cases: pl.DataFrame, asr: pl.DataFrame | None) -> dict[str, tuple[str, str]]:
    out = build_kind(cases, asr)
    return {a: (k, b) for a, k, b in out.select("asn", "build_kind", "build_kind_basis").iter_rows()}


def test_decision_rule():
    far = 10_000.0  # metres apart, so prior-case search stays out of the way unless intended
    cases = _frame(
        [
            _case("old_tower", far * 0, fcc_asr_number="1001000"),
            _case("built_after", far * 1, fcc_asr_number="1330000"),
            _case("granted_unbuilt", far * 2),
            _case("registered_long_ago", far * 3, fcc_asr_number="1200000"),
            _case("faa_built", far * 4, date_built=date(2010, 1, 1)),
            _case("keyword", far * 5, structure_description="Smith Hill - SIDE MNT"),
            _case("rawland_name", far * 6, structure_description="RAWLAND Smith Hill"),
            _case("prior", far * 7),
            _case("prior_earlier", far * 7 + 40, date_entered=date(2025, 1, 1)),
            _case("sibling_same_week", far * 8),
            _case("sibling_b", far * 8 + 30, date_entered=date(2025, 5, 28)),
            _case("unregistered_number", far * 9, fcc_asr_number="9999999"),
            _case("new_beats_prior", far * 10, fcc_asr_number="1330001"),
            _case("prior_for_new", far * 10 + 20, date_entered=date(2024, 1, 1)),
            _case("crane", far * 11, category="crane"),
        ]
    )
    asr = _asr(
        [
            ("old_tower", "asr_number", date(2001, 1, 1), date(2001, 6, 1)),
            ("built_after", "asr_number", date(2025, 8, 1), date(2026, 3, 1)),
            ("granted_unbuilt", "proximity", date(2025, 9, 1), None),
            ("registered_long_ago", "asr_number", date(2019, 1, 1), None),
            ("new_beats_prior", "asr_number", date(2025, 7, 1), date(2026, 1, 1)),
        ]
    )
    k = _kinds(cases, asr)
    assert k["old_tower"] == ("modification", "asr")
    assert k["built_after"] == ("new_build", "asr")
    assert k["granted_unbuilt"] == ("new_build", "asr")
    assert k["registered_long_ago"] == ("modification", "asr")
    assert k["faa_built"] == ("modification", "faa_built")
    assert k["keyword"] == ("modification", "keyword")
    assert k["rawland_name"] == ("unknown", "none")
    assert k["prior"] == ("modification", "prior_case")
    assert k["prior_earlier"] == ("unknown", "none")
    assert k["sibling_same_week"] == ("unknown", "none")
    assert k["unregistered_number"] == ("modification", "asr_number")
    assert k["new_beats_prior"] == ("new_build", "asr")
    assert "crane" not in k


def test_without_asr_and_agreement():
    cases = _frame([_case("a"), _case("b", 50, date_entered=date(2024, 1, 1), structure_description="colo")])
    out = build_kind(cases, None)
    assert dict(out.select("asn", "build_kind").iter_rows()) == {"a": "modification", "b": "modification"}
    agree = signal_agreement(out)
    row = agree.filter((pl.col("signal_a") == "bk_keyword") & (pl.col("signal_b") == "bk_prior_case"))
    assert row["n_both"][0] == 0
