from __future__ import annotations

from conftest import case_xml, fixture_xml
from faa_oe.coverage import completeness, completeness_matrix, structure_group, weekly_counts
from faa_oe.parse import parse_cases


def test_completeness_counts_gaps():
    df = parse_cases(
        case_xml(
            [
                {"asn": f"2025-ASW-{s}-OE", "asnSequence": str(s), "dateEntered": "2025-01-06", "statusCode": st}
                for s, st in [(1, "DET-DNE"), (2, "WRK-Part77"), (4, "DET-DNH"), (10, "NPF")]
            ]
        )
    )
    row = completeness(df).row(0, named=True)
    assert (row["n_cases"], row["max_seq"], row["completeness"]) == (4, 10, 0.4)
    assert (row["n_det"], row["n_wrk"], row["n_npf"]) == (2, 1, 1)
    assert completeness_matrix(df, "OE").columns == ["asn_region", "2025"]


def test_weekly_and_structure_group_on_fixture():
    df = parse_cases(fixture_xml("oe_tx_2026-09-01_07.xml.gz"))
    assert weekly_counts(df, by="asn_region")["n"].sum() == 625
    counts = dict(df.with_columns(structure_group())["structure_group"].value_counts().iter_rows())
    assert counts["WINDMILL"] == 470
    assert "TOWER" in counts and not any("$" in k for k in counts)
