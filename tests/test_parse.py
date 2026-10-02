from __future__ import annotations

import json
from datetime import date, datetime, timezone

import polars as pl

from conftest import case_xml, fixture_xml
from faa_oe.parse import SCHEMA, count_cases, parse_cases

PII_COLUMNS = {"sponsor_phone", "sponsor_address1", "sponsor_address2", "sponsor_postal_code", "sponsor_email"}


def test_live_fixture_count_and_keys():
    xml = fixture_xml("oe_tx_2026-09-01_07.xml.gz")
    df = parse_cases(xml)
    assert df.height == 625 == count_cases(xml)
    assert df["asn"].n_unique() == 625
    assert df["date_entered"].min() == date(2026, 9, 1)
    assert df["date_entered"].max() <= date(2026, 9, 7)
    assert set(df["case_type"].unique()) == {"OE"}
    assert set(df["asn_region"].unique()) == {"ASW", "WTW"}
    assert (df["asn_year"] == 2026).all()
    assert (df["structure_type"] == "WINDMILL").sum() == 470


def test_dtypes():
    df = parse_cases(fixture_xml("oe_tx_2026-09-01_07.xml.gz"))
    assert df.schema == pl.Schema(SCHEMA)
    assert df.schema["date_entered"] == pl.Date
    assert df.schema["date_completed"] == pl.Date
    assert df.schema["received_date"] == pl.Datetime("us", "UTC")
    assert df.schema["agl_structure_height"] == pl.Float64
    assert df.schema["latitude"] == pl.Float64
    assert df.schema["asn_sequence"] == pl.Int64
    row = df.filter(pl.col("asn") == "2026-ASW-14575-OE").row(0, named=True)
    assert row["received_date"] == datetime(2026, 9, 1, 14, 38, 41, 42000, tzinfo=timezone.utc)
    assert row["agl_structure_height"] == 112.0
    assert row["date_completed"] == date(2026, 9, 8)
    assert row["expiration_date"] == date(2028, 3, 8)
    assert df["date_completed"].null_count() == 625 - 129


def test_empty_tags_are_null_and_date_built_parsed():
    df = parse_cases(fixture_xml("oe_ny_2018-03.xml.gz"))
    assert df.height == 40
    assert df["date_built"].drop_nulls().len() == 20
    old = parse_cases(fixture_xml("oe_ny_2012-03.xml.gz"))
    assert old["expiration_date"].null_count() == 2
    assert old["fcc_asr_number"].to_list() == ["1207798", "1004173"]


def test_pii_dropped_and_email_domain_kept():
    xml = case_xml(
        [
            {
                "id": "42",
                "asn": "2021-ACE-1349-OE",
                "asnSequence": "1349",
                "sponsor": "Example Towers LLC",
                "sponsorAddress1": "1 Main St",
                "sponsorCity": "Boca Raton",
                "sponsorState": "FL",
                "sponsorPostalCode": "33431",
                "sponsorEmail": " Jane.Doe@Example-Towers.COM ",
                "sponsorPhone": "555-0100",
                "sponsorFaxNumber": "555-0101",
                "newFieldFromFaa": "kept",
                "aglStructureHeight": "not-a-number",
            }
        ]
    )
    df = parse_cases(xml)
    assert PII_COLUMNS.isdisjoint(df.columns)
    row = df.row(0, named=True)
    assert row["sponsor_email_domain"] == "example-towers.com"
    assert row["sponsor"] == "Example Towers LLC"
    assert (row["sponsor_city"], row["sponsor_state"]) == ("Boca Raton", "FL")
    assert row["case_id"] == 42
    assert row["agl_structure_height"] is None
    assert json.loads(row["extra"]) == {"newFieldFromFaa": "kept"}
    dump = df.write_csv().lower()
    assert "555" not in dump and "jane" not in dump and "main st" not in dump


def test_nra_created_date_backfills_date_entered():
    xml = case_xml(
        [{"asn": "2025-ASW-97-NRA", "asnSequence": "97", "createdDate": "2025-01-06T23:20:24-05:00"}],
        tag="NRACase",
    )
    row = parse_cases(xml).row(0, named=True)
    assert row["date_entered"] == date(2025, 1, 6)
    assert row["created_date"] == datetime(2025, 1, 7, 4, 20, 24, tzinfo=timezone.utc)
    assert row["case_type"] == row["asn_type"] == "NRA"


def test_asn_type_follows_suffix_not_case_type():
    xml = case_xml([{"asn": "2026-ASW-1-NRA", "asnSequence": "1", "caseType": "OE"}])
    row = parse_cases(xml).row(0, named=True)
    assert (row["case_type"], row["asn_type"], row["asn_region"], row["asn_year"]) == ("OE", "NRA", "ASW", 2026)


def test_empty_response():
    df = parse_cases(b'<?xml version="1.0" encoding="UTF-8"?><caseList/>')
    assert df.height == 0 and df.schema == pl.Schema(SCHEMA)
