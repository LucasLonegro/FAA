from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from faa_oe.entities import (
    apply_owner_fallback,
    attribute,
    attribution_rate,
    base_name,
    default_rules,
    normalize_sponsor,
    owner_table,
    unmatched_report,
)


@pytest.mark.parametrize(
    "raw, norm",
    [
        ("American Towers LLC - (LG)", "AMERICAN TOWERS LLC"),
        ("American Towers LLC- KP", "AMERICAN TOWERS LLC"),
        ("* CELLCO PARTNERSHIP", "CELLCO PARTNERSHIP"),
        ("Crown Castle USA.NB", "CROWN CASTLE USA"),
        ("CROWN CASTLE USA MR", "CROWN CASTLE USA"),
        ("CROWN CASTLE USA (CK-MD)", "CROWN CASTLE USA"),
        ("CROWN CASTLE USA - M.B.", "CROWN CASTLE USA"),
        ("HARMONI TOWERS LLC-REGULATORY", "HARMONI TOWERS LLC"),
        ("AT&T ~ dc", "ATT"),
        ("AT&T (ST5066)", "ATT"),
        ("T-MOBILE WEST LLC - NORCAL", "T MOBILE WEST LLC"),
        ("VB-S1 ASSETS, LLC", "VB S1 ASSETS LLC"),
        ("Sun State Towers_MJ", "SUN STATE TOWERS"),
        ("PACIFIC GAS & ELECTRIC COMPANY", "PACIFIC GAS ELECTRIC COMPANY"),
        ("  ", None),
        (None, None),
    ],
)
def test_normalize_sponsor(raw, norm):
    assert normalize_sponsor(raw) == norm


@pytest.mark.parametrize(
    "sponsor, entity, ticker, entity_type",
    [
        ("SBA TOWERS", "SBA Communications", "SBAC", "tower_reit"),
        ("SBA Towers", "SBA Communications", "SBAC", "tower_reit"),
        ("CELLCO PARTNERSHIP - C", "Verizon", "VZ", "carrier"),
        ("* CELLCO PARTNERSHIP", "Verizon", "VZ", "carrier"),
        ("Verizon Wireless (VAW) LLC", "Verizon", "VZ", "carrier"),
        ("Alltel Corporation", "Verizon", "VZ", "carrier"),
        ("American Towers LLC - CS", "American Tower", "AMT", "tower_reit"),
        ("American Towers LLC - (LG)", "American Tower", "AMT", "tower_reit"),
        ("ATC - TM", "American Tower", "AMT", "tower_reit"),
        ("Crown Castle USA - XX", "Crown Castle", "CCI", "tower_reit"),
        ("CROWN CASTLE USA (MD)", "Crown Castle", "CCI", "tower_reit"),
        ("Crown Castle for AT&T - Cassandra Robbins", "Crown Castle", "CCI", "tower_reit"),
        ("CCATT LLC", "Crown Castle", "CCI", "tower_reit"),
        ("Pinnacle Towers LLC", "Crown Castle", "CCI", "tower_reit"),
        ("AT&T", "AT&T", "T", "carrier"),
        ("AT&T Mobility", "AT&T", "T", "carrier"),
        ("ATT-AR", "AT&T", "T", "carrier"),
        ("Los Angeles SMSA Limited Partnership", "AT&T", "T", "carrier"),
        ("T-MOBILE", "T-Mobile", "TMUS", "carrier"),
        ("T-MOBILE WEST CORPORATION", "T-Mobile", "TMUS", "carrier"),
        ("Dish Wireless L.L.C.", "EchoStar (DISH)", "SATS", "carrier"),
        ("US Cellular Corporation Regulatory", "Array Digital Infrastructure", "AD", "towerco"),
        ("Array Digital Infrastructure", "Array Digital Infrastructure", "AD", "towerco"),
        ("THE TOWERS, LLC", "The Towers (Vertical Bridge/Verizon JV)", None, "private_towerco"),
        ("VB-S1 ASSETS, LLC", "Vertical Bridge", None, "private_towerco"),
        ("VB BTS III, LLC", "Vertical Bridge", None, "private_towerco"),
        ("VERTICAL BRIDGE S3 ASSETS, LLC", "Vertical Bridge", None, "private_towerco"),
        ("TILLMAN INFRASTRUCTURE, LLC", "Tillman Infrastructure", None, "private_towerco"),
        ("HARMONI TOWERS LLC-REGULATORY", "Harmoni Towers", None, "private_towerco"),
        ("Skyway Towers, LLC", "Skyway Towers", None, "private_towerco"),
        ("TOWERNORTH DEVELOPMENT, LLC", "TowerNorth", None, "private_towerco"),
        ("PHOENIX TOWER INTERNATIONAL", "Phoenix Tower International", None, "private_towerco"),
        ("Diamond Towers IV LLC", "Diamond Communications", None, "private_towerco"),
        ("TowerCo 2013 LLC", "TowerCo", None, "private_towerco"),
        ("CTGI LLC - MD", "Communications Tower Group", None, "private_towerco"),
        ("Hemphill, LLC", "Hemphill", None, "private_towerco"),
        ("Tarpon Towers II, LLC", "Tarpon Towers", None, "private_towerco"),
        ("Horizon Tower Limited, LLC", "HORIZON TOWER", None, "private_towerco"),
        ("PACIFIC GAS & ELECTRIC COMPANY", "PACIFIC GAS ELECTRIC", None, "utility"),
        ("South Dakota DOT (Sioux Falls) (SD)", "SOUTH DAKOTA DOT", None, "government"),
        ("UNION PACIFIC RAILROAD COMPANY", "UNION PACIFIC RAILROAD", None, "other"),
        ("PALM-TECH CONSULTING, LLC", None, None, "consultant"),
        ("Tower Engineering Professionals", None, None, "consultant"),
        ("Leslie Lindeman", None, None, None),
        ("HEMPHILL SEMINARY, LLC", None, None, None),
    ],
)
def test_sponsor_rules(sponsor, entity, ticker, entity_type):
    m = default_rules().match_name(sponsor)
    assert (m.entity, m.ticker, m.entity_type) == (entity, ticker, entity_type)


def test_base_name():
    assert base_name("TARPON TOWERS II LLC") == "TARPON TOWERS"
    assert base_name("TOWERCO 2013 LLC") == "TOWERCO"
    assert base_name("CT CUBE L P D B A WEST CENTRAL WIRELESS") == "CT CUBE"
    assert base_name("LLC") == "LLC"


@pytest.mark.parametrize(
    "owner, domain, entity",
    [
        ("Pinnacle Towers LLC", "crowncastle.com", "Crown Castle"),
        ("Kentucky RSA No. 1 Partnership", "verizonwireless.com", "Verizon"),
        ("STC Five LLC", "t-mobile.com", "T-Mobile"),
        ("Atlas Tower 1- LLC", "atlastowers.com", "Atlas Tower"),
        ("American Towers LLC", None, "American Tower"),
        ("Optima Towers IV, LLC", "comcast.net", "OPTIMA TOWERS"),
        ("Free Bird Communications, LLC", "gmail.com", "FREE BIRD COMMUNICATIONS"),
        ("Sagebrush Cellular, Inc.", "nemont.coop", "SAGEBRUSH CELLULAR"),
        ("Acme Paging", "state.sd.us", "ACME PAGING"),
    ],
)
def test_owner_match(owner, domain, entity):
    assert default_rules().match_owner(owner, domain).entity == entity


def test_owner_domain_types():
    rs = default_rules()
    assert rs.match_owner("Acme Paging", "state.sd.us").entity_type == "government"
    assert rs.match_domain("gmail.com").entity is None
    assert rs.match_domain("mail.americantower.com").ticker == "AMT"


def _cases() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "asn": [f"2025-AEA-{i}-OE" for i in range(6)],
            "sponsor": ["SBA TOWERS", "Leslie Lindeman", "PALM-TECH CONSULTING, LLC", "Leslie Lindeman", "CELLCO PARTNERSHIP", None],
            "is_tower_signal": [True, True, True, True, True, False],
            "date_entered": [date(2025, 1, 1)] * 6,
        }
    )


def test_attribute_and_unmatched_report():
    df = attribute(_cases())
    assert df["entity"].to_list() == ["SBA Communications", None, None, None, "Verizon", None]
    assert df["entity_source"].to_list() == ["sponsor", None, None, None, "sponsor", None]
    rep = unmatched_report(df, top=5)
    assert rep["sponsor_norm"].to_list() == ["LESLIE LINDEMAN", "PALM TECH CONSULTING LLC"]
    assert rep["n"].to_list() == [2, 1]
    assert rep["consultant"].to_list() == [False, True]
    rate = attribution_rate(df)
    assert rate["n"][0] == 5 and rate["rate"][0] == 0.4


def test_owner_fallback_only_fills_gaps():
    df = attribute(_cases()).with_columns(
        asr_owner=pl.Series(["Crown Castle South LLC", "CitySwitch II, LLC", None, None, "American Towers LLC", None]),
        asr_owner_domain=pl.Series(["crowncastle.com", "cityswitch.com", None, None, "americantower.com", None]),
    )
    df = df.join(owner_table(df), on=["asr_owner", "asr_owner_domain"], how="left", nulls_equal=True)
    out = apply_owner_fallback(df)
    assert out["entity"].to_list() == ["SBA Communications", "CitySwitch", None, None, "Verizon", None]
    assert out["entity_source"].to_list() == ["sponsor", "asr_owner", None, None, "sponsor", None]
