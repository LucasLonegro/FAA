from __future__ import annotations

import os
import time
import zipfile
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from faa_oe import asr
from faa_oe.paths import DataPaths


def _ra(reg: str, status: str, entered: str, constructed: str, height_m: str, study: str, purpose: str = "NE") -> str:
    f = [""] * 49
    f[0:16] = ["RA", "REG", f"A{reg}", reg, "9" + reg, purpose, "", "I", status, entered, entered, entered,
               constructed, "", entered, "C"]
    f[24], f[25], f[28], f[30], f[31], f[32] = "SPRINGFIELD", "IL", height_m, height_m, "300.0", "LTOWER"
    f[33], f[34] = entered, study
    return "|".join(f)


def _co(reg: str, lat: tuple, lon: tuple, ctype: str = "T") -> str:
    return "|".join(["CO", "REG", f"A{reg}", reg, "9" + reg, ctype, *lat[:3], lat[3], "0", *lon[:3], lon[3], "0", "", ""])


def _en(reg: str, etype: str, name: str, email: str) -> str:
    f = [""] * 25
    f[0:10] = ["EN", "REG", f"A{reg}", reg, "9" + reg, etype, "L", "", "L0001", name]
    f[16] = email
    return "|".join(f)


def _hs(reg: str, d: str, event: str) -> str:
    return "|".join(["HS", "REG", f"A{reg}", reg, "9" + reg, d, event])


@pytest.fixture
def asr_zip(tmp_path: Path) -> Path:
    files = {
        "RA.dat": [
            _ra("1001000", "C", "05/01/2024", "06/30/2006", "90.2", "2024-ASO-5144-OE", "MD"),
            _ra("1330000", "G", "09/15/2025", "", "45.7", "25-AGL-0077-OE"),
            _ra("1330500", "C", "10/01/2025", "02/01/2026", "61.0", "2025-ASW-12-OE"),
            _ra("1330900", "A", "11/01/2025", "", "30.0", "N/A", "CA"),
        ],
        "CO.dat": [
            _co("1001000", ("37", "10", "55.4", "N"), ("88", "56", "43.7", "W")),
            _co("1001000", ("37", "11", "0.0", "N"), ("88", "57", "0.0", "W"), ctype="A"),
            _co("1330000", ("41", "0", "0.0", "N"), ("90", "0", "0.0", "W")),
            _co("1330500", ("32", "30", "0.0", "N"), ("97", "15", "0.0", "W")),
            _co("1330900", ("33", "0", "0.0", "S"), ("151", "0", "0.0", "E")),
        ],
        "EN.dat": [
            _en("1001000", "O", "Kentucky RSA No. 1 Partnership", "NetworkRegulatory@VerizonWireless.com"),
            _en("1001000", "R", "Kentucky RSA No. 1 Partnership", "someone@consultant.com"),
            _en("1330000", "O", "SBA Towers XI, LLC", "x@sbasite.com"),
            _en("1330500", "O", "Unknown Tower Owner LLC", ""),
        ],
        "HS.dat": [_hs("1001000", "05/17/2024", "Construction Notification"), _hs("1330000", "09/10/2025", "Application Received")],
    }
    path = tmp_path / "r_tower.zip"
    with zipfile.ZipFile(path, "w") as z:
        for name, lines in files.items():
            z.writestr(name, "\r\n".join(lines) + "\r\n")
    return path


def test_parse_registrations(asr_zip: Path):
    df = asr.parse_registrations(asr_zip)
    assert df.height == 4
    r = df.filter(pl.col("registration_number") == "1001000").row(0, named=True)
    assert r["status"] == "constructed"
    assert r["date_constructed"] == date(2006, 6, 30)
    assert r["overall_height_agl_ft"] == pytest.approx(295.9, abs=0.1)
    assert r["latitude"] == pytest.approx(37 + 10 / 60 + 55.4 / 3600)
    assert r["longitude"] == pytest.approx(-(88 + 56 / 60 + 43.7 / 3600))
    assert r["asr_owner"] == "Kentucky RSA No. 1 Partnership"
    assert r["asr_owner_domain"] == "verizonwireless.com"
    assert r["faa_asn"] == "2024-ASO-5144-OE"
    statuses = dict(df.select("registration_number", "status").iter_rows())
    assert statuses == {"1001000": "constructed", "1330000": "granted", "1330500": "constructed", "1330900": "cancelled"}
    assert df.filter(pl.col("registration_number") == "1330000")["faa_asn"][0] == "2025-AGL-77-OE"
    assert df.filter(pl.col("registration_number") == "1330900")["latitude"][0] == pytest.approx(-33.0)
    assert df.filter(pl.col("registration_number") == "1330500")["asr_owner_domain"][0] is None
    assert (df["registration_date"] <= df["date_entered"]).all()


@pytest.mark.parametrize(
    "raw, norm",
    [("1252613", "1252613"), ("A1252613", "1252613"), ("01252613", "1252613"), ("1252613.0", "1252613"), ("", None), (None, None)],
)
def test_normalize_asr_number(raw, norm):
    assert pl.select(asr.normalize_asr_number(pl.lit(raw, pl.String))).item() == norm


@pytest.mark.parametrize(
    "raw, asn",
    [("2024-ASO-5144-OE", "2024-ASO-5144-OE"), ("96-ACE-0407-OE", "1996-ACE-407-OE"), ("05-AGL-12-NRA", "2005-AGL-12-NRA"),
     ("N/A", None), (None, None)],
)
def test_normalize_study_number(raw, asn):
    assert asr.normalize_study_number(raw) == asn


def _cases() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "asn": ["2025-ASO-1-OE", "2025-AGL-77-OE", "2025-ASW-2-OE", "2025-ASW-3-OE", "2025-ASW-4-OE", "2025-ASW-5-OE"],
            "fcc_asr_number": ["A1001000", None, None, None, None, "1330500"],
            "latitude": [37.18, 41.0, 32.5 + 20 / 111_320, 32.5 + 20 / 111_320, 32.5 + 80 / 111_320, 32.5],
            "longitude": [-88.95, -90.0, -97.25, -97.25, -97.25, -97.25],
            "agl_structure_height": [296.0, 150.0, 195.0, 260.0, 200.0, 200.0],
            "structure_type": ["TOWER$ANTENNA", "TOWER", "POLE$MONO", "TOWER", "TOWER", "CRANE"],
        }
    )


def test_match_cases(asr_zip: Path):
    reg = asr.parse_registrations(asr_zip)
    m = asr.match_cases(_cases(), reg)
    got = {a: (r, meth) for a, r, meth in m.select("asn", "registration_number", "match_method").iter_rows()}
    assert got == {
        "2025-ASO-1-OE": ("1001000", "asr_number"),
        "2025-AGL-77-OE": ("1330000", "faa_study"),
        "2025-ASW-2-OE": ("1330500", "proximity"),
        "2025-ASW-5-OE": ("1330500", "asr_number"),
    }
    table = asr.case_asr_table(m, reg)
    v = table.filter(pl.col("asn") == "2025-ASO-1-OE").row(0, named=True)
    assert (v["asr_owner_entity"], v["asr_owner_ticker"]) == ("Verizon", "VZ")
    assert v["asr_date_constructed"] == date(2006, 6, 30)
    s = table.filter(pl.col("asn") == "2025-AGL-77-OE").row(0, named=True)
    assert s["asr_owner_ticker"] == "SBAC"


def test_download_uses_fresh_cache(tmp_path: Path, monkeypatch):
    paths = DataPaths(tmp_path)
    paths.asr_dir.mkdir(parents=True)
    cached = paths.asr_dir / asr.ZIP_NAME
    cached.write_bytes(b"cached")

    def boom(*a, **kw):
        raise AssertionError("network used")

    monkeypatch.setattr(asr.httpx, "stream", boom)
    assert asr.download(paths) == cached
    old = time.time() - 30 * 86400
    os.utime(cached, (old, old))
    with pytest.raises(AssertionError, match="network used"):
        asr.download(paths)
