from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone
from typing import Any, Callable

import polars as pl
from lxml import etree

INT_FIELDS = ("caseId", "asnSequence", "year")
FLOAT_FIELDS = (
    "siteElevationProposed",
    "aglStructureHeight",
    "aglStructureHeightDet",
    "amslOverallHeightProposed",
    "amslOverallHeightDet",
    "distanceFromNearestAirport",
    "directionFromNearestAirport",
    "latitude",
    "longitude",
)
DATE_FIELDS = ("dateEntered", "dateCompleted", "expirationDate", "dateBuilt")
DATETIME_FIELDS = ("receivedDate", "createdDate")
STR_FIELDS = (
    "asn",
    "caseType",
    "statusCode",
    "structureType",
    "structureDescription",
    "faaGeographyId",
    "nearestState",
    "nearestCity",
    "nearestAirportName",
    "locatorId",
    "sponsor",
    "sponsorCity",
    "sponsorState",
    "sponsorCountry",
    "recommendedMarkLightType",
    "latLongAccuracy",
    "fccAsrNumber",
)
ALIASES = {"id": "caseId"}

PII_DROP = {"sponsorPhone", "sponsorAddress1", "sponsorAddress2", "sponsorPostalCode", "sponsorFax"}
# Unknown future tags that look like contact details are dropped rather than parked in `extra`.
PII_PATTERN = re.compile(r"phone|fax|email|address|postal|zip", re.IGNORECASE)


def snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _to_datetime(s: str) -> datetime | None:
    dt = datetime.fromisoformat(s)
    return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


CONVERTERS: dict[str, tuple[Callable[[str], Any], pl.DataType]] = {}
for _f in INT_FIELDS:
    CONVERTERS[_f] = (int, pl.Int64())
for _f in FLOAT_FIELDS:
    CONVERTERS[_f] = (float, pl.Float64())
for _f in DATE_FIELDS:
    CONVERTERS[_f] = (lambda s: date.fromisoformat(s[:10]), pl.Date())
for _f in DATETIME_FIELDS:
    CONVERTERS[_f] = (_to_datetime, pl.Datetime("us", "UTC"))
for _f in STR_FIELDS:
    CONVERTERS[_f] = (str, pl.String())

_DERIVED: dict[str, pl.DataType] = {
    "sponsor_email_domain": pl.String(),
    "asn_year": pl.Int32(),
    "asn_region": pl.String(),
    "asn_type": pl.String(),
    "extra": pl.String(),
}
_LEADING = ("asn", "asn_year", "asn_region", "asn_sequence", "asn_type", "case_type", "case_id")
_all = {snake(k): dt for k, (_, dt) in CONVERTERS.items()} | _DERIVED
SCHEMA: dict[str, pl.DataType] = {k: _all[k] for k in _LEADING} | {
    k: v for k, v in _all.items() if k not in _LEADING
}

_ASN = re.compile(r"^(\d{4})-([A-Z]+)-(\d+)-(OE|NRA)$")


def _convert(field: str, text: str) -> Any:
    fn, _ = CONVERTERS[field]
    try:
        return fn(text)
    except ValueError:
        return None


def _case_row(case: etree._Element) -> dict[str, Any]:
    row: dict[str, Any] = {}
    raw: dict[str, str] = {}
    extra: dict[str, str] = {}
    for el in case:
        if not isinstance(el.tag, str):
            continue
        tag = ALIASES.get(el.tag, el.tag)
        text = (el.text or "").strip() if len(el) == 0 else etree.tostring(el, encoding="unicode")
        if not text:
            continue
        if tag == "sponsorEmail":
            if "@" in text:
                row["sponsor_email_domain"] = text.rsplit("@", 1)[1].strip().lower() or None
        elif tag in PII_DROP:
            continue
        elif tag in CONVERTERS:
            raw[tag] = text
            row[snake(tag)] = _convert(tag, text)
        elif not PII_PATTERN.search(tag):
            extra[tag] = text

    # NRA cases carry createdDate (a timestamp) instead of dateEntered; use its local calendar date.
    if row.get("date_entered") is None and "createdDate" in raw:
        row["date_entered"] = _convert("dateEntered", raw["createdDate"])

    m = _ASN.match(row.get("asn") or "")
    if m:
        row["asn_year"] = int(m.group(1))
        row["asn_region"] = m.group(2)
        # The ASN suffix names the sequence series; it differs from <caseType> for on-airport NRA-series
        # cases, which the OE endpoint returns with caseType=OE.
        row["asn_type"] = m.group(4)
        row.setdefault("case_type", m.group(4))
    if extra:
        row["extra"] = json.dumps(extra, sort_keys=True)
    return row


def parse_root(xml: bytes) -> etree._Element:
    parser = etree.XMLParser(huge_tree=True, resolve_entities=False, no_network=True)
    return etree.fromstring(xml, parser)


def count_cases(xml: bytes) -> int:
    return sum(1 for el in parse_root(xml) if isinstance(el.tag, str))


def parse_cases(xml: bytes) -> pl.DataFrame:
    rows = [_case_row(c) for c in parse_root(xml) if isinstance(c.tag, str)]
    return pl.DataFrame(rows, schema=SCHEMA) if rows else empty_frame()


def empty_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=SCHEMA)
