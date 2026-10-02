from __future__ import annotations

from datetime import date

import polars as pl

from faa_oe import truth
from faa_oe.truth import Concept


def _fact(start: str, end: str, val: float, filed: str, form: str = "10-Q") -> dict:
    return {"start": start, "end": end, "val": val, "filed": filed, "form": form}


def _facts(by_concept: dict[str, list[dict]]) -> dict:
    return {"facts": {"us-gaap": {c: {"units": {"USD": v}} for c, v in by_concept.items()}}}


def test_quarters_from_ytd_first_filed():
    facts = _facts({"Capex": [
        _fact("2024-01-01", "2024-03-31", 10, "2024-05-01"),
        _fact("2024-01-01", "2024-06-30", 25, "2024-08-01"),
        _fact("2024-01-01", "2024-06-30", 26, "2025-08-01"),  # later recast: ignored
        _fact("2024-01-01", "2024-09-30", 45, "2024-11-01"),
        _fact("2024-01-01", "2024-12-31", 70, "2025-02-15", "10-K"),
        _fact("2024-01-01", "2024-12-31", 99, "2025-03-01", "8-K"),  # not a periodic report
        _fact("2025-01-01", "2025-03-31", 12, "2025-05-01"),
    ]})
    q = truth.quarterly_from_ytd(truth.ytd_facts(facts, (Concept("Capex"),)))
    got = dict(zip(q["period"].to_list(), q["value"].to_list()))
    assert got == {date(2024, 1, 1): 10, date(2024, 4, 1): 15, date(2024, 7, 1): 20, date(2024, 10, 1): 25,
                   date(2025, 1, 1): 12}
    q4 = q.filter(pl.col("period") == date(2024, 10, 1)).row(0, named=True)
    assert q4["method"] == "12M-9M" and q4["filed"] == date(2025, 2, 15)


def test_direct_quarter_fills_missing_ytd_and_concept_priority():
    facts = _facts({
        "Old": [_fact("2016-01-01", "2016-03-31", 5, "2016-05-01"), _fact("2016-01-01", "2016-12-31", 1, "2017-02-01", "10-K")],
        "New": [
            _fact("2016-01-01", "2016-12-31", 999, "2017-02-01", "10-K"),
            _fact("2018-01-01", "2018-06-30", 30, "2018-08-01"),
            _fact("2018-04-01", "2018-06-30", 16, "2018-08-01"),
        ],
    })
    ytd = truth.ytd_facts(facts, (Concept("Old"), Concept("New", min_end=date(2017, 1, 1))))
    assert ytd.filter(pl.col("end") == date(2016, 12, 31))["value"].to_list() == [1.0]
    q = truth.quarterly_from_ytd(ytd)
    got = dict(zip(q["period"].to_list(), q["value"].to_list()))
    assert got[date(2018, 4, 1)] == 16
    assert date(2018, 1, 1) not in got


def test_quarter_labels_and_kpi_csv(tmp_path):
    assert truth.quarter_start("2019Q3") == date(2019, 7, 1)
    assert truth.quarter_start("2019-Q1") == date(2019, 1, 1)
    assert truth.quarter_label(date(2019, 10, 1)) == "2019Q4"
    p = tmp_path / "k.csv"
    p.write_text(
        "# hand-collected; spot-check\n"
        "quarter,entity,metric,value,unit,source_url,note\n"
        '2020Q2,SBAC,towers_built,99,towers,https://x,"built 99 towers, total"\n',
        encoding="utf-8",
    )
    k = truth.load_kpis(p)
    assert k.row(0, named=True)["period"] == date(2020, 4, 1)
    assert k["value"].to_list() == [99.0]
