from __future__ import annotations

import gzip
from datetime import date

import duckdb
import polars as pl

from conftest import FakeSource, case_xml
from faa_oe import ingest
from faa_oe.client import Window
from faa_oe.parse import parse_cases
from faa_oe.paths import DataPaths


def test_backfill_writes_outputs_and_resumes(tmp_path):
    src = FakeSource(per_day=2)
    kw = dict(types=["OE"], regions=["ASW", "AEA"], source=src, data_dir=tmp_path, workers=2, today=date(2025, 3, 15))
    stats = ingest.backfill(2025, 2025, **kw)
    assert stats.windows == 6 and not stats.errors
    assert len(src.calls) == 6
    paths = DataPaths(tmp_path)
    cases = pl.read_parquet(paths.cases)
    assert cases.height == stats.rows == 2 * 2 * (31 + 28 + 15)
    assert cases["date_entered"].max() == date(2025, 3, 15)
    raw = sorted(p.relative_to(paths.raw).as_posix() for p in paths.raw.rglob("*.xml.gz"))
    assert "2025-03-15/OE/ASW/2025-03-01_2025-03-15.xml.gz" in raw and len(raw) == 6

    again = ingest.backfill(2025, 2025, **kw)
    assert again.windows == 0 and again.skipped == 6
    assert len(src.calls) == 6

    # a new day re-fetches only the still-open current month
    ingest.backfill(2025, 2025, **{**kw, "today": date(2025, 3, 16)})
    assert len(src.calls) == 8


def test_backfill_adds_nra_spill_windows(tmp_path):
    src = FakeSource(per_day=0)
    kw = dict(types=["OE", "NRA"], regions=["ASO"], source=src, data_dir=tmp_path, workers=1, today=date(2025, 2, 10))
    stats = ingest.backfill(2025, 2025, **kw)
    assert stats.windows == 2 * 2 + 1
    spill = [w for w in src.calls if w.year != w.start.year]
    assert [(w.case_type, w.year, w.start, w.end) for w in spill] == [("NRA", 2025, date(2024, 1, 1), date(2024, 12, 31))]
    assert any("2024-01-01_2024-12-31_asn2025" in p.name for p in DataPaths(tmp_path).raw.rglob("*.gz"))
    assert ingest.backfill(2025, 2025, **kw).skipped == 5


def test_backfill_retries_failed_windows_next_run(tmp_path):
    class Flaky(FakeSource):
        healed = False

        def case_list(self, w):
            if w.region == "AEA" and not self.healed:
                raise RuntimeError("boom")
            return super().case_list(w)

    src = Flaky()
    kw = dict(types=["OE"], regions=["ASW", "AEA"], source=src, data_dir=tmp_path, workers=1, today=date(2025, 1, 31))
    first = ingest.backfill(2025, 2025, **kw)
    assert len(first.errors) == 1
    src.healed = True
    second = ingest.backfill(2025, 2025, **kw)
    assert second.windows == 1 and not second.errors


def test_raw_never_overwritten(tmp_path):
    paths = DataPaths(tmp_path)
    w = Window("OE", "ASW", date(2025, 1, 1), date(2025, 1, 31))
    a = ingest.save_raw(paths, "2025-02-01", w, b"<caseList/>")
    b = ingest.save_raw(paths, "2025-02-01", w, b"<caseList><OECase/></caseList>")
    assert a != b
    assert gzip.decompress(a.read_bytes()) == b"<caseList/>"


def _cases(status: str) -> pl.DataFrame:
    return parse_cases(
        case_xml([{"asn": "2025-ASW-1-OE", "asnSequence": "1", "dateEntered": "2025-01-02", "statusCode": status}])
    )


def test_upsert_preserves_first_seen_and_tracks_history(tmp_path):
    paths = DataPaths(tmp_path)
    d1, d2, d3 = date(2025, 1, 5), date(2025, 2, 5), date(2025, 3, 5)
    ingest.store([_cases("WRK-Part77")], d1, paths)
    ingest.store([_cases("DET-DNE")], d2, paths)
    ingest.store([_cases("DET-DNE")], d2, paths)

    cases = pl.read_parquet(paths.cases)
    assert cases.height == 1
    row = cases.row(0, named=True)
    assert (row["first_seen_asof"], row["last_seen_asof"], row["status_code"]) == (d1, d2, "DET-DNE")

    # replaying an older snapshot must not clobber newer state
    ingest.upsert_cases(_cases("WRK-Part77"), d1, paths.cases)
    row = pl.read_parquet(paths.cases).row(0, named=True)
    assert (row["first_seen_asof"], row["last_seen_asof"], row["status_code"]) == (d1, d2, "DET-DNE")

    ingest.store([_cases("DET-DNE-EXT")], d3, paths)
    hist = pl.read_parquet(paths.history).sort("asof")
    assert hist.select("asof", "status_code").rows() == [(d1, "WRK-Part77"), (d2, "DET-DNE"), (d3, "DET-DNE-EXT")]

    with duckdb.connect(str(paths.duckdb), read_only=True) as con:
        assert con.execute("select count(*) from case_history").fetchone()[0] == 3
        assert con.execute("select first_seen_asof from cases").fetchone()[0] == d1


def test_compact_folds_leftover_parts_in_asof_order(tmp_path):
    paths = DataPaths(tmp_path)
    d1, d2 = date(2025, 1, 5), date(2025, 2, 5)
    ingest.stage([_cases("DET-DNE")], d2, paths)
    ingest.stage([_cases("WRK-Part77")], d1, paths)
    ingest.compact(paths)
    row = pl.read_parquet(paths.cases).row(0, named=True)
    assert (row["first_seen_asof"], row["last_seen_asof"], row["status_code"]) == (d1, d2, "DET-DNE")
    assert not list(paths.parts.glob("*.parquet"))


def test_incremental_refetches_trailing_window(tmp_path):
    src = FakeSource(per_day=1)
    kw = dict(days=40, types=["OE"], regions=["ASW"], source=src, data_dir=tmp_path, workers=1, today=date(2026, 1, 10))
    stats = ingest.incremental(**kw)
    assert sorted((w.start, w.end) for w in src.calls) == [
        (date(2025, 12, 1), date(2025, 12, 31)),
        (date(2026, 1, 1), date(2026, 1, 10)),
    ]
    assert stats.rows == 41
    ingest.incremental(**kw)
    assert len(src.calls) == 4


def test_rebuild_from_raw(tmp_path):
    src = FakeSource(per_day=1)
    ingest.backfill(2025, 2025, types=["OE"], regions=["ASW"], source=src, data_dir=tmp_path, workers=1, today=date(2025, 2, 10))
    before = pl.read_parquet(DataPaths(tmp_path).cases)
    assert ingest.rebuild(tmp_path) == before.height
    assert pl.read_parquet(DataPaths(tmp_path).cases).equals(before)
