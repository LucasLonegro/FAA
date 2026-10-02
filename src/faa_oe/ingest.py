from __future__ import annotations

import gzip
import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

import duckdb
import polars as pl

from faa_oe.client import CASE_TYPES, REGIONS, CaseListSource, FAAClient, Window, fetch_adaptive, month_windows
from faa_oe.parse import parse_cases
from faa_oe.paths import DataPaths

log = logging.getLogger(__name__)

HISTORY_COLUMNS = (
    "asn",
    "asof",
    "status_code",
    "date_completed",
    "expiration_date",
    "date_built",
    "agl_structure_height",
    "agl_structure_height_det",
    "amsl_overall_height_proposed",
    "amsl_overall_height_det",
)


@dataclass
class RunStats:
    windows: int = 0
    skipped: int = 0
    requests: int = 0
    rows: int = 0
    capped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"windows={self.windows} skipped={self.skipped} requests={self.requests} rows={self.rows} "
            f"capped={len(self.capped)} errors={len(self.errors)}"
        )


class Manifest:
    """Append-only JSONL log of completed windows; a window is only recorded after its rows are stored."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def done_keys(self) -> set[tuple[str, int, str, str, str]]:
        return {(e["case_type"], e["year"], e["region"], e["start"], e["end"]) for e in self.entries()}

    def append(self, entries: Iterable[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")


def window_key(w: Window) -> tuple[str, int, str, str, str]:
    return (w.case_type, w.year, w.region or "ALL", w.start.isoformat(), w.end.isoformat())


def save_raw(paths: DataPaths, asof: str, w: Window, xml: bytes) -> Path:
    d = paths.raw / asof / w.case_type / (w.region or "ALL")
    d.mkdir(parents=True, exist_ok=True)
    stem = f"{w.start}_{w.end}"
    stem += f"_asn{w.year}" if w.year != w.start.year else ""
    stem += f"_{w.state}" if w.state else ""
    for i in range(1000):
        p = d / (f"{stem}.xml.gz" if i == 0 else f"{stem}-{i}.xml.gz")
        try:
            with open(p, "xb") as fh, gzip.GzipFile(fileobj=fh, mode="wb") as gz:
                gz.write(xml)
            return p
        except FileExistsError:
            continue
    raise RuntimeError(f"too many raw files for {p}")


def fetch_window(source: CaseListSource, w: Window, paths: DataPaths, asof: str) -> tuple[pl.DataFrame, dict]:
    frames, files, capped, n = [], [], [], 0
    for f in fetch_adaptive(source, w):
        files.append(save_raw(paths, asof, f.window, f.xml).relative_to(paths.root).as_posix())
        frames.append(parse_cases(f.xml))
        n += f.n_rows
        if f.capped:
            capped.append(str(f.window))
    case_type, year, region, start, end = window_key(w)
    entry = {
        "case_type": case_type,
        "year": year,
        "region": region,
        "start": start,
        "end": end,
        "asof": asof,
        "rows": n,
        "requests": len(files),
        "capped": capped,
        "files": files,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return pl.concat(frames, how="vertical"), entry


def _write_atomic(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    df.write_parquet(tmp)
    # On Windows, replacing a file another process has open (a notebook, DuckDB) fails transiently.
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(1 + attempt)


def upsert_cases(new: pl.DataFrame, asof: date, path: Path) -> pl.DataFrame:
    new = new.with_columns(first_seen_asof=pl.lit(asof), last_seen_asof=pl.lit(asof))
    merged = pl.concat([pl.read_parquet(path), new], how="diagonal_relaxed") if path.exists() else new
    merged = (
        merged.with_columns(pl.col("first_seen_asof").min().over("asn"))
        .sort("last_seen_asof", maintain_order=True)
        .unique("asn", keep="last", maintain_order=True)
        .sort("asn")
    )
    _write_atomic(merged, path)
    return merged


def append_history(new: pl.DataFrame, asof: date, path: Path) -> pl.DataFrame:
    rows = new.with_columns(asof=pl.lit(asof)).select(HISTORY_COLUMNS)
    merged = pl.concat([pl.read_parquet(path), rows], how="diagonal_relaxed") if path.exists() else rows
    merged = merged.unique(["asn", "asof"], keep="last", maintain_order=True)
    _write_atomic(merged, path)
    return merged


def register_views(paths: DataPaths) -> None:
    try:
        with duckdb.connect(str(paths.duckdb)) as con:
            for name, p in (("cases", paths.cases), ("case_history", paths.history)):
                if p.exists():
                    src = p.resolve().as_posix()
                    con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{src}')")
    except duckdb.IOException as e:
        log.warning("could not update %s (open elsewhere?): %s", paths.duckdb, e)


def store(frames: Sequence[pl.DataFrame], asof: date, paths: DataPaths) -> None:
    if not frames:
        return
    df = pl.concat(frames, how="vertical")
    if df.height:
        upsert_cases(df, asof, paths.cases)
        append_history(df, asof, paths.history)
    register_views(paths)


def stage(frames: Sequence[pl.DataFrame], asof: date, paths: DataPaths) -> None:
    """Write a batch as its own part file; `compact` folds parts into the main parquet once per run."""
    df = pl.concat(frames, how="vertical") if frames else None
    if df is None or not df.height:
        return
    paths.parts.mkdir(parents=True, exist_ok=True)
    df.write_parquet(paths.parts / f"{asof.isoformat()}_{uuid.uuid4().hex}.parquet")


def compact(paths: DataPaths) -> None:
    """Fold staged parts (including any left by a crashed run) into cases/history, oldest asof first."""
    by_asof: dict[str, list[Path]] = {}
    for p in sorted(paths.parts.glob("*.parquet")) if paths.parts.exists() else []:
        by_asof.setdefault(p.name.split("_", 1)[0], []).append(p)
    for asof, parts in sorted(by_asof.items()):
        store([pl.read_parquet(p) for p in parts], date.fromisoformat(asof), paths)
        for p in parts:
            p.unlink()


def run_windows(
    windows: Sequence[Window],
    source: CaseListSource,
    paths: DataPaths,
    asof: date,
    workers: int = 4,
    flush_every: int = 100,
) -> RunStats:
    stats = RunStats(windows=len(windows))
    manifest = Manifest(paths.manifest)
    frames: list[pl.DataFrame] = []
    entries: list[dict] = []
    t0 = time.monotonic()

    def flush() -> None:
        stage(frames, asof, paths)
        manifest.append(entries)
        frames.clear()
        entries.clear()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_window, source, w, paths, asof.isoformat()): w for w in windows}
        try:
            for i, fut in enumerate(as_completed(futures), 1):
                w = futures[fut]
                try:
                    df, entry = fut.result()
                except Exception as e:  # one bad window must not abort a multi-hour backfill
                    log.error("window %s failed: %r", w, e)
                    stats.errors.append(f"{w}: {e!r}")
                    continue
                frames.append(df)
                entries.append(entry)
                stats.requests += entry["requests"]
                stats.rows += entry["rows"]
                stats.capped += entry["capped"]
                if len(entries) >= flush_every:
                    flush()
                if i % 25 == 0 or i == len(windows):
                    rate = i / (time.monotonic() - t0)
                    log.info("%d/%d windows, %d rows, ETA %.0f min", i, len(windows), stats.rows, (len(windows) - i) / rate / 60)
        finally:
            for f in futures:
                f.cancel()
            flush()
    compact(paths)
    return stats


# NRA cases are sometimes created in the year before their ASN year (0-75 per region-year in probes); OE never.
SPILL_TYPES = ("NRA",)


def spill_window(case_type: str, region: str, asn_year: int) -> Window:
    return Window(case_type, region, date(asn_year - 1, 1, 1), date(asn_year - 1, 12, 31), asn_year=asn_year)


def _client_or(source: CaseListSource | None) -> CaseListSource:
    return source if source is not None else FAAClient()


def backfill(
    start_year: int,
    end_year: int | None = None,
    types: Sequence[str] = CASE_TYPES,
    regions: Sequence[str] = REGIONS,
    source: CaseListSource | None = None,
    data_dir: Path | str | None = None,
    workers: int = 4,
    today: date | None = None,
) -> RunStats:
    """Region x month windows (plus one prior-year spill window per NRA region-year), newest year first.

    Windows already in the manifest are skipped.
    """
    today = today or date.today()
    end_year = end_year or today.year
    paths = DataPaths.resolve(data_dir)
    done = Manifest(paths.manifest).done_keys()
    todo: list[Window] = []
    skipped = 0
    for year in range(end_year, start_year - 1, -1):
        if year > today.year:
            continue
        months = month_windows("OE", None, date(year, 1, 1), min(date(year, 12, 31), today))
        windows = [Window(t, r, m.start, m.end) for m in months for t in types for r in regions]
        windows += [spill_window(t, r, year) for t in types if t in SPILL_TYPES for r in regions]
        for w in windows:
            if window_key(w) in done:
                skipped += 1
            else:
                todo.append(w)
    log.info("backfill %d-%d: %d windows to fetch, %d already done", start_year, end_year, len(todo), skipped)
    stats = run_windows(todo, _client_or(source), paths, today, workers=workers)
    stats.skipped = skipped
    return stats


def incremental(
    days: int = 180,
    types: Sequence[str] = CASE_TYPES,
    regions: Sequence[str] = REGIONS,
    source: CaseListSource | None = None,
    data_dir: Path | str | None = None,
    workers: int = 4,
    today: date | None = None,
) -> RunStats:
    """Re-pull the trailing window unconditionally so status changes land in case_history."""
    today = today or date.today()
    paths = DataPaths.resolve(data_dir)
    windows = [
        Window(t, r, m.start, m.end)
        for m in month_windows("OE", None, today - timedelta(days=days), today)
        for t in types
        for r in regions
    ]
    windows += [spill_window(t, r, today.year) for t in types if t in SPILL_TYPES for r in regions]
    return run_windows(windows, _client_or(source), paths, today, workers=workers)


def rebuild(data_dir: Path | str | None = None) -> int:
    """Rebuild both parquet files from the immutable raw responses, replaying snapshots in asof order."""
    paths = DataPaths.resolve(data_dir)
    for p in (paths.cases, paths.history):
        p.unlink(missing_ok=True)
    by_asof: dict[str, list[str]] = {}
    for e in Manifest(paths.manifest).entries():
        by_asof.setdefault(e["asof"], []).extend(e["files"])
    n = 0
    for asof in sorted(by_asof):
        frames = [parse_cases(gzip.decompress((paths.root / f).read_bytes())) for f in by_asof[asof]]
        store(frames, date.fromisoformat(asof), paths)
        n += sum(f.height for f in frames)
    return n
