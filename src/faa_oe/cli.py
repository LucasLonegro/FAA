from __future__ import annotations

import argparse
import logging
import sys
from datetime import date

import polars as pl

from faa_oe import asr, backtest, coverage, enrich, entities, features, ingest, report, truth
from faa_oe.client import CASE_TYPES, REGIONS


def _csv(s: str) -> list[str]:
    return [x.strip().upper() for x in s.split(",") if x.strip()]


def _quarter(s: str) -> date:
    return truth.quarter_start(s)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="faa_oe", description="FAA OE/AAA obstruction filings pipeline")
    p.add_argument("--data-dir", help="defaults to $FAA_OE_DATA or <project>/data")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("backfill", help="fetch region x month windows for a year range (resumable)")
    b.add_argument("--from", dest="start", type=int, required=True)
    b.add_argument("--to", dest="end", type=int, default=date.today().year)
    b.add_argument("--types", type=_csv, default=list(CASE_TYPES))
    b.add_argument("--regions", type=_csv, default=list(REGIONS))
    b.add_argument("--workers", type=int, default=4)

    i = sub.add_parser("ingest", help="incremental: re-pull the trailing N days")
    i.add_argument("--days", type=int, default=180)
    i.add_argument("--types", type=_csv, default=list(CASE_TYPES))
    i.add_argument("--regions", type=_csv, default=list(REGIONS))
    i.add_argument("--workers", type=int, default=4)

    c = sub.add_parser("coverage", help="print completeness tables")
    c.add_argument("--long", action="store_true", help="also print the full region x year x type table")

    sub.add_parser("rebuild", help="rebuild parquet outputs from stored raw responses")

    e = sub.add_parser("entities", help="attribution rate of tower filings (overall, by year)")
    e.add_argument("--unmatched", action="store_true", help="also list the top unmatched sponsors")
    e.add_argument("--top", type=int, default=50)

    r = sub.add_parser("asr", help="download FCC ASR registrations and match them to cases")
    r.add_argument("--refresh", action="store_true", help="re-download even if the cached zip is recent")

    n = sub.add_parser("enrich", help="write cases_enriched.parquet (category, entity, build_kind, project, ASR)")
    n.add_argument("--refresh-asr", action="store_true", help="re-download the ASR file")
    n.add_argument("--no-asr", action="store_true", help="skip the ASR step (reuse case_asr.parquet if present)")

    f = sub.add_parser("features", help="write weekly / quarterly point-in-time filing panels")
    f.add_argument("--start", type=date.fromisoformat, help="first date_entered kept in the panels (YYYY-MM-DD)")
    f.add_argument("--prior-lookback-days", type=int, help="limit prior-case evidence to this many days back")

    t = sub.add_parser("truth", help="fetch carrier capex (SEC XBRL) and list ground-truth coverage")
    t.add_argument("--refresh", action="store_true", help="re-download SEC companyfacts")

    k = sub.add_parser("backtest", help="lead-lag and out-of-sample nowcast of KPIs from the filing signal")
    k.add_argument("--start", type=_quarter, help="first signal quarter, YYYY-Qn (default: start of the covered run)")
    k.add_argument("--end", type=_quarter, help="last signal quarter, YYYY-Qn")
    k.add_argument("--min-train", type=int, default=backtest.MIN_TRAIN)

    o = sub.add_parser("report", help="write reports/summary.md")
    o.add_argument("--start", type=_quarter, help="first signal quarter, YYYY-Qn")
    o.add_argument("--end", type=_quarter, help="last signal quarter, YYYY-Qn")

    a = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if a.cmd == "backfill":
        stats = ingest.backfill(a.start, a.end, a.types, a.regions, data_dir=a.data_dir, workers=a.workers)
    elif a.cmd == "ingest":
        stats = ingest.incremental(a.days, a.types, a.regions, data_dir=a.data_dir, workers=a.workers)
    elif a.cmd == "coverage":
        coverage.print_report(a.data_dir, long=a.long)
        return 0
    elif a.cmd == "entities":
        entities.print_report(a.data_dir, unmatched=a.unmatched, top=a.top)
        return 0
    elif a.cmd == "asr":
        registrations, case_asr = asr.run(a.data_dir, refresh=a.refresh)
        print(f"{registrations.height:,} ASR registrations; {case_asr.height:,} cases matched")
        print(case_asr.group_by("match_method").len("n").sort("n", descending=True))
        return 0
    elif a.cmd == "enrich":
        path = enrich.run(a.data_dir, refresh_asr=a.refresh_asr, use_asr=not a.no_asr)
        print(f"wrote {path}")
        enrich.print_summary(a.data_dir)
        return 0
    elif a.cmd == "features":
        out = features.run(a.data_dir, start=a.start, prior_lookback_days=a.prior_lookback_days)
        for path in out.values():
            print(f"wrote {path}")
        return 0
    elif a.cmd == "truth":
        try:
            truth.carrier_capex(a.data_dir, refresh=a.refresh)
        except truth.SecBlocked as e:
            print(f"SEC capex skipped: {e}")
        with pl.Config(tbl_rows=-1, tbl_width_chars=200, tbl_hide_dataframe_shape=True):
            print(truth.coverage_table(truth.load_truth(a.data_dir)))
        return 0
    elif a.cmd == "backtest":
        panel = features.load_panel("Q", a.data_dir)
        start = a.start or features.default_start(panel)
        res = backtest.run(panel, truth.load_truth(a.data_dir), start=start, end=a.end, min_train=a.min_train)
        backtest.save(res, a.data_dir)
        with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250, tbl_hide_dataframe_shape=True, fmt_str_lengths=60):
            print(f"signal from {truth.quarter_label(start) if start else 'first quarter'}")
            print(backtest.wide_leadlag(res.leadlag, "yoy"))
            print(res.oos.select("key", "kpi", "lead", "corr_at_lead", "best_lead_insample", "n_oos", "ratio_vs_last",
                                 "ratio_vs_last_year", "ratio_ar1x_vs_ar1"))
        for s_ in res.skipped:
            print(f"skipped {s_}")
        return 0
    elif a.cmd == "report":
        print(f"wrote {report.run(a.data_dir, start=a.start, end=a.end)}")
        return 0
    else:
        print(f"rebuilt from {ingest.rebuild(a.data_dir):,} raw rows")
        return 0

    print(stats.summary())
    for w in stats.capped:
        print(f"CAPPED: {w}")
    for e in stats.errors:
        print(f"ERROR: {e}")
    return 1 if stats.errors else 0
