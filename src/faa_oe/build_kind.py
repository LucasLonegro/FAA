"""New build vs modification of an existing structure, for tower, monopole and COW filings.

Most REIT `TOWER$ANTENNA` filings are antenna work on towers that already stand. Every filing gets
`build_kind` in {new_build, modification, unknown} plus the evidence used (`build_kind_basis`) and each raw
signal (`bk_*`, values "existing" / "new" / null):

  bk_asr        The FCC ASR registration matched to the case (fcc_asr_number, FAA study number or
                proximity; see asr.py). Existing if the structure was registered more than 365 days before
                the filing or constructed more than 30 days before it; otherwise new (registered and/or built
                after the filing, or granted and not yet built). Registration dates are estimated from the
                sequential registration number, hence the wide margin.
  bk_faa_built  FAA `date_built` (from a 7460-2 completion notice) more than 30 days before date_entered.
  bk_keyword    structure_description says the work is on an existing structure: existing, colo, mod,
                side mount, swap, upgrade, as-built, raise / increase / top hat / extension, add, 5G.
                "New" words (raw land, BTS, greenfield) are not used: they name the original site type and
                mostly sit on towers ASR shows as long built.
  bk_prior_case An earlier tower / monopole / antenna-mount filing within 100 m, entered at least 60 days
                before. Refiled proposals for towers never built also trigger it, so it ranks low; its reach
                grows with the length of the history held.
  bk_asr_number The case cites an ASR number that is not in the current ASR file.

Decision rule: the first non-null signal in the order above decides (existing -> modification,
new -> new_build); with none, `unknown`. Unknowns are mostly sub-200 ft structures that need no ASR
registration and have no earlier filing in the history held.
"""

from __future__ import annotations

import polars as pl

from faa_oe.geo import pairs_within

BUILD_KIND_CATEGORIES = ("tower", "monopole", "cow")
PRIOR_CASE_CATEGORIES = ("tower", "monopole", "antenna_mount")
PRIOR_RADIUS_M = 100.0
PRIOR_MIN_DAYS = 60
SIGNALS = ("bk_asr", "bk_faa_built", "bk_keyword", "bk_prior_case", "bk_asr_number")

EXISTING_KEYWORDS = (
    r"\bEXIST|\bCOLO|\bMOD(IFICATION)?\b|\bSIDE( ?(MNT|MOUNT))?\b|\bSWAP\b|\bUPGRADE|\bAS[- ]?BUILT"
    r"|\bRAISE|\bINCREASE|\bTOP ?HAT|\bEXTENSION|\bADD(ING|ITION)?\b|\b5G\b"
)

INPUT = ["asn", "category", "date_entered", "date_built", "structure_description", "fcc_asr_number",
         "latitude", "longitude"]
ASR_INPUT = ["asn", "match_method", "asr_registration_date", "asr_date_constructed"]


def _days(n: int) -> pl.Expr:
    return pl.duration(days=n)


def prior_cases(cases: pl.DataFrame) -> pl.DataFrame:
    """asn -> n_prior_cases: earlier tower-ish filings within PRIOR_RADIUS_M, entered >= PRIOR_MIN_DAYS before."""
    cols = ["asn", "latitude", "longitude", "date_entered"]
    left = cases.filter(pl.col("category").is_in(BUILD_KIND_CATEGORIES)).select(cols)
    right = cases.filter(pl.col("category").is_in(PRIOR_CASE_CATEGORIES)).select(cols)
    pairs = pairs_within(left, right, PRIOR_RADIUS_M)
    return (
        pairs.filter(pl.col("date_entered_r") <= pl.col("date_entered") - _days(PRIOR_MIN_DAYS))
        .group_by("asn")
        .agg(n_prior_cases=pl.len())
    )


def signals(cases: pl.DataFrame, case_asr: pl.DataFrame | None) -> pl.DataFrame:
    """asn + bk_* signal columns, for build-kind categories only."""
    d = cases.select(INPUT).filter(pl.col("category").is_in(BUILD_KIND_CATEGORIES))
    asr = case_asr.select(ASR_INPUT) if case_asr is not None else pl.DataFrame(
        schema={"asn": pl.String, "match_method": pl.String, "asr_registration_date": pl.Date,
                "asr_date_constructed": pl.Date}
    )
    d = d.join(asr, on="asn", how="left").join(prior_cases(cases.select(INPUT)), on="asn", how="left")
    de = pl.col("date_entered")
    existing, new = pl.lit("existing"), pl.lit("new")
    return d.select(
        "asn",
        bk_asr=pl.when(pl.col("asr_registration_date") < de - _days(365)).then(existing)
        .when(pl.col("asr_date_constructed") < de - _days(30)).then(existing)
        .when(pl.col("match_method").is_not_null()).then(new),
        bk_faa_built=pl.when(pl.col("date_built") < de - _days(30)).then(existing),
        bk_keyword=pl.when(
            pl.col("structure_description").fill_null("").str.to_uppercase().str.contains(EXISTING_KEYWORDS)
        ).then(existing),
        bk_prior_case=pl.when(pl.col("n_prior_cases") > 0).then(existing),
        bk_asr_number=pl.when(pl.col("fcc_asr_number").is_not_null() & pl.col("match_method").is_null()).then(existing),
    )


def decide(sig: pl.DataFrame) -> pl.DataFrame:
    basis = pl.coalesce([pl.when(pl.col(s).is_not_null()).then(pl.lit(s.removeprefix("bk_"))) for s in SIGNALS])
    verdict = pl.coalesce([pl.col(s) for s in SIGNALS])
    return sig.with_columns(
        build_kind=pl.when(verdict == "existing").then(pl.lit("modification"))
        .when(verdict == "new").then(pl.lit("new_build"))
        .otherwise(pl.lit("unknown")),
        build_kind_basis=basis.fill_null("none"),
    )


def build_kind(cases: pl.DataFrame, case_asr: pl.DataFrame | None = None) -> pl.DataFrame:
    """asn -> build_kind, build_kind_basis and the bk_* signals (only build-kind categories get rows)."""
    return decide(signals(cases, case_asr))


def signal_agreement(bk: pl.DataFrame) -> pl.DataFrame:
    """For each pair of signals: filings where both fire, and how often they agree."""
    rows = []
    for i, a in enumerate(SIGNALS):
        for b in SIGNALS[i + 1:]:
            both = bk.filter(pl.col(a).is_not_null() & pl.col(b).is_not_null())
            agree = both.filter(pl.col(a) == pl.col(b)).height
            rows.append((a, b, both.height, agree, round(agree / both.height, 3) if both.height else None))
    return pl.DataFrame(rows, schema=["signal_a", "signal_b", "n_both", "n_agree", "agree_rate"], orient="row")


def signal_coverage(bk: pl.DataFrame) -> pl.DataFrame:
    return pl.DataFrame(
        [
            (s, bk.filter(pl.col(s) == "existing").height, bk.filter(pl.col(s) == "new").height)
            for s in SIGNALS
        ],
        schema=["signal", "n_existing", "n_new"],
        orient="row",
    ).with_columns(share_fired=((pl.col("n_existing") + pl.col("n_new")) / max(bk.height, 1)).round(3))
