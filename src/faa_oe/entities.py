"""Sponsor -> company attribution.

Rules live in config/sponsors.yaml and are matched against a normalised sponsor name. The FAA API no longer
returns sponsor e-mails, so a sponsor that is a filing agent, an engineering firm or a person stays
unattributed here; `enrich` then falls back to the FCC ASR owner of the structure (by owner e-mail domain
first, then owner name, with the same rules).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import polars as pl
import yaml

from faa_oe.classify import classify
from faa_oe.paths import CONFIG_DIR, DataPaths

DEFAULT_CONFIG = CONFIG_DIR / "sponsors.yaml"
ENTITY_COLUMNS = ("entity", "ticker", "entity_type")

# Filer / consultant codes appended to company names: "(CK-MD)", " - CS", "USA.NB", "LLC-REGULATORY",
# "USA MR", "Towers_MJ". Applied repeatedly to the uppercased raw name, before punctuation is removed.
_TRAILING_CODES = [
    re.compile(r"\s*\([^()]*\)\s*$"),
    re.compile(r"\s+[-~_]\s*[A-Z0-9.&' ]{1,15}$"),
    re.compile(r"(?<=\bUSA|\bLLC|\bINC)\s*[-.,_~]\s*[A-Z0-9.&' ]{1,15}$"),
    re.compile(r"(?<=\bUSA)\s+[A-Z]{1,3}$"),
    re.compile(r"_[A-Z]{1,4}$"),
    re.compile(r"[\s\-~_.,]+$"),
]
_LEGAL_SUFFIX = re.compile(
    r"(\s+(LLC|L L C|INC|LP|L P|LLP|LLLP|PLLC|LTD|CORP|CORPORATION|CO|COMPANY|LIMITED|PARTNERSHIP"
    r"|I{1,3}|IV|VI{0,3}|IX|XI{0,3}|[A-Z]?\d+[A-Z]?))+$"
)


def normalize_sponsor(name: str | None) -> str | None:
    if name is None:
        return None
    s = re.sub(r"\s+", " ", name.upper()).strip()
    prev = None
    while s != prev:
        prev = s
        for pat in _TRAILING_CODES:
            s = pat.sub("", s).strip()
    s = re.sub(r"[&'’]", "", s)
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def base_name(norm: str) -> str:
    """Company name without d/b/a clause, legal suffixes or fund numbering: 'TARPON TOWERS II LLC' -> 'TARPON TOWERS'."""
    s = re.split(r"\b(?:D B A|DBA)\b", norm)[0].strip()
    stripped = _LEGAL_SUFFIX.sub("", s).strip()
    return stripped or s


@dataclass(frozen=True)
class Rule:
    pattern: re.Pattern
    entity: str | None
    ticker: str | None
    entity_type: str
    from_name: bool = False
    domains: tuple[str, ...] = ()


@dataclass(frozen=True)
class Match:
    entity: str | None
    ticker: str | None
    entity_type: str | None


NO_MATCH = Match(None, None, None)


@dataclass(frozen=True)
class RuleSet:
    rules: tuple[Rule, ...]
    ignored_domains: frozenset[str]

    def match_name(self, name: str | None) -> Match:
        return _match_name(self, normalize_sponsor(name))

    def match_domain(self, domain: str | None) -> Match:
        d = (domain or "").strip().lower()
        if not d or d in self.ignored_domains:
            return NO_MATCH
        for r in self.rules:
            if any(d == x or d.endswith("." + x) for x in r.domains):
                return Match(r.entity, r.ticker, r.entity_type)
        if d.endswith(".gov") or d.endswith(".mil") or re.search(r"(^|\.)(state|k12)\.[a-z]{2}\.us$", d):
            return Match(None, None, "government")
        return NO_MATCH

    def match_owner(self, name: str | None, domain: str | None) -> Match:
        """ASR owner: the e-mail domain is the more stable key (Crown Castle registers as CCATT, Pinnacle, ...).

        An owner no rule knows still names a company, so it becomes its own entity (type from the domain if
        that says government, else "other").
        """
        by_domain = self.match_domain(domain)
        if by_domain.entity is not None:
            return by_domain
        by_name = self.match_name(name)
        if by_name.entity is not None:
            return by_name
        norm = normalize_sponsor(name)
        if norm and by_name.entity_type != "consultant":
            return Match(base_name(norm), None, by_domain.entity_type or by_name.entity_type or "other")
        return NO_MATCH


@lru_cache(maxsize=65536)
def _match_name(ruleset: RuleSet, norm: str | None) -> Match:
    if not norm:
        return NO_MATCH
    for r in ruleset.rules:
        if r.pattern.search(norm):
            entity = base_name(norm) if r.from_name and r.entity is None else r.entity
            return Match(entity, r.ticker, r.entity_type)
    return NO_MATCH


def load_rules(path: Path | str | None = None) -> RuleSet:
    cfg = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    rules = tuple(
        Rule(
            pattern=re.compile(r["pattern"]),
            entity=r.get("entity"),
            ticker=r.get("ticker"),
            entity_type=r["entity_type"],
            from_name=bool(r.get("from_name", False)),
            domains=tuple(d.lower() for d in r.get("domains", [])),
        )
        for r in cfg["rules"]
    )
    return RuleSet(rules=rules, ignored_domains=frozenset(d.lower() for d in cfg.get("ignored_domains", [])))


@lru_cache(maxsize=1)
def default_rules() -> RuleSet:
    return load_rules()


def sponsor_table(sponsors: list[str | None], rules: RuleSet | None = None) -> pl.DataFrame:
    rs = rules or default_rules()
    rows = []
    for s in dict.fromkeys(sponsors):
        if s is None:
            continue
        norm = normalize_sponsor(s)
        m = _match_name(rs, norm)
        rows.append((s, norm, m.entity, m.ticker, m.entity_type))
    return pl.DataFrame(
        rows, schema={"sponsor": pl.String, "sponsor_norm": pl.String, **{c: pl.String for c in ENTITY_COLUMNS}}, orient="row"
    )


def attribute(df: pl.DataFrame | pl.LazyFrame, rules: RuleSet | None = None) -> pl.DataFrame | pl.LazyFrame:
    """Add sponsor_norm, entity, ticker, entity_type (consultants keep entity null) and entity_source."""
    lazy = df.lazy()
    sponsors = lazy.select(pl.col("sponsor").unique()).collect()["sponsor"].to_list()
    out = lazy.join(sponsor_table(sponsors, rules).lazy(), on="sponsor", how="left").with_columns(
        entity_source=pl.when(pl.col("entity").is_not_null()).then(pl.lit("sponsor"))
    )
    return out if isinstance(df, pl.LazyFrame) else out.collect()


def owner_table(owners: pl.DataFrame, rules: RuleSet | None = None) -> pl.DataFrame:
    """Distinct (asr_owner, asr_owner_domain) -> asr_owner_entity / _ticker / _entity_type."""
    rs = rules or default_rules()
    pairs = owners.select("asr_owner", "asr_owner_domain").unique()
    rows = [(n, d, *_as_tuple(rs.match_owner(n, d))) for n, d in pairs.iter_rows()]
    return pl.DataFrame(
        rows,
        schema={
            "asr_owner": pl.String,
            "asr_owner_domain": pl.String,
            "asr_owner_entity": pl.String,
            "asr_owner_ticker": pl.String,
            "asr_owner_entity_type": pl.String,
        },
        orient="row",
    )


def _as_tuple(m: Match) -> tuple[str | None, str | None, str | None]:
    return m.entity, m.ticker, m.entity_type


def apply_owner_fallback(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """Fill entity/ticker/entity_type from the ASR owner where the sponsor gave no entity."""
    use = pl.col("entity").is_null() & pl.col("asr_owner_entity").is_not_null()
    return df.with_columns(
        entity=pl.when(use).then(pl.col("asr_owner_entity")).otherwise(pl.col("entity")),
        ticker=pl.when(use).then(pl.col("asr_owner_ticker")).otherwise(pl.col("ticker")),
        entity_type=pl.when(use).then(pl.col("asr_owner_entity_type")).otherwise(pl.col("entity_type")),
        entity_source=pl.when(use).then(pl.lit("asr_owner")).otherwise(pl.col("entity_source")),
    )


def attribution_rate(df: pl.DataFrame, by: str | pl.Expr | None = None) -> pl.DataFrame:
    """Share of tower-signal filings with an entity."""
    d = df.filter(pl.col("is_tower_signal"))
    aggs = [
        pl.len().alias("n"),
        pl.col("entity").is_not_null().sum().alias("n_attributed"),
        pl.col("entity").is_not_null().mean().round(3).alias("rate"),
        (pl.col("entity_source") == "sponsor").fill_null(False).mean().round(3).alias("rate_sponsor_only"),
        pl.col("ticker").is_not_null().mean().round(3).alias("rate_listed"),
    ]
    if by is None:
        return d.select(aggs)
    out = d.group_by(by).agg(aggs)
    return out.sort(out.columns[0])


def unmatched_report(cases: pl.DataFrame, top: int = 50) -> pl.DataFrame:
    """Most frequent tower-signal sponsors that end up with no entity."""
    d = cases.filter(pl.col("is_tower_signal") & pl.col("entity").is_null())
    total = cases.filter(pl.col("is_tower_signal")).height
    return (
        d.group_by(pl.col("sponsor_norm").fill_null("<no sponsor>"))
        .agg(
            n=pl.len(),
            example=pl.col("sponsor").drop_nulls().mode().first(),
            consultant=(pl.col("entity_type") == "consultant").any(),
        )
        .with_columns(share=(pl.col("n") / max(total, 1)).round(4))
        .sort("n", descending=True)
        .head(top)
    )


ATTRIBUTION_INPUT = ["asn", "sponsor", "structure_type", "date_entered"]


def load_attributed(data_dir: Path | str | None = None) -> pl.DataFrame:
    """Cases (projected) with category and entity columns; uses the ASR owner fallback when case_asr exists."""
    paths = DataPaths.resolve(data_dir)
    lf = attribute(classify(pl.scan_parquet(paths.cases).select(ATTRIBUTION_INPUT)))
    if paths.case_asr.exists():
        owner = pl.scan_parquet(paths.case_asr).select(
            "asn", "asr_owner_entity", "asr_owner_ticker", "asr_owner_entity_type"
        )
        lf = apply_owner_fallback(lf.join(owner, on="asn", how="left"))
    return lf.collect()


def print_report(data_dir: Path | str | None = None, unmatched: bool = False, top: int = 50) -> None:
    df = load_attributed(data_dir)
    has_asr = "asr_owner_entity" in df.columns
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, tbl_hide_dataframe_shape=True, fmt_str_lengths=60):
        print(f"Tower-signal filings: {df['is_tower_signal'].sum():,} of {df.height:,} cases")
        print("ASR owner fallback: " + ("on" if has_asr else "off (run `enrich` or `asr` first)"))
        print("\nAttribution rate, overall")
        print(attribution_rate(df))
        print("\nBy year of date_entered")
        print(attribution_rate(df, pl.col("date_entered").dt.year().alias("year")))
        tw = df.filter(pl.col("is_tower_signal"))
        print("\nBy entity_type")
        print(tw.group_by("entity_type").len("n").with_columns(share=(pl.col("n") / tw.height).round(3)).sort("n", descending=True))
        print("\nTop entities")
        print(tw.group_by("entity", "ticker", "entity_type").len("n").sort("n", descending=True).head(30))
        if unmatched:
            print(f"\nTop {top} unmatched tower-signal sponsors")
            print(unmatched_report(df, top))
