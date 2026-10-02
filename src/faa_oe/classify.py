from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import polars as pl
import yaml

from faa_oe.paths import CONFIG_DIR

DEFAULT_CONFIG = CONFIG_DIR / "structure_types.yaml"
OTHER = "other"


@dataclass(frozen=True)
class Taxonomy:
    rules: tuple[tuple[str, frozenset[str], tuple[re.Pattern, ...]], ...]
    tower_signal: frozenset[str]

    def categorize(self, structure_type: str | None) -> str:
        st = (structure_type or "").strip().upper()
        if not st:
            return OTHER
        for category, types, patterns in self.rules:
            if st in types or any(p.search(st) for p in patterns):
                return category
        return OTHER

    @property
    def categories(self) -> list[str]:
        return list(dict.fromkeys([c for c, _, _ in self.rules] + [OTHER]))


def load_taxonomy(path: Path | str | None = None) -> Taxonomy:
    cfg = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    rules = tuple(
        (
            r["category"],
            frozenset(t.upper() for t in r.get("types", [])),
            tuple(re.compile(p) for p in r.get("patterns", [])),
        )
        for r in cfg["categories"]
    )
    return Taxonomy(rules=rules, tower_signal=frozenset(cfg["tower_signal"]))


@lru_cache(maxsize=1)
def default_taxonomy() -> Taxonomy:
    return load_taxonomy()


def classify(df: pl.DataFrame | pl.LazyFrame, taxonomy: Taxonomy | None = None) -> pl.DataFrame | pl.LazyFrame:
    """Add `category` and `is_tower_signal` from `structure_type`."""
    tx = taxonomy or default_taxonomy()
    lazy = df.lazy()
    types = lazy.select(pl.col("structure_type").unique()).collect()["structure_type"].to_list()
    mapping = {t: tx.categorize(t) for t in types if t is not None}
    out = lazy.with_columns(
        category=pl.col("structure_type").replace_strict(mapping, default=OTHER, return_dtype=pl.String).fill_null(OTHER)
    ).with_columns(is_tower_signal=pl.col("category").is_in(sorted(tx.tower_signal)))
    return out if isinstance(df, pl.LazyFrame) else out.collect()
