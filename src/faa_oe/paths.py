from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


def default_data_dir() -> Path:
    return Path(os.environ.get("FAA_OE_DATA", PROJECT_ROOT / "data"))


@dataclass(frozen=True)
class DataPaths:
    root: Path

    @classmethod
    def resolve(cls, data_dir: Path | str | None = None) -> DataPaths:
        return cls(Path(data_dir) if data_dir is not None else default_data_dir())

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def parquet(self) -> Path:
        return self.root / "parquet"

    @property
    def cases(self) -> Path:
        return self.parquet / "cases.parquet"

    @property
    def parts(self) -> Path:
        return self.parquet / "parts"

    @property
    def history(self) -> Path:
        return self.parquet / "case_history.parquet"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.jsonl"

    @property
    def duckdb(self) -> Path:
        return self.root / "faa.duckdb"

    @property
    def asr_dir(self) -> Path:
        return self.root / "asr"

    @property
    def asr(self) -> Path:
        return self.parquet / "asr.parquet"

    @property
    def case_asr(self) -> Path:
        return self.parquet / "case_asr.parquet"

    @property
    def enriched(self) -> Path:
        return self.parquet / "cases_enriched.parquet"
