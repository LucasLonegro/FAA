from __future__ import annotations

import gzip
from datetime import timedelta
from pathlib import Path

from faa_oe.client import Window

FIXTURES = Path(__file__).parent / "fixtures"


def fixture_xml(name: str) -> bytes:
    return gzip.decompress((FIXTURES / name).read_bytes())


def case_xml(cases: list[dict[str, str]], tag: str = "OECase") -> bytes:
    body = "".join(f"<{tag}>" + "".join(f"<{k}>{v}</{k}>" for k, v in c.items()) + f"</{tag}>" for c in cases)
    return f'<?xml version="1.0" encoding="UTF-8"?><caseList>{body}</caseList>'.encode()


class FakeSource:
    """Serves `per_day` synthetic cases per day (or `per_state` per day for state-filtered calls)."""

    def __init__(self, per_day: int = 3, per_state: int | None = None, status: str = "WRK-Part77") -> None:
        self.per_day = per_day
        self.per_state = per_state
        self.status = status
        self.calls: list[Window] = []

    def case_list(self, w: Window) -> bytes:
        self.calls.append(w)
        per = self.per_state if (w.state and self.per_state is not None) else self.per_day
        region = w.region or "ASW"
        cases = []
        d = w.start
        while d <= w.end:
            for i in range(per):
                seq = d.timetuple().tm_yday * 1000 + i
                cases.append(
                    {
                        "asn": f"{d.year}-{region}-{seq}{w.state or ''}-{w.case_type}",
                        "asnSequence": str(seq),
                        "dateEntered": d.isoformat(),
                        "statusCode": self.status,
                        "faaGeographyId": region,
                        "nearestState": w.state or "TX",
                        "structureType": "TOWER$ANTENNA",
                    }
                )
            d += timedelta(days=1)
        return case_xml(cases)

