from __future__ import annotations

import polars as pl
import pytest

from faa_oe.classify import classify, default_taxonomy


@pytest.mark.parametrize(
    "structure_type, category",
    [
        ("TOWER$ANTENNA", "tower"),
        ("TOWER", "tower"),
        ("tower$antenna", "tower"),
        ("POLE$MONO", "monopole"),
        ("TOWER$COW", "cow"),
        ("WATER_TOWER", "antenna_mount"),
        ("CRANE$MOBILE", "crane"),
        ("CRANE$WINDMILL", "crane"),
        ("CONSTRUCTION$VEHICLE", "crane"),
        ("WINDMILL", "wind"),
        ("WINDMILL_FARMS$MET", "wind"),
        ("TOWER$MET", "wind"),
        ("RIG$DRILLING", "rig"),
        ("TRANSMISSION_LINE$T_L_TOWER", "transmission"),
        ("POLE$UTILITY", "transmission"),
        ("POLE$LIGHTING", "lighting"),
        ("TOWER$SOLAR", "solar"),
        ("BUILDING$COMMERCIAL", "building"),
        ("HANGAR - Construction", "building"),
        ("RUNWAY - Rehabilitate Runway", "airport"),
        ("TOWER$NON", "other"),
        ("POLE", "other"),
        ("SOMETHING NEW", "other"),
        (None, "other"),
        ("", "other"),
    ],
)
def test_categorize(structure_type, category):
    assert default_taxonomy().categorize(structure_type) == category


def test_tower_signal_is_towers_and_monopoles_only():
    assert default_taxonomy().tower_signal == {"tower", "monopole"}


def test_classify_frame_and_lazy():
    df = pl.DataFrame({"structure_type": ["TOWER$ANTENNA", "POLE$MONO", "TOWER$COW", "WINDMILL", None]})
    out = classify(df)
    assert out["category"].to_list() == ["tower", "monopole", "cow", "wind", "other"]
    assert out["is_tower_signal"].to_list() == [True, True, False, False, False]
    assert classify(df.lazy()).collect().equals(out)
