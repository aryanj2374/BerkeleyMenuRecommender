"""Crowdedness estimation service for dining halls.

Provides a heuristic fallback today and leaves hooks for real-time data sources.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class Crowdedness:
    score: float  # 0 = packed, 1 = empty
    label: str
    updated_at: str
    source: str


BASE_PROFILES: Dict[str, Dict[str, str]] = {
    "Cafe 3": {"breakfast": "medium", "lunch": "high", "dinner": "high"},
    "Crossroads": {"breakfast": "medium", "lunch": "high", "dinner": "high"},
    "Foothill": {"breakfast": "low", "lunch": "medium", "dinner": "medium"},
    "Clark Kerr": {"breakfast": "low", "lunch": "medium", "dinner": "high"},
}

LABEL_TO_SCORE = {"low": 0.8, "medium": 0.5, "high": 0.2}


def estimate_crowdedness(location_name: str, when: Optional[dt.datetime] = None) -> Crowdedness:
    now = when or dt.datetime.now(dt.timezone.utc)
    profile = BASE_PROFILES.get(location_name, {})
    meal_bucket = _bucketize_time(now.astimezone(dt.timezone(dt.timedelta(hours=-7))))
    label = profile.get(meal_bucket, profile.get("default", "medium"))
    score = LABEL_TO_SCORE.get(label, 0.5)
    return Crowdedness(
        score=score,
        label=label.title(),
        updated_at=now.isoformat(timespec="seconds"),
        source="heuristic-profile-v1",
    )


def _bucketize_time(timestamp: dt.datetime) -> str:
    hour = timestamp.hour
    if 5 <= hour < 11:
        return "breakfast"
    if 11 <= hour < 16:
        return "lunch"
    if 16 <= hour < 22:
        return "dinner"
    return "default"
