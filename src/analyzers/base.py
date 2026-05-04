"""
Analyzer interface.
All plant-health analyzers (cloud, local, hybrid) return the same dict schema
so downstream code (annotation, storage) doesn't care which one ran.

Schema:
{
    "overall_health": "excellent|good|fair|poor|critical|unknown",
    "health_score": int 0-100,
    "watering_status": "underwatered|optimal|overwatered|unknown",
    "watering_confidence": int 0-100,
    "watering_reasoning": str,
    "sunlight_status": "insufficient|optimal|excessive|unknown",
    "sunlight_confidence": int 0-100,
    "sunlight_reasoning": str,
    "observations": str,
    "issues_detected": [{"issue": str, "severity": "mild|moderate|severe", "location": str}],
    "recommendations": [str],
    "plant_bounding_box": {"x": float, "y": float, "w": float, "h": float},
    "species_guess": str,
    "analyzer": str,           # "cloud" | "local" | which one actually ran
    "analyzer_version": str,
}
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class Analyzer(ABC):
    """Abstract base class for plant health analyzers."""

    name: str = "abstract"
    version: str = "0"

    @abstractmethod
    def analyse(self, image_path: Path, plant_context: Optional[dict] = None) -> dict:
        """Analyse a plant image and return a dict matching the schema above."""
        raise NotImplementedError

    def is_available(self) -> bool:
        """Check whether this analyzer is ready to run (deps installed, API reachable, etc.)."""
        return True


def empty_result(analyzer_name: str, reason: str) -> dict:
    """Fallback result for when analysis fails completely."""
    return {
        "overall_health": "unknown",
        "health_score": 0,
        "watering_status": "unknown",
        "watering_confidence": 0,
        "watering_reasoning": reason,
        "sunlight_status": "unknown",
        "sunlight_confidence": 0,
        "sunlight_reasoning": reason,
        "observations": f"Analysis unavailable: {reason}",
        "issues_detected": [],
        "recommendations": [],
        "plant_bounding_box": {"x": 0.1, "y": 0.1, "w": 0.8, "h": 0.8},
        "species_guess": "unknown",
        "analyzer": analyzer_name,
        "analyzer_version": "0",
    }
