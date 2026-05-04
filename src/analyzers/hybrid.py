"""Hybrid analyzer — tries cloud first, falls back to local if unavailable."""

import logging
from pathlib import Path
from typing import Optional

from .base import Analyzer
from .cloud import CloudAnalyzer
from .local import LocalAnalyzer

log = logging.getLogger("plant_monitor.hybrid")


class HybridAnalyzer(Analyzer):
    name = "hybrid"
    version = "1.0"

    def __init__(self):
        self.cloud = CloudAnalyzer()
        self.local = LocalAnalyzer()

    def is_available(self) -> bool:
        return True  # local is always available

    def analyse(self, image_path: Path, plant_context: Optional[dict] = None) -> dict:
        if self.cloud.is_available():
            log.info("Cloud available — using Claude vision API")
            result = self.cloud.analyse(image_path, plant_context)
            # If cloud errored out (returned unknown), fall back to local
            if result.get("overall_health") != "unknown":
                return result
            log.warning("Cloud returned unknown — falling back to local analyzer")
        else:
            log.info("Cloud unavailable (offline or no API key) — using local analyzer")

        return self.local.analyse(image_path, plant_context)


def get_analyzer(mode: str = "auto") -> Analyzer:
    """Factory that selects an analyzer by mode name."""
    mode = (mode or "auto").lower()
    if mode == "cloud":
        return CloudAnalyzer()
    if mode == "local":
        return LocalAnalyzer()
    if mode in ("auto", "hybrid"):
        return HybridAnalyzer()
    raise ValueError(f"Unknown analyzer mode: {mode}")
