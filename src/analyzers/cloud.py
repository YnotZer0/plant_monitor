"""Cloud analyzer — uses Claude's vision API for high-quality analysis."""

import os
import json
import base64
import logging
import socket
from pathlib import Path
from typing import Optional

from .base import Analyzer, empty_result

log = logging.getLogger("plant_monitor.cloud")

SYSTEM_PROMPT = """You are an expert botanist and plant-care specialist.
You will be shown a photo of a plant and must assess its health.

Return ONLY valid JSON matching this exact schema — no prose, no markdown fences:

{
  "overall_health": "excellent|good|fair|poor|critical",
  "health_score": <integer 0-100>,
  "watering_status": "underwatered|optimal|overwatered|unknown",
  "watering_confidence": <integer 0-100>,
  "watering_reasoning": "<short explanation>",
  "sunlight_status": "insufficient|optimal|excessive|unknown",
  "sunlight_confidence": <integer 0-100>,
  "sunlight_reasoning": "<short explanation>",
  "observations": "<2-4 sentences describing what you see>",
  "issues_detected": [
    {"issue": "<name>", "severity": "mild|moderate|severe", "location": "<where on plant>"}
  ],
  "recommendations": ["<action 1>", "<action 2>"],
  "plant_bounding_box": {"x": <0-1>, "y": <0-1>, "w": <0-1>, "h": <0-1>},
  "species_guess": "<best guess of species or 'unknown'>"
}

Guidelines:
- watering: yellow/wilting lower leaves + soggy soil = overwatered; crispy/drooping + dry soil = underwatered
- sunlight: leggy growth, pale leaves, leaning = insufficient; scorched/bleached patches = excessive
- be honest about uncertainty — use "unknown" and low confidence when the image doesn't show it
- bounding box coordinates are fractions of image width/height (0.0-1.0), tight around the plant
"""


def _has_internet(host: str = "api.anthropic.com", port: int = 443, timeout: float = 3.0) -> bool:
    """Quick check for internet connectivity to the API host."""
    try:
        socket.setdefaulttimeout(timeout)
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, socket.gaierror, OSError):
        return False


class CloudAnalyzer(Analyzer):
    name = "cloud"
    version = "claude-opus-4-7"

    def __init__(self, model: str = "claude-opus-4-7"):
        self.model = model
        self.version = model
        self._client = None

    def is_available(self) -> bool:
        """True only if API key is set AND we can reach the internet."""
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return False
        return _has_internet()

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def analyse(self, image_path: Path, plant_context: Optional[dict] = None) -> dict:
        try:
            client = self._get_client()
            with open(image_path, "rb") as f:
                image_data = base64.standard_b64encode(f.read()).decode("utf-8")

            context_note = ""
            if plant_context:
                context_note = (
                    f"\n\nContext about this plant:\n"
                    f"- Name: {plant_context.get('name', 'unknown')}\n"
                    f"- Species: {plant_context.get('species', 'unknown')}\n"
                    f"- Location: {plant_context.get('location', 'unknown')}\n"
                )

            message = client.messages.create(
                model=self.model,
                max_tokens=1500,
                system=SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {
                            "type": "base64", "media_type": "image/jpeg", "data": image_data,
                        }},
                        {"type": "text", "text": f"Analyse this plant's health.{context_note}"},
                    ],
                }],
            )

            raw = message.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            result = json.loads(raw)
            result["analyzer"] = self.name
            result["analyzer_version"] = self.version
            return result

        except Exception as e:
            log.exception("Cloud analysis failed")
            return empty_result(self.name, f"cloud error: {e}")
