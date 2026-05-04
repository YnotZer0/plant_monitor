"""
Smoke test for the local analyzer.
Creates three synthetic 'plant' images representing healthy, overwatered, and
scorched conditions, and runs the local analyzer on each. We're not asserting
exact values — just that the analyzer runs, returns valid schema, and produces
roughly directionally correct signals.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
from PIL import Image, ImageDraw
from analyzers import LocalAnalyzer


def _draw_leaves(img, color):
    """Draw a canopy of overlapping leaf blobs in the given color."""
    for cx, cy, r in [(300, 300, 180), (220, 250, 90), (380, 280, 100),
                       (300, 200, 80), (250, 380, 75), (360, 370, 85)]:
        y, x = np.ogrid[:600, :600]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
        img[mask] = color


def _discolor_within_plant(img, plant_green, color, centers):
    """Paint discoloured patches, but only over pixels that are currently the green plant."""
    y, x = np.ogrid[:600, :600]
    is_green = np.all(img == np.array(plant_green), axis=-1)
    for cx, cy, r in centers:
        patch = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
        # Only replace green pixels, not background
        img[patch & is_green] = color


GREEN = [60, 140, 55]


def make_healthy_plant(path):
    img = np.full((600, 600, 3), 230, dtype=np.uint8)
    _draw_leaves(img, GREEN)
    Image.fromarray(img).save(path, "JPEG", quality=90)


def make_overwatered_plant(path):
    img = np.full((600, 600, 3), 230, dtype=np.uint8)
    _draw_leaves(img, GREEN)
    # Large yellow patches — ~10% of plant area
    _discolor_within_plant(img, GREEN, [210, 200, 50],
                            [(250, 330, 90), (330, 340, 80), (280, 270, 75),
                             (360, 300, 70), (270, 220, 60)])
    Image.fromarray(img).save(path, "JPEG", quality=90)


def make_scorched_plant(path):
    img = np.full((600, 600, 3), 230, dtype=np.uint8)
    _draw_leaves(img, GREEN)
    # Larger bleached patches — real scorch covers meaningful area
    _discolor_within_plant(img, GREEN, [248, 248, 243],
                            [(260, 230, 65), (340, 290, 55), (300, 310, 50),
                             (250, 300, 45), (330, 240, 55)])
    Image.fromarray(img).save(path, "JPEG", quality=90)


def main():
    tmp = Path("/tmp/plant_test")
    tmp.mkdir(exist_ok=True)

    scenarios = {
        "healthy":     (make_healthy_plant,     tmp / "healthy.jpg"),
        "overwatered": (make_overwatered_plant, tmp / "overwatered.jpg"),
        "scorched":    (make_scorched_plant,    tmp / "scorched.jpg"),
    }

    analyzer = LocalAnalyzer()
    print(f"Analyzer: {analyzer.name} v{analyzer.version}")
    print(f"Disease model available: {analyzer.classifier.available()}")
    print()

    for label, (maker, path) in scenarios.items():
        maker(path)
        result = analyzer.analyse(path)
        print(f"--- Scenario: {label} ---")
        print(f"  Overall:  {result['overall_health']} ({result['health_score']}/100)")
        print(f"  Water:    {result['watering_status']} ({result['watering_confidence']}%)")
        print(f"            {result['watering_reasoning']}")
        print(f"  Light:    {result['sunlight_status']} ({result['sunlight_confidence']}%)")
        print(f"            {result['sunlight_reasoning']}")
        print(f"  Issues:   {result['issues_detected']}")
        print()

        # Schema sanity checks
        assert result["analyzer"] == "local"
        assert 0 <= result["health_score"] <= 100
        assert result["watering_status"] in ("underwatered", "optimal", "overwatered", "unknown")
        assert result["sunlight_status"] in ("insufficient", "optimal", "excessive", "unknown")
        assert all(k in result["plant_bounding_box"] for k in ("x", "y", "w", "h"))

    print("✅ All schema checks passed")


if __name__ == "__main__":
    main()
