"""
Plant Health Monitor - Core
Handles image capture, analysis (via pluggable analyzers), annotation, and storage.
"""

import os
import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

# -------- Configuration --------
BASE_DIR = Path(__file__).resolve().parent.parent
CAPTURES_DIR = BASE_DIR / "captures"
DB_PATH = BASE_DIR / "db" / "plants.db"
LOG_PATH = BASE_DIR / "plant_monitor.log"
MODELS_DIR = BASE_DIR / "models"

# Choose which analyzer to use: "cloud", "local", or "auto" (hybrid)
ANALYZER_MODE = os.environ.get("PLANT_ANALYZER", "auto")

for d in (CAPTURES_DIR, DB_PATH.parent, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
log = logging.getLogger("plant_monitor")


# -------- Data Models --------
@dataclass
class HealthReading:
    plant_id: int
    timestamp: str
    overall_health: str
    health_score: int
    watering_status: str
    watering_confidence: int
    sunlight_status: str
    sunlight_confidence: int
    observations: str
    issues_detected: str
    recommendations: str
    original_image_path: str
    annotated_image_path: str
    raw_analysis: str
    analyzer: str


# -------- Database --------
SCHEMA = """
CREATE TABLE IF NOT EXISTS plants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    species TEXT,
    location TEXT,
    date_added TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plant_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    overall_health TEXT,
    health_score INTEGER,
    watering_status TEXT,
    watering_confidence INTEGER,
    sunlight_status TEXT,
    sunlight_confidence INTEGER,
    observations TEXT,
    issues_detected TEXT,
    recommendations TEXT,
    original_image_path TEXT,
    annotated_image_path TEXT,
    raw_analysis TEXT,
    analyzer TEXT,
    FOREIGN KEY (plant_id) REFERENCES plants(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS care_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plant_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    event_type TEXT NOT NULL,
    notes TEXT,
    FOREIGN KEY (plant_id) REFERENCES plants(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_readings_plant_time ON readings(plant_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_care_plant_time ON care_events(plant_id, timestamp);
"""


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with get_db() as conn:
        conn.executescript(SCHEMA)
        # Add analyzer column if upgrading from v1
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(readings)").fetchall()}
        if "analyzer" not in cols:
            conn.execute("ALTER TABLE readings ADD COLUMN analyzer TEXT")
        conn.commit()
    log.info("Database initialised at %s", DB_PATH)


# -------- Camera --------
def capture_image(output_path: Path, resolution=(2304, 1296)) -> Path:
    """Capture via picamera2; fall back to placeholder if unavailable."""
    try:
        from picamera2 import Picamera2  # type: ignore
        picam2 = Picamera2()
        config = picam2.create_still_configuration(main={"size": resolution})
        picam2.configure(config)
        picam2.start()
        import time
        time.sleep(2)
        picam2.capture_file(str(output_path))
        picam2.stop()
        picam2.close()
        log.info("Captured image to %s", output_path)
        return output_path
    except ImportError:
        log.warning("picamera2 not available — using placeholder image")
        img = Image.new("RGB", resolution, (40, 90, 40))
        img.save(output_path, "JPEG", quality=90)
        return output_path


# -------- Image Annotation --------
def annotate_image(original_path: Path, analysis: dict, output_path: Path) -> Path:
    """Draw bounding box + health summary onto image. Shows which analyzer ran."""
    img = Image.open(original_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    W, H = img.size

    colours = {
        "excellent": (46, 204, 113), "good": (46, 204, 113),
        "fair": (241, 196, 15), "poor": (230, 126, 34), "critical": (231, 76, 60),
    }
    health = analysis.get("overall_health", "unknown").lower()
    box_colour = colours.get(health, (149, 165, 166))

    bb = analysis.get("plant_bounding_box") or {}
    if all(k in bb for k in ("x", "y", "w", "h")):
        x0 = int(bb["x"] * W); y0 = int(bb["y"] * H)
        x1 = int((bb["x"] + bb["w"]) * W); y1 = int((bb["y"] + bb["h"]) * H)
        for offset in range(4):
            draw.rectangle([x0 - offset, y0 - offset, x1 + offset, y1 + offset],
                           outline=box_colour)

    try:
        font_md = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except OSError:
        font_md = ImageFont.load_default()
        font_sm = ImageFont.load_default()

    panel_lines = [
        f"Health: {analysis.get('overall_health', '?').upper()} ({analysis.get('health_score', '?')}/100)",
        f"Water:  {analysis.get('watering_status', '?')} ({analysis.get('watering_confidence', 0)}%)",
        f"Light:  {analysis.get('sunlight_status', '?')} ({analysis.get('sunlight_confidence', 0)}%)",
    ]
    padding = 16
    line_h = 36
    panel_h = line_h * len(panel_lines) + padding * 2 + 20
    panel_w = 620
    draw.rectangle([0, 0, panel_w, panel_h], fill=(0, 0, 0, 180))
    draw.rectangle([0, 0, 8, panel_h], fill=box_colour)

    for i, line in enumerate(panel_lines):
        draw.text((padding + 8, padding + i * line_h), line, fill=(255, 255, 255), font=font_md)

    # Analyzer tag at bottom of panel
    analyzer_tag = f"Analyzer: {analysis.get('analyzer', '?')} ({analysis.get('analyzer_version', '?')})"
    draw.text((padding + 8, padding + len(panel_lines) * line_h),
              analyzer_tag, fill=(200, 200, 200), font=font_sm)

    # Timestamp
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    bbox = draw.textbbox((0, 0), ts, font=font_md)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([W - tw - 2 * padding, H - th - 2 * padding, W, H], fill=(0, 0, 0, 160))
    draw.text((W - tw - padding, H - th - padding), ts, fill=(255, 255, 255), font=font_md)

    img.save(output_path, "JPEG", quality=92)
    log.info("Annotated image saved to %s", output_path)
    return output_path


# -------- Orchestration --------
def capture_and_analyse(plant_id: int, analyzer_mode: Optional[str] = None) -> HealthReading:
    """End-to-end: capture, analyse, annotate, persist. analyzer_mode overrides env var."""
    # Lazy import so this module can be imported without analyzer deps
    from analyzers import get_analyzer

    mode = analyzer_mode or ANALYZER_MODE
    analyzer = get_analyzer(mode)
    log.info("Using analyzer: %s (mode=%s)", analyzer.name, mode)

    with get_db() as conn:
        plant = conn.execute("SELECT * FROM plants WHERE id = ?", (plant_id,)).fetchone()
        if not plant:
            raise ValueError(f"No plant with id {plant_id}")
        plant = dict(plant)

    ts = datetime.now()
    stamp = ts.strftime("%Y%m%d_%H%M%S")
    plant_dir = CAPTURES_DIR / f"plant_{plant_id}"
    plant_dir.mkdir(exist_ok=True)

    original_path = plant_dir / f"{stamp}_original.jpg"
    annotated_path = plant_dir / f"{stamp}_annotated.jpg"

    capture_image(original_path)
    analysis = analyzer.analyse(original_path, plant)
    annotate_image(original_path, analysis, annotated_path)

    reading = HealthReading(
        plant_id=plant_id,
        timestamp=ts.isoformat(),
        overall_health=analysis.get("overall_health", "unknown"),
        health_score=int(analysis.get("health_score", 0)),
        watering_status=analysis.get("watering_status", "unknown"),
        watering_confidence=int(analysis.get("watering_confidence", 0)),
        sunlight_status=analysis.get("sunlight_status", "unknown"),
        sunlight_confidence=int(analysis.get("sunlight_confidence", 0)),
        observations=analysis.get("observations", ""),
        issues_detected=json.dumps(analysis.get("issues_detected", [])),
        recommendations=json.dumps(analysis.get("recommendations", [])),
        original_image_path=str(original_path.relative_to(BASE_DIR)),
        annotated_image_path=str(annotated_path.relative_to(BASE_DIR)),
        raw_analysis=json.dumps(analysis),
        analyzer=analysis.get("analyzer", "unknown"),
    )

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO readings (
                plant_id, timestamp, overall_health, health_score,
                watering_status, watering_confidence,
                sunlight_status, sunlight_confidence,
                observations, issues_detected, recommendations,
                original_image_path, annotated_image_path, raw_analysis, analyzer
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                reading.plant_id, reading.timestamp,
                reading.overall_health, reading.health_score,
                reading.watering_status, reading.watering_confidence,
                reading.sunlight_status, reading.sunlight_confidence,
                reading.observations, reading.issues_detected,
                reading.recommendations,
                reading.original_image_path, reading.annotated_image_path,
                reading.raw_analysis, reading.analyzer,
            ),
        )
        conn.commit()

    log.info("Reading stored for plant %s: %s (%s/100) via %s",
             plant["name"], reading.overall_health, reading.health_score, reading.analyzer)
    return reading


# -------- Plant CRUD helpers --------
def add_plant(name: str, species: str = "", location: str = "", notes: str = "") -> int:
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO plants (name, species, location, date_added, notes) VALUES (?,?,?,?,?)",
            (name, species, location, datetime.now().isoformat(), notes),
        )
        conn.commit()
        return cur.lastrowid


def log_care_event(plant_id: int, event_type: str, notes: str = ""):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO care_events (plant_id, timestamp, event_type, notes) VALUES (?,?,?,?)",
            (plant_id, datetime.now().isoformat(), event_type, notes),
        )
        conn.commit()


if __name__ == "__main__":
    init_db()
    print(f"Database initialised. Analyzer mode: {ANALYZER_MODE}")
