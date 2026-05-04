"""Plant Health Monitor - Web UI (with analyzer mode support)."""

import json
import threading
from datetime import datetime, timedelta

from flask import (
    Flask, render_template, request, redirect, url_for,
    send_from_directory, jsonify, flash,
)

from plant_monitor import (
    init_db, get_db, add_plant, capture_and_analyse, log_care_event,
    CAPTURES_DIR, ANALYZER_MODE,
)
from analyzers import get_analyzer, CloudAnalyzer

#app = Flask(__name__)
from pathlib import Path

_BASE = Path(__file__).resolve().parent.parent
app = Flask(
    __name__,
    template_folder=str(_BASE / "templates"),
    static_folder=str(_BASE / "static") if (_BASE / "static").exists() else None,
)

app.secret_key = "change-me-in-production"

_capture_lock = threading.Lock()


@app.context_processor
def inject_globals():
    """Make analyzer status available to all templates."""
    cloud = CloudAnalyzer()
    return {
        "analyzer_mode": ANALYZER_MODE,
        "cloud_available": cloud.is_available(),
    }


@app.route("/")
def index():
    with get_db() as conn:
        plants = conn.execute("""
            SELECT p.*,
                   (SELECT r.overall_health FROM readings r WHERE r.plant_id = p.id
                    ORDER BY r.timestamp DESC LIMIT 1) AS latest_health,
                   (SELECT r.health_score FROM readings r WHERE r.plant_id = p.id
                    ORDER BY r.timestamp DESC LIMIT 1) AS latest_score,
                   (SELECT r.timestamp FROM readings r WHERE r.plant_id = p.id
                    ORDER BY r.timestamp DESC LIMIT 1) AS latest_ts,
                   (SELECT r.annotated_image_path FROM readings r WHERE r.plant_id = p.id
                    ORDER BY r.timestamp DESC LIMIT 1) AS latest_image,
                   (SELECT r.analyzer FROM readings r WHERE r.plant_id = p.id
                    ORDER BY r.timestamp DESC LIMIT 1) AS latest_analyzer
            FROM plants p
            ORDER BY p.name
        """).fetchall()
    return render_template("index.html", plants=plants)


@app.route("/plant/new", methods=["GET", "POST"])
def new_plant():
    if request.method == "POST":
        pid = add_plant(
            name=request.form["name"].strip(),
            species=request.form.get("species", "").strip(),
            location=request.form.get("location", "").strip(),
            notes=request.form.get("notes", "").strip(),
        )
        flash(f"Added plant (id {pid})")
        return redirect(url_for("plant_detail", plant_id=pid))
    return render_template("new_plant.html")


@app.route("/plant/<int:plant_id>")
def plant_detail(plant_id):
    with get_db() as conn:
        plant = conn.execute("SELECT * FROM plants WHERE id = ?", (plant_id,)).fetchone()
        if not plant:
            return "Plant not found", 404
        readings = conn.execute(
            "SELECT * FROM readings WHERE plant_id = ? ORDER BY timestamp DESC",
            (plant_id,),
        ).fetchall()
        care = conn.execute(
            "SELECT * FROM care_events WHERE plant_id = ? ORDER BY timestamp DESC LIMIT 20",
            (plant_id,),
        ).fetchall()

    parsed_readings = []
    for r in readings:
        d = dict(r)
        d["issues"] = json.loads(d["issues_detected"] or "[]")
        d["recs"] = json.loads(d["recommendations"] or "[]")
        parsed_readings.append(d)

    chart_data = [
        {"t": r["timestamp"], "score": r["health_score"],
         "water": r["watering_status"], "light": r["sunlight_status"],
         "analyzer": r["analyzer"] or "?"}
        for r in reversed(parsed_readings)
    ]

    return render_template(
        "plant_detail.html",
        plant=plant,
        readings=parsed_readings,
        care=care,
        chart_data=json.dumps(chart_data),
    )


@app.route("/plant/<int:plant_id>/capture", methods=["POST"])
def trigger_capture(plant_id):
    if not _capture_lock.acquire(blocking=False):
        flash("Another capture is in progress — try again in a moment.")
        return redirect(url_for("plant_detail", plant_id=plant_id))
    try:
        mode = request.form.get("analyzer_mode")  # optional override
        reading = capture_and_analyse(plant_id, analyzer_mode=mode)
        flash(
            f"Captured via {reading.analyzer} — "
            f"health: {reading.overall_health} ({reading.health_score}/100)"
        )
    except Exception as e:
        flash(f"Capture failed: {e}")
    finally:
        _capture_lock.release()
    return redirect(url_for("plant_detail", plant_id=plant_id))


@app.route("/plant/<int:plant_id>/care", methods=["POST"])
def add_care(plant_id):
    log_care_event(
        plant_id=plant_id,
        event_type=request.form["event_type"],
        notes=request.form.get("notes", "").strip(),
    )
    flash("Care event logged")
    return redirect(url_for("plant_detail", plant_id=plant_id))


@app.route("/captures/<path:filename>")
def serve_capture(filename):
    return send_from_directory(CAPTURES_DIR, filename)


@app.route("/api/plants/<int:plant_id>/history")
def api_history(plant_id):
    days = int(request.args.get("days", 30))
    since = (datetime.now() - timedelta(days=days)).isoformat()
    with get_db() as conn:
        rows = conn.execute(
            "SELECT timestamp, health_score, watering_status, sunlight_status, analyzer "
            "FROM readings WHERE plant_id = ? AND timestamp >= ? ORDER BY timestamp",
            (plant_id, since),
        ).fetchall()
    return jsonify([dict(r) for r in rows])


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=False)
