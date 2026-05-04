"""Plant Health Monitor - CLI tool.

Examples:
    python cli.py init
    python cli.py add-plant "Kitchen Monstera" --species "Monstera deliciosa"
    python cli.py list
    python cli.py capture 1                 # uses PLANT_ANALYZER env var (default: auto)
    python cli.py capture 1 --mode local    # force local offline analysis
    python cli.py capture 1 --mode cloud    # force cloud (Claude) analysis
    python cli.py history 1
    python cli.py care 1 watered --notes "Gave about 200ml"
"""

import argparse
import json
from plant_monitor import (
    init_db, get_db, add_plant, capture_and_analyse, log_care_event,
)


def cmd_init(_):
    init_db()
    print("✅ Database initialised.")


def cmd_add_plant(args):
    pid = add_plant(args.name, args.species or "", args.location or "", args.notes or "")
    print(f"✅ Added plant '{args.name}' (id {pid})")


def cmd_list(_):
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM plants ORDER BY name").fetchall()
    if not rows:
        print("No plants yet. Use 'add-plant' to add one.")
        return
    print(f"\n{'ID':<4} {'Name':<25} {'Species':<25} {'Location':<20}")
    print("-" * 76)
    for r in rows:
        print(f"{r['id']:<4} {r['name']:<25} {(r['species'] or '-'):<25} {(r['location'] or '-'):<20}")


def cmd_capture(args):
    print(f"Capturing plant {args.plant_id} (mode={args.mode or 'auto'})...")
    reading = capture_and_analyse(args.plant_id, analyzer_mode=args.mode)
    print(f"\n✅ Analyzer:  {reading.analyzer}")
    print(f"   Health:    {reading.overall_health} ({reading.health_score}/100)")
    print(f"   Water:     {reading.watering_status} ({reading.watering_confidence}%)")
    print(f"   Light:     {reading.sunlight_status} ({reading.sunlight_confidence}%)")
    print(f"   Observations: {reading.observations}")
    recs = json.loads(reading.recommendations)
    if recs:
        print("   Recommendations:")
        for r in recs:
            print(f"     • {r}")
    print(f"\n   Annotated: {reading.annotated_image_path}")


def cmd_history(args):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT timestamp, overall_health, health_score, watering_status, sunlight_status, analyzer "
            "FROM readings WHERE plant_id = ? ORDER BY timestamp DESC LIMIT ?",
            (args.plant_id, args.limit),
        ).fetchall()
    if not rows:
        print("No readings yet for this plant.")
        return
    print(f"\n{'When':<19} {'Health':<11} {'Score':<6} {'Water':<15} {'Light':<15} {'Analyzer':<10}")
    print("-" * 82)
    for r in rows:
        when = r["timestamp"][:16].replace("T", " ")
        print(f"{when:<19} {r['overall_health']:<11} {r['health_score']:<6} "
              f"{r['watering_status']:<15} {r['sunlight_status']:<15} {(r['analyzer'] or '?'):<10}")


def cmd_care(args):
    log_care_event(args.plant_id, args.event, args.notes or "")
    print(f"✅ Logged '{args.event}' for plant {args.plant_id}")


def cmd_status(_):
    """Show which analyzers are available."""
    from analyzers import CloudAnalyzer, LocalAnalyzer
    cloud = CloudAnalyzer()
    local = LocalAnalyzer()
    print("Analyzer status:")
    print(f"  cloud (Claude API): {'✅ available' if cloud.is_available() else '❌ unavailable (no key or offline)'}")
    print(f"  local (offline CV): {'✅ available' if local.is_available() else '❌ unavailable'}")
    print(f"  local disease model: {'✅ loaded' if local.classifier.available() else '⚠️  not installed (optional)'}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init").set_defaults(func=cmd_init)
    sub.add_parser("status").set_defaults(func=cmd_status)

    s = sub.add_parser("add-plant")
    s.add_argument("name")
    s.add_argument("--species")
    s.add_argument("--location")
    s.add_argument("--notes")
    s.set_defaults(func=cmd_add_plant)

    sub.add_parser("list").set_defaults(func=cmd_list)

    s = sub.add_parser("capture")
    s.add_argument("plant_id", type=int)
    s.add_argument("--mode", choices=["auto", "cloud", "local"],
                   help="Override analyzer mode (default: env var PLANT_ANALYZER or 'auto')")
    s.set_defaults(func=cmd_capture)

    s = sub.add_parser("history")
    s.add_argument("plant_id", type=int)
    s.add_argument("--limit", type=int, default=20)
    s.set_defaults(func=cmd_history)

    s = sub.add_parser("care")
    s.add_argument("plant_id", type=int)
    s.add_argument("event", choices=["watered", "fertilized", "repotted", "moved", "pruned", "treated"])
    s.add_argument("--notes")
    s.set_defaults(func=cmd_care)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
