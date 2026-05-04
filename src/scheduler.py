"""Plant Health Monitor - Scheduler (daily auto-capture)."""

import time
import schedule

from plant_monitor import init_db, get_db, capture_and_analyse, log

CAPTURE_TIME = "09:00"  # 24h local time


def run_daily_captures():
    log.info("Starting scheduled capture run")
    with get_db() as conn:
        plant_ids = [row["id"] for row in conn.execute("SELECT id FROM plants").fetchall()]

    for pid in plant_ids:
        try:
            capture_and_analyse(pid)
            time.sleep(5)
        except Exception as e:
            log.exception("Failed to capture plant %s: %s", pid, e)

    log.info("Scheduled capture run complete (%d plants)", len(plant_ids))


def main():
    init_db()
    schedule.every().day.at(CAPTURE_TIME).do(run_daily_captures)
    log.info("Scheduler running. Daily capture at %s.", CAPTURE_TIME)
    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
