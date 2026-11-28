import time
from datetime import datetime, date
from zoneinfo import ZoneInfo
import schedule
from pipeline_runner import run_full_pipeline
from config import SCHEDULE_TIME
from logger import get_logger

log = get_logger("scheduler")
NY_TZ = ZoneInfo("America/New_York")


def _is_nyse_holiday(dt: date) -> bool:
    y = dt.year
    fixed = {
        date(y, 1, 1),
        date(y, 6, 19),
        date(y, 7, 4),
        date(y, 12, 25),
    }
    observed = set()
    for h in fixed:
        if h.weekday() == 5:
            observed.add(date(y, h.month, max(1, h.day - 1)))
        elif h.weekday() == 6:
            observed.add(date(y, h.month, min(28, h.day + 1)))
        else:
            observed.add(h)
    return dt in observed


def job() -> None:
    now_ny = datetime.now(NY_TZ)
    if now_ny.weekday() >= 5 or _is_nyse_holiday(now_ny.date()):
        log.info(f"NYSE kapali ({now_ny.date()}), calisma atlandi.")
        return

    log.info(f"Zamanlanmis gorev tetiklendi (NY: {now_ny.strftime('%Y-%m-%d %H:%M %Z')})")
    try:
        run_full_pipeline(skip_fetch=False, notify=True)
    except Exception as exc:
        log.error(f"Zamanlanmis gorev hatasi: {exc}", exc_info=True)


def main() -> None:
    schedule.every().day.at(SCHEDULE_TIME, "America/New_York").do(job)
    log.info(f"Otonom Zamanlayici aktif: Her NYSE islem gunu {SCHEDULE_TIME} America/New_York.")
    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
