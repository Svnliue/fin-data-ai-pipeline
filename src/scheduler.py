import time
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
import schedule
from pipeline_runner import run_full_pipeline
from config import SCHEDULE_TIME
from logger import get_logger

log = get_logger("scheduler")
NY_TZ = ZoneInfo("America/New_York")


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + (n - 1) * 7)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    nxt = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    last_day = nxt - timedelta(days=1)
    offset = (last_day.weekday() - weekday) % 7
    return last_day - timedelta(days=offset)


def _good_friday(year: int) -> date:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day) - timedelta(days=2)


def _is_nyse_holiday(dt: date) -> bool:
    y = dt.year
    holidays = {
        _nth_weekday(y, 1, 0, 3),   # MLK Day
        _nth_weekday(y, 2, 0, 3),   # Presidents' Day
        _good_friday(y),            # Good Friday
        _last_weekday(y, 5, 0),     # Memorial Day
        _nth_weekday(y, 9, 0, 1),   # Labor Day
        _nth_weekday(y, 11, 3, 4),  # Thanksgiving Day
    }
    fixed = [
        date(y, 1, 1),    # New Year's Day
        date(y, 6, 19),   # Juneteenth
        date(y, 7, 4),    # Independence Day
        date(y, 12, 25),  # Christmas Day
    ]
    for h in fixed:
        if h.weekday() == 5:
            if h.month != 1:
                holidays.add(h - timedelta(days=1))
        elif h.weekday() == 6:
            holidays.add(h + timedelta(days=1))
        else:
            holidays.add(h)
    return dt in holidays


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
