import subprocess
import sys
import time
from datetime import datetime
from logger import get_logger
from config import SCHEDULE_TIME

log = get_logger("scheduler")


def run_once() -> None:
    log.info("Zamanlanmış otonom pipeline tetikleniyor...")
    res = subprocess.run([sys.executable, "src/run_pipeline.py"])
    if res.returncode == 0:
        log.info("Zamanlanmış pipeline başarıyla tamamlandı.")
    else:
        log.error(f"Zamanlanmış pipeline hata koduyla sonlandı: {res.returncode}")


def main() -> None:
    if "--now" in sys.argv:
        run_once()
        return

    log.info(f"Otonom Zamanlayıcı Aktif. Hafta içi her gün saat {SCHEDULE_TIME} bekleniyor (Çıkmak için Ctrl+C)...")
    last_run_date = None

    while True:
        now = datetime.now()
        is_weekday = now.weekday() < 5
        current_hhmm = now.strftime("%H:%M")

        if is_weekday and current_hhmm == SCHEDULE_TIME and last_run_date != now.date():
            run_once()
            last_run_date = now.date()

        time.sleep(20)


if __name__ == "__main__":
    main()
