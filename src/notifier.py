import os
import requests
from dotenv import load_dotenv
from logger import get_logger

load_dotenv()
log = get_logger("notifier")


def send_telegram_alert(report: dict) -> bool:
    """
    Aksiyon gerektiren sinyalleri Telegram'a gonderir.
    TELEGRAM_BOT_TOKEN veya TELEGRAM_CHAT_ID bos ise sessizce atlar.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return False

    actionable = []
    for item in report.get("portfolio", []):
        dec = item["decision"]
        if not dec.startswith("TUT"):
            actionable.append(f"• [PORTFÖY] {item['symbol']}: {dec} (Fiyat: ${item['close']:.2f})")

    for item in report.get("radar", []):
        dec = item["decision"]
        if dec.startswith("AL") or dec.startswith("GÜÇLÜ AL"):
            risk = item["risk"]
            actionable.append(
                f"• [RADAR] {item['symbol']}: {dec} | Lot: {risk['shares']} | SL: ${risk['stop_loss']:.2f} | TP1: ${risk['partial_tp']:.2f}"
            )

    if not actionable:
        log.info("Telegram bildirimi atlandı: Aksiyon gerektiren yeni sinyal yok.")
        return False

    macro = report.get("macro", {})
    regime = "BOĞA" if macro.get("is_bull") else "AYI"
    lines = [
        f"📊 *ALPHA ENGINE RAPORU* (QQQ: ${macro.get('close', 0):.2f} | {regime})",
        "",
        *actionable,
    ]
    text = "\n".join(lines)

    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
        resp.raise_for_status()
        log.info("Telegram sinyalleri başarıyla gönderildi.")
        return True
    except Exception as exc:
        log.error(f"Telegram bildirimi gönderilemedi: {exc}")
        return False
