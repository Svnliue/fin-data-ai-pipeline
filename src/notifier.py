import os
import requests
from logger import get_logger

log = get_logger("notifier")


def format_report_message(report: dict) -> str:
    macro = report.get("macro", {})
    cap = report.get("capital", {})
    held = report.get("held", [])
    radar = report.get("radar", [])
    warnings = report.get("warnings") or []

    macro_status = "BOGA (SMA200 Ustu)" if macro.get("is_bull") else "AYI (SMA200 Alti)"
    lines = [
        "ALPHA ENGINE v4.9.16 RAPORU",
        f"Makro Rejim [QQQ]: ${macro.get('close', 0.0):.2f} vs SMA-200 ${macro.get('sma_200', 0.0):.2f} -> {macro_status}",
        (
            f"Sermaye: Efektif ${cap.get('effective_capital', 0.0):,.2f} "
            f"(Realized: ${cap.get('realized_pnl', 0.0):+,.2f}) | "
            f"Serbest Nakit: ${cap.get('available_cash', 0.0):,.2f} | "
            f"Pozisyon: {cap.get('open_positions_count', 0)}/{cap.get('max_positions', 3)}"
        ),
        "",
        "ACIK POZISYONLARIM:",
    ]

    if not held:
        lines.append("- Acik pozisyon yok.")
    else:
        for item in held:
            lines.append(
                f"- {item.get('symbol')}: {item.get('decision')} | "
                f"Fiyat: ${item.get('price', 0.0):.2f} | PnL: {item.get('pnl_pct', 0.0):+.2f}% | "
                f"SL: ${item.get('stop_loss', 0.0):.2f} | TP1: ${item.get('partial_tp', 0.0):.2f}"
            )

    actionable = [r for r in radar if str(r.get("decision", "")).startswith(("AL", "GÜÇLÜ AL", "BEKLE (KİLİTLİ"))]
    lines.append("")
    lines.append("RADAR (ONEMLI SINYALLER):")
    if not actionable:
        lines.append("- Yeni giris veya kilitli sinyal yok.")
    else:
        for r in actionable:
            lines.append(f"- {r.get('symbol')} [{r.get('asset_type')}]: {r.get('decision')} (Skor: {r.get('score', 0.0):+.2f})")

    if warnings:
        lines.append("")
        lines.append("UYARILAR:")
        for w in warnings:
            lines.append(f"- {w}")

    return "\n".join(lines)


def send_telegram_message(text: str) -> bool:
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    chat_id = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()
    if not token or not chat_id:
        log.info("Telegram ayarlari (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID) yapilandirilmamis, bildirim atlandi.")
        return False

    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=10,
        )
        resp.raise_for_status()
        log.info("Telegram bildirimi basariyla gonderildi.")
        return True
    except Exception as exc:
        log.error(f"Telegram bildirimi gonderilemedi: {exc}")
        return False


def notify_report(report: dict) -> bool:
    msg = format_report_message(report)
    return send_telegram_message(msg)
