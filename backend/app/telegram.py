from __future__ import annotations

import json
import os
from typing import Literal
from urllib import request

from models import TelegramButton, TelegramPreview, TelegramTestResponse, VideoEvent


def build_preview(event: VideoEvent) -> TelegramPreview:
    mode: Literal["telegram", "mock"] = "telegram" if os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID") else "mock"
    return TelegramPreview(
        mode=mode,
        text=(
            f"Событие / Оқиға: {event.title}\n"
            f"Камера: {event.camera_name}\n"
            f"Зона / Аймақ: {event.zone}\n"
            f"Оценка / Баға: {round(event.confidence * 100)}%\n"
            f"Время / Уақыт: {event.detected_at}"
        ),
        buttons=[
            TelegramButton(label="Подтвердить / Растау", action="confirmed", callback_data=f"feedback:{event.id}:confirmed"),
            TelegramButton(label="Ложная тревога / Жалған дабыл", action="dismissed", callback_data=f"feedback:{event.id}:dismissed"),
        ],
    )


def send_message(text: str, event: VideoEvent) -> TelegramTestResponse:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    preview = build_preview(event)
    if not token or not chat_id:
        return TelegramTestResponse(
            configured=False,
            sent=False,
            mode="mock",
            detail="TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID are not configured.",
            inline_feedback=True,
            preview=preview,
        )
    payload = {
        "chat_id": chat_id,
        "text": text,
        "reply_markup": {
            "inline_keyboard": [[
                {"text": "Подтвердить / Растау", "callback_data": f"feedback:{event.id}:confirmed"},
                {"text": "Ложная тревога / Жалған дабыл", "callback_data": f"feedback:{event.id}:dismissed"},
            ]]
        },
    }
    req = request.Request(
        url=f"https://api.telegram.org/bot{token}/sendMessage",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=8) as response:
            result = json.loads(response.read())
            sent = 200 <= response.status < 300 and result.get("ok") is True
    except OSError as exc:
        return TelegramTestResponse(
            configured=True,
            sent=False,
            mode="telegram",
            detail=str(exc),
            inline_feedback=True,
            preview=preview,
        )
    return TelegramTestResponse(
        configured=True,
        sent=sent,
        mode="telegram",
        detail="Telegram request completed.",
        inline_feedback=True,
        preview=preview,
    )
