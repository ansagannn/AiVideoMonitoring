"""Persistent event queue with retries; only risk events, including marked demos."""
import logging
import os
import threading
import time
import db
import telegram


class Notifier:
    def __init__(self):
        self.halt = threading.Event()
        self.thread = None

    def start(self):
        if not os.getenv('TELEGRAM_BOT_TOKEN') or not os.getenv('TELEGRAM_CHAT_ID'): return
        with db.connect() as con:
            con.execute('CREATE TABLE IF NOT EXISTS alert_attempts (event_id TEXT PRIMARY KEY, next_attempt REAL NOT NULL)')
        self.halt.clear()
        self.thread = threading.Thread(target=self.loop, daemon=True, name='alert-dispatcher')
        self.thread.start()

    def dispatch(self):
        with db.connect() as con:
            rows = con.execute("SELECT e.id FROM events e LEFT JOIN alert_attempts a ON e.id=a.event_id WHERE e.telegram_sent=0 AND e.status='new' AND e.type IN ('suspicious','fight','fall','danger','system_stream_lost') AND (a.next_attempt IS NULL OR a.next_attempt<=?) ORDER BY e.detected_at LIMIT 5", (time.time(),)).fetchall()
        for row in rows:
            event = db.get_event(row['id'])
            preview = telegram.build_preview(event)
            prefix = 'ДЕМО / ДЕМО\n' if event.detection_mode == 'demo' else ''
            result = telegram.send_message(prefix+preview.text, event)
            with db.connect() as con:
                if result.sent: con.execute('UPDATE events SET telegram_sent=1 WHERE id=?', (event.id,))
                con.execute('INSERT OR REPLACE INTO alert_attempts VALUES (?,?)', (event.id,time.time()+60))

    def loop(self):
        while not self.halt.is_set():
            try: self.dispatch()
            except Exception: logging.exception('Alert dispatch failed')
            self.halt.wait(3)

    def stop(self):
        self.halt.set()
        if self.thread: self.thread.join(timeout=2)

notifier = Notifier()
