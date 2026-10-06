import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'app'))
os.environ['AI_MONITOR_DISABLE_RUNTIME']='1'
import pytest
import db
from stream_capture import FrameCache

@pytest.fixture(autouse=True)
def database(tmp_path,monkeypatch):
    monkeypatch.setattr(db,'DB_PATH',tmp_path/'test.db')
    monkeypatch.delenv('TELEGRAM_BOT_TOKEN',raising=False)
    monkeypatch.delenv('TELEGRAM_CHAT_ID',raising=False)
    FrameCache.get_instance()._frames.clear()
    db.init_db()
