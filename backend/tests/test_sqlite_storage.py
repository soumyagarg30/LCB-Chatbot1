import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_store_chat_message_persists_to_sqlite(tmp_path, monkeypatch):
    db_path = tmp_path / "test.sqlite"
    monkeypatch.setenv("DATABASE_PATH", str(db_path))

    import db_utils

    db_utils = importlib.reload(db_utils)
    db_utils.init_db()

    message_id = db_utils.store_chat_message("session-123", "user", "Hello from SQLite")

    assert message_id is not None

    messages = db_utils.get_chat_messages("session-123", limit=10)
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello from SQLite"
