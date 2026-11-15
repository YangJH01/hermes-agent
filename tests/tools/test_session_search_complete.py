"""Search completes both conversational and scheduled history without truncating on time."""

import json

from hermes_state import SessionDB
from hermes_state_substring import build_index
from tools.session_search_tool import session_search


def test_conversation_and_cron_search_are_complete(tmp_path, monkeypatch):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    for sid, source in (("chat", "telegram"), ("scheduled", "cron")):
        db.create_session(sid, source=source)
        db.append_message(sid, role="tool", content="padding " * 2000 + "needle")
    build_index(path)
    original = db.search_messages

    def search(*args, **kwargs):
        assert kwargs.get("timeout_seconds") is None
        return original(*args, **kwargs)

    monkeypatch.setattr(db, "search_messages", search)
    try:
        result = json.loads(session_search("needle", role_filter="tool", db=db))
        assert result["success"] is True
        assert [r["session_id"] for r in result["results"]] == ["chat", "scheduled"]
        assert not result.get("incomplete")
    finally:
        db.close()
