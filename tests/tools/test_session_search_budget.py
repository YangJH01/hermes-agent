"""Conversation recall stays useful when automated history exhausts its budget."""

import json

from hermes_state import SessionDB
from tools.session_search_tool import session_search


def test_cron_timeout_preserves_conversation_hits_and_reports_incomplete(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("chat", source="telegram")
    db.append_message("chat", role="tool", content="needle")
    original = db.search_messages

    def search(*args, **kwargs):
        if kwargs.get("source_filter") == ["cron"]:
            raise TimeoutError("Session search exceeded its time budget")
        return original(*args, **kwargs)

    monkeypatch.setattr(db, "search_messages", search)
    try:
        result = json.loads(session_search("needle", role_filter="tool", db=db))
        assert result["success"] is True
        assert result["results"][0]["session_id"] == "chat"
        assert result["incomplete"] is True
        assert "time budget" in result["message"]
        empty = json.loads(session_search("absent", role_filter="tool", db=db))
        assert empty["incomplete"] is True
        assert "No matching sessions found" not in empty["message"]
    finally:
        db.close()
