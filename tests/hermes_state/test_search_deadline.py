"""Interactive recall cancels SQLite work and leaves the reader usable."""

import pytest

from hermes_state import SessionDB


def test_search_deadline_cancels_sql_and_cleans_up(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("chat", source="telegram")
    db.append_message("chat", role="tool", content="needle")
    original = db._search_messages_impl

    def expensive_query(*args, **kwargs):
        return db._search_read_all(
            "WITH RECURSIVE n(x) AS (VALUES(0) UNION ALL SELECT x+1 FROM n "
            "WHERE x<100000000) SELECT sum(x) FROM n")

    try:
        monkeypatch.setattr(db, "_search_messages_impl", expensive_query)
        with pytest.raises(TimeoutError, match="time budget"):
            db.search_messages("needle", timeout_seconds=0.01)
        monkeypatch.setattr(db, "_search_messages_impl", original)
        assert db.search_messages("needle", role_filter=["tool"], timeout_seconds=2)[0]["session_id"] == "chat"
        assert db.search_messages("needle", role_filter=["tool"])[0]["session_id"] == "chat"
    finally:
        db.close()
