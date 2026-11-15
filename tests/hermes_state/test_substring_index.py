"""Compact candidates must return exactly the canonical full-body LIKE results."""

import pytest

from hermes_state import SessionDB
from hermes_state_substring import INDEX, READY, _install, build_index, index_ready


@pytest.fixture
def seeded(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    for sid, source in (("chat", "telegram"), ("job", "cron")):
        db.create_session(sid, source=source)
    for i, body in enumerate((
        "padding " * 2000 + "한글검색 마지막문장 NEEDLE",
        "abc xyz without contiguous abcxyz",
        "abc bcx xyz are all present but not a contiguous phrase",
        "abcxyz and emoji 🐈🐕🐀 and café École",
        "한글 검색 짧은 말 needle 10% under_score back\\slash",
        "alpha beta gamma delta",
        "ALPHA delta",
        "we exclude nothing without an exact canonical match",
    )):
        db.append_message("job" if i % 2 else "chat", role="tool", content=body,
                          tool_name="outside_column", tool_calls=[{"name": "terminal", "arguments": "crossfield"}])
    for role, active, compacted in (("user", 1, 0), ("assistant", 0, 1), ("tool", 0, 0)):
        mid = db.append_message("chat", role=role, content="needle archivedvisibility")
        db._conn.execute("UPDATE messages SET active=?,compacted=? WHERE id=?", (active, compacted, mid))
    db._conn.commit()
    try:
        yield db, path
    finally:
        db.close()


def test_compact_index_matches_canonical_queries_and_filters(seeded):
    db, path = seeded
    queries = ["한글검색", "마지막문장", "needle", "abcxyz", "alpha NOT beta", "alpha OR 한글검색",
               "한글 OR needle", '"alpha beta"', "outside_column crossfield", "10%", "under_score",
               "back\\slash", "🐈🐕🐀", "café", "École", "école", "말", "missing", "NOT AND needle"]
    options = [dict(role_filter=["tool"], sort=sort, source_filter=source, fields=("id", "snippet"))
               for sort in ("newest", "oldest") for source in (None, ["cron"], ["telegram"])]
    options += [dict(role_filter=["user", "assistant", "tool"], fields=("id", "snippet"), **filters)
                for filters in (dict(include_inactive=True), dict(before_ts=1),
                                dict(after_ts=1), dict(limit=1, offset=1))]
    expected = {(query, i): db.search_messages(query, **option)
                for query in queries for i, option in enumerate(options)}
    build_index(path)
    assert db._read_one("SELECT value FROM state_meta WHERE key=?", (READY,))[0] == "1"
    for query in queries:
        for i, option in enumerate(options):
            assert db.search_messages(query, **option) == expected[query, i], (query, option)


def test_backfill_and_live_mutations_preserve_membership(seeded):
    db, path = seeded
    _install(db._conn)
    # Updating/deleting a not-yet-indexed row must not issue a fictitious FTS delete.
    first = db.get_messages("chat")[0]["id"]
    db._conn.execute("UPDATE messages SET content='beforebackfill' WHERE id=?", (first,))
    deleted = db.get_messages("job")[0]["id"]
    db._conn.execute("DELETE FROM messages WHERE id=?", (deleted,))
    db._conn.commit()

    build_index(path, batch_size=2)
    new = db.append_message("chat", role="tool", content="newtoken")
    assert db.search_messages("newtoken", role_filter=["tool"])[0]["id"] == new
    db._conn.execute("UPDATE messages SET content='changedtoken' WHERE id=?", (new,))
    db._conn.commit()
    assert not db.search_messages("newtoken", role_filter=["tool"])
    assert db.search_messages("changedtoken", role_filter=["tool"])[0]["id"] == new
    db._conn.execute("DELETE FROM messages WHERE id=?", (new,))
    db._conn.commit()
    assert not db.search_messages("changedtoken", role_filter=["tool"])
    db._conn.execute(f"INSERT INTO {INDEX}({INDEX},rank) VALUES('integrity-check',1)")
    db._conn.commit()


def test_fts_detach_preserves_writes_and_rebuild_covers_the_gap(seeded):
    db, path = seeded
    build_index(path)
    with db._conn:
        db._drop_fts_triggers(db._conn)
    assert not index_ready(db)
    mid = db.append_message("chat", role="tool", content="writtenwhileindexdetached")
    build_index(path)
    assert index_ready(db)
    assert db.search_messages("writtenwhileindexdetached", role_filter=["tool"])[0]["id"] == mid


def test_corrupt_substring_index_cannot_prevent_message_storage(seeded):
    db, path = seeded
    build_index(path)
    db._conn.execute(f"UPDATE {INDEX}_data SET block=X'DEADBEEFDEADBEEF'")
    db._conn.commit()
    mid = db.append_message("chat", role="user", content="durable despite a broken search index")
    assert any(row["id"] == mid for row in db.get_messages("chat"))
    assert not index_ready(db)
