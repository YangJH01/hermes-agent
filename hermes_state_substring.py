"""Opt-in compact substring candidates; canonical LIKE remains the final authority.

Only term/document membership is stored: no duplicate bodies, positions or lengths.
The projection folds ASCII only, exactly like SQLite's default LIKE. Candidate
trigrams may overmatch across columns or positions; the original predicate removes
those false positives. Short terms and unready indexes retain the complete scan.
"""

import argparse
import re
import sqlite3
import time
from pathlib import Path

INDEX = "messages_fts_substring"
READY = "substring_index_version"
PENDING = "substring_index_pending"
STALE = "substring_index_stale"
_ASCII_LOWER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")
_TOKENS = re.compile(r'"[^\"]+"|\S+')
_COLUMNS = ("content", "tool_name", "tool_calls")


def like_groups(query):
    """The canonical LIKE route's existing boolean subset, shared with its prefilter."""
    groups, negate = [[]], False
    for token in _TOKENS.findall(query):
        op = token.upper()
        if op == "OR":
            if groups[-1]:
                groups.append([])
            negate = False
        elif op in {"AND", "NEAR"}:
            continue
        elif op == "NOT":
            negate = True
        elif term := token.strip('"').strip("*").strip():
            groups[-1].append((term, negate))
            negate = False
    return [group for group in groups if any(not negated for _, negated in group)]


def candidate_match(groups):
    """A necessary condition for every OR arm, or None when no safe filter exists.

    Never negate a lossy candidate: a false positive under NOT would lose a real
    result. Three sampled trigrams per positive term suffice as a prefilter; the
    full literal/phrase, including escaped LIKE wildcards, is verified afterward.
    """
    arms = []
    for group in groups:
        grams = set()
        for term, negated in group:
            if negated or len(term) < 3 or "\x00" in term:
                continue
            folded = term.translate(_ASCII_LOWER)
            for start in (0, (len(folded) - 3) // 2, len(folded) - 3):
                grams.add('"' + folded[start:start + 3].replace('"', '""') + '"')
        if not grams:
            return None
        arms.append("(" + " AND ".join(sorted(grams)) + ")")
    return " OR ".join(arms) or None


def index_ready(db):
    """Read readiness on every search; existing read-only handles see a finished build."""
    return db._read_one(
        "SELECT 1 FROM state_meta WHERE key=? AND value='1' "
        "AND EXISTS(SELECT 1 FROM sqlite_master WHERE name=?) "
        "AND (SELECT count(*) FROM sqlite_master WHERE type='trigger' AND name IN (?,?,?))=3",
        (READY, INDEX, *(f"{INDEX}_{op}" for op in ("insert", "update", "delete")))) is not None


def detach_index(cursor):
    """Join the existing FTS fail-open path: a derived index must never block message writes."""
    if not cursor.execute("SELECT 1 FROM sqlite_master WHERE name=?", (INDEX,)).fetchone():
        return
    for op in ("insert", "update", "delete"):
        cursor.execute(f"DROP TRIGGER IF EXISTS {INDEX}_{op}")
    cursor.execute("DELETE FROM state_meta WHERE key=?", (READY,))
    cursor.execute("INSERT OR REPLACE INTO state_meta(key,value) VALUES(?,'1')", (STALE,))


def _install(conn):
    """One transaction installs the index, backlog and synchronous write maintenance."""
    folded_new = ",".join(f"lower(new.{col})" for col in _COLUMNS)
    folded_old = ",".join(f"lower(old.{col})" for col in _COLUMNS)
    with conn:
        conn.execute("BEGIN IMMEDIATE")
        exists = conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (INDEX,)).fetchone()
        triggers = conn.execute("SELECT count(*) FROM sqlite_master WHERE type='trigger' AND name IN (?,?,?)",
                                tuple(f"{INDEX}_{op}" for op in ("insert", "update", "delete"))).fetchone()[0]
        if exists and triggers == 3:
            return  # Ready or an interrupted, still-maintained backfill.
        # Repair/old-layout maintenance may have detached or removed the index.
        # Rebuild from canonical rows; never re-certify an unmaintained index.
        for op in ("insert", "update", "delete"):
            conn.execute(f"DROP TRIGGER IF EXISTS {INDEX}_{op}")
        conn.execute(f"DROP TABLE IF EXISTS {INDEX}")
        conn.execute(f"DROP VIEW IF EXISTS {INDEX}_src")
        conn.execute(f"DROP TABLE IF EXISTS {PENDING}")
        conn.execute("DELETE FROM state_meta WHERE key IN (?,?,?)", (READY, STALE, "substring_build_baseline"))
        conn.execute(f"CREATE VIEW {INDEX}_src AS SELECT id, " +
                     ",".join(f"lower({col}) AS {col}" for col in _COLUMNS) + " FROM messages")
        conn.execute(f"CREATE VIRTUAL TABLE {INDEX} USING fts5(content,tool_name,tool_calls, "
                     f"content='{INDEX}_src',content_rowid='id',detail=none,columnsize=0, "
                     "tokenize='trigram case_sensitive 1')")
        conn.execute(f"CREATE TABLE {PENDING}(id INTEGER PRIMARY KEY)")
        conn.execute(f"INSERT INTO {PENDING} SELECT id FROM messages")
        conn.execute(f"""CREATE TRIGGER {INDEX}_insert AFTER INSERT ON messages BEGIN
            INSERT INTO {INDEX}(rowid,content,tool_name,tool_calls)
            VALUES(new.id,{folded_new}); END""")
        conn.execute(f"""CREATE TRIGGER {INDEX}_delete AFTER DELETE ON messages BEGIN
            INSERT INTO {INDEX}({INDEX},rowid,content,tool_name,tool_calls)
            SELECT 'delete',old.id,{folded_old} WHERE NOT EXISTS(SELECT 1 FROM {PENDING} WHERE id=old.id);
            DELETE FROM {PENDING} WHERE id=old.id; END""")
        conn.execute(f"""CREATE TRIGGER {INDEX}_update AFTER UPDATE OF content,tool_name,tool_calls ON messages
            WHEN (new.content IS NOT old.content OR new.tool_name IS NOT old.tool_name OR new.tool_calls IS NOT old.tool_calls)
            AND NOT EXISTS(SELECT 1 FROM {PENDING} WHERE id=old.id) BEGIN
            INSERT INTO {INDEX}({INDEX},rowid,content,tool_name,tool_calls) VALUES('delete',old.id,{folded_old});
            INSERT INTO {INDEX}(rowid,content,tool_name,tool_calls) VALUES(new.id,{folded_new}); END""")


def build_index(path, *, max_growth_mib=8192, batch_size=100):
    """Resumable opt-in build; each transaction indexes only a small backlog batch.

    No startup migration and no model calls. New writes are indexed by SQLite
    triggers in their original transaction. A failed/paused build is never used
    to prune search candidates. The growth ceiling includes unrelated DB growth,
    conservatively stopping the build instead of exhausting the volume.
    """
    from hermes_cli.sqlite_safe_read import connect_tracked

    if max_growth_mib <= 0 or batch_size <= 0:
        raise ValueError("Growth budget and batch size must be positive")
    conn = connect_tracked(Path(path), timeout=30)
    try:
        conn.execute("PRAGMA cache_size=-65536")
        page_size = conn.execute("PRAGMA page_size").fetchone()[0]
        initial_used = (conn.execute("PRAGMA page_count").fetchone()[0] -
                        conn.execute("PRAGMA freelist_count").fetchone()[0]) * page_size
        _install(conn)
        with conn:
            conn.execute("INSERT OR IGNORE INTO state_meta(key,value) VALUES('substring_build_baseline',?)",
                         (str(initial_used),))
        initial_used = int(conn.execute("SELECT value FROM state_meta WHERE key='substring_build_baseline'").fetchone()[0])
        if conn.execute("SELECT 1 FROM state_meta WHERE key=?", (READY,)).fetchone():
            return
        done, last_report = 0, time.monotonic()
        while True:
            with conn:
                conn.execute("BEGIN IMMEDIATE")
                rows = conn.execute(f"SELECT m.id,m.content,m.tool_name,m.tool_calls FROM {PENDING} p "
                                    "JOIN messages m ON m.id=p.id ORDER BY p.id LIMIT ?", (batch_size,)).fetchall()
                if not rows:
                    break
                conn.executemany(f"INSERT INTO {INDEX}(rowid,content,tool_name,tool_calls) "
                                 "VALUES(?,lower(?),lower(?),lower(?))", rows)
                conn.executemany(f"DELETE FROM {PENDING} WHERE id=?", [(row[0],) for row in rows])
            done += len(rows)
            if time.monotonic() - last_report >= 10:
                used = (conn.execute("PRAGMA page_count").fetchone()[0] -
                        conn.execute("PRAGMA freelist_count").fetchone()[0]) * page_size
                growth = used - initial_used
                print(f"indexed={done} growth_mib={growth / 2**20:.1f}", flush=True)
                if growth > max_growth_mib * 2**20:
                    raise RuntimeError("Substring build reached its growth budget; incomplete index remains unused")
                last_report = time.monotonic()
            time.sleep(0.02)  # Yield the writer between batches; live bot writes keep running.
        with conn:
            used = (conn.execute("PRAGMA page_count").fetchone()[0] -
                    conn.execute("PRAGMA freelist_count").fetchone()[0]) * page_size
            if used - initial_used > max_growth_mib * 2**20:
                raise RuntimeError("Substring build reached its growth budget; incomplete index remains unused")
            # Structural check only here. rank=1 re-tokenizes the entire corpus while
            # holding the writer lock; use it on an offline copy for full verification.
            conn.execute(f"INSERT INTO {INDEX}({INDEX}) VALUES('integrity-check')")
            conn.execute("INSERT OR REPLACE INTO state_meta(key,value) VALUES(?,'1')", (READY,))
        print(f"complete indexed={done}", flush=True)
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--max-growth-mib", type=int, default=8192)
    args = parser.parse_args()
    build_index(args.database, max_growth_mib=args.max_growth_mib)


if __name__ == "__main__":
    main()
