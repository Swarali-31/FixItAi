"""Thread-safe persistence layer for FixItAI query history and feedback.

Supports:
1. Local SQLite database (ACID compliant, zero setup, thread-safe).
2. Automatic migration from legacy query_history.json.
3. MongoDB support via FIXITAI_MONGODB_URI if explicitly configured.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
_db_lock = threading.Lock()


def _get_db_path(fallback_json_path: str | Path) -> Path:
    p = Path(fallback_json_path)
    return p.parent / "fixitai.db"


def _init_sqlite(db_path: Path, legacy_json_path: Path):
    """Initialize SQLite schema and migrate legacy JSON history if present."""
    with _db_lock:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    result_ids TEXT NOT NULL,
                    feedback TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_history_created ON query_history(created_at DESC)")
            conn.commit()

            # Migrate legacy JSON once if database is empty and JSON exists
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM query_history")
            count = cursor.fetchone()[0]
            if count == 0 and legacy_json_path.exists():
                try:
                    data = json.loads(legacy_json_path.read_text(encoding="utf-8"))
                    if isinstance(data, list) and data:
                        for item in data:
                            conn.execute(
                                "INSERT INTO query_history (query, result_ids, feedback) VALUES (?, ?, ?)",
                                (
                                    item.get("query", ""),
                                    json.dumps(item.get("result_ids", [])),
                                    item.get("feedback")
                                )
                            )
                        conn.commit()
                        logger.info("Migrated %d history records from JSON to SQLite.", len(data))
                except Exception as exc:
                    logger.warning("Could not migrate legacy JSON history: %s", exc)


def _mongo_collection(name: str):
    uri = os.getenv("FIXITAI_MONGODB_URI")
    if not uri:
        return None
    try:
        from pymongo import MongoClient
        client = MongoClient(uri, serverSelectionTimeoutMS=800)
        client.admin.command("ping")
        return client[os.getenv("FIXITAI_MONGODB_DATABASE", "fixitai")][name]
    except Exception:
        return None


def load_history(path: str | Path) -> list[dict]:
    """Load latest search history records."""
    # 1. Check MongoDB
    col = _mongo_collection("query_history")
    if col is not None:
        try:
            return [
                {
                    "query": d.get("query", ""),
                    "result_ids": d.get("result_ids", []),
                    "feedback": d.get("feedback"),
                    "created_at": str(d.get("created_at", ""))
                }
                for d in col.find({}, {"_id": 0}).sort("created_at", 1).limit(200)
            ]
        except Exception:
            pass

    # 2. SQLite
    json_path = Path(path)
    db_path = _get_db_path(json_path)
    try:
        _init_sqlite(db_path, json_path)
        with _db_lock, sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT query, result_ids, feedback, created_at
                FROM query_history
                ORDER BY id ASC
                LIMIT 200
            """)
            rows = cursor.fetchall()
            results = []
            for q, r_ids, fb, dt in rows:
                try:
                    ids = json.loads(r_ids)
                except Exception:
                    ids = []
                results.append({"query": q, "result_ids": ids, "feedback": fb, "created_at": dt})
            return results
    except Exception as exc:
        logger.warning("SQLite read error, falling back to JSON: %s", exc)

    # 3. JSON Fallback
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception:
            return []
    return []


def append_history(path: str | Path, item: dict):
    """Append a new query history entry."""
    # 1. MongoDB
    col = _mongo_collection("query_history")
    if col is not None:
        try:
            col.insert_one({**item, "created_at": datetime.now(timezone.utc)})
            return
        except Exception:
            pass

    # 2. SQLite
    json_path = Path(path)
    db_path = _get_db_path(json_path)
    try:
        _init_sqlite(db_path, json_path)
        with _db_lock, sqlite3.connect(db_path) as conn:
            conn.execute(
                "INSERT INTO query_history (query, result_ids, feedback) VALUES (?, ?, ?)",
                (
                    item.get("query", ""),
                    json.dumps(item.get("result_ids", [])),
                    item.get("feedback")
                )
            )
            conn.commit()
            return
    except Exception as exc:
        logger.warning("SQLite write error, falling back to JSON: %s", exc)

    # 3. JSON Fallback
    try:
        old = []
        if json_path.exists():
            old = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(old, list):
            old = []
        old.append(item)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(old[-200:], indent=2), encoding="utf-8")
    except Exception:
        pass


def save_feedback(path: str | Path, index_from_newest: int, feedback: str):
    """Save user feedback ('Useful' / 'Not useful') for a query at offset from newest."""
    # 1. MongoDB
    col = _mongo_collection("query_history")
    if col is not None:
        try:
            doc = col.find().sort("created_at", -1).skip(index_from_newest).limit(1)
            item = next(doc, None)
            if item:
                col.update_one({"_id": item["_id"]}, {"$set": {"feedback": feedback}})
                return
        except Exception:
            pass

    # 2. SQLite
    json_path = Path(path)
    db_path = _get_db_path(json_path)
    try:
        _init_sqlite(db_path, json_path)
        with _db_lock, sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id FROM query_history
                ORDER BY id DESC
                LIMIT 1 OFFSET ?
            """, (index_from_newest,))
            row = cursor.fetchone()
            if row:
                conn.execute(
                    "UPDATE query_history SET feedback = ? WHERE id = ?",
                    (feedback, row[0])
                )
                conn.commit()
                return
    except Exception as exc:
        logger.warning("SQLite update error, falling back to JSON: %s", exc)

    # 3. JSON Fallback
    try:
        if json_path.exists():
            rows = json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(rows, list) and 0 <= index_from_newest < len(rows):
                rows[-1 - index_from_newest]["feedback"] = feedback
                json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    except Exception:
        pass


def clear_history(path: str | Path) -> bool:
    """Clear all history records."""
    json_path = Path(path)
    db_path = _get_db_path(json_path)
    success = True
    try:
        _init_sqlite(db_path, json_path)
        with _db_lock, sqlite3.connect(db_path) as conn:
            conn.execute("DELETE FROM query_history")
            conn.commit()
    except Exception:
        success = False

    try:
        if json_path.exists():
            json_path.write_text("[]", encoding="utf-8")
    except Exception:
        pass
    return success


