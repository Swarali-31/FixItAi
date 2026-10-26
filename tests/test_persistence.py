"""Unit tests for SQLite and JSON persistence layer."""
import tempfile
from pathlib import Path
from src.persistence import append_history, load_history, save_feedback, clear_history


def test_sqlite_persistence_lifecycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "query_history.json"

        # 1. Append
        item = {
            "query": "Java NullPointerException in Controller",
            "result_ids": ["doc-1", "doc-2"],
            "feedback": None
        }
        append_history(json_path, item)

        # 2. Load
        records = load_history(json_path)
        assert len(records) == 1
        assert records[0]["query"] == "Java NullPointerException in Controller"

        # 3. Save feedback
        save_feedback(json_path, 0, "Useful")
        records_after_fb = load_history(json_path)
        assert records_after_fb[0]["feedback"] == "Useful"

        # 4. Clear
        clear_history(json_path)
        records_cleared = load_history(json_path)
        assert len(records_cleared) == 0
