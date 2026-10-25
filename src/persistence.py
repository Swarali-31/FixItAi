"""Small MongoDB adapter with local JSON fallback for FixItAI history."""
import json, os
from pathlib import Path

def _collection(name):
    uri=os.getenv("FIXITAI_MONGODB_URI")
    if not uri:return None
    try:
        from pymongo import MongoClient
        client=MongoClient(uri,serverSelectionTimeoutMS=800)
        client.admin.command("ping")
        return client[os.getenv("FIXITAI_MONGODB_DATABASE","fixitai")][name]
    except Exception:return None

def _safe_read_json(p: Path, default=None):
    if default is None:
        default = []
    if not p.exists():
        return default
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else default
    except Exception:
        return default

def load_history(path: str|Path):
    col=_collection("query_history")
    if col is not None:
        try:return [{"query":d.get("query",""),"result_ids":d.get("result_ids",[]),"feedback":d.get("feedback")} for d in col.find({}, {"_id":0}).sort("created_at",1).limit(200)]
        except Exception:pass
    return _safe_read_json(Path(path), [])

def append_history(path: str|Path, item: dict):
    col=_collection("query_history")
    if col is not None:
        try:
            from datetime import datetime, timezone
            col.insert_one({**item,"created_at":datetime.now(timezone.utc)});return
        except Exception:pass
    try:
        p=Path(path)
        old=_safe_read_json(p, [])
        old.append(item)
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(old[-200:],indent=2),encoding="utf-8")
    except Exception:
        pass

def save_feedback(path: str|Path, index_from_newest: int, feedback: str):
    col=_collection("query_history")
    if col is not None:
        try:
            doc=col.find().sort("created_at",-1).skip(index_from_newest).limit(1)
            item=next(doc,None)
            if item:col.update_one({"_id":item["_id"]},{"$set":{"feedback":feedback}});return
        except Exception:pass
    try:
        p=Path(path)
        rows=_safe_read_json(p, [])
        if 0<=index_from_newest<len(rows):
            rows[-1-index_from_newest]["feedback"]=feedback
            p.write_text(json.dumps(rows,indent=2),encoding="utf-8")
    except Exception:
        pass

