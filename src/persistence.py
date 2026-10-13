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

def load_history(path: str|Path):
    col=_collection("query_history")
    if col is not None:
        try:return [{"query":d.get("query",""),"result_ids":d.get("result_ids",[]),"feedback":d.get("feedback")} for d in col.find({}, {"_id":0}).sort("created_at",1).limit(200)]
        except Exception:pass
    p=Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []

def append_history(path: str|Path, item: dict):
    col=_collection("query_history")
    if col is not None:
        try:
            from datetime import datetime, timezone
            col.insert_one({**item,"created_at":datetime.now(timezone.utc)});return
        except Exception:pass
    p=Path(path);old=json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    old.append(item);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(old[-200:],indent=2),encoding="utf-8")

def save_feedback(path: str|Path, index_from_newest: int, feedback: str):
    col=_collection("query_history")
    if col is not None:
        try:
            doc=col.find().sort("created_at",-1).skip(index_from_newest).limit(1)
            item=next(doc,None)
            if item:col.update_one({"_id":item["_id"]},{"$set":{"feedback":feedback}});return
        except Exception:pass
    p=Path(path);rows=json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    if 0<=index_from_newest<len(rows):rows[-1-index_from_newest]["feedback"]=feedback;p.write_text(json.dumps(rows,indent=2),encoding="utf-8")
