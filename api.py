"""Local FastAPI service for the FixItAI React application."""
from __future__ import annotations

import json
import threading
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
_BERT_MODEL_DIR = ROOT / "models" / "fixitai_bert_200k"
_BERT_ARTIFACTS_READY = all(
    (_BERT_MODEL_DIR / name).is_file()
    for name in ("bert_encoder/config.json", "tokenizer.json", "indexed_chunks.json", "passage_embeddings.npz")
)
MODEL_DIR = _BERT_MODEL_DIR if _BERT_ARTIFACTS_READY else ROOT / "models" / "fixitai"
BERT_MODEL_DIR = ROOT / "models" / "fixitai_bert_200k"
SEED = ROOT / "data" / "knowledge_base"
STORE = ROOT / "data" / "fixitai_chunks.json"
HISTORY = ROOT / "data" / "query_history.json"
FRONTEND = ROOT / "frontend" / "dist"
_runtime_lock = threading.Lock()


def active_model_dir() -> Path:
    """Switch to BERT only after its weights and complete vector index exist."""
    required = (
        BERT_MODEL_DIR / "bert_encoder" / "config.json",
        BERT_MODEL_DIR / "bert_encoder" / "model.safetensors",
        BERT_MODEL_DIR / "tokenizer.json",
        BERT_MODEL_DIR / "indexed_chunks.json",
        BERT_MODEL_DIR / "passage_embeddings.npz",
    )
    return BERT_MODEL_DIR if all(path.is_file() for path in required) else MODEL_DIR

app = FastAPI(title="FixItAI API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    technology: str = "All"
    mode: str = "Hybrid"
    top_k: int = Field(default=5, ge=1, le=10)


class FeedbackRequest(BaseModel):
    index_from_newest: int = Field(ge=0)
    feedback: str


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


@lru_cache(maxsize=1)
def _load_chunks(seed_stamp: int, store_stamp: int):
    from src.fixitai import load_chunk_store
    return load_chunk_store(STORE, SEED)


def load_chunks():
    return _load_chunks(
        SEED.stat().st_mtime_ns if SEED.exists() else 0,
        STORE.stat().st_mtime_ns if STORE.exists() else 0,
    )


@lru_cache(maxsize=1)
def _load_runtime(model_key: str, index_stamp: int, model_stamp: int, seed_stamp: int, store_stamp: int):
    import numpy as np
    from src.fixitai import HybridRetriever
    from src.bert_encoder import encode_fixitai_texts, load_fixitai_model
    from src.scratch_encoder import require_cuda

    device = require_cuda()
    model_dir = Path(model_key)
    chunks = read_json(model_dir / "indexed_chunks.json", None)
    if chunks is None:
        raise RuntimeError("The trained FixItAI passage index is missing.")
    with np.load(model_dir / "passage_embeddings.npz", allow_pickle=False) as archive:
        stored_ids = list(archive["chunk_ids"])
        base_embeddings = archive["embeddings"].copy()
    if stored_ids != [item["chunk_id"] for item in chunks]:
        raise RuntimeError("The passage index and embeddings do not match. Rebuild the selected model's passage index.")

    current = load_chunks()
    known = {item["chunk_id"] for item in chunks}
    extra = [item for item in current if item["chunk_id"] not in known]
    model, tokenizer, device = load_fixitai_model(model_dir, device)
    extra_vectors = (
        encode_fixitai_texts(
            model,
            tokenizer,
            [item["title"] + ". " + " ".join(item.get("technologies", [])) + ". " + item["text"] for item in extra],
            device,
        )
        if extra
        else np.empty((0, base_embeddings.shape[1]), dtype="float32")
    )
    all_chunks = chunks + extra
    all_vectors = np.vstack([base_embeddings, extra_vectors])
    retriever = HybridRetriever(all_chunks, model_dir=model_dir)
    retriever.set_scratch_model(model, tokenizer, device, all_vectors)
    return retriever, all_chunks, str(device)


def get_runtime():
    with _runtime_lock:
        model_dir = active_model_dir()
        index_path = model_dir / "indexed_chunks.json"
        model_path = model_dir / "bert_encoder" / "model.safetensors" if model_dir == BERT_MODEL_DIR else model_dir / "model.pt"
        return _load_runtime(
            str(model_dir),
            index_path.stat().st_mtime_ns if index_path.exists() else 0,
            model_path.stat().st_mtime_ns if model_path.exists() else 0,
            SEED.stat().st_mtime_ns if SEED.exists() else 0,
            STORE.stat().st_mtime_ns if STORE.exists() else 0,
        )


@app.get("/api/health")
def health():
    model_dir = active_model_dir()
    training = read_json(model_dir / "training_metadata.json", {})
    status = read_json(model_dir / "training_status.json", {})
    index = read_json(model_dir / "index_metadata.json", {})
    ready = all((model_dir / filename).is_file() for filename in ("tokenizer.json", "indexed_chunks.json", "passage_embeddings.npz"))
    ready = ready and ((model_dir / "bert_encoder" / "model.safetensors").is_file() if model_dir == BERT_MODEL_DIR else (model_dir / "model.pt").is_file())
    return {
        "status": "ready" if ready else "missing_model",
        "model": "random-init BERT encoder" if model_dir == BERT_MODEL_DIR else "scratch Transformer encoder",
        "training_status": status.get("status", "unknown"),
        "indexed_passages": int(index.get("passages", 0)),
        "gpu": training.get("gpu", "CUDA model"),
        "pretrained": bool(training.get("pretrained_weights", False)),
    }


@app.post("/api/search")
def search(payload: SearchRequest):
    if not payload.query.strip():
        raise HTTPException(status_code=422, detail="Enter a technical question to search.")
    try:
        retriever, _, device = get_runtime()
        from src.fixitai import analyze_query
        results = retriever.search(payload.query, payload.top_k, payload.technology, payload.mode)
        if results:
            from src.persistence import append_history
            append_history(HISTORY, {"query": payload.query, "result_ids": [item["chunk_id"] for item in results], "feedback": None})
        return {"results": results, "analysis": analyze_query(payload.query), "device": device}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"GPU search could not start: {exc}") from exc


@lru_cache(maxsize=1)
def _load_library(model_key: str, index_stamp: int, seed_stamp: int, store_stamp: int):
    indexed = read_json(Path(model_key) / "indexed_chunks.json", [])
    current = load_chunks()
    seen = {item.get("chunk_id") for item in indexed}
    return indexed + [item for item in current if item.get("chunk_id") not in seen]


def load_library_items():
    model_dir = active_model_dir()
    index_path = model_dir / "indexed_chunks.json"
    return _load_library(
        str(model_dir),
        index_path.stat().st_mtime_ns if index_path.exists() else 0,
        SEED.stat().st_mtime_ns if SEED.exists() else 0,
        STORE.stat().st_mtime_ns if STORE.exists() else 0,
    )

@app.get("/api/knowledge-base")
def knowledge_base(
    search_text: str = Query(default="", max_length=300),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=30, ge=1, le=100),
):
    items = load_library_items()
    needle = search_text.strip().casefold()
    if needle:
        items = [item for item in items if needle in " ".join((str(item.get("title", "")), str(item.get("source", "")), str(item.get("text", "")), " ".join(item.get("technologies", [])))).casefold()]
    total = len(items)
    visible = items[offset:offset + limit]
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": [{key: item.get(key) for key in ("chunk_id", "title", "source", "category", "technologies", "text", "url", "question_url", "attribution")} for item in visible],
    }


class UploadAdapter:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getvalue(self):
        return self._data


@app.post("/api/knowledge-base/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    from src.fixitai import save_uploaded_documents

    allowed = {".txt", ".md", ".markdown", ".pdf", ".csv", ".json"}
    uploads = []
    for upload in files:
        suffix = Path(upload.filename or "").suffix.casefold()
        if suffix not in allowed:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix or '(none)'}")
        uploads.append(UploadAdapter(Path(upload.filename or "upload").name, await upload.read()))
    try:
        _, added = save_uploaded_documents(uploads, STORE)
        _load_chunks.cache_clear()
        _load_library.cache_clear()
        _load_runtime.cache_clear()
        return {"added": added, "message": f"Added {added} new passages."}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/evaluation")
def evaluation():
    return read_json(ROOT / "outputs" / "fixitai_evaluation.json", {})


@app.get("/api/history")
def history():
    return {"items": list(reversed(read_json(HISTORY, [])))}


@app.post("/api/feedback")
def feedback(payload: FeedbackRequest):
    if payload.feedback not in {"Useful", "Not useful"}:
        raise HTTPException(status_code=400, detail="Feedback must be Useful or Not useful.")
    rows = read_json(HISTORY, [])
    if payload.index_from_newest >= len(rows):
        raise HTTPException(status_code=404, detail="That search is no longer in history.")
    from src.persistence import save_feedback
    save_feedback(HISTORY, payload.index_from_newest, payload.feedback)
    return {"ok": True}


@app.get("/{asset_path:path}", include_in_schema=False)
def frontend_app(asset_path: str):
    index = FRONTEND / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="React build missing. Run npm install and npm run build in frontend/.")
    candidate = (FRONTEND / asset_path).resolve()
    if candidate.is_file() and FRONTEND.resolve() in candidate.parents:
        return FileResponse(candidate)
    return FileResponse(index)

