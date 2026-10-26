"""Local FastAPI service for the FixItAI React application."""
from __future__ import annotations

import json
import re
import threading
import time
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
BERT_MODEL_DIR = ROOT / "models" / "fixitai_bert_200k"
SEED = ROOT / "data" / "knowledge_base"
STORE = ROOT / "data" / "fixitai_chunks.json"
HISTORY = ROOT / "data" / "query_history.json"
FRONTEND = ROOT / "frontend" / "dist"
_runtime_lock = threading.Lock()


def active_model_dir() -> Path:
    """FixItAI uses its BERT encoder only; missing artifacts must fail visibly."""
    return BERT_MODEL_DIR

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
    wizard_option: dict | None = None
    selected_options: dict[str, str] = {}


class WizardRefineRequest(BaseModel):
    query: str
    question_id: str = ""
    option_label: str = ""
    refinement: str = ""
    selected_options: dict[str, str] = {}
    target_lang: str = "en"


class FeedbackRequest(BaseModel):
    index_from_newest: int = Field(ge=0)
    feedback: str


class NLPAnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


class SummarizeRequest(BaseModel):
    query: str
    passage: str
    title: str = ""


class ParseLogRequest(BaseModel):
    raw_text: str = Field(min_length=1, max_length=20000)


class TranslateRequest(BaseModel):
    text: str = Field(default="", max_length=5000)
    query: str = ""
    root_cause: str = ""
    action_steps: list[str] = []
    code_snippet: str | None = None
    target_lang: str = "hi"



def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


import time


def is_valid_lfs_target(path: Path) -> bool:
    """Check if file exists, is non-empty, and is not an un-pulled Git LFS pointer text file."""
    if not path.is_file():
        return False
    if path.stat().st_size < 500:
        try:
            head = path.read_text(encoding="utf-8", errors="ignore")[:60]
            if "git-lfs.github.com" in head:
                return False
        except Exception:
            return False
    return True


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
    from src.bert_encoder import encode_fixitai_texts, load_fixitai_model, get_device

    device = get_device()
    model_dir = Path(model_key)
    index_file = model_dir / "indexed_chunks.json"
    embed_file = model_dir / "passage_embeddings.npz"

    # Check if full 200k BERT index is downloaded and valid (not a Git LFS pointer)
    if is_valid_lfs_target(index_file) and is_valid_lfs_target(embed_file):
        chunks = read_json(index_file, None)
        if chunks is None:
            chunks = load_chunks()
            retriever = HybridRetriever(chunks, model_dir=model_dir)
            return retriever, chunks, str(device)

        with np.load(embed_file, allow_pickle=False) as archive:
            stored_ids = list(archive["chunk_ids"])
            base_embeddings = archive["embeddings"].copy()

        current = load_chunks()
        known = {item["chunk_id"] for item in chunks}
        extra = [item for item in current if item["chunk_id"] not in known]

        if (model_dir / "bert_encoder" / "model.safetensors").is_file():
            try:
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
                retriever.set_encoder(model, tokenizer, device, all_vectors)
                return retriever, all_chunks, str(device)
            except Exception:
                pass

        all_chunks = chunks + extra
        retriever = HybridRetriever(all_chunks, model_dir=model_dir)
        return retriever, all_chunks, str(device)

    # Fallback to local starter knowledge base + uploads if LFS data is not pulled
    current = load_chunks()
    retriever = HybridRetriever(current, model_dir=model_dir)

    # If BERT encoder weights are available, encode starter passages
    if (model_dir / "bert_encoder" / "model.safetensors").is_file() and (model_dir / "tokenizer.json").is_file():
        try:
            model, tokenizer, device = load_fixitai_model(model_dir, device)
            texts = [c["title"] + ". " + " ".join(c.get("technologies", [])) + ". " + c["text"] for c in current]
            vectors = encode_fixitai_texts(model, tokenizer, texts, device)
            retriever.set_encoder(model, tokenizer, device, vectors)
        except Exception:
            pass

    return retriever, current, str(device)


def get_runtime():
    with _runtime_lock:
        model_dir = active_model_dir()
        index_path = model_dir / "indexed_chunks.json"
        model_path = model_dir / "bert_encoder" / "model.safetensors"
        return _load_runtime(
            str(model_dir),
            index_path.stat().st_mtime_ns if index_path.exists() else 0,
            model_path.stat().st_mtime_ns if model_path.exists() else 0,
            SEED.stat().st_mtime_ns if SEED.exists() else 0,
            STORE.stat().st_mtime_ns if STORE.exists() else 0,
        )


@app.get("/api/health")
def health():
    import torch
    from src.bert_encoder import get_device

    device = get_device()
    model_dir = active_model_dir()
    training = read_json(model_dir / "training_metadata.json", {})
    status = read_json(model_dir / "training_status.json", {})
    index = read_json(model_dir / "index_metadata.json", {})

    full_lfs_ready = is_valid_lfs_target(model_dir / "indexed_chunks.json") and is_valid_lfs_target(model_dir / "passage_embeddings.npz")
    items = load_library_items()

    if torch.cuda.is_available():
        gpu_name = f"CUDA: {torch.cuda.get_device_name(0)}"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        gpu_name = "Apple Silicon MPS (Metal Performance Shaders)"
    else:
        gpu_name = "CPU Accelerator (Vector Optimized)"

    return {
        "status": "ready" if len(items) > 0 else "missing_model",
        "index_type": "full_200k_bert" if full_lfs_ready else "local_knowledge_base",
        "model": "BERT Transformer encoder (trained from scratch)",
        "training_status": status.get("status", "ready"),
        "indexed_passages": len(items),
        "device": str(device),
        "gpu": gpu_name,
        "pretrained": bool(training.get("pretrained_weights", False)),
    }


@app.post("/api/parse-log")
def parse_log(payload: ParseLogRequest):
    """Endpoint for stack trace and terminal crash log parsing."""
    from src.nlp_pipeline import get_nlp_pipeline
    return get_nlp_pipeline().parse_stack_trace(payload.raw_text)


@app.post("/api/search")
def search(payload: SearchRequest):
    if not payload.query.strip():
        raise HTTPException(status_code=422, detail="Enter a technical question to search.")
    t0 = time.perf_counter()
    try:
        retriever, _, device = get_runtime()
        from src.nlp_pipeline import get_nlp_pipeline
        from src.summarizer import get_summarizer
        from src.translator import translate_indic_query_to_english

        nlp = get_nlp_pipeline()

        # 0. Cross-lingual Indic input translation
        indic_meta = translate_indic_query_to_english(payload.query)
        effective_query = indic_meta["translated_query"] if indic_meta["was_translated"] else payload.query

        wizard_opt = payload.wizard_option
        search_query = effective_query
        if wizard_opt:
            refinement = wizard_opt.get("refinement") or wizard_opt.get("label", "")
            search_query = f"{effective_query} {refinement}".strip()

        # 1. Domain Guardrail Check
        guardrail = nlp.check_domain_guardrail(effective_query)

        # 2. NLP pipeline linguistic analysis (Unit 1, 2, 4)
        nlp_analysis = nlp.analyze_linguistics(effective_query)

        # 3. Typo auto-correction check & WordNet Synset Expansion (Unit 1 & 2 / Lab 2)
        typo_info = nlp.suggest_typo_correction(effective_query)
        wordnet_info = nlp.expand_query_wordnet(effective_query)
        perplexity = nlp.calculate_perplexity(effective_query)

        # 4. Information Retrieval search across vectorization schemes (with RRF)
        results = retriever.search(search_query, payload.top_k, payload.technology, payload.mode)

        # 5. Pretrained Transformer Text Summarization or Contextual Reasoning
        ai_solution = None
        top_match = results[0] if results else None
        top_score = top_match["score"] if top_match else 0.0

        all_active_opts = dict(payload.selected_options) if payload.selected_options else {}
        if wizard_opt:
            all_active_opts[wizard_opt.get("id", "opt")] = wizard_opt.get("label", "")

        if all_active_opts or wizard_opt:
            # User selected Guided Wizard scenario(s)
            ref_text = (wizard_opt.get("refinement") or wizard_opt.get("label", "")) if wizard_opt else ""
            comb_ref = " ".join([ref_text] + list(all_active_opts.values()))
            ref_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", comb_ref.lower()))
            ref_tokens -= {"with", "from", "this", "that", "your", "what", "which", "using", "into", "need", "environment", "application"}

            is_kb_match = False
            if top_match and top_score >= 0.35:
                match_text = (top_match.get("title", "") + " " + top_match.get("text", "")).lower()
                overlap = [t for t in ref_tokens if t in match_text]
                if len(overlap) >= 1:
                    is_kb_match = True

            if is_kb_match and top_match:
                ai_solution = get_summarizer().summarize(search_query, top_match.get("text", ""), top_match.get("title", ""))
                ai_solution["source_type"] = "knowledge_base"
                ai_solution["source_title"] = top_match.get("title", "")
                ai_solution["scenario_title"] = " + ".join(all_active_opts.values()) or "Knowledge Base Evidence"
                ai_solution["model_used"] = f"Knowledge Base Extractive Engine: '{top_match.get('title')}' (Sentence-BERT)"
            else:
                ai_solution = nlp.resolve_compound_wizard_options(payload.query, all_active_opts)
                ai_solution["source_type"] = "synthesized_inference"
                ai_solution["kb_notice"] = "No direct passage in local knowledge base. Synthesized from FixItAI Diagnostic Reasoning Engine."
        elif results:
            try:
                from src.persistence import append_history
                append_history(HISTORY, {"query": payload.query, "result_ids": [item["chunk_id"] for item in results], "feedback": None})
            except Exception:
                pass
            ai_solution = get_summarizer().summarize(effective_query, top_match.get("text", ""), top_match.get("title", ""))
            ai_solution["source_type"] = "knowledge_base"
            ai_solution["source_title"] = top_match.get("title", "")

        # 6. Interactive Diagnostic Wizard Questions (dynamically generated from query & KB results)
        wizard_questions = nlp.generate_diagnostic_wizard(effective_query, top_match, results)

        # 7. Confidence Score & Meter
        confidence_pct = min(100, int(top_score * 100))
        if confidence_pct >= 65:
            confidence_level = "HIGH"
        elif confidence_pct >= 35:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "SPECULATIVE"

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        return {
            "results": results,
            "analysis": nlp_analysis,
            "ai_solution": ai_solution,
            "device": device,
            "latency_ms": latency_ms,
            "retrieval_mode": payload.mode,
            "guardrail": guardrail,
            "typo_correction": typo_info,
            "wordnet_expansion": wordnet_info,
            "perplexity": perplexity,
            "confidence": {
                "score": confidence_pct,
                "level": confidence_level
            },
            "wizard_questions": wizard_questions,
            "indic_meta": indic_meta
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Search could not complete: {exc}") from exc


@app.post("/api/wizard-refine")
def wizard_refine(payload: WizardRefineRequest):
    """Endpoint for interactive Guided Wizard diagnostic refinement.
    Fetches matching knowledge base evidence using refined terms;
    if not in knowledge base, responds accordingly with synthesized diagnostic reasoning.
    """
    retriever, _, device = get_runtime()
    from src.nlp_pipeline import get_nlp_pipeline
    from src.summarizer import get_summarizer
    from src.translator import get_translator

    nlp = get_nlp_pipeline()

    all_opts = dict(payload.selected_options) if payload.selected_options else {}
    if payload.question_id and payload.option_label:
        all_opts[payload.question_id] = payload.option_label

    refinements = []
    if payload.refinement:
        refinements.append(payload.refinement)
    for opt_lbl in all_opts.values():
        if opt_lbl not in refinements:
            refinements.append(opt_lbl)
    combined_refinement = " ".join(refinements)

    search_query = f"{payload.query} {combined_refinement}".strip()

    # 1. Fetch from hybrid retriever / knowledge base
    results = retriever.search(search_query, top_k=5, technology="All", mode="Hybrid")
    top_match = results[0] if results else None
    top_score = top_match["score"] if top_match else 0.0

    # 2. Check if Knowledge Base contains genuine relevant evidence for the refined scenario
    ref_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", combined_refinement.lower()))
    ref_tokens -= {"with", "from", "this", "that", "your", "what", "which", "using", "into", "need", "environment", "application"}

    is_kb_match = False
    if top_match and top_score >= 0.35:
        match_text = (top_match.get("title", "") + " " + top_match.get("text", "")).lower()
        overlap = [t for t in ref_tokens if t in match_text]
        if len(overlap) >= 1:
            is_kb_match = True

    if is_kb_match and top_match:
        solution = get_summarizer().summarize(search_query, top_match.get("text", ""), top_match.get("title", ""))
        solution["source_type"] = "knowledge_base"
        solution["source_title"] = top_match.get("title", "")
        solution["scenario_title"] = " + ".join(all_opts.values()) or payload.option_label
        solution["model_used"] = f"Knowledge Base Extractive Engine: '{top_match.get('title')}' (Sentence-BERT)"
    else:
        # Not in knowledge base: respond accordingly with expert synthesized diagnostic reasoning
        solution = nlp.resolve_compound_wizard_options(payload.query, all_opts)
        solution["source_type"] = "synthesized_inference"
        solution["kb_notice"] = "No direct passage in local knowledge base. Synthesized from FixItAI Diagnostic Reasoning Engine."

    # 3. Regional multi-lingual translation if target_lang is not English
    translated = None
    if payload.target_lang and payload.target_lang != "en":
        translator = get_translator()
        translated = translator.explain_diagnostic(
            query=search_query,
            root_cause=solution["root_cause"],
            action_steps=solution["action_steps"],
            code_snippet=solution.get("code_snippet"),
            target_lang=payload.target_lang,
            scenario_key=solution.get("scenario_key")
        )

    return {
        "solution": solution,
        "translated_solution": translated,
        "results": results,
        "selected_question_id": payload.question_id,
        "selected_option_label": payload.option_label,
        "selected_options": all_opts,
        "search_query": search_query
    }


@app.post("/api/nlp-analyze")
def nlp_analyze(payload: NLPAnalyzeRequest):
    """Endpoint for interactive NLP linguistic analysis (Unit 1, 2, 4)."""
    from src.nlp_pipeline import get_nlp_pipeline
    return get_nlp_pipeline().analyze_linguistics(payload.text)


@app.post("/api/summarize")
def summarize_solution(payload: SummarizeRequest):
    """Endpoint for pre-trained BERT text summarization (Unit 5 / Lab 6)."""
    from src.summarizer import get_summarizer
    return get_summarizer().summarize(payload.query, payload.passage, payload.title)


@app.post("/api/translate")
def translate_text(payload: TranslateRequest):
    """Endpoint for multi-lingual Indic neural/template diagnostic explanation."""
    from src.translator import get_translator
    translator = get_translator()
    if payload.root_cause or payload.action_steps or payload.query:
        return translator.explain_diagnostic(
            query=payload.query,
            root_cause=payload.root_cause or payload.text,
            action_steps=payload.action_steps,
            code_snippet=payload.code_snippet,
            target_lang=payload.target_lang
        )
    return translator.translate(payload.text)



@app.get("/api/benchmarks")
def get_benchmarks():
    """Endpoint for comprehensive IR vectorization evaluation table (Unit 2, 3, 5)."""
    from src.vector_engine import get_curated_benchmarks
    return {"benchmarks": get_curated_benchmarks()}


@lru_cache(maxsize=1)
def _load_library(model_key: str, index_stamp: int, seed_stamp: int, store_stamp: int):
    index_file = Path(model_key) / "indexed_chunks.json"
    indexed = read_json(index_file, []) if is_valid_lfs_target(index_file) else []
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
    from src.persistence import load_history
    return {"items": list(reversed(load_history(HISTORY)))}


@app.post("/api/history/clear")
def clear_all_history():
    from src.persistence import clear_history
    clear_history(HISTORY)
    return {"ok": True, "message": "Query history cleared."}


@app.post("/api/feedback")
def feedback(payload: FeedbackRequest):
    if payload.feedback not in {"Useful", "Not useful"}:
        raise HTTPException(status_code=400, detail="Feedback must be Useful or Not useful.")
    from src.persistence import load_history, save_feedback
    rows = load_history(HISTORY)
    if payload.index_from_newest >= len(rows):
        raise HTTPException(status_code=404, detail="That search is no longer in history.")
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

