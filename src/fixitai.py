"""FixItAI document ingestion and hybrid retrieval (all components run locally).

Includes:
- Linguistic query entity & error extraction
- Multi-format ingestion (TXT, Markdown, PDF, CSV, JSON)
- Overlapping sliding-window chunking
- Multi-modal retrieval: TF-IDF, Okapi BM25, and dense BERT
- State-of-the-art Reciprocal Rank Fusion (RRF)
- Device-portable execution across CUDA, Apple Silicon MPS, and CPU
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Keep model artifacts inside the project workspace
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / ".cache" / "huggingface"))

TECHS = [
    "Java", "Python", "SQL", "MySQL", "PostgreSQL", "Docker", "Linux",
    "Windows", "Git", "HTTP", "REST", "DNS", "networking", "pip", "Kubernetes"
]

ERROR_RE = re.compile(
    r"\b(?:\w*(?:Exception|Error|Failure)\w*|ECONNREFUSED|ORA-\d+|HTTP\s*\d{3}|\d{3}\s+\w+|NullPointerException)\b",
    re.I
)
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_./:#-]*|\d+(?:\.\d+)*")


def analyze_query(query: str) -> dict:
    """Extract normalized tokens, technology tags, and explicit error signatures."""
    q = html.unescape(query or "").strip()
    technologies = sorted({t for t in TECHS if re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", q, re.I)})
    errors = list(dict.fromkeys(m.group(0) for m in ERROR_RE.finditer(q)))
    terms = list(dict.fromkeys(TOKEN_RE.findall(q)))
    return {
        "normalized": re.sub(r"\s+", " ", q),
        "technologies": technologies,
        "errors": errors,
        "terms": terms
    }


def _read_upload(name: str, raw: bytes) -> list[dict]:
    suffix = Path(name).suffix.casefold()
    text = raw.decode("utf-8-sig", errors="replace")

    if suffix in {".txt", ".md", ".markdown"}:
        return [{"title": Path(name).stem, "text": text, "source": name}]

    if suffix == ".csv":
        rows = list(csv.DictReader(text.splitlines()))
        results = []
        for i, r in enumerate(rows, 1):
            title = r.get("title") or r.get("question") or f"{name} row {i}"
            body = "\n".join(
                str(v) for k, v in r.items()
                if k.casefold() not in {"title", "url", "source", "category", "technology", "technologies"}
            )
            results.append({
                "title": title,
                "text": body,
                "source": r.get("source") or name,
                "url": r.get("url", ""),
                "category": r.get("category", ""),
                "technologies": r.get("technologies") or r.get("technology", "")
            })
        return results

    if suffix == ".json":
        data = json.loads(text)
        data = data.get("documents", data) if isinstance(data, dict) else data
        if isinstance(data, dict):
            data = [data]
        return [{**d, "source": d.get("source") or name} for d in data if isinstance(d, dict)]

    if suffix == ".pdf":
        try:
            import io
            from pypdf import PdfReader
            pdf = PdfReader(io.BytesIO(raw))
            return [{
                "title": Path(name).stem,
                "text": "\n".join(p.extract_text() or "" for p in pdf.pages),
                "source": name
            }]
        except ImportError as exc:
            raise ValueError("PDF support requires pypdf. Install project requirements.") from exc

    raise ValueError("Supported formats: TXT, Markdown, PDF, CSV, JSON.")


def chunk_documents(documents: list[dict], chunk_size: int = 1100, overlap: int = 180) -> list[dict]:
    """Split documents into overlapping token windows with deterministic sha256 IDs."""
    chunks = []
    for d in documents:
        content = re.sub(r"\s+", " ", html.unescape(str(d.get("text", d.get("content", ""))))).strip()
        if not content:
            continue
        digest = hashlib.sha256(content.encode()).hexdigest()
        words = content.split()
        start = 0
        pos = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))
            body = " ".join(words[start:end])
            tech = d.get("technologies", [])
            if isinstance(tech, str):
                tech = [x.strip() for x in re.split(r"[,;|]", tech) if x.strip()]

            chunks.append({
                "document_id": digest[:20],
                "chunk_id": f"{digest[:12]}-{pos}",
                "title": str(d.get("title") or d.get("name") or "Untitled technical note"),
                "source": str(d.get("source") or d.get("url") or "uploaded document"),
                "url": str(d.get("url") or ""),
                "category": str(d.get("category") or "General"),
                "technologies": tech,
                "text": body,
                "position": pos,
                "document_hash": digest
            })
            if end == len(words):
                break
            start = end - overlap
            pos += 1
    return chunks


def reciprocal_rank_fusion(
    rankings: dict[str, np.ndarray],
    total_items: int,
    k: int = 60,
    weights: dict[str, float] | None = None
) -> np.ndarray:
    """Combine ranked retrieval lists using Reciprocal Rank Fusion (RRF).

    Formula: RRF(d) = SUM_m [ w_m / (k + rank_m(d)) ]
    """
    rrf_scores = np.zeros(total_items, dtype=float)
    weights = weights or {}

    for modality, scores in rankings.items():
        w = weights.get(modality, 1.0)
        # Order items descending by score
        order = np.argsort(scores)[::-1]
        for rank, idx in enumerate(order):
            if scores[idx] > 0:
                rrf_scores[idx] += w / (k + rank + 1)

    return rrf_scores


class HybridRetriever:
    """Multi-stage hybrid information retrieval engine with RRF and device auto-detection."""

    def __init__(self, chunks: list[dict], model_dir: str | Path | None = None):
        self.chunks = chunks
        self.model_dir = Path(model_dir) if model_dir else Path(__file__).resolve().parents[1] / "models" / "fixitai_bert_200k"
        self.vectorizer = None
        self.matrix = None
        self.embeddings = None
        self.encoder = None
        self.tokenizer = None
        self.device = None
        self.bm25 = None

        from sklearn.feature_extraction.text import TfidfVectorizer
        texts = [
            c["title"] + " " + " ".join(c.get("technologies", [])) + " " + c["text"]
            for c in chunks
        ]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            token_pattern=r"(?u)\b[\w./:#-]+\b"
        )
        if texts:
            try:
                self.matrix = self.vectorizer.fit_transform(texts)
            except ValueError:
                self.matrix = None

        try:
            from rank_bm25 import BM25Okapi
            tokenized_corpus = [TOKEN_RE.findall(t.casefold()) for t in texts]
            self.bm25 = BM25Okapi(tokenized_corpus) if texts else None
        except ImportError:
            self.bm25 = None

    def set_encoder(self, model, tokenizer, device, embeddings=None):
        self.encoder = model
        self.tokenizer = tokenizer
        self.device = device
        self.embeddings = embeddings

    def build_embeddings(self, device: str | None = None):
        from .bert_encoder import encode_fixitai_texts, get_device, load_fixitai_model
        import torch

        if device:
            dev = torch.device(device)
        else:
            dev = get_device()

        model, tokenizer, dev = load_fixitai_model(self.model_dir, dev)
        self.set_encoder(model, tokenizer, dev)
        texts = [
            c["title"] + ". " + " ".join(c.get("technologies", [])) + ". " + c["text"]
            for c in self.chunks
        ]
        self.embeddings = encode_fixitai_texts(model, tokenizer, texts, dev)
        return str(dev)

    def search(self, query: str, top_k: int = 5, technology: str | None = None, mode: str = "Hybrid"):
        from sklearn.metrics.pairwise import cosine_similarity

        q = analyze_query(query)
        n = len(self.chunks)
        if not n:
            return []

        tfidf = np.zeros(n)
        bm = np.zeros(n)
        sem = np.zeros(n)

        # 1. Sparse Lexical: TF-IDF
        if self.matrix is not None:
            qv = self.vectorizer.transform([q["normalized"]])
            tfidf = cosine_similarity(qv, self.matrix)[0]

        # 2. Probabilistic: Okapi BM25
        if self.bm25:
            tokens = TOKEN_RE.findall(q["normalized"].casefold())
            if tokens:
                raw_bm = np.asarray(self.bm25.get_scores(tokens), dtype=float)
                if raw_bm.max() > 0:
                    bm = raw_bm / raw_bm.max()

        # 3. Dense Semantic: Scratch BERT
        if self.embeddings is not None and len(self.embeddings) == n:
            if self.encoder is None:
                from .bert_encoder import get_device, load_fixitai_model
                self.encoder, self.tokenizer, self.device = load_fixitai_model(self.model_dir, get_device())
            from .bert_encoder import encode_fixitai_texts
            query_vector = encode_fixitai_texts(self.encoder, self.tokenizer, [q["normalized"]], self.device, batch_size=1)[0]
            sem = query_vector @ self.embeddings.T
            sem = np.maximum(sem, 0)

        # 4. Score Aggregation / Fusion
        has_semantic = self.embeddings is not None and len(self.embeddings) == n

        if mode == "Hybrid":
            # State-of-the-art Reciprocal Rank Fusion (RRF)
            rank_inputs = {"TF-IDF": tfidf, "BM25": bm}
            weights = {"TF-IDF": 1.0, "BM25": 1.2}
            if has_semantic:
                rank_inputs["Semantic"] = sem
                weights["Semantic"] = 1.5
            scores = reciprocal_rank_fusion(rank_inputs, n, k=60, weights=weights)
        elif mode == "Linear-Hybrid":
            scores = (0.25 * tfidf + 0.25 * bm + 0.5 * sem) if has_semantic else (0.5 * tfidf + 0.5 * bm)
        elif mode == "TF-IDF":
            scores = tfidf.copy()
        elif mode == "BM25":
            scores = bm.copy()
        elif mode == "Semantic":
            scores = sem.copy() if has_semantic else tfidf.copy()
        else:
            scores = (0.25 * tfidf + 0.25 * bm + 0.5 * sem) if has_semantic else (0.5 * tfidf + 0.5 * bm)

        # 5. Targeted Diagnostic Reranking (Error Signatures & Technology Tags)
        rerank = scores.copy()
        query_techs = [t.casefold() for t in q["technologies"]]

        for i, c in enumerate(self.chunks):
            title = c["title"].casefold()
            body = c["text"].casefold()
            tags = [str(t).casefold() for t in c.get("technologies", [])]

            # Bonus for matching technology tags
            if query_techs and any(t in tags or t in title for t in query_techs):
                rerank[i] += 0.04

            # Strong boost or penalty for exact runtime error tokens
            for err in q["errors"]:
                key = err.casefold()
                if key in title:
                    rerank[i] += 0.25
                elif key in body:
                    rerank[i] += 0.08
                else:
                    rerank[i] *= 0.85

        # 6. Candidate Filtering & Output Formatting
        results = []
        for i in np.argsort(rerank)[::-1]:
            c = self.chunks[int(i)]
            tags = [str(x).casefold() for x in c.get("technologies", [])]

            if technology and technology != "All" and technology.casefold() not in tags and technology.casefold() not in c["text"].casefold():
                continue

            # Minimum score threshold
            if rerank[i] < 0.005:
                continue

            hit_terms = [t for t in q["terms"] if len(t) > 1 and t.casefold() in c["text"].casefold()]
            results.append({
                **c,
                "score": float(rerank[i]),
                "base_score": float(scores[i]),
                "tfidf_score": float(tfidf[i]),
                "bm25_score": float(bm[i]),
                "semantic_score": float(sem[i]),
                "matched_terms": hit_terms,
                "retrieval_mode": mode
            })
            if len(results) >= top_k:
                break

        return results


def load_documents(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    docs = []
    for f in p.iterdir():
        if f.is_file() and f.suffix.casefold() in {".txt", ".md", ".markdown", ".pdf", ".csv", ".json"}:
            docs.extend(_read_upload(f.name, f.read_bytes()))
    return docs


def save_uploaded_documents(files, store_path: str | Path):
    """Persist indexed chunks in a portable JSON fallback store with hash deduplication."""
    p = Path(store_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    fresh = []
    for f in files:
        fresh.extend(chunk_documents(_read_upload(f.name, f.getvalue())))
    known = {c["chunk_id"] for c in old}
    merged = old + [c for c in fresh if c["chunk_id"] not in known]
    p.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    return merged, len(merged) - len(old)


def load_chunk_store(path: str | Path, seed_path: str | Path) -> list[dict]:
    p = Path(path)
    saved = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    combined = chunk_documents(load_documents(seed_path)) + saved
    seen = set()
    unique = []
    for c in combined:
        if c["chunk_id"] not in seen:
            unique.append(c)
            seen.add(c["chunk_id"])
    return unique

