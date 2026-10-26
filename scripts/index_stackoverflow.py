"""Extract and index high-quality Stack Overflow technical Q&A pairs into FixItAI's chunk store."""
from __future__ import annotations

import csv
import html
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA_DIR = ROOT / "data"
ARCHIVE_DIR = DATA_DIR / "archive"
OUTPUT_FILE = DATA_DIR / "fixitai_chunks.json"


def clean_text(raw: str, max_chars: int = 2500) -> str:
    if not raw:
        return ""
    text = html.unescape(raw)
    # Strip HTML tags
    text = re.sub(r"<pre><code>([\s\S]*?)</code></pre>", r"\n```\n\1\n```\n", text)
    text = re.sub(r"<code>([^<]+)</code>", r"`\1`", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()[:max_chars]


def build_stackoverflow_chunks(max_items: int = 100000, min_q_score: int = 4):
    t0 = time.time()
    print("=" * 65)
    print(f"FixItAI Stack Overflow Dataset Indexer ({max_items:,} Target Items)")
    print("=" * 65)

    q_file = ARCHIVE_DIR / "Questions.csv"
    a_file = ARCHIVE_DIR / "Answers.csv"
    t_file = ARCHIVE_DIR / "Tags.csv"

    if not q_file.is_file() or not a_file.is_file() or not t_file.is_file():
        print(f"Error: Required CSV files not found in {ARCHIVE_DIR}")
        sys.exit(1)

    # 1. Scan Questions
    print(f"[1/5] Scanning Questions.csv (filtering score >= {min_q_score})...")
    questions = {}
    with open(q_file, encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                score = int(row.get("Score", 0) or 0)
            except ValueError:
                score = 0
            qid = str(row.get("Id", "")).strip()
            title = (row.get("Title") or "").strip()
            if not qid or not title or score < min_q_score:
                continue

            clean_title = re.sub(r"<[^>]+>", " ", html.unescape(title)).strip()
            clean_body = clean_text(row.get("Body", ""), max_chars=800)
            questions[qid] = {
                "id": qid,
                "title": clean_title,
                "body": clean_body,
                "score": score
            }
            if len(questions) >= int(max_items * 1.4):
                break

    print(f"  -> Extracted {len(questions):,} high-quality technical questions in {round(time.time() - t0, 1)}s")

    # 2. Scan Tags
    t1 = time.time()
    print("[2/5] Mapping technologies from Tags.csv...")
    tags_map = {qid: [] for qid in questions}
    with open(t_file, encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = str(row.get("Id", "")).strip()
            tag = (row.get("Tag") or "").strip().lower()
            if qid in tags_map and tag and tag not in tags_map[qid]:
                tags_map[qid].append(tag)

    print(f"  -> Mapped tags across questions in {round(time.time() - t1, 1)}s")

    # 3. Scan Answers
    t2 = time.time()
    print("[3/5] Finding highest-voted technical answers in Answers.csv...")
    best_answers = {}
    scanned = 0
    with open(a_file, encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scanned += 1
            parent = str(row.get("ParentId", "")).strip()
            if parent not in questions:
                continue
            try:
                score = int(row.get("Score", 0) or 0)
            except ValueError:
                score = 0
            aid = str(row.get("Id", "")).strip()
            body = clean_text(row.get("Body", ""), max_chars=3000)
            if len(body) < 50:
                continue

            if parent not in best_answers or score > best_answers[parent]["score"]:
                best_answers[parent] = {
                    "aid": aid,
                    "score": score,
                    "body": body
                }

    print(f"  -> Matched {len(best_answers):,} top answers from {scanned:,} rows in {round(time.time() - t2, 1)}s")

    # 4. Assemble FixItAI Chunks
    t3 = time.time()
    print("[4/5] Constructing FixItAI chunk store with verified Stack Overflow URLs...")
    so_chunks = []
    sorted_qids = sorted(best_answers.keys(), key=lambda q: questions[q]["score"] + best_answers[q]["score"], reverse=True)

    for qid in sorted_qids[:max_items]:
        q = questions[qid]
        a = best_answers[qid]
        tags = tags_map.get(qid, [])

        chunk = {
            "document_id": f"stackoverflow-{qid}",
            "chunk_id": f"stackoverflow-answer-{a['aid']}",
            "title": q["title"],
            "source": f"Stack Overflow question #{qid} · answer #{a['aid']}",
            "url": f"https://stackoverflow.com/a/{a['aid']}",
            "question_url": f"https://stackoverflow.com/questions/{qid}",
            "category": "Community technical Q&A",
            "technologies": tags,
            "text": a["body"],
            "position": 0,
            "question_score": q["score"],
            "answer_score": a["score"],
            "document_hash": f"so-{qid}-{a['aid']}"
        }
        so_chunks.append(chunk)

    # Load local 22 starter markdown knowledge base chunks
    from src.fixitai import load_chunk_store
    seed_chunks = load_chunk_store(Path("/nonexistent"), ROOT / "data" / "knowledge_base")
    print(f"  -> Combined {len(seed_chunks)} starter curated guides with {len(so_chunks):,} Stack Overflow Q&As.")
    all_chunks = seed_chunks + so_chunks

    OUTPUT_FILE.write_text(json.dumps(so_chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    indexed_json_file = ROOT / "models" / "fixitai_bert_200k" / "indexed_chunks.json"
    indexed_json_file.write_text(json.dumps(all_chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  -> Saved {len(all_chunks):,} total chunks to {indexed_json_file} ({round(indexed_json_file.stat().st_size / (1024*1024), 2)} MB).")

    # 5. Precompute BERT Embeddings
    print("[5/5] Precomputing BERT embeddings with Apple Silicon MPS...")
    import numpy as np
    from src.bert_encoder import load_fixitai_model, encode_fixitai_texts, get_device

    device = get_device()
    model_dir = ROOT / "models" / "fixitai_bert_200k"
    model, tokenizer, device = load_fixitai_model(model_dir, device)

    texts = [c["title"] + ". " + " ".join(c.get("technologies", [])) + ". " + c["text"] for c in all_chunks]
    
    t_embed_start = time.time()
    batch_size = 256
    sub_chunk_size = 10000
    all_vectors = []
    for offset in range(0, len(texts), sub_chunk_size):
        sub_texts = texts[offset:offset + sub_chunk_size]
        sub_vecs = encode_fixitai_texts(model, tokenizer, sub_texts, device, batch_size=batch_size)
        all_vectors.append(sub_vecs)
        pct = min(100.0, round((offset + len(sub_texts)) / len(texts) * 100, 1))
        rate = round((offset + len(sub_texts)) / max(0.1, time.time() - t_embed_start), 1)
        print(f"  -> Encoded {offset + len(sub_texts):,}/{len(texts):,} passages ({pct}%) at {rate:,} items/sec on {device}...")
    embeddings = np.vstack(all_vectors)
    t_embed_elapsed = time.time() - t_embed_start
    print(f"  -> Finished encoding {len(embeddings):,} embeddings in {round(t_embed_elapsed, 1)}s ({round(len(embeddings) / max(0.1, t_embed_elapsed), 1)} passages/sec) on {device}!")

    embed_npz_path = model_dir / "passage_embeddings.npz"
    np.savez_compressed(
        embed_npz_path,
        chunk_ids=np.array([c["chunk_id"] for c in all_chunks]),
        embeddings=embeddings.astype(np.float32),
    )
    print(f"  -> Saved embeddings to {embed_npz_path} ({round(embed_npz_path.stat().st_size / (1024*1024), 2)} MB).")
    print(f"Total indexing and embedding pipeline completed in {round(time.time() - t0, 1)}s!")
    print("=" * 65)


if __name__ == "__main__":
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    build_stackoverflow_chunks(max_items=target, min_q_score=4)
