"""Create FixItAI's indexed answer passages from the completed local BERT run."""
from __future__ import annotations

import json, sys, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    from src.bert_encoder import encode_texts, load_bert_model, require_cuda
    from src.fixitai import chunk_documents, load_documents
    from src.training_data import clean_html, load_pairs

    model_dir = ROOT / "models" / "fixitai_bert_200k"
    status_path = model_dir / "training_status.json"
    if not status_path.is_file():
        raise SystemExit(f"Training status not found: {status_path}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "completed":
        raise SystemExit(f"BERT training is not complete (status={status.get('status')}); refusing to index an incomplete run.")
    device = require_cuda()
    pairs, quality = load_pairs(ROOT / "data" / "archive.zip", 200000, -2147483648, -2147483648, 42)
    documents = []
    for p in pairs:
        documents.append({
            "document_id": f"stackoverflow-{p['id']}",
            "chunk_id": f"stackoverflow-answer-{p['answer_id']}",
            "title": p["title"],
            "source": f"Stack Overflow question #{p['id']} · answer #{p['answer_id']}",
            "url": f"https://stackoverflow.com/a/{p['answer_id']}",
            "question_url": p["url"],
            "category": "Community technical Q&A",
            "technologies": p["tags"],
            "text": p["answer"], "position": 0,
            "question_score": p["score"], "answer_score": p["answer_score"],
            "question_owner_id": p["owner"], "answer_owner_id": p["answer_owner"],
            "attribution": f"Stack Overflow post {p['id']} (question) and {p['answer_id']} (answer); user IDs {p['owner'] or 'unavailable'} and {p['answer_owner'] or 'unavailable'}; CC BY-SA; source links above.",
            "document_hash": f"{p['id']}:{p['answer_id']}",
        })
    documents.extend(chunk_documents(load_documents(ROOT / "data" / "knowledge_base")))
    print(f"Indexing {len(documents):,} passages with the trained CUDA BERT encoder.", flush=True)
    model, tokenizer, device = load_bert_model(model_dir, device)
    vectors = []
    batch_size = 16
    for start in range(0, len(documents), batch_size):
        batch = documents[start:start + batch_size]
        texts = [d["title"] + ". " + " ".join(d.get("technologies", [])) + ". " + d["text"] for d in batch]
        vectors.append(encode_texts(model, tokenizer, texts, device, batch_size=batch_size))
        if (start // batch_size + 1) % 500 == 0:
            print(f"Embedded {min(start + batch_size, len(documents)):,}/{len(documents):,} passages", flush=True)
    embeddings = np.concatenate(vectors).astype(np.float32, copy=False)
    ids = np.asarray([d["chunk_id"] for d in documents])
    np.savez_compressed(model_dir / "passage_embeddings.npz", embeddings=embeddings, chunk_ids=ids)
    index_path = model_dir / "indexed_chunks.json"
    with index_path.open("w", encoding="utf-8") as handle:
        encoder = json.JSONEncoder(ensure_ascii=False, separators=(",", ":"))
        handle.write("[")
        for idx, doc in enumerate(documents):
            if idx: handle.write(",")
            handle.writelines(encoder.iterencode(doc))
        handle.write("]")
    index_metadata = {"model": "FixItAI BERT encoder initialized from random weights", "model_dir": str(model_dir), "device": "cuda", "gpu": __import__("torch").cuda.get_device_name(device), "passages": len(documents), "embedding_dimension": int(embeddings.shape[1]), "pretrained": False}
    (model_dir / "index_metadata.json").write_text(json.dumps(index_metadata, indent=2), encoding="utf-8")
    quality.update({"indexed_passages": len(documents), "weak_supervision": True})
    (model_dir / "data_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    print(f"Saved BERT passage index ({len(documents):,} passages) and CUDA embeddings.", flush=True)


if __name__ == "__main__": main()
