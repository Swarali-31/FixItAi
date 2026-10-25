"""Shared Stack Overflow parsing and retrieval evaluation for BERT training."""
from __future__ import annotations

import csv
import html
import random
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)

    def handle_starttag(self, tag, attrs):
        if tag in {"p", "br", "pre", "li", "tr", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"p", "pre", "li", "tr", "h1", "h2", "h3"}:
            self.parts.append("\n")


def clean_html(value: str, max_chars=7000) -> str:
    parser = TextExtractor()
    try:
        parser.feed(value or "")
    except Exception:
        pass
    text = html.unescape(" ".join(parser.parts))
    return re.sub(r"\s+", " ", text).strip()[:max_chars]


def _entry(archive, name_hint):
    for name in archive.namelist():
        if Path(name).name.casefold() == name_hint.casefold():
            return name
    raise ValueError(f"ZIP is missing required file {name_hint}.")


def load_pairs(zip_path: Path, max_questions: int, min_question_score: int,
               min_answer_score: int, seed: int):
    """Stream Stack Exchange CSVs from ZIP and select the highest-voted answer."""
    rng = random.Random(seed)
    with zipfile.ZipFile(zip_path) as archive:
        qname = _entry(archive, "Questions.csv")
        aname = _entry(archive, "Answers.csv")
        tname = _entry(archive, "Tags.csv")
        reservoir, seen, rows_seen = [], 0, 0
        with archive.open(qname) as raw:
            rows = csv.DictReader(__import__("io").TextIOWrapper(
                raw, encoding="utf-8-sig", errors="replace", newline=""))
            for row in rows:
                rows_seen += 1
                if rows_seen % 500000 == 0:
                    print(f"Scanned {rows_seen:,} question rows; {seen:,} meet the score filter.", flush=True)
                try:
                    score = int(row.get("Score", 0) or 0)
                except ValueError:
                    score = 0
                title, qid = (row.get("Title") or "").strip(), (row.get("Id") or "").strip()
                if not qid or not title or score < min_question_score:
                    continue
                item = {"id": qid, "owner": row.get("OwnerUserId") or "", "score": score,
                        "_title": title[:1000], "_body": (row.get("Body", "") or "")[:3000]}
                seen += 1
                if len(reservoir) < max_questions:
                    reservoir.append(item)
                else:
                    j = rng.randrange(seen)
                    if j < max_questions:
                        reservoir[j] = item
        for question in reservoir:
            question["title"] = clean_html(question.pop("_title"), 400)
            question["body"] = clean_html(question.pop("_body"))
        selected = {q["id"]: q for q in reservoir}
        print(f"Question scan: {seen:,} met score >= {min_question_score}; sampled {len(selected):,}.", flush=True)
        tags = {qid: [] for qid in selected}
        with archive.open(tname) as raw:
            rows = csv.DictReader(__import__("io").TextIOWrapper(
                raw, encoding="utf-8-sig", errors="replace", newline=""))
            for row in rows:
                qid = str(row.get("Id", ""))
                if qid in tags and row.get("Tag"):
                    tags[qid].append(row["Tag"].strip("<> "))
        best, answers_seen = {}, 0
        with archive.open(aname) as raw:
            rows = csv.DictReader(__import__("io").TextIOWrapper(
                raw, encoding="utf-8-sig", errors="replace", newline=""))
            for row in rows:
                answers_seen += 1
                if answers_seen % 1000000 == 0:
                    print(f"Scanned {answers_seen:,} answer rows; matched {len(best):,} sampled questions so far.", flush=True)
                parent = str(row.get("ParentId", ""))
                if parent not in selected:
                    continue
                try:
                    score = int(row.get("Score", 0) or 0)
                except ValueError:
                    score = 0
                if score < min_answer_score:
                    continue
                if parent not in best or score > best[parent]["answer_score"]:
                    body = clean_html(row.get("Body", ""))
                    if len(body) >= 40:
                        best[parent] = {"answer_id": str(row.get("Id", "")),
                                        "answer_owner": row.get("OwnerUserId") or "",
                                        "answer_score": score, "answer": body[:3000]}
    pairs = []
    for qid, question in selected.items():
        answer = best.get(qid)
        if not answer:
            continue
        tag_list = sorted(set(t for t in tags[qid] if t))
        query = (question["title"] + " " + " ".join(tag_list) + " " + question["body"]).strip()
        if len(query) < 25:
            continue
        pairs.append({**question, **answer, "tags": tag_list, "query": query,
                      "url": f"https://stackoverflow.com/questions/{qid}"})
    print(f"Matched {len(pairs):,} questions to an answer with score >= {min_answer_score}; selected the highest-scored match.", flush=True)
    if not pairs:
        raise ValueError("No question/answer pairs matched. Check that this ZIP contains the expected Stack Overflow CSVs.")
    quality = {"source_zip": zip_path.name, "question_candidates": seen,
               "sampled_questions": len(selected), "matched_pairs": len(pairs),
               "question_min_score": min_question_score, "answer_min_score": min_answer_score,
               "label_method": "highest answer vote score among answers sharing ParentId; weak relevance label, not accepted-answer ground truth"}
    return pairs, quality


def make_tokenizer(pairs, train_indices, vocab_size, max_length, path):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors, normalizers
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=special,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())

    def corpus():
        for i in train_indices:
            yield pairs[i]["query"]
            yield pairs[i]["answer"]

    tokenizer.train_from_iterator(corpus(), trainer=trainer, length=2 * len(train_indices))
    cls_id, sep_id = tokenizer.token_to_id("[CLS]"), tokenizer.token_to_id("[SEP]")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)])
    tokenizer.enable_truncation(max_length=max_length)
    tokenizer.enable_padding(length=max_length, pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]")
    tokenizer.save(str(path))
    return tokenizer


def batch_encode(tokenizer, texts, device):
    import torch
    encodings = tokenizer.encode_batch(texts)
    ids = torch.tensor([e.ids for e in encodings], dtype=torch.long, device=device)
    mask = torch.tensor([e.attention_mask for e in encodings], dtype=torch.bool, device=device)
    return ids, mask


def evaluate(model, tokenizer, pairs, indices, device, batch_size=32, max_eval=2500, candidate_chunk=8192):
    import torch
    indices = indices[:max_eval]
    query_vectors, answer_batches = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            part = [pairs[i] for i in indices[start:start + batch_size]]
            qi, qm = batch_encode(tokenizer, [p["query"] for p in part], device)
            with torch.autocast("cuda", dtype=torch.float16):
                query_vectors.append(model(qi, qm))
        for start in range(0, len(pairs), batch_size):
            part = pairs[start:start + batch_size]
            ai, am = batch_encode(tokenizer, [p["answer"] for p in part], device)
            with torch.autocast("cuda", dtype=torch.float16):
                answer_batches.append(model(ai, am))
        queries, answers = torch.cat(query_vectors).float(), torch.cat(answer_batches).float()
        gold = torch.tensor(indices, device=device, dtype=torch.long)
        gold_scores = (queries * answers.index_select(0, gold)).sum(dim=1)
        ranks = torch.ones(len(indices), device=device, dtype=torch.long)
        for start in range(0, len(pairs), candidate_chunk):
            scores = queries @ answers[start:start + candidate_chunk].T
            ranks += (scores > gold_scores.unsqueeze(1)).sum(dim=1)
        return {"queries": len(indices), "candidate_answers": len(pairs),
                "recall_at_1": float((ranks <= 1).float().mean()),
                "recall_at_5": float((ranks <= 5).float().mean()),
                "mrr": float((1 / ranks.float()).mean())}
