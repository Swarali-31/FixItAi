"""FixItAI document ingestion and hybrid retrieval (all components run locally)."""
from __future__ import annotations
import csv, hashlib, html, json, re, os
from pathlib import Path
from typing import Any

# Keep model artifacts inside the project workspace, including on restricted desktops.
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1]/".cache"/"huggingface"))

TECHS = ["Java", "Python", "SQL", "MySQL", "PostgreSQL", "Docker", "Linux", "Windows", "Git", "HTTP", "REST", "DNS", "networking", "pip", "Kubernetes"]
ERROR_RE = re.compile(r"\b(?:\w*(?:Exception|Error|Failure)\w*|ECONNREFUSED|ORA-\d+|HTTP\s*\d{3}|\d{3}\s+\w+|NullPointerException)\b", re.I)
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_./:#-]*|\d+(?:\.\d+)*")

def analyze_query(query: str) -> dict:
    q = html.unescape(query or "").strip()
    technologies = sorted({t for t in TECHS if re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", q, re.I)})
    errors = list(dict.fromkeys(m.group(0) for m in ERROR_RE.finditer(q)))
    terms = list(dict.fromkeys(TOKEN_RE.findall(q)))
    return {"normalized": re.sub(r"\s+", " ", q), "technologies": technologies, "errors": errors, "terms": terms}

def _read_upload(name: str, raw: bytes) -> list[dict]:
    suffix = Path(name).suffix.casefold()
    text = raw.decode("utf-8-sig", errors="replace")
    if suffix in {".txt", ".md", ".markdown"}:
        return [{"title": Path(name).stem, "text": text, "source": name}]
    if suffix == ".csv":
        return [{"title": r.get("title") or r.get("question") or f"{name} row {i}", "text": "\n".join(str(v) for k,v in r.items() if k.casefold() not in {"title","url","source","category","technology","technologies"}), "source": r.get("source") or name, "url": r.get("url",""), "category": r.get("category",""), "technologies": r.get("technologies") or r.get("technology","")} for i,r in enumerate(csv.DictReader(text.splitlines()),1)]
    if suffix == ".json":
        data = json.loads(text); data = data.get("documents",data) if isinstance(data,dict) else data
        if isinstance(data,dict): data=[data]
        return [{**d,"source":d.get("source") or name} for d in data if isinstance(d,dict)]
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
            import io
            pdf=PdfReader(io.BytesIO(raw)); return [{"title":Path(name).stem,"text":"\n".join(p.extract_text() or "" for p in pdf.pages),"source":name}]
        except ImportError as exc: raise ValueError("PDF support requires pypdf. Install project requirements.") from exc
    raise ValueError("Supported formats: TXT, Markdown, PDF, CSV, JSON.")

def chunk_documents(documents: list[dict], chunk_size: int=1100, overlap: int=180) -> list[dict]:
    chunks=[]
    for d in documents:
        content=re.sub(r"\s+"," ",html.unescape(str(d.get("text",d.get("content",""))))).strip()
        if not content: continue
        digest=hashlib.sha256(content.encode()).hexdigest()
        words=content.split(); start=0; pos=0
        while start<len(words):
            end=min(start+chunk_size,len(words)); body=" ".join(words[start:end])
            tech=d.get("technologies",[])
            if isinstance(tech,str): tech=[x.strip() for x in re.split(r"[,;|]",tech) if x.strip()]
            chunks.append({"document_id":digest[:20],"chunk_id":f"{digest[:12]}-{pos}","title":str(d.get("title") or d.get("name") or "Untitled technical note"),"source":str(d.get("source") or d.get("url") or "uploaded document"),"url":str(d.get("url") or ""),"category":str(d.get("category") or "General"),"technologies":tech,"text":body,"position":pos,"document_hash":digest})
            if end==len(words): break
            start=end-overlap; pos+=1
    return chunks

class HybridRetriever:
    def __init__(self, chunks: list[dict], model_dir: str|Path|None=None):
        self.chunks=chunks; self.model_dir=Path(model_dir) if model_dir else Path(__file__).resolve().parents[1]/"models"/"fixitai"; self.vectorizer=None; self.matrix=None; self.embeddings=None; self.encoder=None; self.tokenizer=None; self.device=None
        from sklearn.feature_extraction.text import TfidfVectorizer
        texts=[c["title"]+" "+" ".join(c.get("technologies",[]))+" "+c["text"] for c in chunks]
        self.vectorizer=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,token_pattern=r"(?u)\b[\w./:#-]+\b")
        if texts:
            try: self.matrix=self.vectorizer.fit_transform(texts)
            except ValueError: self.matrix=None
        try:
            from rank_bm25 import BM25Okapi
            self.bm25=BM25Okapi([TOKEN_RE.findall(t.casefold()) for t in texts]) if texts else None
        except ImportError: self.bm25=None

    def set_encoder(self,model,tokenizer,device,embeddings=None):
        if getattr(device,"type",None)!="cuda":raise RuntimeError("FixItAI semantic model must run on CUDA.")
        self.encoder=model;self.tokenizer=tokenizer;self.device=device;self.embeddings=embeddings

    def build_embeddings(self, device: str|None=None):
        from .bert_encoder import encode_fixitai_texts,load_fixitai_model,require_cuda
        import numpy as np
        cuda=require_cuda()
        if device and str(device).casefold() not in {"cuda","cuda:0"}:raise RuntimeError("FixItAI embeddings are CUDA-only.")
        model,tokenizer,cuda=load_fixitai_model(self.model_dir,cuda)
        self.set_encoder(model,tokenizer,cuda)
        texts=[c["title"]+". "+" ".join(c.get("technologies",[]))+". "+c["text"] for c in self.chunks]
        self.embeddings=encode_fixitai_texts(model,tokenizer,texts,cuda)
        return str(cuda)

    def search(self, query: str, top_k=5, technology: str|None=None, mode="Hybrid"):
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        q=analyze_query(query); n=len(self.chunks)
        if not n:return []
        tfidf=np.zeros(n); bm=np.zeros(n); sem=np.zeros(n)
        if self.matrix is not None:
            qv=self.vectorizer.transform([q["normalized"]]); tfidf=cosine_similarity(qv,self.matrix)[0]
            if self.bm25:
                bm=np.asarray(self.bm25.get_scores(TOKEN_RE.findall(q["normalized"].casefold())),dtype=float)
                if bm.max()>0:bm/=bm.max()
        if self.embeddings is not None:
            if self.encoder is None:
                from .bert_encoder import load_fixitai_model
                self.encoder,self.tokenizer,self.device=load_fixitai_model(self.model_dir)
            from .bert_encoder import encode_fixitai_texts
            query_vector=encode_fixitai_texts(self.encoder,self.tokenizer,[q["normalized"]],self.device,batch_size=1)[0]
            sem=query_vector @ self.embeddings.T
            sem=np.maximum(sem,0)
        use_sem=self.embeddings is not None
        scores={"TF-IDF":tfidf,"BM25":bm,"Semantic":sem,"Hybrid":(0.25*tfidf+0.25*bm+0.5*sem if use_sem else 0.5*tfidf+0.5*bm)}.get(mode)
        if scores is None:scores=(0.25*tfidf+0.25*bm+0.5*sem if use_sem else 0.5*tfidf+0.5*bm)
        # Preserve explicit error strings and technology tags as strong retrieval cues.
        rerank=scores.copy()
        query_techs=[t.casefold() for t in q["technologies"]]
        for i,c in enumerate(self.chunks):
            title=c["title"].casefold(); body=c["text"].casefold(); tags=[str(t).casefold() for t in c.get("technologies",[])]
            if query_techs and any(t in tags or t in title for t in query_techs):rerank[i]+=0.04
            for err in q["errors"]:
                key=err.casefold()
                if key in title:rerank[i]+=0.25
                elif key in body:rerank[i]+=0.08
                else:rerank[i]*=0.8
        results=[]
        for i in np.argsort(rerank)[::-1]:
            c=self.chunks[int(i)]; tags=[str(x).casefold() for x in c.get("technologies",[])]
            if technology and technology!="All" and technology.casefold() not in tags and technology.casefold() not in c["text"].casefold():continue
            if rerank[i]<0.10:continue
            hit_terms=[t for t in q["terms"] if len(t)>1 and t.casefold() in c["text"].casefold()]
            results.append({**c,"score":float(rerank[i]),"base_score":float(scores[i]),"tfidf_score":float(tfidf[i]),"bm25_score":float(bm[i]),"semantic_score":float(sem[i]),"matched_terms":hit_terms})
            if len(results)>=top_k:break
        return results

def load_documents(path: str|Path) -> list[dict]:
    p=Path(path)
    if not p.exists():return []
    docs=[]
    for f in p.iterdir():
        if f.is_file() and f.suffix.casefold() in {".txt",".md",".markdown",".pdf",".csv",".json"}:
            docs.extend(_read_upload(f.name,f.read_bytes()))
    return docs

def save_uploaded_documents(files, store_path: str|Path):
    """Persist indexed chunks in a portable JSON fallback store with hash deduplication."""
    p=Path(store_path); p.parent.mkdir(parents=True,exist_ok=True)
    old=json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    fresh=[]
    for f in files:fresh.extend(chunk_documents(_read_upload(f.name,f.getvalue())))
    known={c["chunk_id"] for c in old}; merged=old+[c for c in fresh if c["chunk_id"] not in known]
    p.write_text(json.dumps(merged,ensure_ascii=False,indent=2),encoding="utf-8")
    return merged,len(merged)-len(old)

def load_chunk_store(path: str|Path, seed_path: str|Path) -> list[dict]:
    p=Path(path)
    saved=json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    combined=chunk_documents(load_documents(seed_path))+saved; seen=set(); unique=[]
    for c in combined:
        if c["chunk_id"] not in seen:unique.append(c);seen.add(c["chunk_id"])
    return unique
