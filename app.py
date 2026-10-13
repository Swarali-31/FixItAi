"""FixItAI: local, explainable technical solution retrieval."""
from pathlib import Path
import json, sys
import pandas as pd
import streamlit as st
import plotly.express as px
import torch
ROOT=Path(__file__).resolve().parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.fixitai import HybridRetriever, analyze_query, chunk_documents, load_chunk_store, save_uploaded_documents
from src.persistence import append_history, load_history, save_feedback
from src.scratch_encoder import encode_texts, load_scratch_model, require_cuda

SEED=ROOT/"data"/"knowledge_base"; STORE=ROOT/"data"/"fixitai_chunks.json"; HISTORY=ROOT/"data"/"query_history.json"
st.set_page_config(page_title="FixItAI | Technical Troubleshooting",page_icon="🛠️",layout="wide")
st.title("🛠️ FixItAI")
st.caption("Context-aware technical troubleshooting and explainable solution retrieval")

@st.cache_data(show_spinner=False)
def get_chunks(seed_stamp, store_stamp):return load_chunk_store(STORE,SEED)
@st.cache_resource(show_spinner=False)
def get_scratch_model(checkpoint_stamp):return load_scratch_model(ROOT/"models"/"fixitai")

trained_index=ROOT/"models"/"fixitai"
try:
    cuda=require_cuda()
    required=[trained_index/"model.pt",trained_index/"tokenizer.json",trained_index/"indexed_chunks.json",trained_index/"passage_embeddings.npz",trained_index/"index_metadata.json"]
    if not all(p.exists() for p in required):
        st.error("The scratch-trained FixItAI model is missing. Run `python train_fixitai.py` with archive.zip available before starting the app.")
        st.stop()
    model,tokenizer,device=get_scratch_model((trained_index/"model.pt").stat().st_mtime_ns)
    trained_chunks=json.loads((trained_index/"indexed_chunks.json").read_text(encoding="utf-8"))
    import numpy as np
    stored=np.load(trained_index/"passage_embeddings.npz",allow_pickle=False)
    if list(stored["chunk_ids"])!=[c["chunk_id"] for c in trained_chunks]:raise ValueError("The saved embedding index does not match its passage metadata; rerun train_fixitai.py.")
    current=get_chunks(SEED.stat().st_mtime_ns if SEED.exists() else 0,STORE.stat().st_mtime_ns if STORE.exists() else 0)
    known={c["chunk_id"] for c in trained_chunks};extra=[c for c in current if c["chunk_id"] not in known]
    chunks=trained_chunks+extra
    extra_vectors=encode_texts(model,tokenizer,[c["title"]+". "+" ".join(c.get("technologies",[]))+". "+c["text"] for c in extra],device) if extra else np.empty((0,stored["embeddings"].shape[1]),dtype="float32")
    vectors=np.vstack([stored["embeddings"],extra_vectors])
    retriever=HybridRetriever(chunks,model_dir=trained_index);retriever.set_scratch_model(model,tokenizer,device,vectors)
except Exception as e:st.error(f"GPU-only FixItAI startup failed: {e}");st.stop()
pages=["Ask FixItAI","Knowledge Base Explorer","Evaluation Dashboard","Query History & Feedback","About the Project"]
page=st.sidebar.radio("Navigate",pages)
st.sidebar.metric("Indexed passages",len(chunks));st.sidebar.caption(f"Scratch-trained Transformer on {torch.cuda.get_device_name(0)} · inference on CUDA only")

if page=="Ask FixItAI":
    st.subheader("Describe the problem")
    examples=["My Java application fails to connect to MySQL with a communications link failure.","Why am I getting a NullPointerException in Java?","How do I resolve a MySQL connection timeout?","Why does Docker say that a port is already allocated?","Why is pip unable to install a Python package?","How do I fix a Git merge conflict?","Why does my REST API return HTTP 500?","Why is my application unable to resolve a DNS hostname?"]
    selected=st.selectbox("Try an example",["Choose an example…"]+examples)
    q=st.text_area("Your technical question",value="" if selected=="Choose an example…" else selected,placeholder="Include the error message, technology, version, and what you already tried…",height=110)
    a,b,c=st.columns([2,1,1]);techs=sorted({t for d in chunks for t in d.get("technologies",[])})
    with a:technology=st.selectbox("Technology filter",["All"]+techs)
    with b:mode=st.selectbox("Retrieval mode",["Hybrid","TF-IDF","BM25","Semantic"])
    with c:top_k=st.slider("Results",1,10,5)
    analyze=st.checkbox("Show query analysis",True)
    if st.button("Find troubleshooting solutions",type="primary",disabled=not q.strip()):
        with st.spinner("Searching technical documentation…"):
            analysis=analyze_query(q);results=retriever.search(q,top_k,technology,mode)
        st.session_state["last_query"]=q;st.session_state["last_results"]=results;st.session_state["last_analysis"]=analysis
        if results:
            item={"query":q,"result_ids":[r["chunk_id"] for r in results],"feedback":None}
            append_history(HISTORY,item)
    if analyze and "last_analysis" in st.session_state:
        with st.expander("Query analysis",expanded=True):st.json(st.session_state["last_analysis"])
    results=st.session_state.get("last_results",[])
    if results:
        st.subheader("Retrieved evidence")
        st.info("Recommendations below summarize retrieved documents. Verify commands and settings against your environment. FixItAI does not invent a fix when the knowledge base has no relevant evidence.")
        if results[0]["score"]<0.2:st.warning("The top match is weak by this project's uncalibrated retrieval score. Treat these passages as leads and verify against current official documentation.")
        import re
        extractive=[s.strip() for s in re.split(r"(?<=[.!?])\s+",results[0]["text"]) if s.strip()][:3]
        st.markdown("### Evidence-based explanation")
        st.caption("Extracted directly from the highest-ranked passage; this is not a generated or independently verified diagnosis.")
        for sentence in extractive:st.write(f"- {sentence}")
        for i,r in enumerate(results,1):
            with st.expander(f"{i}. {r['title']} · relevance {r['score']:.3f}",expanded=i==1):
                links=(f" · answer {r['url']}" if r.get("url") else "")+(f" · question {r['question_url']}" if r.get("question_url") else "")
                st.caption(f"Source: {r['source']}"+links)
                st.write(r["text"])
                if r.get("attribution"):
                    st.caption(r["attribution"])
                st.write("**Why retrieved:** matched terms: "+(", ".join(r["matched_terms"]) or "overall lexical/semantic similarity"))
                st.caption(f"TF-IDF {r['tfidf_score']:.3f} · BM25 {r['bm25_score']:.3f} · semantic {r['semantic_score']:.3f}")
    elif "last_query" in st.session_state:st.warning("I couldn't find enough evidence in the indexed knowledge base. Add an authorized technical document or try a query with the exact error text and technology.")

elif page=="Knowledge Base Explorer":
    st.subheader("Knowledge Base Explorer")
    st.caption("Upload technical documents you own or are authorized to index. TXT, Markdown, PDF, CSV, and JSON are supported; duplicate content is skipped by stable content hash.")
    uploaded=st.file_uploader("Add documents",type=["txt","md","markdown","pdf","csv","json"],accept_multiple_files=True)
    if uploaded and st.button("Index uploaded documents",type="primary"):
        try:
            merged,count=save_uploaded_documents(uploaded,STORE);get_chunks.clear();st.success(f"Added {count} new passages. Reload the page to use the updated GPU index.")
        except Exception as e:st.error(str(e))
    frame=pd.DataFrame([{k:c.get(k) for k in ("title","source","category","technologies","position","text")} for c in chunks])
    if not frame.empty:
        filt=st.text_input("Filter by title, source, technology, or passage text")
        if filt:frame=frame[frame.astype(str).apply(lambda col:col.str.contains(filt,case=False,regex=False)).any(axis=1)]
        st.dataframe(frame.drop(columns=["text"]),use_container_width=True)
        n=st.selectbox("Inspect passage",frame.index.tolist(),format_func=lambda i:f"{frame.loc[i,'title']} · {frame.loc[i,'source']}")
        st.write(frame.loc[n,"text"])
        st.download_button("Download indexed chunks (JSON)",json.dumps(chunks,ensure_ascii=False,indent=2),"fixitai_chunks.json","application/json")
    if st.button("Rebuild local index cache"):
        get_chunks.clear();st.success("Index cache cleared; it will rebuild on the next rerun.")

elif page=="Evaluation Dashboard":
    st.subheader("Retrieval evaluation")
    p=ROOT/"outputs"/"fixitai_evaluation.json"
    if p.exists():
        metrics=json.loads(p.read_text(encoding="utf-8"));st.json(metrics)
        if metrics.get("per_query"):st.plotly_chart(px.bar(pd.DataFrame(metrics["per_query"]),x="query_id",y=[c for c in ("reciprocal_rank","recall_at_5","ndcg_at_5") if c in metrics["per_query"][0]],barmode="group"),use_container_width=True)
    else:
        st.info("No training evaluation exists yet. Run `python train_fixitai.py` to train on the supplied archive and create held-out Recall@1, Recall@5, and MRR metrics.")
    st.markdown("**Evaluation caution:** validation retrieves the highest-voted associated answer. The dataset lacks accepted-answer labels, so these metrics are against weak labels and do not measure verified technical correctness.")

elif page=="Query History & Feedback":
    st.subheader("Query history and feedback")
    hist=load_history(HISTORY)
    if hist:
        for i,row in enumerate(reversed(hist)):
            st.write(f"**{row['query']}** · feedback: {row.get('feedback') or 'not provided'}")
        idx=st.number_input("History item (newest is 0)",min_value=0,max_value=len(hist)-1,value=0)
        sentiment=st.radio("Was the retrieved evidence useful?",["Useful","Not useful"],horizontal=True)
        if st.button("Save feedback"):
            save_feedback(HISTORY,int(idx),sentiment);st.success("Feedback saved.")
    else:st.info("Your locally processed queries will appear here.")
    st.caption("History uses the local JSON fallback by default. Set FIXITAI_MONGODB_URI and optionally FIXITAI_MONGODB_DATABASE to persist query history and feedback in MongoDB.")

elif page=="About the Project":
    st.subheader("About FixItAI")
    st.markdown("""FixItAI retrieves technical troubleshooting passages. Its dual encoder is a 2-layer Transformer initialized randomly and trained locally on the supplied Stack Overflow Q&A archive. No pretrained model weights or hosted generation API are used. Training and dense inference require CUDA; the app stops if CUDA is unavailable. TF-IDF and BM25 are lexical baselines and run on CPU.

**Dataset quality:** the archive has technical posts, votes, parent-question links, and tags, making it relevant to solution retrieval. It has no accepted-answer field. Training uses the highest-voted answer with positive score as a weak target; held-out metrics measure retrieval of that noisy target and do not prove technical correctness.

Results preserve question/answer IDs and source links. Stack Overflow posts have CC BY-SA licensing; follow attribution and share-alike terms. The starter knowledge-base notes were authored locally. Retrieved community answers may be outdated, so check their source and version context.""")
