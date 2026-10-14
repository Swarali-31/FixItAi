"""FixItAI: a polished local workspace for technical troubleshooting."""
from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fixitai import HybridRetriever, analyze_query, chunk_documents, load_chunk_store, save_uploaded_documents
from src.persistence import append_history, load_history, save_feedback
from src.scratch_encoder import encode_texts, load_scratch_model, require_cuda

SEED = ROOT / "data" / "knowledge_base"
STORE = ROOT / "data" / "fixitai_chunks.json"
HISTORY = ROOT / "data" / "query_history.json"
MODEL_DIR = ROOT / "models" / "fixitai"
PAGES = [
    ("Ask FixItAI", "⌕", "ask"),
    ("Knowledge Base", "▤", "knowledge"),
    ("Evaluation", "◫", "evaluation"),
    ("History & Feedback", "◷", "history"),
    ("About FixItAI", "ⓘ", "about"),
]

st.set_page_config(
    page_title="FixItAI | Technical Support Workspace",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink:#14243b; --muted:#62748a; --line:#e4ebf3; --blue:#2864dc; --mint:#dbf6ed; }
    .stApp { background:linear-gradient(180deg,#f6f9fd 0,#f8fafc 360px,#f8fafc 100%); color:var(--ink); }
    [data-testid="stSidebar"] { background:#fff; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] > div:first-child { padding-top:1.25rem; }
    [data-testid="stSidebar"] div.stButton > button { min-height:2.9rem; border-radius:12px; border-color:transparent; text-align:left; font-weight:600; }
    [data-testid="stSidebar"] div.stButton > button:hover { border-color:#cbd9ef; background:#f3f7ff; color:#174fb9; }
    [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] { background:#eaf1ff; color:#174fb9; border:1px solid #d7e4ff; }
    .block-container { max-width:1440px; padding-top:1.4rem; padding-bottom:3rem; }
    .brand { display:flex; align-items:center; gap:12px; padding:0 8px 18px; border-bottom:1px solid var(--line); margin-bottom:18px; }
    .brand-mark { width:42px; height:42px; border-radius:14px; display:grid; place-items:center; color:#fff; font-size:21px; background:linear-gradient(145deg,#4385f5,#2454ce); box-shadow:0 8px 18px #2864dc30; }
    .brand-title { color:var(--ink); font-size:19px; font-weight:800; letter-spacing:-.5px; line-height:1.1; }
    .brand-sub { color:var(--muted); font-size:11px; margin-top:4px; }
    .workspace-label { color:#8492a5; font-size:10px; font-weight:800; letter-spacing:1.3px; text-transform:uppercase; padding:4px 8px 9px; }
    .topline { display:flex; justify-content:space-between; align-items:center; margin:0 0 22px; }
    .crumb { color:#6d7e93; font-size:12px; font-weight:700; letter-spacing:.7px; text-transform:uppercase; }
    .gpu-pill { display:inline-flex; align-items:center; gap:7px; color:#187251; background:#e7f7ef; padding:7px 11px; border:1px solid #ceefdf; border-radius:99px; font-size:11px; font-weight:700; }
    .gpu-dot { width:7px; height:7px; border-radius:50%; background:#23a26d; display:inline-block; }
    .page-head { margin:0 0 23px; }
    .eyebrow { color:#2864dc; font-size:11px; font-weight:800; letter-spacing:1.4px; text-transform:uppercase; margin-bottom:7px; }
    .page-title { color:#14243b; font-size:32px; line-height:1.15; letter-spacing:-1px; font-weight:800; margin:0; }
    .page-description { color:#687b91; font-size:14px; margin-top:8px; max-width:820px; }
    .hero { background:linear-gradient(115deg,#173975 0%,#2864dc 58%,#5e9bf5 100%); color:white; padding:27px 30px; border-radius:22px; margin:0 0 20px; box-shadow:0 16px 36px #1c4c9924; }
    .hero-kicker { color:#c9dcff; font-size:11px; font-weight:800; letter-spacing:1.2px; text-transform:uppercase; }
    .hero-title { color:white; font-size:29px; font-weight:800; letter-spacing:-.8px; margin:7px 0 5px; }
    .hero-copy { color:#e5efff; font-size:14px; max-width:690px; line-height:1.55; }
    .section-title { color:#192c45; font-size:19px; font-weight:750; letter-spacing:-.3px; margin:20px 0 4px; }
    .section-copy { color:#718198; font-size:13px; margin-bottom:13px; }
    div[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); padding:15px 17px; border-radius:16px; box-shadow:0 4px 14px #19395b08; }
    div[data-testid="stMetricLabel"] p { color:#75859a; font-weight:650; }
    div[data-testid="stMetricValue"] { color:#1b3150; }
    div[data-testid="stForm"] { background:#fff; border:1px solid var(--line); padding:18px; border-radius:18px; }
    div[data-testid="stVerticalBlockBorderWrapper"] { border-radius:17px; }
    div.stButton > button, div.stDownloadButton > button { border-radius:11px; min-height:2.7rem; font-weight:650; }
    .result-score { color:#287453; background:#e7f7ef; border:1px solid #d5efe1; border-radius:99px; display:inline-block; padding:4px 9px; font-size:11px; font-weight:750; }
    .footer-note { color:#8a98aa; font-size:11px; border-top:1px solid var(--line); padding-top:15px; margin-top:30px; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">🛠</div><div><div class="brand-title">FixItAI</div><div class="brand-sub">Technical support workspace</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="workspace-label">Workspace</div>', unsafe_allow_html=True)
    if "active_page" not in st.session_state:
        st.session_state.active_page = PAGES[0][0]
    for label, icon, slug in PAGES:
        if st.button(
            f"{icon}    {label}",
            key=f"nav_{slug}",
            type="primary" if st.session_state.active_page == label else "secondary",
            use_container_width=True,
        ):
            st.session_state.active_page = label
    st.divider()
    st.markdown("**System status**")
    st.caption("CUDA model · local retrieval")
    st.caption("Your questions stay in this workspace unless MongoDB is configured.")


@st.cache_data(show_spinner=False)
def get_chunks(seed_stamp: int, store_stamp: int):
    return load_chunk_store(STORE, SEED)


@st.cache_resource(show_spinner=False)
def get_scratch_model(checkpoint_stamp: int):
    return load_scratch_model(MODEL_DIR)


def page_heading(eyebrow: str, title: str, description: str):
    st.markdown(
        f'<div class="page-head"><div class="eyebrow">{eyebrow}</div><h1 class="page-title">{title}</h1><div class="page-description">{description}</div></div>',
        unsafe_allow_html=True,
    )


def section_heading(title: str, description: str | None = None):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if description:
        st.markdown(f'<div class="section-copy">{description}</div>', unsafe_allow_html=True)


try:
    cuda = require_cuda()
    required = [
        MODEL_DIR / "model.pt",
        MODEL_DIR / "tokenizer.json",
        MODEL_DIR / "indexed_chunks.json",
        MODEL_DIR / "passage_embeddings.npz",
        MODEL_DIR / "index_metadata.json",
    ]
    if not all(path.exists() for path in required):
        st.error("The scratch-trained FixItAI model is missing. Run `python train_fixitai.py` with `data/archive.zip` available before starting the app.")
        st.stop()

    model, tokenizer, device = get_scratch_model((MODEL_DIR / "model.pt").stat().st_mtime_ns)
    trained_chunks = json.loads((MODEL_DIR / "indexed_chunks.json").read_text(encoding="utf-8"))
    import numpy as np

    with np.load(MODEL_DIR / "passage_embeddings.npz", allow_pickle=False) as stored:
        stored_ids = list(stored["chunk_ids"])
        base_embeddings = stored["embeddings"].copy()
    if stored_ids != [chunk["chunk_id"] for chunk in trained_chunks]:
        raise ValueError("The saved embedding index does not match its passage metadata; rerun train_fixitai.py.")

    current = get_chunks(
        SEED.stat().st_mtime_ns if SEED.exists() else 0,
        STORE.stat().st_mtime_ns if STORE.exists() else 0,
    )
    known = {chunk["chunk_id"] for chunk in trained_chunks}
    extra = [chunk for chunk in current if chunk["chunk_id"] not in known]
    extra_vectors = (
        encode_texts(
            model,
            tokenizer,
            [chunk["title"] + ". " + " ".join(chunk.get("technologies", [])) + ". " + chunk["text"] for chunk in extra],
            device,
        )
        if extra
        else np.empty((0, base_embeddings.shape[1]), dtype="float32")
    )
    chunks = trained_chunks + extra
    vectors = np.vstack([base_embeddings, extra_vectors])
    retriever = HybridRetriever(chunks, model_dir=MODEL_DIR)
    retriever.set_scratch_model(model, tokenizer, device, vectors)
except Exception as exc:
    st.error(f"GPU-only FixItAI startup failed: {exc}")
    st.stop()

page = st.session_state.active_page
gpu_name = html.escape(torch.cuda.get_device_name(0))
st.markdown(
    f'<div class="topline"><div class="crumb">FixItAI <span style="color:#aab6c5">/</span> {html.escape(page)}</div><div class="gpu-pill"><span class="gpu-dot"></span> CUDA active · {gpu_name}</div></div>',
    unsafe_allow_html=True,
)

if page == "Ask FixItAI":
    st.markdown(
        '<div class="hero"><div class="hero-kicker">Technical troubleshooting</div><div class="hero-title">Find a useful next step.</div><div class="hero-copy">Describe the error and the technology involved. FixItAI searches its technical knowledge base and shows the source passages behind each result.</div></div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    with cols[0]:
        st.metric("Indexed passages", f"{len(chunks):,}")
    with cols[1]:
        st.metric("Search methods", "3", "TF-IDF · BM25 · Transformer")
    with cols[2]:
        st.metric("Model runtime", "CUDA", "Scratch-trained")

    section_heading("Ask a technical question", "Include the exact error, language or tool, and what you have already tried.")
    examples = [
        "My Java application fails to connect to MySQL with a communications link failure.",
        "Why am I getting a NullPointerException in Java?",
        "How do I resolve a MySQL connection timeout?",
        "Why does Docker say that a port is already allocated?",
        "Why is pip unable to install a Python package?",
        "How do I fix a Git merge conflict?",
        "Why does my REST API return HTTP 500?",
        "Why is my application unable to resolve a DNS hostname?",
    ]
    selected = st.selectbox("Start with an example (optional)", ["Choose an example…"] + examples)
    default_question = "" if selected == "Choose an example…" else selected
    question = st.text_area(
        "Your question",
        value=default_question,
        placeholder="Example: Python 3.12 raises ModuleNotFoundError after I install the package…",
        height=115,
        label_visibility="collapsed",
    )
    controls = st.columns([2, 1, 1, 1])
    technologies = sorted({tag for doc in chunks for tag in doc.get("technologies", [])})
    with controls[0]:
        technology = st.selectbox("Technology", ["All"] + technologies)
    with controls[1]:
        mode = st.selectbox("Search mode", ["Hybrid", "Semantic", "BM25", "TF-IDF"])
    with controls[2]:
        top_k = st.selectbox("Show results", [3, 5, 8, 10], index=1)
    with controls[3]:
        show_analysis = st.toggle("Query details", value=False)

    search_col, hint_col = st.columns([1, 3])
    with search_col:
        search_clicked = st.button("⌕  Search solutions", type="primary", use_container_width=True, disabled=not question.strip())
    with hint_col:
        st.caption("Results include links, matching details, and community attribution where available.")

    if search_clicked:
        with st.spinner("Searching the knowledge base…"):
            analysis = analyze_query(question)
            results = retriever.search(question, top_k, technology, mode)
        st.session_state["last_query"] = question
        st.session_state["last_results"] = results
        st.session_state["last_analysis"] = analysis
        if results:
            append_history(HISTORY, {"query": question, "result_ids": [result["chunk_id"] for result in results], "feedback": None})

    if show_analysis and "last_analysis" in st.session_state:
        with st.expander("Query signals", expanded=True):
            st.json(st.session_state["last_analysis"])

    results = st.session_state.get("last_results", [])
    if results:
        section_heading("Recommended evidence", f"Showing {len(results)} relevant passages for: {st.session_state.get('last_query', '')}")
        st.info("These are retrieved community and project notes, not generated or independently verified diagnoses. Check commands against your environment.")
        if results[0]["score"] < 0.2:
            st.warning("The top match is weak by this project's uncalibrated retrieval score. Treat it as a lead and verify against current official documentation.")
        first_sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", results[0]["text"]) if sentence.strip()][:3]
        with st.container(border=True):
            st.markdown("**Top match summary**")
            st.caption("Extracted from the highest-ranked passage")
            for sentence in first_sentences:
                st.write(f"• {sentence}")
        for index, result in enumerate(results, 1):
            with st.expander(f"{index}. {result['title']}  ·  relevance {result['score']:.3f}", expanded=index == 1):
                st.caption(f"Source: {result['source']}")
                st.write(result["text"])
                if result.get("url") or result.get("question_url"):
                    links = st.columns(2)
                    if result.get("url"):
                        links[0].link_button("Open answer source", result["url"])
                    if result.get("question_url"):
                        links[1].link_button("Open question source", result["question_url"])
                if result.get("attribution"):
                    st.caption(result["attribution"])
                st.markdown("**Why this matched**")
                st.write(", ".join(result["matched_terms"]) or "Overall lexical and semantic similarity")
                score_cols = st.columns(3)
                score_cols[0].caption(f"TF-IDF · {result['tfidf_score']:.3f}")
                score_cols[1].caption(f"BM25 · {result['bm25_score']:.3f}")
                score_cols[2].caption(f"Transformer · {result['semantic_score']:.3f}")
    elif "last_query" in st.session_state:
        st.warning("No strong passage matched that question. Try the exact error text or add a technical document in the Knowledge Base page.")

elif page == "Knowledge Base":
    page_heading("Your technical library", "Knowledge Base", "Browse the indexed passages and add technical documents you own or are authorized to use.")
    cols = st.columns(3)
    with cols[0]:
        st.metric("Total passages", f"{len(chunks):,}")
    with cols[1]:
        st.metric("Starter notes", f"{sum(1 for chunk in chunks if not str(chunk.get('source', '')).startswith('Stack Overflow')):,}")
    with cols[2]:
        st.metric("Supported formats", "6", "TXT · MD · PDF · CSV · JSON")

    with st.expander("＋  Add your documents", expanded=False):
        st.caption("Uploads are saved locally in the project index. Duplicate passages are skipped by content hash.")
        uploaded = st.file_uploader("Select files", type=["txt", "md", "markdown", "pdf", "csv", "json"], accept_multiple_files=True)
        if uploaded and st.button("Index selected documents", type="primary"):
            try:
                _, added_count = save_uploaded_documents(uploaded, STORE)
                get_chunks.clear()
                st.session_state["upload_notice"] = f"Added {added_count} new passages. They are now available in search."
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    upload_notice = st.session_state.pop("upload_notice", None)
    if upload_notice:
        st.success(upload_notice)

    section_heading("Explore passages", "Filter the collection, then select a row to inspect its full text.")
    frame = pd.DataFrame([{key: chunk.get(key) for key in ("title", "source", "category", "technologies", "position", "text")} for chunk in chunks])
    if not frame.empty:
        filt = st.text_input("Filter by title, source, technology, or passage text", placeholder="Try Java, Docker, or a specific error…")
        if filt:
            frame = frame[frame.astype(str).apply(lambda column: column.str.contains(filt, case=False, regex=False)).any(axis=1)]
        st.dataframe(frame.drop(columns=["text"]), use_container_width=True, hide_index=True)
        if not frame.empty:
            selected_row = st.selectbox("Inspect a passage", frame.index.tolist(), format_func=lambda i: f"{frame.loc[i, 'title']} · {frame.loc[i, 'source']}")
            with st.container(border=True):
                st.markdown(f"**{frame.loc[selected_row, 'title']}**")
                st.caption(str(frame.loc[selected_row, "source"]))
                st.write(frame.loc[selected_row, "text"])
        else:
            st.info("No passages match that filter.")
        st.download_button("Download passage list (JSON)", json.dumps(chunks, ensure_ascii=False, indent=2), "fixitai_chunks.json", "application/json")
    else:
        st.info("The knowledge base is empty. Add a document to get started.")

elif page == "Evaluation":
    page_heading("Model performance", "Evaluation", "Review the scratch-trained Transformer’s held-out retrieval results and training progress.")
    report_path = ROOT / "outputs" / "fixitai_evaluation.json"
    metadata_path = MODEL_DIR / "training_metadata.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        validation = report.get("validation", {})
        metric_cols = st.columns(4)
        metric_cols[0].metric("Recall @ 1", f"{validation.get('recall_at_1', 0):.1%}")
        metric_cols[1].metric("Recall @ 5", f"{validation.get('recall_at_5', 0):.1%}")
        metric_cols[2].metric("MRR", f"{validation.get('mrr', 0):.3f}")
        metric_cols[3].metric("Validation queries", f"{validation.get('queries', 0):,}")
        st.caption(f"Device: {report.get('gpu', 'CUDA')} · Best epoch: {validation.get('epoch', '—')} · Candidate answers: {validation.get('candidate_answers', 0):,}")
        history = report.get("all_epochs", [])
        if history:
            section_heading("Training progress", "Training loss and held-out retrieval quality by epoch.")
            history_frame = pd.DataFrame(history)
            chart = px.line(history_frame, x="epoch", y=["train_loss", "recall_at_1", "recall_at_5", "mrr"], markers=True, labels={"value": "Score / loss", "variable": "Metric", "epoch": "Epoch"})
            chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white", legend_title_text="", margin=dict(l=10, r=10, t=20, b=10), height=360)
            st.plotly_chart(chart, use_container_width=True)
        with st.expander("Dataset and evaluation notes"):
            if metadata_path.exists():
                meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                st.json(meta.get("data_quality", {}))
            st.write(report.get("metrics_caution", "Metrics reflect retrieval against weak labels and do not prove technical correctness."))
    else:
        st.info("No training evaluation exists yet. Run `python train_fixitai.py` with the archive available to create held-out retrieval metrics.")

elif page == "History & Feedback":
    page_heading("Your recent activity", "History & Feedback", "Review past searches and tell FixItAI whether the retrieved evidence was useful.")
    history = load_history(HISTORY)
    if history:
        reversed_history = list(reversed(history))
        st.metric("Saved searches", f"{len(history):,}", "Stored locally by default")
        history_frame = pd.DataFrame([
            {"Search": row.get("query", ""), "Feedback": row.get("feedback") or "Not provided", "Sources": len(row.get("result_ids", []))}
            for row in reversed_history
        ])
        st.dataframe(history_frame, use_container_width=True, hide_index=True)
        section_heading("Leave feedback", "Choose a recent search and record whether its retrieved evidence helped.")
        idx = st.selectbox("Recent search", range(len(reversed_history)), format_func=lambda i: reversed_history[i].get("query", "(empty query)"))
        sentiment = st.radio("Was the retrieved evidence useful?", ["Useful", "Not useful"], horizontal=True)
        if st.button("Save feedback", type="primary"):
            save_feedback(HISTORY, idx, sentiment)
            st.success("Feedback saved.")
            st.rerun()
    else:
        st.info("Your searches will appear here after you ask a question.")
    st.caption("History is saved to a local JSON file by default. Set FIXITAI_MONGODB_URI to store history in MongoDB instead.")

elif page == "About FixItAI":
    page_heading("How this workspace works", "About FixItAI", "A local technical troubleshooting and evidence retrieval project.")
    cards = st.columns(3)
    with cards[0]:
        st.metric("Transformer", "2 layers", "128 hidden · 4 attention heads")
    with cards[1]:
        st.metric("Initialization", "From scratch", "No pretrained weights")
    with cards[2]:
        st.metric("Dense inference", "CUDA only", html.escape(torch.cuda.get_device_name(0)))
    section_heading("Retrieval pipeline", "Each query is matched against technical passages using three complementary signals.")
    pipeline_cols = st.columns(3)
    for col, title, body in zip(
        pipeline_cols,
        ["TF-IDF", "BM25", "Scratch Transformer"],
        ["Finds exact words and phrases.", "Ranks term relevance across passages.", "Adds semantic similarity from a locally trained encoder."],
    ):
        with col:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(body)
    st.markdown(
        """
        ### Dataset quality and limits
        The Stack Overflow archive contains relevant technical questions, answers, votes, parent-question IDs, and tags. It does not include an accepted-answer field. Training therefore treats the highest-voted answer as a **weak relevance label**; votes do not prove that an answer is correct or current.

        ### Privacy and sources
        Searches are processed locally. Query history is saved locally unless MongoDB is configured. Retrieved Stack Overflow passages retain links and attribution; review the applicable CC BY-SA terms before redistributing them.
        """
    )

st.markdown('<div class="footer-note">FixItAI · Local technical retrieval · Always verify commands and version-specific guidance against official documentation.</div>', unsafe_allow_html=True)