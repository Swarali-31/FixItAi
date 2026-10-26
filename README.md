# FixItAI

FixItAI is a local, offline technical diagnostic and information retrieval workspace. It features a modern React + Vite frontend, a FastAPI REST backend, multi-stage Information Retrieval with Reciprocal Rank Fusion (RRF), an extractive Transformer diagnostic summarizer, an English-to-Hindi vernacular diagnostic engine, and thread-safe SQLite persistence.

## System Requirements & Portability

FixItAI runs natively across:
- **macOS (Apple Silicon M1/M2/M3/M4):** Accelerated via PyTorch Metal Performance Shaders (MPS).
- **Windows / Linux with NVIDIA GPU:** Accelerated via PyTorch CUDA 12.x.
- **CPU Fallback:** Vector-optimized float32 inference on any standard machine.

---

## Quickstart

### 1. Install Backend Dependencies

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Start the Application

You can run the frontend and backend together or independently:

**Single Combined Service (FastAPI serves compiled React app at `http://127.0.0.1:8000`):**
```bash
# Build the React frontend
cd frontend
npm install
npm run build
cd ..

# Start FastAPI
uvicorn api:app --host 127.0.0.1 --port 8000
```
Open `http://127.0.0.1:8000` in your browser.

**Development Mode (Hot Reloading):**
- **Terminal 1 (Backend):** `uvicorn api:app --reload --host 127.0.0.1 --port 8000`
- **Terminal 2 (Frontend):** `cd frontend && npm run dev` (Runs at `http://127.0.0.1:5173`)

---

## Running Automated Tests

Run the test suite using `pytest`:
```bash
pytest tests/ -v
```

---

## Git LFS & Large Datasets

Large dataset indices (`indexed_chunks.json`, `passage_embeddings.npz`) are tracked via Git LFS. If you have Git LFS installed, pull the complete 173k passage index:
```bash
git lfs pull
```
*Note: If Git LFS is not installed, FixItAI automatically falls back to indexing the curated technical guides in `data/knowledge_base/` and user-uploaded documents without crashing.*

---

## Information Retrieval & Neural Architecture

- **Multi-Stage Fusion:** Okapi BM25 probabilistic scoring, sublinear TF-IDF (1-2 n-grams), and dense BERT embeddings fused via **Reciprocal Rank Fusion (RRF)**:
  $$RRF(d) = \sum_{m} \frac{w_m}{k + \text{rank}_m(d)}$$
- **Extractive Summarizer:** Pretrained Transformer extractor isolating root causes, action checklists, and verified code fixes.
- **Neural Translator:** Seq2Seq MarianMT with technical token preservation and domain-specific Hindi explanations.
- **Persistence:** Thread-safe local SQLite database (`data/fixitai.db`) with automatic migration from legacy JSON.


## Dataset

The archive contains `Questions.csv` (1,264,216 rows, 7 columns), `Answers.csv` (2,014,516 rows, 6 columns), and `Tags.csv` (3,750,994 rows, 2 columns). It is about 1.19 GB compressed and 3.60 GB uncompressed.

Stack Overflow posts are licensed under CC BY-SA terms. The index retains post links and attribution; review those terms before redistributing the derived dataset.

## Project structure

- `frontend/` - React and Vite website
- `api.py` - local FastAPI routes for search, uploads, evaluation, history, and feedback
- `src/bert_encoder.py` - BERT checkpoint loading and CUDA embeddings
- `train_fixitai_bert.py` - BERT training and retrieval evaluation
- `src/fixitai.py` - document ingestion and hybrid retrieval
- `data/knowledge_base/` - starter technical notes
- `models/fixitai_bert_200k/` - from-scratch BERT checkpoint, tokenizer, passage index, and training history
- `reports/FixItAI_Report_BERT.html` - editable report source; final Word and PDF versions are in the project root
## Ask general technical questions

Ask FixItAI accepts natural-language technical questions, including app and device symptoms such as missing Microsoft Teams notifications. It retrieves the closest guidance from the local index and displays the strongest match with its evidence and source. It does not generate answers beyond the indexed material; weak matches are labeled so you can add a relevant guide or provide more detail.


The Knowledge Base page browses the indexed Stack Overflow answer passages along with local project notes and uploaded documents. It loads 60 items at a time; use search or Load more to browse the index.
