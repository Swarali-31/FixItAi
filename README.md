# FixItAI

FixItAI is a local technical troubleshooting website. Its React and Vite frontend uses a FastAPI backend, a BERT Transformer encoder trained from scratch, and indexed technical passages.

## Start the website in development

Open two PowerShell terminals.

**Terminal 1 - Python API**

```powershell
cd D:\NLP_PROJECT
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

**Terminal 2 - React frontend**

```powershell
cd D:\NLP_PROJECT\frontend
npm.cmd install
npm.cmd run dev
```

Open `http://127.0.0.1:5173`. The home page and non-search sections do not need to initialize CUDA. Search loads the BERT encoder and requires the NVIDIA CUDA runtime.

## Build a single local website

```powershell
cd D:\NLP_PROJECT\frontend
npm.cmd install
npm.cmd run build
cd D:\NLP_PROJECT
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. FastAPI serves the compiled React website and its API from the same local process.

## BERT retrieval encoder

The active model is a compact bidirectional BERT encoder trained from scratch on the question/answer sample with a contrastive retrieval objective. Its parameters start from random initialization; it uses no pretrained weights. Archive parsing and tokenization are CPU-side; BERT forward/backward training requires CUDA and has no CPU fallback.

```powershell
cd D:\NLP_PROJECT
.\.venv\Scripts\python.exe train_fixitai_bert.py --epochs 4 --batch-size 8 --max-questions 200000
```

The tokenizer, BERT checkpoint, epoch history, run status, and indexed passage vectors are saved under `models\fixitai_bert_200k`. Training completed four CUDA epochs on 173,857 matched question/answer pairs (156,472 train; 17,385 validation). Best held-out ranking was epoch 3: Recall@1 6.60%, Recall@5 19.60%, MRR 0.1313, measured on 1,000 queries against 173,857 candidates using vote-derived weak labels. The run resumed from epoch-2 weights; AdamW state restarted because optimizer state was not checkpointed. These results do not establish technical correctness.

To build the BERT passage index and CUDA embeddings:

```powershell
.\.venv\Scripts\python.exe index_fixitai_bert.py
```

The API uses BERT exclusively and reports an error if its checkpoint or passage index is missing; it does not fall back to another encoder. The final CA-3 report is `FixItAI_Report_BERT_Results.docx` and `FixItAI_Report_BERT_Results.pdf`.

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
