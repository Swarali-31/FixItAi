# FixItAI

FixItAI is a local technical troubleshooting website. Its frontend is built with React and Vite; a FastAPI service connects it to the locally trained Transformer and indexed technical passages.

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

Open `http://127.0.0.1:5173`. The React homepage has navigation links and hoverable cards for troubleshooting, the knowledge base, evaluation, search history, and project information. The home page and non-search sections do not need to initialize CUDA. The first search loads the scratch-trained model and requires the NVIDIA CUDA runtime.

## Build a single local website

```powershell
cd D:\NLP_PROJECT\frontend
npm.cmd install
npm.cmd run build
cd D:\NLP_PROJECT
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. FastAPI serves the compiled React website and its API from the same local process.

## Train the Transformer from scratch

The supplied Stack Overflow archive belongs at `data\archive.zip`. The 1.19 GB archive is excluded from GitHub. Training streams its CSV files without extracting the full archive.

```powershell
cd D:\NLP_PROJECT
.\.venv\Scripts\python.exe train_fixitai.py
```

Training requires CUDA and stops if CUDA is unavailable; it does not fall back to CPU or load pretrained weights. The model is a randomly initialized two-layer, 128-hidden Transformer trained with contrastive question-to-answer retrieval. The saved checkpoint and index are in `models\fixitai`.

The dataset has no accepted-answer column. Training uses the highest-voted associated answer as a weak relevance label. Votes do not verify answer correctness or freshness. Recorded held-out results are Recall@1 3.0%, Recall@5 8.1%, and MRR 0.063 against this noisy label setup.

## Train the BERT retrieval encoder from scratch

This experiment randomly initializes a compact bidirectional BERT encoder (no pretrained weights or downloads) and trains it on the question/answer sample with a contrastive retrieval objective. Archive parsing and tokenization are CPU-side; BERT forward/backward training requires CUDA and has no CPU fallback.

```powershell
cd D:\NLP_PROJECT
.\.venv\Scripts\python.exe train_fixitai_bert.py --epochs 4 --batch-size 8 --max-questions 200000
```

The tokenizer, BERT checkpoint, epoch history, and run status are saved under `models\fixitai_bert_200k`. Training completed four CUDA epochs on 173,857 matched question/answer pairs (156,472 train; 17,385 validation). Best held-out ranking was epoch 3: Recall@1 6.60%, Recall@5 19.60%, MRR 0.1313, measured on 1,000 queries against 173,857 candidates using vote-derived weak labels. The run resumed from epoch-2 weights; AdamW state restarted because the optimizer state was not checkpointed. These results do not establish technical correctness.

To build the BERT passage index and CUDA embeddings:

```powershell
.\.venv\Scripts\python.exe index_fixitai_bert.py
```

The API selects BERT only after its checkpoint and matching passage index are present. Until index generation is completed, the already-indexed `models\fixitai` model remains active. The final CA-3 report is `FixItAI_Report_BERT_Results.docx` and `FixItAI_Report_BERT_Results.pdf`.

## Dataset

The archive contains `Questions.csv` (1,264,216 rows, 7 columns), `Answers.csv` (2,014,516 rows, 6 columns), and `Tags.csv` (3,750,994 rows, 2 columns). It is about 1.19 GB compressed and 3.60 GB uncompressed.

Stack Overflow posts are licensed under CC BY-SA terms. The index retains post links and attribution; review those terms before redistributing the derived dataset.

## Project structure

- `frontend/` - React and Vite website
- `api.py` - local FastAPI routes for search, uploads, evaluation, history, and feedback
- `src/scratch_encoder.py` - randomly initialized Transformer; CUDA-only model loading and embeddings
- `train_fixitai.py` - data matching, tokenizer training, CUDA model training, and retrieval evaluation
- `src/fixitai.py` - document ingestion and hybrid retrieval
- `data/knowledge_base/` - starter technical notes
- `models/fixitai/` - trained checkpoint, tokenizer, and retrieval index
- `models/fixitai_bert_200k/` - from-scratch BERT checkpoint, tokenizer, and measured epoch history
- `reports/FixItAI_Report_BERT.html` - editable report source; final Word and PDF versions are in the project root
## Ask general technical questions

Ask FixItAI accepts natural-language technical questions, including app and device symptoms such as missing Microsoft Teams notifications. It retrieves the closest guidance from the local index and displays the strongest match with its evidence and source. It does not generate answers beyond the indexed material; weak matches are labeled so you can add a relevant guide or provide more detail.


The Knowledge Base page browses the indexed Stack Overflow answer passages along with local project notes and uploaded documents. It loads 60 items at a time; use search or Load more to browse the index.

- `models/fixitai_bert_200k/` - BERT-from-scratch checkpoint and completed training history
