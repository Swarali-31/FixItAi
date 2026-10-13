# FixItAI

FixItAI retrieves technical answers for troubleshooting questions. It combines technical-aware query parsing, TF-IDF, BM25, and a small Transformer dual encoder trained **from random initialization on CUDA**. It does not load Sentence-BERT or any other pretrained weights, call an LLM, or fall back to CPU for model training or inference.

## Dataset audit

The supplied `archive.zip` contains `Questions.csv`, `Answers.csv`, and `Tags.csv` from Stack Overflow. The columns include question IDs, titles, bodies, scores, answer parent IDs, answer bodies/scores, and question tags. This is technically relevant community Q&A for software troubleshooting and is a workable retrieval corpus.

It does **not** contain an `AcceptedAnswerId` field. FixItAI therefore uses the highest-voted answer associated with each question as a weak positive training pair. Votes are community feedback, not verified correctness; old or high-scoring answers can be obsolete or wrong. The model's held-out retrieval metrics are agreement with this noisy pairing setup, not proof of technical correctness. Keep that qualification in academic reporting.

The training script reads the CSV files directly from the ZIP, samples a bounded subset of questions with positive question scores, matches answers via `ParentId`, and attaches tags. It preserves source post IDs, links, vote scores, and available user IDs for traceability. Stack Overflow contributions are under versions of CC BY-SA; review and follow attribution/share-alike conditions before redistribution. [Stack Overflow licensing](https://stackoverflow.com/help/licensing).

The 1.19 GB source archive is excluded from this GitHub repository. To train from scratch, place your copy at `data\archive.zip`; the script streams its CSVs without extracting the archive. The committed trained checkpoint and index let you run the app without retraining.

## Train from scratch on CUDA (required)

The CUDA model uses a byte-level BPE tokenizer trained on the training split and a randomly initialized 2-layer, 128-hidden Transformer encoder. It learns question-to-answer retrieval with symmetric in-batch contrastive loss. Validation uses held-out question/answer pairs and reports Recall@1, Recall@5, and MRR. The answer selection labels are weak, as described above.

On this workstation, from PowerShell:

```powershell
cd D:\NLP_PROJECT
.\.venv\Scripts\python.exe train_fixitai.py
```

The default is 30,000 sampled questions, 4 epochs, and batch size 16 to fit the RTX 3050's 4 GB VRAM. Use `--max-questions`, `--epochs`, or `--batch-size` to adjust. Training **stops with an error** if CUDA is unavailable; it never quietly trains on CPU. No pretrained model is downloaded.

Training creates `models/fixitai/model.pt`, `tokenizer.json`, `model_config.json`, `training_metadata.json`, `training_history.json`, `indexed_chunks.json`, and `passage_embeddings.npz`. Check `training_metadata.json` for the actual device, sampled/matched counts, weak-label definition, and validation results.

### Recorded run on the supplied archive

On the RTX 3050 Laptop GPU, the run sampled 30,000 of 594,057 positive-score questions, matched 22,814 positive-score answers, trained on 20,533 pairs, and validated on 2,281 queries against all 22,814 candidate answers. The best epoch (4) reported Recall@1 **3.0%**, Recall@5 **8.1%**, and MRR **0.063**. These are modest retrieval results against the highest-voted-answer heuristic, not an expert correctness score. The evaluation dashboard and `models/fixitai/training_metadata.json` contain the complete epoch history.

## Run the app

Train first, then:

```powershell
cd D:\NLP_PROJECT
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The app also refuses startup when CUDA or the trained checkpoint/index is missing. Dense query encoding runs on CUDA only. TF-IDF and BM25 are lexical baselines implemented by scikit-learn/rank-bm25; they run on CPU. The interface shows retrieved source passages, Stack Overflow post links/attribution, component scores, and a low-evidence message.

## Verify CUDA

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

This environment should report CUDA available and the RTX 3050. Install dependencies if needed with `python -m pip install -r requirements.txt`, then install a matching CUDA PyTorch wheel from the [official selector](https://pytorch.org/get-started/locally/). `torch` in `requirements.txt` alone may install a CPU build depending on the index, so always verify the command above.

## Main project files

- `src/scratch_encoder.py`: random-init Transformer, CUDA-only loading and embeddings.
- `train_fixitai.py`: Stack Overflow ZIP audit, weak-pair creation, tokenizer/model training, held-out retrieval evaluation, and CUDA indexing.
- `src/fixitai.py`: document ingestion, query analysis, TF-IDF/BM25/dense hybrid ranking.
- `app.py`: Streamlit troubleshooting interface, document explorer, evaluation, and feedback.
- `data/knowledge_base/`: locally authored starter notes, included alongside the dataset passages.

