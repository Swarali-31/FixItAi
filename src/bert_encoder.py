"""CUDA-only loader and sentence embedding helpers for FixItAI's BERT encoder."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("FixItAI BERT requires CUDA; refusing CPU inference.")
    return torch.device("cuda:0")


def load_bert_model(model_dir: str | Path, device: torch.device | None = None):
    from tokenizers import Tokenizer
    from transformers import BertModel

    base = Path(model_dir)
    encoder_dir = base / "bert_encoder"
    if not (encoder_dir / "config.json").is_file():
        raise FileNotFoundError(f"BERT checkpoint is missing: {encoder_dir}")
    if not (base / "tokenizer.json").is_file():
        raise FileNotFoundError(f"BERT tokenizer is missing: {base / 'tokenizer.json'}")
    device = device or require_cuda()
    if device.type != "cuda":
        raise RuntimeError("FixItAI BERT inference is CUDA-only.")
    model = BertModel.from_pretrained(
        encoder_dir, add_pooling_layer=False, local_files_only=True,
        use_safetensors=True,
    ).to(device).eval()
    tokenizer = Tokenizer.from_file(str(base / "tokenizer.json"))
    max_length = int(model.config.max_position_embeddings)
    pad_id = int(model.config.pad_token_id or 0)
    tokenizer.enable_truncation(max_length=max_length)
    tokenizer.enable_padding(length=max_length, pad_id=pad_id, pad_token="[PAD]")
    return model, tokenizer, device


@torch.inference_mode()
def encode_texts(model, tokenizer, texts: list[str], device: torch.device, batch_size: int = 16):
    if device.type != "cuda":
        raise RuntimeError("FixItAI BERT embeddings are CUDA-only.")
    vectors = []
    for start in range(0, len(texts), batch_size):
        enc = tokenizer.encode_batch(texts[start:start + batch_size])
        ids = torch.tensor([e.ids for e in enc], dtype=torch.long, device=device)
        mask = torch.tensor([e.attention_mask for e in enc], dtype=torch.bool, device=device)
        with torch.autocast("cuda", dtype=torch.float16):
            hidden = model(input_ids=ids, attention_mask=mask.long()).last_hidden_state
            weights = mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
            out = F.normalize(pooled.float(), dim=-1)
        vectors.append(out.cpu())
    return torch.cat(vectors).numpy() if vectors else np.empty((0, model.config.hidden_size), dtype=np.float32)


def load_fixitai_model(model_dir: str | Path, device: torch.device | None = None):
    """Load the project's BERT checkpoint; never substitute another encoder."""
    return load_bert_model(model_dir, device)


def encode_fixitai_texts(model, tokenizer, texts: list[str], device: torch.device, batch_size: int = 16):
    if model.__class__.__name__ != "BertModel":
        raise TypeError("FixItAI only supports its BERT Transformer encoder.")
    return encode_texts(model, tokenizer, texts, device, batch_size)
