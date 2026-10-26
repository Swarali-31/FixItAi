"""Device-aware loader and sentence embedding helpers for FixItAI's BERT encoder.

Supports NVIDIA CUDA, Apple Silicon MPS, and CPU inference with automatic detection.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
# pyrefly: ignore[missing-import]
import torch
from torch.nn import functional as F

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Detect and return the best available accelerator (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def require_cuda() -> torch.device:
    """Legacy helper preserved for compatibility; delegates to optimal available device."""
    device = get_device()
    if device.type != "cuda":
        logger.info("CUDA not available. Using '%s' for BERT inference.", device.type)
    return device


def load_bert_model(model_dir: str | Path, device: torch.device | None = None):
    from tokenizers import Tokenizer
    from transformers import BertModel

    base = Path(model_dir)
    encoder_dir = base / "bert_encoder"
    if not (encoder_dir / "config.json").is_file():
        raise FileNotFoundError(f"BERT checkpoint is missing: {encoder_dir}")
    if not (base / "tokenizer.json").is_file():
        raise FileNotFoundError(f"BERT tokenizer is missing: {base / 'tokenizer.json'}")

    device = device or get_device()
    model = BertModel.from_pretrained(
        encoder_dir,
        add_pooling_layer=False,
        local_files_only=True,
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
    vectors = []
    use_cuda_autocast = (device.type == "cuda")

    for start in range(0, len(texts), batch_size):
        enc = tokenizer.encode_batch(texts[start:start + batch_size])
        ids = torch.tensor([e.ids for e in enc], dtype=torch.long, device=device)
        mask = torch.tensor([e.attention_mask for e in enc], dtype=torch.bool, device=device)

        if use_cuda_autocast:
            with torch.autocast("cuda", dtype=torch.float16):
                hidden = model(input_ids=ids, attention_mask=mask.long()).last_hidden_state
                weights = mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
                out = F.normalize(pooled.float(), dim=-1)
        else:
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

