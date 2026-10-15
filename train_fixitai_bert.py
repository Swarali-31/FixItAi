"""Train a randomly initialized, compact BERT encoder for FixItAI retrieval on CUDA.

This is not a pretrained model and does not download weights. It learns a tokenizer
from the training split and optimizes a bidirectional BERT encoder with the same
weakly supervised question/answer contrastive objective as the original baseline.
"""
from __future__ import annotations

import argparse, json, random, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from train_fixitai import load_pairs, make_tokenizer, batch_encode, evaluate


def main():
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Subset
    from transformers import BertConfig, BertModel

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-zip", type=Path, default=ROOT / "data" / "archive.zip")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "models" / "fixitai_bert_200k")
    ap.add_argument("--max-questions", type=int, default=200000)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--vocab-size", type=int, default=12000)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; BERT training has no CPU fallback.")
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed); np.random.seed(args.seed)
    if not args.dataset_zip.is_file():
        raise SystemExit(f"Dataset ZIP not found: {args.dataset_zip}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"CUDA device: {torch.cuda.get_device_name(0)}; BERT weights: random initialization; pretrained weights: none", flush=True)

    pairs, quality = load_pairs(args.dataset_zip, args.max_questions, -2147483648, -2147483648, args.seed)
    order = list(range(len(pairs))); random.shuffle(order)
    n_val = max(1, int(0.1 * len(order)))
    val_indices, train_indices = order[:n_val], order[n_val:]
    tok_path = args.output_dir / "tokenizer.json"
    if not tok_path.exists():
        make_tokenizer(pairs, train_indices, args.vocab_size, args.max_length, tok_path)
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tok_path))
    pad_id = tokenizer.token_to_id("[PAD]")
    tokenizer.enable_truncation(max_length=args.max_length)
    tokenizer.enable_padding(length=args.max_length, pad_id=pad_id, pad_token="[PAD]")

    config = BertConfig(
        vocab_size=tokenizer.get_vocab_size(), hidden_size=128, num_hidden_layers=2,
        num_attention_heads=4, intermediate_size=512, hidden_act="gelu",
        hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
        max_position_embeddings=args.max_length, type_vocab_size=2,
        pad_token_id=pad_id, position_embedding_type="absolute",
    )
    checkpoint = args.output_dir / "bert_encoder"
    model = (BertModel.from_pretrained(checkpoint, add_pooling_layer=False).to(device)
             if (checkpoint / "model.safetensors").is_file() else BertModel(config, add_pooling_layer=False).to(device))
    params = sum(p.numel() for p in model.parameters())
    print(f"Initialized BERT encoder from scratch: {params:,} parameters; config={config.to_json_string().strip()}", flush=True)

    def embed(ids, mask):
        out = model(input_ids=ids, attention_mask=mask.long()).last_hidden_state
        weights = mask.unsqueeze(-1).to(out.dtype)
        pooled = (out * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return nn.functional.normalize(pooled.float(), dim=-1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    def collate(rows): return [p["query"] for p in rows], [p["answer"] for p in rows]
    loader = DataLoader(Subset(pairs, train_indices), batch_size=args.batch_size, shuffle=True,
                        drop_last=True, collate_fn=collate, num_workers=0)
    if not len(loader): raise SystemExit("Not enough pairs for one batch.")
    history_path = args.output_dir / "training_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    start_epoch = len(history)
    if start_epoch >= args.epochs:
        raise SystemExit(f"Already completed {start_epoch}/{args.epochs} epochs.")
    print(f"Resuming from saved epoch {start_epoch}; will train through epoch {args.epochs}.", flush=True)
    (args.output_dir / "training_status.json").write_text(json.dumps({"status":"running","model":"BERT encoder, random initialization","gpu":torch.cuda.get_device_name(0),"training_pairs":len(train_indices),"validation_pairs":len(val_indices),"started_at":time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=2), encoding="utf-8")
    try:
        for epoch in range(start_epoch, args.epochs):
            model.train(); total = 0.0; started = time.time()
            for step, (queries, answers) in enumerate(loader, 1):
                qi, qm = batch_encode(tokenizer, queries, device); ai, am = batch_encode(tokenizer, answers, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    qv, av = embed(qi, qm), embed(ai, am)
                    logits = qv @ av.T / 0.07
                    labels = torch.arange(len(queries), device=device)
                    loss = 0.5 * (nn.functional.cross_entropy(logits, labels) + nn.functional.cross_entropy(logits.T, labels))
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); total += float(loss.detach())
                if step % 100 == 0:
                    print(json.dumps({"epoch":epoch+1,"epochs":args.epochs,"step":step,"steps_per_epoch":len(loader),"mean_loss":round(total/step,5),"elapsed_min":round((time.time()-started)/60,1),"gpu_allocated_gb":round(torch.cuda.memory_allocated()/1024**3,2)}), flush=True)
            # Retrieval evaluation uses the original evaluator's CUDA model interface.
            model.eval()
            class EvalWrapper(nn.Module):
                def forward(self, ids, mask): return embed(ids, mask)
            metrics = evaluate(EvalWrapper(), tokenizer, pairs, val_indices, device, batch_size=8, max_eval=1000)
            record = {"epoch":epoch+1,"train_loss":total/len(loader),**metrics,"epoch_minutes":(time.time()-started)/60}
            history.append(record)
            model.save_pretrained(args.output_dir / "bert_encoder", safe_serialization=True)
            (args.output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            status = {"status":"running" if epoch+1<args.epochs else "completed","model":"BERT encoder, random initialization","pretrained_weights":False,"gpu":torch.cuda.get_device_name(0),"completed_epochs":epoch+1,"requested_epochs":args.epochs,"record":record}
            (args.output_dir / "training_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
            print(json.dumps(record), flush=True)
        metadata = {"model":"BERT bidirectional encoder initialized randomly","pretrained_weights":False,"objective":"symmetric in-batch contrastive question-answer retrieval","device":"cuda","gpu":torch.cuda.get_device_name(0),"architecture":config.to_dict(),"parameters":params,"epochs":args.epochs,"history":history,"data_quality":quality,"validation_note":"Highest-voted associated answer is weak supervision, not verified technical correctness."}
        (args.output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"BERT run completed. Artifacts saved to {args.output_dir}", flush=True)
    except BaseException as exc:
        status = {"status":"interrupted_or_failed","model":"BERT encoder, random initialization","gpu":torch.cuda.get_device_name(0),"completed_epochs":len(history),"error":repr(exc)}
        (args.output_dir / "training_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__": main()
