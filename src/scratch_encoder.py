"""Small, randomly initialized Transformer dual encoder for FixItAI.

This model has no downloaded or pretrained weights. Training and embedding
generation require CUDA; callers must choose a CUDA device explicitly.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

class ScratchEncoder(nn.Module):
    def __init__(self, vocab_size: int, max_length: int=128, hidden_size: int=128, layers: int=2, heads: int=4, dropout: float=0.1, pad_id: int=0):
        super().__init__()
        self.config={"vocab_size":vocab_size,"max_length":max_length,"hidden_size":hidden_size,"layers":layers,"heads":heads,"dropout":dropout,"pad_id":pad_id}
        self.token_embedding=nn.Embedding(vocab_size,hidden_size,padding_idx=pad_id)
        self.position_embedding=nn.Embedding(max_length,hidden_size)
        layer=nn.TransformerEncoderLayer(d_model=hidden_size,nhead=heads,dim_feedforward=hidden_size*4,dropout=dropout,activation="gelu",batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,num_layers=layers,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(hidden_size)
        self.projection=nn.Linear(hidden_size,hidden_size,bias=False)
        self.apply(self._init_weights)
        with torch.no_grad():self.token_embedding.weight[pad_id].zero_()

    @staticmethod
    def _init_weights(module):
        if isinstance(module,(nn.Linear,nn.Embedding)):
            nn.init.normal_(module.weight,mean=0.0,std=0.02)
            if isinstance(module,nn.Linear) and module.bias is not None:nn.init.zeros_(module.bias)
        elif isinstance(module,nn.LayerNorm):nn.init.ones_(module.weight);nn.init.zeros_(module.bias)

    def forward(self,input_ids:torch.Tensor,attention_mask:torch.Tensor)->torch.Tensor:
        length=input_ids.shape[1]
        pos=torch.arange(length,device=input_ids.device).unsqueeze(0)
        x=self.token_embedding(input_ids)+self.position_embedding(pos)
        x=self.encoder(x,src_key_padding_mask=~attention_mask.bool())
        weights=attention_mask.unsqueeze(-1).to(x.dtype)
        pooled=(x*weights).sum(dim=1)/weights.sum(dim=1).clamp_min(1)
        return F.normalize(self.projection(self.norm(pooled)).float(),p=2,dim=-1)

def require_cuda()->torch.device:
    if not torch.cuda.is_available():raise RuntimeError("FixItAI is configured GPU-only. CUDA is unavailable; refusing CPU training or inference.")
    return torch.device("cuda:0")

def load_scratch_model(model_dir:str|Path,device:torch.device|None=None):
    device=device or require_cuda(); p=Path(model_dir)
    metadata=json.loads((p/"model_config.json").read_text(encoding="utf-8"))
    config=metadata["architecture"]
    model=ScratchEncoder(**config)
    state=torch.load(p/"model.pt",map_location="cpu",weights_only=True)
    model.load_state_dict(state,strict=True); model.to(device).eval()
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(p/"tokenizer.json"))
    tokenizer.enable_truncation(max_length=config["max_length"])
    tokenizer.enable_padding(length=config["max_length"],pad_id=config["pad_id"],pad_token="[PAD]")
    return model,tokenizer,device

@torch.inference_mode()
def encode_texts(model,tokenizer,texts:list[str],device:torch.device,batch_size:int=64):
    if device.type!="cuda":raise RuntimeError("Embedding generation is CUDA-only.")
    vectors=[]
    for start in range(0,len(texts),batch_size):
        enc=tokenizer.encode_batch(texts[start:start+batch_size])
        ids=torch.tensor([e.ids for e in enc],dtype=torch.long,device=device)
        masks=torch.tensor([e.attention_mask for e in enc],dtype=torch.bool,device=device)
        with torch.autocast("cuda",dtype=torch.float16):out=model(ids,masks)
        vectors.append(out.float())
    return torch.cat(vectors).cpu().numpy() if vectors else __import__("numpy").empty((0,model.config["hidden_size"]),dtype="float32")
