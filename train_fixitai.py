"""Train FixItAI's own Transformer on the supplied Stack Overflow Q&A ZIP.

The archive is streamed in place; it is not extracted. This script is GPU-only,
uses random initialization, and selects the highest-voted answer as a noisy
positive because this archive has no accepted-answer column.
"""
from __future__ import annotations
import argparse, csv, html, json, pickle, random, re, sys, zipfile
from html.parser import HTMLParser
from pathlib import Path

ROOT=Path(__file__).resolve().parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.scratch_encoder import ScratchEncoder, encode_texts, require_cuda

class TextExtractor(HTMLParser):
    def __init__(self):super().__init__();self.parts=[]
    def handle_data(self,data):self.parts.append(data)
    def handle_starttag(self,tag,attrs):
        if tag in {"p","br","pre","li","tr","h1","h2","h3"}:self.parts.append("\n")
    def handle_endtag(self,tag):
        if tag in {"p","pre","li","tr","h1","h2","h3"}:self.parts.append("\n")
def clean_html(value:str,max_chars=7000)->str:
    parser=TextExtractor()
    try:parser.feed(value or "")
    except Exception:pass
    text=html.unescape(" ".join(parser.parts)); text=re.sub(r"\s+"," ",text).strip()
    return text[:max_chars]

def _entry(z,name_hint):
    for name in z.namelist():
        if Path(name).name.casefold()==name_hint.casefold():return name
    raise ValueError(f"ZIP is missing required file {name_hint}.")

def load_pairs(zip_path:Path,max_questions:int,min_question_score:int,min_answer_score:int,seed:int):
    """Read Stack Exchange CSVs directly from ZIP and choose one top-voted answer per question."""
    rng=random.Random(seed)
    with zipfile.ZipFile(zip_path) as z:
        qname=_entry(z,"Questions.csv");aname=_entry(z,"Answers.csv");tname=_entry(z,"Tags.csv")
        reservoir=[];seen=0;rows_seen=0
        with z.open(qname) as raw:
            rows=csv.DictReader(__import__("io").TextIOWrapper(raw,encoding="utf-8-sig",errors="replace",newline=""))
            for row in rows:
                rows_seen+=1
                if rows_seen%500000==0:print(f"Scanned {rows_seen:,} question rows; {seen:,} meet the score filter.",flush=True)
                try:score=int(row.get("Score",0) or 0)
                except ValueError:score=0
                title=(row.get("Title") or "").strip(); qid=(row.get("Id") or "").strip()
                if not qid or not title or score<min_question_score:continue
                item={"id":qid,"owner":row.get("OwnerUserId") or "","score":score,"_title":title[:1000],"_body":(row.get("Body", "") or "")[:3000]}
                seen+=1
                if len(reservoir)<max_questions:reservoir.append(item)
                else:
                    j=rng.randrange(seen)
                    if j<max_questions:reservoir[j]=item
        for q in reservoir:
            q["title"]=clean_html(q.pop("_title"),400);q["body"]=clean_html(q.pop("_body"))
        selected={q["id"]:q for q in reservoir};print(f"Question scan: {seen:,} met score >= {min_question_score}; sampled {len(selected):,}.",flush=True)
        tags={qid:[] for qid in selected}
        with z.open(tname) as raw:
            rows=csv.DictReader(__import__("io").TextIOWrapper(raw,encoding="utf-8-sig",errors="replace",newline=""))
            for row in rows:
                qid=str(row.get("Id", ""))
                if qid in tags and row.get("Tag"):tags[qid].append(row["Tag"].strip("<> "))
        best={};answers_seen=0
        with z.open(aname) as raw:
            rows=csv.DictReader(__import__("io").TextIOWrapper(raw,encoding="utf-8-sig",errors="replace",newline=""))
            for row in rows:
                answers_seen+=1
                if answers_seen%1000000==0:print(f"Scanned {answers_seen:,} answer rows; matched {len(best):,} sampled questions so far.",flush=True)
                parent=str(row.get("ParentId", ""))
                if parent not in selected:continue
                try:score=int(row.get("Score",0) or 0)
                except ValueError:score=0
                if score<min_answer_score:continue
                if parent not in best or score>best[parent]["answer_score"]:
                    body=clean_html(row.get("Body", ""))
                    if len(body)>=40:
                        body=body[:3000]
                        best[parent]={"answer_id":str(row.get("Id", "")),"answer_owner":row.get("OwnerUserId") or "","answer_score":score,"answer":body}
    pairs=[]
    for qid,q in selected.items():
        a=best.get(qid)
        if not a:continue
        tag_list=sorted(set(t for t in tags[qid] if t))
        query=(q["title"]+" "+" ".join(tag_list)+" "+q["body"]).strip()
        if len(query)<25:continue
        pairs.append({**q,**a,"tags":tag_list,"query":query,"url":f"https://stackoverflow.com/questions/{qid}"})
    print(f"Matched {len(pairs):,} questions to an answer with score >= {min_answer_score}; selected the highest-scored match.",flush=True)
    if not pairs:raise ValueError("No question/answer pairs matched. Check that this ZIP contains the expected Stack Overflow CSVs.")
    return pairs,{"source_zip":zip_path.name,"question_candidates":seen,"sampled_questions":len(selected),"matched_pairs":len(pairs),"question_min_score":min_question_score,"answer_min_score":min_answer_score,"label_method":"highest answer vote score among answers sharing ParentId; weak relevance label, not accepted-answer ground truth"}

def make_tokenizer(pairs,train_indices,vocab_size,max_length,path):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors, normalizers
    tok=Tokenizer(models.BPE(unk_token="[UNK]"));tok.normalizer=normalizers.Sequence([normalizers.NFKC(),normalizers.Lowercase()]);tok.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=True);tok.decoder=decoders.ByteLevel()
    special=["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]"];trainer=trainers.BpeTrainer(vocab_size=vocab_size,min_frequency=2,special_tokens=special,initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    def corpus():
        for i in train_indices:yield pairs[i]["query"];yield pairs[i]["answer"]
    tok.train_from_iterator(corpus(),trainer=trainer,length=2*len(train_indices))
    cls_id=tok.token_to_id("[CLS]");sep_id=tok.token_to_id("[SEP]")
    tok.post_processor=processors.TemplateProcessing(single="[CLS] $A [SEP]",special_tokens=[("[CLS]",cls_id),("[SEP]",sep_id)])
    tok.enable_truncation(max_length=max_length);tok.enable_padding(length=max_length,pad_id=tok.token_to_id("[PAD]"),pad_token="[PAD]")
    tok.save(str(path));return tok

def batch_encode(tokenizer,texts,device):
    import torch
    enc=tokenizer.encode_batch(texts)
    ids=torch.tensor([e.ids for e in enc],dtype=torch.long,device=device)
    mask=torch.tensor([e.attention_mask for e in enc],dtype=torch.bool,device=device)
    return ids,mask

def evaluate(model,tokenizer,pairs,indices,device,batch_size=32,max_eval=2500,candidate_chunk=8192):
    import torch
    indices=indices[:max_eval];qvec=[];all_answer_vectors=[]
    model.eval()
    with torch.inference_mode():
        for start in range(0,len(indices),batch_size):
            part=[pairs[i] for i in indices[start:start+batch_size]]
            qi,qm=batch_encode(tokenizer,[p["query"] for p in part],device)
            with torch.autocast("cuda",dtype=torch.float16):qvec.append(model(qi,qm))
        for start in range(0,len(pairs),batch_size):
            part=pairs[start:start+batch_size];ai,am=batch_encode(tokenizer,[p["answer"] for p in part],device)
            with torch.autocast("cuda",dtype=torch.float16):all_answer_vectors.append(model(ai,am))
        queries=torch.cat(qvec).float()
        answer_vectors=torch.cat(all_answer_vectors).float()
        gold=torch.tensor(indices,device=device,dtype=torch.long)
        gold_scores=(queries*answer_vectors.index_select(0,gold)).sum(dim=1)
        ranks=torch.ones(len(indices),device=device,dtype=torch.long)
        for start in range(0,len(pairs),candidate_chunk):
            scores=queries@answer_vectors[start:start+candidate_chunk].T
            ranks+=(scores>gold_scores.unsqueeze(1)).sum(dim=1)
        return {"queries":len(indices),"candidate_answers":len(pairs),"recall_at_1":float((ranks<=1).float().mean()),"recall_at_5":float((ranks<=5).float().mean()),"mrr":float((1/ranks.float()).mean())}
def main():
    import numpy as np,torch
    from torch import nn
    from torch.utils.data import DataLoader
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-zip",type=Path,default=ROOT/"data"/"archive.zip")
    ap.add_argument("--output-dir",type=Path,default=ROOT/"models"/"fixitai")
    ap.add_argument("--max-questions",type=int,default=200000,help="Maximum questions to sample from the archive")
    ap.add_argument("--min-question-score",type=int,default=-2147483648)
    ap.add_argument("--min-answer-score",type=int,default=-2147483648)
    ap.add_argument("--epochs",type=int,default=4)
    ap.add_argument("--resume-from",type=Path,default=None,help="Resume model weights from a saved state_dict")
    ap.add_argument("--start-epoch",type=int,default=0,help="Zero-based epoch index for a resumed run")
    ap.add_argument("--batch-size",type=int,default=16)
    ap.add_argument("--max-length",type=int,default=128)
    ap.add_argument("--vocab-size",type=int,default=12000)
    ap.add_argument("--learning-rate",type=float,default=3e-4)
    ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args();device=require_cuda();torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed);random.seed(args.seed);np.random.seed(args.seed)
    if not args.dataset_zip.is_file():raise SystemExit(f"Dataset ZIP not found: {args.dataset_zip}")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    print(f"Device: {device} â€” {torch.cuda.get_device_name(device)}; no CPU fallback and no pretrained weights.",flush=True)
    pairs,quality=load_pairs(args.dataset_zip,args.max_questions,args.min_question_score,args.min_answer_score,args.seed)
    order=list(range(len(pairs)));random.shuffle(order);n_val=max(1,int(0.1*len(order)));val_indices=order[:n_val];train_indices=order[n_val:]
    tokenizer=make_tokenizer(pairs,train_indices,args.vocab_size,args.max_length,args.output_dir/"tokenizer.json")
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(args.output_dir/"tokenizer.json"));tokenizer.enable_truncation(max_length=args.max_length);pad_id=tokenizer.token_to_id("[PAD]");tokenizer.enable_padding(length=args.max_length,pad_id=pad_id,pad_token="[PAD]")
    vocab_size=tokenizer.get_vocab_size()
    architecture={"vocab_size":vocab_size,"max_length":args.max_length,"hidden_size":128,"layers":2,"heads":4,"dropout":0.1,"pad_id":pad_id}
    model=ScratchEncoder(**architecture).to(device)
    if args.resume_from is not None:
        model.load_state_dict(torch.load(args.resume_from,map_location=device,weights_only=True))
        print(f"Resumed model weights from {args.resume_from}.",flush=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=0.01)
    scaler=torch.amp.GradScaler("cuda")
    def collate(items):return [x["query"] for x in items],[x["answer"] for x in items]
    train_rows=torch.utils.data.Subset(pairs,train_indices)
    loader=DataLoader(train_rows,batch_size=args.batch_size,shuffle=True,drop_last=True,collate_fn=collate,num_workers=0,pin_memory=False)
    if len(loader)==0:raise SystemExit("Not enough pairs for a full training batch; lower --batch-size or increase --max-questions.")
    best=-1;history=[]
    prior_history=args.output_dir/"training_history.json"
    if args.resume_from is not None and prior_history.is_file():
        history=json.loads(prior_history.read_text(encoding="utf-8"))
        best=max((item.get("mrr",-1) for item in history),default=-1)
    for epoch in range(args.start_epoch,args.epochs):
        model.train();running=0.0
        for step,(queries,answers) in enumerate(loader,1):
            qi,qm=batch_encode(tokenizer,queries,device);ai,am=batch_encode(tokenizer,answers,device);optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16):
                qv=model(qi,qm);av=model(ai,am);logits=(qv@av.T)/0.07;labels=torch.arange(len(queries),device=device);loss=(nn.functional.cross_entropy(logits,labels)+nn.functional.cross_entropy(logits.T,labels))*0.5
            scaler.scale(loss).backward();scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(),1.0);scaler.step(optimizer);scaler.update();running+=float(loss.detach())
            if step%100==0:print(f"epoch {epoch+1}/{args.epochs} step {step}/{len(loader)} loss {running/step:.4f}",flush=True)
        metrics=evaluate(model,tokenizer,pairs,val_indices,device)
        record={"epoch":epoch+1,"train_loss":running/len(loader),**metrics};history.append(record)
        print(json.dumps(record),flush=True)
        if metrics["mrr"]>best:
            best=metrics["mrr"];torch.save(model.state_dict(),args.output_dir/"model.pt")
            with (args.output_dir/"model.pkl").open("wb") as handle:pickle.dump({"architecture":architecture,"state_dict":{key:value.detach().cpu() for key,value in model.state_dict().items()},"optimizer_state_dict":{key:{k:(v.detach().cpu() if torch.is_tensor(v) else v) for k,v in value.items()} if isinstance(value,dict) else value for key,value in optimizer.state_dict().items()},"scaler_state_dict":scaler.state_dict(),"completed_epoch":epoch+1},handle,protocol=pickle.HIGHEST_PROTOCOL)
            (args.output_dir/"model_config.json").write_text(json.dumps({"architecture":architecture,"model_type":"ScratchEncoder","pretrained":False},indent=2),encoding="utf-8")
    # Model-selected passages retain IDs and source links for attribution and audit.
    documents=[]
    for p in pairs:
        documents.append({"document_id":f"stackoverflow-{p['id']}","chunk_id":f"stackoverflow-answer-{p['answer_id']}","title":p["title"],"source":f"Stack Overflow question #{p['id']} Â· answer #{p['answer_id']}","url":f"https://stackoverflow.com/a/{p['answer_id']}","question_url":p["url"],"category":"Community technical Q&A","technologies":p["tags"],"text":p["answer"],"position":0,"question_score":p["score"],"answer_score":p["answer_score"],"question_owner_id":p["owner"],"answer_owner_id":p["answer_owner"],"attribution":f"Stack Overflow post {p['id']} (question) and {p['answer_id']} (answer); user IDs {p['owner'] or 'unavailable'} and {p['answer_owner'] or 'unavailable'}; CC BY-SA; source links above.","document_hash":str(p["id"])+":"+str(p["answer_id"])})
    seed_dir=ROOT/"data"/"knowledge_base"
    from src.fixitai import load_documents,chunk_documents
    documents.extend(chunk_documents(load_documents(seed_dir)))
    # Encode and save final corpus with only the scratch-trained CUDA model.
    model.eval();texts=[d["title"]+". "+" ".join(d.get("technologies",[]))+". "+d["text"] for d in documents]
    vectors=encode_texts(model,tokenizer,texts,device,batch_size=max(8,args.batch_size))
    np.savez_compressed(args.output_dir/"passage_embeddings.npz",embeddings=vectors,chunk_ids=np.asarray([d["chunk_id"] for d in documents]))
    (args.output_dir/"indexed_chunks.json").write_text(json.dumps(documents,ensure_ascii=False),encoding="utf-8")
    quality.update({"training_pairs":len(train_indices),"validation_pairs":len(val_indices),"sampled_answer_ids":len(documents),"answer_selection_is_weak_supervision":True})
    report={"device":"cuda","gpu":torch.cuda.get_device_name(device),"pretrained_weights":False,"training_method":"scratch Transformer dual encoder with symmetric in-batch contrastive loss","architecture":architecture,"epochs":args.epochs,"history":history,"best_validation_mrr":best,"data_quality":quality,"license_note":"Stack Overflow content is CC BY-SA; indexed passages retain post IDs and links. Review applicable attribution/share-alike terms before redistributing this derived corpus."}
    (args.output_dir/"training_metadata.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    (args.output_dir/"index_metadata.json").write_text(json.dumps({"model":"FixItAI ScratchEncoder","model_dir":str(args.output_dir),"device":"cuda","passages":len(documents),"embedding_dimension":int(vectors.shape[1]),"pretrained":False},indent=2),encoding="utf-8")
    (args.output_dir/"training_history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
    eval_report={"model":"FixItAI scratch Transformer dual encoder","device":"cuda","gpu":torch.cuda.get_device_name(device),"pretrained_weights":False,"weak_supervision":quality["label_method"],"validation":max(history,key=lambda r:r["mrr"]),"all_epochs":history,"data_quality":quality,"metrics_caution":"Validation retrieves the highest-voted answer selected by a weak heuristic; it does not measure verified technical correctness."}
    eval_path=ROOT/"outputs"/"fixitai_evaluation.json";eval_path.parent.mkdir(parents=True,exist_ok=True);eval_path.write_text(json.dumps(eval_report,indent=2),encoding="utf-8")
    print(f"Saved scratch-trained CUDA model and {len(documents):,} passages to {args.output_dir}.",flush=True)

if __name__=="__main__":main()

