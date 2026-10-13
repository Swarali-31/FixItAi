"""Export the most recent held-out CUDA retrieval evaluation as dashboard JSON."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def main():
    source=ROOT/"models"/"fixitai"/"training_metadata.json"
    if not source.exists():raise SystemExit("No trained model report found. Run train_fixitai.py first.")
    metadata=json.loads(source.read_text(encoding="utf-8"))
    history=metadata.get("history",[])
    best=max(history,key=lambda row:row.get("mrr",0),default={})
    report={"model":"FixItAI scratch Transformer dual encoder","device":metadata.get("device"),"gpu":metadata.get("gpu"),"pretrained_weights":metadata.get("pretrained_weights"),"weak_supervision":metadata.get("data_quality",{}).get("label_method"),"validation":best,"all_epochs":history,"data_quality":metadata.get("data_quality"),"metrics_caution":"Validation retrieves the highest-voted answer selected by a weak heuristic; it does not measure verified technical correctness."}
    out=ROOT/"outputs"/"fixitai_evaluation.json";out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
