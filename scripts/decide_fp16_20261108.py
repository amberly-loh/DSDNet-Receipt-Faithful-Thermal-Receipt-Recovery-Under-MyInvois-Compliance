"""Adopt half precision for the Fast preset only if, on validation, it keeps field accuracy and CER
(no lower field accuracy, CER not higher by more than 0.005) relative to the Fast preset in full precision."""
import json, sys
from pathlib import Path
import pandas as pd
D = Path(sys.argv[1])
F = ["merchant", "transaction_date", "total_amount"]
def summ(name):
    rows = [json.loads(l) for l in open(D / f"{name}.jsonl", encoding="utf-8")][1:]
    d = pd.DataFrame([r for r in rows if r.get("status") == 200])
    d["fa"] = d[[f + "_correct" for f in F]].astype(float).mean(axis=1)
    s = d.groupby("source_id")[["fa", "cer", "word_f1", "seconds"]].mean().mean()
    return {k: float(v) for k, v in s.items()} | {"failed": int(sum(r.get("status") != 200 for r in rows))}
a, b = summ("fast_fp32"), summ("fast_fp16")
ok = b["fa"] >= a["fa"] and b["cer"] <= a["cer"] + 0.005 and b["failed"] <= a["failed"]
out = {"fast_fp32": a, "fast_fp16": b, "adopt": ok, "rule": "field accuracy not lower; CER not higher by more than 0.005; no extra failures"}
(D / "fp16_decision.json").write_text(json.dumps(out, indent=1)); print(out)
sys.exit(0 if ok else 3)
