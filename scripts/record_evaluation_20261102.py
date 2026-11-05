from pathlib import Path
import csv,hashlib,json,platform,statistics,sys
from collections import defaultdict
import numpy as np
import torch
root=Path(__file__).resolve().parents[1]
base=root/'artifacts/validation_restoration/exact_excluded_6000_full_gpu'
with (base/'records.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
grouped=defaultdict(list)
for row in rows: grouped[row['method']].append(float(row['seconds']))
times={k:{'n':len(v),'median_s':float(np.median(v)),'p95_s':float(np.percentile(v,95)),'mean_s':statistics.mean(v)} for k,v in grouped.items()}
files={'checkpoint':root/'artifacts/training_exact_duplicate_excluded_20261102_snapshot/checkpoint_snapshot.pt','generation_manifest':root/'artifacts/protocol_lock/exact_duplicate_excluded_20261102/generation_manifest.jsonl','training_config':root/'artifacts/protocol_lock/exact_duplicate_excluded_20261102/training_config.json','evaluation_script':root/'scripts/evaluate_restoration_validation.py','baseline_code':root/'src/dsdnet_receipt/baselines.py','records':base/'records.csv','summary':base/'summary.json','paired_significance':base/'paired_significance.json'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
provenance={k:{'path':str(p),'sha256':sha(p)} for k,p in files.items()}
report={'schema_version':1,'scope':'exploratory validation; fixed 6000-step candidate; exact cross-partition SHA-256 duplicate groups excluded','source_groups':85,'variants':255,'crop_size':[128,128],'environment':{'python':sys.version,'platform':platform.platform(),'pytorch':torch.__version__,'cuda_available_at_recording':torch.cuda.is_available(),'gpu_name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},'files':provenance,'per_crop_timing':times,'timing_limit':'Exploratory wall times from one run on 128x128 crops; method timing boundaries differ, include warmup and are not full receipt latency or a fair speed benchmark.','limitations':['Validation and test partitions previously inspected.','Exact-byte duplicates removed; potential near duplicates have not been resolved.','Synthetic centre crops and 6000-step checkpoint, not complete training or a real-world usability assessment.','Some baseline PSNR aggregates are infinity due to identical inputs and are omitted from comparisons.']}
(base/'provenance_and_performance.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({'per_crop_timing':times,'output':str(base/'provenance_and_performance.json')},indent=2))
