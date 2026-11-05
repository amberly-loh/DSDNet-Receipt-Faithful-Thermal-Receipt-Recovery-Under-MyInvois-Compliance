from pathlib import Path
import hashlib,json,platform,sys
import torch
root=Path(__file__).resolve().parents[1]
base=root/'artifacts/validation_restoration/exact_excluded_6000_exploratory_test'
files={'checkpoint':root/'artifacts/training_exact_duplicate_excluded_20261102_snapshot/checkpoint_snapshot.pt','generation_manifest':root/'artifacts/protocol_lock/exact_duplicate_excluded_20261102/generation_manifest.jsonl','training_config':root/'artifacts/protocol_lock/exact_duplicate_excluded_20261102/training_config.json','evaluation_script':root/'scripts/evaluate_restoration_validation.py','baseline_code':root/'src/dsdnet_receipt/baselines.py','records':base/'records.csv','summary':base/'summary.json'}
files['paired_significance']=base/'paired_significance.json'
files['significance_script']=root/'scripts/compare_validation_significance.py'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
report={'scope':'exploratory previously inspected test partition; not independent confirmatory evidence','frozen_checkpoint_iteration':6000,'source_groups':92,'variants':276,'crop_size':[128,128],'environment':{'python':sys.version,'platform':platform.platform(),'pytorch':torch.__version__,'gpu_name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},'files':{k:{'path':str(p),'sha256':sha(p)} for k,p in files.items()},'limitations':['Test partition previously inspected; not a fresh independent test.','Known exact-byte cross-partition duplicates excluded; potential near duplicates not resolved.','Synthetic centre crops do not prove full-receipt or human usability performance.','Model trained for 6000 iterations, not complete one-million-iteration run.']}
(base/'provenance.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(base/'provenance.json')
