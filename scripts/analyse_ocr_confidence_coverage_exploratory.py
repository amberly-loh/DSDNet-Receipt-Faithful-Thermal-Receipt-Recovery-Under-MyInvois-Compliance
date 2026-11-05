"""Pre-specified exploratory OCR confidence sweep; not a source-support gate."""
import hashlib,json,math
from pathlib import Path
root=Path(__file__).resolve().parents[1]
records_path=root/'artifacts/training_candidate_20261102/field_predictions_validation_20261102.jsonl'
audit=json.loads((root/'artifacts/training_candidate_20261102/dataset_partition_audit.json').read_text(encoding='utf-8'))
excluded={x for group in audit['exact_hash_cross_partition_duplicates'] for x in group['sources']}
thresholds=[0.0,0.5,0.7,0.8,0.9,0.95]
def split(s):return 'calibration' if int(hashlib.sha256(s.encode()).hexdigest()[:8],16)%2==0 else 'evaluation'
def evaluate(rows,threshold):
 accepted=[r for r in rows if r['prediction'] is not None and isinstance(r['confidence'],(int,float)) and math.isfinite(r['confidence']) and r['confidence']>=threshold]
 errors=sum(not r['correct'] for r in accepted)
 return {'labelled':len(rows),'accepted':len(accepted),'errors':errors,'coverage':len(accepted)/len(rows) if rows else None,'risk':errors/len(accepted) if accepted else None}
rows=[json.loads(x) for x in records_path.read_text(encoding='utf-8').splitlines() if x]
rows=[r for r in rows if r['source_id'] not in excluded]
results={}
for field in ['date','total']:
 cal=[r for r in rows if r['field']==field and split(r['source_id'])=='calibration']
 val=[r for r in rows if r['field']==field and split(r['source_id'])=='evaluation']
 curves=[{'threshold':t,**evaluate(cal,t)} for t in thresholds]
 eligible=[v for v in curves if v['accepted']>=10 and v['risk']<=.10]
 chosen=max(eligible,key=lambda v:(v['accepted'],v['threshold'])) if eligible else None
 results[field]={'calibration_curve':curves,'selected_threshold':chosen['threshold'] if chosen else None,'selection_rule':'maximum accepted count among pre-specified thresholds with at least 10 accepted calibration fields and observed error <=10%; tie to higher threshold','evaluation':evaluate(val,chosen['threshold']) if chosen else None,'evaluation_baseline':evaluate(val,0.0)}
out={'scope':'exploratory OCR-confidence only; previously inspected validation sources; not calibrated source support or independent test','split':'SHA-256(source_id) first 32 bits modulo 2','excluded_exact_duplicate_source_ids':sorted(excluded),'thresholds':thresholds,'results':results,'warning':'Selection and evaluation are small and based on previously inspected data; no statistical risk guarantee, no applicability to handwritten invoices or source-trace decisions.'}
dest=root/'artifacts/training_candidate_20261102/ocr_confidence_coverage_exploratory.json';dest.write_text(json.dumps(out,indent=2),encoding='utf-8');print(dest)
