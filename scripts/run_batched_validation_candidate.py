"""Evaluate a fixed candidate in isolated source batches and merge only complete results."""
import csv,hashlib,json,os,subprocess,sys
from collections import defaultdict
from pathlib import Path
from evaluate_restoration_validation import source_aggregate
root=Path(__file__).resolve().parents[1]
checkpoint=root/'artifacts/training_exact_duplicate_excluded_20261102_snapshot/checkpoint_snapshot.pt'
manifest=root/'artifacts/protocol_lock/exact_duplicate_excluded_20261102/generation_manifest.jsonl'
base=root/'artifacts/validation_restoration/exact_excluded_6000_batched'
base.mkdir(parents=True,exist_ok=True)
records=[json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line]
sources=sorted({r['source_id'] for r in records if r['project_partition']=='validation'})
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='-1';env['OMP_NUM_THREADS']='2'
combined=[]
for offset in range(0,len(sources),10):
    output=base/f'batch_{offset:03d}'
    if (output/'records.csv').exists():
        raise SystemExit(f'Refusing to overwrite existing batch {output}')
    command=[sys.executable,str(root/'scripts/evaluate_restoration_validation.py'),'--checkpoint',str(checkpoint),'--manifest',str(manifest),'--partition','validation','--source-offset',str(offset),'--max-sources','10','--output-dir',str(output)]
    print(f'Starting source batch {offset+1}-{min(offset+10,len(sources))} of {len(sources)}',flush=True)
    result=subprocess.run(command,cwd=root,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
    if result.returncode:
        (base/'failure.json').write_text(json.dumps({'offset':offset,'returncode':result.returncode,'stderr_tail':result.stderr[-4000:]},indent=2),encoding='utf-8')
        raise SystemExit(f'Batch {offset} failed with exit {result.returncode}; previous batches retained')
    with (output/'records.csv').open(encoding='utf-8-sig',newline='') as f:batch=list(csv.DictReader(f))
    got={r['source_id'] for r in batch}
    if got!=set(sources[offset:offset+10]):raise SystemExit('Batch source mismatch; refusing merge')
    combined.extend(batch)
    print(f'Finished batch with {len(got)} source groups',flush=True)
groups=defaultdict(list)
for row in combined:groups[row['method']].append(row)
if any(len({r['source_id'] for r in rows})!=len(sources) for rows in groups.values()):raise SystemExit('Incomplete method coverage')
with (base/'records.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=combined[0].keys());writer.writeheader();writer.writerows(combined)
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
result={'partition':'validation','formal_test_evidence':False,'source_groups':len(sources),'checkpoint':str(checkpoint),'checkpoint_sha256':digest(checkpoint),'manifest':str(manifest),'manifest_sha256':digest(manifest),'methods':{k:source_aggregate(v) for k,v in groups.items()},'limitations':['Previously inspected validation data; candidate checkpoint not a final trained model.','Exact-byte duplicate groups excluded; near-duplicate pairs require review.','128 x 128 synthetic centre crops do not represent full receipt quality.']}
(base/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print('Complete: '+str(base/'summary.json'),flush=True)
