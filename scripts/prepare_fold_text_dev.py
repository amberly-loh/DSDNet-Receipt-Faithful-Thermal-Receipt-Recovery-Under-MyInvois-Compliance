"""Prepare source-disjoint development pairs; targets remain real scans, not pristine truth."""
import json
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from train_dsdnet_receipt import add_natural_receipt_degradation

source = ROOT / 'artifacts/low_resolution_experiment_2x/manifest.jsonl'
out = ROOT / 'artifacts/fold_text_dev'
out.mkdir(parents=True, exist_ok=True)
records = [json.loads(line) for line in source.read_text().splitlines()]
settings = {'enabled': True, 'probability': 1.0, 'texture_amplitude': [.005, .035],
            'texture_scale_pixels': [2., 12.], 'crease_probability': .9,
            'crease_count': [1, 4], 'crease_strength': [.04, .22],
            'crease_thickness': [1, 4], 'crease_blur_sigma': [.5, 3.],
            'crease_shadow_offset': 5, 'fade_probability': .8,
            'fade_scale_pixels': [12., 40.], 'fade_strength': [.05, .35],
            'blur_probability': .5, 'blur_sigma': [.3, .9]}
prepared=[]
for row in records:
    if row['project_partition'] not in ('train','validation'): continue
    target=cv2.imread(row['clean_reference'],0)
    observed=cv2.imread(row['degraded_path'],0)
    if target is None or observed is None or target.shape != observed.shape:
        raise ValueError('Missing or mismatched pair: '+row['source_id'])
    # Score targets without looking at the later model output. This is a
    # relative selection criterion, not a claim of pristine ground truth.
    paper=target[target>=np.percentile(target,60)]
    quality=float(np.percentile(target,90)-np.percentile(target,10)) / max(float(np.std(paper)),1.)
    row['target_quality_proxy']=quality
    prepared.append((row,observed))
selected=[]
for partition in ('train','validation'):
    part=[item for item in prepared if item[0]['project_partition']==partition]
    cutoff=float(np.quantile([item[0]['target_quality_proxy'] for item in part],.25))
    for row,observed in part:
        if row['target_quality_proxy']<cutoff: continue
        degraded,metadata=add_natural_receipt_degradation(observed,settings,row['seed']+230001)
        folder=out/partition/row['source_id'];folder.mkdir(parents=True,exist_ok=True)
        path=folder/'observed.png';cv2.imwrite(str(path),degraded)
        selected.append({**row,'degraded_path':str(path.resolve()),'added_degradation':metadata})
train={r['source_id'] for r in selected if r['project_partition']=='train'}
val={r['source_id'] for r in selected if r['project_partition']=='validation'}
assert not train.intersection(val)
(out/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selected))
print(json.dumps({'train_sources':len(train),'validation_sources':len(val),'test_sources_read':0,
                  'target':'quality-filtered scan-derived target; not pristine', 'selection':'top 75 percent per partition by contrast and paper variation'}))
