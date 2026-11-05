"""Held-out development comparison, never the project's test partition."""
import hashlib
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dsdnet_receipt.config import load_config
from dsdnet_receipt.dsdnet_adapter import DSDNetReceiptAdapter, gaussian_kernel
from dsdnet_receipt.metrics import psnr, ssim
from dsdnet_receipt.validation import stroke_fidelity_metrics
from train_dsdnet_receipt import crop_and_augment
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--scale', type=int, choices=(1, 2), default=1)
args = parser.parse_args()
folder = ROOT / ('artifacts/low_resolution_experiment' if args.scale == 1 else 'artifacts/low_resolution_experiment_2x')
records = [json.loads(line) for line in (folder / 'manifest.jsonl').read_text().splitlines()]
validation = [r for r in records if r['project_partition'] == 'validation']
train_ids = {r['source_id'] for r in records if r['project_partition'] == 'train'}
assert not train_ids.intersection(r['source_id'] for r in validation)
protocol, _ = load_config(ROOT / 'configs/default.json')
model = DSDNetReceiptAdapter(protocol['dsdnet']['source_path']).cuda().eval()
kernel = gaussian_kernel(0).cuda()
models = {'previous': ROOT / 'artifacts/training_generalization_v2/checkpoint_0000500.pt',
          'step250': ROOT / 'artifacts/training_low_resolution_dev_v1/checkpoint_0000250.pt',
          'step500': ROOT / 'artifacts/training_low_resolution_dev_v1/checkpoint_0000500.pt'}
if args.scale == 2:
    models = {'previous': models['previous'], 'native500': models['step500'],
              'edge250': ROOT / 'artifacts/training_low_resolution_dev_v2/checkpoint_0000250.pt',
              'edge500': ROOT / 'artifacts/training_low_resolution_dev_v2/checkpoint_0000500.pt'}
results, previews = {}, {}
for label, checkpoint in models.items():
    model.load_state_dict(torch.load(checkpoint, map_location='cuda')['model'])
    metrics = []
    for index, row in enumerate(validation):
        clean = cv2.imread(row['clean_reference'], 0)
        observed = cv2.imread(row['degraded_path'], 0)
        clean, observed = crop_and_augment(clean, observed, 128, random.Random(50000 + row['seed']), 1.0)
        x = torch.from_numpy(observed).float()[None,None].cuda()/255
        with torch.inference_mode():
            output = model(x, kernel, retain_diagnostics=False)['restored'].clamp(0,1)
        recovered = (output[0,0].cpu().numpy()*255).round().astype(np.uint8)
        fidelity = stroke_fidelity_metrics(clean, recovered, clean < 190)
        metrics.append({'source': row['source_id'], 'psnr': psnr(clean,recovered), 'ssim': ssim(clean,recovered), **fidelity})
        if index < 8:
            previews.setdefault(index, [clean, observed]).append(recovered)
    summary = {key: float(np.mean([r[key] for r in metrics])) for key in ['psnr','ssim','stroke_recall','deleted_stroke_rate','false_stroke_rate']}
    results[label] = {'checkpoint': str(checkpoint), 'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(), 'summary': summary, 'records': metrics}
    print(label, json.dumps(summary), flush=True)
    # Real complaint image is qualitative evidence only; no target or tuning.
    real = cv2.imread(str(ROOT / 'artifacts/complaint_regression/petronas_registered.png'),0)
    if args.scale == 2:
        real = cv2.resize(real,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
    tensor = torch.from_numpy(real).float()[None,None].cuda()/255
    h,w = real.shape
    tensor = F.pad(tensor,(0,(-w)%64,0,(-h)%64),mode='replicate')
    with torch.inference_mode():
        pred = model(tensor,kernel,retain_diagnostics=False)['restored'][0,0,:h,:w].clamp(0,1)
    cv2.imwrite(str(folder / f'petronas_{label}.png'),(pred.cpu().numpy()*255).round().astype(np.uint8))
rows = [np.concatenate([cv2.resize(p,(256,256),interpolation=cv2.INTER_NEAREST) for p in panels],axis=1) for panels in previews.values()]
cv2.imwrite(str(folder / 'validation_comparison.jpg'), np.concatenate(rows,axis=0))
report = {'scope': 'development validation; synthetic low-resolution observations; scan-derived targets', 'validation_sources': len(validation), 'source_overlap': 0, 'test_access': False, 'methods': results}
(folder / 'evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
