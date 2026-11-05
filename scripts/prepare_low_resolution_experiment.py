"""Development pairs from existing source-disjoint train/validation receipts.

Targets are area-resampled source scans, not OCR renderings or feedback outputs.
The existing test partition is never read. Native blur is unknown to the model.
"""
import hashlib
import json
import argparse
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--scale', type=int, choices=(1, 2), default=1)
args = parser.parse_args()
out = ROOT / ('artifacts/low_resolution_experiment' if args.scale == 1 else 'artifacts/low_resolution_experiment_2x')
out.mkdir(exist_ok=True)
manifest = ROOT / 'artifacts/controlled_degradation_sroie/generation_manifest.jsonl'
sources = {}
for line in manifest.read_text(encoding='utf-8').splitlines():
    row = json.loads(line)
    if row['project_partition'] in ('train', 'validation'):
        sources.setdefault(row['source_id'], row)
rows = []
for index, (source_id, row) in enumerate(sorted(sources.items())):
    seed = int(hashlib.sha256((source_id + '-native-v1').encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    source = cv2.imread(row['clean_reference'], 0)
    width = int(rng.integers(160, 241))
    scale = width / source.shape[1]
    native_height = max(128, round(source.shape[0] * scale))
    target = cv2.resize(source, (width * args.scale, native_height * args.scale), interpolation=cv2.INTER_AREA)
    native = cv2.resize(source, (width, native_height), interpolation=cv2.INTER_AREA)
    sigma = float(rng.uniform(.35, .85))
    observed = cv2.GaussianBlur(native.astype(np.float32), (0, 0), sigma)
    contrast = float(rng.uniform(.45, .85))
    observed = 245 - (255 - observed) * contrast
    y, x = np.mgrid[:native.shape[0], :native.shape[1]]
    center = float(rng.uniform(0, native.shape[0]))
    shadow = rng.uniform(5, 22) * np.exp(-((y - center - .2*x) / rng.uniform(6, 18))**2)
    observed -= shadow
    observed += rng.normal(0, rng.uniform(.3, 1.2), observed.shape)
    observed = np.clip(observed, 0, 255).round().astype(np.uint8)
    if args.scale == 2:
        observed = cv2.resize(observed, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_CUBIC)
    folder = out / row['project_partition'] / source_id
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / 'target.png'), target)
    cv2.imwrite(str(folder / 'observed.png'), observed)
    # Evaluation uses this only for stroke fidelity, not for model input.
    cv2.imwrite(str(folder / 'clean_foreground.png'), np.where(target < 190, 0, 255).astype(np.uint8))
    rows.append({'source_id': source_id, 'source_group': source_id,
                 'project_partition': row['project_partition'], 'variant_id': 'NATIVE-V1',
                 'clean_reference': str(folder / 'target.png'), 'degraded_path': str(folder / 'observed.png'),
                 'clean_foreground': str(folder / 'clean_foreground.png'),
                 'parameters': {'blur_sigma': 0.0, 'actual_blur_sigma': sigma, 'contrast': contrast},
                 'target_policy': 'area-resampled source; no rendered glyphs', 'seed': seed})
# Shuffle source order once: a short run must not see only adjacent source IDs.
np.random.default_rng(56).shuffle(rows)
(out / 'manifest.jsonl').write_text('\n'.join(json.dumps(row) for row in rows) + '\n', encoding='utf-8')
report = {'train_sources': sum(r['project_partition']=='train' for r in rows),
          'validation_sources': sum(r['project_partition']=='validation' for r in rows),
          'test_sources_read': 0, 'target_limit': 'source scans, not pristine ground truth'}
(out / 'dataset_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report))
