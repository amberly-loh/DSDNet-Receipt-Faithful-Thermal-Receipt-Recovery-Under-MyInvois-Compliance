"""Compare development checkpoints on source-held-out synthetic pairs."""
import json, random, sys, hashlib
from pathlib import Path
import cv2, numpy as np, torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from dsdnet_receipt.config import load_config
from dsdnet_receipt.dsdnet_adapter import DSDNetReceiptAdapter,gaussian_kernel
from dsdnet_receipt.metrics import psnr,ssim,cer
from dsdnet_receipt.validation import stroke_fidelity_metrics
from dsdnet_receipt.ocr import PaddleOCRAdapter
torch.set_num_threads(1)
def evaluation_crop(clean, degraded, seed, size=128):
    height,width=clean.shape
    if min(height,width)<size:
        scale=max(size/height,size/width)
        clean=cv2.resize(clean,(int(np.ceil(width*scale)),int(np.ceil(height*scale))),interpolation=cv2.INTER_CUBIC)
        degraded=cv2.resize(degraded,(clean.shape[1],clean.shape[0]),interpolation=cv2.INTER_CUBIC)
        height,width=clean.shape
    rng=random.Random(seed)
    threshold,_=cv2.threshold(clean,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    foreground=np.argwhere(clean<threshold)
    if len(foreground):
        row,col=foreground[rng.randrange(len(foreground))]
        top=int(np.clip(int(row)-rng.randrange(size),0,height-size))
        left=int(np.clip(int(col)-rng.randrange(size),0,width-size))
    else:
        top=rng.randrange(height-size+1);left=rng.randrange(width-size+1)
    return clean[top:top+size,left:left+size],degraded[top:top+size,left:left+size]
rows=[json.loads(line) for line in (ROOT/'artifacts/fold_text_dev/manifest.jsonl').read_text().splitlines()]
train={r['source_id'] for r in rows if r['project_partition']=='train'}
validation=[r for r in rows if r['project_partition']=='validation']
assert not train.intersection(r['source_id'] for r in validation)
cfg,_=load_config(ROOT/'configs/default.json'); model=DSDNetReceiptAdapter(cfg['dsdnet']['source_path']).cuda().eval();kernel=gaussian_kernel(0).cuda()
paths={'prior':ROOT/'artifacts/training_low_resolution_dev_v2/checkpoint_0000500.pt',
       'candidate':ROOT/'artifacts/training_fold_text_dev_v1/checkpoint_0000500.pt'}
samples=[]
for r in validation:
    target=cv2.imread(r['clean_reference'],0);obs=cv2.imread(r['degraded_path'],0)
    target,obs=evaluation_crop(target,obs,50000+r['seed'])
    samples.append((r,target,obs))
results={};previews=[]
for label,path in paths.items():
    model.load_state_dict(torch.load(path,map_location='cpu')['model']); model.cuda();metrics=[];images=[]
    for row,target,obs in samples:
        with torch.inference_mode():
            prediction=model(torch.from_numpy(obs).float()[None,None].cuda()/255,kernel,retain_diagnostics=False)['restored']
        output=(prediction[0,0].clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)
        fidelity=stroke_fidelity_metrics(target,output,target<190)
        metrics.append({'source':row['source_id'],'psnr':psnr(target,output),'ssim':ssim(target,output),**fidelity})
        images.append(output)
    results[label]={'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'summary':{key:float(np.mean([m[key] for m in metrics])) for key in ('psnr','ssim','deleted_stroke_rate','false_stroke_rate')},'records':metrics}
    previews.append(images)
    print(label,results[label]['summary'],flush=True)
ocr=PaddleOCRAdapter(cfg);ocr_pairs=[]
for i,(row,target,obs) in enumerate(samples[:12]):
    reference=ocr.run(target).text.strip()
    if len(reference)<10:continue
    ocr_pairs.append({'source':row['source_id'],'reference_characters':len(reference),
        'prior_cer':cer(reference,ocr.run(previews[0][i]).text.strip()),
        'candidate_cer':cer(reference,ocr.run(previews[1][i]).text.strip())})
report={'scope':'development validation; synthetic folds, source-derived targets; target OCR is an unverified proxy, not ground-truth transcription',
        'validation_sources':len(samples),'train_validation_overlap':0,'test_images_read':0,'models':results,'ocr_proxy':ocr_pairs}
out=ROOT/'artifacts/fold_text_dev';(out/'evaluation.json').write_text(json.dumps(report,indent=2))
panels=[np.concatenate([target,obs,previews[0][i],previews[1][i]],axis=1) for i,(_,target,obs) in enumerate(samples[:12])]
cv2.imwrite(str(out/'comparison.png'),np.concatenate(panels,axis=0))
print('ocr_proxy_cases',len(ocr_pairs),flush=True)
