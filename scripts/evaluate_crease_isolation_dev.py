"""Compare crease masks on deterministic, source-disjoint synthetic folds."""
import json,sys,random
from pathlib import Path
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from dsdnet_receipt.artifact_mask import _load_model
from train_dsdnet_receipt import add_natural_receipt_degradation
torch.set_num_threads(1)
cfg=json.loads((ROOT/'configs/artifact_crease_isolation_dev.json').read_text())
rows=[json.loads(line) for line in (ROOT/'artifacts/fold_text_dev/manifest.jsonl').read_text().splitlines()]
train={r['source_id'] for r in rows if r['project_partition']=='train'}
val=[r for r in rows if r['project_partition']=='validation']
assert not train.intersection(r['source_id'] for r in val)
models={'previous':ROOT/'artifacts/training_artifact_mask_v4/checkpoint_0001250.pt',
        'candidate':ROOT/'artifacts/training_artifact_crease_isolation_dev_v1/best.pt'}
results={};previews=[]
for label,path in models.items():
    model,device=_load_model(str(path.resolve()));scores={str(t):dict(tp=0,fn=0,fp=0,bg=0,ink_fp=0,ink_bg=0) for t in (.8,.95,.98)}
    for index,row in enumerate(val):
        img=cv2.imread(row['clean_reference'],0);h,w=img.shape;rng=random.Random(9011+row['seed']);size=128
        if min(h,w)<size:img=cv2.resize(img,(max(w,size),max(h,size)));h,w=img.shape
        # Select ink-rich areas without reading training sources.
        _,bw=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU);pts=np.argwhere(img<bw)
        if len(pts):
            y,x=pts[rng.randrange(len(pts))];top=int(np.clip(int(y)-rng.randrange(size),0,h-size));left=int(np.clip(int(x)-rng.randrange(size),0,w-size))
        else:top=rng.randrange(h-size+1);left=rng.randrange(w-size+1)
        clean=img[top:top+size,left:left+size]
        obs,_,masks=add_natural_receipt_degradation(clean,cfg['natural_receipt_augmentation'],800001+index,return_masks=True)
        truth=masks['crease'];ink=clean<190
        with torch.inference_mode():prob=torch.sigmoid(model(torch.from_numpy(obs).float()[None,None].to(device)/255))[0,0].cpu().numpy()
        for threshold in (.8,.95,.98):
            pred=prob>=threshold;s=scores[str(threshold)];s['tp']+=int((pred&truth).sum());s['fn']+=int((~pred&truth).sum());s['fp']+=int((pred&~truth).sum());s['bg']+=int((~truth).sum());s['ink_fp']+=int((pred&ink&~truth).sum());s['ink_bg']+=int((ink&~truth).sum())
        if index<8:previews.append((label,index,clean,obs,(prob*255).astype(np.uint8)))
    results[label]={t:{'iou':s['tp']/max(1,s['tp']+s['fn']+s['fp']),'recall':s['tp']/max(1,s['tp']+s['fn']),'false_positive_rate':s['fp']/max(1,s['bg']),'noncrease_ink_flagged':s['ink_fp']/max(1,s['ink_bg'])} for t,s in scores.items()}
    print(label,json.dumps(results[label]),flush=True)
out=ROOT/'artifacts/crease_isolation_dev';out.mkdir(exist_ok=True)
(out/'evaluation.json').write_text(json.dumps({'scope':'development validation, synthetic folds only; no real-mask ground truth','validation_sources':len(val),'train_overlap':0,'test_images_read':0,'results':results},indent=2))
for label in models:
    panels=[np.concatenate([clean,obs,pred],axis=1) for name,_,clean,obs,pred in previews if name==label]
    cv2.imwrite(str(out/(label+'_validation.png')),np.concatenate(panels,axis=0))
