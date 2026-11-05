"""Development fine-tune using sparse real-fold scribbles and synthetic replay."""
import json,random,sys
from pathlib import Path
import cv2,numpy as np,torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from dsdnet_receipt.artifact_mask import ReceiptArtifactMaskNet
from train_artifact_mask import make_batch,dice_loss
torch.set_num_threads(1);torch.manual_seed(2026);random.seed(2026);np.random.seed(2026)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg=json.loads((ROOT/'configs/artifact_crease_isolation_dev.json').read_text())
rows=json.loads((ROOT/'artifacts/real_fold_scribbles/manifest.json').read_text())
real=[r for r in rows if r['split']=='train'];held=[r for r in rows if r['split']=='validation']
assert not {r['row'] for r in real}.intersection(r['row'] for r in held)
synthetic=[json.loads(x) for x in (ROOT/'artifacts/fold_text_dev/manifest.jsonl').read_text().splitlines()]
synthetic=[x for x in synthetic if x['project_partition']=='train']
checkpoint=ROOT/'artifacts/training_artifact_crease_isolation_dev_v1/best.pt'
payload=torch.load(checkpoint,map_location=device);model=ReceiptArtifactMaskNet(24).to(device);model.load_state_dict(payload['model']);model.train()
optimizer=torch.optim.AdamW(model.parameters(),lr=0.00002)
arrays=[(cv2.imread(r['image'],0),cv2.imread(r['labels'],0)) for r in real]
out=ROOT/'artifacts/real_fold_scribbles';log=[]
for step in range(1,501):
    optimizer.zero_grad(set_to_none=True)
    if step%2:
        observed,truth=make_batch(synthetic,cfg['natural_receipt_augmentation'],128,2,30000+step)
        observed=observed.to(device);truth=truth.to(device);logits=model(observed)
        loss=F.binary_cross_entropy_with_logits(logits,truth)+dice_loss(logits,truth)
    else:
        images=[];labels=[]
        for _ in range(2):
            img,mask=random.choice(arrays);points=np.argwhere(mask==1);y,x=points[random.randrange(len(points))]
            top=int(np.clip(int(y)-random.randrange(128),0,img.shape[0]-128));left=int(np.clip(int(x)-random.randrange(128),0,img.shape[1]-128))
            images.append(img[top:top+128,left:left+128]);labels.append(mask[top:top+128,left:left+128])
        observed=torch.from_numpy(np.stack(images)).float()[:,None].to(device)/255
        lab=torch.from_numpy(np.stack(labels)).to(device);valid=lab!=255;truth=(lab==1).float();logits=model(observed)[:,0]
        # Balance sparse positive and negative supervision without assuming
        # unmarked pixels are paper. Neither label type is pristine truth.
        loss=F.binary_cross_entropy_with_logits(logits[valid],truth[valid],pos_weight=torch.tensor(2.,device=device))
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
    if step%100==0:print('step',step,'loss',round(float(loss.detach()),4),flush=True)
    log.append(float(loss.detach()))
torch.save({'model':model.state_dict(),'config':cfg,'iterations':500,'training_rows':[r['row'] for r in real]},out/'candidate.pt')
(out/'training.json').write_text(json.dumps({'steps':500,'train_rows':[r['row'] for r in real],'validation_rows':[r['row'] for r in held],'synthetic_replay_sources':len(synthetic),'labels':'sparse approximate folds and weak text negatives','loss_every_100':[log[i-1] for i in range(100,501,100)]},indent=2))
