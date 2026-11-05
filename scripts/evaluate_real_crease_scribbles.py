"""Compare two fold detectors on two held-out, sparsely labelled receipts."""
import json,sys
from pathlib import Path
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from dsdnet_receipt.artifact_mask import _load_model
from dsdnet_receipt.restoration import make_complete_document_view,suppress_learned_paper_artifacts
from dsdnet_receipt.config import load_config
from dsdnet_receipt.ocr import PaddleOCRAdapter
torch.set_num_threads(1);cfg,_=load_config(ROOT/'configs/default.json');ocr=PaddleOCRAdapter(cfg)
out=ROOT/'artifacts/real_fold_scribbles';rows=json.loads((out/'manifest.json').read_text());val=[r for r in rows if r['split']=='validation']
models={'previous':ROOT/'artifacts/training_artifact_crease_isolation_dev_v1/best.pt','candidate':out/'candidate.pt'}
report=[]
for row in val:
    image=cv2.imread(row['image'],0);labels=cv2.imread(row['labels'],0);h,w=image.shape;pad=np.pad(image,((0,(-h)%4),(0,(-w)%4)),mode='edge')
    lines=ocr.run(image).lines;base=make_complete_document_view(image,thermal=w/h<.7,ocr_lines=lines);renders=[];record={'row':row['row']}
    for name,path in models.items():
        model,device=_load_model(str(path.resolve()))
        with torch.inference_mode():prob=torch.sigmoid(model(torch.from_numpy(pad).float()[None,None].to(device)/255))[0,0,:h,:w].cpu().numpy()
        pred=prob>=.8;positive=labels==1;negative=labels==0
        record[name]={'marked_fold_recall':float(pred[positive].mean()),'weak_text_flagged':float(pred[negative].mean()),'image_mask_fraction':float(pred.mean())}
        candidate=suppress_learned_paper_artifacts(base,image,lines,path,.8)
        record[name]['pixels_changed']=int((candidate!=base).sum());renders.append(candidate)
        cv2.imwrite(str(out/f'row_{row["row"]:02}_{name}_mask.png'),pred.astype(np.uint8)*255)
    cv2.imwrite(str(out/f'row_{row["row"]:02}_comparison.png'),np.concatenate([image,base,*renders],axis=1));report.append(record)
    print(json.dumps(record),flush=True)
(out/'evaluation.json').write_text(json.dumps({'scope':'two held-out receipts, sparse approximate fold strokes and weak automatic ink negatives; no full pixel ground truth','threshold':.8,'records':report},indent=2))
