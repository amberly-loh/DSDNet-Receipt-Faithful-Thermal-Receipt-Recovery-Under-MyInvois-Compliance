"""Non-commercial research-only handwritten digit-string experiment.

ORAND-CAR-2014 is CC BY-NC-ND 4.0. The resulting model is experimental;
no permission to use it commercially is implied. Test set is not used.
"""
from pathlib import Path
import json, random, re, sys
import cv2
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from train_handwritten_date_pilot import Recognizer, ALPHABET
from finetune_trocr_handwriting import AMOUNTS, image_for

DATA=ROOT/'artifacts'/'handwriting_datasets'/'orand'/'ORAND-CAR-2014'

def load_records():
    out=[]
    for group in ('a','b'):
        folder=DATA/f'CAR-{group.upper()}'
        for row in (folder/f'{group}_train_gt.txt').read_text(encoding='utf8').splitlines():
            parts=row.split('\t')
            if len(parts)<2: continue
            name,label=parts[0].strip(),parts[1].strip()
            if re.fullmatch(r'\d{1,15}',label): out.append((folder/f'{group}_train_images'/name,label))
    return out

def pixels(path, augment=False):
    image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if image is None: raise RuntimeError(f'Missing image: {path}')
    image=cv2.resize(image,(192,48),interpolation=cv2.INTER_AREA)
    if augment:
        matrix=cv2.getRotationMatrix2D((96,24),random.uniform(-2,2),random.uniform(.96,1.04))
        image=cv2.warpAffine(image,matrix,(192,48),borderValue=230)
    return torch.from_numpy(image.astype('float32')/255).unsqueeze(0)

def decode(out):
    values=out.argmax(2).squeeze(1).tolist(); result=[]; last=0
    for n in values:
        if n and n!=last: result.append(ALPHABET[n-1])
        last=n
    return ''.join(result)

def main():
    random.seed(56);np.random.seed(56);torch.manual_seed(56);torch.set_num_threads(2)
    records=load_records();random.shuffle(records)
    split=int(len(records)*.9); train,valid=records[:split],records[split:]
    if not train or not valid: raise RuntimeError('No labelled ORAND training data')
    device='cuda' if torch.cuda.is_available() else 'cpu'; model=Recognizer().to(device)
    optim=torch.optim.Adam(model.parameters(),lr=.001);ctc=nn.CTCLoss(blank=0,zero_infinity=True)
    steps=1600
    for step in range(steps):
        batch=random.choices(train,k=24);x=torch.stack([pixels(p,True) for p,_ in batch]).to(device)
        encoded=[[ALPHABET.index(ch)+1 for ch in label] for _,label in batch]
        y=torch.tensor([n for seq in encoded for n in seq],device=device)
        out=model(x); loss=ctc(out,y,torch.full((len(batch),),out.shape[0],dtype=torch.long),torch.tensor([len(seq) for seq in encoded]))
        optim.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optim.step()
        if (step+1)%200==0: print(f'step {step+1}/{steps}, loss {loss.item():.4f}',flush=True)
    model.eval(); n=0; correct=0
    with torch.inference_mode():
        for p,label in valid[:200]:
            pred=decode(model(pixels(p).unsqueeze(0).to(device)));correct+=pred==label;n+=1
        receipt=[]
        for case in AMOUNTS:
            if case[3]!='heldout': continue
            image=np.asarray(image_for(case).convert('L'));image=cv2.resize(image,(192,48),interpolation=cv2.INTER_AREA)
            pred=decode(model(torch.from_numpy(image.astype('float32')/255).unsqueeze(0).unsqueeze(0).to(device)))
            # Exact transcription of visible digits, not inferred decimal places.
            expected={'Document_20250512_0060.jpg':'175000','Invoice_20240316_0035.jpg':'45700','Invoice_20250512_0032.jpg':'550'}[case[0]]
            receipt.append({'file':case[0],'expected_visible_digits_provisional':expected,'prediction':pred,'match':pred==expected})
    output=ROOT/'artifacts'/'orand_amount_pilot';output.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':model.state_dict(),'alphabet':ALPHABET,'research_only':True},output/'research_only_checkpoint.pt')
    report={'source':'ORAND-CAR-2014','license':'CC BY-NC-ND 4.0; research only','steps':steps,'train':len(train),'validation':n,'validation_exact':correct,'validation_scope':'random holdout from ORAND training set, not writer-disjoint','receipt_results':receipt,'warning':'Not validated or licensed for commercial deployment'}
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
