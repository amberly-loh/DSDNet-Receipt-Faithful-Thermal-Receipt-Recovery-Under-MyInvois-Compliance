"""Experimental date recognizer on six manually transcribed crops, with three held out.

This is a tiny feasibility experiment, not a production model. Rectangles and
labels are provisional and must not be used to report general handwriting accuracy.
"""
from pathlib import Path
import argparse, json, random
import cv2
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / 'FYP datasets' / 'Real life'
CASES = [
 ('Document_20240511_0008.jpg',(.73,.25,.97,.31),'19/3/24','train'),
 ('Document_20240511_0070.jpg',(.75,.24,.98,.31),'26/3/24','train'),
 ('Document_20240511_0081.jpg',(.74,.25,.97,.31),'13/7/24','train'),
 ('Document_20250512_0027.jpg',(.72,.32,.96,.39),'21/11/24','train'),
 ('Document_20250512_0041.jpg',(.73,.25,.97,.31),'26/6/24','train'),
 ('Document_20250512_0042.jpg',(.73,.25,.97,.31),'15/7/24','train'),
 ('Document_20240511_0045.jpg',(.72,.25,.94,.32),'16/8/24','train'),
 ('Document_20240511_0047.jpg',(.70,.22,.96,.29),'13/4/24','train'),
 ('Document_20250512_0010.jpg',(.72,.25,.97,.32),'18/10/24','train'),
 ('Document_20250512_0075.jpg',(.70,.33,.98,.40),'14-11-24','train'),
 ('Invoice_20240316_0003.jpg',(.75,.055,.98,.15),'1/2/24','train'),
 ('Invoice_20240316_0013.jpg',(.72,.24,.97,.32),'25/1/24','train'),
 ('Invoice_20240316_0021.jpg',(.65,.25,.96,.34),'17/11/24','train'),
 ('Document_20250512_0060.jpg',(.71,.26,.95,.32),'25/6/24','heldout'),
 ('Invoice_20240316_0035.jpg',(.75,.22,.97,.28),'27/2/24','heldout'),
 ('Invoice_20250512_0032.jpg',(.75,.25,.97,.30),'30.11.2024','heldout'),
]
ALPHABET = '0123456789/.'

class Recognizer(nn.Module):
 def __init__(self):
  super().__init__()
  self.cnn=nn.Sequential(nn.Conv2d(1,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(64,96,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((1,48)))
  self.rnn=nn.LSTM(96,96,bidirectional=True,batch_first=True)
  self.out=nn.Linear(192,len(ALPHABET)+1)
 def forward(self,x):
  x=self.cnn(x).squeeze(2).permute(0,2,1)
  return self.out(self.rnn(x)[0]).log_softmax(2).permute(1,0,2)

def crop(case, augment=False):
 name,(x1,y1,x2,y2),_,_=case
 image=cv2.imread(str(DATA/name),cv2.IMREAD_GRAYSCALE)
 if image is None: raise RuntimeError(name)
 h,w=image.shape; image=image[int(y1*h):int(y2*h),int(x1*w):int(x2*w)]
 image=cv2.resize(image,(192,48),interpolation=cv2.INTER_AREA)
 if augment:
  m=cv2.getRotationMatrix2D((96,24),random.uniform(-2,2),random.uniform(.96,1.04))
  m[:,2]+=np.array([random.uniform(-4,4),random.uniform(-2,2)])
  image=cv2.warpAffine(image,m,(192,48),borderValue=230)
  image=np.clip(image.astype(np.float32)*random.uniform(.85,1.1)+random.uniform(-12,12),0,255).astype('uint8')
 return torch.from_numpy(image.astype('float32')/255).unsqueeze(0)

def decode(out):
 indices=out.argmax(2).squeeze(1).tolist(); result=[]; previous=0
 for index in indices:
  if index and index!=previous: result.append(ALPHABET[index-1])
  previous=index
 return ''.join(result)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=1000); args=parser.parse_args()
 random.seed(56); np.random.seed(56); torch.manual_seed(56); torch.set_num_threads(2)
 device='cuda' if torch.cuda.is_available() else 'cpu'; model=Recognizer().to(device)
 optim=torch.optim.Adam(model.parameters(),lr=.001); loss_fn=nn.CTCLoss(blank=0,zero_infinity=True)
 train=[c for c in CASES if c[3]=='train']
 for step in range(args.steps):
  batch=[random.choice(train) for _ in range(12)]
  x=torch.stack([crop(c,True) for c in batch]).to(device)
  encoded=[[ALPHABET.index(ch)+1 for ch in c[2]] for c in batch]
  labels=torch.tensor([c for seq in encoded for c in seq],device=device)
  out=model(x); loss=loss_fn(out,labels,torch.full((len(batch),),out.shape[0],dtype=torch.long),torch.tensor([len(v) for v in encoded],dtype=torch.long))
  optim.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optim.step()
  if (step+1)%200==0: print(f'step {step+1}: loss {loss.item():.4f}',flush=True)
 model.eval(); results=[]
 with torch.no_grad():
  for case in CASES:
   pred=decode(model(crop(case).unsqueeze(0).to(device)))
   results.append({'file':case[0],'split':case[3],'expected':case[2],'prediction':pred,'correct':pred==case[2]})
 output=ROOT/'artifacts'/'handwriting_date_pilot';output.mkdir(parents=True,exist_ok=True)
 torch.save({'state_dict':model.state_dict(),'alphabet':ALPHABET,'steps':args.steps,'seed':56},output/'checkpoint.pt')
 report={'device':device,'train_crops':len(train),'heldout_crops':len(CASES)-len(train),'steps':args.steps,'results':results,'warning':'Small provisional pilot; not validated for deployment.'}
 (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__': main()
