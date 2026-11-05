"""Research-only ORAND adaptation of TrOCR; never deploy commercially."""
from pathlib import Path
import argparse, json, random, sys
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tmp'/'trocr_deps'));sys.path.insert(0,str(ROOT/'scripts'))
from train_orand_amount_pilot import load_records
from finetune_trocr_handwriting import AMOUNTS,image_for
from transformers import TrOCRProcessor,VisionEncoderDecoderModel

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=1000);parser.add_argument('--layers',type=int,default=1);parser.add_argument('--run-name',default='default');args=parser.parse_args()
 if args.steps < 0 or not 1 <= args.layers <= 4 or not args.run_name.replace('_','').isalnum(): raise ValueError('Invalid training options')
 random.seed(56);torch.manual_seed(56);torch.set_num_threads(2)
 records=load_records();random.shuffle(records);cut=int(len(records)*.9);train,valid=records[:cut],records[cut:]
 base=ROOT/'tmp'/'trocr_model';processor=TrOCRProcessor.from_pretrained(str(base),local_files_only=True)
 model=VisionEncoderDecoderModel.from_pretrained(str(base),local_files_only=True)
 model.config.decoder_start_token_id=processor.tokenizer.cls_token_id;model.config.pad_token_id=processor.tokenizer.pad_token_id
 for p in model.parameters():p.requires_grad=False
 for layer in model.decoder.model.decoder.layers[-args.layers:]:
  for p in layer.parameters():p.requires_grad=True
 for p in model.decoder.output_projection.parameters():p.requires_grad=True
 device='cuda' if torch.cuda.is_available() else 'cpu';model.to(device).train()
 optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-5)
 order=[];used=set()
 for step in range(args.steps):
  if not order: order=list(range(len(train)));random.shuffle(order)
  index=order.pop();used.add(index);path,label=train[index]
  with Image.open(path) as image: image=image.convert('RGB')
  pixels=processor(images=image,return_tensors='pt').pixel_values.to(device)
  labels=processor.tokenizer(label,return_tensors='pt').input_ids.to(device)
  loss=model(pixel_values=pixels,labels=labels).loss
  optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1);optimizer.step()
  if (step+1)%500==0:print(f'step {step+1}/{args.steps}, loss {loss.item():.4f}',flush=True)
 model.eval();correct=0;rows=[]
 with torch.inference_mode():
  for path,label in valid[:100]:
   with Image.open(path) as img: image=img.convert('RGB')
   pixels=processor(images=image,return_tensors='pt').pixel_values.to(device)
   pred=processor.batch_decode(model.generate(pixels,max_new_tokens=20),skip_special_tokens=True)[0].strip()
   correct+=pred==label
  expected={'Document_20250512_0060.jpg':'175000','Invoice_20240316_0035.jpg':'45700','Invoice_20250512_0032.jpg':'550'}
  for case in AMOUNTS:
   if case[3]!='heldout':continue
   pixels=processor(images=image_for(case),return_tensors='pt').pixel_values.to(device)
   pred=processor.batch_decode(model.generate(pixels,max_new_tokens=20),skip_special_tokens=True)[0].strip()
   rows.append({'file':case[0],'expected_visible_digits_provisional':expected[case[0]],'prediction':pred,'match':pred==expected[case[0]]})
 output=ROOT/'artifacts'/'orand_amount_pilot';output.mkdir(parents=True,exist_ok=True)
 if args.steps: torch.save({n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},output/f'trocr_{args.run_name}_research_only.pt')
 report={'license':'ORAND-CAR-2014 CC BY-NC-ND 4.0, non-commercial research','base':'microsoft/trocr-base-handwritten','steps':args.steps,'decoder_layers_trained':args.layers,'training_images_available':len(train),'training_images_seen':len(used),'validation_images':100,'validation_exact':correct,'validation_scope':'random subset from original training split, not writer-independent','receipt_results':rows,'warning':'Not for commercial deployment; provisional labels, manually located crops; receipts previously examined during development'}
 (output/f'trocr_{args.run_name}_report.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
