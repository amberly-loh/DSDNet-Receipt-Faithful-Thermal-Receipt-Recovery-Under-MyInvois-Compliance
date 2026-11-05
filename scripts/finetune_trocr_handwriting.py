"""Tiny experimental TrOCR adaptation, with three receipts held out."""
from pathlib import Path
import argparse, json, random, sys
import torch
from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tmp' / 'trocr_deps'))
sys.path.insert(0, str(ROOT / 'scripts'))
from train_handwritten_date_pilot import CASES, DATA
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

AMOUNTS = [
 ('Document_20240511_0008.jpg',(.78,.83,.98,.89),'340.00','train'),
 ('Document_20240511_0070.jpg',(.78,.82,.98,.89),'116.00','train'),
 ('Document_20240511_0081.jpg',(.78,.82,.98,.89),'45.00','train'),
 ('Document_20250512_0027.jpg',(.78,.88,.97,.95),'20.00','train'),
 ('Document_20250512_0041.jpg',(.78,.82,.97,.90),'480.00','train'),
 ('Document_20250512_0042.jpg',(.75,.84,.98,.91),'262.00','train'),
 ('Document_20250512_0060.jpg',(.70,.82,.97,.88),'1750.00','heldout'),
 ('Invoice_20240316_0035.jpg',(.78,.84,.98,.92),'457.00','heldout'),
 ('Invoice_20250512_0032.jpg',(.75,.73,.97,.80),'550.00','heldout'),
]

def image_for(case, augment=False):
    name, (x1,y1,x2,y2), _, _ = case
    with Image.open(DATA/name) as src:
        w,h=src.size
        image=src.convert('RGB').crop((int(x1*w),int(y1*h),int(x2*w),int(y2*h)))
    if augment:
        image=image.rotate(random.uniform(-1.5,1.5),fillcolor='white')
        image=ImageEnhance.Contrast(image).enhance(random.uniform(.85,1.15))
    return image

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--field',choices=('date','amount'),default='date'); args=parser.parse_args()
    cases=CASES if args.field=='date' else AMOUNTS
    random.seed(56); torch.manual_seed(56); torch.set_num_threads(2)
    path=ROOT/'tmp'/'trocr_model'
    processor=TrOCRProcessor.from_pretrained(str(path),local_files_only=True)
    model=VisionEncoderDecoderModel.from_pretrained(str(path),local_files_only=True)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    device='cuda' if torch.cuda.is_available() else 'cpu'
    for p in model.parameters(): p.requires_grad=False
    # Only the last decoder layer and projection are updated; all other weights
    # stay fixed, so this fits the available GPU and remains experimental.
    for p in model.decoder.model.decoder.layers[-1].parameters(): p.requires_grad=True
    for p in model.decoder.output_projection.parameters(): p.requires_grad=True
    model.to(device); model.train()
    optim=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-5)
    train=[c for c in cases if c[3]=='train']
    steps=400
    for step in range(steps):
        case=random.choice(train)
        pixel=processor(images=image_for(case,True),return_tensors='pt').pixel_values.to(device)
        labels=processor.tokenizer(case[2],return_tensors='pt').input_ids.to(device)
        loss=model(pixel_values=pixel,labels=labels).loss
        optim.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1);optim.step()
        if (step+1)%40==0: print(f'step {step+1}: loss {loss.item():.4f}',flush=True)
    model.eval();rows=[]
    with torch.inference_mode():
        for case in cases:
            pixel=processor(images=image_for(case),return_tensors='pt').pixel_values.to(device)
            prediction=processor.batch_decode(model.generate(pixel,max_new_tokens=24),skip_special_tokens=True)[0].strip()
            rows.append({'file':case[0],'split':case[3],'expected':case[2],'prediction':prediction,'correct':prediction==case[2]})
            print(f'{case[3]} {case[0]}: {prediction} (expected {case[2]})',flush=True)
    output=ROOT/'artifacts'/'handwriting_trocr_pilot';output.mkdir(parents=True,exist_ok=True)
    changed={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad}
    torch.save(changed,output/f'{args.field}_decoder_adaptation.pt')
    (output/f'{args.field}_finetuned_results.json').write_text(json.dumps({'field':args.field,'steps':steps,'base_revision':'eaacaf452b06415df8f10bb6fad3a4c11e609406','labels':'provisional normalized values','warning':'Small experiment, not approved for app use','results':rows},indent=2),encoding='utf-8')

if __name__=='__main__': main()
