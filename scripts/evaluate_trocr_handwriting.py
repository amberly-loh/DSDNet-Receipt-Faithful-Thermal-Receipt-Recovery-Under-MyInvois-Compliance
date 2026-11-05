"""Evaluate pretrained TrOCR on provisional, manually located date crops."""
from pathlib import Path
import json, sys
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tmp' / 'trocr_deps'))
sys.path.insert(0, str(ROOT / 'scripts'))
from train_handwritten_date_pilot import CASES, DATA
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

MODEL = ROOT / 'tmp' / 'trocr_model'

def main():
    processor = TrOCRProcessor.from_pretrained(str(MODEL), local_files_only=True)
    model = VisionEncoderDecoderModel.from_pretrained(str(MODEL), local_files_only=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval()
    rows = []
    for filename, (x1,y1,x2,y2), expected, split in CASES:
        with Image.open(DATA / filename) as source:
            w,h = source.size
            image = source.convert('RGB').crop((int(x1*w),int(y1*h),int(x2*w),int(y2*h)))
        pixels = processor(images=image, return_tensors='pt').pixel_values.to(device)
        with torch.inference_mode():
            generated = model.generate(pixels, max_new_tokens=24)
        prediction = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        rows.append({'file':filename, 'expected':expected, 'prediction':prediction, 'split':split})
        print(f'{filename}: expected {expected}, predicted {prediction}', flush=True)
    output = ROOT / 'artifacts' / 'handwriting_trocr_pilot'
    output.mkdir(parents=True,exist_ok=True)
    (output / 'pretrained_results.json').write_text(json.dumps({'model':'microsoft/trocr-base-handwritten', 'revision':'eaacaf452b06415df8f10bb6fad3a4c11e609406', 'manually_located':True,'provisional_labels':True,'results':rows},indent=2),encoding='utf-8')

if __name__ == '__main__': main()
