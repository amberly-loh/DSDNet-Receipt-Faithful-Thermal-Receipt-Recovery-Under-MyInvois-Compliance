"""End-to-end native worker probe in an isolated database, not user history."""
import json
import sqlite3
import sys
import uuid
import argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--batch',action='store_true');args=parser.parse_args()
    from dsdnet_receipt.storage import ReceiptStore
    from dsdnet_receipt.inference_worker import IsolatedReceiptWorker
    folder=ROOT/'artifacts/release_audit';folder.mkdir(exist_ok=True)
    path=folder/'worker_probe.sqlite3'
    store=ReceiptStore(path)
    marker=uuid.uuid4().hex
    user=store.create_user(marker,'Worker Probe',marker+'@example.invalid','disabled-test-account')
    db=sqlite3.connect(ROOT/'artifacts/app_data/receipts.sqlite3')
    names=['Screenshot 2026-07-13 080718.png']
    if args.batch:
        names += ['Invoice_20240316_0039.jpg','Invoice_20240316_0035.jpg','Document_20250512_0027.jpg',
                  'Invoice_20241101_0003.jpg','Invoice_20241101_0021.jpg','Document_20240511_0052.jpg','Receipt_20240316_0053.jpg']
    worker=IsolatedReceiptWorker(timeout=300)
    try:
        reports=[]
        for name in names:
            source=db.execute('SELECT input_path FROM receipts WHERE source_name=? ORDER BY id DESC LIMIT 1',(name,)).fetchone()[0]
            try:
                result=worker.run({'bytes':Path(source).read_bytes(),'filename':name,'mimetype':'image/png',
                    'config':str(ROOT/'configs/default.json'),'database':str(path),'user':user})
                report={'source':name,'saved_in_isolated_database':bool(result.get('receipt')),'runtime_seconds':result.get('runtime_seconds'),
                    'native_refinement':result['audit'].get('native_refinement'),'ocr_characters':len(result.get('ocr_text',''))}
            except Exception as error:
                report={'source':name,'error':str(error)}
            reports.append(report)
            print(json.dumps(report),flush=True)
            (folder/('worker_batch_probe.json' if args.batch else 'worker_probe.json')).write_text(json.dumps(reports if args.batch else report,indent=2))
    finally:worker.close()

if __name__=='__main__':main()
