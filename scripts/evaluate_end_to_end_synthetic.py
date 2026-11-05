"""Small, isolated synthetic upload-to-export performance/integration smoke test."""
import io,json,os,sys,time,traceback
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dsdnet_receipt.config import load_config
from dsdnet_receipt.web_app import create_app
OUT=ROOT/'artifacts/e2e_synthetic_20261102'
OUT.mkdir(parents=True,exist_ok=True)
config,_=load_config(ROOT/'configs/default.json')
config['artifacts_root']=str(OUT/'sandbox_artifacts')
config_path=OUT/'config_snapshot.json';config_path.write_text(json.dumps(config,indent=2),encoding='utf-8')
report={'scope':'synthetic local single-client integration and latency smoke test; not a human usability study','cases':[],'errors':[]}
try:
 app=create_app(config_path,testing=True,database_path=OUT/'synthetic.sqlite3')
 client=app.test_client(); token=client.get('/api/auth').get_json()['csrf_token']
 session=client.post('/api/guest',headers={'X-CSRF-Token':token})
 if session.status_code!=201:raise RuntimeError(f'Guest failed: {session.status_code}')
 token=session.get_json()['csrf_token']
 for index in range(3):
  img=np.full((500,340,3),255,np.uint8)
  for y,text in [(70,'TEST HARDWARE'),(120,'DATE 02/11/2024'),(170,'INVOICE T100'),(220,'TOTAL RM 12.50')]:
   cv2.putText(img,text,(12,y),cv2.FONT_HERSHEY_SIMPLEX,.60,(0,0,0),1,cv2.LINE_AA)
  if index==1: img=cv2.GaussianBlur(img,(3,3),.6)
  if index==2: img=cv2.convertScaleAbs(img,alpha=.65,beta=80)
  success,encoded=cv2.imencode('.png',img)
  if not success:raise RuntimeError('PNG encoding failed')
  start=time.perf_counter()
  response=client.post('/api/analyze',data={'receipt':(io.BytesIO(encoded.tobytes()),f'synthetic_{index}.png')},headers={'X-CSRF-Token':token},content_type='multipart/form-data')
  elapsed=time.perf_counter()-start
  body=response.get_json(silent=True) or {}
  record={'case':index,'elapsed_wall_seconds':elapsed,'http_status':response.status_code,'restoration_status':body.get('restoration_status'),'ocr_nonempty':bool(body.get('ocr_text'))}
  if response.status_code==200:
   rid=body['receipt']['public_id']
   record['detail_status']=client.get('/api/receipts/'+rid).status_code
   record['xlsx_status']=client.get('/api/receipts/'+rid+'/ocr.xlsx').status_code
  else: record['error']=body.get('error')
  report['cases'].append(record)
  print(json.dumps(record),flush=True)
except Exception as error:
 report['errors'].append({'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
finally:
 report['note']='Times are for synthetic images in a local test client, include concurrent workload and cold start, and do not establish real-user performance.'
 (OUT/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print('Report:',OUT/'results.json',flush=True)
if report['errors'] or any(r['http_status']!=200 for r in report['cases']):sys.exit(1)
