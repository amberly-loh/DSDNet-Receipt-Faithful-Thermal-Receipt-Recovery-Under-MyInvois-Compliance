"""Focused local privacy and security regression checks with synthetic data."""
import json
import io
import csv
from pathlib import Path
import openpyxl
from dsdnet_receipt.storage import utc_now
from dsdnet_receipt.web_app import create_app

CONFIG=Path(__file__).resolve().parents[1]/'configs/default.json'

def guest(client):
    token=client.get('/api/auth').get_json()['csrf_token']
    response=client.post('/api/guest',headers={'X-CSRF-Token':token})
    assert response.status_code==201
    return response.get_json()['user'],response.get_json()['csrf_token']

def test_security_headers_and_anonymous_access(tmp_path):
    app=create_app(CONFIG,testing=True,database_path=tmp_path/'test.sqlite3')
    client=app.test_client()
    response=client.get('/api/receipts')
    assert response.status_code==401
    assert response.headers['X-Content-Type-Options']=='nosniff'
    assert response.headers['X-Frame-Options']=='DENY'
    assert response.headers['Cache-Control']=='no-store'
    assert "frame-ancestors 'none'" in response.headers['Content-Security-Policy']
    assert app.config['SESSION_COOKIE_HTTPONLY'] is True
    assert app.config['SESSION_COOKIE_SAMESITE']=='Lax'

def test_cross_user_data_cannot_be_read_changed_or_deleted(tmp_path):
    app=create_app(CONFIG,testing=True,database_path=tmp_path/'test.sqlite3')
    owner=app.test_client(); other=app.test_client()
    user,owner_token=guest(owner); _,other_token=guest(other)
    source=tmp_path/'synthetic.png'; source.write_bytes(b'synthetic data')
    now=utc_now()
    app.extensions['receipt_store'].create_receipt({'public_id':'EVAL-PRIVATE-001','user_id':user['id'],'source_name':'synthetic.png','created_at':now,'updated_at':now,'status':'needs_review','merchant':'Test Shop','transaction_date':'2024-01-01','receipt_number':'SYNTHETIC','total_amount':'5.00','tags':'[]','notes':'private synthetic marker','ocr_text':'private synthetic marker','audit_json':json.dumps({}),'input_path':str(source),'restored_path':str(source),'runtime_seconds':0,'restoration_status':'ok','support_status':'unavailable'})
    assert owner.get('/api/receipts/EVAL-PRIVATE-001').status_code==200
    assert other.get('/api/receipts/EVAL-PRIVATE-001').status_code==404
    assert other.get('/api/receipts/EVAL-PRIVATE-001/image/original').status_code==404
    assert other.patch('/api/receipts/EVAL-PRIVATE-001',json={'merchant':'Intruder'},headers={'X-CSRF-Token':other_token}).status_code==404
    assert other.delete('/api/receipts/EVAL-PRIVATE-001',headers={'X-CSRF-Token':other_token}).status_code==404
    assert b'private synthetic marker' not in other.get('/api/export.csv').data
    assert other.get('/api/admin/overview').status_code==403
    assert owner.get('/api/receipts/EVAL-PRIVATE-001').get_json()['merchant']=='Test Shop'

def test_passwords_stored_hashed(tmp_path):
    app=create_app(CONFIG,testing=True,database_path=tmp_path/'test.sqlite3')
    client=app.test_client(); token=client.get('/api/auth').get_json()['csrf_token']
    password='uniquely strong local example 2026'
    result=client.post('/api/register',json={'name':'Example User','email':'example@valid-domain.my','password':password,'password_confirmation':password,'policy_accepted':True},headers={'X-CSRF-Token':token})
    assert result.status_code==201
    account=app.extensions['receipt_store'].user_by_email('example@valid-domain.my')
    assert account['password_hash']!=password
    assert account['password_hash'].startswith('scrypt:')

def test_formula_like_export_values_are_written_as_text(tmp_path):
    app=create_app(CONFIG,testing=True,database_path=tmp_path/'test.sqlite3')
    client=app.test_client(); user,_=guest(client)
    source=tmp_path/'synthetic.png'; source.write_bytes(b'synthetic')
    now=utc_now()
    app.extensions['receipt_store'].create_receipt({'public_id':'EVAL-FORMULA-001','user_id':user['id'],'source_name':'synthetic.png','created_at':now,'updated_at':now,'status':'needs_review','merchant':'  =1+1','transaction_date':'2024-01-01','receipt_number':'SYNTHETIC','total_amount':'5.00','tags':'[]','notes':'\t@SYNTHETIC','ocr_text':'','audit_json':'{}','input_path':str(source),'restored_path':str(source),'runtime_seconds':0,'restoration_status':'ok','support_status':'unavailable'})
    csv_response=client.get('/api/export.csv')
    assert csv_response.status_code==200
    rows=list(csv.DictReader(io.StringIO(csv_response.data.decode('utf-8-sig'))))
    assert any(value.startswith("'  =") for value in rows[0].values() if value)
    assert any(value.startswith("'\t@") for value in rows[0].values() if value)
    workbook=openpyxl.load_workbook(io.BytesIO(client.get('/api/export.xlsx').data))
    values=[cell for row in workbook.active.iter_rows(min_row=2) for cell in row if isinstance(cell.value,str) and cell.value.startswith("'")]
    assert any(cell.value.startswith("'  =") and cell.data_type=='s' for cell in values)
    assert any(cell.value.startswith("'\t@") and cell.data_type=='s' for cell in values)
