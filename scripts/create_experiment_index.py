"""Index completed evaluation artefacts without asserting formal conformance."""
from pathlib import Path
import hashlib,json,platform,sys
from datetime import datetime,timezone
import torch
root=Path(__file__).resolve().parents[1]
sources={
 'validation_summary':'artifacts/validation_restoration/exact_excluded_6000_full_gpu/summary.json',
 'validation_records':'artifacts/validation_restoration/exact_excluded_6000_full_gpu/records.csv',
 'validation_statistics':'artifacts/validation_restoration/exact_excluded_6000_full_gpu/paired_significance.json',
 'validation_provenance':'artifacts/validation_restoration/exact_excluded_6000_full_gpu/provenance_and_performance.json',
 'exploratory_test_summary':'artifacts/validation_restoration/exact_excluded_6000_exploratory_test/summary.json',
 'exploratory_test_records':'artifacts/validation_restoration/exact_excluded_6000_exploratory_test/records.csv',
 'exploratory_test_provenance':'artifacts/validation_restoration/exact_excluded_6000_exploratory_test/provenance.json',
 'data_partition_audit':'artifacts/training_candidate_20261102/dataset_partition_audit.json',
 'near_duplicate_candidates':'artifacts/training_candidate_20261102/near_duplicate_candidate_audit.json',
 'split_preparation':'artifacts/protocol_lock/exact_duplicate_excluded_20261102/preparation_record.json',
 'synthetic_e2e':'artifacts/e2e_synthetic_20261102/results.json',
 'usability_static':'artifacts/training_candidate_20261102/usability_static_audit.json',
 'regression_junit':'artifacts/training_candidate_20261102/post_export_fix_regression.xml',
 'regression_console':'artifacts/training_candidate_20261102/post_export_fix_regression.log',
 'image_ocr_field_results':'artifacts/training_candidate_20261102/field_image_ocr_validation_148.json',
 'field_reference_results':'artifacts/training_candidate_20261102/field_reference_validation.json',
 'ocr_confidence_analysis':'artifacts/training_candidate_20261102/ocr_confidence_coverage_exploratory.json',
 'ocr_confidence_predictions':'artifacts/training_candidate_20261102/field_predictions_validation_20261102.jsonl',
 'ocr_confidence_analysis_code':'scripts/analyse_ocr_confidence_coverage_exploratory.py',
 'training_resume_manifest':'artifacts/training_exact_excluded_resume_12000_20261102/run_manifest.json',
 'current_application_code':'src/dsdnet_receipt/web_app.py',
 'security_tests':'tests/test_evaluation_security.py',
}
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
files={k:{'path':str(root/v),'sha256':digest(root/v)} for k,v in sources.items()}
report={'generated_utc':datetime.now(timezone.utc).isoformat(),'environment':{'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},'files':files,'status':{'functional_and_integration':'automated synthetic checks completed; not exhaustive','security_privacy':'local regression checks completed; no independent security audit','usability':'static checks completed; no human participant study','performance':'synthetic 3-receipt CPU smoke and fixed-crop timings; no production load benchmark','baselines':'85-source validation plus previously inspected 92-source exploratory test at 6000 iterations','training':'one-million-iteration run ongoing, not complete','risk_coverage':'exploratory OCR confidence analysis completed; source-support threshold and independent validation not established'},'warning':'Do not represent exploratory results as independent confirmation or completed formal protocol.'}
out=root/'artifacts/training_candidate_20261102/experiment_index_20261102.json';out.write_text(json.dumps(report,indent=2),encoding='utf-8');print(out)
