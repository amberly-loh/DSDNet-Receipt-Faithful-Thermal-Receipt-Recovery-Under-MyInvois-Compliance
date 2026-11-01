import numpy as np
from dsdnet_receipt.ocr import PaddleOCRAdapter,OCRResult


def test_identical_ocr_views_are_recognized_once_without_losing_text(monkeypatch):
    adapter=object.__new__(PaddleOCRAdapter)
    adapter.version='probe'
    calls=[]
    def recognize(image):
        calls.append(image)
        return OCRResult('TOTAL 12.50',[{'text':'TOTAL 12.50','confidence':.9,'box':[[0,0],[5,0],[5,5],[0,5]]}],'probe','ok')
    monkeypatch.setattr(adapter,'run',recognize)
    source=np.full((20,20),240,np.uint8)
    result=adapter.run_views([('original',source),('source_fallback',source.copy())])
    assert len(calls)==1
    assert result.text=='TOTAL 12.50'
