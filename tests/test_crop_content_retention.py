import cv2
import numpy as np
import json
from dsdnet_receipt.preprocessing import _rectify_receipt,_crop_retains_marks
from dsdnet_receipt.preprocessing import _content_crop
from dsdnet_receipt.preprocessing import preprocess


def test_table_rectangle_is_not_allowed_to_crop_off_header_or_signature():
    image=np.full((700,500),235,np.uint8)
    cv2.putText(image,'COMPANY NAME',(30,65),0,.8,40,2)
    cv2.rectangle(image,(30,170),(470,610),30,2)
    for y in range(220,600,50):
        cv2.line(image,(30,y),(470,y),30,1)
    assert not _crop_retains_marks(image,np.array([[30,170],[470,170],[470,610],[30,610]]))
    result,record=_rectify_receipt(image)
    assert record['status']!='rectified'
    assert np.array_equal(result,image)


def test_whole_page_candidate_keeps_handwriting_and_all_scripts():
    image=np.full((300,200),230,np.uint8)
    cv2.rectangle(image,(10,10),(190,290),80,1)
    cv2.line(image,(30,30),(70,80),50,2)
    assert _crop_retains_marks(image,np.array([[1,1],[198,1],[198,298],[1,298]]))


def test_content_crop_keeps_large_logo_outside_small_text_components():
    image=np.full((700,500),240,np.uint8)
    cv2.rectangle(image,(10,10),(420,95),35,-1)
    for y in range(280,600,35):
        cv2.putText(image,'ITEM AMOUNT',(180,y),0,.5,65,1)
    result,metadata=_content_crop(image)
    assert metadata['status']=='rejected_content_loss'
    assert np.array_equal(result,image)


def test_cropped_receipt_metadata_can_be_saved_as_json():
    image=np.full((500,300),245,np.uint8)
    for y in range(80,400,30):
        cv2.putText(image,'TOTAL 12.50',(40,y),0,.5,65,1)
    _,metadata=preprocess(image,{'preprocessing':{'max_long_edge':1600}})
    assert json.loads(json.dumps(metadata))['boundary_detection']
