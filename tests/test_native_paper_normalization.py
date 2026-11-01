import cv2
import numpy as np

from dsdnet_receipt.restoration import make_complete_document_view
from dsdnet_receipt.restoration import sharpen_measured_print_edges
from dsdnet_receipt.restoration import make_faithful_content_view


def test_small_print_keeps_strokes_while_suppressing_broad_fold_shading():
    y, x = np.mgrid[:320, :168]
    paper = 228 - 35 * np.exp(-((y - 130 - x * .2) / 13) ** 2)
    source = paper.round().astype(np.uint8)
    ink = np.zeros_like(source)
    cv2.putText(ink, 'REF 012345', (8, 80), cv2.FONT_HERSHEY_SIMPLEX, .36, 255, 1)
    cv2.putText(ink, 'RM 55.51', (8, 190), cv2.FONT_HERSHEY_SIMPLEX, .36, 255, 1)
    source[ink > 0] -= 55
    result = make_complete_document_view(source, thermal=True)
    fold = (y > 130) & (y < 145) & (x > 20) & (x < 140)
    assert float(result[fold].mean()) > 245
    assert float(result[ink > 0].mean()) < float(source[ink > 0].mean()) - 30
    assert result.shape == source.shape


def test_edge_sharpening_keeps_flat_paper_and_does_not_close_white_gaps():
    source = np.full((160, 200), 255, np.uint8)
    for y in range(20, 150, 20):
        cv2.putText(source, 'REF 012345', (5, y), 0, .4, 65, 1)
    source = cv2.GaussianBlur(source, (0, 0), .55)
    result = sharpen_measured_print_edges(source)
    assert np.array_equal(result[:, 140:], source[:, 140:])
    assert np.all(result[source == 255] == 255)
    assert float(result[source < 180].mean()) < float(source[source < 180].mean())


def test_white_canvas_does_not_keep_underexposed_paper_dark():
    source=np.full((400,300),255,np.uint8)
    source[40:360,35:265]=125
    for y in range(80,330,30):
        cv2.putText(source,'REF 12.50',(50,y),0,.45,65,1)
    result=make_complete_document_view(source,thermal=True)
    assert float(result[200:220,190:220].mean())>240
    assert result[80:330,50:180].min()<100


def test_extremely_weak_dark_capture_preserves_source_ordering():
    source=np.full((320,280),130,np.uint8)
    for y in range(40,300,30):
        cv2.putText(source,'REF 12.50',(20,y),0,.4,125,1)
    result=make_faithful_content_view(source,source,[])
    assert float(result[source==130].mean())>=245
    assert result[source==125].max()<result[source==130].min()-5
