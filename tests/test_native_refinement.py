import numpy as np
import pytest
from dsdnet_receipt.native_refinement import combine_supported_edges, refine_native_print, _load


def test_model_cannot_add_ink_to_clean_paper_or_erase_dark_strokes():
    base = np.array([[255,250,235,230,180,90,0]],dtype=np.uint8)
    dark = combine_supported_edges(base,np.zeros_like(base))
    bright = combine_supported_edges(base,np.full_like(base,255))
    assert np.array_equal(dark[base>=235],base[base>=235])
    assert np.array_equal(bright[base>=235],base[base>=235])
    assert np.all(base.astype(int)-dark.astype(int)<=45)
    assert np.all(bright.astype(int)-base.astype(int)<=15)
    assert bright[0,-2] <= 105


def test_refinement_rejects_mismatched_geometry():
    with pytest.raises(ValueError):
        combine_supported_edges(np.zeros((4,4),np.uint8),np.zeros((5,4),np.uint8))


def test_disabled_refinement_never_loads_weights():
    source = np.full((350,168),240,np.uint8)
    result,record = refine_native_print(source,source,{'enabled':False})
    assert np.array_equal(result,source)
    assert record['applied'] is False


def test_checkpoint_mismatch_is_rejected_before_model_loading(tmp_path):
    checkpoint=tmp_path/'weights.pt'
    checkpoint.write_bytes(b'not a trusted model')
    with pytest.raises(ValueError,match='integrity'):
        _load(str(checkpoint),'missing architecture','0'*64)
