"""Source-bounded use of the development low-resolution edge checkpoint."""
from functools import lru_cache
from pathlib import Path
from threading import Lock
import hashlib
import cv2
import numpy as np

_inference_lock = Lock()


def combine_supported_edges(base: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Keep clean paper exact; permit bounded changes only on measured ink.

    This restriction deliberately prevents the model filling white gaps. It
    cannot promise recovery of information absent from the input.
    """
    if base.shape != predicted.shape:
        raise ValueError('Refinement and source view must share geometry')
    b = base.astype(np.float32)
    weight = np.clip((235.0 - b) / 50.0, 0, 1)
    change = np.clip((predicted.astype(np.float32) - b) * .7, -45, 15)
    return np.clip(b + weight * change, 0, 255).round().astype(np.uint8)


@lru_cache(maxsize=1)
def _load(checkpoint: str, source_path: str, expected_hash: str = ''):
    import torch
    from .dsdnet_adapter import DSDNetReceiptAdapter
    digest=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    if expected_hash and digest!=expected_hash:
        raise ValueError('Native refinement checkpoint integrity check failed')
    model = DSDNetReceiptAdapter(source_path)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu')['model'])
    return model.eval(),digest


def refine_native_print(source: np.ndarray, base: np.ndarray, config: dict) -> tuple[np.ndarray, dict]:
    import torch
    import torch.nn.functional as F
    from .dsdnet_adapter import gaussian_kernel
    audit = {'applied': False, 'synthetic_glyph_rendering': False}
    if (not config.get('enabled') or source.size > 200000 or min(source.shape) >= 250
            or source.shape[1] / source.shape[0] >= .7
            or float(np.percentile(source,85)-np.percentile(source,5)) >= 90):
        return base, audit
    checkpoint = str(Path(config['checkpoint']).resolve())
    if not Path(checkpoint).is_file():
        return base, {**audit, 'reason': 'checkpoint unavailable'}
    # Source-based interpolation is explicit; the model refines at 2x but the
    # input capture still has its original information limit.
    enlarged = cv2.resize(source, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    reference = cv2.resize(base, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    h,w = enlarged.shape
    tensor = torch.from_numpy(enlarged.astype(np.float32)/255)[None,None]
    tensor = F.pad(tensor,(0,(-w)%64,0,(-h)%64),mode='replicate')
    try:
        with _inference_lock, torch.inference_mode():
            model,digest = _load(checkpoint, config['source_path'],config.get('expected_sha256',''))
            # Prefer measured available GPU capacity, not a second permanently
            # resident model. Oversized/contended work retains the CPU path.
            required=max(1.8*1024**3,enlarged.size*5000+.5*1024**3)
            device='cuda' if torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>required else 'cpu'
            try:
                model.to(device)
                try:
                    prediction=model(tensor.to(device),gaussian_kernel(0).to(device),retain_diagnostics=False)['restored']
                except torch.cuda.OutOfMemoryError:
                    model.cpu()
                    torch.cuda.empty_cache()
                    device='cpu'
                    prediction=model(tensor,gaussian_kernel(0),retain_diagnostics=False)['restored']
                prediction=(prediction[0,0,:h,:w].clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)
            finally:
                model.cpu()
        pixels = prediction.astype(np.float32)
        paper = cv2.morphologyEx(pixels,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
        paper = cv2.GaussianBlur(paper,(0,0),1)
        paper = np.maximum(paper,float(np.percentile(pixels,85))*.85)
        darkness = np.maximum(250-pixels/np.maximum(paper,1)*250,0)
        darkness = np.maximum(darkness-12*(1-np.exp(-darkness/12)),0)
        toned = np.clip(255-darkness*3,0,255).round().astype(np.uint8)
        output = combine_supported_edges(reference,toned)
        audit.update(applied=True, checkpoint=checkpoint, checkpoint_sha256=digest, device=device, model_output_scale=2,
                     background_policy='pixels >=235 unchanged from interpolated source-based view',
                     change_bounds={'maximum_darkening':45,'maximum_lightening':15},
                     scope='source-bounded development checkpoint; not pristine reconstruction')
        return output,audit
    except (RuntimeError, ValueError, OSError) as error:
        return base,{**audit,'reason':str(error)}
