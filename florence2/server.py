import sys, types, importlib.util, gc, io, base64, logging
from contextlib import asynccontextmanager
from PIL import Image
import torch, numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("florence2")

# ── torchvision mock ──────────────────────────────────────────────────────────
def _make_mock(name):
    m = types.ModuleType(name)
    m.__spec__ = importlib.util.spec_from_loader(name, loader=None)
    m.__path__ = []; m.__package__ = name
    return m

class _MockObj:
    def __getattr__(self, n): return _MockObj()
    def __call__(self, *a, **k): return _MockObj()

class FrozenBatchNorm2d(_MockObj): pass

for name in ["torchvision","torchvision.models","torchvision.models.feature_extraction",
             "torchvision.ops","torchvision.ops.misc","torchvision.ops.boxes",
             "torchvision.transforms","torchvision.transforms.functional"]:
    sys.modules[name] = _make_mock(name)

tv = sys.modules["torchvision"]
tv.models = sys.modules["torchvision.models"]
tv.models.feature_extraction = sys.modules["torchvision.models.feature_extraction"]
tv.models.feature_extraction.create_feature_extractor = lambda *a,**k: None
tv.ops = sys.modules["torchvision.ops"]
tv.ops.misc = sys.modules["torchvision.ops.misc"]
tv.ops.misc.FrozenBatchNorm2d = FrozenBatchNorm2d
tv.ops.boxes = sys.modules["torchvision.ops.boxes"]
tv.transforms = sys.modules["torchvision.transforms"]
tv.transforms.functional = sys.modules["torchvision.transforms.functional"]

def _batched_nms(boxes, scores, idxs, iou_threshold):
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long)
    keep_all = []
    for cls_id in torch.unique(idxs):
        mask = idxs == cls_id
        idx_map = torch.where(mask)[0]
        b, s = boxes[mask], scores[mask]
        order = s.argsort(descending=True)
        b, s, idx_map = b[order], s[order], idx_map[order]
        keep = []
        while len(b):
            keep.append(idx_map[0].item())
            if len(b) == 1: break
            xx1 = torch.maximum(b[0,0], b[1:,0])
            yy1 = torch.maximum(b[0,1], b[1:,1])
            xx2 = torch.minimum(b[0,2], b[1:,2])
            yy2 = torch.minimum(b[0,3], b[1:,3])
            inter = (xx2-xx1).clamp(0) * (yy2-yy1).clamp(0)
            area0 = (b[0,2]-b[0,0])*(b[0,3]-b[0,1])
            areas = (b[1:,2]-b[1:,0])*(b[1:,3]-b[1:,1])
            iou = inter / (area0 + areas - inter + 1e-6)
            mask2 = iou <= iou_threshold
            b, s, idx_map = b[1:][mask2], s[1:][mask2], idx_map[1:][mask2]
        keep_all.extend(keep)
    return torch.tensor(keep_all, dtype=torch.long)

def _box_area(boxes):
    return (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])

tv.ops.boxes.batched_nms = _batched_nms
tv.ops.boxes.box_area = _box_area
sys.modules["torchvision.ops.boxes"].batched_nms = _batched_nms
sys.modules["torchvision.ops.boxes"].box_area = _box_area

def _resize(img, size, interpolation=None, **kw):
    if isinstance(img, Image.Image):
        if isinstance(size, int): size = (size, size)
        return img.resize((size[1], size[0]), Image.BILINEAR)
    return img

def _to_pil_image(x, mode=None):
    if isinstance(x, torch.Tensor):
        arr = x.cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] in (1,3,4): arr = np.transpose(arr, (1,2,0))
        if arr.ndim == 3 and arr.shape[2] == 1: arr = arr[:,:,0]
        if arr.dtype != np.uint8:
            arr = (arr * 255).clip(0,255).astype(np.uint8)
        return Image.fromarray(arr)
    if isinstance(x, np.ndarray):
        if x.dtype != np.uint8:
            x = (x * 255).clip(0,255).astype(np.uint8)
        return Image.fromarray(x)
    return x

sys.modules["torchvision.transforms.functional"].resize = _resize
sys.modules["torchvision.transforms.functional"].to_pil_image = _to_pil_image

import transformers.dynamic_module_utils as dmu
_orig_check = dmu.check_imports
def _patched_check(f):
    try: return _orig_check(f)
    except ImportError as e:
        if "flash_attn" in str(e): return []
        raise
dmu.check_imports = _patched_check

# ── Model globals ─────────────────────────────────────────────────────────────
_processor = None
_model = None

def load_model():
    global _processor, _model
    from transformers import AutoProcessor, AutoModelForCausalLM
    log.info("Loading Florence-2-base...")
    _processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-base", trust_remote_code=True,
        torch_dtype=torch.float16, attn_implementation="eager").cuda()
    _model.eval()
    log.info("Florence-2 ready.")

def run_task(img: Image.Image, task: str, text: str = "") -> dict:
    W, H = img.size
    inputs = _processor(text=task + text, images=img, return_tensors="pt")
    inputs = {k: v.cuda().half() if hasattr(v,"cuda") and v.dtype==torch.float32
              else v.cuda() if hasattr(v,"cuda") else v for k,v in inputs.items()}
    with torch.no_grad():
        ids = _model.generate(**inputs, max_new_tokens=1024, num_beams=3)
    txt = _processor.batch_decode(ids, skip_special_tokens=False)[0]
    return _processor.post_process_generation(txt, task=task, image_size=(W, H))

# ── FastAPI app ───────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(lifespan=lifespan)

class PredictRequest(BaseModel):
    image: str          # base64-encoded JPEG/PNG
    task: str           # e.g. "<DENSE_REGION_CAPTION>", "<OPEN_VOCABULARY_DETECTION>"
    text: Optional[str] = ""  # extra text for OVD queries like ". trophy."

class PredictResponse(BaseModel):
    result: dict
    width: int
    height: int

@app.get("/health")
def health():
    return {"status": "ok", "model": "Florence-2-base"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        img_bytes = base64.b64decode(req.image)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    result = run_task(img, req.task, req.text or "")
    return PredictResponse(result=result, width=img.width, height=img.height)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8086)
