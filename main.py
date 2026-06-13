# main.py
import io
import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models import convnext_tiny

from pipeline import VisionPipeline
from schemas import PipelineResult

app = FastAPI(title="Production Vision Pipeline API")
pipe = None

COMPLETED_JOBS = {}

os.makedirs("static/videos", exist_ok=True)

@app.on_event("startup")
def load_models():
    global pipe
    print("Loading deep learning models into system memory...")
    detector = maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
    classifier = convnext_tiny(weights="DEFAULT")
    
    COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    imagenet_classes = [f"imagenet_class_{i}" for i in range(1000)]
    pipe = VisionPipeline(detector, classifier, COCO_INSTANCE_CATEGORY_NAMES, imagenet_classes)
    print("Inference engine ready.")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.post("/detect", response_model=PipelineResult)
async def detect_endpoint(file: UploadFile):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="Unsupported image format.")
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Corrupt image data.")
    return pipe.run(img, image_id=file.filename or "upload")

def process_video_background(job_id: str, input_path: str, output_path: str):
    """This function runs silently in the background without blocking the web server."""
    try:
        result = pipe.process_video(input_path, output_path, video_id=job_id)
        
        result.output_video_path = f"/static/videos/output_{job_id}.mp4"
        COMPLETED_JOBS[job_id] = result
    except Exception as e:
        COMPLETED_JOBS[job_id] = {"error": str(e)}
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

@app.post("/start-tracking")
def start_tracking_endpoint(file: UploadFile, background_tasks: BackgroundTasks):
    is_video = file.content_type and file.content_type.startswith("video/")
    if not is_video and file.filename:
        is_video = file.filename.lower().endswith(('.mp4', '.webm', '.mov', '.avi'))
        
    if not is_video:
        raise HTTPException(status_code=400, detail="Must upload a video file.")
    
    job_id = str(uuid.uuid4())[:8]
    input_path = f"static/videos/input_{job_id}.mp4"
    output_path = f"static/videos/output_{job_id}.mp4"
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    background_tasks.add_task(process_video_background, job_id, input_path, output_path)

    return {"job_id": job_id, "status": "started"}

@app.get("/status/{job_id}")
def get_job_status(job_id: str):
    """The browser calls this every 2 seconds to check on the background task."""

    if job_id in COMPLETED_JOBS:
        data = COMPLETED_JOBS[job_id]
        if isinstance(data, dict) and "error" in data:
            return {"status": "error", "detail": data["error"]}
        return {"status": "completed", "result": data}
    
    if pipe and "video_progress" in pipe.metrics_log:
        progress = pipe.metrics_log["video_progress"].get(job_id)
        if progress:
            return {"status": "processing", "progress": progress}
            
    return {"status": "starting"}

app.mount("/static", StaticFiles(directory="static"), name="static")