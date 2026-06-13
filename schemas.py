# schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple

class Detection(BaseModel):
    box: Tuple[float, float, float, float]
    score: float = Field(ge=0, le=1)
    class_id: int = Field(ge=0)
    class_name: str
    track_id: Optional[int] = None
    mask_rle: Optional[str] = None

class Classification(BaseModel):
    detection_index: int
    class_id: int
    class_name: str
    score: float = Field(ge=0, le=1)

class PipelineResult(BaseModel):
    image_id: str
    detections: List[Detection]
    classifications: List[Classification]
    inference_ms: float

# NEW: Contract for video processing
class VideoJobResult(BaseModel):
    video_id: str
    total_frames: int
    processed_frames: int
    fps: float
    output_video_path: str
    total_time_sec: float