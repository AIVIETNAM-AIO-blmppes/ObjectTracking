# pipeline.py
import time
import json
import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from deep_sort_realtime.deepsort_tracker import DeepSort
from schemas import Detection, Classification, PipelineResult, VideoJobResult

class VisionPipeline:
    def __init__(self, detector, classifier, detector_classes, classifier_names, device="cpu", min_crop=32):
        self.detector = detector.to(device).eval()
        self.classifier = classifier.to(device).eval()
        self.detector_classes = detector_classes
        self.classifier_names = classifier_names
        self.device = device
        self.min_crop = min_crop
        
        self.tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)
        
        self.metrics_log = {"preprocess": [], "detect": [], "classify": [], "total": [], "video_progress": {}}

    def save_state(self, save_dir: str = "checkpoints"):
        """Safely serializes model weights, progress saves, and performance metrics."""
        path = Path(save_dir)
        path.mkdir(exist_ok=True)
        
        torch.save(self.detector.state_dict(), path / "detector_weights.pth")
        torch.save(self.classifier.state_dict(), path / "classifier_weights.pth")
        
        with open(path / "metrics_history.json", "w") as f:
            json.dump(self.metrics_log, f, indent=2)

    def preprocess(self, image):
        if isinstance(image, Image.Image):
            image = np.asarray(image.convert("RGB")).copy()
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return tensor.to(self.device)

    @torch.no_grad()
    def detect(self, image_tensor):
        return self.detector([image_tensor])[0]

    @torch.no_grad()
    def classify(self, crops):
        if len(crops) == 0:
            return []
        batch = torch.stack(crops).to(self.device)
        logits = self.classifier(batch)
        probs = logits.softmax(-1)
        scores, cls = probs.max(-1)
        return list(zip(cls.tolist(), scores.tolist()))
    
    def run(self, image, image_id="anonymous"):
        t_start = time.perf_counter()
        
        t0 = time.perf_counter()
        tensor = self.preprocess(image)
        self.metrics_log["preprocess"].append((time.perf_counter() - t0) * 1000)

        t1 = time.perf_counter()
        det = self.detect(tensor)
        self.metrics_log["detect"].append((time.perf_counter() - t1) * 1000)

        crops, detections, valid_indices = [], [], []
        
        for i, (box, score, cls) in enumerate(zip(det["boxes"], det["scores"], det["labels"])):
            if score < 0.5:
                continue
                
            x1, y1, x2, y2 = [max(0, int(b)) for b in box.tolist()]
            x2, y2 = min(x2, tensor.shape[-1]), min(y2, tensor.shape[-2])
            
            det_cls_id = int(cls)
            det_label = self.detector_classes[det_cls_id] if det_cls_id < len(self.detector_classes) else f"obj_{det_cls_id}"
            
            detections.append(Detection(
                box=(x1, y1, x2, y2),
                score=float(score),
                class_id=det_cls_id,
                class_name=det_label 
            ))
            
            if (x2 - x1) < self.min_crop or (y2 - y1) < self.min_crop:
                continue
                
            crop = tensor[:, y1:y2, x1:x2]
            crop = torch.nn.functional.interpolate(
                crop.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
            )[0]
            crops.append(crop)
            valid_indices.append(len(detections) - 1)

        t2 = time.perf_counter()
        class_preds = self.classify(crops)
        self.metrics_log["classify"].append((time.perf_counter() - t2) * 1000)

        classifications = []
        for valid_idx, (cls_id, cls_score) in zip(valid_indices, class_preds):
            classifications.append(Classification(
                detection_index=valid_idx,
                class_id=int(cls_id),
                class_name=self.classifier_names[cls_id],
                score=float(cls_score),
            ))

        total_ms = (time.perf_counter() - t_start) * 1000
        self.metrics_log["total"].append(total_ms)
        self.save_state()

        return PipelineResult(
            image_id=image_id,
            detections=detections,
            classifications=classifications,
            inference_ms=total_ms,
        )
    
    def process_video(self, video_path: str, output_path: str, video_id: str) -> VideoJobResult:
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"\n[{video_id}] --- STARTING VIDEO JOB ---")
        print(f"[{video_id}] Resolution: {width}x{height} @ {fps} FPS")
        print(f"[{video_id}] Total Frames: {total_frames}")
        
        t_start = time.perf_counter()
        frame_idx = 0
        track_class_cache = {}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = self.preprocess(rgb_frame)
            
            det = self.detect(tensor)
            bbs = []
            
            for box, score, cls in zip(det["boxes"], det["scores"], det["labels"]):
                if score < 0.5:
                    continue
                x1, y1, x2, y2 = [int(b) for b in box.tolist()]
                w, h = x2 - x1, y2 - y1
                bbs.append(([x1, y1, w, h], float(score), int(cls)))
                
            tracks = self.tracker.update_tracks(bbs, frame=rgb_frame)
            
            crops = []
            valid_tracks = []
            
            for track in tracks:
                if not track.is_confirmed():
                    continue
                    
                track_id = track.track_id
                ltrb = track.to_ltrb()
                
                x1 = max(0, min(int(ltrb[0]), tensor.shape[-1]))
                y1 = max(0, min(int(ltrb[1]), tensor.shape[-2]))
                x2 = max(0, min(int(ltrb[2]), tensor.shape[-1]))
                y2 = max(0, min(int(ltrb[3]), tensor.shape[-2]))
                
                if x1 >= x2 or y1 >= y2:
                    continue
                # -------------------------------------------------------------
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if track_id not in track_class_cache and (x2 - x1) >= self.min_crop and (y2 - y1) >= self.min_crop:
                    crop = tensor[:, y1:y2, x1:x2]
                    crop = torch.nn.functional.interpolate(crop.unsqueeze(0), size=(224, 224), mode="bilinear")[0]
                    crops.append(crop)
                    valid_tracks.append(track_id)
            
            if crops:
                class_preds = self.classify(crops)
                for tid, (cls_id, _) in zip(valid_tracks, class_preds):
                    track_class_cache[tid] = self.classifier_names[cls_id]
            
            for track in tracks:
                if track.is_confirmed() and track.track_id in track_class_cache:
                    ltrb = track.to_ltrb()
                    cv2.putText(frame, track_class_cache[track.track_id], (int(ltrb[0]), int(ltrb[1]) + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            out.write(frame)
            
            if frame_idx % 10 == 0:
                elapsed = time.perf_counter() - t_start
                current_fps = frame_idx / elapsed
                percent = (frame_idx / total_frames) * 100
                print(f"[{video_id}] Progress: {frame_idx}/{total_frames} frames ({percent:.1f}%) | Speed: {current_fps:.1f} it/s | Active Tracks: {len(valid_tracks)}")
                
                self.metrics_log["video_progress"][video_id] = {"frames": frame_idx, "total": total_frames}
                self.save_state()

        cap.release()
        out.release()
        
        total_time = time.perf_counter() - t_start
        print(f"\n[{video_id}] --- JOB COMPLETE ---")
        print(f"[{video_id}] Processed {frame_idx} frames in {total_time:.1f} seconds (Avg: {frame_idx/total_time:.1f} it/s)")
        print(f"[{video_id}] Output saved to: {output_path}\n")
        
        self.metrics_log["video_progress"][video_id] = {"frames": frame_idx, "status": "completed"}
        self.save_state()

        return VideoJobResult(
            video_id=video_id,
            total_frames=total_frames,
            processed_frames=frame_idx,
            fps=fps,
            output_video_path=output_path,
            total_time_sec=total_time
        )