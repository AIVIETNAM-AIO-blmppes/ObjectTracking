# benchmark.py
import numpy as np
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models import convnext_tiny
from pipeline import VisionPipeline

def run_benchmark():
    detector = maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
    classifier = convnext_tiny(weights="DEFAULT")
    class_names = [f"class_{i}" for i in range(1000)]
    
    pipe = VisionPipeline(detector, classifier, class_names)
    
    print("Warming up pipeline...")
    img = (np.random.rand(400, 600, 3) * 255).astype(np.uint8)
    pipe.run(img)
    
    print("Running benchmark (20 iterations)...")
    for _ in range(20):
        test_img = (np.random.rand(400, 600, 3) * 255).astype(np.uint8)
        pipe.run(test_img)
        
    print("\n--- Latency Report ---")
    for stage, times in pipe.metrics_log.items():
        if not times:
            continue
        times.sort()
        p50 = times[len(times)//2]
        p95 = times[int(len(times)*0.95)]
        print(f"{stage:12s} | p50: {p50:7.1f} ms | p95: {p95:7.1f} ms")
        
    pipe.save_state("benchmark_checkpoints")
    print("Benchmark completed. Metrics and weights saved to 'benchmark_checkpoints/'.")

if __name__ == "__main__":
    run_benchmark()