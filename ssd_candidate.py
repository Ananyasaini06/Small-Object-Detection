"""Evaluate COCO SSDLite320 on the project's sorted VisDrone validation subset."""
import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large,
)

from comparison import COCO, NAMES, ground_truth, small
from evaluate import calculate_ap50


def evaluate(images, labels, output, device):
    paths = sorted(images.glob("*.jpg"))[:100]
    if len(paths) != 100:
        raise ValueError("Expected the first 100 sorted validation JPGs")
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = ssdlite320_mobilenet_v3_large(weights=weights, score_thresh=0.001).eval().to(device)
    transform = weights.transforms()
    targets, predictions, times = [], [], []
    for index, path in enumerate(paths):
        with Image.open(path) as source:
            image = source.convert("RGB")
        targets.append(ground_truth(labels / (path.stem + ".txt"), image.size))
        with torch.inference_mode():
            if index == 0:
                model([transform(image).to(device)])
            if device == "cuda":
                torch.cuda.synchronize()
            start = perf_counter()
            result = model([transform(image).to(device)])[0]
            rows = [{"class": COCO[label], "score": score, "box": box}
                    for label, score, box in zip(result["labels"].tolist(), result["scores"].tolist(),
                                                 result["boxes"].tolist()) if label in COCO]
            times.append((perf_counter()-start)*1000)
        predictions.append(rows)
    row = {"model": "SSDLite320 MobileNetV3-Large", "device": device, "images": len(paths)}
    for key, gt, pred in [("AP50_pct", targets, predictions),
                          ("small_AP50_pct", list(map(small, targets)), list(map(small, predictions)))]:
        scores = [calculate_ap50(gt, pred, cls) for cls in NAMES]
        row[key] = 100*np.mean([s for s in scores if s is not None])
    row.update(latency_ms=np.mean(times), detections_per_image=np.mean(list(map(len, predictions))),
               images_with_shared_detection_at_025=sum(any(p["score"] >= .25 for p in rows) for rows in predictions))
    with output.open("x") as f:
        writer = csv.DictWriter(f, fieldnames=row)
        writer.writeheader()
        writer.writerow(row)
    print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("images", "labels", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    evaluate(args.images, args.labels, args.output, args.device)
