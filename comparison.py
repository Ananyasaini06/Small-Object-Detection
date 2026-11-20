"""Three detectors evaluated in the same five-class VisDrone label space."""
from time import perf_counter
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from PIL import Image
from evaluate import calculate_ap50

NAMES = {2: "bicycle", 3: "car", 5: "truck", 8: "bus", 9: "motor"}
COCO = {2: 2, 3: 3, 8: 5, 6: 8, 4: 9}
METHODS = ("YOLO11n", "YOLO11n + SAHI", "Faster R-CNN")


class Detectors:
    def __init__(self, weights):
        import torch
        from ultralytics import YOLO
        from sahi import AutoDetectionModel
        from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
        self.yolo = YOLO(str(weights))
        self.sahi = AutoDetectionModel.from_pretrained(model_type="ultralytics",
            model_path=str(weights), confidence_threshold=0.001, device="cuda:0")
        self.categories = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta["categories"]
        self.rcnn = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
            box_score_thresh=0.001, box_detections_per_img=3000).eval().cuda()

    def predict(self, method, image, threshold, diagnostics=None):
        import torch
        from torchvision.transforms.functional import to_tensor
        from sahi.predict import get_sliced_prediction
        torch.cuda.synchronize()
        start = perf_counter()
        if method == METHODS[0]:
            boxes = self.yolo.predict(image, imgsz=640, conf=threshold, max_det=3000, verbose=False)[0].boxes
            rows = boxes.data.cpu().tolist()
        elif method == METHODS[1]:
            self.sahi.confidence_threshold = threshold
            result = get_sliced_prediction(np.asarray(image), self.sahi, slice_height=512, slice_width=512,
                overlap_height_ratio=0.2, overlap_width_ratio=0.2, verbose=0)
            rows = [[*o.bbox.to_xyxy(), o.score.value, o.category.id] for o in result.object_prediction_list]
        else:
            with torch.inference_mode():
                self.rcnn.roi_heads.score_thresh = min(0.001, threshold) if diagnostics is not None else threshold
                out = self.rcnn([to_tensor(image).cuda()])[0]
            labels = out["labels"].cpu().tolist()
            scores = out["scores"].cpu().tolist()
            boxes = out["boxes"].cpu().tolist()
            rows = [[*box, score, COCO[label]] for box, score, label in zip(boxes, scores, labels)
                    if label in COCO]
            if diagnostics is not None:
                for label, score in zip(labels, scores):
                    diagnostics.append([self.categories[label], round(score, 5),
                                        NAMES.get(COCO.get(label), "Outside shared classes"),
                                        label in COCO and score >= threshold])
        preds = [{"class": int(c), "score": s, "box": [x,y,r,b]} for x,y,r,b,s,c in rows if c in NAMES and s >= threshold]
        torch.cuda.synchronize()
        return preds, (perf_counter()-start)*1000


def ground_truth(path, size):
    w,h = size
    boxes = []
    for line in path.read_text().splitlines():
        cls,x,y,bw,bh = map(float,line.split())
        if cls in NAMES:
            boxes.append({"class":int(cls),"box":[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]})
    return boxes


def small(rows):
    return [o for o in rows if o["box"][2]-o["box"][0] <= 32 and o["box"][3]-o["box"][1] <= 32]


def selection(frame):
    ranked = frame.sort_values(["small_AP50_pct", "latency_ms", "AP50_pct"],
                               ascending=[False, True, False])
    best = ranked.iloc[0]
    fastest = frame.loc[frame.latency_ms.idxmin()]
    overall = frame.sort_values(["AP50_pct", "latency_ms"], ascending=[False, True]).iloc[0]
    if overall.model == fastest.model:
        summary = f"Recommended for overall accuracy and speed: {overall.model}."
    else:
        summary = f"Overall AP50 leader: {overall.model}. Fastest: {fastest.model}."
    return (f"{summary} Highest overall AP50: {overall.AP50_pct:.4f}%. "
            f"Lowest latency: {fastest.latency_ms:.4f} ms/image. "
            f"For small-object accuracy, choose {best.model}: {best.small_AP50_pct:.4f}% "
            f"small-object AP50 at {best.latency_ms:.4f} ms/image.")


def benchmark(weights, val_dir, label_dir, destination):
    from matplotlib import pyplot as plt
    paths = sorted(val_dir.glob("*.jpg"))[:100]
    if len(paths) != 100:
        raise ValueError("Expected 100 validation images")
    destination.mkdir(parents=True, exist_ok=False)
    (destination/"images.txt").write_text("\n".join(p.name for p in paths))
    (destination/"checkpoint.txt").write_text(str(weights)+"\n"+hashlib.sha256(Path(weights).read_bytes()).hexdigest())
    models = Detectors(weights)
    records = []
    for method in METHODS:
        truth, predictions, times = [], [], []
        for index, path in enumerate(paths):
            with Image.open(path) as source:
                image = source.convert("RGB")
            truth.append(ground_truth(label_dir/(path.stem+".txt"), image.size))
            if index == 0:
                models.predict(method, image, 0.001)
            pred, ms = models.predict(method, image, 0.001)
            predictions.append(pred)
            times.append(ms)
        row = {"model":method, "latency_ms":np.mean(times),
               "detections_per_image":np.mean([len(p) for p in predictions])}
        for key, gt, pred in [("AP50_pct",truth,predictions),
                ("small_AP50_pct",list(map(small,truth)),list(map(small,predictions)))]:
            aps = [calculate_ap50(gt,pred,c) for c in NAMES]
            row[key] = 100*np.mean([a for a in aps if a is not None])
        records.append(row)
    frame = pd.DataFrame(records)
    frame.to_csv(destination/"comparison.csv",index=False)
    decision = selection(frame)
    (destination/"selection.txt").write_text(decision)
    print(decision)
    fig, ax = plt.subplots()
    for row in records:
        ax.scatter(row["latency_ms"],row["small_AP50_pct"],label=row["model"])
    ax.set(xlabel="Mean latency (ms/image)",ylabel="Shared-class small AP50 (%)")
    ax.legend()
    fig.savefig(destination/"accuracy-speed.png",bbox_inches="tight")
    plt.close(fig)
    return frame
