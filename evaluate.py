
from pathlib import Path
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from PIL import Image

def read_small_ground_truth(image_path, label_dir):
    label_path = label_dir / f"{image_path.stem}.txt"
    if not label_path.exists():
        return []

    with Image.open(image_path) as im:
        image_w, image_h = im.size

    boxes = []
    for line in label_path.read_text().splitlines():
        values = line.split()
        if len(values) < 5:
            continue

        cls, xc, yc, w, h = map(float, values[:5])
        w_px, h_px = w * image_w, h * image_h

        if w_px <= 32 and h_px <= 32:
            boxes.append({
                "class": int(cls),
                "box": [
                    (xc - w / 2) * image_w,
                    (yc - h / 2) * image_h,
                    (xc + w / 2) * image_w,
                    (yc + h / 2) * image_h
                ]
            })
    return boxes

def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def calculate_ap50(all_gt, all_pred, class_id):
    # Only evaluate classes with small ground-truth objects
    gt_by_image = {
        i: [g["box"] for g in gt if g["class"] == class_id]
        for i, gt in enumerate(all_gt)
    }
    total_gt = sum(len(boxes) for boxes in gt_by_image.values())
    if total_gt == 0:
        return None

    matched = {i: set() for i in range(len(all_gt))}
    predictions = []
    for i, preds in enumerate(all_pred):
        for p in preds:
            if p["class"] == class_id:
                predictions.append((p["score"], i, p["box"]))

    predictions.sort(key=lambda x: x[0], reverse=True)
    tp, fp = [], []

    for score, image_i, pred_box in predictions:
        gt_boxes = gt_by_image[image_i]
        best_iou, best_j = 0.0, -1

        for j, gt_box in enumerate(gt_boxes):
            if j in matched[image_i]:
                continue
            overlap = box_iou(pred_box, gt_box)
            if overlap > best_iou:
                best_iou, best_j = overlap, j

        if best_iou >= 0.50:
            tp.append(1)
            fp.append(0)
            matched[image_i].add(best_j)
        else:
            tp.append(0)
            fp.append(1)

    if not tp:
        return 0.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / total_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)

    # 101-point interpolated AP
    ap = np.mean([
        np.max(precision[recall >= r]) if np.any(recall >= r) else 0.0
        for r in np.linspace(0, 1, 101)
    ])
    return float(ap)

def evaluate(model_path, val_dir, label_dir, out_dir, tile=512):
    out_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(val_dir.glob("*.jpg"))[:100]

    assert len(image_paths) == 100, (
        f"Expected 100 images, found {len(image_paths)}"
    )

    # 2. Rebuild ground truth for these exact 100 images
    all_gt = [read_small_ground_truth(p, label_dir) for p in image_paths]
    class_ids = sorted({
        obj["class"] for image_gt in all_gt for obj in image_gt
    })

    print("Images:", len(image_paths))
    print("Ground-truth entries:", len(all_gt))
    print("Small ground-truth objects:", sum(map(len, all_gt)))
    print("Classes:", class_ids)

    # 3. Load models
    normal_model = YOLO(model_path)

    sahi_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=0.001,
        device="cuda:0"
    )

    normal_predictions = []
    sahi_predictions = []

    # 4. Run both methods on every image
    for i, image_path in enumerate(image_paths):
        # Normal YOLO
        result = normal_model.predict(
            source=str(image_path),
            conf=0.001,
            max_det=3000,
            verbose=False
        )[0]

        preds = []
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                if (x2 - x1) <= 32 and (y2 - y1) <= 32:
                    preds.append({
                        "class": int(box.cls.item()),
                        "score": float(box.conf.item()),
                        "box": [x1, y1, x2, y2]
                    })
        normal_predictions.append(preds)

        # SAHI with 512 x 512 tiles
        sliced = get_sliced_prediction(
            str(image_path),
            sahi_model,
            slice_height=tile,
            slice_width=tile,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            verbose=0
        )

        preds = []
        for obj in sliced.object_prediction_list:
            x1, y1, x2, y2 = obj.bbox.to_xyxy()
            if (x2 - x1) <= 32 and (y2 - y1) <= 32:
                preds.append({
                    "class": int(obj.category.id),
                    "score": float(obj.score.value),
                    "box": [x1, y1, x2, y2]
                })
        sahi_predictions.append(preds)

        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/100 images")

    # 5. Calculate custom small-object AP50 on the same ground truth
    normal_aps = [
        calculate_ap50(all_gt, normal_predictions, cls)
        for cls in class_ids
    ]
    sahi_aps = [
        calculate_ap50(all_gt, sahi_predictions, cls)
        for cls in class_ids
    ]

    normal_ap50 = 100 * np.mean(normal_aps)
    sahi_ap50 = 100 * np.mean(sahi_aps)

    comparison = pd.DataFrame([
        {
            "method": "Normal YOLO",
            "images": len(image_paths),
            "small_object_macro_AP50_percent": round(normal_ap50, 2)
        },
        {
            "method": f"SAHI {tile}x{tile}",
            "images": len(image_paths),
            "small_object_macro_AP50_percent": round(sahi_ap50, 2)
        }
    ])

    print(comparison.to_string(index=False))

    comparison.to_csv(
        out_dir / f"normal_vs_sahi_{tile}.csv",
        index=False
    )

    print("Normal predictions:", len(normal_predictions))
    print("SAHI predictions:", len(sahi_predictions))
    print("Corrected results saved to:", out_dir)
