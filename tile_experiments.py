"""Reproduce the archived tile study and figures with a supplied checkpoint."""

def run(weights, val_dir, label_dir, destination):
    from IPython.display import display
    from pathlib import Path
    import json
    import time
    import numpy as np
    import pandas as pd
    from ultralytics import YOLO
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw
    from evaluate import calculate_ap50, read_small_ground_truth

    model_path = str(weights)

    # Use the same 100 images as our earlier comparison
    image_paths = sorted(val_dir.glob("*.jpg"))[:100]

    print("Model exists:", Path(model_path).exists())
    print("Validation images:", len(image_paths))
    print("Tile sizes to test:", [256, 512, 640])

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=0.25,
        device="cuda:0"
    )

    tile_results = []

    for tile_size in [256, 512, 640]:
        times = []
        detection_counts = []

        print(f"\nTesting tile size {tile_size}x{tile_size}...")

        for image_path in image_paths:
            start = time.perf_counter()

            result = get_sliced_prediction(
                str(image_path),
                detection_model,
                slice_height=tile_size,
                slice_width=tile_size,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
                verbose=0
            )

            times.append(time.perf_counter() - start)
            detection_counts.append(len(result.object_prediction_list))

        tile_results.append({
            "tile_size": f"{tile_size}x{tile_size}",
            "images_evaluated": len(image_paths),
            "avg_latency_ms": round(np.mean(times) * 1000, 2),
            "avg_detections": round(np.mean(detection_counts), 2)
        })

    tile_df = pd.DataFrame(tile_results)
    display(tile_df)

    output_dir = destination
    output_dir.mkdir(parents=True, exist_ok=True)

    tile_df.to_csv(output_dir / "tile_size_comparison.csv", index=False)

    with open(output_dir / "tile_size_comparison.json", "w") as f:
        json.dump(tile_results, f, indent=2)

    print("Saved tile-size results to:", output_dir)

    # Use a low confidence threshold for AP evaluation
    detector = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=0.001,
        device="cuda:0"
    )

    all_gt = [read_small_ground_truth(p, label_dir) for p in image_paths]
    class_ids = sorted({g["class"] for gt in all_gt for g in gt})
    tile_ap_results = []

    for tile_size in [256, 512, 640]:
        print(f"Evaluating {tile_size}x{tile_size} tiles...")
        all_pred = []

        for image_path in image_paths:
            result = get_sliced_prediction(
                str(image_path),
                detector,
                slice_height=tile_size,
                slice_width=tile_size,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
                verbose=0
            )

            image_preds = []
            for obj in result.object_prediction_list:
                x1, y1, x2, y2 = obj.bbox.to_xyxy()
                w, h = x2-x1, y2-y1

                # Match the same small-object definition as ground truth
                if w <= 32 and h <= 32:
                    image_preds.append({
                        "class": int(obj.category.id),
                        "score": float(obj.score.value),
                        "box": [x1, y1, x2, y2]
                    })
            all_pred.append(image_preds)

        per_class_ap = {
            cls: calculate_ap50(all_gt, all_pred, cls)
            for cls in class_ids
        }
        valid_aps = [v for v in per_class_ap.values() if v is not None]
        macro_ap = 100 * np.mean(valid_aps) if valid_aps else 0.0

        tile_ap_results.append({
            "tile_size": f"{tile_size}x{tile_size}",
            "small_object_macro_AP50_percent": round(macro_ap, 2),
            "classes_evaluated": len(valid_aps)
        })

        print("Small-object macro AP50:", round(macro_ap, 2), "%")

    ap_df = pd.DataFrame(tile_ap_results)
    display(ap_df)

    ap_df.to_csv(output_dir / "tile_size_small_object_AP50.csv", index=False)
    with open(output_dir / "tile_size_small_object_AP50.json", "w") as f:
        json.dump(tile_ap_results, f, indent=2)

    print("AP50 results saved in", output_dir)

    out = destination

    summary = tile_df.merge(ap_df, on="tile_size")
    summary.to_csv(out / "final_tile_comparison.csv", index=False)

    display(summary)

    # Plot small-object AP50 by tile size
    plt.figure(figsize=(8, 5))
    plt.bar(summary["tile_size"],
            summary["small_object_macro_AP50_percent"])
    plt.xlabel("Tile size")
    plt.ylabel("Small-object macro AP50 (%)")
    plt.title("Effect of Tile Size on Small-Object Detection")
    plt.tight_layout()
    plt.savefig(out / "tile_size_AP50.png", dpi=200)
    plt.show()

    print("Saved combined table and graph to", out)

    save_dir = destination / "qualitative_comparisons"
    save_dir.mkdir(parents=True, exist_ok=True)

    yolo_model = YOLO(model_path)

    sahi_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=0.25,
        device="cuda:0"
    )

    image_paths = sorted(val_dir.glob("*.jpg"))[:3]

    for i, image_path in enumerate(image_paths, start=1):
        # Normal YOLO detection
        normal_result = yolo_model.predict(
            source=str(image_path),
            conf=0.25,
            verbose=False
        )[0]

        normal_image = Image.fromarray(
            normal_result.plot()[..., ::-1]
        ).convert("RGB")

        # SAHI tiled detection at 512 x 512
        sahi_result = get_sliced_prediction(
            str(image_path),
            sahi_model,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            verbose=0
        )

        sahi_image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(sahi_image)

        for obj in sahi_result.object_prediction_list:
            x1, y1, x2, y2 = map(int, obj.bbox.to_xyxy())
            name = obj.category.name
            score = obj.score.value

            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text(
                (x1, max(0, y1 - 12)),
                f"{name} {score:.2f}",
                fill="red"
            )

        # Put the two predictions side by side
        width, height = normal_image.size
        sahi_image = sahi_image.resize((width, height))

        panel = Image.new("RGB", (width * 2, height + 35), "white")
        panel.paste(normal_image, (0, 35))
        panel.paste(sahi_image, (width, 35))

        draw_panel = ImageDraw.Draw(panel)
        draw_panel.text((10, 10), "Normal YOLO", fill="black")
        draw_panel.text((width + 10, 10), "SAHI 512x512", fill="black")

        output_path = save_dir / f"comparison_{i}.jpg"
        panel.save(output_path, quality=95)
        print("Saved:", output_path)

        plt.figure(figsize=(16, 6))
        plt.imshow(panel)
        plt.axis("off")
        plt.show()

    print("All comparisons saved in:", save_dir)

    # Plot accuracy against latency
    plt.figure(figsize=(8, 5))
    plt.plot(
        summary["avg_latency_ms"],
        summary["small_object_macro_AP50_percent"],
        marker="o"
    )

    for _, row in summary.iterrows():
        plt.annotate(
            row["tile_size"],
            (row["avg_latency_ms"],
             row["small_object_macro_AP50_percent"]),
            xytext=(5, 5),
            textcoords="offset points"
        )

    plt.xlabel("Average latency per image (ms) — lower is faster")
    plt.ylabel("Small-object macro AP50 (%) — higher is better")
    plt.title("Tile Size: Accuracy vs Inference Speed")
    plt.grid(True)
    plt.tight_layout()

    graph_path = out / "latency_vs_accuracy.png"
    plt.savefig(graph_path, dpi=200)
    plt.show()

    print("Saved graph:", graph_path)
    display(summary)
