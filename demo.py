import gradio as gr
import numpy as np
from PIL import ImageDraw, ImageOps
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO


def launch(weights):
    normal = YOLO(str(weights))
    sliced = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=str(weights),
        confidence_threshold=0.25, device="cuda:0"
    )

    def compare(image):
        if image is None:
            raise gr.Error("Upload a JPG or PNG image.")
        image = ImageOps.exif_transpose(image).convert("RGB")
        result = normal.predict(image, conf=0.25, verbose=False)[0]
        tiled = get_sliced_prediction(
            np.asarray(image), sliced, slice_height=512, slice_width=512,
            overlap_height_ratio=0.2, overlap_width_ratio=0.2, verbose=0
        )
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        for obj in tiled.object_prediction_list:
            box = obj.bbox.to_xyxy()
            draw.rectangle(box, outline="lime", width=2)
            draw.text(box[:2], f"{obj.category.name} {obj.score.value:.2f}", fill="yellow")
        return result.plot(pil=False)[:, :, ::-1], overlay

    gr.Interface(compare, gr.Image(type="pil", sources=["upload"]),
                 [gr.Image(label="Normal YOLO"), gr.Image(label="SAHI 512 × 512")],
                 title="Small-object detection in aerial scenes").launch(share=True)
