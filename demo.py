from PIL import ImageDraw, ImageOps
from comparison import Detectors, METHODS, NAMES


def launch(weights):
    import gradio as gr
    models = Detectors(weights)

    def compare(image, threshold):
        if image is None:
            raise gr.Error("Upload a JPG or PNG image.")
        image = ImageOps.exif_transpose(image).convert("RGB")
        outputs, statistics = [], []
        for method in METHODS:
            predictions, ms = models.predict(method, image, threshold)
            canvas = image.copy()
            draw = ImageDraw.Draw(canvas)
            for obj in predictions:
                draw.rectangle(obj["box"], outline="lime", width=2)
                draw.text(obj["box"][:2], f"{NAMES[obj['class']]} {obj['score']:.2f}", fill="yellow")
            outputs.append(canvas)
            statistics.append([method, len(predictions), round(ms,1)])
        return *outputs, statistics

    with gr.Blocks() as app:
        gr.Markdown("# Small-object detection in aerial scenes\nCompare bicycle, car, truck, bus and motor detections.")
        source = gr.Image(type="pil", sources=["upload"], label="Upload a JPG or PNG aerial image")
        threshold = gr.Slider(0.01, 0.95, value=0.25, step=0.01, label="Confidence threshold")
        button = gr.Button("Compare detections")
        with gr.Row():
            outputs = [gr.Image(label=name, format="png", show_download_button=True) for name in METHODS]
        stats = gr.Dataframe(headers=["Model","Detections","Time (ms)"], interactive=False)
        button.click(compare,[source,threshold],[*outputs,stats],concurrency_limit=1)
    return app.launch(share=True, prevent_thread_lock=True)
