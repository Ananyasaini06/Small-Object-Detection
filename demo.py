from PIL import ImageDraw, ImageOps
from pathlib import Path
import hashlib
import pandas as pd
from comparison import Detectors, METHODS, NAMES, selection


def launch(weights, project=None):
    import gradio as gr
    models = Detectors(weights)

    def compare(image, threshold):
        if image is None:
            raise gr.Error("Upload a JPG or PNG image.")
        image = ImageOps.exif_transpose(image).convert("RGB")
        outputs, statistics = [], []
        diagnostics = []
        for method in METHODS:
            predictions, ms = models.predict(method, image, threshold, diagnostics if method == METHODS[2] else None)
            canvas = image.copy()
            draw = ImageDraw.Draw(canvas)
            for obj in predictions:
                draw.rectangle(obj["box"], outline="lime", width=2)
                draw.text(obj["box"][:2], f"{NAMES[obj['class']]} {obj['score']:.2f}", fill="yellow")
            outputs.append(canvas)
            statistics.append([method, len(predictions), round(ms,1)])
        return *outputs, statistics, diagnostics

    with gr.Blocks() as app:
        gr.Markdown("# Small-object detection in aerial scenes\nVehicles occupy only a few pixels in drone images. Upload one scene to see how full-image, tiled and region-based detectors handle it. All panels show the same five vehicle classes.")
        source = gr.Image(type="pil", sources=["upload"], label="Upload a JPG or PNG aerial image")
        threshold = gr.Slider(0.01, 0.95, value=0.25, step=0.01, label="Confidence threshold")
        button = gr.Button("Compare detections")
        with gr.Row():
            outputs = [gr.Image(label=name, format="png", show_download_button=True) for name in METHODS]
        stats = gr.Dataframe(headers=["Model","Detections","Time (ms)"], interactive=False)
        with gr.Accordion("Why did Faster R-CNN return this count?", open=False):
            gr.Markdown("COCO predictions after native postprocessing at a 0.001 score floor, before the shared-class filter. Only shared labels above the slider threshold appear in the image. The live time includes this diagnostic pass.")
            raw = gr.Dataframe(headers=["COCO label", "Candidates", "Top score", "Above slider", "Shared class"], interactive=False)
        button.click(compare,[source,threshold],[*outputs,stats,raw],concurrency_limit=1)
        assets = Path(__file__).resolve().parent
        with gr.Accordion("Training, validation and testing evidence", open=False):
            gr.Markdown("**Training.** Original 15-epoch VisDrone run. The curves and table use rounded validation logs from the archived notebook.")
            gr.Image(str(assets/"figures/historical_epochs.png"), interactive=False, label="Historical validation curves")
            gr.Dataframe(value=pd.read_csv(assets/"figures/historical_epochs.csv"), interactive=False)
            run = Path(weights).parent.parent
            if Path(weights).parent.name == "weights":
                if (run/"results.csv").is_file():
                    gr.Dataframe(value=pd.read_csv(run/"results.csv"), interactive=False, label="Loaded run: epoch history")
                if (run/"results.png").is_file():
                    gr.Image(str(run/"results.png"), label="Loaded run: training curves", interactive=False)
            gr.Markdown("**Validation.** Historical custom small AP50: YOLO 16.38%, SAHI 512 28.42%. "
                        "Among the tested tiles, 512 pixels gave the highest small AP50. "
                        "These ten-class results support SAHI 512 as the historical accuracy-first choice.")
            gr.Dataframe(value=pd.read_csv(assets/"final_tile_comparison.csv"), interactive=False)
            gr.Markdown("**Testing.** Use a new aerial photo for a qualitative check. The upload has no ground truth; counts and times describe that image only.")
        with gr.Accordion("Three-model selection on shared classes", open=True):
            digest = hashlib.sha256(Path(weights).read_bytes()).hexdigest()
            for csv in sorted((Path(project)/"evaluation").glob("*/comparison.csv"), reverse=True) if project else []:
                stamp = csv.with_name("checkpoint.txt")
                if stamp.is_file() and stamp.read_text().splitlines()[-1:] == [digest]:
                    table = pd.read_csv(csv)
                    if len(table) == 3 and set(table.model) == set(METHODS) and not table.isna().any().any():
                        gr.Markdown(selection(table))
                        gr.Dataframe(value=table, interactive=False, label="Saved validation comparison")
                        graph = csv.with_name("accuracy-speed.png")
                        if graph.is_file():
                            gr.Image(str(graph), label="Small-object accuracy versus latency", interactive=False)
                        break
            else:
                table = pd.read_csv(assets/"results"/"verified-five-class.csv")
                gr.Markdown("Verified project benchmark on 100 validation images (five shared classes).")
                gr.Markdown(selection(table))
                gr.Dataframe(value=table, interactive=False, label="Verified project comparison")
    return app.launch(share=True, prevent_thread_lock=True)
