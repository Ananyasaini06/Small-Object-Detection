from PIL import ImageDraw, ImageOps
from pathlib import Path
import hashlib
import pandas as pd
from comparison import Detectors, METHODS, NAMES, selection


def build_demo(models, weights, project=None):
    import gradio as gr
    assets = Path(__file__).resolve().parent

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
                label = f"{NAMES[obj['class']]} {obj['score']:.2f}"
                x, y = obj["box"][:2]
                x = max(0, min(x, canvas.width - draw.textlength(label) - 5))
                y = max(0, y - 15)
                draw.rectangle(obj["box"], outline="#36ebcf", width=2)
                draw.rectangle((x, y, x + draw.textlength(label) + 4, y + 14), fill="#103539")
                draw.text((x + 2, y), label, fill="white")
            outputs.append(canvas)
            statistics.append([method, len(predictions), round(ms,1)])
        return *outputs, statistics, diagnostics

    css = """
    .gradio-container {max-width: 1320px !important; margin: auto;}
    #overview {padding: 24px 28px; border-radius: 14px; background: #103539;}
    #overview h1, #overview p {color: #f2fffc;}
    """
    with gr.Blocks(title="P10 · Aerial object detection", theme=gr.themes.Soft(primary_hue="teal"), css=css) as app:
        gr.Markdown("# Small-object detection in aerial scenes\n**Project overview** · Vehicles can occupy only a few pixels in a drone image. Compare how three detectors handle the same scene.", elem_id="overview")
        gr.Markdown("## Live detection\nUpload a JPG or PNG, or select a VisDrone sample. All three panels use the same five vehicle classes.")
        with gr.Row():
            source = gr.Image(type="pil", sources=["upload"], label="Aerial image", height=235)
            with gr.Column(min_width=240):
                gr.Examples(examples=[str(p) for p in sorted((assets/"samples").glob("*.jpg"))],
                            inputs=source, label="VisDrone validation samples", cache_examples=False)
        threshold = gr.Slider(0.01, 0.95, value=0.25, step=0.01, label="Confidence threshold")
        button = gr.Button("Compare detections", variant="primary")
        with gr.Row():
            outputs = [gr.Image(label=name, min_width=230, height=260, interactive=False, format="png", show_download_button=True) for name in METHODS]
        stats = gr.Dataframe(headers=["Model","Detections","Time (ms)"], interactive=False)
        with gr.Accordion("Why did Faster R-CNN return this count?", open=False):
            gr.Markdown("COCO predictions after native postprocessing at a 0.001 score floor, before the shared-class filter. Only shared labels above the slider threshold appear in the image. The live time includes this diagnostic pass.")
            raw = gr.Dataframe(headers=["COCO label", "Confidence", "VisDrone mapping", "Kept for display"], interactive=False)
        button.click(compare,[source,threshold],[*outputs,stats,raw],concurrency_limit=1)
        gr.Markdown("## Model comparison\nVerified results on 100 validation images and five shared vehicle classes. Counts alone cannot establish accuracy: extra boxes can include false positives.")
        verified = pd.read_csv(assets/"results/verified-five-class.csv")
        gr.Dataframe(value=verified.rename(columns={"model":"Model", "AP50_pct":"AP50 (%)",
                     "small_AP50_pct":"Small AP50 (%)", "latency_ms":"Latency (ms/image)",
                     "detections_per_image":"Detections/image"}).round(2), interactive=False)
        gr.ScatterPlot(verified, x="latency_ms", y="small_AP50_pct", color="model",
                       x_title="Mean latency (ms/image)", y_title="Small-object AP50 (%)",
                       title="Small-object accuracy versus inference time")
        gr.Markdown("## Training evidence\nFinal YOLO validation: precision **39.7%**, recall **30.0%**, mAP50 **27.5%**, mAP50–95 **15.1%**.")
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
        gr.Markdown("## Final model selection\n**YOLO11n** is the general-purpose choice: it has the highest overall AP50 and the lowest latency. **Faster R-CNN** leads on small-object AP50, at about ten times YOLO11n’s latency. **SAHI** improved small-object AP50 in the earlier ten-class experiment; the two evaluations average over different classes.")
        with gr.Accordion("Measurements for the loaded checkpoint", open=False):
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
                gr.Markdown("No separate evaluation was found for these weights. The verified project measurements are shown above.")
    return app


def launch(weights, project=None):
    app = build_demo(Detectors(weights), weights, project)
    return app.launch(share=True, prevent_thread_lock=True)
