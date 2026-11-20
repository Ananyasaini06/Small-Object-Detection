# Small-object detection in aerial scenes

A car in a drone image can occupy just a few pixels. This project asks whether running a detector on smaller crops helps it find those objects, and how much extra time that takes. The experiments use the VisDrone dataset and a Google Colab T4 GPU.

## Models and evaluation

- **YOLO11n** predicts boxes and classes in one pass over the resized image. The original experiment fine-tuned pretrained weights on VisDrone for 15 epochs at an image size of 640.
- **YOLO11n with SAHI** uses the same weights. SAHI runs the detector on overlapping tiles and merges the predictions. The tiles overlap by 20%. The 512-pixel configuration is used in the three-model comparison.
- **Faster R-CNN** first proposes regions, then classifies them and adjusts their boxes. The notebook uses Torchvision’s ResNet-50 FPN model with COCO weights. Only YOLO received VisDrone fine-tuning, so the comparison reflects training data as well as architecture.

Section 5 uses the first 100 validation JPGs sorted by filename. All three models are scored on five shared classes: bicycle, car, truck, bus and motor. COCO groups pedestrian and people under one person label and lacks direct matches for van, tricycle and awning-tricycle.

AP50 measures precision and recall with a box overlap threshold of 0.50. The small-object calculation keeps boxes at most 32 pixels wide **and** 32 pixels tall. The project averages AP across classes with ground truth. This is a custom subset evaluation. The three-model run uses a confidence threshold of 0.001 and reports overall AP50, small-object AP50, latency and detections per image. It also saves an accuracy-versus-speed plot.

## Recorded results

The original baseline validation reported about 27.5% mAP50 and 15.1% mAP50–95. On the historical ten-class, 100-image small-object evaluation, full-image YOLO scored **16.38% AP50**. The 512-pixel SAHI run scored **28.42%**, a gain of 12.04 percentage points. The historical tile study measured the following results.

| Tile size | Small-object AP50 | Mean time per image | Mean detections per image |
|---|---:|---:|---:|
| 256 × 256 | 13.92% | 301.31 ms | 72.27 |
| 512 × 512 | 28.42% | 133.47 ms | 52.88 |
| 640 × 640 | 20.81% | 95.42 ms | 44.39 |

The 256-pixel tiles produced the most boxes. The 512-pixel tiles gave the highest AP50. These timings depend on hardware and inference settings. Use the new five-class CSV for a comparison with Faster R-CNN. [Historical tile measurements](final_tile_comparison.csv)

![Sample VisDrone image with full-image and sliced predictions](figures/comparison_1.jpg)

![Historical tile-size accuracy and latency](figures/latency_vs_accuracy.png)

## Run in Google Colab

Select **Runtime → Change runtime type → T4 GPU**. For a demo with saved YOLO weights, run **1 → 2 → 6**. For evaluation with saved weights, run **1 → 2 → 3 → 5**. [Open the working notebook](https://colab.research.google.com/github/Ananyasaini06/Small-Object-Detection/blob/main/Small_Object_Detection.ipynb)

| Section | What to do |
|---|---|
| 1. Setup | Install the dependencies and load the project code. First use downloads pretrained weights. |
| 2. Recover weights | Mount Drive. If your trained checkpoint is elsewhere, set `manual` to its full path. |
| 3. Prepare data | Download VisDrone and locate the validation images and labels. |
| 4. Train or resume | If recovery found no usable weights, set `TRAIN=True`. To continue an interrupted Drive run, select its checkpoint in section 2 and set `RESUME=True`. Resume requires optimizer state in the original run’s last.pt. The training cell verifies best.pt and last.pt in Drive. |
| 5. Evaluate | Run the shared-class comparison. The optional switches repeat the historical protocol, full YOLO validation or tile study. Results go to a new dated folder under MyDrive/Small-Object-Detection/evaluation. |
| 6. Try an image | Upload a JPG or PNG, choose a confidence threshold and click **Compare detections**. Each panel has a download button. Counts and timings appear below. The Gradio share link is public while the demo runs. |
| 7. Back up | Save a dated ZIP to Drive with available checkpoints, this session’s results, package versions and Git revision. |

## Project files

The working notebook calls `comparison.py` for the three-model benchmark, `evaluate.py` for the original AP50 calculation, `tile_experiments.py` for the tile study and `demo.py` for Gradio. The ZIP files contain historical results and figures. The archive keeps all 55 original cells, including pilot training and saved outputs. [Original notebook](archive/Small_Object_Detection.ipynb)

## Report and live presentation

[Read the project report (PDF)](report/P10-report.pdf). For the live presentation, run section 5 before class. It saves the three-model measurements and selection.txt beside the speed–accuracy graph. Selection prioritizes shared-class small AP50, then latency, then overall AP50. Section 6 shows the latest complete comparison matching the loaded checkpoint fingerprint. Use its evidence panel for the historical training curve, epoch table, tile results and any saved training outputs alongside the checkpoint.

For the two-day submission, Faster R-CNN remains a COCO-pretrained transfer baseline. A controlled VisDrone fine-tune needs a new classification head, a separate training and validation run, and time to evaluate the resulting checkpoint. We have no measured training throughput or guaranteed T4 session for that additional experiment. The historical accuracy-first choice is SAHI 512. Run the shared-class benchmark to determine the three-way choice. Launch the temporary Gradio link during the presentation; the runtime must remain active.
