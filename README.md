# Small Object Detection in Aerial Scenes Using YOLO and SAHI

## Project Overview

Small objects such as cars and pedestrians are difficult to detect in drone images because they occupy very few pixels. This project investigates whether sliced image inference can improve small object detection compared with standard YOLO inference.

The project uses a pretrained YOLO11n model fine-tuned on the VisDrone dataset and evaluates SAHI (Slicing Aided Hyper Inference) with different tile sizes.

## Objectives

- Train a YOLO11n object detector on VisDrone.
- Compare standard inference with SAHI sliced inference.
- Study the effect of tile size on small object detection.
- Measure the trade-off between detection performance and inference latency.
- Visualize predictions to examine objects detected by each approach.

## Technologies Used

- Python
- Google Colab and NVIDIA T4 GPU
- Ultralytics YOLO11n
- SAHI
- PyTorch
- VisDrone dataset
- Pandas, NumPy, Matplotlib, and Pillow

## Methodology

1. Set up the Google Colab environment and download the VisDrone dataset.
2. Fine-tune pretrained YOLO11n for 15 epochs at an image size of 640 pixels.
3. Evaluate standard YOLO inference and SAHI sliced inference.
4. Compare tile sizes of 256 × 256, 512 × 512, and 640 × 640 pixels.
5. Record inference latency and detections per image.
6. Evaluate small object AP50 on a 100-image validation subset using a custom evaluation script.
7. Generate side-by-side visual comparisons and performance graphs.

## Results: Tile-Size Comparison

Experiments were conducted on a subset of 100 VisDrone validation images using a fine-tuned YOLO11n model and SAHI sliced inference.

### Small-Object Detection Performance

| Method | Small-Object Macro AP50 |
|---|---:|
| Normal YOLO | 16.38% |
| SAHI with 512 × 512 tiles | 28.42% |
| Absolute improvement | +12.04 percentage points |

SAHI achieved approximately 73.5% relative improvement in custom small-object AP50 over normal YOLO on this subset.

### Effect of Tile Size

| Tile Size | Small-Object Macro AP50 | Average Latency per Image | Average Detections per Image |
|---|---:|---:|---:|
| 256 × 256 | 13.92% | 301.31 ms | 72.27 |
| 512 × 512 | 28.42% | 133.47 ms | 52.88 |
| 640 × 640 | 20.81% | 95.42 ms | 44.39 |

### Observations

- The 512 × 512 tile size achieved the highest measured small-object AP50 among the tested settings.
- The 640 × 640 tile size had the lowest measured latency.
- The 256 × 256 tile size generated the most detections but was the slowest.
- Higher detection counts do not necessarily imply higher accuracy because detections can include false positives.

### Evaluation Notes

These are preliminary results from 100 validation images using a custom small-object evaluation procedure. They are not official VisDrone benchmark metrics. The custom evaluation defines a small object as a ground-truth or predicted box with both width and height no greater than 32 pixels and uses an IoU threshold of 0.50.

SAHI may automatically change its postprocessing configuration at very low confidence thresholds. Therefore, differences in postprocessing remain a limitation when comparing SAHI with standard YOLO. Full-dataset evaluation and further checks are needed before generalizing these results.
### Key Observations

- The 512 × 512 tile setting achieved the highest measured small object AP50 among the three tested settings.
- The 640 × 640 setting had the lowest average inference latency.
- The 256 × 256 setting produced the most detections but also had the highest latency.
- More detections do not necessarily indicate better accuracy because some predictions may be false positives.

These results suggest a trade-off between detection performance and inference speed.

## Limitations

- The evaluation uses a subset of 100 validation images rather than the complete validation dataset.
- Small object AP50 is calculated using a custom evaluation script and is not directly equivalent to official VisDrone mAP.
- Results may vary with hardware, confidence thresholds, overlap settings, and postprocessing.
- Further evaluation is required before drawing conclusions about full-dataset performance.

## How to Run

The experiments were developed in Google Colab.

1. Open `P10_Small_Object_Detection.ipynb` in Google Colab.
2. Select a GPU runtime.
3. Install the required packages, including Ultralytics and SAHI.
4. Download and prepare the VisDrone dataset.
5. Run the notebook cells in order.

The trained model checkpoint and dataset are not included in this repository.

## Future Improvements

- Evaluate on the complete validation dataset.
- Make confidence thresholds and postprocessing consistent across comparisons.
- Test additional tile sizes and overlap ratios.
- Measure memory usage and inference latency more systematically.
- Explore larger input resolutions and alternative detection models.

## Project Status

The initial model training, sliced inference experiments, tile-size comparison, and preliminary visual analysis are complete. Full validation and more rigorous evaluation remain future work.
