# Smart Surveillance & Crowd Analytics System
### Lab #14 — Complex Computing Activity
**Project Name:** Smart Surveillance & Crowd Analytics System
**Roll No:** 23-ai-26, 23-ai-04, 23-ai-50
**Course:** Computer Vision

---

## Project Overview

This project implements a real-time crowd analytics pipeline using a fixed surveillance camera feed. It detects people using YOLOv8, tracks them across frames using a custom centroid tracker, monitors a defined restricted zone, and raises visual alerts when crowd density exceeds a set threshold. The system outputs a fully annotated video, a density-over-time graph, and a heatmap of movement patterns.

---

## Folder Structure

```
Lab14_Surveillance/
│
├── Lab14_Surveillance.ipynb       # Main Jupyter Notebook (all code)
├── README.md                      # This file
│
├── outputs/
│   ├── output_annotated.avi       # Annotated output video
│   ├── classical_cv_analysis.png  # Gaussian / Canny / Sobel grid
│   ├── density_over_time.png      # Crowd density graph
│   └── fps_benchmark.png          # Processing FPS chart
```

---

## Lab Concepts Integrated

| Lab # | Concept | Where Used |
|-------|---------|------------|
| Lab 3 | Gaussian Blur + Canny + Sobel Edge Detection | Cell 5 — Scene structure analysis |
| Lab 7 | Deep Learning Object Detection (YOLOv8) | Cell 6, 11 — Person detection per frame |
| Lab 9 | Object Tracking (Centroid-based) | Cell 8, 11 — ID assignment and trail drawing |
| Lab 2 | Video Pipeline (VideoCapture / VideoWriter) | Cell 4, 11 — Frame-level processing loop |
| Lab 6 | Spatial Analysis + Heatmap | Cell 9, 11 — Zone check + density accumulator |

---

## Requirements

### Python Version
```
Python 3.8 or higher
```

### Required Libraries

```bash
pip install ultralytics
pip install opencv-python
pip install numpy
pip install matplotlib
```

> **Note:** If running on Google Colab or Kaggle, `opencv-python`, `numpy`, and `matplotlib` are pre-installed. Only `ultralytics` needs to be installed.

```python
# In Colab — run this in the first cell
!pip install ultralytics -q
```

### YOLOv8 Model Weight
The model (`yolov8n.pt`) is downloaded automatically on first run by the `ultralytics` library. No manual download needed. Requires an internet connection on first execution (~6 MB).

---

## How to Run

### Option A — Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `Lab14_Surveillance.ipynb` via `File → Upload Notebook`
3. Go to `Runtime → Change runtime type → T4 GPU` (if available, for faster processing)
4. Run Cell 1 to install dependencies
5. Run Cell 3 — a file upload button will appear. Upload your `.mp4` video
6. Run all remaining cells in order (Cell 4 → Cell 15)
7. Run Cell 15 to download all output files to your local machine

### Option B — Kaggle Notebooks

1. Create a new notebook at [kaggle.com](https://kaggle.com)
2. Under Settings → Accelerator → select **GPU T4 x2**
3. Upload your video file in the **Data** panel (right sidebar)
4. Change `VIDEO_PATH` in Cell 3 to:
   ```python
   VIDEO_PATH = "/kaggle/input/your-dataset-name/your_video.mp4"
   ```
5. Run all cells in order

### Option C — Local Machine (CPU only)

```bash
# 1. Clone or copy the project folder
cd Lab14_Surveillance

# 2. Install dependencies
pip install ultralytics opencv-python numpy matplotlib

# 3. Launch Jupyter
jupyter notebook Lab14_Surveillance.ipynb

# 4. In Cell 3, replace the upload cell with:
VIDEO_PATH = "path/to/your/video.mp4"
```

> AMD GPU users: PyTorch does not support AMD GPUs on Windows via CUDA. The system will automatically fall back to CPU. Set `PROCESS_EVERY_N = 3` in Cell 11 to compensate for slower CPU speed.

---

## Key Configuration Parameters

These are the main values you may want to adjust before running:

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `ZONE_POINTS` | Cell 7 | Door area coords | Polygon defining the monitored zone |
| `ALERT_THRESHOLD` | Cell 7 | `3` | Person count that triggers the alert |
| `CONF_THRESHOLD` | Cell 11 | `0.40` | YOLO confidence cutoff (0.0–1.0) |
| `PROCESS_EVERY_N` | Cell 11 | `2` | Process 1 in every N frames (1 = all frames) |
| `max_disappeared` | Cell 8 | `5` | Frames before a lost track is deleted |
| `max_distance` | Cell 8 | `90` | Max pixel distance for track matching |

---

## Output Files Description

| File | Description |
|------|-------------|
| `output_annotated.avi` | Full annotated video with bounding boxes, trails, heatmap, zone, HUD, and alert |
| `classical_cv_analysis.png` | 3×4 grid showing Blur, Canny, Sobel on 3 sample frames |
| `density_over_time.png` | Line chart of zone occupancy across all frames with alert threshold line |
| `fps_benchmark.png` | Per-frame processing FPS with mean and minimum markers |

---

## Known Limitations

- YOLOv8n may miss heavily occluded people in dense crowds
- Centroid tracker can swap IDs when two people cross paths closely
- On Colab CPU, processing speed is approximately 2–6 FPS (real-time requires GPU)
- Heatmap is cumulative — does not distinguish between different time periods

---

## References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com)
- [OpenCV Documentation](https://docs.opencv.org)
- [COCO Dataset — Class Labels](https://cocodataset.org)
