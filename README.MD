# 🎯 Video Tracking Analytics Pipeline

Real-time multi-object tracking, counting, and movement analysis using **YOLOv8 + ByteTrack**, optimised for Apple M1 (MPS acceleration).

![Demo](assets/demo.gif)

> **Portfolio project** — detects and tracks objects across video frames with persistent IDs, trajectory trails, heatmaps, virtual line counters, and a live stats dashboard. Exports full tracking results as CSV/JSON.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 Multi-object detection | YOLOv8 (nano → xlarge) with MPS/CPU support |
| 🔗 Persistent tracking | ByteTrack — stable IDs across frames |
| 📊 Virtual line counter | IN/OUT counting per class with crossing events |
| 🌡️ Movement heatmap | Gaussian-blurred density overlay, PNG export |
| 🛤️ Trajectory trails | Per-object fading polyline history |
| 📈 Live HUD | FPS, active tracks, unique total, line counts |
| 💾 Full export | CSV + JSON tracking data, summary report |
| ⚡ M1 optimised | PyTorch MPS acceleration on Apple Silicon |

---

## 🛠️ Setup

### Requirements
- macOS Apple M1/M2/M3 (or Linux/Windows)
- Python 3.10+
- Conda or venv

### Install

```bash
# 1. Clone
git clone https://github.com/yourusername/video-tracking-analytics.git
cd video-tracking-analytics

# 2. Create environment
conda create -n video-tracking python=3.10 -y
conda activate video-tracking

# 3. Install dependencies
pip install -r requirements.txt
pip install -e .

# 4. Verify MPS
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

### Prepare sample videos

```bash
# Generate a synthetic test video (instant, no download)
python -m data.prepare_video --synthetic

# Inspect any video
python -m data.prepare_video --info data/sample_videos/synthetic_test.mp4

# Resize a video to 1280×720
python -m data.prepare_video --input myvideo.mp4 --width 1280 --height 720
```

---

## 🚀 Usage

### Process a video file

```bash
python -m inference.process_video \
    --config configs/tracking_config.yaml \
    --source data/sample_videos/your_video.mp4 \
    --show
```

### Live webcam stream

```bash
python -m inference.live_stream \
    --config configs/tracking_config.yaml \
    --source 0
```

### Record the live stream

```bash
python -m inference.live_stream \
    --config configs/tracking_config.yaml \
    --source 0 --save
```

### Batch process a folder

```bash
python -m inference.process_video \
    --config configs/tracking_config.yaml \
    --batch data/sample_videos/
```

### Keyboard controls (both modes)

| Key | Action |
|---|---|
| `q` | Quit |
| `h` | Toggle heatmap overlay |
| `t` | Toggle trajectory trails |
| `b` | Toggle bounding boxes |
| `l` | Toggle counting lines |
| `u` | Toggle HUD |
| `s` | Save snapshot (live mode) |

---

## ⚙️ Configuration

All parameters live in `configs/tracking_config.yaml`. Key settings:

```yaml
model:
  size: n           # n | s | m | l | x
  confidence: 0.4
  device: mps       # mps | cpu

classes:
  filter: [0, 2, 7] # 0=person  2=car  7=truck

tracker:
  type: bytetrack
  track_buffer: 30  # frames to keep lost tracks alive

counting:
  enabled: true
  lines:
    - name: "Line A"
      points: [[0.5, 0.0], [0.5, 1.0]]  # vertical center line

heatmap:
  enabled: true
  colormap: HOT     # HOT | JET | INFERNO | PLASMA | TURBO
  alpha: 0.5
```

---

## 📊 Benchmark

Tested on **MacBook Air M1** · YOLOv8n · 100 frames · `street_video.mp4`

```bash
python benchmark.py --source data/sample_videos/street_video.mp4 --frames 100
```

### Speed

```
──────────────────────────────────────────────────
│ Device │ Detect  │ Track  │ Total   │ FPS      │
──────────────────────────────────────────────────
│ MPS    │ 40.0 ms │ 1.6 ms │ 41.5 ms │ 24.1 fps │
│ CPU    │ 39.3 ms │ 1.3 ms │ 40.5 ms │ 24.7 fps │
──────────────────────────────────────────────────
```

> MPS and CPU perform comparably at this resolution because YOLOv8n is
> lightweight enough that GPU dispatch overhead is a factor. MPS advantage
> becomes significant with larger models (yolov8s/m) or higher resolutions.

### Tracking Metrics

```
──────────────────────────────────────
│ Metric               │ Value       │
──────────────────────────────────────
│ Unique tracks        │ 25          │
│ ID switches          │ 37          │
│ Total matched frames │ 1245        │
│ Avg track duration   │ 49.8 frames │
│ MOTA (proxy)         │ 97.03 %     │
│ MOTP (proxy)         │ 32.77 %     │
──────────────────────────────────────
```

> MOTA/MOTP are self-consistency proxies (no ground-truth annotations).
> For official scores, evaluate against MOT17 using `py-motmetrics`.

---

## 📂 Project Structure

```
video-tracking-analytics/
├── assets/
│   └── demo.gif                # portfolio demo
├── data/
│   ├── prepare_video.py        # download / validate / resize / synthetic
│   └── sample_videos/          # input videos (gitignored)
├── tracking/
│   ├── detector.py             # YOLOv8 wrapper → (N,6) detections
│   ├── tracker.py              # ByteTrack wrapper → TrackedObject list
│   ├── motion_estimator.py     # velocity, speed, direction per track
│   └── trajectory.py           # centroid history store
├── analytics/
│   ├── counter.py              # virtual line crossing counter
│   ├── statistics.py           # rolling FPS, counts, speed stats
│   ├── heatmap_generator.py    # density heatmap, PNG export
│   └── report_generator.py     # CSV / JSON / TXT export
├── inference/
│   ├── process_video.py        # offline pipeline
│   ├── live_stream.py          # real-time webcam pipeline
│   └── visualization.py        # all drawing (boxes, trails, HUD)
├── configs/
│   └── tracking_config.yaml    # master config
├── results/                    # auto-generated (gitignored)
│   ├── tracked_videos/
│   ├── heatmaps/
│   ├── statistics/
│   └── reports/
├── weights/                    # YOLO weights (gitignored)
├── benchmark.py                # speed + MOT metrics
├── setup.py
└── requirements.txt
```

---

## 📤 Output Files

After processing a video named `street.mp4`:

| File | Description |
|---|---|
| `results/tracked_videos/street_tracked.mp4` | Annotated output video |
| `results/heatmaps/street_heatmap.png` | Movement density heatmap |
| `results/reports/street_tracks.csv` | Per-track flat table |
| `results/reports/street_tracks.json` | Full trajectory data |
| `results/statistics/street_summary.json` | Machine-readable summary |
| `results/statistics/street_summary.txt` | Human-readable report |

---

## 🏗️ Tech Stack

- **Detection**: [YOLOv8](https://github.com/ultralytics/ultralytics) via `ultralytics`
- **Tracking**: [ByteTrack](https://github.com/ifzhang/ByteTrack) via `boxmot`
- **Vision**: OpenCV, NumPy, SciPy
- **Acceleration**: PyTorch MPS (Apple Silicon)

---

## 📄 License

MIT License — free to use and adapt.