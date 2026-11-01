# End-to-End Traffic Violation Enforcement Pipeline
### Automated Motorcycle Helmet, Car Seatbelt & Mobile Phone Usage Detection

An edge-optimized, real-time computer vision enforcement pipeline tailored for **NVIDIA GeForce RTX 3050 (4 GB VRAM)** and production CCTV feeds, leveraging cascaded inference, class-separated tracking, 4-point quadrilateral check-zone geometry, and confidence-weighted temporal voting.

---

## 🚦 System Architecture & Detection Pipeline

```
[ CCTV RTSP Stream / Video Input ]
                │
                ▼
      [ Decoupled StreamBuffer ]
  (Threaded FIFO queue, zero lag,
   auto-reconnect, frame-drop)
                │
                ▼
  ┌───────────────────────────────┐
  │   Primary Object Detection    │
  │     (YOLOv8s on CUDA FP16)    │
  └──────┬─────────────────┬──────┘
         │                 │
   [Motorcycles]         [Cars]
         │                 │
         ▼                 ▼
  ┌───────────────┐ ┌───────────────┐
  │ ByteTrack #1  │ │ ByteTrack #2  │
  │ (Bike Tracker)│ │ (Car Tracker) │
  └──────┬────────┘ └──────┬────────┘
         │                 │
         ▼                 ▼
  ┌────────────────────────────────────────────────┐
  │           Check Zone Trigger Logic             │
  │   - 4-point convex polygon ray-casting         │
  │   - Bottom-center contact point tracking       │
  │   - States: APPROACHING -> IN_ZONE -> EXITED   │
  └──────────────┬──────────────────┬──────────────┘
                 │                  │
    (Inside / Traversing Zone)      │
                 │                  │
                 ▼                  ▼
  ┌────────────────────────┐ ┌──────────────────────────────────────┐
  │  Helmet Detection      │ │  Car Cabin Inspection                │
  │  - Upper 70% rider ROI │ │  - CLAHE glare suppression           │
  │  - Vector projection   │ │  - Seatbelt Classifier (384px)       │
  │    (driver vs pillion) │ │  - Driver Quadrant Isolation         │
  │  - Helmet/No-Helmet    │ │  - Distracted Driving Phone Detector │
  └──────────┬─────────────┘ └──────────────────┬───────────────────┘
             │                                  │
             ▼                                  ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │               Multi-Frame Temporal Voting Engine                │
  │  (Aggregates predictions across zone frames; eliminates noise)  │
  └──────────────────────────────┬──────────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                   Event Dispatch & Evidence                     │
  │  - Conditional ANPR License Plate Extraction (EasyOCR on CUDA)  │
  │  - SQLite Database log (`data/violations.db`)                   │
  │  - High-res cropped snapshot archive (`data/snapshots/`)        │
  │  - Live Real-Time Streamlit Dashboard (`dashboard/app.py`)      │
  └─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Repository Layout

```text
Vehicle/
├── configs/
│   ├── pipeline_config.yaml      # Resolution, VRAM thresholds, voting buffers
│   └── zones.json                # Quadrilateral check-zone polygons per camera ID
├── src/
│   ├── detection/
│   │   ├── anpr.py               # Indian ANPR regex parser, bilateral filtering & OCR
│   │   ├── phone_detector.py     # Driver quadrant phone usage / distraction model
│   │   └── secondary_detectors.py# Cascaded Helmet & Seatbelt YOLO models + adaptive rider head extractor
│   ├── geometry/
│   │   ├── movement_tracker.py   # Origin -> Destination road directional movement gating
│   │   └── zone.py               # Ray-Casting point-in-polygon & zone lifecycle
│   ├── heuristics/
│   │   ├── cabin_roi.py          # Dynamic windshield crop + LAB-space CLAHE
│   │   └── rider_filter.py       # Velocity-vector projection (driver vs pillion)
│   ├── ingestion/
│   │   └── stream_buffer.py      # Non-blocking threaded FIFO capture buffer
│   ├── tracking/
│   │   └── tracker.py            # Independent class-separated ByteTrack instances
│   ├── voting/
│   │   └── temporal_voter.py     # Rolling multi-frame confidence accumulator
│   ├── dispatch/
│   │   └── event_store.py        # SQLite event & movement database & snapshot writer
│   └── pipeline.py               # Unified end-to-end execution loop
├── tools/
│   ├── zone_calibrator.py        # Interactive OpenCV 4-point polygon calibration GUI
│   ├── train_models.py           # VRAM-safe model training runner for RTX 3050 (4 GB)
│   ├── export_engine.py          # ONNX & TensorRT FP16 export optimizer
│   ├── reencode_h264.py          # PyAV HTML5 browser-compatible H.264 transcoder
│   └── generate_sample_video.py  # Synthetic test video generator
├── dashboard/
│   └── app.py                    # Real-time Streamlit monitoring dashboard
├── tests/
│   ├── test_anpr.py              # Indian ANPR regex syntax & crop tests
│   ├── test_movement_tracker.py  # Directional road movement & displacement tests
│   ├── test_pipeline_e2e.py      # End-to-end integration test suite
│   ├── test_phone_detector.py    # Driver quadrant and distraction unit tests
│   └── test_stream_buffer.py     # Threaded stream buffer verification
├── data/
│   ├── violations.db             # SQLite citations database
│   ├── snapshots/                # High-resolution violation evidence crops
│   ├── helmet_dataset.yaml       # Helmet YOLO dataset configuration
│   ├── seatbelt_dataset.yaml     # Seatbelt YOLO dataset configuration
│   └── phone_dataset.yaml        # Distracted driving dataset configuration
├── test_core_engines.py          # Pure mathematical & geometric test suite
└── pyproject.toml                # Pinned uv project dependencies
```

---

## ⚡ Quick Start

### 1. Requirements & Environment
The project is configured via `uv` with Python 3.12 and PyTorch with CUDA 12.4 support:

```bash
# Verify environment and CUDA device
uv run python test_core_engines.py
```

### 2. Run the Pipeline

#### Live Webcam Stream (Interactive Display)
```bash
uv run python -m src.pipeline --source 0
```

#### CCTV RTSP Feed (Live Non-blocking Buffer)
```bash
uv run python -m src.pipeline --source "rtsp://admin:pass@192.168.1.100:554/stream1" --output data/output_annotated.mp4
```

#### Video File (Headless / Automated Processing)
```bash
uv run python -m src.pipeline --source data/sample_traffic.mp4 --output data/output_annotated.mp4 --headless
```

#### Driver Configuration (Right-Hand vs Left-Hand Drive)
```bash
# For India / UK / Australia (Default: RIGHT)
uv run python -m src.pipeline --source data/sample_traffic.mp4 --driver-side RIGHT

# For US / Europe / Middle East
uv run python -m src.pipeline --source data/sample_traffic.mp4 --driver-side LEFT
```

---

## 🖥️ Live Monitoring Dashboard

Launch the real-time web UI to monitor live metrics, query SQLite citation history, and inspect high-resolution evidence snapshots:

```bash
uv run streamlit run dashboard/app.py
```

Features:
- **KPI Metrics**: Total Infractions, Helmet Violations, Seatbelt Violations, Phone Violations.
- **Interactive Citation Log**: Searchable table showing Track ID, Timestamp, Vehicle Type, Violation Type, and Confidence.
- **Evidence Inspector**: View cropped high-resolution evidence images for any cited vehicle.
- **Video Playback**: Embedded replay of recorded pipeline output clips.

---

## 📐 Tools & Utilities

### 1. Interactive Check-Zone Calibrator
Calibrate a 4-point quadrilateral road zone for any camera angle:
```bash
uv run python tools/zone_calibrator.py --source data/sample_traffic.mp4 --camera-id camera_01
```
* **Left-click**: Place 4 points on the road.
* **Right-click**: Undo point.
* **`s`**: Save directly to `configs/zones.json`.
* **`c`**: Clear points.
* **`q`**: Quit.

### 2. Automated Model Training (Configured for RTX 3050 - 4 GB VRAM)
Train dedicated secondary YOLO models without running out of CUDA VRAM:
```bash
# Helmet model (imgsz=320, batch=16)
uv run python tools/train_models.py --type helmet --data data/helmet_dataset.yaml --epochs 40

# Seatbelt model (imgsz=384, batch=12)
uv run python tools/train_models.py --type seatbelt --data data/seatbelt_dataset.yaml --epochs 40

# Mobile phone model (imgsz=320, batch=16)
uv run python tools/train_models.py --type phone --data data/phone_dataset.yaml --epochs 40
```

### 3. Model Export & Optimization
Export PyTorch weights to ONNX or TensorRT FP16 engines for maximum throughput:
```bash
# Export to ONNX
uv run python tools/export_engine.py --model yolov8s.pt --format onnx --imgsz 640

# Export to TensorRT FP16 engine
uv run python tools/export_engine.py --model yolov8s.pt --format engine --imgsz 640
```

---

## 🧪 Test Suites

Run all automated test suites to verify system integrity:

```bash
# 1. Geometry, Ray-Casting & Temporal Voting
uv run python test_core_engines.py

# 2. End-to-End Pipeline & Tracker Persistence
uv run python tests/test_pipeline_e2e.py

# 3. Driver Quadrant & Phone Detector
uv run python tests/test_phone_detector.py

# 4. Threaded RTSP StreamBuffer
uv run python tests/test_stream_buffer.py

# 5. Automatic Number Plate Recognition (ANPR) & Regex Normalizer
uv run python tests/test_anpr.py
```

