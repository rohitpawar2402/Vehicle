# End-to-End Traffic Violation Detection Pipeline: Helmet & Seatbelt Enforcement

## 1. System Architecture & End-to-End Flow

```
[ CCTV RTSP Stream / Video Input ]
                │
                ▼
      [ Frame Ingestion & Queue ]
   (Decoupled thread, frame-skipping,
     resolution normalization 1080p)
                │
                ▼
 ┌───────────────────────────────┐
 │   Primary Object Detection    │
 │ (YOLOv8/v11 Multi-Class Detect)│
 └──────┬─────────────────┬──────┘
        │                 │
  [Motorcycles]         [Cars]
        │                 │
        ▼                 ▼
 ┌───────────────┐ ┌───────────────┐
 │ BoT-SORT #1   │ │ BoT-SORT #2   │
 │ (Bike Tracker)│ │ (Car Tracker) │
 └──────┬────────┘ └──────┬────────┘
        │                 │
        ▼                 ▼
 ┌────────────────────────────────────────────────┐
 │           Check Zone Trigger Logic             │
 │   - Road polygon defined via homography/ROI   │
 │   - Bounding-box bottom-center crossing check │
 └──────────────┬──────────────────┬──────────────┘
                │                  │
    (Inside / Traversing Zone)     │
                │                  │
                ▼                  ▼
 ┌────────────────────────┐ ┌────────────────────────┐
 │ Secondary YOLO: Helmet │ │ Secondary YOLO: Seatbelt│
 │ 1. Crop bike bounding  │ │ 1. Crop windshield/cabin│
 │ 2. Detect heads/helmets│ │ 2. Detect driver seatbelt│
 │ 3. Spatial sort: front │ │ 3. Handle glare/tinted  │
 │    rider = driver      │ │    reflections          │
 └──────────┬─────────────┘ └──────────┬─────────────┘
            │                          │
            ▼                          ▼
 ┌───────────────────────────────────────────────────┐
 │       Multi-Frame Temporal Voting Engine          │
 │ (Aggregates N frames while in zone; filters noise)│
 └──────────────────────────┬────────────────────────┘
                            │
                            ▼
 ┌───────────────────────────────────────────────────┐
 │          Violation Verdict & Event Dispatch       │
 │ - Bike: HELMET / NO HELMET (Driver only)          │
 │ - Car:  SEATBELT / NO SEATBELT                    │
 └──────────────────────────┬────────────────────────┘
                            │
                            ▼
 ┌───────────────────────────────────────────────────┐
 │           Live Dashboard & Event Store            │
 │ - High-res violation crop + annotated frame       │
 │ - SQLite / PostgreSQL event log with timestamps   │
 │ - Live compliance KPIs, counters & zoom overlay   │
 └───────────────────────────────────────────────────┘
```

---

## 2. Core Technical Components & Engineering Logic

### Component A: Frame Ingestion & RTSP Handling
* **Decoupled Buffer Queue**: CCTV RTSP feeds drop packets or suffer latency if inference blocks ingestion. Use a dedicated worker thread with `cv2.VideoCapture` that continuously drains frames into a fixed-length FIFO queue (`collections.deque(maxlen=2)`), dropping stale frames when the queue is full.
* **Resolution Strategy**: Process detection at $1280 \times 720$ or $1920 \times 1080$ to preserve small-object details (helmet straps, seatbelt bands). Avoid downscaling to standard $640 \times 640$ unless deploying TensorRT on low-compute edge devices (e.g., Jetson Orin).

### Component B: Dual-Tracker Separation (BoT-SORT)
* Running separate tracker instances for `car` and `motorcycle` prevents cross-class ID switches in dense traffic.
* **Camera Motion Compensation (CMC)**: BoT-SORT incorporates camera motion estimation (via sparse optical flow), stabilizing IDs during camera vibrations caused by wind or pole sway common in real CCTV deployments.

### Component C: Check Zone & Spatial Geometry
* **Polygon Representation**: Define the check zone as a 4-point convex quadrilateral on the road plane where the camera angle provides an optimal, unoccluded view into the car cabin and rider's head.
* **Trigger Mechanism**:
  1. Compute the bottom-center coordinate $(x_c, y_{max})$ of each tracked vehicle bounding box.
  2. Use the Ray-Casting algorithm (`cv2.pointPolygonTest`) to determine entry, presence, and exit.
  3. Status states: `OUTSIDE` $\rightarrow$ `ENTERED` $\rightarrow$ `IN_ZONE` (sampling frames) $\rightarrow$ `EXITED` (final verdict locked).

### Component D: Driver vs. Passenger Disambiguation (Bikes)
* **The Problem**: A pillion passenger wearing or not wearing a helmet must not flip the driver's compliance record.
* **Spatial Disambiguation Logic**:
  1. Within the motorcycle bounding box, run the Helmet/Head detector.
  2. For multiple detected heads/riders, compute the motion trajectory vector $\vec{v}$ of the bike.
  3. The driver is consistently the foremost rider along the travel vector (closest to the front axle/handlebars) and lower or level in relative vertical offset.
  4. Only the head bounding box assigned to the front index is queried for the `Helmet` vs `No-Helmet` class label.

### Component E: Windshield Localization & Seatbelt Detection (Cars)
* **Two-Stage Cascaded Inference**:
  1. Primary model crops the vehicle bounding box.
  2. Upper-body / windshield sub-crop ($[0:0.6 \cdot H, 0:1.0 \cdot W]$) is passed to the specialized seatbelt model.
  3. Detect classes: `Seatbelt`, `No-Seatbelt`, `Driver_Torso`.
* **Glare & Reflection Robustness**: Apply adaptive histogram equalization (CLAHE) on the cabin crop to mitigate windshield glare before feeding it into the secondary classifier.

### Component F: Temporal Voting Engine
* Never issue a violation citation on a single frame.
* While vehicle $ID_k$ is inside the check zone, collect predictions across $M$ consecutive frames (typically 5–15 frames).
* Apply confidence-weighted majority voting:
  $$\text{Score} = \sum_{t=1}^M \text{Conf}_t \cdot \mathbb{I}(\text{Class}_t = \text{Violation})$$
* A violation is registered only if the aggregated confidence passes a strict threshold (e.g., $\ge 0.75$).

---

## 3. 20-Day Implementation Schedule (Day-by-Day Work Plan)

| Phase | Days | Focus Area |
| :--- | :--- | :--- |
| **Phase 1** | Days 1–4 | Dataset Curation, Annotation & Environment Setup |
| **Phase 2** | Days 5–9 | Model Training, Fine-Tuning & Evaluation |
| **Phase 3** | Days 10–13 | Video Pipeline, BoT-SORT Tracking & Check-Zone Logic |
| **Phase 4** | Days 14–17 | Two-Stage Cascaded Inference & Temporal Voting Logic |
| **Phase 5** | Days 18–20 | Live Dashboard, Edge Optimization & Real-World Validation |

---

### Detailed Day-by-Day Breakdown

#### Phase 1: Environment Setup & Dataset Preparation (Days 1–4)
* **Day 1: System Baseline & Environment Setup**
  * Install PyTorch (CUDA-accelerated), Ultralytics YOLO, ByteTrack/BoT-SORT, OpenCV, and SuperVision.
  * Set up RTSP ingestion test scripts with threading and frame-drop logic.
  * Benchmark baseline FPS on target GPU (RTX 4090 or Jetson Orin).
* **Day 2: CCTV Video Acquisition & Frame Extraction**
  * Collect real CCTV footage across diverse conditions: bright sunlight, overcast, evening, and varied camera mounting angles ($15^\circ$ to $45^\circ$).
  * Extract representative candidate frames (dense traffic, high-speed movement, occluded vehicles).
* **Day 3: Helmet Dataset Annotation**
  * Curate and format annotations for: `motorcycle`, `rider_head_helmet`, `rider_head_no_helmet`.
  * Ensure balanced representation of full-face helmets, half-helmets, turbans/caps, and bare heads.
* **Day 4: Seatbelt Dataset Annotation**
  * Curate vehicle cabin crops.
  * Annotate classes: `seatbelt_on`, `seatbelt_off`, `windshield_obscured`.
  * Augment data with contrast adjustments, glare overlays, and shadow patterns.

#### Phase 2: Model Training & Evaluation (Days 5–9)
* **Day 5: Vehicle Base Detection (YOLOv8x/v11x)**
  * Train or fine-tune vehicle detector for `car`, `motorcycle`, `bus`, `truck`.
  * Evaluate mAP@0.5 and mAP@0.5:0.95 on held-out CCTV test splits.
* **Day 6: Helmet Detection Model Training**
  * Train YOLOv8m/v11m on rider-head crops.
  * Optimize anchors and hyper-parameters for small-target detection.
* **Day 7: Seatbelt Detection Model Training**
  * Train secondary seatbelt model on driver torso/cabin ROI.
  * Target precision $> 92\%$ to minimize false positives on dark clothing.
* **Day 8: Model Quantization & Export**
  * Export models to ONNX and compile to TensorRT engine (`FP16` precision) for ultra-low latency inference.
* **Day 9: Offline Batch Validation**
  * Run offline inference over 1-hour uncut CCTV video.
  * Compute confusion matrices and identify failure modes (glare, rain, heavy occlusion).

#### Phase 3: Tracking & Spatial Zone Geometry (Days 10–13)
* **Day 10: Multi-Object Tracking Pipeline (BoT-SORT)**
  * Integrate BoT-SORT tracker with camera motion compensation.
  * Tune track buffer, minimum matching threshold, and lost-frame grace periods.
* **Day 11: Interactive Road Calibration Tool**
  * Build a lightweight configuration tool to draw and save 4-point quadrilateral check zones on any camera angle.
  * Store polygon coordinates in a JSON configuration per camera ID.
* **Day 12: Trajectory & Zone Ingestion Logic**
  * Implement point-in-polygon checks for vehicle bounding box contact points.
  * Track vehicle states: `APPROACHING`, `INSIDE_ZONE`, `EVALUATED`, `EXITED`.
* **Day 13: ID Switch Prevention & Stress Testing**
  * Test tracker persistence during stop-and-go traffic and overlapping vehicles.
  * Prevent duplicate event creation for vehicles idling in the check zone.

#### Phase 4: Cascaded Inspection & Temporal Aggregation (Days 14–17)
* **Day 14: Bike Rider Disambiguation Pipeline**
  * Implement spatial ordering logic: calculate vehicle forward vector, assign front-most head as driver, ignore rear pillions.
  * Verify edge case: solo driver vs. driver + 1 passenger vs. triple-riding.
* **Day 15: Car Windshield Cropping & Dynamic ROI**
  * Implement dynamic windshield cropping based on car bounding box aspect ratio and orientation.
  * Integrate CLAHE / gamma correction preprocessing for tinted glass and glare.
* **Day 16: Temporal Voting & Confidence Aggregator**
  * Write the rolling voting buffer for each active track ID inside the zone.
  * Apply decaying exponential weights or median filtering over 8–12 frames.
* **Day 17: End-to-End Pipeline Integration**
  * Combine Vehicle Detector + Dual BoT-SORT + Secondary Classifiers into a unified pipeline.
  * Profile end-to-end processing latency per frame (target $\le 33\text{ ms}$ for 30 FPS).

#### Phase 5: Dashboard, Event Logging & Edge Deployment (Days 18–20)
* **Day 18: Live Dashboard UI (Streamlit / FastAPI + React)**
  * Build real-time video feed with violation highlight boxes.
  * Add live zoom picture-in-picture window focused on the latest detected violation.
  * Display running KPIs: total vehicle count, helmet compliance %, seatbelt compliance %.
* **Day 19: Event Logging & Evidence Export**
  * Save violation crops, timestamped video clips (3 seconds before/after), vehicle ID, and confidence score.
  * Implement SQLite / PostgreSQL schema and automated CSV/PDF report generator.
* **Day 20: Real-World CCTV Stress Testing & Tuning**
  * Deploy pipeline on live CCTV RTSP stream across morning, midday, and twilight.
  * Fine-tune detection thresholds to eliminate false violation alarms.
  * Document operational runbook and camera positioning guidelines.

---

## 4. Key Edge Challenges & Mitigation Strategies

| Challenge | Real-World Impact | Engineering Solution |
| :--- | :--- | :--- |
| **Windshield Glare / Tint** | Driver torso obscured; false "no seatbelt" verdicts | Dynamic ROI contrast stretching (CLAHE); polarized camera filters; multi-frame accumulation |
| **Pillion Helmet Status** | Passenger without helmet triggers violation on driver | Vector projection along travel path: isolate front-most head bounding box only |
| **Camera Pole Sway** | False trajectory drift and bounding-box jitter | BoT-SORT camera motion compensation (CMC) using optical flow |
| **ID Switching in Congestion** | One vehicle inherits another's violation status | Class-specific tracker instances; tight spatial distance gating in Kalman filter |
| **Dark Clothing vs Black Belt** | Contrast loss causes false seatbelt violation | Train model on high-contrast edge features and texture gradients; require higher confidence threshold |

---

## 5. Recommended Production Tech Stack

* **Core Models**: Ultralytics YOLOv8 / YOLOv11 (`yolov8x` for base vehicle, `yolov8m` for helmet/seatbelt)
* **Tracking**: BoT-SORT (ReID-enabled + Camera Motion Compensation)
* **Runtime Acceleration**: TensorRT / ONNX Runtime (`FP16` execution)
* **Vision & Geometry**: OpenCV, SuperVision (`supervision.PolygonZone`, `ByteTrack`)
* **Backend & API**: Python 3.11, FastAPI, Asyncio, SQLite/PostgreSQL
* **Frontend Dashboard**: Streamlit (rapid prototype) or React/Vite + WebSockets (production)
