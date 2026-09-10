---
title: HydroSentry Backend
emoji: 🌊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# SIH26057 — Sonar Anomaly & Debris Detection

## Overview

A software-based **side-scan sonar anomaly detection system** for identifying underwater objects and previously unseen anomalies in sonar imagery.

The system combines:

* **YOLOv8** for known-object detection
* **PatchCore** for anomaly detection and localization
* **Dual-branch fusion** to distinguish corroborated detections from single-model detections
* A lightweight **operator triage system** using `HIGH`, `REVIEW`, and `REJECT` buckets

The ML pipeline is designed around the practical challenge of **limited labeled sonar data**, where a supervised detector alone cannot reliably cover every possible underwater object.

---

## Repository Structure

```text
SIH26057/
│
├── ml/
│   ├── sonar_pipeline.py
│   ├── sih26057_ml_handoff.json
│   ├── requirements.txt
│   ├── README.md
│   └── golden/
│       └── golden_cases.json
│
├── backend/
│   └── # Backend/API integration
│
└── README.md
```

> The `backend/` folder is reserved for the backend integration and will be added separately.

---

# ML Pipeline

```text
                    Side-Scan Sonar Image
                              |
              +---------------+---------------+
              |                               |
              v                               v
          YOLOv8                         PatchCore
       Known Objects                    Anomaly Detection
              |                               |
              |                         Image Score
              |                               |
              |                         Gate = 0.3103
              |                               |
              |                         Raw Anomaly Map
              |                               |
              |                       Threshold = 0.35
              |                               |
              |                    Connected Components
              |                               |
              |                         Min Area = 200
              |                               |
              |                       Merge Nearby Boxes
              |                               |
              +---------------+---------------+
                              |
                              v
                       Box-Level Fusion
                         IoU >= 0.10
                              |
                +-------------+-------------+
                |             |             |
                v             v             v
              BOTH       YOLO_ONLY    PATCHCORE_ONLY
                |             |             |
                +-------------+-------------+
                              |
                              v
                    Operator Triage Bucket
                              |
                 +------------+------------+
                 |            |            |
                HIGH        REVIEW       REJECT
```

---

# Models

## YOLOv8

YOLOv8 handles **known-object detection**.

Current classes:

```text
0 = aircraft
1 = shipwreck
```

There is **no mine class** in the current two-class model.

YOLO inference confidence threshold:

```text
0.25
```

---

## PatchCore

PatchCore is used as the **unknown-anomaly detection branch**.

It learns the appearance of normal seafloor imagery and identifies regions that deviate from that normal distribution.

The PatchCore branch operates at two levels:

1. **Image-level anomaly score**
2. **Pixel/region-level anomaly localization**

The image-level gate is:

```text
0.3103
```

If the image anomaly score is below this threshold, the PatchCore anomaly branch is disabled for that image.

---

# Anomaly Localization

PatchCore's raw anomaly map is processed as follows:

```text
Raw PatchCore anomaly map
        |
        v
256 × 256 resolution
        |
        v
Absolute threshold = 0.35
        |
        v
Binary anomaly mask
        |
        v
Connected components
        |
        v
Remove components < 200 px
        |
        v
Merge boxes within 0.02 normalized distance
        |
        v
Candidate anomaly boxes
```

The threshold is **absolute**, not per-image normalized.

---

# Detection Fusion

YOLO detections and PatchCore anomaly boxes are matched using:

```text
IoU >= 0.10
```

The resulting source is one of:

```text
both
yolo_only
patchcore_only
```

### `both`

Both YOLO and PatchCore identify the same region.

This provides stronger corroboration and maps to the highest-confidence operator bucket.

### `yolo_only`

Only YOLO detects the region.

A YOLO-only detection must meet:

```text
YOLO confidence >= 0.70
```

to enter the `REVIEW` bucket.

### `patchcore_only`

Only PatchCore detects the region.

A PatchCore-only anomaly must meet:

```text
anomaly_score >= 0.55
```

to enter the `REVIEW` bucket.

---

# Final Output

Each detection follows this structure:

```json
{
  "bbox": {
    "x_min": 0.41,
    "y_min": 0.22,
    "x_max": 0.53,
    "y_max": 0.38
  },
  "src": "both",
  "class_name": "shipwreck",
  "yolo_conf": 0.87,
  "anomaly_score": 0.61,
  "anomaly_mean": 0.44,
  "bucket": "HIGH"
}
```

Bounding boxes are normalized to:

```text
0–1
```

The frontend can multiply these coordinates by the rendered image dimensions.

---

# Operator Triage

The system intentionally separates model output from the final operator-facing decision.

```text
HIGH
```

Both branches corroborate the detection.

```text
REVIEW
```

Only one branch detects the object, but the detection exceeds the corresponding admission threshold.

```text
REJECT
```

The detection does not satisfy the required admission criteria.

`REJECT` detections are still returned by the ML module. The backend/frontend can decide whether to hide them, collapse them, or expose them to the operator.

---

# Current ML Results

The current evaluated pipeline produced:

```text
Total candidate boxes: 80

HIGH_CONFIDENCE: 30
REVIEW_REQUIRED: 35
REJECT: 15
```

Final confidence statistics:

```text
Mean:   0.624581
Std:    0.192077
Min:    0.334800
Max:    1.000000
```

The physics-verification experiment was evaluated separately and is **not part of the current production pipeline**.

---

# Important: What We Can Claim

The current system should **not** be presented as definitively beating YOLOv8.

On the available evaluation:

```text
Best F1:
Fusion pipeline = 0.907
Tuned YOLOv8    = 0.894
```

However, the difference was **not statistically resolvable** on the available data.

The stronger demonstrated result is the effect of requiring model corroboration:

```text
Naive union:
Precision = 0.447
False positives = 140

Corroborated detections:
Precision = 0.928
False positives = 8
```

Therefore, the primary demonstrated value is **better precision / operator triage through fusion**, rather than claiming a definitive overall detection-performance improvement over YOLOv8.

---

# Why PatchCore?

A conventional supervised detector is constrained by its labeled classes.

For underwater sonar imagery, previously unseen objects can appear that were not represented in the training labels.

PatchCore provides a complementary mechanism:

```text
Known object
    ↓
YOLOv8
    ↓
classified detection


Previously unseen / unusual region
    ↓
PatchCore
    ↓
unknown anomaly
```

This allows the system to surface potentially relevant anomalies outside the known-object classes.

---

# Current ML Status

## Completed

* YOLOv8 known-object detection
* Two-class YOLO model
* PatchCore training
* PatchCore image-level anomaly detection
* PatchCore anomaly localization
* Connected-component box generation
* Anomaly-box filtering
* YOLO + PatchCore fusion
* Confidence scoring
* HIGH / REVIEW / REJECT decision buckets
* Golden-case validation
* ML/backend handoff package

## Not Part of Current Pipeline

* Physics verification
* Physics-based filtering
* Previous 3-class `mine` classification
* Previous `patchcore_handoff.json`
* Per-image relative heatmap thresholding

---

# Configuration

All production configuration values are stored in:

```text
sih26057_ml_handoff.json
```

The important values are:

```text
yolo_conf                  = 0.25
image_gate                 = 0.3103
heat_threshold_abs         = 0.35
min_area_px                = 200
merge_gap_norm             = 0.02
fusion_iou                 = 0.10
review_yolo_only_conf      = 0.70
review_patchcore_only_amax = 0.55
anomaly_map_hw             = 256 × 256
```

The JSON configuration should be treated as the **source of truth** rather than duplicating these values throughout the codebase.

---

# ML Usage

```python
from sonar_pipeline import SonarDetector, autodiscover

detector = SonarDetector(autodiscover())

detections = detector.detect("image.jpg")
```

For batch inference:

```python
detections = detector.detect_dir("folder/")
```

Batch inference is preferred when processing multiple sonar images because the models are loaded once.

---

# Model Files

The repository does not need to store large model weights directly in Git.

The expected model files are:

```text
weights/
├── best.pt
└── model.ckpt
```

Where:

```text
best.pt
    YOLOv8 two-class checkpoint

model.ckpt
    PatchCore seafloor checkpoint
```

Model checksums and expected versions are documented in the ML handoff package.

---

# Backend Integration

The backend should treat the ML module as an inference service/module rather than implementing ML logic itself.

Expected flow:

```text
Frontend
   |
   | sonar image
   v
Backend API
   |
   | image
   v
ML Pipeline
   |
   +--> YOLOv8
   |
   +--> PatchCore
   |
   +--> Fusion
   |
   v
Detection JSON
   |
   v
Backend
   |
   v
Frontend / Database
```

The backend should consume the output contract from `sonar_pipeline.py` and avoid reimplementing:

* YOLO inference
* PatchCore inference
* anomaly-map processing
* box generation
* fusion logic
* confidence calculation
* decision thresholds

---

# Validation

Before integration, run:

```bash
python sonar_pipeline.py --selfcheck golden/golden_cases.json
```

Expected output:

```text
SELFCHECK PASS
```

If the self-check fails, the ML integration should be stopped and investigated before proceeding.

---

# Reproducibility

The ML environment depends on specific library versions.

Install:

```bash
pip install -r requirements.txt
```

Do not casually change the pinned versions.

In particular, PatchCore anomaly-score behavior depends on the Anomalib version used during development.

---

# Current Goal

The goal is **not simply to run YOLO on sonar images**.

The intended contribution is a practical sonar detection system that combines:

```text
Supervised detection
        +
Unsupervised anomaly detection
        +
Spatial localization
        +
Cross-model corroboration
        +
Operator triage
```

The system is therefore designed to handle both:

```text
Known objects
    → aircraft / shipwreck

Unknown anomalies
    → potentially relevant objects or debris not represented
      in the supervised training classes
```

The long-term evaluation goal is to determine whether this architecture can provide a meaningful advantage over a conventional supervised-only baseline, while being explicit about what the current experiments do and do not establish.

---

# Current Roadmap

```text
[✓] Dataset preparation
[✓] YOLOv8 baseline
[✓] PatchCore anomaly detector
[✓] Anomaly localization
[✓] YOLO + PatchCore fusion
[✓] Confidence / triage logic
[✓] Golden-case validation
[✓] Backend handoff

[ ] Backend integration
[ ] Frontend/dashboard integration
[ ] End-to-end API testing
[ ] Deployment / model serving
[ ] ONNX / edge benchmarking
[ ] Geolocation transformation
[ ] Temporal deduplication
[ ] Final comparative evaluation
[ ] Final demo
```

---

# Repository Principle

Keep the architecture modular:

```text
ML
 ↓
sonar_pipeline.py
 ↓
structured detection output
 ↓
Backend
 ↓
API / storage / orchestration
 ↓
Frontend
```

The ML layer should remain independently testable and reproducible, while the backend owns API handling, persistence, authentication/orchestration, and communication with the frontend.
