# SIH26057 — Sonar Anomaly & Object Detection

An AI-assisted side-scan sonar analysis system for detecting known underwater objects and identifying previously unseen anomalies in sonar imagery.

The system combines supervised object detection with unsupervised anomaly detection to provide a practical detection and operator-triage workflow.

## Overview

Side-scan sonar imagery can contain known objects such as aircraft and shipwrecks, while also containing previously unseen objects or unusual seabed structures.

A single supervised detector is limited to the classes it was trained on. To address this, our system uses two complementary ML approaches:

- **YOLOv8** — detects known object classes
- **PatchCore** — identifies regions that differ from normal seafloor imagery
- **Spatial fusion** — correlates detections from both branches
- **Operator triage** — categorises detections as `HIGH`, `REVIEW`, or `REJECT`

```text
                    Sonar Image
                         │
              ┌──────────┴──────────┐
              │                     │
          YOLOv8                  PatchCore
              │                     │
       Known objects         Anomaly detection
              │                     │
              │              Image-level gate
              │                     │
              │              Anomaly localization
              │                     │
              └──────────┬──────────┘
                         │
                  IoU-based Fusion
                         │
                 Detection Buckets
                         │
              ┌──────────┼──────────┐
              │          │          │
             HIGH      REVIEW      REJECT

Current Detection Classes

The current YOLO model uses two classes:

Class ID	Class
0	Aircraft
1	Shipwreck

Unknown objects are not assigned a YOLO class. They can be surfaced through the PatchCore anomaly branch.

ML Architecture
1. YOLOv8

YOLOv8 is used as the supervised object detector for known sonar objects.

Input: sonar image

Output:

bounding box
class
detector confidence

Current inference confidence threshold:

yolo_conf = 0.25
2. PatchCore

PatchCore is trained using normal seafloor imagery rather than requiring every possible anomaly to be labelled.

This allows the system to identify visually unusual regions that are not represented by the supervised YOLO classes.

The anomaly branch operates in two stages:

PatchCore
   ↓
Image-level anomaly score
   ↓
Gate
   ↓
Raw anomaly map
   ↓
Absolute threshold
   ↓
Connected components
   ↓
Candidate anomaly boxes

Current configuration:

image_gate       = 0.3103
heat_threshold   = 0.35
anomaly_map      = 256 × 256
min_area_px      = 200
merge_gap_norm   = 0.02

The heatmap threshold is an absolute threshold on the raw anomaly map.

3. Detection Fusion

YOLO and PatchCore detections are matched spatially using IoU.

fusion_iou = 0.10

A detection can therefore originate from:

both — YOLO and PatchCore agree spatially
yolo_only — detected only by YOLO
patchcore_only — detected only by PatchCore
4. Operator Triage

The fused detections are assigned one of three buckets:

HIGH
REVIEW
REJECT

The purpose of these buckets is operator triage, not to claim an improvement in raw object-detection accuracy.

Current admission thresholds:

review_yolo_only_conf       = 0.70
review_patchcore_only_amax  = 0.55
Output Contract

The ML pipeline returns one detection dictionary per candidate:

{
  "bbox": {
    "x_min": 0.41,
    "y_min": 0.22,
    "x_max": 0.53,
    "y_max": 0.38
  },
  "src": "both",
  "class_name": "aircraft",
  "yolo_conf": 0.87,
  "anomaly_score": 0.61,
  "anomaly_mean": 0.44,
  "bucket": "HIGH"
}

Bounding boxes are normalised to [0, 1].

Repository Structure

The repository is being developed with separate ML and backend components:

SIH26057/
│
├── ml/
│   ├── sonar_pipeline.py
│   ├── sih26057_ml_handoff.json
│   ├── requirements.txt
│   └── README.md
│
├── backend/
│   └── ...
│
└── README.md

The backend directory will contain the API/server layer that exposes the ML pipeline to the application.

ML Usage

The ML pipeline is designed to provide a simple inference interface:

from sonar_pipeline import SonarDetector, autodiscover

detector = SonarDetector(autodiscover())

detections = detector.detect("image.jpg")

For batch inference:

detections = detector.detect_dir("folder/")

Models are loaded once and reused across inference requests.

Model Weights

Model weights are intentionally not stored in this repository.

The ML pipeline expects:

weights/
├── best.pt
└── model.ckpt

Where:

best.pt    → trained YOLOv8 model
model.ckpt → trained PatchCore model

See the ML handoff documentation for the required checkpoints and verification information.

Backend Integration

The backend will be responsible for:

Receiving sonar imagery from the frontend.
Passing the image to the ML pipeline.
Returning the detection results as JSON.
Handling inference errors and validation.
Serving the results to the frontend/dashboard.

Conceptually:

Frontend
   │
   │ sonar image
   ▼
Backend API
   │
   ▼
SonarDetector
   │
   ├── YOLOv8
   ├── PatchCore
   └── IoU Fusion
   │
   ▼
Detection JSON
   │
   ▼
Frontend / Dashboard
Reproducibility

The ML environment should use the versions specified in:

ml/requirements.txt

The dependency versions are important because PatchCore anomaly-score calibration and threshold behaviour depend on the Anomalib environment.

The repository also includes a self-check:

python sonar_pipeline.py --selfcheck golden/golden_cases.json

Expected result:

SELFCHECK PASS

The self-check should pass before integrating the ML pipeline into the backend.

Evaluation

The current evaluation shows that the fused pipeline should not be presented as simply "better than YOLO".

Current results:

Tuned YOLOv8:
F1 = 0.894

Fused system:
F1 = 0.907

The difference is not statistically resolvable on the current evaluation.

The more meaningful result is the effect of requiring spatial corroboration:

Naive union:
Precision = 0.447
False positives = 140

Corroborated fusion:
Precision = 0.928
False positives = 8

Therefore, the key system objective is:

Reduce false-positive burden and surface both known and previously unseen sonar anomalies in an operator-friendly triage workflow.

Status
ML
 YOLOv8 known-object detector
 PatchCore anomaly detector
 Anomaly-map based candidate localization
 Spatial IoU fusion
 Confidence/triage buckets
 ML inference contract
 Golden self-check
Backend
 API integration
 Model loading/service layer
 Image upload endpoint
 ML inference endpoint
 Frontend integration
 Deployment
Future Work
Confidence calibration
Geolocation transformation
Temporal deduplication
ONNX/edge optimisation and benchmarking
Production deployment
Larger and more diverse sonar datasets
Important Notes
Do not add mine as a YOLO class. The current model is 2-class.
Do not use the older patchcore_handoff.json.
Do not use the superseded relative heatmap threshold of 0.70.
The current heatmap threshold is absolute 0.35.
Do not interpret HIGH, REVIEW, and REJECT as calibrated probabilities.
Do not claim that the system has established state-of-the-art performance based on the current evaluation.
SIH 2026 — Problem Statement 26057

Side-Scan Sonar Object and Anomaly Detection

Built as a modular ML + backend system for practical sonar-image analysis and operator-assisted underwater object detection.
