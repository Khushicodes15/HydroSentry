# SIH26057 - Smart Scan Strategy: ML handoff

Side-scan sonar debris/anomaly detection. YOLOv8 (closed set) + PatchCore (anomaly) ->
spatial fusion -> decision bucket. This package is the ML deliverable: model, weights,
output contract, measured performance, and a day-one integration test.

## Read this before you build on it

This pipeline does **not** outperform a well-tuned YOLOv8 alone on our evaluation data.
Best F1 0.907 vs 0.894, and that difference is not statistically resolvable
(95% CI [-0.021, +0.049], bootstrap over 124 source scenes).

What *is* measured and resolvable: the fusion operator matters enormously. On identical
models, a naive union of the two branches scores F1 0.612 (precision 0.447, 140 false
positives) while requiring the two branches to corroborate scores F1 0.907 (precision
0.928, 8 false positives). That is the finding this system demonstrates.

Do not quote an earlier "F1 0.909 vs 0.822" figure from any older notebook or manifest.
It was measured on a leaked split and is invalid. See `data_leakage` in the manifest.

## Get the weights first

Weights are not in this repo (the PatchCore checkpoint is ~315 MB, over GitHub's file limit).
Download both files from the tagged Release into `ml/weights/`, then verify their checksums
against `ml/weights/README.md`. `autodiscover()` finds them there automatically.

## Quickstart

    pip install -r requirements.txt
    python sonar_pipeline.py --selfcheck golden/golden_cases.json    # must print SELFCHECK PASS
    python sonar_pipeline.py path/to/image.jpg --out result.json
    python sonar_pipeline.py path/to/folder/ --buckets HIGH,REVIEW --out results.json

From Python:

    from sonar_pipeline import SonarDetector, autodiscover
    det = SonarDetector(autodiscover())          # finds ./weights/best.pt and ./weights/model.ckpt
    dets = det.detect("image.jpg")               # single image
    all_ = det.detect_dir("folder/")             # batch - the anomaly branch runs once, much faster

**Run the selfcheck first, on your own host.** If it fails, a library version drifted and
every threshold in the manifest is invalid. It will not raise an exception on its own.

## Output contract

Each detection:

    {"bbox": {"x_min": 0.41, "y_min": 0.22, "x_max": 0.53, "y_max": 0.38},   # normalised 0-1
     "src": "both",                    # "both" | "yolo_only" | "patchcore_only"
     "class_name": "shipwreck",        # aircraft | shipwreck; null for patchcore_only
     "yolo_conf": 0.87,                # null for patchcore_only
     "anomaly_score": 0.61,            # max anomaly value inside bbox
     "anomaly_mean": 0.44,
     "bucket": "HIGH"}                 # HIGH | REVIEW | REJECT

Buckets: HIGH = both branches agree. REVIEW = one branch only, above its admit threshold
(yolo_conf >= 0.70, or anomaly_score >= 0.55). REJECT = everything else.

There is no `mine` class. Class index 1 is `shipwreck`.

`detect()` returns REJECT detections too - filter them yourself.

## Two behaviours that will surprise you

**The image gate can suppress good detections.** If PatchCore's image score is below 0.3103
the anomaly branch contributes nothing, so every YOLO box becomes `yolo_only` and is only
promoted to REVIEW at conf >= 0.70. A confident-ish detection at conf 0.4 on such an image
is bucketed REJECT. If recall matters more than operator load, surface REJECT boxes.

**The bucket structure is a triage workflow, not a performance gain.** HIGH+REVIEW combined
measures the same as YOLO alone (F1 +0.003, CI [-0.030, +0.035]). Its value is telling an
operator which detections need eyes on them, not finding more objects.

## Do not use

- Any dataset directory named `Mine-detection-1` - stale 3-class data, source of a wrong
  `mine` label.
- `patchcore_handoff.json.STALE_PRE_LEAKAGE_DO_NOT_USE` - pre-leakage numbers and an
  obsolete relative-mode heatmap threshold.
- A PatchCore `v0` checkpoint if you find one. The shipped weights are from `v1`.

## Files

    sonar_pipeline.py            inference module, no notebook dependencies
    sih26057_ml_handoff.json     config, metrics with CIs, negative results, limitations
    requirements.txt             pinned versions (not advisory)
    weights/best.pt              YOLOv8
    weights/model.ckpt           PatchCore (anomalib lightning checkpoint)
    golden/golden_cases.json     expected outputs for the integration test
    golden/images/               the test images
    CHECKSUMS.sha256             integrity

## Security

An earlier version of the training notebook contained a hardcoded Roboflow API key. It must
be rotated in Roboflow and loaded from an environment variable or secret store. No key is
present anywhere in this package.

Physics highlight/shadow verification is deliberately absent. It was built, measured, and did
not work on this data (unbiased AUC 0.578; shadow-right-darker only 40.2% of the time). The
evidence is in `negative_results` in the manifest.

## Reproducibility, and what a selfcheck failure means

On the machine that produced this package, `detect()` is bit-identical across processes: all
11 golden detections reproduce at a difference of exactly 0.000000. So the selfcheck is not
noisy here, and a failure on your host means something really did change.

`selfcheck` compares bucket, source branch and class name **exactly**, and anomaly scores to
a tolerance of 2e-3. The tolerance exists because scores move at the 1e-3 level across
GPU/CPU and across hosts, while the failure that matters -- a library version bump -- moves
them by 1e-1 or more. That is because anomalib normalises `pred_score` against a threshold
stored at fit time, so a version change silently rescales every score and invalidates the
image gate (0.3103) and the REVIEW admit thresholds along with it.

One caveat worth knowing before you debug: the tightest bucket-threshold margin in the golden
set is 0.0056 (yolo_only/REJECT in 000132_jpg.rf.c5f531751b93cedd4ce214261d87c3f5.jpg). If that single detection is the only thing that differs, you are
almost certainly seeing float drift on a knife-edge detection, not a broken install. Each
golden detection carries a `threshold_margin` field and the selfcheck will point this out.
If several detections differ, or scores differ by more than 0.01, treat the manifest's
thresholds as invalid until you re-derive them.

The same knife-edge caveat applies in production, not just in the test. Both admit
thresholds -- yolo_conf 0.70 and anomaly_score 0.55 -- are hard cutoffs with no hysteresis, so a
detection sitting a few thousandths either side of one is genuinely borderline and REVIEW versus
REJECT for it is not a confident judgement. The tight case in this golden set is a yolo_only
detection with yolo_conf = 0.6944, which lands 0.0056 below 0.7 and is therefore bucketed REJECT.


---
pipeline 1.0.0 | module sha256 1bfc139397610651 | assembled 2026-09-05
