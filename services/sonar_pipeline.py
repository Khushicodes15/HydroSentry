"""SIH26057 - Smart Scan Strategy: side-scan sonar detection, self-contained inference.

Two branches -> spatial fusion -> decision bucket:
  YOLOv8 (closed set: aircraft, shipwreck)  +  PatchCore (anomaly / normality model)

HONEST PERFORMANCE NOTE (read before deploying):
  On the leakage-corrected 143-image evaluation set this pipeline does NOT outperform a
  well-tuned YOLOv8 alone: best F1 0.907 vs 0.894, and that difference is not statistically
  resolvable (95% CI [-0.021, +0.049]). Its measured contribution is the fusion operator -
  requiring corroboration scores F1 0.907 where a naive union of the same two branches
  scores 0.612. Numbers, confidence intervals and negative results: sih26057_ml_handoff.json.

KNOWN BEHAVIOURAL CONSEQUENCE - READ THIS:
  If PatchCore's image score falls below `image_gate`, the anomaly branch contributes no boxes,
  so every YOLO detection on that image becomes `yolo_only` and is bucketed REVIEW only when
  yolo_conf >= review_yolo_only_conf (0.70), otherwise REJECT. This pipeline can therefore
  reject a mid-confidence detection that a plain detector would have surfaced. If recall
  matters more than operator load, surface REJECT boxes in the UI too - they are returned.

Physics highlight/shadow verification is intentionally absent: it was implemented, measured, and
did not work on this data (unbiased AUC 0.578, shadow-right-darker only 40.2%).

Thresholds are checkpoint-bound. PatchCore's pred_score is normalised against the adaptive
threshold fitted at train time, so image_gate=0.3103 is valid ONLY for the shipped checkpoint
under the pinned anomalib version. A version bump moves it with no error raised.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from dataclasses import dataclass, asdict

import cv2
import numpy as np

PIPELINE_VERSION = "1.0.0"
CLASS_NAMES = {0: "aircraft", 1: "shipwreck"}   # there is no 'mine' class


@dataclass(frozen=True)
class Config:
    yolo_weights: str
    patchcore_ckpt: Optional[str] = None
    yolo_conf: float = 0.25            # F1 flat 0.25-0.50; 0.25 is the operating point
    image_gate: float = 0.3103         # PatchCore pred_score gate (checkpoint-bound)
    heat_threshold_abs: float = 0.35   # ABSOLUTE threshold on the raw anomaly map
    min_area_px: int = 200             # measured on the 256x256 anomaly map
    merge_gap_norm: float = 0.02
    fusion_iou: float = 0.10
    review_yolo_only_conf: float = 0.70
    review_patchcore_only_amax: float = 0.55
    anomaly_map_hw: tuple = (256, 256)
    # close_px is deliberately absent: morphological closing is a no-op once merge_gap_norm
    # is 0.02 (~5 px at 256x256). Box counts were identical for 0/3/5/7.


# ------------------------------------------------------------------ geometry
def _xy(b):
    if isinstance(b, dict):
        if "bbox" in b:
            b = b["bbox"]
        return (b["x_min"], b["y_min"], b["x_max"], b["y_max"])
    return tuple(b[:4])


def _area(a):
    return max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])


def _inter(a, b):
    return (max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            * max(0.0, min(a[3], b[3]) - max(a[1], b[1])))


def _iou(p, g):
    i = _inter(p, g)
    u = _area(p) + _area(g) - i
    return i / u if u > 0 else 0.0


def _bb(b):
    x = _xy(b)
    return {"x_min": float(x[0]), "y_min": float(x[1]),
            "x_max": float(x[2]), "y_max": float(x[3])}


# ------------------------------------------------------- anomaly -> boxes
def boxes_from_amap(amap, score, cfg):
    """Absolute thresholding on the raw anomaly map.

    Per-image min-max normalisation is NOT used: it is a structural defect that guarantees
    boxes on any image that passes the gate, because the max always maps to 1.0.
    """
    if score < cfg.image_gate:
        return []
    mask = (amap >= cfg.heat_threshold_abs).astype(np.uint8) * 255
    n, _, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, w = amap.shape[:2]
    out = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < cfg.min_area_px:
            continue
        x, y = st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP]
        bw, bh = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        out.append({"x_min": x / w, "y_min": y / h,
                    "x_max": (x + bw) / w, "y_max": (y + bh) / h})
    return out


def merge_boxes(boxes, gap):
    """Union-merge boxes whose gap-dilated extents touch.

    Fixes the one-to-one matching artifact where several blobs on one object would each be
    scored as a separate false positive. Per-box anomaly scores are intentionally NOT carried
    through here - they are recomputed from the map on the final fused bbox (see _box_stats).
    """
    B = [list(_xy(b)) for b in boxes]
    out = []
    while B:
        c, grew = B.pop(0), True
        while grew:
            grew, keep = False, []
            for o in B:
                if _inter((c[0] - gap, c[1] - gap, c[2] + gap, c[3] + gap), tuple(o)) > 0:
                    c = [min(c[0], o[0]), min(c[1], o[1]),
                         max(c[2], o[2]), max(c[3], o[3])]
                    grew = True
                else:
                    keep.append(o)
            B = keep
        out.append(c)
    return [{"x_min": b[0], "y_min": b[1], "x_max": b[2], "y_max": b[3]} for b in out]


def _box_stats(amap, bb):
    H, W = amap.shape[:2]
    x0 = int(np.clip(bb["x_min"] * W, 0, W - 1))
    x1 = int(np.clip(np.ceil(bb["x_max"] * W), x0 + 1, W))
    y0 = int(np.clip(bb["y_min"] * H, 0, H - 1))
    y1 = int(np.clip(np.ceil(bb["y_max"] * H), y0 + 1, H))
    r = amap[y0:y1, x0:x1]
    return float(r.max()), float(r.mean())


# ------------------------------------------------------------------ fusion
def fuse(pc_boxes, yl_boxes, iou_thr):
    """Greedy best-IoU one-to-one pairing, then tag each survivor by provenance.

    A pair that agrees adopts the YOLO box, because the detector localises better than the
    anomaly blob. NOTE: this means a 'patchcore only' evaluation that reuses paired boxes is
    flattered - it borrows YOLO's localisation wherever the two agree.
    """
    cand = sorted(((_iou(_xy(p), _xy(y)), i, j)
                   for i, p in enumerate(pc_boxes) for j, y in enumerate(yl_boxes)),
                  key=lambda z: -z[0])
    used_p, used_y, out = set(), set(), []
    for q, i, j in cand:
        if q < iou_thr or i in used_p or j in used_y:
            continue
        used_p.add(i)
        used_y.add(j)
        out.append({"bbox": _bb(yl_boxes[j]), "src": "both",
                    "class_name": yl_boxes[j].get("class_name"),
                    "yolo_conf": float(yl_boxes[j]["confidence"])})
    out += [{"bbox": _bb(p), "src": "patchcore_only", "class_name": None, "yolo_conf": None}
            for i, p in enumerate(pc_boxes) if i not in used_p]
    out += [{"bbox": _bb(y), "src": "yolo_only", "class_name": y.get("class_name"),
             "yolo_conf": float(y["confidence"])}
            for j, y in enumerate(yl_boxes) if j not in used_y]
    return out


def assign_bucket(d, cfg):
    if d["src"] == "both":
        return "HIGH"
    if d["src"] == "yolo_only" and (d["yolo_conf"] or 0.0) >= cfg.review_yolo_only_conf:
        return "REVIEW"
    if d["src"] == "patchcore_only" and d["anomaly_score"] >= cfg.review_patchcore_only_amax:
        return "REVIEW"
    return "REJECT"


# ------------------------------------------------------------------ detector
class SonarDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        self._yolo = None
        self._pc = None
        self._engine = None
        if not os.path.exists(cfg.yolo_weights):
            raise FileNotFoundError("missing YOLO weights: " + str(cfg.yolo_weights))

    def _load(self):
        if self._yolo is not None:
            return
        from ultralytics import YOLO
        self._yolo = YOLO(self.cfg.yolo_weights)
        if self.cfg.patchcore_ckpt and os.path.exists(self.cfg.patchcore_ckpt):
            try:
                from anomalib.models import Patchcore
                from anomalib.engine import Engine
                self._pc = Patchcore.load_from_checkpoint(self.cfg.patchcore_ckpt, weights_only=False)
                self._engine = Engine()
            except Exception as e:
                import logging
                logging.getLogger("hydrosentry").warning("PatchCore load skipped/failed: %s", e)
                self._pc = None

    @staticmethod
    def _pred_path(p):
        v = getattr(p, "image_path", None) or getattr(p, "image_paths", None)
        return str(v[0] if isinstance(v, (list, tuple)) else v)

    @staticmethod
    def _pred_score(p):
        s = getattr(p, "pred_score", None)
        if s is None:
            return float("nan")
        try:
            s = s.detach().cpu().numpy()
        except AttributeError:
            pass
        return float(np.asarray(s).reshape(-1)[0])

    def _anomaly(self, path):
        """path may be a single image or a directory. Returns {basename: (score, amap)}."""
        self._load()
        from anomalib.data import PredictDataset
        preds = self._engine.predict(model=self._pc,
                                     dataset=PredictDataset(path=path),
                                     ckpt_path=self.cfg.patchcore_ckpt)
        out = {}
        for p in preds or []:
            m = p.anomaly_map.squeeze().detach().cpu().numpy().astype(np.float32)
            if m.ndim != 2:
                raise RuntimeError("unexpected anomaly_map shape " + str(m.shape))
            out[os.path.basename(self._pred_path(p))] = (self._pred_score(p), m)
        return out

    def _yolo_boxes(self, path):
        self._load()
        r = self._yolo.predict(path, conf=self.cfg.yolo_conf, verbose=False)[0]
        bs = []
        if r.boxes is not None and len(r.boxes):
            xy = r.boxes.xyxyn.cpu().numpy()
            cl = r.boxes.cls.cpu().numpy()
            cf = r.boxes.conf.cpu().numpy()
            for k in range(len(cf)):
                bs.append({"bbox": {"x_min": float(xy[k, 0]), "y_min": float(xy[k, 1]),
                                    "x_max": float(xy[k, 2]), "y_max": float(xy[k, 3])},
                           "class_name": self._yolo.names[int(cl[k])],
                           "confidence": float(cf[k])})
        return bs

    def _assemble(self, amap, score, yl):
        pcb = merge_boxes(boxes_from_amap(amap, score, self.cfg), self.cfg.merge_gap_norm)
        dets = fuse(pcb, yl, self.cfg.fusion_iou)
        for d in dets:
            amax, amean = _box_stats(amap, d["bbox"])
            d["anomaly_score"] = amax
            d["anomaly_mean"] = amean
            d["bucket"] = assign_bucket(d, self.cfg)
        order = {"HIGH": 0, "REVIEW": 1, "REJECT": 2}
        return sorted(dets, key=lambda d: (order[d["bucket"]], -(d["yolo_conf"] or 0.0)))

    def detect(self, image_path):
        """Single image -> detections matching the manifest output contract.

        Every detection is returned, including bucket == 'REJECT'. Filter downstream.
        """
        if self._pc is not None:
            try:
                an = self._anomaly(image_path)
                key = os.path.basename(image_path)
                if key in an:
                    score, amap = an[key]
                    return self._assemble(amap, score, self._yolo_boxes(image_path))
            except Exception as e:
                import logging
                logging.getLogger("hydrosentry").warning("Anomaly branch failed, falling back to YOLO: %s", e)

        # Fallback to YOLO-only detections
        yl = self._yolo_boxes(image_path)
        dets = []
        for y in yl:
            conf = y.get("confidence", 0.0)
            bucket = "HIGH" if conf >= 0.70 else "REVIEW" if conf >= self.cfg.yolo_conf else "REJECT"
            dets.append({
                "bbox": y["bbox"],
                "src": "yolo_only",
                "class_name": y.get("class_name"),
                "yolo_conf": conf,
                "anomaly_score": 0.0,
                "anomaly_mean": 0.0,
                "bucket": bucket,
            })
        order = {"HIGH": 0, "REVIEW": 1, "REJECT": 2}
        return sorted(dets, key=lambda d: (order[d["bucket"]], -(d["yolo_conf"] or 0.0)))

    def detect_dir(self, folder, exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif")):
        """Batch form. Much faster than looping detect() - the anomaly branch runs once."""
        if self._pc is not None:
            try:
                an = self._anomaly(folder)
                out = {}
                for name, (score, amap) in an.items():
                    if not name.lower().endswith(exts):
                        continue
                    out[name] = self._assemble(amap, score,
                                               self._yolo_boxes(os.path.join(folder, name)))
                return out
            except Exception as e:
                import logging
                logging.getLogger("hydrosentry").warning("Anomaly batch pass failed: %s", e)

        out = {}
        for name in os.listdir(folder):
            if not name.lower().endswith(exts):
                continue
            out[name] = self.detect(os.path.join(folder, name))
        return out

    def image_score(self, image_path):
        if self._pc is not None:
            return self._anomaly(image_path)[os.path.basename(image_path)][0]
        return 0.0


# ------------------------------------------------------------------ helpers
def autodiscover(root=None):
    """Locate weights. Prefers <module dir>/weights/, then <project root>/weights/, then falls back to search root."""
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
    lb, lc = os.path.join(local, "best.pt"), os.path.join(local, "model.ckpt")
    if os.path.exists(lb) and os.path.exists(lc):
        return Config(yolo_weights=lb, patchcore_ckpt=lc)
    proj_weights = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
    pb, pc = os.path.join(proj_weights, "best.pt"), os.path.join(proj_weights, "model.ckpt")
    if os.path.exists(pb) and os.path.exists(pc):
        return Config(yolo_weights=pb, patchcore_ckpt=pc)
    # Check if best.pt exists alone (YOLO mode)
    if os.path.exists(pb):
        return Config(yolo_weights=pb, patchcore_ckpt=None)
    if os.path.exists(lb):
        return Config(yolo_weights=lb, patchcore_ckpt=None)
    root = root or "/kaggle/working"
    ck = sorted(glob.glob(os.path.join(root, "**/weights/lightning/model.ckpt"), recursive=True))
    yw = sorted(glob.glob(os.path.join(root, "**/weights/best.pt"), recursive=True))
    if yw:
        return Config(yolo_weights=yw[-1], patchcore_ckpt=ck[-1] if ck else None)
    raise FileNotFoundError(
        "could not locate weights. Expected best.pt in " + local
        + " or " + proj_weights)


def selfcheck(det, golden_json, atol=2e-3):
    """Reproduce the frozen golden outputs. Returns True on pass, False on failure.

    Detections are matched to the golden set by bounding box, NOT by list position. The
    order in which same-bucket, same-confidence detections come back is an artefact of
    connected-component labelling and is deliberately not part of the output contract.

    atol is 2e-3, not 0, because anomaly scores shift at the 1e-3 level across GPU/CPU and
    across hosts. It is loose enough to survive different hardware and tight enough to catch
    what actually matters: a library version bump, which moves scores by 1e-1 or more because
    anomalib normalises pred_score against a threshold stored at fit time. Bucket, source
    branch and class name are compared EXACTLY -- those must never drift.
    """
    _CO = ("x_min", "y_min", "x_max", "y_max")
    base = os.path.dirname(os.path.abspath(golden_json))
    G = json.load(open(golden_json))
    ok, worst = True, 0.0
    for case in G["cases"]:
        p = case["image_path"]
        if not os.path.isabs(p):
            p = os.path.join(base, p)
        name = os.path.basename(p)
        got, exp = det.detect(p), list(case["expected"])
        if len(got) != len(exp):
            print("FAIL %s: expected %d detections, got %d" % (name, len(exp), len(got)))
            ok = False
            continue
        pool = list(range(len(got)))
        for e in exp:
            i = min(pool, key=lambda k: max(abs(got[k]["bbox"][c] - e["bbox"][c]) for c in _CO))
            g = got[i]
            pool.remove(i)
            bd = max(abs(g["bbox"][c] - e["bbox"][c]) for c in _CO)
            if bd > 1e-4:
                print("FAIL %s: no box matches golden %s/%s (closest differs by %.6f)"
                      % (name, e["src"], e["bucket"], bd))
                ok = False
                continue
            if (g["src"], g["bucket"], g.get("class_name")) != \
               (e["src"], e["bucket"], e.get("class_name")):
                m = e.get("threshold_margin")
                hint = ""
                if m is not None and m < 0.01:
                    hint = ("\n     NOTE: this detection sits %.4f from the threshold that "
                            "decides its bucket.\n     A flip here is most likely float drift "
                            "on this host rather than a regression -- check whether the "
                            "anomaly_score lines above are clean." % m)
                print("FAIL %s: got %s/%s/%s, expected %s/%s/%s%s"
                      % (name, g["src"], g["bucket"], g.get("class_name"),
                         e["src"], e["bucket"], e.get("class_name"), hint))
                ok = False
            ad = abs((g["anomaly_score"] or 0.0) - (e["anomaly_score"] or 0.0))
            worst = max(worst, ad)
            if ad > atol:
                print("FAIL %s: anomaly_score %.6f, expected %.6f (tolerance %g)"
                      % (name, g["anomaly_score"] or 0.0, e["anomaly_score"] or 0.0, atol))
                ok = False
    print("worst anomaly_score drift %.6f (tolerance %g)" % (worst, atol))
    if ok:
        print("SELFCHECK PASS")
    else:
        print("SELFCHECK FAIL - do not trust the thresholds in the manifest on this host")
    return ok

def main():
    ap = argparse.ArgumentParser(description="SIH26057 sonar detection inference")
    ap.add_argument("target", nargs="?", help="image file or directory")
    ap.add_argument("--yolo-weights")
    ap.add_argument("--patchcore-ckpt")
    ap.add_argument("--out", help="write JSON here")
    ap.add_argument("--buckets", default="HIGH,REVIEW,REJECT",
                    help="comma-separated buckets to keep")
    ap.add_argument("--selfcheck", help="path to golden_cases.json")
    a = ap.parse_args()

    cfg = (Config(yolo_weights=a.yolo_weights, patchcore_ckpt=a.patchcore_ckpt)
           if a.yolo_weights and a.patchcore_ckpt else autodiscover())
    det = SonarDetector(cfg)

    if a.selfcheck:
        raise SystemExit(0 if selfcheck(det, a.selfcheck) else 1)
    if not a.target:
        ap.error("target is required unless --selfcheck is given")

    keep = set(a.buckets.split(","))
    t0 = time.time()
    res = (det.detect_dir(a.target) if os.path.isdir(a.target)
           else {os.path.basename(a.target): det.detect(a.target)})
    res = {k: [d for d in v if d["bucket"] in keep] for k, v in res.items()}
    payload = {"pipeline_version": PIPELINE_VERSION, "config": asdict(cfg),
               "elapsed_s": round(time.time() - t0, 2), "results": res}
    if a.out:
        with open(a.out, "w") as f:
            json.dump(payload, f, indent=2)
        print("wrote " + a.out)
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
