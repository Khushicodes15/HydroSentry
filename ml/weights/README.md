# Weights are not in git

The PatchCore checkpoint is ~315 MB, over GitHub's 100 MB per-file limit. Both weight
files are attached to the tagged GitHub Release instead. Download them into this folder:

    ml/weights/best.pt
    ml/weights/model.ckpt

`autodiscover()` looks here first, so once they are in place everything runs with no
path configuration. Verify before use:

    sha256sum best.pt model.ckpt

    6cc66a749f7f22ebb50b4ada6fd1dea4935447136d0c7258f96c6fd03949843b  best.pt
    d806340829c93526d8f59ac6bc2fa685b38dca5dc29b56a3796d0883c6db70fc  model.ckpt

If a checksum does not match, stop. A stale PatchCore checkpoint reproduces nothing:
anomalib normalises scores against a threshold stored at fit time, so the wrong file
shifts every score and silently invalidates the image gate and both admit thresholds.
