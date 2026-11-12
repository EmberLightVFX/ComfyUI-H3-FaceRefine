"""The pure half of multi-face refinement: pairing, the subject plan, the job folder.

Import boundary is the whole point of this module (ADR 0006): numpy plus the standard
library, never ComfyUI, never torch, never `folder_paths`. `nodes.py` stays untouched
(ADR 0002), and the geometry below stays testable with synthetic boxes on a machine with
no GPU, no weights and no ComfyUI on the path. Anything that needs the ComfyUI world -
the output folder, the detector, pixel tensors - belongs in `multi_face_nodes.py`.

Coordinate conventions, used everywhere here:

* a FACE box is ``[x1, y1, x2, y2]``, the same shape `nodes.py` detects and
  `H3FACEPICK` carries;
* a RECT is ``[x, y, w, h]``, the same shape the tracker's crop box has;
* plate sizes are ``[width, height]`` in pixels.
"""

from __future__ import annotations

import json
import os

import numpy as np

# Bumped when the plan's shape changes, so a job folder written by an older pack is
# recognised rather than half-read.
PLAN_VERSION = 1
PLAN_FILENAME = "plan.json"

SUBJECT_SOURCES = ("target_clip", "reference_still", "union")

# How far the two plates' aspect ratios may drift before pairing calls it a different
# framing. Rounding of one true aspect (1080 against 1088) sits well inside this;
# 16:9 against 4:3 does not.
ASPECT_TOLERANCE = 0.01

# The tracker's defaults, mirrored so the plan can record the rectangle the tracker
# WOULD use before any pass has run. `crop_factor` and `canvas_mode="manual"` 768x768
# are `H3FaceTrackCrop`'s shipped defaults (nodes.py:1031, nodes.py:1043).
DEFAULT_CROP_FACTOR = 2.5
DEFAULT_CANVAS_ASPECT = 1.0

# ----------------------------------------------------------------------------
# sizes
# ----------------------------------------------------------------------------

def _size(size, label: str) -> tuple[int, int]:
    """One plate's [width, height], validated. Plates are measured in whole pixels."""
    try:
        w, h = int(size[0]), int(size[1])
    except (TypeError, ValueError, IndexError):
        raise ValueError(f"{label} size must be given as [width, height] in pixels, "
                         f"got {size!r}") from None
    if w <= 0 or h <= 0:
        raise ValueError(f"{label} size must be positive pixels, got {w}x{h}")
    return w, h


def _plates(target_size, still_size) -> tuple[int, int, int, int, float]:
    """Both plates' sizes plus the factor that carries a still coordinate into the clip.

    A shared aspect ratio is a hard precondition. Pairing is positional - a still face is
    matched to the clip face in the same place - so two differently-framed plates put
    every box in the wrong place, and scaling one into the other would only hide that.
    """
    tw, th = _size(target_size, "target clip")
    sw, sh = _size(still_size, "reference still")
    clip_aspect, still_aspect = tw / float(th), sw / float(sh)
    if abs(clip_aspect - still_aspect) > ASPECT_TOLERANCE * clip_aspect:
        raise ValueError(
            f"cannot pair: the target clip {tw}x{th} (aspect {clip_aspect:.3f}) and the "
            f"reference still {sw}x{sh} (aspect {still_aspect:.3f}) are not framed the "
            f"same way. Pairing is positional, so a different aspect ratio puts every "
            f"face in the wrong place instead of matching it to its own second "
            f"detection. Crop or letterbox the still to the clip's framing, or use a "
            f"still cut from the clip itself."
        )
    return tw, th, sw, sh, tw / float(sw)


def _face_array(boxes, label: str) -> np.ndarray:
    """Face boxes as an (n, 4) float64 array of [x1, y1, x2, y2]. No faces is empty."""
    if boxes is None:
        return np.zeros((0, 4), dtype=np.float64)
    arr = np.asarray([[float(v) for v in box] for box in boxes], dtype=np.float64)
    if arr.size and arr.shape[1] != 4:
        raise ValueError(f"{label}: every face box needs 4 numbers "
                         f"[x1, y1, x2, y2], got {arr.shape[1]}")
    return arr


def _overlap_area(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise intersection area of boxes `a` against boxes `b`, in a's coordinates."""
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    left = np.maximum(a[:, None, 0], b[None, :, 0])
    top = np.maximum(a[:, None, 1], b[None, :, 1])
    right = np.minimum(a[:, None, 2], b[None, :, 2])
    bottom = np.minimum(a[:, None, 3], b[None, :, 3])
    return np.clip(right - left, 0.0, None) * np.clip(bottom - top, 0.0, None)


# ----------------------------------------------------------------------------
# pairing: positional, measured on both plates at the match frame
# ----------------------------------------------------------------------------

def pair_faces(target_boxes, still_boxes, target_size, still_size,
               tolerance: float) -> dict:
    """Pair the target clip's faces with the reference still's, by position.

    Both plates were detected on the frame whose positions line up (the match frame), so a
    face and its own second detection sit in the same place. The still's boxes are scaled
    into clip coordinates first, which is what lets a still of a different resolution pair
    just as well.

    A pair is accepted when the two centres are within `tolerance` AND the boxes overlap.
    `tolerance` is a fraction of the TARGET frame's diagonal, so the same number means the
    same thing at every resolution. Overlap is required as well as proximity because
    proximity alone will happily pair two bystanders standing near each other.

    Returns:

    * ``pairs``: ``[{"target": i, "still": j, "distance": d}, ...]`` ordered by target face
      index, the indices being into the lists that were passed in;
    * ``scale``: multiply a still coordinate by this to land in clip coordinates;
    * ``tolerance_px``: what `tolerance` came to in clip pixels;
    * ``unpaired_target`` / ``unpaired_still``: the faces that found no partner.
    """
    tw, th, sw, sh, scale = _plates(target_size, still_size)
    tol = float(tolerance)
    if tol < 0.0:
        raise ValueError(f"match_tolerance cannot be negative, got {tol}")

    clip = _face_array(target_boxes, "target clip faces")
    still = _face_array(still_boxes, "reference still faces")
    # Still -> clip. xyxy boxes measured from a common origin scale by one factor on all
    # four numbers.
    still_in_clip = still * scale

    reach = tol * float(np.hypot(float(tw), float(th)))
    out = {"scale": float(scale), "tolerance_px": float(reach), "pairs": [],
           "unpaired_target": list(range(len(clip))),
           "unpaired_still": list(range(len(still)))}
    if len(clip) == 0 or len(still) == 0:
        return out

    cx = (clip[:, 0] + clip[:, 2]) / 2.0
    cy = (clip[:, 1] + clip[:, 3]) / 2.0
    sx = (still_in_clip[:, 0] + still_in_clip[:, 2]) / 2.0
    sy = (still_in_clip[:, 1] + still_in_clip[:, 3]) / 2.0
    distance = np.hypot(cx[:, None] - sx[None, :], cy[:, None] - sy[None, :])
    possible = (distance <= reach) & (_overlap_area(clip, still_in_clip) > 0.0)

    # Nearest first, so nobody steals a closer partner: the still's detection order is its
    # own, and pairing by index would match whoever shares a row rather than whoever shares
    # a face.
    candidates = sorted((float(distance[i, j]), int(i), int(j))
                        for i in range(len(clip)) for j in range(len(still))
                        if possible[i, j])
    taken_clip: set[int] = set()
    taken_still: set[int] = set()
    for dist, i, j in candidates:
        if i in taken_clip or j in taken_still:
            continue
        taken_clip.add(i)
        taken_still.add(j)
        out["pairs"].append({"target": i, "still": j, "distance": round(dist, 3)})
    out["pairs"].sort(key=lambda p: p["target"])
    out["unpaired_target"] = [i for i in range(len(clip)) if i not in taken_clip]
    out["unpaired_still"] = [j for j in range(len(still)) if j not in taken_still]
    return out

