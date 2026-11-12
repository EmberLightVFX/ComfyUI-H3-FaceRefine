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


# ----------------------------------------------------------------------------
# the rectangle a subject reference is cut with
# ----------------------------------------------------------------------------

def tracker_rect(face_box, frame_size, crop_factor: float = DEFAULT_CROP_FACTOR,
                 aspect: float = DEFAULT_CANVAS_ASPECT) -> list[float]:
    """The crop rectangle `H3FaceTrackCrop` computes for this face on this frame.

    Mirrored here (the arithmetic is nodes.py:1829-1840) instead of imported, so the plan
    can record the framing a pass WILL use before any pass has run - and so a subject
    reference is cut to that same framing rather than to a similar one. Side as a multiple
    of face HEIGHT, width from the canvas aspect, shrunk to fit the frame, centred on the
    face, then clamped inside the frame.

    Returns ``[x, y, w, h]`` floats in that plate's own pixels.
    """
    frame_w, frame_h = _size(frame_size, "plate")
    aspect = float(aspect)
    if aspect <= 0.0:
        raise ValueError(f"crop aspect must be positive, got {aspect}")
    box = _face_array([face_box], "face box")[0]
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    bh = (box[3] - box[1]) * float(crop_factor)
    bw = bh * aspect
    # Keep the canvas aspect while fitting inside the plate.
    if bw > frame_w:
        bw, bh = float(frame_w), float(frame_w) / aspect
    if bh > frame_h:
        bh, bw = float(frame_h), float(frame_h) * aspect
    x = min(max(cx - bw / 2.0, 0.0), max(0.0, frame_w - bw))
    y = min(max(cy - bh / 2.0, 0.0), max(0.0, frame_h - bh))
    return [float(x), float(y), float(bw), float(bh)]


def crop_rect_for_reference(subject_rect, still_size, target_size) -> list[int]:
    """The rectangle to cut out of the reference still for one subject's reference.

    `subject_rect` is the rect the tracker computes for that subject, in target-clip
    pixels (see `tracker_rect`). It is scaled into still pixels and kept inside the still.

    It is never upscaled. A crop is cut from the still's own pixels, so when the still is
    smaller than the clip the cut shrinks with it, and a rect wider than the still is
    clamped to the still: anything else would hand H3 detail the still never had, dressed
    up as sharpness.

    Returns whole still pixels as ``[x, y, w, h]``.
    """
    tw, th, sw, sh, scale = _plates(target_size, still_size)
    rect = [float(v) for v in subject_rect]
    if len(rect) != 4:
        raise ValueError(f"a crop rect needs 4 numbers [x, y, w, h], got {subject_rect!r}")

    # clip -> still is the inverse of the pairing scale.
    back = 1.0 / scale
    x, y = rect[0] * back, rect[1] * back
    w, h = rect[2] * back, rect[3] * back
    # Never more than the still holds: the cut is at native resolution or nothing.
    w = max(1.0, min(w, float(sw)))
    h = max(1.0, min(h, float(sh)))
    x = min(max(x, 0.0), max(0.0, float(sw) - w))
    y = min(max(y, 0.0), max(0.0, float(sh) - h))

    wi, hi = max(1, min(int(round(w)), sw)), max(1, min(int(round(h)), sh))
    xi = max(0, min(int(round(x)), sw - wi))
    yi = max(0, min(int(round(y)), sh - hi))
    return [xi, yi, wi, hi]




# ----------------------------------------------------------------------------
# the excluded_faces widget
# ----------------------------------------------------------------------------

def parse_excluded_faces(value) -> list[int]:
    """Read the excluded_faces widget: ``"2, 5"`` -> ``[2, 5]``, empty -> ``[]``.

    A typed widget rather than a front end: the numbers are the ones printed on the pairing
    preview, so switching a face off is a text edit, and text edits travel with the saved
    workflow.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        chunks: list = list(value)
    else:
        chunks = str(value).replace(";", ",").replace("/", ",").split(",")
    out: list[int] = []
    for chunk in chunks:
        if isinstance(chunk, (int, float)) and not isinstance(chunk, bool):
            out.append(int(chunk))
            continue
        text = str(chunk).strip()
        if not text:
            continue
        try:
            out.append(int(text))
        except ValueError:
            raise ValueError(
                f"excluded_faces has to be comma-separated face numbers, like '0, 3' - "
                f"{text!r} is not one. The numbers are the ones printed on the pairing "
                f"preview."
            ) from None
    return out


def _checked_excluded(parsed, n_target: int, match_frame: int) -> list[int]:
    """Exclusions sorted, deduplicated, and checked against the faces actually found."""
    unique = sorted({int(i) for i in parsed})
    for i in unique:
        if i < 0 or i >= n_target:
            raise ValueError(
                f"excluded_faces names face {i} on the target clip, but only {n_target} "
                f"face(s) were found there on frame {match_frame}. The numbers come from "
                f"the pairing preview, so re-read them after changing detector, "
                f"confidence or match_frame."
            )
    return unique


def _confs(values, n: int) -> list[float]:
    """Detector score per face. A detector that reports none reads as 1.0, as in nodes.py."""
    out = [float(v) for v in (values or [])][:n]
    if len(out) < n:
        out += [1.0] * (n - len(out))
    return out



# ----------------------------------------------------------------------------
# the subject plan
# ----------------------------------------------------------------------------

def build_subject_plan(target_boxes, target_confs, still_boxes, still_confs,
                       target_size, still_size, match_frame: int = 0,
                       subject_source: str = "union", tolerance: float = 0.05,
                       excluded=(), max_subjects: int = 0) -> dict:
    """Settle who the subjects of this job are, as a plain JSON-serialisable dict.

    Both plates were detected on the match frame, so pairing is the positional kind
    (`pair_faces`). `subject_source` decides which plate's faces seed the subjects:

    * ``target_clip`` - the clip's faces, the plate a subject has to be tracked through;
    * ``reference_still`` - the still's faces, when the still is the plate that saw more;
    * ``union`` (the default) - every face found on either plate.

    Every subject records whether it has a **subject reference** and, when it does, the
    rectangle it gets cut with. `excluded` names faces on the target clip to switch off;
    their partner on the other plate goes with them. `max_subjects` (0 = no cap) is
    recorded for later passes to act on, not applied here.
    """
    tw, th, sw, sh, _scale = _plates(target_size, still_size)
    if subject_source not in SUBJECT_SOURCES:
        raise ValueError(
            f"subject_source must be one of {' | '.join(SUBJECT_SOURCES)}, got "
            f"{subject_source!r}. target_clip seeds the subjects from the clip, "
            f"reference_still from the reference still, union from both."
        )
    frame = int(match_frame)
    if frame < 0:
        raise ValueError(f"match_frame cannot be negative, got {frame}")

    clip = _face_array(target_boxes, "target clip faces")
    still = _face_array(still_boxes, "reference still faces")
    clip_confs = _confs(target_confs, len(clip))
    still_confs = _confs(still_confs, len(still))
    excluded_faces = _checked_excluded(parse_excluded_faces(excluded), len(clip), frame)
    off = set(excluded_faces)

    pairing = pair_faces(clip, still, (tw, th), (sw, sh), tolerance)
    scale = pairing["scale"]
    still_in_clip = still * scale if len(still) else still
    still_of = {p["target"]: p["still"] for p in pairing["pairs"]}
    target_of = {p["still"]: p["target"] for p in pairing["pairs"]}

    if subject_source == "target_clip":
        seeds: list[tuple[str, int]] = [("clip", i) for i in range(len(clip))
                                        if i not in off]
    elif subject_source == "reference_still":
        seeds = [("still", j) for j in range(len(still))
                 if target_of.get(j) is None or target_of[j] not in off]
    else:
        # A still face that found a partner is already represented by that partner, so
        # union adds only the faces the clip never saw.
        seeds = ([("clip", i) for i in range(len(clip)) if i not in off]
                 + [("still", j) for j in pairing["unpaired_still"]])

    def left_to_right(seed: tuple[str, int]) -> tuple:
        kind, i = seed
        # Still boxes go into clip coordinates before they are compared, or a still at a
        # different resolution would sort its faces by pixel position rather than by where
        # the person stands.
        box = clip[i] if kind == "clip" else still_in_clip[i]
        return (float((box[0] + box[2]) / 2.0), float((box[1] + box[3]) / 2.0), kind, i)



    subjects: list[dict] = []
    for index, (kind, i) in enumerate(sorted(seeds, key=left_to_right)):
        clip_face = i if kind == "clip" else target_of.get(i)
        still_face = still_of.get(i) if kind == "clip" else i
        subject = {
            "index": index,
            "clip_face": clip_face,
            "clip_box": ([float(v) for v in clip[clip_face]]
                         if clip_face is not None else None),
            "clip_conf": clip_confs[clip_face] if clip_face is not None else None,
            "still_face": still_face,
            # In the still's own pixels: this is a face to cut out, not a position in the
            # clip. `ref_crop` is where it gets cut.
            "still_box": ([float(v) for v in still[still_face]]
                          if still_face is not None else None),
            "still_conf": still_confs[still_face] if still_face is not None else None,
            "paired": clip_face is not None and still_face is not None,
            "has_reference": still_face is not None,
            "ref_crop": None,
            # Filled in by the node once the crop has actually been written out, so no
            # later pass has to load the still again.
            "ref_file": None,
        }
        if subject["has_reference"]:
            if clip_face is not None:
                # The same rectangle the tracker would crop this face with, in the still's
                # pixels at the still's native resolution.
                subject["ref_crop"] = crop_rect_for_reference(
                    tracker_rect(clip[clip_face], (tw, th)), (sw, sh), (tw, th))
            else:
                # The clip never saw this person, so the framing comes from the still's own
                # face at the same crop_factor.
                subject["ref_crop"] = crop_rect_for_reference(
                    tracker_rect(still[still_face], (sw, sh)), (sw, sh), (sw, sh))
        subjects.append(subject)

    return {
        "version": PLAN_VERSION,
        "match_frame": frame,
        "inputs": {
            "target_size": [tw, th],
            "still_size": [sw, sh],
            "match_frame": frame,
            "subject_source": subject_source,
            "match_tolerance": float(tolerance),
            "excluded_faces": excluded_faces,
            "max_subjects": int(max_subjects or 0),
        },
        # What the match frame measured, kept so a later pass can rebuild one subject's
        # pick payload without re-detecting anything or loading the still again.
        "target_boxes": [[float(v) for v in b] for b in clip],
        "target_confs": clip_confs,
        "still_boxes": [[float(v) for v in b] for b in still],
        "still_confs": still_confs,
        "scale": float(scale),
        "tolerance_px": float(pairing["tolerance_px"]),
        "pairs": pairing["pairs"],
        "unpaired_target": pairing["unpaired_target"],
        "unpaired_still": pairing["unpaired_still"],
        "excluded_faces": excluded_faces,
        "subjects": subjects,
        "subject_count": len(subjects),
    }



# ----------------------------------------------------------------------------
# the job folder: where the plan and the subject references live
# ----------------------------------------------------------------------------

def _safe_job_name(name) -> str:
    """A job name is free text off a widget, so it may not walk out of the output tree."""
    text = str(name or "").strip()
    for sep in (os.sep, os.altsep or "", "/", "\\"):
        if sep:
            text = text.replace(sep, "_")
    while text.startswith("."):
        text = "_" + text[1:]
    return text.strip()


def job_dir_for(base_dir, job_name) -> str:
    """The job folder for one multi-face job, inside `base_dir`.

    `base_dir` is handed in by the node - ComfyUI's output folder - because this module has
    no idea where ComfyUI puts anything, and must not ask.
    """
    name = _safe_job_name(job_name)
    if not name:
        raise ValueError(
            "job_name is empty: H3 Multi-Face Plan derives one from the clip when the "
            "widget is left blank, so this is a caller that skipped that step"
        )
    return os.path.join(str(base_dir), name)


def plan_path(job_dir) -> str:
    """Where a job's subject plan lives."""
    return os.path.join(str(job_dir), PLAN_FILENAME)


def _json_default(value):
    """Every measurement here came through numpy; JSON wants plain Python numbers."""
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"a subject plan cannot hold a {type(value).__name__}")


def write_plan(job_dir, plan: dict) -> str:
    """Write the subject plan into the job folder, making the folder if need be.

    Written whole to a temporary name and then renamed: a job that dies mid-write must not
    leave half a plan behind that the next pass reads as a settled answer.
    """
    if not isinstance(plan, dict):
        raise ValueError(f"a subject plan is a dict, got {type(plan).__name__}")
    record = dict(plan)
    record.setdefault("version", PLAN_VERSION)

    folder = str(job_dir)
    os.makedirs(folder, exist_ok=True)
    final = plan_path(folder)
    staging = final + ".writing"
    with open(staging, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, default=_json_default)
        fh.write("\n")
    os.replace(staging, final)
    return final


def read_plan(job_dir) -> dict:
    """The job's subject plan, exactly as written - the record every later pass checks
    itself against before it reuses anything."""
    path = plan_path(job_dir)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"no subject plan at {path}. Run H3 Multi-Face Plan with this job_name first, "
            f"and check the job_name here matches the one that node wrote."
        )
    with open(path, encoding="utf-8") as fh:
        plan = json.load(fh)
    if not isinstance(plan, dict):
        raise ValueError(f"{path} does not hold a subject plan")
    return plan

