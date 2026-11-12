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
# how much two subjects' crop regions sit on each other, at the match frame
# ----------------------------------------------------------------------------
#
# Each pass feathers its crop back onto the clip. When two subjects' crop regions overlap, one
# pass's feathered edge lands on pixels another pass has already refined, so the finished clip
# shows a soft seam across somebody's face. That is worth SAYING about, and nothing more: two
# people standing close together are still two subjects, and the composite is still the answer.
# Hence a threshold, and hence a warning rather than an error.

DEFAULT_OVERLAP_THRESHOLD = 0.25


def _plan_size(plan, key: str):
    """One plate size out of a plan, in either place `pairing_fingerprint` reads it from."""
    value = ((plan or {}).get("inputs") or {}).get(key)
    if value is None:
        value = (plan or {}).get(key)
    return value


def _checked_threshold(threshold) -> float:
    try:
        value = float(threshold)
    except (TypeError, ValueError):
        value = float("nan")
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"overlap_threshold must be a fraction between 0 and 1, got {threshold!r}. It is "
            f"how much of the two crops' combined area they share at the match frame.")
    return value


def subject_crop_rect(plan, subject, crop_factor: float = DEFAULT_CROP_FACTOR):
    """The rectangle one subject's crop covers, in TARGET-CLIP pixels at the match frame.

    It is `tracker_rect` - the same arithmetic the plan cut the subject reference with - so the
    overlap measure is the region that will actually be feathered back, not the face box inside
    it. A subject the clip never saw is measured from the still's own face and scaled into clip
    coordinates, which is how it was framed in the first place. ``None`` when the plan recorded
    no face for that subject on either plate.

    Returns ``[x, y, w, h]`` floats in clip pixels.
    """
    box = (subject or {}).get("clip_box")
    if box is not None:
        return tracker_rect(box, _size(_plan_size(plan, "target_size"), "target clip"),
                            crop_factor)

    still_box = (subject or {}).get("still_box")
    if still_box is None:
        return None
    still_size = _plan_size(plan, "still_size") or _plan_size(plan, "target_size")
    rect = tracker_rect(still_box, _size(still_size, "reference still"), crop_factor)
    scale = (plan or {}).get("scale")
    scale = 1.0 if scale is None else float(scale)
    return [float(v) * scale for v in rect]


def overlapping_subject_pairs(plan, threshold: float = DEFAULT_OVERLAP_THRESHOLD,
                              crop_factor: float = DEFAULT_CROP_FACTOR) -> list[dict]:
    """Every pair of subjects whose crop regions overlap by more than ``threshold``.

    The measure is intersection over union of the two crop rectangles at the match frame, so 0
    means the crops merely touch and 1 means they are the same region. Exactly on the threshold
    is NOT a warning: a plan planned at a round number would otherwise warn about itself.

    Pairs come back in plan order, the lower-numbered subject first, which is also the left to
    right order the plan was settled in. Nothing here raises for a plan with two subjects on top
    of each other - this reports, it does not judge.

    Returns ``[{"a", "b", "overlap", "rect_a", "rect_b"}, ...]``.
    """
    limit = _checked_threshold(threshold)
    subjects = (plan or {}).get("subjects") or []

    rects = []
    for position, subject in enumerate(subjects):
        if subject is None:
            rects.append((position, None))
            continue
        index = subject.get("index")
        index = position if index is None else _subject_index(index)
        rects.append((index, subject_crop_rect(plan, subject, crop_factor)))

    pairs = []
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            index_a, rect_a = rects[i]
            index_b, rect_b = rects[j]
            if rect_a is None or rect_b is None:
                continue
            box_a = np.array([[rect_a[0], rect_a[1], rect_a[0] + rect_a[2], rect_a[1] + rect_a[3]]])
            box_b = np.array([[rect_b[0], rect_b[1], rect_b[0] + rect_b[2], rect_b[1] + rect_b[3]]])
            shared = float(_overlap_area(box_a, box_b)[0][0])
            union = (rect_a[2] * rect_a[3]) + (rect_b[2] * rect_b[3]) - shared
            if union <= 0.0:
                continue
            overlap = shared / union
            if overlap > limit:
                pairs.append({"a": index_a, "b": index_b, "overlap": float(overlap),
                              "rect_a": [float(v) for v in rect_a],
                              "rect_b": [float(v) for v in rect_b]})
    return pairs




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


# ----------------------------------------------------------------------------
# the pass manifest: which subjects already had their pass
# ----------------------------------------------------------------------------
#
# READ side only, on purpose: H3 Subject Select has to answer "which subject is next"
# before anything in the job has been saved, and it must not have to write anything to
# do it. # ticket #5 writes this - H3 Pass Save records a pass complete here once its
# crops and its stitched result are on disk. The shape below is the contract that
# ticket writes, stated here so the two ends agree without either importing the other:
#
#   {"version": 1, "passes": {"<subject index>": {"complete": true, ...}}}
#
# Keys are strings because JSON object keys are. A subject with no entry, or an entry
# whose `complete` is not true, has not finished.

MANIFEST_VERSION = 1
MANIFEST_FILENAME = "manifest.json"


def manifest_path(job_dir) -> str:
    """Where a job's pass manifest lives."""
    return os.path.join(str(job_dir), MANIFEST_FILENAME)


def read_manifest(job_dir) -> dict:
    """The job's pass manifest, or an empty one when there is no manifest yet.

    A job that has never completed a pass has no manifest: that is the normal state of a
    fresh job, not a missing file, and asking "who is next" on a job that has not started
    has to work.
    """
    path = manifest_path(job_dir)
    if not os.path.isfile(path):
        return {"version": MANIFEST_VERSION, "passes": {}}
    with open(path, encoding="utf-8") as fh:
        try:
            manifest = json.load(fh)
        except ValueError:
            raise ValueError(
                f"{path} is not readable JSON, so no pass in this job can be trusted as "
                f"complete. Delete it to start the bookkeeping over, or fix it by hand."
            ) from None
    if not isinstance(manifest, dict):
        raise ValueError(f"{path} does not hold a pass manifest")
    return manifest


def _pass_is_complete(record) -> bool:
    """One manifest entry, read as "did this subject's pass finish?".

    Forgiving about the wrapper (a bare ``true`` counts) because the writer is a later
    ticket; unforgiving about everything else, so a half-written record never reads as
    "already done" and silently skips a subject.
    """
    if isinstance(record, dict):
        return bool(record.get("complete"))
    return bool(record)


def completed_subjects(plan, job_dir) -> list[int]:
    """Subject indices this plan holds whose pass the manifest records complete.

    Only subjects the plan actually has are counted, and they come back in plan order: a
    manifest left in the folder from a different plan cannot answer for this job.
    """
    passes = read_manifest(job_dir).get("passes") or {}
    if not isinstance(passes, dict):
        raise ValueError(
            f"{manifest_path(job_dir)} has no 'passes' object, so its record of completed "
            f"passes cannot be read"
        )
    done = {str(key) for key, record in passes.items() if _pass_is_complete(record)}
    out = []
    for subject in plan.get("subjects") or []:
        index = int(subject["index"])
        if str(index) in done:
            out.append(index)
    return out


def next_subject(plan, job_dir):
    """The first subject in plan order with no recorded completed pass.

    ``None`` when every subject has one: the whole job is done, and inventing a subject to
    run again would cost a generation for nothing.
    """
    done = set(completed_subjects(plan, job_dir))
    for subject in plan.get("subjects") or []:
        if int(subject["index"]) not in done:
            return subject
    return None
# ----------------------------------------------------------------------------
# the pass manifest: the WRITE side, and the pass layout inside the job folder
# ----------------------------------------------------------------------------
#
#   <job folder>/
#     manifest.json
#     passes/subject_0000/crops/frame_0000.png     <- what H3 was fed, one PNG per frame
#     passes/subject_0000/crops.json               <- the crop positions (H3FACEXFORM)
#     passes/subject_0000/stitched/frame_0000.png  <- the pass composited back on the clip
#
# Names are deterministic and zero-padded, never timestamped: re-running a pass overwrites
# its own frames instead of leaving a second copy nobody can line up with frame 7, and every
# image is an ordinary PNG a user can open to see what H3 was actually given.

PASS_STAGES = ("crops", "stitched")
PASSES_DIRNAME = "passes"
POSITIONS_FILENAME = "crops.json"
FRAME_EXTENSION = ".png"


def _subject_index(subject) -> int:
    try:
        index = int(subject)
    except (TypeError, ValueError):
        raise ValueError(f"subject must be a subject number, got {subject!r}") from None
    if index < 0:
        raise ValueError(f"subject cannot be negative, got {index}")
    return index


def checked_stage(stage) -> str:
    name = str(stage)
    if name not in PASS_STAGES:
        raise ValueError(
            f"stage must be one of {' | '.join(PASS_STAGES)}, got {stage!r}. crops is what "
            f"H3 was fed; stitched is that pass composited back onto the clip.")
    return name


def pass_dir(job_dir, subject) -> str:
    """One subject's folder inside the job."""
    return os.path.join(str(job_dir), PASSES_DIRNAME,
                        f"subject_{_subject_index(subject):04d}")


def stage_dir(job_dir, subject, stage) -> str:
    return os.path.join(pass_dir(job_dir, subject), checked_stage(stage))


def frame_name(index) -> str:
    """The name of one frame of a pass: viewable, ordered, and stable across a resume."""
    return f"frame_{int(index):04d}{FRAME_EXTENSION}"


def stage_files(job_dir, subject, stage, count) -> list[str]:
    """The paths of ``count`` frames of one stage, in frame order."""
    folder = stage_dir(job_dir, subject, stage)
    return [os.path.join(folder, frame_name(i)) for i in range(int(count))]


def positions_path(job_dir, subject) -> str:
    """Where a subject's crop positions are kept, readable, next to their frames."""
    return os.path.join(pass_dir(job_dir, subject), POSITIONS_FILENAME)


def empty_manifest() -> dict:
    return {"version": MANIFEST_VERSION, "passes": {}}


def _manifest_passes(manifest: dict, job_dir) -> dict:
    passes = manifest.get("passes")
    if passes is None:
        passes = manifest["passes"] = {}
    if not isinstance(passes, dict):
        raise ValueError(
            f"{manifest_path(job_dir)} has no 'passes' object, so its record of completed "
            f"passes cannot be read")
    return passes


def write_manifest(job_dir, manifest: dict) -> str:
    """Write the manifest whole, through a temporary name, exactly like the plan is written.

    A job that dies mid-write must not leave half a manifest behind that the next pass reads
    as "this subject is done" and skips it.
    """
    if not isinstance(manifest, dict):
        raise ValueError(f"a pass manifest is a dict, got {type(manifest).__name__}")
    folder = str(job_dir)
    os.makedirs(folder, exist_ok=True)
    final = manifest_path(folder)
    staging = final + ".writing"
    with open(staging, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=_json_default)
        fh.write("\n")
    os.replace(staging, final)
    return final


def record_pass_stage(job_dir, subject, stage, record=None) -> dict:
    """Record one subject's stage as written to disk, and return the manifest it wrote.

    ``complete`` is DERIVED here, never handed in: a pass is complete once its stitched stage
    is recorded and not before. The whole resume cursor rests on that rule, so it is computed
    in one place rather than left to each caller to get right. ``record`` is the caller's note
    of what landed on disk - frame count, file names - and is stored as given.
    """
    index = _subject_index(subject)
    name = checked_stage(stage)
    manifest = read_manifest(job_dir)
    passes = _manifest_passes(manifest, job_dir)
    key = str(index)

    entry = passes.get(key)
    if not isinstance(entry, dict):
        # The read side forgives a bare `true` as complete. Such a record can only honestly
        # mean the stitched stage was written without detail, so keep the completion rather
        # than quietly unwinding a subject's finished pass.
        entry = {"stages": ({"stitched": {}} if _pass_is_complete(entry) else {})}
    stages = entry.get("stages")
    if not isinstance(stages, dict):
        stages = {}
    if record is None:
        stages[name] = {}
    elif isinstance(record, dict):
        stages[name] = dict(record)
    else:
        stages[name] = {"record": record}
    entry["stages"] = stages
    entry["complete"] = "stitched" in stages
    passes[key] = entry
    manifest.setdefault("version", MANIFEST_VERSION)
    write_manifest(job_dir, manifest)
    return manifest


def _manifest_entry(job_dir, subject):
    passes = read_manifest(job_dir).get("passes")
    if not isinstance(passes, dict):
        return None
    return passes.get(str(_subject_index(subject)))


def manifest_stage_record(job_dir, subject, stage):
    """What the manifest records for one subject's stage, or ``None`` when it records none.

    The question Pass Load asks before it looks for a single file on disk. A record written
    by an older manifest that only says "complete" still counts for the stitched stage.
    """
    entry = _manifest_entry(job_dir, subject)
    name = checked_stage(stage)
    if isinstance(entry, dict):
        stages = entry.get("stages")
        if isinstance(stages, dict) and name in stages:
            return stages[name]
        return True if (name == "stitched" and entry.get("complete")) else None
    return {} if (_pass_is_complete(entry) and name == "stitched") else None


def pass_is_complete(job_dir, subject) -> bool:
    """Does the manifest record this subject's pass as complete (stitched stage and all)?"""
    return _pass_is_complete(_manifest_entry(job_dir, subject))


def plan_subject_indices(plan) -> list[int]:
    """The subject numbers this plan actually has."""
    return [int(subject["index"]) for subject in (plan or {}).get("subjects") or []]


def check_manifest_subjects(plan, job_dir) -> list[int]:
    """Raise when the manifest names a subject this plan does not have.

    `completed_subjects` deliberately ignores manifest keys the plan does not know - "who is
    next" must not be held hostage by a stray record. A pass about to WRITE into that
    subject's folder is a different matter: a stale entry is how two jobs in one output tree
    start reading each other's passes. Returns the recorded subject indices.
    """
    passes = read_manifest(job_dir).get("passes") or {}
    if not isinstance(passes, dict):
        raise ValueError(
            f"{manifest_path(job_dir)} has no 'passes' object, so its record of completed "
            f"passes cannot be read")
    known = set(plan_subject_indices(plan))
    stray = []
    for key in passes:
        try:
            index = int(key)
        except (TypeError, ValueError):
            stray.append(repr(key))
            continue
        if index not in known:
            stray.append(index)
    if stray:
        raise ValueError(
            f"{manifest_path(job_dir)} records a pass for subject(s) "
            f"{', '.join(str(i) for i in stray)}, which this plan does not have - its "
            f"subjects are {', '.join(str(i) for i in sorted(known)) or 'none'}. That "
            f"manifest came from another plan, or this job was re-planned. Set job_name to a "
            f"new job, or delete the manifest to start the bookkeeping over.")
    return sorted(int(key) for key in passes)


# ----------------------------------------------------------------------------
# one pass's crop positions: the H3FACEXFORM record as readable JSON
# ----------------------------------------------------------------------------
#
# Not a pickle and not a tensor: `crops.json` is readable in an editor, and it is what lets a
# resumed pass stitch its frames back without running the tracker again. `canvas` and
# `src_size` are tuples in the live payload and `segments` is a list of them; JSON has only
# arrays, so they are restored on the way back in and the payload a resumed pass gets is the
# one the tracker built.

_TUPLE_FIELDS = ("canvas", "src_size")
# Per-frame lists whose entries were tuples when the tracker built them. Restored by arity:
# `segments` is (start, end) in whole frames, `face_rect` is a float rect in canvas pixels.
_TUPLE_ENTRY_FIELDS = {"segments": (2, int), "face_rect": (4, float)}


def write_positions(job_dir, subject, transform) -> str:
    """Write one subject's crop positions (the ``H3FACEXFORM`` record) as readable JSON."""
    if not isinstance(transform, dict):
        raise ValueError(f"the crop positions of a pass are a dict, got "
                         f"{type(transform).__name__}")
    path = positions_path(job_dir, subject)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    staging = path + ".writing"
    with open(staging, "w", encoding="utf-8") as fh:
        json.dump(transform, fh, indent=2, default=_json_default)
        fh.write("\n")
    os.replace(staging, path)
    return path


def read_positions(job_dir, subject) -> dict:
    """The crop positions back off disk, tuples restored.

    A missing one is a real gap, not an empty answer: without the positions a stitched pass
    cannot be put back where it came from, so it says so rather than returning ``{}``.
    """
    path = positions_path(job_dir, subject)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"no crop positions at {path}. Without them the crops cannot be stitched back "
            f"where they came from. Run H3 Pass Save with stage=crops for this subject, with "
            f"transform wired from H3 Face Track + Crop.")
    with open(path, encoding="utf-8") as fh:
        try:
            payload = json.load(fh)
        except ValueError:
            raise ValueError(
                f"{path} is not readable JSON. Delete it and re-run H3 Pass Save with "
                f"stage=crops for this subject.") from None
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not hold a crop-position record")
    for field in _TUPLE_FIELDS:
        value = payload.get(field)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            payload[field] = (int(value[0]), int(value[1]))
    for field, (arity, cast) in _TUPLE_ENTRY_FIELDS.items():
        value = payload.get(field)
        if isinstance(value, list):
            payload[field] = [tuple(cast(v) for v in part) for part in value
                              if isinstance(part, (list, tuple)) and len(part) == arity]
    return payload






# ----------------------------------------------------------------------------
# the pairing fingerprint: refuse before two pairings mix in one job folder
# ----------------------------------------------------------------------------
#
# A job folder is a promise: the plan, the subject references and every pass under `passes/`
# were decided from the SAME plates, the SAME match frame and the SAME exclusions. Change one
# widget and re-run with the same job_name and the new pairing writes its passes beside the
# old ones, then stitches two different ideas of who is who into one clip. So the inputs that
# decide the pairing get a fingerprint, and every pass checks it.
#
# `max_subjects` is deliberately NOT part of it: it is recorded and not applied (#10), so
# changing it changes nothing on disk. What it would change is the subject set below, which is
# compared whatever the widgets say.

PAIRING_INPUT_FIELDS = ("target_size", "still_size", "match_frame", "subject_source",
                        "match_tolerance", "excluded_faces")


def _normalise(value):
    """One input, made to read the same from a plan in memory as from plan.json."""
    if isinstance(value, (list, tuple)):
        return [_normalise(v) for v in value]
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return round(float(value), 6)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if value is None:
        return None
    return str(value)


def pairing_fingerprint(plan) -> dict:
    """The part of a plan that says WHICH pairing this job was planned on.

    Both plate sizes, the match frame, the seeding plate, the tolerance, the exclusions, and
    the subject set itself - index, clip face and still face for every subject. The subject
    set is in here because it is what actually breaks: a re-detect that pairs someone else at
    the same widgets is still a different job.
    """
    inputs = (plan or {}).get("inputs") or {}
    record = {}
    for field in PAIRING_INPUT_FIELDS:
        value = inputs.get(field)
        if value is None:
            value = (plan or {}).get(field)
        record[field] = _normalise(value)
    record["subjects"] = [[_normalise((subject or {}).get("index")),
                           _normalise((subject or {}).get("clip_face")),
                           _normalise((subject or {}).get("still_face"))]
                          for subject in (plan or {}).get("subjects") or []]
    return record


def _describe(value) -> str:
    return json.dumps(value, sort_keys=True)


def _subjects_line(before: list, after: list) -> str:
    def label(row) -> str:
        index, clip_face, still_face = row
        return f"subject {index} (clip face {clip_face}, still face {still_face})"

    moved_out = [row for row in before if row not in after]
    moved_in = [row for row in after if row not in before]
    parts = []
    if moved_out:
        parts.append("in the job folder: " + "; ".join(label(row) for row in moved_out))
    if moved_in:
        parts.append("in this plan: " + "; ".join(label(row) for row in moved_in))
    if not parts:
        parts.append(f"the same {len(before)} subject(s), paired differently")
    return "subject set: " + " | ".join(parts)


def fingerprint_difference(stored_plan, current_plan) -> list[str]:
    """Every pairing input the two plans disagree about, one line each.

    Empty means the same pairing: re-planning it into the same job folder is a resume, not a
    mix, and has to be allowed or no job could ever be re-run.
    """
    before = pairing_fingerprint(stored_plan)
    after = pairing_fingerprint(current_plan)
    lines = []
    for field in PAIRING_INPUT_FIELDS:
        if before[field] != after[field]:
            lines.append(f"{field}: the job folder has {_describe(before[field])}, this plan "
                         f"has {_describe(after[field])}")
    if before["subjects"] != after["subjects"]:
        lines.append(_subjects_line(before["subjects"], after["subjects"]))
    return lines


def check_pairing_inputs(stored_plan, current_plan, job_dir, advice: str) -> None:
    """Raise unless the plan in the graph and the plan in the job folder are one pairing.

    ``advice`` is what the caller can do about it, and it differs per node: the plan node is
    the one that would overwrite, a pass is the one that would mix.
    """
    lines = fingerprint_difference(stored_plan, current_plan)
    if not lines:
        return None
    raise ValueError(
        f"{plan_path(job_dir)} was written from different pairing inputs, so this would mix "
        f"two pairings into one job folder:\n"
        + "\n".join(f"  - {line}" for line in lines)
        + f"\n{advice}")


# ----------------------------------------------------------------------------
# one subject's shot account: real absence, and the tracker losing them (#6)
# ----------------------------------------------------------------------------
#
# Absence is not invented here. The released payload already says "this person is not in this
# shot" with a pick entry `{"absent": true, "segment": [start, end]}` (nodes.py:1334): no frame
# of that shot gets a subject, the crop there is a centred box of padding rather than a face
# (nodes.py:1747), and the composite fades out across it, leaving the target clip's own pixels.
# What was missing is the read-out: WHICH shot dropped WHICH subject, over which frames of the
# clip, and whether that shot is empty because the plan said so or because the tracker lost the
# face. Only one of those is worth fixing, so they cannot share a sentence.
#
# The evidence sits in the crop-position record (`H3FACEXFORM`, built at nodes.py:1908). Every
# list in it is per KEPT frame, in crop order:
#
#   "source"        which clip frame each kept crop came from
#   "segments"      the shot boundaries, in CROP index space (nodes.py:1898) - so they are not
#                   frame numbers once anything has been dropped
#   "absent"        this frame sits in a shot the pick marked absent
#   "detected"      the tracker resolved this subject's face on this frame
#
# Frames dropped out of the batch (nodes.py:1792) never appear in `source` at all, and the
# tracker only ever drops frames it already marked absent - so a gap in `source` is absence,
# never a lost subject. Frames the tracker reached with the subject selected and produced
# nobody for are the opposite: `absent` false, `detected` false.

SHOT_PAIRED = "paired"
SHOT_ABSENT_PLANNED = "absent_planned"
SHOT_LOST = "lost"

# The two ways a report can end. Deliberately different: one tells the user nothing is wrong,
# the other names the widgets that would fix it.
ABSENT_ACTION = ("original pixels are kept there, which is the intended result for an absence - "
                 "if this person IS in this shot, plan on a match frame they share with the "
                 "reference still, or name them with Pick faces")
LOST_ACTION = ("the tracker stopped finding them there - check identity_track and "
               "identity_threshold, or lower confidence, on H3 Face Track + Crop")


def _subject_number(subject) -> int:
    """The subject number, whether it arrives as a number or as the plan's subject entry."""
    if isinstance(subject, dict):
        return _subject_index(subject.get("index"))
    return _subject_index(subject)


def _plan_subject_entry(plan, index: int) -> dict:
    for entry in (plan or {}).get("subjects") or []:
        if isinstance(entry, dict) and int(entry.get("index", -1)) == index:
            return entry
    return {"index": index}


def _checked_positions(transform) -> dict:
    """The crop-position record, or a plain refusal. ``None`` means no pass has run yet."""
    if transform is None:
        return {}
    if not isinstance(transform, dict):
        raise ValueError(
            "crop positions (H3FACEXFORM) are a dict of per-frame lists - H3 Face Track + "
            f"Crop's transform output, or the pass folder's crops.json - got "
            f"{type(transform).__name__}")
    return transform


def _clip_frame_count(plan, transform, pick) -> int:
    """How many frames the clip has, from whichever payload knows."""
    transform = _checked_positions(transform)
    for value in ((transform.get("source_frames")),
                  (((plan or {}).get("run") or {}).get("frames")),
                  ((pick or {}).get("frames"))):
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count > 0:
            return count
    source = (transform or {}).get("source") or []
    return len(source)


def _frame_evidence(transform, total: int) -> dict:
    """Clip frame -> what this pass produced there. Empty when there are no crop positions.

    Anything that is not a crop-position record is refused rather than guessed at: an account
    built from the wrong dict would tell a user their subject is fine when it is not.
    """
    transform = _checked_positions(transform)
    if not transform:
        return {}
    source = [int(f) for f in (transform.get("source") or [])]
    absent = list(transform.get("absent") or [])
    detected = list(transform.get("detected") or [])
    if not source:
        # No `source` means one crop per clip frame, in order.
        source = list(range(max(len(absent), len(detected))))
    evidence: dict = {}
    for k, frame in enumerate(source):
        if not 0 <= frame < total:
            continue
        evidence[frame] = {
            "absent": bool(absent[k]) if k < len(absent) else False,
            # No flag at all is not a lost subject. Only what the tracker DID say about a
            # frame can be evidence of a failure.
            "detected": bool(detected[k]) if k < len(detected) else True,
        }
    return evidence


def _shot_ranges(transform, pick, total: int) -> list:
    """(shot number, first frame, last frame) - inclusive, in the CLIP's frame numbers.

    The pick payload's segments are already clip frames, so when it is wired it wins and the
    shot numbers match the ones Subject Select prints. Crop positions alone carry shot bounds in
    CROP index space, so their starts are read back through `source`, and each shot runs to the
    frame before the next one starts: the shots tile the clip, so frames dropped out of the
    batch land inside whichever shot they belong to instead of vanishing from the report.
    """
    if pick:
        given = [(int(a), int(b)) for a, b in (pick.get("segments") or [])]
        ranges = [(k + 1, max(0, a), min(total, b) - 1)
                  for k, (a, b) in enumerate(given) if b > a]
        if ranges:
            return ranges

    source = [int(f) for f in ((transform or {}).get("source") or [])]
    if not source:
        source = list(range(len(((transform or {}).get("absent") or []))))
    starts = []
    for a, b in [(int(x), int(y)) for x, y in ((transform or {}).get("segments") or [])]:
        inside = [source[i] for i in range(max(0, a), min(len(source), b))]
        if inside:
            starts.append(min(inside))
    starts = sorted(set(starts))
    if not starts:
        return [(1, 0, max(0, total - 1))]
    ranges = []
    for k, start in enumerate(starts):
        end = starts[k + 1] - 1 if k + 1 < len(starts) else total - 1
        ranges.append((k + 1, max(0, start), max(start, end)))
    if ranges[0][1] > 0:
        # Leading frames dropped out of the batch belong to the first shot, not to a phantom
        # shot of their own.
        ranges[0] = (ranges[0][0], 0, ranges[0][2])
    return ranges


def _lost_reason(pick, lost_frames) -> str:
    """Why nobody was produced for these frames, as far as the payloads can tell.

    A lost subject has two causes and the user fixes them differently, so the pick payload's
    boxes are read: a shot that never held as many faces as this subject's NUMBER had nobody to
    detect there, while a shot that held plenty of faces lost this subject's identity.
    """
    boxes = (pick or {}).get("boxes") or []
    if not boxes:
        return ("the tracker reached this shot with this subject selected and produced no "
                "subject for those frames - their identity fell below the threshold, or the "
                "face was never detected")
    number = int((pick or {}).get("select_index", 0) or 0)
    widest = max((len(boxes[f]) for f in lost_frames if 0 <= f < len(boxes)), default=0)
    if widest <= number:
        return (f"no face was detected in this shot at this subject's position - the detector "
                f"reported at most {widest} face(s) there, so there was nobody to track")
    return ("faces were detected in this shot but none of them matched this subject - their "
            "identity fell below the threshold")


def _absent_reason(entry, pick, shot_number: int, dropped: int) -> str:
    """Why this shot holds no subject for them, and who decided that."""
    picks = (pick or {}).get("picks") or []
    index = shot_number - 1
    planned = bool(picks[index].get("absent")) if 0 <= index < len(picks) else False
    if planned:
        return ("the plan put an absent entry on this shot - no frame of it shows this subject, "
                "so there is nobody here for the pass to refine")
    if dropped and not picks:
        return ("every frame of this shot was marked absent and dropped out of the batch, so "
                "the tracker was never asked for a face there")
    return ("the tracker's absent-shot check found none of this subject's identity in this shot "
            "and marked the whole shot absent")


def shot_account(plan=None, subject=None, transform=None, pick=None) -> list:
    """This subject's state in every shot of one pass, in shot order.

    Three answers, and the difference between the last two is the point:

    ``paired``          the pass has a subject for that shot;
    ``absent_planned``  the plan (or the clip's own picks, or the tracker's absent-shot check)
                        put an absent entry there. Expected: those frames keep the clip's own
                        pixels, and that is the released behaviour, not a failure;
    ``lost``            the tracker reached the shot with this subject selected and produced
                        nobody for them. A failure worth telling the user about.

    Frame ranges are the clip's frame numbers, inclusive, so they line up with what a scrubbing
    timeline shows. A shot that was marked absent AND dropped out of the batch is still named:
    the user cannot see dropped frames, so a report that only covered the kept ones would hide
    the absence entirely.
    """
    index = _subject_number(subject)
    entry = _plan_subject_entry(plan, index)
    transform = _checked_positions(transform)
    total = _clip_frame_count(plan, transform, pick)
    evidence = _frame_evidence(transform, total)
    picks = (pick or {}).get("picks") or []
    shots = []
    for number, start, end in _shot_ranges(transform, pick, total):
        frames = list(range(start, end + 1))
        kept = [f for f in frames if f in evidence]
        absent = [f for f in kept if evidence[f]["absent"]]
        lost = [f for f in kept
                if not evidence[f]["absent"] and not evidence[f]["detected"]]
        present = [f for f in kept
                   if not evidence[f]["absent"] and evidence[f]["detected"]]
        dropped = [f for f in frames if transform and f not in evidence]
        pick_entry = picks[number - 1] if 0 <= number - 1 < len(picks) else None
        planned = bool(pick_entry.get("absent")) if pick_entry else False

        if lost:
            state = SHOT_LOST
        elif present:
            state = SHOT_PAIRED
        elif absent or dropped or planned:
            state = SHOT_ABSENT_PLANNED
        else:
            state = SHOT_PAIRED

        record = {
            "shot": number,
            "start": start, "end": end, "frames": end - start + 1,
            "state": state,
            "frames_present": len(present), "frames_absent": len(absent),
            "frames_lost": len(lost), "frames_dropped": len(dropped),
            "planned_by_pick": planned,
            # Whether the tracker itself said anything about this shot. Without it the account
            # can only speak for what the plan decided, not for what the tracker managed.
            "evidence": bool(kept),
            "reason": "", "action": "",
        }
        if state == SHOT_LOST:
            record["lost_start"], record["lost_end"] = lost[0], lost[-1]
            record["reason"] = _lost_reason(pick, lost)
            record["action"] = LOST_ACTION
        elif state == SHOT_ABSENT_PLANNED:
            record["reason"] = _absent_reason(entry, pick, number, len(dropped))
            record["action"] = ABSENT_ACTION
        elif absent or dropped:
            record["reason"] = (f"{len(absent) + len(dropped)} of {record['frames']} frame(s) "
                                f"of this shot were absent; the rest were refined")
            record["action"] = ("the absent frames inside this shot keep the clip's own pixels, "
                                "so only part of this shot was refined")
        shots.append(record)
    return shots


def absence_lines(shots, subject) -> list:
    """The report lines, worded so an editor can act on them without reading code.

    An absence and a lost track never share a sentence shape: one says the original pixels were
    kept and that this is the intended result, the other says the tracker stopped finding them
    and names the widgets to change. Both name the shot and its frame range, because the thing
    being checked is a moment in the clip.
    """
    index = _subject_number(subject)
    lines = []
    for shot in shots:
        where = f"shot {shot['shot']} (frames {shot['start']}-{shot['end']})"
        state = shot["state"]
        if state == SHOT_ABSENT_PLANNED:
            lines.append(f"subject {index} is not in {where}: {shot['reason']} - "
                         f"{ABSENT_ACTION}")
        elif state == SHOT_LOST:
            span = (f"{shot['frames_lost']} of {shot['frames']} frame(s)"
                    + (f" (frames {shot['lost_start']}-{shot['lost_end']})"
                       if (shot["lost_start"], shot["lost_end"]) != (shot["start"], shot["end"])
                       else ""))
            lines.append(f"subject {index} was lost in {where}: {span} produced nobody - "
                         f"{shot['reason']} - {LOST_ACTION}")
        elif shot["frames_absent"] or shot["frames_dropped"]:
            lines.append(f"subject {index} is partly absent inside {where}: {shot['reason']} - "
                         f"{shot['action']}")
    if not lines:
        if any(s.get("evidence") for s in shots):
            lines.append(f"subject {index} is in every shot of this pass "
                         f"- no absence, nothing lost")
        else:
            # Nothing has run the tracker for this subject yet, so a clean bill of health would
            # be a claim about a pass nobody has made.
            lines.append(f"subject {index} is in every shot of the plan - no absence planned, "
                         f"and whether the tracker keeps them per shot is read from the crop "
                         f"positions once this pass has run")
    return lines


def absence_account(plan=None, subject=None, transform=None, pick=None) -> dict:
    """One subject's per-shot account of a pass: the shots, the rollup, and the wording.

    ``transform`` is a pass's crop positions (the tracker's ``H3FACEXFORM``, or a checkpointed
    pass's ``crops.json``) and says what actually happened; ``pick`` is a narrowed
    ``H3FACEPICK`` and says what was PLANNED, plus which frames the detector saw at all. Either
    alone gives a partial account - Subject Select runs before the tracker, so it can only know
    the planned half unless this subject's crop positions are already in the job folder - and
    both together give the whole one.
    """
    index = _subject_number(subject)
    shots = shot_account(plan, subject, transform=transform, pick=pick)
    return {
        "subject": index,
        "shots": shots,
        "present": [s["shot"] for s in shots if s["state"] == SHOT_PAIRED],
        "absent_planned": [s["shot"] for s in shots if s["state"] == SHOT_ABSENT_PLANNED],
        "lost": [s["shot"] for s in shots if s["state"] == SHOT_LOST],
        "lines": absence_lines(shots, index),
    }




