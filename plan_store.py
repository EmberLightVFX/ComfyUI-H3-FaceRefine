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
#
#   1  the plan as first written
#   2  #10: max_subjects truncates the subject set, so a plan gained `capped_out`
#   3  #11 review: a plan records the `crop_factor` its subject references were cut with
#      (the tracker takes it from a widget, so it was never the plan's to assume), and
#      `excluded_faces` can name a face on either plate - a bare number for the target clip,
#      `s2` for the reference still
#
# Version 1 and 2 stay READABLE rather than migrated or refused for being old. They already recorded
# `max_subjects` in `inputs` even though nothing acted on it at first, so the honest rule is the cheap
# one: read an old plan as it stands, say which version the folder was written with, and let the
# pairing fingerprint decide - it compares max_subjects now, so an old folder that genuinely
# disagrees about the cap is refused naming the difference, and one that agrees is the same job.
# A version 1/2 folder records no crop_factor because it hard-coded the tracker's shipped default,
# so `pairing_fingerprint` reads that default back out of it rather than calling it a difference.
# Anything outside PLAN_READABLE_VERSIONS is refused by number rather than half-read.
PLAN_VERSION = 3
PLAN_READABLE_VERSIONS = (1, 2, PLAN_VERSION)
PLAN_FILENAME = "plan.json"

# The pairing preview image, written by the plan node into the job folder (#9). It is the same
# image the node already returns as its `preview` output - both plates, every face outlined and
# numbered, a pair sharing a colour, a switched-off face struck through - kept on disk so the
# read-only panel can show a pairing without re-detecting or re-pairing anything. The plan node
# draws it; nothing else writes it.
PREVIEW_FILENAME = "pairing_preview.png"

SUBJECT_SOURCES = ("target_clip", "reference_still", "union")

# What to do with a face that found no partner on the other plate (#8). This belongs to the PLAN
# and not to any single pass because it decides the SUBJECT SET, and the subject set is what the
# whole downstream chain is gated on: the composite needs a stitched pass for every subject in
# the plan, and the pairing fingerprint compares the subject set to keep two pairings out of one
# job folder. Deciding it once here means "who is a subject" is answered in `plan.json`.
UNMATCHED_FACES = ("leave", "no_reference", "with_still")
DEFAULT_UNMATCHED_FACES = "leave"

# What kind of subject reference a subject has, so no pass has to infer it from whether
# `ref_crop` happens to be set:
#   crop   cut out of the still with `crop_rect_for_reference`, at the still's native pixels;
#   still  the WHOLE reference still - the face was never resolved in it, so there is nothing
#          to cut (#8, `unmatched_faces=with_still`);
#   none   no subject reference at all: the pass runs with nothing wired into H3's reference
#          slots, which is legal because those slots autogrow from zero.
REF_CROP, REF_STILL, REF_NONE = "crop", "still", "none"
REF_KINDS = (REF_CROP, REF_STILL, REF_NONE)

# How far the two plates' aspect ratios may drift before pairing calls it a different
# framing. Rounding of one true aspect (1080 against 1088) sits well inside this;
# 16:9 against 4:3 does not.
ASPECT_TOLERANCE = 0.01

# How an exclusion names a face on the REFERENCE STILL instead of the target clip: ``s2``. A bare
# number has always been a clip face, and `union` seeding makes a subject of a face only the photo
# ever saw - so without a second numbering (spec story 6's bystander) that subject could never be
# switched off.
STILL_FACE_PREFIX = "s"

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


def plan_crop_factor(plan) -> float:
    """The crop factor this plan's subject references were cut at.

    A job folder written before the plan recorded one (plan versions 1 and 2) cut every reference
    at the tracker's shipped default, so that is how it is read back: an old folder is not
    suddenly a different job because the field is missing.
    """
    recorded = ((plan or {}).get("inputs") or {}).get("crop_factor")
    if recorded is None:
        return DEFAULT_CROP_FACTOR
    try:
        return _checked_crop_factor(recorded)
    except ValueError:
        return DEFAULT_CROP_FACTOR


def subject_crop_rect(plan, subject, crop_factor: float = None):
    """The rectangle one subject's crop covers, in TARGET-CLIP pixels at the match frame.

    It is `tracker_rect` - the same arithmetic the plan cut the subject reference with - so the
    overlap measure is the region that will actually be feathered back, not the face box inside
    it. A subject the clip never saw is measured from the still's own face and scaled into clip
    coordinates, which is how it was framed in the first place. ``None`` when the plan recorded
    no face for that subject on either plate.

    ``crop_factor`` defaults to the factor the PLAN was built at rather than to the shipped 2.5:
    the tracker takes it from a widget, so a job run at another factor would otherwise have its
    crop regions, its overlap warnings and its preview measured at a size no pass ever used.

    Returns ``[x, y, w, h]`` floats in clip pixels.
    """
    if crop_factor is None:
        crop_factor = plan_crop_factor(plan)
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
                              crop_factor: float = None) -> list[dict]:
    """Every pair of subjects whose crop regions overlap by more than ``threshold``.

    The measure is intersection over union of the two crop rectangles at the match frame, so 0
    means the crops merely touch and 1 means they are the same region. Exactly on the threshold
    is NOT a warning: a plan planned at a round number would otherwise warn about itself.
    ``crop_factor`` defaults to the factor the plan was built at, so the warning is about the
    crops the passes will actually cut.

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

def _digits(text: str) -> bool:
    """``"12"`` / ``"-3"`` -> True, ``""`` / ``"s2"`` / ``" bystander"`` -> False."""
    body = text[1:] if text.startswith("-") else text
    return bool(body) and all(char.isdigit() for char in body)


def parse_excluded_faces(value) -> list:
    """Read the excluded_faces widget: ``"2, s5"`` -> ``[2, "s5"]``, empty -> ``[]``.

    A typed widget rather than a front end: the numbers are the ones printed on the pairing
    preview, so switching a face off is a text edit, and text edits travel with the saved
    workflow.

    A bare number names a face on the **target clip** - the numbering it has always had. An
    ``s``-prefixed one names a face on the **reference still**, which is the only way to switch
    off a bystander the photo saw and the clip never did: with the shipped ``union`` seeding such
    a face becomes a subject, and a subject with no clip face has no clip number to switch it
    off with.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        chunks: list = list(value)
    else:
        chunks = str(value).replace(";", ",").replace("/", ",").split(",")
    out: list = []
    for chunk in chunks:
        if isinstance(chunk, (int, float)) and not isinstance(chunk, bool):
            out.append(int(chunk))
            continue
        text = str(chunk).strip()
        if not text:
            continue
        if text[0] in ("s", "S"):
            tail = text[1:]
            if _digits(tail):
                out.append(STILL_FACE_PREFIX + tail)
                continue
        try:
            out.append(int(text))
        except ValueError:
            raise ValueError(
                f"excluded_faces has to be comma-separated face numbers, like '0, 3' - "
                f"{text!r} is not one. A bare number is a target clip face; a reference still "
                f"face is written with an s, like 's2'. The numbers are the ones printed on "
                f"the pairing preview."
            ) from None
    return out


def exclusion_plates(excluded) -> tuple[list[int], list[int]]:
    """An exclusion list split into ``(target clip faces, reference still faces)``.

    Downstream - the preview, the report, the chips - each plate has to be counted and named
    separately, and none of them should have to re-read the ``s`` prefix for itself.
    """
    clip: list[int] = []
    still: list[int] = []
    for item in excluded or []:
        if isinstance(item, str):
            still.append(int(item[1:]))
        else:
            clip.append(int(item))
    return clip, still


def _checked_excluded(parsed, n_target: int, n_still: int,
                      match_frame: int) -> list:
    """Exclusions sorted, deduplicated, and checked against the faces actually found.

    Each one is checked against the plate it names, so an out-of-range ``s2`` says the reference
    still only had two faces rather than blaming a clip it was never pointing at.
    """
    clip, still = exclusion_plates(parsed)
    clip, still = sorted(set(clip)), sorted(set(still))
    for i in clip:
        if i < 0 or i >= n_target:
            raise ValueError(
                f"excluded_faces names target clip face {i}, but only {n_target} face(s) "
                f"were found on the target clip on frame {match_frame}. A reference still "
                f"face is written s{i}. The numbers come from the pairing preview, so "
                f"re-read them after changing detector, confidence or match_frame."
            )
    for j in still:
        if j < 0 or j >= n_still:
            raise ValueError(
                f"excluded_faces names reference still face {j}, but only {n_still} face(s) "
                f"were found on the reference still on frame {match_frame}, so the reference "
                f"still faces run 0-{max(0, n_still - 1)}. A target clip face is written {j} "
                f"without the s. The numbers come from the pairing preview, so re-read them "
                f"after changing detector, confidence or match_frame."
            )
    return clip + [STILL_FACE_PREFIX + str(j) for j in still]


def describe_exclusions(excluded) -> str:
    """The exclusions as one readable run-on: ``clip face(s) 1, reference still face(s) s2``."""
    clip, still = exclusion_plates(excluded)
    named = []
    if clip:
        named.append("clip face(s) " + ", ".join(str(i) for i in clip))
    if still:
        named.append("reference still face(s) "
                     + ", ".join(STILL_FACE_PREFIX + str(j) for j in still))
    return ", ".join(named)


def _confs(values, n: int) -> list[float]:
    """Detector score per face. A detector that reports none reads as 1.0, as in nodes.py."""
    out = [float(v) for v in (values or [])][:n]
    if len(out) < n:
        out += [1.0] * (n - len(out))
    return out



# ----------------------------------------------------------------------------
# the subject plan
# ----------------------------------------------------------------------------

def _checked_crop_factor(value) -> float:
    """The tracker's ``crop_factor`` as a usable number.

    It is the crop side as a multiple of the detected face HEIGHT, so below 1.0 there is no crop
    the tracker could make - the face would not fit in it. The plan records it because the
    **subject reference** has to be framed at the same factor the tracker crops at, and the
    tracker takes that from a widget: it was never this module's to assume.
    """
    try:
        factor = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"crop_factor must be a number, got {value!r}") from None
    if not factor >= 1.0:
        raise ValueError(
            f"crop_factor {value!r} cannot frame a face: it is the crop side as a multiple of "
            f"face HEIGHT, so it has to be at least 1.0 (H3 Face Track + Crop ships 2.5).")
    return factor


def build_subject_plan(target_boxes, target_confs, still_boxes, still_confs,
                       target_size, still_size, match_frame: int = 0,
                       subject_source: str = "union", tolerance: float = 0.05,
                       excluded=(), max_subjects: int = 0,
                       unmatched_faces: str = DEFAULT_UNMATCHED_FACES,
                       crop_factor: float = DEFAULT_CROP_FACTOR) -> dict:
    """Settle who the subjects of this job are, as a plain JSON-serialisable dict.

    Both plates were detected on the match frame, so pairing is the positional kind
    (`pair_faces`). `subject_source` decides which plate's faces seed the subjects:

    * ``target_clip`` - the clip's faces, the plate a subject has to be tracked through;
    * ``reference_still`` - the still's faces, when the still is the plate that saw more;
    * ``union`` (the default) - every face found on either plate.

    Every subject records whether it has a **subject reference** and, when it does, the
    rectangle it gets cut with and its `ref_kind` (`crop`, `still` or `none`). `excluded` names
    faces to switch off - a bare number on the target clip, `s2` on the reference still - and the
    face paired to each one goes with it, so the person it was matched to falls to whatever
    `unmatched_faces` says.
    `max_subjects` (0 = no cap) is recorded for later passes to act on, not applied here.

    `unmatched_faces` decides what becomes of a face that found no partner on the other plate:

    * ``leave`` (the default) - it is not made a subject at all, so no pass ever touches it and
      it keeps its original pixels. Recorded in ``left_out`` so the report can name it.
    * ``no_reference`` - it becomes a subject with no subject reference, and its pass runs with
      nothing wired into H3's reference slots.
    * ``with_still`` - it becomes a subject whose subject reference is the whole reference still.

    A face that found a partner is identical in all three modes.

    `crop_factor` is **H3 Face Track + Crop's own crop factor**, which that node takes from a
    widget. Every **subject reference** is cut with the rectangle that factor gives at the match
    frame, so a plan built at a factor the tracker will not use mis-frames every reference in the
    job - the detail it carries lands somewhere else in the crop.
    """
    tw, th, sw, sh, _scale = _plates(target_size, still_size)
    if subject_source not in SUBJECT_SOURCES:
        raise ValueError(
            f"subject_source must be one of {' | '.join(SUBJECT_SOURCES)}, got "
            f"{subject_source!r}. target_clip seeds the subjects from the clip, "
            f"reference_still from the reference still, union from both."
        )
    if unmatched_faces not in UNMATCHED_FACES:
        raise ValueError(
            f"unmatched_faces must be one of {' | '.join(UNMATCHED_FACES)}, got "
            f"{unmatched_faces!r}. leave does not make a subject of a face that found no "
            f"partner, so it keeps its original pixels; no_reference makes one with no subject "
            f"reference; with_still makes one conditioned on the whole reference still."
        )
    frame = int(match_frame)
    if frame < 0:
        raise ValueError(f"match_frame cannot be negative, got {frame}")
    factor = _checked_crop_factor(crop_factor)

    clip = _face_array(target_boxes, "target clip faces")
    still = _face_array(still_boxes, "reference still faces")
    clip_confs = _confs(target_confs, len(clip))
    still_confs = _confs(still_confs, len(still))
    excluded_faces = _checked_excluded(parse_excluded_faces(excluded), len(clip),
                                       len(still), frame)
    off_clip, off_still = exclusion_plates(excluded_faces)
    off_clip, off_still = set(off_clip), set(off_still)

    pairing = pair_faces(clip, still, (tw, th), (sw, sh), tolerance)
    scale = pairing["scale"]
    still_in_clip = still * scale if len(still) else still
    still_of = {p["target"]: p["still"] for p in pairing["pairs"]}
    target_of = {p["still"]: p["target"] for p in pairing["pairs"]}

    if subject_source == "target_clip":
        seeds: list[tuple[str, int]] = [("clip", i) for i in range(len(clip))
                                        if i not in off_clip]
    elif subject_source == "reference_still":
        seeds = [("still", j) for j in range(len(still))
                 if j not in off_still
                 and (target_of.get(j) is None or target_of[j] not in off_clip)]
    else:
        # A still face that found a partner is already represented by that partner, so
        # union adds only the faces the clip never saw.
        seeds = ([("clip", i) for i in range(len(clip)) if i not in off_clip]
                 + [("still", j) for j in pairing["unpaired_still"]
                    if j not in off_still])

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
        # A face switched off on either plate cannot hold up a pair: the person it was matched to
        # loses its partner and falls to whatever `unmatched_faces` says, which is what switching
        # one detection off has to mean.
        if clip_face in off_clip:
            clip_face = None
        if still_face in off_still:
            still_face = None
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
            "ref_kind": REF_NONE,
            # Filled in by the node once the crop has actually been written out, so no
            # later pass has to load the still again.
            "ref_file": None,
        }
        if subject["has_reference"]:
            subject["ref_kind"] = REF_CROP
            if clip_face is not None:
                # The same rectangle the tracker would crop this face with, in the still's
                # pixels at the still's native resolution.
                subject["ref_crop"] = crop_rect_for_reference(
                    tracker_rect(clip[clip_face], (tw, th), factor), (sw, sh), (tw, th))
            else:
                # The clip never saw this person, so the framing comes from the still's own
                # face at the same crop_factor.
                subject["ref_crop"] = crop_rect_for_reference(
                    tracker_rect(still[still_face], (sw, sh), factor), (sw, sh), (sw, sh))
        elif unmatched_faces == "with_still":
            # Nothing to cut: this face was never resolved in the still, so there is no
            # rectangle to map into it. `crop_rect_for_reference` is for CUT references and is
            # deliberately not used here - the reference is the whole still, at the still's own
            # pixels, and the pass is conditioned on the whole frame instead of on a face.
            subject["has_reference"] = True
            subject["ref_kind"] = REF_STILL
            subject["ref_crop"] = [0, 0, int(sw), int(sh)]
        subjects.append(subject)

    # `leave` takes the unpaired faces out of the subject set altogether, and records them
    # separately: they are not subjects, so they get no pass, no reference file and no number -
    # but the report has to name every one of them, or a face quietly missing from the finished
    # clip looks like a bug in the refinement rather than the safe default working.
    left_out: list[dict] = []
    if unmatched_faces == "leave":
        kept = []
        for subject in subjects:
            if subject["has_reference"]:
                kept.append(subject)
            else:
                subject["index"] = None      # left out of the job, so it has no subject number
                left_out.append(subject)
        subjects = kept
    # Renumbered after any leaving, so the subject set stays contiguous: passes, the manifest
    # keys, the reference file names and the composite order all count on that.
    for index, subject in enumerate(subjects):
        subject["index"] = index

    # The cap (#10) spends the generations on purpose. The first N faces in the order the job
    # runs them are the job; the tail is capped, and capped LAST because `leave` (#8) already
    # took the unpaired faces out first - nobody is named twice. A capped face follows #8's rule
    # for a left-out face: `index: None`, so it can never be numbered, selected, reach `next`, or
    # be indexed into a pass folder. It keeps where it stood on the match frame so the report can
    # name each one instead of cutting the tail quietly.
    cap = int(max_subjects or 0)
    capped_out: list = []
    if cap and len(subjects) > cap:
        for subject in subjects[cap:]:
            # A clip face is named by its own box; a face only the still ever saw is named by
            # that box scaled into the clip, which is where it would have stood in the order.
            box = subject.get("clip_box")
            if box is None and subject.get("still_box") is not None:
                box = [float(v) * float(scale) for v in subject["still_box"]]
            if box:
                subject["match_x"] = round((float(box[0]) + float(box[2])) / 2.0, 1)
                subject["match_y"] = round((float(box[1]) + float(box[3])) / 2.0, 1)
            else:
                subject["match_x"] = subject["match_y"] = None
            subject["capped_by"] = cap
            subject["index"] = None
            capped_out.append(subject)
        subjects = subjects[:cap]
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
            "unmatched_faces": str(unmatched_faces),
            "max_subjects": int(max_subjects or 0),
            # The tracker's own crop factor, recorded because it is the rectangle every
            # subject reference in this job was cut with - and the passes read that
            # rectangle back from the plan rather than re-guessing it.
            "crop_factor": float(factor),
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
        # Faces `unmatched_faces=leave` took out of the subject set. Not subjects, so no pass
        # ever runs for them - which is why they keep their original pixels.
        "left_out": left_out,
        # Faces `max_subjects` caps so the generations get spent elsewhere (#10). They found a
        # partner and would have been subjects; the cap said otherwise. This is where that
        # decision stays readable afterwards, and why the report can still name them.
        "capped_out": capped_out,
    }



# ----------------------------------------------------------------------------
# who each subject is, and what the report says about the faces that found no partner (#8)
# ----------------------------------------------------------------------------

KIND_PAIRED = "paired"
KIND_NO_REFERENCE = "unpaired_no_reference"
KIND_WHOLE_STILL = "unpaired_whole_still"
KIND_LEFT_OUT = "left_out"


def subject_kind(subject) -> str:
    """Which kind of subject this entry is, in one word.

    ``paired`` has a subject reference of its own - a crop of the face the still resolved at the
    same place as the clip's; ``unpaired_no_reference`` is a subject with nothing in H3's
    reference slots; ``unpaired_whole_still`` is a subject conditioned on the whole reference
    still. The fourth kind, ``left_out``, never reaches a pass at all, so it is counted from
    ``plan["left_out"]`` rather than from a subject entry.
    """
    entry = subject or {}
    if entry.get("ref_kind") == REF_STILL:
        return KIND_WHOLE_STILL
    if entry.get("has_reference"):
        return KIND_PAIRED
    return KIND_NO_REFERENCE


def subject_kind_counts(plan) -> dict:
    """How many subjects of each kind this plan holds, plus how many faces were left out."""
    counts = {KIND_PAIRED: 0, KIND_NO_REFERENCE: 0, KIND_WHOLE_STILL: 0, KIND_LEFT_OUT: 0}
    for subject in (plan or {}).get("subjects") or []:
        counts[subject_kind(subject)] += 1
    counts[KIND_LEFT_OUT] = len((plan or {}).get("left_out") or [])
    return counts


def _face_position(subject, frame) -> str:
    """Where this face was on the match frame, in the clip's pixels - the numbers the user can
    read straight off the pairing preview."""
    entry = subject or {}
    where = f"frame {frame}" if frame is not None else "the match frame"
    box = entry.get("clip_box")
    if not box:
        return f"a face the clip never saw on {where}"
    cx = int(round((float(box[0]) + float(box[2])) / 2.0))
    cy = int(round((float(box[1]) + float(box[3])) / 2.0))
    return f"clip face {int(entry['clip_face'])} (at {cx}, {cy} on {where})"


def unmatched_lines(plan) -> list:
    """The sentences about faces that found no partner, one per face or subject.

    Built here and not in the node module for the reason #6 found: the words a user reads are
    the deliverable, so CI pins them. `leave` in particular has to say out loud that leaving is
    not a failure - otherwise the safe default reads like a pairing bug.
    """
    inputs = (plan or {}).get("inputs") or {}
    mode = str(inputs.get("unmatched_faces") or DEFAULT_UNMATCHED_FACES)
    frame = (plan or {}).get("match_frame")
    lines: list[str] = []
    if mode == "leave":
        for subject in (plan or {}).get("left_out") or []:
            lines.append(
                f"{_face_position(subject, frame)} found no partner in the reference still, so "
                f"it is not a subject at all: unmatched_faces=leave - no pass runs for it and "
                f"its original pixels are kept. That is not a failure: it is the safe default "
                f"when the pairing might be wrong.")
    elif mode == "no_reference":
        for subject in (plan or {}).get("subjects") or []:
            if subject_kind(subject) != KIND_NO_REFERENCE:
                continue
            lines.append(
                f"{_face_position(subject, frame)} is subject {subject['index']} with no "
                f"subject reference: unmatched_faces=no_reference runs its pass with nothing "
                f"wired into H3's reference slots, so the refinement is driven by identity and "
                f"quality only and will be weaker than a paired subject.")
    elif mode == "with_still":
        sw, sh = (inputs.get("still_size") or [0, 0])[:2]
        for subject in (plan or {}).get("subjects") or []:
            if subject_kind(subject) != KIND_WHOLE_STILL:
                continue
            lines.append(
                f"{_face_position(subject, frame)} is subject {subject['index']} with the whole "
                f"reference still as its subject reference ({int(sw)}x{int(sh)}px): "
                f"unmatched_faces=with_still, so the conditioning is whole-frame and H3 chooses "
                f"which face in the still it applies to - not a guarantee the right face is "
                f"used.")
    return lines


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


def _capped_label(subject, frame) -> str:
    """How a capped face is named in the report.

    The numbers are the ones a user reads off the pairing preview, so a clip face is named in the
    clip's own pixels and a face only the still ever saw is named where it would have stood.
    """
    entry = subject or {}
    where = f"frame {frame}" if frame is not None else "the match frame"
    box = entry.get("clip_box")
    if box:
        cx = int(round((float(box[0]) + float(box[2])) / 2.0))
        cy = int(round((float(box[1]) + float(box[3])) / 2.0))
        return f"clip face {int(entry['clip_face'])} (at {cx}, {cy} on {where})"
    x, y = entry.get("match_x"), entry.get("match_y")
    still_face = entry.get("still_face")
    name = f"still face {int(still_face)}" if still_face is not None else "a face the clip never saw"
    if x is None or y is None:
        return f"{name} (on {where})"
    return f"{name} (at {int(round(float(x)))}, {int(round(float(y)))} on {where})"


def capped_lines(plan) -> list:
    """The truncation sentence, naming every capped face.

    A tail cut in silence reads like the pairing went wrong, so this never returns a bare count:
    each capped face is named with the position a user can find it at, and the sentence says what
    to do about it. Empty when nothing was capped.
    """
    inputs = (plan or {}).get("inputs") or {}
    cap = int(inputs.get("max_subjects") or 0)
    capped = (plan or {}).get("capped_out") or []
    if not cap or not capped:
        return []
    frame = (plan or {}).get("match_frame")
    named = ", ".join(_capped_label(subject, frame) for subject in capped)
    return [f"subject cap {cap}: {len(capped)} face(s) capped - {named}. "
            f"Raise max_subjects to include them, or switch a face off with excluded_faces to "
            f"spend those generations elsewhere on purpose."]


def uncapped_cost_line(plan, frames) -> str:
    """What the job would have cost without the cap, next to what it costs with it.

    The cap's whole point is generations saved, so the number it saved has to sit beside the
    number it spends. Empty when the cap changed nothing.
    """
    capped = len((plan or {}).get("capped_out") or [])
    if not capped:
        return ""
    running = int((plan or {}).get("subject_count") or 0)
    total = running + capped
    return (f"{running} subject(s) of {total} to generate; uncapped cost would be "
            f"{total} subject(s) x {int(frames)} frame(s)")


def plan_version_note(plan) -> str:
    """What to tell the user about the plan format version a job folder was written with.

    An old folder is understood, not migrated - so this is an explanation, not a warning. The
    pairing fingerprint is what refuses a genuinely different job; the version number only says
    which pack wrote the folder.
    """
    version = int((plan or {}).get("version") or 0)
    if version in PLAN_READABLE_VERSIONS and version != PLAN_VERSION:
        return (f"this job folder was written with plan format version {version}; this pack "
                f"writes version {PLAN_VERSION}. The plan is read as it stands - the pairing "
                f"fingerprint, not the version number, is what refuses a different job.")
    return ""


def chain_lines(check) -> list:
    """The composite's chain sentence, in the voice of what was actually recorded.

    Never a flat statement that the passes WERE chained: it names which passes declared their
    base, and where nothing was declared it says the claim is unproven rather than hiding it.
    """
    check = check or {}
    declared = check.get("declared") or {}
    missing = check.get("missing") or []
    lines = ["chain as declared by H3 Pass Save: "
             + "; ".join(f"subject {index} onto {chain_label(value)}"
                         for index, value in sorted(declared.items()))]
    if missing:
        named = ", ".join(f"subject {index}" for index in missing)
        lines.append(
            f"{named} recorded no chained_from, so nothing in this job folder says what those "
            f"passes were composited onto: the last stitched pass being the finished clip is "
            f"unproven for them. Re-save those stages with chained_from set, or check the "
            f"stitched folders yourself.")
    else:
        lines.append("every pass named the plate it was composited onto, so the last subject's "
                     "stitched frames carry the earlier passes into the finished clip because "
                     "each pass took the previous one's result as its base - nothing is "
                     "re-blended here.")
    return lines


def chain_error(check) -> str:
    """The refusal sentence for a declared base that contradicts the plan order."""
    parts = []
    for index, declared_value, expected in (check or {}).get("wrong") or []:
        parts.append(
            f"subject {index} declared chained_from {declared_value} (it composited onto "
            f"{chain_label(declared_value)}), but this plan composites it onto "
            f"{chain_label(expected)}. Wiring a pass onto the target clip instead of the previous "
            f"pass's result drops every earlier subject from the finished clip: their refined "
            f"faces revert to the pixels the clip arrived with, and the report would still name "
            f"them. Set chained_from on H3 Pass Save to the subject whose stitched result that "
            f"pass was composited onto - {expected} for this one - and re-save its stitched "
            f"stage.")
    return " ".join(parts)


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
    version = int(plan.get("version") or 0)
    if version not in PLAN_READABLE_VERSIONS:
        readable = ", ".join(str(v) for v in PLAN_READABLE_VERSIONS)
        raise ValueError(
            f"{path} was written with plan format version {version}, which this pack cannot "
            f"read - it reads version {readable}. Re-run H3 Multi-Face Plan to write a plan this "
            f"pack understands, into a job_name of its own.")
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


# What H3 Pass Save's `chained_from` widget means: -1 is the target clip, any other number is the
# subject whose stitched result that pass was composited onto.
CHAINED_TO_CLIP = -1


def chain_check(plan, job_dir) -> dict:
    """What each pass DECLARED it composited onto, against the plan's own composite order.

    **H3 Multi-Face Composite** hands back the last subject's stitched frames on the claim that
    every earlier pass is already carried inside them, because each pass took the previous pass's
    result as its base. Nothing in the graph proves that - the wiring is the user's - so H3 Pass
    Save records what each pass said it composited onto, and the composite checks that against the
    plan order it prints. A pass that says the target clip when the plan says "after subject 0"
    silently reverts subject 0 in the finished clip while the report claims a chain, so that is a
    refusal and not a note.

    Returns ``{"declared": {subject: value}, "missing": [subject, ...],
    "wrong": [(subject, declared, expected), ...]}``.

    A missing declaration only counts against a subject that is not first in the order: the first
    pass sits on the target clip by definition - which is also what the widget's default says - so
    a job folder written before the field existed is silent exactly where silence is harmless.
    """
    order = plan_subject_indices(plan)
    declared, missing, wrong = {}, [], []
    for position, index in enumerate(order):
        record = manifest_stage_record(job_dir, index, "stitched")
        value = (record or {}).get("chained_from") if isinstance(record, dict) else None
        expected = CHAINED_TO_CLIP if position == 0 else order[position - 1]
        if value is None:
            if position:
                missing.append(index)
            continue
        declared[index] = int(value)
        if int(value) != int(expected):
            wrong.append((index, int(value), int(expected)))
    return {"declared": declared, "missing": missing, "wrong": wrong}


def checked_chained_from(plan, subject, value) -> int:
    """A ``chained_from`` declaration that means something for THIS plan.

    -1 (the target clip) or one of the plan's own subjects, and never the subject itself: a pass
    composited onto its own earlier output is not a chain, it is a loop nobody meant.
    """
    try:
        index = int(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"chained_from must be -1 (the target clip) or a subject number, got {value!r}"
        ) from None
    if index == CHAINED_TO_CLIP:
        return CHAINED_TO_CLIP
    if index < CHAINED_TO_CLIP:
        raise ValueError(
            f"chained_from {index} means nothing: -1 is the target clip and every other value is "
            f"a subject number.")
    if index == _subject_index(subject):
        raise ValueError(
            f"chained_from {index} is subject {index} itself. A pass cannot composite onto its "
            f"own output: name the subject whose stitched result it was composited onto, or -1 "
            f"for the target clip.")
    if index not in plan_subject_indices(plan):
        raise ValueError(
            f"chained_from {index} is not a subject of this plan. The subjects are "
            f"{', '.join(str(i) for i in plan_subject_indices(plan))}, or -1 for the target clip.")
    return index


def chain_label(value) -> str:
    """A chained_from value in words: which plate that pass was composited onto."""
    return "the target clip" if int(value) == CHAINED_TO_CLIP else f"subject {int(value)}"


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
# `max_subjects` USED to be left out because it was recorded and not applied; #10 made it
# truncate the subject set, so two caps are two jobs and it is in the list below.

PAIRING_INPUT_FIELDS = ("target_size", "still_size", "match_frame", "subject_source",
                        "match_tolerance", "excluded_faces", "unmatched_faces",
                        # #10: the cap used to be inert, so it stayed out of the fingerprint. It
                        # truncates the subject set now, so two caps are two jobs - and the
                        # fingerprint, not the plan version number, is what says so.
                        "max_subjects",
                        # #11 review: the crop factor decides the rectangle every subject
                        # reference in the folder was cut with, so a plan at another factor is a
                        # different job even when the pairing itself is identical.
                        "crop_factor")

# What the match frame actually MEASURED (#11 review). The subject indices say who was paired;
# these say which detections that pairing came from. A different still of the same size, or a
# different detector, can pair different faces into the SAME indices - and a resume on those
# composites passes cut from two different pairings into one clip.
PAIRING_EVIDENCE_FIELDS = ("target_boxes", "still_boxes", "scale")

# How the plates were detected, from the plan's own `run` record: same boxes by different
# settings is a coincidence worth refusing on, because the settings decide what else would have
# been found at that frame.
PAIRING_RUN_FIELDS = ("detector", "confidence")


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

    Both plate sizes, the match frame, the seeding plate, the tolerance, the exclusions, what was
    decided about faces that found no partner, the crop factor the references were cut at, and the
    subject set itself - index, clip face and still face for every subject. The subject set is in
    here because it is what actually breaks: a re-detect that pairs someone else at the same
    widgets is still a different job. `unmatched_faces` is in here for the same reason (#8): it
    does not re-run the pairing, it decides which of the paired faces become subjects at all.

    #11 review added the rest of the evidence: the boxes both plates measured at the match frame,
    their scale, and the detector and confidence that produced them. The subject set is the
    OUTCOME of a pairing; these are the measurements it came from, and two different pairings can
    land on the same outcome.
    """
    inputs = (plan or {}).get("inputs") or {}
    record = {}
    for field in PAIRING_INPUT_FIELDS:
        value = inputs.get(field)
        if value is None:
            value = (plan or {}).get(field)
        if field == "crop_factor":
            # A folder written before the plan recorded a factor cut its references at the
            # shipped default, so a missing field is that default and not a difference.
            value = plan_crop_factor(plan)
        record[field] = _normalise(value)
    for field in PAIRING_EVIDENCE_FIELDS:
        record[field] = _normalise((plan or {}).get(field))
    run = (plan or {}).get("run") or {}
    for field in PAIRING_RUN_FIELDS:
        record[field] = _normalise(run.get(field))
    record["subjects"] = [[_normalise((subject or {}).get("index")),
                           _normalise((subject or {}).get("clip_face")),
                           _normalise((subject or {}).get("still_face"))]
                          for subject in (plan or {}).get("subjects") or []]
    return record


def _describe(value) -> str:
    return json.dumps(value, sort_keys=True)


def _boxes_differ(a, b) -> bool:
    return [round(float(v), 3) for v in a] != [round(float(v), 3) for v in b]


def _boxes_line(field: str, before, after) -> str:
    """One box-list difference, in words a person can act on.

    A face box is four numbers and a plate has several faces, so dumping every coordinate of
    every box reads as noise. What matters is how many faces each side measured and which one
    moved first - that is the frame to go and look at.
    """
    before = list(before or [])
    after = list(after or [])
    parts = [f"{field}: the job folder has {len(before)} face(s) at the match frame, this plan "
             f"has {len(after)}"]
    shared = min(len(before), len(after))
    first = next((k for k in range(shared) if _boxes_differ(before[k], after[k])), None)
    if first is None and len(before) != len(after):
        parts.append(f"the extra face(s) start at index {shared}")
    elif first is not None:
        parts.append(f"first difference at face {first}: there {_describe(before[first])}, "
                     f"here {_describe(after[first])}")
    return "; ".join(parts)


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

    The lines are written for a person reading an error at a node: `field: the job folder has X,
    this plan has Y`, and for the box lists a count plus the first box that moved rather than
    every coordinate of every face.
    """
    before = pairing_fingerprint(stored_plan)
    after = pairing_fingerprint(current_plan)
    lines = []
    for field in PAIRING_INPUT_FIELDS:
        if before[field] != after[field]:
            lines.append(f"{field}: the job folder has {_describe(before[field])}, this plan "
                         f"has {_describe(after[field])}")
    for field in PAIRING_EVIDENCE_FIELDS + PAIRING_RUN_FIELDS:
        if before[field] == after[field]:
            continue
        if field.endswith("_boxes"):
            lines.append(_boxes_line(field, before[field], after[field]))
        else:
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
# the other names the widgets that would fix it. The lost half used to restate the diagnosis
# before the fix ("the tracker stopped finding them there"), which made the sentence three
# clauses long beside an absence line that is two - and the state is already carried by the
# `state` field and by the words "was lost", so the fix is all that is left to say.
ABSENT_ACTION = ("original pixels are kept there, which is the intended result for an absence - "
                 "if this person IS in this shot, plan on a match frame they share with the "
                 "reference still, or name them with Pick faces")
LOST_ACTION = ("check identity_track and identity_threshold, or lower confidence, "
               "on H3 Face Track + Crop")

# A subject with no subject reference (#8) is absent in a different way again: the pass never
# had a picture of them. Telling them to fix a pairing would send them back to a decision the
# plan already recorded, so their action names the widget that would change it instead.
UNPAIRED_ABSENT_ACTION = (
    "the frames in this shot keep the clip's own pixels, which is what a subject with no "
    "subject reference gets - unmatched_faces=with_still on H3 Multi-Face Plan would "
    "condition them on the whole reference still instead")


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
    detect there, while a shot that held plenty of faces lost this subject's identity. One clause
    each - the sentence these feed is read while a job is being debugged, and it has a fix to
    state as well.
    """
    boxes = (pick or {}).get("boxes") or []
    if not boxes:
        return ("the tracker reached this shot with this subject selected and produced nobody "
                "for those frames - their identity fell below the threshold, or the face was "
                "never detected")
    number = int((pick or {}).get("select_index", 0) or 0)
    widest = max((len(boxes[f]) for f in lost_frames if 0 <= f < len(boxes)), default=0)
    if widest <= number:
        return (f"no face was detected in this shot at this subject's position, the detector "
                f"reporting at most {widest} face(s) there, so there was nobody to track")
    return ("faces were detected in this shot but none of them matched this subject's identity")


def _absent_reason(entry, pick, shot_number: int, dropped: int) -> str:
    """Why this shot holds no subject for them, and who decided that.

    #6 left `entry` in the signature unused; #8 uses it. A subject with no subject reference has
    a different reason to explain - the pass had no picture of them at all - and the wording for
    a paired subject points the user at a pairing to fix, which for them is a decision that has
    already been taken and written into the plan.
    """
    picks = (pick or {}).get("picks") or []
    index = shot_number - 1
    planned = bool(picks[index].get("absent")) if 0 <= index < len(picks) else False
    if entry.get("has_reference") is False:
        if planned:
            return ("this subject has no subject reference and the plan marked this shot "
                    "absent for them, so the pass was never asked for a face here")
        return ("this subject has no subject reference, so the pass reached this shot with no "
                "picture of them to anchor on")
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
            # #8: a subject with no subject reference is told what that costs them here, not sent
            # off to fix a pairing the plan has already settled. An entry the plan does not hold
            # keeps the paired wording, because nothing here says otherwise.
            record["action"] = (UNPAIRED_ABSENT_ACTION
                                if (entry or {}).get("has_reference") is False
                                else ABSENT_ACTION)
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
                         f"{shot.get('action') or ABSENT_ACTION}")
        elif state == SHOT_LOST:
            span = (f"{shot['frames_lost']} of {shot['frames']} frame(s)"
                    + (f" (frames {shot['lost_start']}-{shot['lost_end']})"
                       if (shot["lost_start"], shot["lost_end"]) != (shot["start"], shot["end"])
                       else ""))
            lines.append(f"subject {index} was lost in {where}: {span} produced nobody - "
                         f"{shot['reason']}; {LOST_ACTION}")
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




