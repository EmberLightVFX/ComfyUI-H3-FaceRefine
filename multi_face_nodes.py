"""The multi-face nodes of ComfyUI-H3-FaceRefine, kept out of the released `nodes.py`
(ADR 0002, ADR 0006).

Ticket #2 lands one node: **H3 Multi-Face Plan**, which detects faces on both plates at the
match frame, pairs them, and settles the **subject plan** - plus the pairing preview, the
report, and the job folder the plan and every **subject reference** crop are written into.
The plan itself is decided in `plan_store.py`, which is pure; this module is where the
ComfyUI world is allowed to touch it.

Import boundary: nothing that needs ComfyUI, torch or the detector is imported at module
level. `INPUT_TYPES()` has to be callable in a test run that has only numpy installed, and
the contract that pins the widget order runs there (ComfyUI stores widget values
positionally, so that order is a promise - see `tests/test_node_contract.py`).

Ticket #4 adds the last node of the chain, **H3 Multi-Face Composite**: it reads every
subject's `stitched` stage out of the job folder and hands back the LAST one's frames, because
each pass composited onto the previous pass's result - so the finished clip is that subject's
stitched output, not a blend of all of them. It re-blends nothing (a second feather over a
feathered edge softens the work), and it refuses when a subject's pass is missing rather than
returning a half-composited clip.
"""

from __future__ import annotations

import os

import numpy as np

try:  # loaded as part of the pack
    from . import plan_store
except ImportError:  # imported by name from the pack root, as the unit tests do
    import plan_store

# The one new custom type in the feature (ADR 0005): the subject plan.
H3FACEPLAN = "H3FACEPLAN"
CUSTOM_TYPES = (H3FACEPLAN,)

# The released pick type, used as-is on purpose (ADR 0005): a pass emits an `H3FACEPICK`, so
# `H3FaceTrackCrop`, `H3PerFrameDenoise` and `H3FaceStitch` run unmodified inside every pass.
# Deliberately NOT in CUSTOM_TYPES - it is registered by `nodes.py`, and re-registering the
# released types is how saved graphs stop meaning what they meant (ADR 0002).
H3FACEPICK = "H3FACEPICK"

# The Subject Select widget's first option and its default: the cursor through a job.
NEXT_SUBJECT = "next"

# Subject numbers the widget offers, matching the released tracker's 0-63 face ceiling. The
# list cannot be the plan's own subjects: `INPUT_TYPES()` is a classmethod that never sees the
# wired plan, so naming the subjects is the report's job (and a front end's, later).
MAX_SUBJECT_CHOICES = 64

# Subject Select always processes the next subject with no saved pass (plan_store.next_subject);
# there is no subject/mode widget to set. The walk position is the job folder's pass manifest, so
# queueing the same graph advances one subject per completed pass, and a queue that crashes
# mid-pass is retried rather than skipped. To start the whole job over, press Reset run on the plan
# node (plan_store.reset_run), not a control here.


def _subject_at_index(plan, index):
    """The plan's subject dict with this index, or None (so the caller can name the choices)."""
    for subject in (plan or {}).get("subjects") or []:
        if int(subject["index"]) == int(index):
            return subject
    return None


def _subject_walk_note(plan, job, current):
    """The 'where am I in the walk' line under the always-next model: the subject being refined
    now, how many are already complete, and how many queues are still to run after this one."""
    counts = plan_store.run_counts(plan, job)
    note = (f"you are on subject {current} of {counts['total']}   "
            f"{counts['complete']} already complete, {max(counts['remaining'] - 1, 0)} still to "
            f"run after this one")
    if counts["remaining"] <= 1:
        note += "   (this is the last subject - the job is done once this pass is saved)"
    done = plan_store.completed_subjects(plan, job) if job else []
    if done:
        note += f"\n   recorded complete: subject(s) {', '.join(str(i) for i in done)}"
    return note


def _run_progress_line(plan, job) -> str:
    """The plan node's 'how many queues do I need' line: the subject count is the number of
    queues a fresh job runs (found faces minus the ignored ones), lowered by any pass already
    saved, and it names the subject the next queue will take."""
    counts = plan_store.run_counts(plan, job)
    if counts["total"] == 0:
        return "passes: this plan holds no subject (every face was excluded) - nothing to refine"
    parts = [f"passes: {counts['total']} subject(s) = {counts['total']} queue(s), one per subject"]
    if counts["complete"]:
        parts.append(f"{counts['complete']} already recorded complete")
    if counts["remaining"]:
        nxt = counts["next_index"]
        tail = f"   next queue takes subject {nxt}" if nxt is not None else ""
        parts.append(f"{counts['remaining']} queue(s) still to run{tail}")
    else:
        parts.append("all complete - press Reset run on this node to start the passes over")
    line = "; ".join(parts)
    warn = plan_store.frame_consistency_note(plan, job)
    return f"{line}\n{warn}" if warn else line


CATEGORY = "MiniMax H3/Face Refine"

# The pack's own tracker defaults. `H3FaceTrackCrop` ships crop_factor 2.5 and a 768x768 canvas.
#
# The CANVAS is only ever quoted as an estimate in the report: it is set on the tracker, on
# widgets this node cannot see, and it can switch to auto - so the report says which default it
# is standing on instead of reading like a measurement of the job.
#
# The CROP FACTOR is different: this node cuts the subject references, so it has to be asked for
# the factor (the `crop_factor` widget) and the plan records it. A reference cut at a factor the
# tracker will not use is a mis-framed reference.
DEFAULT_CROP_FACTOR = plan_store.DEFAULT_CROP_FACTOR
DEFAULT_CANVAS = (768, 768)

# One colour per subject, so a face on the clip and its partner in the still read as the
# same person across the gap between the plates.
_PALETTE = (
    (0.10, 0.75, 1.00), (1.00, 0.55, 0.05), (0.20, 0.95, 0.35), (0.95, 0.35, 0.85),
    (0.95, 0.90, 0.15), (0.55, 0.40, 1.00), (0.10, 0.90, 0.85), (0.85, 0.25, 0.20),
)
# A switched-off face: red, and struck through. Never a palette colour, so it cannot be
# mistaken for a subject.
_EXCLUDED_COLOUR = (1.00, 0.10, 0.10)
# A face that was found but did not become a subject (only possible when the subjects are
# seeded from the other plate).
_UNSEEDED_COLOUR = (0.55, 0.60, 0.68)


def _colour_of(index: int):
    return _PALETTE[int(index) % len(_PALETTE)]


def _detector_choices() -> list:
    """The detector list the tracker offers.

    Asked for lazily, and falling back to the shipped default when ComfyUI's model folders
    are not reachable, so a schema-only run (the contract tests) still gets a combo.
    """
    try:
        from .nodes import _detector_list
        return _detector_list()
    except Exception:  # no ComfyUI here: the schema still has to build
        return ["face_yolov8m.pt"]


def _derived_job_name(images, job_name) -> str:
    """The job folder name, from the widget or from the clip.

    An IMAGE batch carries no filename: neither this pack's loader nor VideoHelperSuite puts
    one on it. So the fallback names the batch by shape, and a loader that does have a name
    can hand it over on the batch (`images.h3_video_name`). The widget exists so a job can
    be named for the folder you already have.
    """
    wanted = str(job_name or "").strip()
    if wanted:
        return wanted
    hint = str(getattr(images, "h3_video_name", "") or "").strip()
    if hint:
        return os.path.splitext(os.path.basename(hint))[0]
    return f"clip_{int(images.shape[0]):04d}f_{int(images.shape[2])}x{int(images.shape[1])}"


def _draw_slash(img, x0, y0, x1, y1, colour, thickness: int) -> None:
    """Two strokes through a box: switched off, readable from across the room."""
    import torch

    steps = max(int(abs(x1 - x0)), int(abs(y1 - y0)), 1)
    stroke = torch.tensor(colour, dtype=img.dtype, device=img.device)
    h, w = int(img.shape[0]), int(img.shape[1])
    t = max(1, int(thickness))
    for k in range(steps + 1):
        f = k / float(steps)
        py = y0 + (y1 - y0) * f
        for px in (x0 + (x1 - x0) * f, x1 + (x0 - x1) * f):
            cx, cy = int(round(px)), int(round(py))
            for dy in range(t):
                yy = cy + dy
                if 0 <= yy < h:
                    for dx in range(t):
                        xx = cx + dx
                        if 0 <= xx < w:
                            img[yy, xx] = stroke


def _write_reference_png(path, frame, rect) -> tuple[int, int]:
    """Cut one subject's reference out of the still and write it at native resolution.

    A slice, not a resize: `plan_store.crop_rect_for_reference` already made sure the rect
    only covers pixels the still actually has.
    """
    from PIL import Image

    x, y, w, h = (int(v) for v in rect)
    crop = frame[y:y + h, x:x + w, :3]
    arr = (crop.clamp(0.0, 1.0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(path)
    return int(crop.shape[1]), int(crop.shape[0])


def _clip_crop_rect(box, W: int, H: int, crop_factor: float = None):
    """Turn a clip face's box into the crop rect the pairing panel shows for it (#9).

    Cut at the SAME crop_factor the subject reference was cut at - `plan_store.tracker_rect` in the
    clip's own pixels - so the clip-face thumb and the reference thumb frame the subject at one
    scale and can be compared side by side. (The old face-box + small padding read as far more
    zoomed in than the reference.) `tracker_rect` keeps the canvas aspect, centres on the face and
    clamps inside the frame; returns None if the crop collapses, so a degenerate detection just
    shows no clip crop rather than a bad slice.
    """
    if crop_factor is None:
        crop_factor = plan_store.DEFAULT_CROP_FACTOR
    x, y, w, h = plan_store.tracker_rect(box, (int(W), int(H)), float(crop_factor))
    x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
    x1, y1 = min(int(W), int(round(x + w))), min(int(H), int(round(y + h)))
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return [x0, y0, x1 - x0, y1 - y0]



def _write_preview_png(path, preview) -> str:
    """Write the pairing preview image the node already built into the job folder (#9).

    The same tensor the node returns as its `preview` output, as the same bytes: no second
    drawing pass, so the file the panel shows cannot disagree with what this node decided. The
    tensor is one BCHW image of both plates side by side, so it goes out at the height the
    preview is drawn at.
    """
    from PIL import Image

    frame = preview[0] if int(preview.dim()) == 4 else preview
    arr = (frame[..., :3].clamp(0.0, 1.0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(path)
    return path


def _pairing_preview(clip_frame, still_frame, clip_boxes, still_boxes, plan):
    """Both plates side by side: every face outlined and numbered, each pair in one colour,
    every switched-off face struck through.

    The numbers are the numbers `excluded_faces` takes, per plate, which is the point of
    showing them: pairing gets checked before a render is committed, not after.
    """
    import torch
    from .nodes import _draw_label, _draw_rect, _label_size

    h, w = int(clip_frame.shape[0]), int(clip_frame.shape[1])
    thick = max(2, int(round(min(h, w) / 240.0)))
    glyph = max(2, int(round(min(h, w) / 140.0)))
    _, label_h = _label_size("0", glyph)

    # Same height for both plates so the eye can carry a colour across the gap. The aspect
    # ratios already match - plan_store raises when they do not - so one resize is enough.
    if (int(still_frame.shape[0]), int(still_frame.shape[1])) != (h, w):
        resized = torch.nn.functional.interpolate(
            still_frame[..., :3].permute(2, 0, 1).unsqueeze(0), size=(h, w),
            mode="bilinear", align_corners=False)
        still_plate = resized[0].permute(1, 2, 0)
    else:
        still_plate = still_frame[..., :3]
    clip_plate = clip_frame[..., :3].clone()
    # Two plates can arrive on different devices or in different dtypes; the concat below is
    # one image, so they are put on the same terms as the clip.
    still_plate = still_plate.clone().to(dtype=clip_plate.dtype,
                                         device=clip_plate.device)

    # One colour per subject, shared by the two detections of that person.
    clip_colour: dict[int, tuple] = {}
    still_colour: dict[int, tuple] = {}
    for subject in plan["subjects"]:
        colour = _colour_of(subject["index"])
        if subject["clip_face"] is not None:
            clip_colour[int(subject["clip_face"])] = colour
        if subject["still_face"] is not None:
            still_colour[int(subject["still_face"])] = colour
    # The exclusions are per plate: a bare number strikes through the clip's face, an `s` number
    # the still's. Striking both at the same index would say a bystander had been switched off on
    # a plate that never had that face at all.
    off_clip, off_still = plan_store.exclusion_plates(plan["excluded_faces"])
    for face in off_clip:
        clip_colour[int(face)] = _EXCLUDED_COLOUR
    for face in off_still:
        still_colour[int(face)] = _EXCLUDED_COLOUR

    for plate, boxes, colours, scale, off in ((clip_plate, clip_boxes, clip_colour, 1.0,
                                               set(off_clip)),
                                              (still_plate, still_boxes, still_colour,
                                               float(plan["scale"]), set(off_still))):
        for i, box in enumerate(boxes):
            x0, y0, x1, y1 = (float(v) * scale for v in box[:4])
            colour = colours.get(i, _UNSEEDED_COLOUR)
            _draw_rect(plate, x0, y0, x1, y1, colour, thick * (2 if i in colours else 1))
            if i in off:
                _draw_slash(plate, x0, y0, x1, y1, colour, thick)
            _draw_label(plate, x0, max(0.0, y0 - label_h), str(i), glyph,
                        (0.0, 0.0, 0.0), colour)

    gap = torch.zeros((h, max(4, w // 60), 3), dtype=clip_plate.dtype,
                      device=clip_plate.device)
    return torch.cat([clip_plate, gap, still_plate], dim=1).unsqueeze(0)


def _line(label: str, text: str) -> str:
    """One report row: a label column, then what it means."""
    return f"{label:<16}{text}"


def _format_report(plan, frames: int, job_dir: str, refs: int, notes) -> str:
    """What the job costs, who is in it, and what was written where.

    Subjects, frames per pass and canvas size are the three numbers that decide whether to
    run the job at all, so they lead.
    """
    canvas_w, canvas_h = DEFAULT_CANVAS
    factor = plan_store.plan_crop_factor(plan)
    tw, th = plan["inputs"]["target_size"]
    sw, sh = plan["inputs"]["still_size"]
    scale = float(plan["scale"])
    lines = [
        _line("job", plan["run"]["job_name"]),
        _line("target clip", f"{tw}x{th}   {frames} frame(s)"
                             f"   match frame {plan['match_frame']}"),
        _line("reference still", f"{sw}x{sh}   still -> clip scale {scale:.3f}"
                                 f"   (1 clip px = {1.0 / scale:.3f} still px)"),
        # An estimate, and labelled as one: the canvas is H3 Face Track + Crop's, set on that
        # node, and this node cannot see it. Naming the count is the point; naming a size the
        # plan never measured as if it had is not.
        _line("cost (estimate)", f"{plan['subject_count']} subject(s) x {frames} frame(s)"
                                 f" @ {canvas_w}x{canvas_h}"
                                 f"   = {plan['subject_count'] * frames} frame(s) to generate"),
        _line("", f"({canvas_w}x{canvas_h} is H3 Face Track + Crop's crop canvas at its shipped "
                  f"default - what this job generates at is that node's canvas_width/"
                  f"canvas_height, or its auto canvas)"),
        # The one rectangle this node does decide, so it is stated with the number that decided
        # it: a reference cut at a different factor from the tracker's is a mis-framed reference.
        _line("crop factor", f"{factor:g}   each subject reference was cut at crop_factor "
                             f"{factor:g} x face height - it must match the crop_factor H3 Face "
                             f"Track + Crop uses for this job"),
        _line("pairing", f"{plan['inputs']['subject_source']}"
                         f"   tolerance {plan['inputs']['match_tolerance']:.3f}"
                         f" = {plan['tolerance_px']:.0f}px of the {tw}x{th} diagonal"
                         f"   {len(plan['pairs'])} of {len(plan['target_boxes'])}"
                         f" clip face(s) paired"),
    ]
    if plan["excluded_faces"]:
        lines.append(_line(
            "excluded",
            f"{plan_store.describe_exclusions(plan['excluded_faces'])}"
            f" - switched off, so they keep their original pixels"))
    else:
        lines.append(_line("excluded", "none"))
    if plan["inputs"]["max_subjects"]:
        lines.append(_line("max_subjects", str(plan["inputs"]["max_subjects"])))
    # #10: a tail cut has to be loud. Every dropped face is named where the user can find it, and
    # the number the cap saved sits next to the number the cap spends.
    for cap_line in plan_store.capped_lines(plan):
        lines.append(_line("capped", cap_line))
    capped_cost = plan_store.uncapped_cost_line(plan, frames)
    if capped_cost:
        lines.append(_line("", capped_cost))

    counts = plan_store.subject_kind_counts(plan)
    lines.append(_line(
        "unmatched",
        f"{plan['inputs'].get('unmatched_faces', plan_store.DEFAULT_UNMATCHED_FACES)}   "
        f"{counts[plan_store.KIND_PAIRED]} paired"
        f"   {counts[plan_store.KIND_NO_REFERENCE]} subject(s) with no reference"
        f"   {counts[plan_store.KIND_WHOLE_STILL]} subject(s) on the whole still"
        f"   {counts[plan_store.KIND_CLIP_REGION]} subject(s) on a clip-aligned still crop"
        f"   {counts[plan_store.KIND_LEFT_OUT]} face(s) left out"))

    for subject in plan["subjects"]:
        where = (f"clip face {subject['clip_face']}" if subject["clip_face"] is not None
                 else "not seen in the clip on this frame")
        crop = subject["ref_crop"]
        if subject.get("ref_kind") == plan_store.REF_STILL:
            ref = (f"reference is the WHOLE reference still, {crop[2]}x{crop[3]}px"
                   f" -> {subject['ref_file']}   whole-frame conditioning: H3 picks which face"
                   f" in it to apply")
        elif (subject["has_reference"] and subject["still_face"] is None
              and subject.get("ref_kind") == plan_store.REF_CROP):
            ref = (f"reference cut from the reference still at THIS clip face's region,"
                   f" {crop[2]}x{crop[3]}px -> {subject['ref_file']}   clip_region: right only if"
                   f" the still shares the clip's framing")
        elif subject["has_reference"]:
            ref = (f"reference from still face {subject['still_face']},"
                   f" {crop[2]}x{crop[3]}px at the still's native resolution"
                   f" -> {subject['ref_file']}")
        else:
            ref = ("NO subject reference - this pass runs with nothing wired into H3's"
                   " reference slots, so it is weaker than a paired subject")
        lines.append(_line(f"subject {subject['index']}", where + "   " + ref))

    lines.append(_line("written", job_dir))
    lines.append(_line("", f"{plan_store.PLAN_FILENAME} + {refs} subject reference"
                           f" file(s) + {plan_store.PREVIEW_FILENAME}"))
    if notes:
        lines.append(_line("notes", ""))
        lines.extend(_line("", f"- {note}") for note in notes)
    return "\n".join(lines)


def build_clip_face_pick(frames, src_size, detector, confidence, boxes, confs) -> dict:
    """The clip's `H3FACEPICK`, in exactly the shape `H3 Load Video + Face Select` writes (nodes.py:3353).

    The plan node already runs a detector on the clip, so it can carry every frame's boxes
    onward and spare the graph a separate picker node: plug the image sequence into `images` and
    the reference frame into `reference_still`, and wire this `face_pick` output into `H3 Subject
    Select` (and `H3 Face Track + Crop`) - no `H3 Load Video + Face Select` anywhere. The whole
    sequence is treated as ONE shot segment (this node does no cut detection); the per-frame
    boxes/confs travel untouched and Subject Select re-points the single pick at each subject in
    turn, then the released tracker crops from them without detecting.

    The key set matches the released payload verbatim: the tracker reads boxes, confs, segments,
    picks, frames, src_size, detector and confidence and nothing else, so the remaining keys are
    informational and mirror what the picker records. `picks` is one placeholder for the single
    segment - Subject Select replaces it with this subject's own per-segment pick.
    """
    n = int(frames)
    boxes = [list(b) for b in (boxes or [])]
    confs = [list(c) for c in (confs or [])]
    present = bool(boxes and boxes[0])
    return {
        "version": 1,
        "boxes": boxes,
        "confs": confs,
        "segments": [(0, n)],
        "picks": [{"segment": [0, n], "frame": 0 if present else -1,
                   "box": 0 if present else -1, "index": 0, "absent": not present}],
        "frames": n,
        "src_size": (int(src_size[0]), int(src_size[1])),
        "detector": str(detector),
        "confidence": float(confidence),
        "select": "rule",
        "select_index": 0,
        "rank_by": "left_most",
        "identity_model": None,
        "identity_threshold": None,
        "identity_scores": [],
    }


class H3MultiFacePlan:
    """Decide who the subjects of this job are.

    Detects the target clip at the match frame and the reference still, pairs the two
    positionally, and writes the subject plan plus every subject reference crop into the job
    folder - so no later pass needs the still loaded again. It also detects EVERY frame of the
    clip and emits that per-frame pick payload as a `face_pick` (`H3FACEPICK`) output, so the
    graph can be driven from just this node + a `Load Image` reference - no separate
    `H3 Load Video + Face Select` - by wiring `face_pick` into `H3 Subject Select`.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # WIDGET ORDER IS PERMANENT. ComfyUI stores widget values by position, so inserting
        # or reordering one silently reinterprets a saved workflow (ADR 0002). Anything new
        # goes at the end. tests/test_node_contract.py pins this list.
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "The target clip - the footage being fixed. Wire it from H3 "
                               "Load Video + Face Select or from VideoHelperSuite; this node "
                               "does not load video itself."}),
                "reference_still": ("IMAGE", {
                    "tooltip": "One high-quality image of the same scene, framed the same way "
                               "as the clip. Its faces supply the detail. Only the first "
                               "image of the batch is used."}),
                "detector": (_detector_choices(), {
                    "tooltip": "The face detector used on BOTH plates, so the two sets of "
                               "boxes mean the same thing."}),
                "confidence": ("FLOAT", {
                    "default": 0.35, "min": 0.05, "max": 0.95, "step": 0.05,
                    "tooltip": "The detector's score floor, as in the other nodes here. "
                               "Lower finds more small and turned faces on both plates."}),
                "match_frame": ("INT", {
                    "default": 0, "min": 0, "max": 8191, "step": 1,
                    "tooltip": "The frame of the target clip whose face positions line up "
                               "with the reference still - where pairing happens. Pick one "
                               "where every subject is clearly in shot; it need not be "
                               "frame 0."}),
                "subject_source": (list(plan_store.SUBJECT_SOURCES), {
                    "default": "union",
                    "tooltip": "Whose faces become subjects.\n"
                               "union (default): every face found on either plate, each "
                               "flagged for whether it has a subject reference - nothing that "
                               "was detected is dropped silently.\n"
                               "target_clip: only the clip's faces, when the clip detected "
                               "better.\n"
                               "reference_still: only the still's faces, when the soft clip "
                               "missed faces the still can see."}),
                "match_tolerance": ("FLOAT", {
                    "default": 0.05, "min": 0.0, "max": 0.5, "step": 0.005,
                    "tooltip": "How far apart the two detections of one face may sit and "
                               "still pair - a FRACTION OF THE CLIP'S DIAGONAL, so it means "
                               "the same at any resolution. The boxes must also overlap. "
                               "Raise it if the still is framed slightly tighter than the "
                               "clip; lower it in a crowd."}),
                "excluded_faces": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Face numbers to switch off, comma-separated, exactly as "
                               "numbered on this node's preview: 0, 3. A bare number is a "
                               "TARGET CLIP face; s2 is a REFERENCE STILL face - the only way "
                               "to switch off a bystander the photo saw and the clip never "
                               "did. An excluded face never becomes a subject and keeps its "
                               "original pixels, and the face paired to it falls to "
                               "unmatched_faces. A text widget, so it survives a reload with "
                               "no front end."}),
                "max_subjects": ("INT", {
                    "default": 0, "min": 0, "max": 64, "step": 1,
                    "tooltip": "0 = no cap. The cap is recorded in the plan for the passes to "
                               "act on."}),
                "job_name": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "The job folder under ComfyUI's output folder, holding the plan "
                               "and every subject reference crop. Left blank it is named "
                               "after the clip. Keep the same name to resume a job."}),
                "unmatched_faces": (list(plan_store.UNMATCHED_FACES), {
                    "default": plan_store.DEFAULT_UNMATCHED_FACES,
                    "tooltip": "What becomes of a face on the target clip that found no "
                               "matching face in the reference still. This decides the SUBJECT "
                               "SET, so it is recorded in plan.json and the pairing fingerprint "
                               "refuses to mix two answers into one job folder.\n"
                               "leave (default): do not make a subject of it at all - no pass "
                               "ever touches it, so it keeps its original pixels. Not a "
                               "failure: the safe default when the pairing might be wrong.\n"
                               "no_reference: a subject with no subject reference. Its pass "
                               "runs with nothing wired into H3's reference slots, which the "
                               "slots allow; the refinement is identity and quality driven and "
                               "comes out weaker than a paired subject.\n"
                               "with_still: a subject whose reference is the WHOLE reference "
                               "still. Conditioning is whole-frame, so H3 chooses which face in "
                               "the still it applies - not a guarantee the right face is used.\n"
                                "clip_region: a subject whose reference is cut from the still at "
                                "the region where this clip face sits - the same crop a paired "
                                "subject gets. Right where the still shares the clip's framing; on "
                                "a differently framed still it crops the wrong area, so prefer "
                                "with_still there.\\n"

                               "A face that did find its partner is unaffected by this."}),
                # crop_factor (#11 review) is not the newest entry any more - `pair_overrides`
                # (manual pairing, #12) follows it - but both sit at the end, and ComfyUI
                # replays saved graphs by position (ADR 0002), so everything older stays exactly
                # where it was. It is a widget on this node only because the subject reference has
                # to be cut with the tracker's rectangle; the plan still owns the geometry.
                "crop_factor": ("FLOAT", {
                    "default": DEFAULT_CROP_FACTOR, "min": 1.2, "max": 8.0, "step": 0.1,
                    "tooltip": "Set this to the crop_factor that H3 Face Track + Crop will use "
                               "for THIS job - the same number, on that node.\n\n"
                               "Each subject reference is cut with the rectangle that factor "
                               "gives: crop_factor x face height, centred on the face, scaled "
                               "into the reference still at its native resolution. Left at a "
                               "different number from the tracker's, EVERY subject reference in "
                               "the job is mis-framed - the detail H3 is handed sits somewhere "
                               "else in the crop than the face it belongs to.\n\n"
                               "Recorded in plan.json, so the overlap warnings, the pairing "
                               "preview and a resumed job all measure the crops this job really "
                               "cuts. It joins the pairing fingerprint, because the references "
                               "already in the job folder were cut at the old number."}),
                # pair_overrides (#12, manual pairing) is the newest widget, so it goes last. It
                # lets the Pairing preview panel re-point one face's partner to another face when
                # the automatic pairing matched the wrong two people.
                "pair_overrides": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Manual pairings, for when the automatic pairing matched the WRONG "
                               "two faces (close faces in a crowd, a detector that mixed up two "
                               "people). Click a face image in the Pairing preview panel to set "
                               "or write it by hand: comma-separated, each entry face=partner.\n\n"
                               "A face is a bare number for the TARGET CLIP or s2 for a REFERENCE "
                               "STILL face - the same numbers as excluded_faces and the preview. "
                               "So 0=s2 says clip face 0's reference is still face 2; s1=0 says it "
                               "the other way round; and 2=none or s3=none switch a face's partner "
                               "off so it falls to unmatched_faces.\n\n"
                               "A manual pairing over-rides the positional guess - it need not be "
                               "near or overlapping. It re-pairs the whole job, so the passes for "
                               "it must be re-run. Recorded in plan.json and part of the pairing "
                               "fingerprint, so a changed pairing cannot be mixed into a folder "
                               "planned without it. A text widget, so it survives a reload."}),
            },
        }



    RETURN_TYPES = (H3FACEPLAN, "IMAGE", "STRING", "INT", H3FACEPICK)
    RETURN_NAMES = ("plan", "preview", "report", "subject_count", "face_pick")
    FUNCTION = "run"
    # The preview and the report exist to be read BEFORE anything is queued, so this is an
    # output node: no preview node has to be wired on to check a pairing. The plan output
    # still feeds downstream.
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, job_name="", **kwargs):
        """Decide whether ComfyUI may reuse this node's cached output.

        ``run()`` writes ``plan.json`` as a SIDE EFFECT, and ComfyUI's output cache skips
        ``run()`` entirely whenever a node's inputs are unchanged. The pairing panel's
        **Process Plan** button therefore cannot re-plan into the same ``job_name``: its
        ``/replace`` route deletes ``plan.json``, the plan node's inputs are otherwise
        identical, so the cached output is served back and no fresh plan is ever written -
        the panel then reports "no plan yet".

        Keying the cache signature on the plan file's presence busts that deadlock:

        - an ordinary queue (``plan.json`` on disk, inputs unchanged) returns a stable value
          and stays cached, so the per-subject pass chain never re-detects;
        - a plan that has been cleared by Process Plan, or not written yet, returns ``NaN``
          (which never equals itself), forcing ``run()`` so the plan node writes a fresh plan;
        - a blank ``job_name`` derives its folder from the images, which IS_CHANGED is not
          handed reliably (ComfyUI guarantees only constants here), so it re-plans too - the
          reuse branch in ``run()`` keeps an already-planned job non-destructive.

        A changed widget still re-runs the node through the normal input-hash path: IS_CHANGED
        only adds invalidation, it never suppresses it, so the pairing-fingerprint guard and
        the "your settings differ, NOT applied" notice stay exactly as before.
        """
        name = str(job_name if job_name is not None else "").strip()
        if not name:
            return float("nan")
        try:
            import folder_paths
            job = plan_store.job_dir_for(folder_paths.get_output_directory(), name)
            if not os.path.isfile(plan_store.plan_path(job)):
                return float("nan")   # cleared or unwritten -> run() so a plan is (re)written
        except Exception:
            return float("nan")        # cannot locate the folder -> re-plan; run() re-derives
        return 0

    def run(self, images, reference_still, detector, confidence, match_frame,
            subject_source, match_tolerance, excluded_faces, max_subjects, job_name,
            unmatched_faces=plan_store.DEFAULT_UNMATCHED_FACES,
            crop_factor=DEFAULT_CROP_FACTOR, pair_overrides: str = ""):
        import folder_paths
        from .nodes import _load_detector, _to_bgr_u8

        if images is None or int(images.dim()) != 4:
            raise ValueError(
                "images: wire the target clip in as an IMAGE batch - from H3 Load Video + "
                "Face Select, or from VideoHelperSuite. This node does not load video.")
        frames = int(images.shape[0])
        if frames < 1:
            raise ValueError("the target clip arrived with no frames")
        height, width = int(images.shape[1]), int(images.shape[2])

        if reference_still is None:
            raise ValueError("reference_still: one high-quality image of the same scene, "
                             "framed the same way as the clip.")
        still_batch = reference_still
        if int(still_batch.dim()) == 3:
            still_batch = still_batch.unsqueeze(0)
        if int(still_batch.dim()) != 4 or int(still_batch.shape[0]) < 1:
            raise ValueError("reference_still: an IMAGE - one image of the same scene.")
        still = still_batch[0]
        notes: list[str] = []
        if int(still_batch.shape[0]) > 1:
            notes.append(f"reference_still held {int(still_batch.shape[0])} images; the "
                         f"first is the one paired")

        frame = int(match_frame)
        if not 0 <= frame < frames:
            used = min(max(frame, 0), frames - 1)
            notes.append(f"match_frame {frame} is not in this clip; paired on frame {used}")
            frame = used

        model = _load_detector(detector)

        def _detect(plate):
            found = model.predict(_to_bgr_u8(plate), conf=float(confidence),
                                  verbose=False)[0]
            if len(found.boxes):
                boxes = [[float(v) for v in q] for q in found.boxes.xyxy.tolist()]
                scores = getattr(found.boxes, "conf", None)
                confs = ([float(c) for c in scores.tolist()] if scores is not None
                         else [1.0] * len(boxes))
            else:
                boxes, confs = [], []
            return boxes, confs

        # Detect EVERY frame of the clip, not just the match frame. The match-frame detection
        # drives the plan; the whole-clip boxes become the `face_pick` payload this node emits so
        # the released tracker can crop every subject without a separate Load Video + Face Select
        # (ADR 0005 keeps it the one H3FACEPICK shape). One detection per frame; ultralytics is
        # deterministic, so the match-frame boxes here are exactly the boxes the plan pairs on.
        all_boxes, all_confs = [], []
        for _f in range(frames):
            _b, _c = _detect(images[_f])
            all_boxes.append(_b)
            all_confs.append(_c)
        clip_boxes, clip_confs = all_boxes[frame], all_confs[frame]
        still_boxes, still_confs = _detect(still)
        if not clip_boxes and not still_boxes:
            raise ValueError(
                f"No face found on either plate - clip frame {frame}, detector {detector}, "
                f"confidence {confidence}. Lower confidence, or check that reference_still "
                f"is the same scene as this frame of the target clip.")

        plan = plan_store.build_subject_plan(
            clip_boxes, clip_confs, still_boxes, still_confs,
            (width, height), (int(still.shape[1]), int(still.shape[0])),
            frame, str(subject_source), float(match_tolerance),
            plan_store.parse_excluded_faces(excluded_faces), int(max_subjects),
            str(unmatched_faces), float(crop_factor), str(pair_overrides))

        if not plan["subjects"]:
            if plan["excluded_faces"]:
                raise ValueError(
                    f"nothing to plan: every face found on frame {frame} is named in "
                    f"excluded_faces ({plan_store.describe_exclusions(plan['excluded_faces'])}). "
                    f"Clear the numbers that are not bystanders.")
            if plan["left_out"]:
                # `leave` spent the whole job. Empty is not an answer anyone wants to queue,
                # and the two modes that would make somebody of these faces are named.
                raise ValueError(
                    f"nothing to plan: unmatched_faces={unmatched_faces} left all "
                    f"{len(plan['left_out'])} face(s) found on frame {frame} out of the subject "
                    f"set, because none of them found a partner in the reference still. Use "
                    f"unmatched_faces=no_reference to refine them from the clip alone, or "
                    f"with_still to condition them on the whole still - or check that "
                    f"reference_still is framed like this frame of the clip.")
            raise ValueError(
                f"nothing to plan: subject_source={subject_source} seeds its subjects from a "
                f"plate holding no faces on frame {frame}. Use union, or lower confidence.")



        name = _derived_job_name(images, job_name)
        job = plan_store.job_dir_for(folder_paths.get_output_directory(), name)
        # What this plan was measured with, recorded BEFORE the job folder is read: the pairing
        # fingerprint compares the detector and the confidence, and a plan that has not yet said
        # which detector it used cannot be compared with a job folder that did. `still_note` says
        # which image was paired when a batch was wired in.
        plan["run"] = {"job_name": name, "frames": frames, "detector": str(detector),
                       "confidence": float(confidence),
                       "clip_source": (str(getattr(images, "h3_video_name", "") or "")
                                       or None)}
        plan["still_note"] = ("the first image of an N-image reference_still batch"
                              if int(reference_still.shape[0]) > 1 else "")
        # Kept so a reused (stored) plan still reports the detector/frame this run measured with.
        run_record = plan["run"]
        still_note = plan["still_note"]
        # Per-run bookkeeping for the pairing panel: reset first, mark_resumed sets it only if the
        # saved plan is reused below. A first plan (no folder) leaves it cleared, so the panel knows.
        plan_store.clear_resumed_flag(job)
        if os.path.isfile(plan_store.plan_path(job)):
            # The plan node only decides a pairing when there is nothing to reuse. A job folder
            # already holds one pairing's plan, references and passes, so re-running the flow here
            # REUSES that saved plan rather than re-cutting it: every output stays consistent with
            # the passes already in the folder, and re-running is never destructive. To write a new
            # plan into this name the user presses the pairing panel's Process Plan (which clears the
            # whole job - plan artifacts AND every saved pass - so the job starts over) or picks a new
            # job_name. If the node's widgets now
            # differ from how the job was planned, that change is deliberately NOT applied to this
            # run - applying it would orphan the existing passes - so the report says so and names
            # what differs, instead of raising the pass nodes' refusal.
            stored = plan_store.read_plan(job)
            changed = plan_store.changed_widget_lines(stored, plan)
            plan = stored
            plan["run"] = run_record
            plan["still_note"] = still_note
            plan_store.mark_resumed(job, changed)
            old_plan = plan_store.plan_version_note(stored)
            if old_plan:
                # Read, not migrated: the folder is only stale if the fingerprint says so.
                notes.append(old_plan)
            notes.append(
                f"reused the saved plan in {name} without re-cutting it: H3 Multi-Face Plan only "
                f"plans when there is no job yet, or after the pairing panel's Process Plan clears "
                f"this folder. The references and passes already in it are unchanged.")
            if changed:
                notes.append(
                    "NOTICE - your settings on this node differ from the plan already saved here, "
                    "so THIS run did NOT apply them (it reused the saved subject set so the "
                    "existing passes stay valid). To apply them, press the pairing panel's "
                    "Process Plan button (it WIPES this job - the plan AND every saved pass - and "
                    "re-plans from scratch, so every subject is then re-rendered) or set a new "
                    "job_name. What differs:")
                notes.extend("  - " + line for line in changed)
            refs = sum(1 for subject in plan["subjects"] if subject.get("ref_file"))
            written = plan_store.plan_path(job)
            # The preview output socket still has to carry an image, so redraw it from the saved
            # plan over the freshly loaded plates (drawing only - the file on disk is untouched).
            preview = _pairing_preview(images[frame], still, clip_boxes, still_boxes, plan)
        else:
            os.makedirs(job, exist_ok=True)
            # The crops are cut and written BEFORE the plan is written, so the plan records the
            # file each subject's reference lives in. That is what lets a later pass resume a job
            # with neither plate loaded again.
            refs = 0
            for subject in plan["subjects"]:
                # A `crop` reference is a cut of the still and a `still` reference (#8,
                # unmatched_faces=with_still) is all of it; either way it goes out through the same
                # writer, under the same subject_N.png name, so a later pass reads the two the same
                # way and never loads the still again.
                if subject["ref_crop"]:
                    file_name = f"subject_{int(subject['index'])}.png"
                    _write_reference_png(os.path.join(job, file_name), still,
                                         subject["ref_crop"])
                    subject["ref_file"] = file_name
                    refs += 1
            # Also cut each subject's face out of the match-frame clip, so the pairing panel can
            # show the clip face beside the reference (#9). Only a subject with a clip face on the
            # match frame gets one; a still-only subject has no clip_box, so the panel shows a
            # placeholder for it instead.
            clip_W, clip_H = int(images[frame].shape[1]), int(images[frame].shape[0])
            for subject in plan["subjects"]:
                box = subject.get("clip_box")
                if box:
                    # Cut the clip thumb at the crop_factor the reference was cut at (recorded in
                    # the plan), so the two thumbs frame the subject at the same scale.
                    rect = _clip_crop_rect(box, clip_W, clip_H, plan_store.plan_crop_factor(plan))
                    if rect is not None:
                        clip_name = f"clip_{int(subject['index'])}.png"
                        _write_reference_png(os.path.join(job, clip_name), images[frame], rect)
                        subject["clip_file"] = clip_name
            # #9: the pairing preview is written here too, before the plan is, so a folder that has
            # a plan also has everything the plan says it wrote. The panel reads this file rather
            # than drawing anything, which is what lets it stay read-only.
            preview = _pairing_preview(images[frame], still, clip_boxes, still_boxes, plan)
            _write_preview_png(os.path.join(job, plan_store.PREVIEW_FILENAME), preview)
            written = plan_store.write_plan(job, plan)

        # #8: what the plan decided about faces that found no partner, in the plan module's own
        # words - `leave` names every face it left out and says that leaving is not a failure.
        notes.extend(plan_store.unmatched_lines(plan))
        manual = plan_store.parse_pair_overrides(pair_overrides)
        if manual:
            notes.append("manual pairing: " + plan_store.describe_pair_overrides(manual))
        unseeded = [int(j) for j in plan["unpaired_still"]
                    if not any(s["still_face"] == j for s in plan["subjects"])]
        if unseeded:
            notes.append(
                f"reference still face(s) {', '.join(str(j) for j in unseeded)} are not "
                f"subjects while subject_source={subject_source}")

        report = _format_report(plan, frames, job, refs,
                                notes + [_run_progress_line(plan, job), f"plan: {written}"])
        # The whole-clip pick, in the released H3FACEPICK shape, so the graph needs no separate
        # Load Video + Face Select: wire this into H3 Subject Select's (and the tracker's) face_pick.
        face_pick = build_clip_face_pick(frames, (width, height), detector, confidence,
                                         all_boxes, all_confs)
        return (plan, preview, report, int(plan["subject_count"]), face_pick)


# ----------------------------------------------------------------------------
# H3 Subject Select: one subject out of the plan, one single-subject pick out to the
# released tracker (ADR 0005)
# ----------------------------------------------------------------------------

def subject_label(subject) -> str:
    """How a subject is named to a human: their number, their face in the clip, and the face
    in the reference still that became their subject reference."""
    index = int(subject["index"])
    clip_face = subject.get("clip_face")
    still_face = subject.get("still_face")
    where = f"clip face {clip_face}" if clip_face is not None else "not seen in the clip"
    if still_face is not None:
        ref = f"still face {still_face}"
    elif subject.get("ref_kind") == plan_store.REF_STILL:
        ref = "the whole reference still"      # unmatched_faces=with_still (#8)
    else:
        ref = "no subject reference"
    return f"{index}: {where} -> {ref}"


def resolve_subject(plan, choice, job_dir=None) -> dict:
    """The subject the widget asked for.

    ``next`` reads the job folder's pass manifest, so re-queueing the same graph walks the
    plan instead of re-running whoever was last. A bare subject number works, and so does a
    whole label - including one saved by an older label format, since the number leads it.
    Anything else raises with the choices spelled out: the plan is the only place the names
    come from, so a silent fallback would refine the wrong person.
    """
    subjects = (plan or {}).get("subjects") or []
    if not subjects:
        raise ValueError(
            "this subject plan holds no subjects, so there is nobody to select. Re-run "
            "H3 Multi-Face Plan - every face may have landed in excluded_faces."
        )
    wanted = str(NEXT_SUBJECT if choice is None else choice).strip()

    if wanted == NEXT_SUBJECT:
        if not job_dir:
            raise ValueError(
                "'next' has to read the job folder to know which passes are already done, "
                "and this plan records no job folder of its own. Run H3 Multi-Face Plan "
                "with job_name set, or name the subject instead of 'next'."
            )
        subject = plan_store.next_subject(plan, job_dir)
        if subject is None:
            raise ValueError(
                f"every subject in this plan already has a completed pass recorded in "
                f"{plan_store.manifest_path(job_dir)}. Nothing is left to refine: composite "
                f"the passes, or delete that manifest to run the job again."
            )
        return subject

    for subject in subjects:
        if subject_label(subject) == wanted:
            return subject
    head = wanted.split(":", 1)[0].strip()
    for subject in subjects:
        if wanted == str(subject["index"]) or head == str(subject["index"]):
            return subject

    raise ValueError(
        f"subject {choice!r} is not one this plan holds. Use {NEXT_SUBJECT} for the next "
        f"subject with no completed pass, or one of: "
        f"{', '.join(subject_label(s) for s in subjects)}"
    )


def _planned_face_index(plan_box, on_frame, clip_face, match_frame) -> int:
    """Which of the payload's boxes on the match frame is the face the plan measured.

    Overlap is the only honest way to say it, since the plan's detection pass and the picker's
    may have returned the same faces in a different order. The measure itself is geometry, so it
    lives in ``plan_store.box_iou`` and this is only the ranking over the payload's boxes.
    """
    best, best_iou = None, 0.0
    for i, box in enumerate(on_frame):
        score = plan_store.box_iou(plan_box, box)
        if score > best_iou:
            best, best_iou = i, score
    if best is not None and best_iou > 0.0:
        return best
    if 0 <= int(clip_face) < len(on_frame):
        # Nothing to overlap with: the two detections of this face sit far apart, so geometry
        # cannot settle it. Take the index the plan recorded, which is the order its own
        # detection came back in.
        return int(clip_face)
    raise ValueError(
        f"the pick found {len(on_frame)} face(s) on the plan's match frame {match_frame}, "
        f"and none of them is the face the plan recorded for this subject (clip face "
        f"{clip_face}). The plan and the pick are not two detections of the same frame. "
        f"Re-run H3 Multi-Face Plan with the same detector, confidence, match_frame and "
        f"frame range as H3 Load Video + Face Select."
    )


def _shot_lock(all_boxes, start: int, end: int, face_number: int, match_frame: int):
    """Where a shot locks onto the subject: the frame it measures them on, and which box.

    Searches from the match frame when the shot holds it - the frame the plan actually
    measured - then forward, then back to the shot start: the order the released
    ``_auto_pick`` uses for a hand-picked shot. A shot that never shows as many faces as the
    subject's number has nobody at that number to name, so it answers ``None`` and the caller
    marks it absent rather than handing the tracker whoever happens to be nearest.
    """
    first = match_frame if start <= match_frame < end else start
    for frame in list(range(first, end)) + list(range(start, first)):
        if len(all_boxes[frame]) > face_number:
            return frame, plan_store.left_to_right_face_order(all_boxes[frame])[face_number]
    return None


def _plan_match_frame(plan) -> int:
    """The frame the plan measured its subjects on - the plan's fact, not the payload's.

    #11 review: this used to be copied into the narrowed `H3FACEPICK` as `plan_match_frame`.
    ADR 0005 promises the tracker the released picker's key set verbatim, so a value the
    pass needs has to be asked of the plan (or said in the report), not smuggled through a
    payload that is meant to be indistinguishable from the loader's. -1 when no plan says.
    """
    plan = plan if isinstance(plan, dict) else {}
    return int(plan.get("match_frame",
                        (plan.get("inputs") or {}).get("match_frame", -1)))


def narrow_pick_to_subject(plan, face_pick, subject) -> dict:
    """One subject's `H3FACEPICK`: the clip's boxes untouched, `picks` rewritten for one person.

    The plan records the match frame only - it detects both plates once, by design - so the
    clip's own payload, from H3 Load Video + Face Select, carries every other frame. Nothing
    here detects, re-pairs or re-picks anything: detection stays exactly once in the graph.

    The plan's match-frame choice becomes a pick per shot through the subject's face NUMBER,
    counted left to right, because a cut renumbers the boxes: a pick entry names a box index
    on ONE frame, so carrying the plan's index across a cut would track a different person.
    A shot that never shows that many faces gets ``absent: true``, which is how the released
    payload already says "this person is not in this shot" (nodes.py:1334) - those frames keep
    their original pixels rather than having whoever is nearest refined in their place.

    The input payload is never modified. Two passes in one graph read the same payload, and a
    shared dict would leak one subject's picks into the next subject's run.
    """
    if not isinstance(face_pick, dict):
        raise ValueError(
            "face_pick is not a pick payload - wire H3 Load Video + Face Select's face_pick "
            "output here. Its boxes are what the pass crops, so detecting them again is not "
            "an option."
        )
    payload_boxes = face_pick.get("boxes")
    frames = int(face_pick.get("frames", -1))
    if not payload_boxes or frames < 0:
        raise ValueError(
            "face_pick carries no boxes, so it is not the payload H3 Load Video + Face "
            "Select emits. Wire that node's face_pick output into this one."
        )
    if len(payload_boxes) != frames:
        raise ValueError(
            f"face_pick says {frames} frame(s) but holds {len(payload_boxes)} - it is half a "
            f"payload. Re-run H3 Load Video + Face Select."
        )

    match_frame = _plan_match_frame(plan)
    if not 0 <= match_frame < frames:
        raise ValueError(
            f"the plan's match frame {match_frame} is outside the {frames} frame(s) this "
            f"pick describes, so the plan cannot name the subject in it. The plan and the "
            f"pick have to come from the same clip, untrimmed and un-resized between them."
        )

    src = face_pick.get("src_size") or (0, 0)
    planned = ((plan or {}).get("inputs") or {}).get("target_size")
    if planned and src and (int(src[0]) != int(planned[0]) or int(src[1]) != int(planned[1])):
        raise ValueError(
            f"the plan measured a {planned[0]}x{planned[1]} target clip, but this pick's "
            f"src_size is {src[0]}x{src[1]} - every box in it would land in the wrong place. "
            f"Wire the plan and the pick from the same clip: no resize, crop or scale between "
            f"H3 Load Video + Face Select and this node."
        )

    clip_face, plan_box = subject.get("clip_face"), subject.get("clip_box")
    if clip_face is None or plan_box is None:
        raise ValueError(
            f"subject {subject['index']} has no clip face - the plan found that head only in "
            f"the reference still, so no frame of the target clip can be named for it and a "
            f"pass would refine whoever else was in shot. Switch it off with excluded_faces "
            f"on H3 Multi-Face Plan, or select a different subject."
        )

    on_match = payload_boxes[match_frame]
    face_index = _planned_face_index(plan_box, on_match, clip_face, match_frame)
    face_number = plan_store.left_to_right_face_order(on_match).index(face_index)

    segments = [(int(a), int(b)) for a, b in (face_pick.get("segments") or [])]
    if not segments:
        segments = [(0, frames)]

    picks = []
    for a, b in segments:
        a, b = max(0, a), min(frames, b)
        lock = _shot_lock(payload_boxes, a, b, face_number, match_frame)
        if lock is None:
            picks.append({"segment": [a, b], "frame": -1, "box": -1,
                          "index": face_number, "absent": True})
        else:
            picks.append({"segment": [a, b], "frame": int(lock[0]), "box": int(lock[1]),
                          "index": face_number, "absent": False})

    narrowed = dict(face_pick)
    narrowed["picks"] = picks
    # Only keys the payload ALREADY has get new values, and only their values: ADR 0005 hands
    # the tracker the released picker's key set verbatim (nodes.py:3353), because a payload
    # with a key `H3FaceSelect` never writes is a different type labelled `H3FACEPICK`. The
    # tracker reads boxes, confs, segments, picks, frames, src_size, detector and confidence
    # from it and nothing else (nodes.py:1315-1551), so these three are informational,
    # re-pointed at this subject - and a payload that arrived without one leaves without it.
    # Anything the pass has to say about itself lives on the plan or in the report: whose pass
    # this is is `subject["index"]` (and the node's `subject_index` output), and the frame it
    # was measured on is the plan's `match_frame`.
    for key, value in (("select", "subject_plan"), ("select_index", face_number),
                       ("rank_by", "left_most")):
        if key in narrowed:
            narrowed[key] = value
    return narrowed


def absent_shots(narrowed) -> list:
    """The shot numbers this subject is not in, 1-based, as the report names them."""
    return [k + 1 for k, pick in enumerate(narrowed.get("picks") or []) if pick.get("absent")]


def read_pass_positions(job_dir, subject):
    """A subject's checkpointed crop positions, or ``None`` when that pass was never saved.

    Not an error: a first pass has nothing saved yet, and a report that failed on it would make
    the node unusable the one time the answer is most wanted.
    """
    if not job_dir:
        return None
    try:
        return plan_store.read_positions(job_dir, subject)
    except (FileNotFoundError, ValueError):
        return None


def shot_account_lines(plan, subject, transform=None, job_dir=None, pick=None) -> tuple:
    """This subject's shot account as report lines, plus the account itself (#6).

    The wording lives in `plan_store`, so the sentences pinned by the pure tests are the ones a
    user reads here: which shots this subject is absent from, and each one marked either
    ``absent_planned`` (the plan leaves them out, and the clip's own pixels stay) or lost (the
    tracker got there and produced nobody). No widget asks for anything: the crop positions are
    already wired into H3 Pass Save, and otherwise they are read back from the pass folder.
    """
    if transform is None:
        transform = read_pass_positions(job_dir, subject)
    account = plan_store.absence_account(plan, subject, transform=transform, pick=pick)
    return [_line("shots", text) for text in account["lines"]], account



def subject_report(plan, subject, narrowed, job_dir=None, reference_note: str = "",
                   cursor_note: str = "") -> str:
    """Who was chosen, which frame each shot locks onto, and where the reference is.

    A pass gets queued over and over without anyone re-reading the graph, so the report is the
    review step: it names the subject and their pairing, says which shots they are absent from
    and what that costs, and says what the tracker must NOT be handed - the identity anchor is
    rebuilt from this subject's own crops in the target clip, never from the still (ADR 0003).
    """
    subjects = (plan or {}).get("subjects") or []
    picks = narrowed.get("picks") or []
    face_number = int(narrowed.get("select_index", 0))
    # The match frame comes from the plan (#11 review): the payload the tracker sees carries
    # the released picker's keys and nothing else, so the report asks the plan that measured
    # it rather than reading a key Subject Select invented.
    match_frame = _plan_match_frame(plan)
    paired = subject.get("still_face") is not None
    on_match = 0
    try:
        if match_frame >= 0:
            on_match = len((narrowed.get("boxes") or [])[int(match_frame)])
    except (IndexError, TypeError, ValueError):
        on_match = 0

    if paired:
        pairing = f"still face {subject['still_face']} (paired)"
    elif subject.get("ref_kind") == plan_store.REF_STILL:
        # #8: this subject has a reference - the whole still - so saying they have none would be
        # wrong, and the cost of that reference belongs where the pass is being read.
        pairing = ("the whole reference still (unpaired - whole-frame conditioning, so H3 "
                   "chooses which face in it to apply)")
    else:
        pairing = "no subject reference (unpaired)"
    lines = [
        _line("chosen", f"subject {subject['index']} of {len(subjects)}   "
                        f"clip face {subject.get('clip_face')}   ->   {pairing}"),
        _line("pick", f"{len(picks)} shot(s) at face {face_number + 1} of {on_match},"
                      f"   measured on match frame {match_frame}"),
    ]
    for k, pick in enumerate(picks):
        a, b = int(pick["segment"][0]), int(pick["segment"][1])
        if pick.get("absent"):
            where = ("absent - no frame of this shot shows this face number, so those frames"
                     " keep their original pixels. If this person IS in this shot, name them"
                     " with Pick faces or plan on a match frame they share with the still.")
        else:
            where = f"locks on frame {pick['frame']}, box {pick['box']}"
        lines.append(_line("", f"shot {k + 1} frames {a}-{b - 1} ({b - a}): {where}"))

    lines.append(_line(
        "pass", f"no detection - the clip's boxes ride in on the pick payload"
                f"   (detector {narrowed.get('detector')}, confidence "
                f"{narrowed.get('confidence')}, {narrowed.get('frames')} frame(s))"))
    lines.append(_line(
        "anchor", "leave identity_reference to the tracker: it rebuilds this subject's "
                  "identity anchor from their matched crops in the target clip (ADR 0003). "
                  "The subject reference is H3's picture input, not an identity input; "
                  "wiring it into identity_reference anchors the track on the wrong domain."))
    if job_dir:
        lines.append(_line("job", job_dir))
    if reference_note:
        lines.append(_line("reference", reference_note))
    if cursor_note:
        lines.append(_line("next", cursor_note))
    gone = absent_shots(narrowed)
    if gone:
        lines.append(_line(
            "absent", f"shot(s) {', '.join(str(k) for k in gone)} left untouched for this "
                      f"subject - they cost no generated frames here, and their faces stay "
                      f"exactly as they are in this pass"))
    # #6: the same account H3 Pass Save prints, in the one place a user reads before queueing
    # the pass. Planned absence is the plan's decision and is expected; a lost track is not, and
    # the two are worded differently so nobody chases a non-problem. If this subject already has
    # crop positions in the job folder, the tracker's half of the answer is read from them.
    shot_lines, _account = shot_account_lines(plan, subject, job_dir=job_dir, pick=narrowed)
    lines.extend(shot_lines)
    return "\n".join(lines)


def _output_root() -> str:
    """ComfyUI's output folder, asked for where importing ComfyUI is allowed."""
    import folder_paths

    return str(folder_paths.get_output_directory())


def job_dir_for_plan(plan):
    """The job folder the plan was written into, or ``None`` when the plan records none.

    The plan carries its own ``run.job_name``, so a pass needs no widget to find its own
    folder - and two jobs in one output tree cannot read each other's references.
    """
    name = ((plan or {}).get("run") or {}).get("job_name")
    if not name:
        return None
    return plan_store.job_dir_for(_output_root(), str(name))


def _read_reference_png(path):
    """One subject reference crop off disk, as a one-frame IMAGE batch.

    Here, and not in `plan_store`, because this is the ComfyUI half of the job folder: the
    crop is already cut at the still's native resolution, so a pass never loads the reference
    still again - which is also why a reference can never be upscaled.
    """
    from PIL import Image

    import torch

    with Image.open(path) as plate:
        arr = np.asarray(plate.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _raise_reference_to_h3_floor(reference):
    """Make a subject reference big enough for H3 to condition on, if it is not already.

    H3 VAE-encodes the reference with a /16 downsample and a DiT patch conv with a 7-wide
    kernel, so a crop whose short side gives a latent under 7 crashes the model. A freshly
    built plan never cuts such a crop (`crop_rect_for_reference` floors the cut), but a job
    folder cut before that floor - or a still genuinely smaller than the floor - still reaches a
    pass off disk, so the IMAGE itself is lifted to the floor here. This resamples rather than
    leaving H3 to raise: the identity anchor is the target-clip crop (ADR 0003), the reference
    is secondary conditioning, so an enlarged reference is a weaker cue, not a broken graph.

    Returns ``(reference, upscaled_bool)``.
    """
    import torch

    h, w = int(reference.shape[1]), int(reference.shape[2])
    edge = int(plan_store.MIN_H3_REF_EDGE)
    if h >= edge and w >= edge:
        return reference, False
    factor = max(edge / float(h), edge / float(w))
    nh, nw = max(edge, int(round(h * factor))), max(edge, int(round(w * factor)))
    img = reference.permute(0, 3, 1, 2)  # IMAGE is [B,H,W,C] -> [B,C,H,W] for interpolate
    img = torch.nn.functional.interpolate(img.float(), size=(nh, nw),
                                          mode="bicubic", align_corners=False)
    img = img.clamp_(0.0, 1.0).permute(0, 2, 3, 1).contiguous()
    return img, True


def load_subject_reference(job_dir, subject):
    """``(IMAGE or None, 1/0, what the report says about it)``.

    A subject with no subject reference is a normal state of a plan, not an error: that pass
    runs with nothing wired into H3's reference slots. ``None`` rather than a black
    placeholder, so a graph that insists on a reference fails loudly instead of generating
    from mush dressed up as detail.
    """
    name = (subject or {}).get("ref_file")
    index = (subject or {}).get("index")
    kind = (subject or {}).get("ref_kind")
    if not name:
        why = ("unmatched_faces made them a subject without one"
               if kind == plan_store.REF_NONE else
               "no face in the reference still paired with this clip face")
        return None, 0, (f"subject {index} has no subject reference: {why}, so this pass runs "
                         f"with nothing wired into H3's reference slots")
    if not job_dir:
        return None, 0, (f"the plan records no job folder, so {name} cannot be read: run "
                         f"H3 Multi-Face Plan with job_name set")
    path = os.path.join(str(job_dir), str(name))
    if not os.path.isfile(path):
        return None, 0, (f"{name} is not in {job_dir}: this subject has no subject reference "
                         f"on disk, so this pass runs with nothing wired into H3's reference "
                         f"slots")
    reference = _read_reference_png(path)
    reference, upscaled = _raise_reference_to_h3_floor(reference)
    size = f"{int(reference.shape[2])}x{int(reference.shape[1])}px"
    bump = "" if not upscaled else (
        f"   NOTE: the saved crop was under H3's minimum reference size "
        f"({plan_store.MIN_H3_REF_EDGE}px), so it was enlarged to that floor - a smaller one "
        f"would crash H3's reference encode, and the still held no more detail at that face")
    if kind == plan_store.REF_STILL:
        # The whole still went in, so the whole still comes back out (#8): same file
        # convention, same read path, whole-frame conditioning instead of a face.
        return (reference, 1,
                f"{name}   the WHOLE reference still, {size}, read from the job folder - the "
                f"conditioning is whole-frame, so H3 chooses which face in it to apply" + bump)
    return (reference, 1,
            f"{name}   {size}, read from the job folder - the reference still is not loaded "
            f"again" + bump)


class H3SubjectSelect:
    """Choose one **subject** out of the subject plan and emit a pick payload for that one
    person, in the shape `H3 Face Track + Crop` already accepts (ADR 0005).

    Two inputs, because two things decide this: the **plan** says which face is the subject,
    which still crop is their subject reference, and whether they were paired at all; the
    clip's own **pick payload** carries a box for every frame of the clip, which the plan never
    measured. Detection therefore stays exactly once in your graph - H3 Load Video + Face
    Select does it, this node reads it, and the tracker that follows runs no detection pass at
    all.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # WIDGET ORDER IS PERMANENT. ComfyUI stores widget values by position, so inserting or
        # reordering one silently reinterprets a saved workflow (ADR 0002). Anything new goes
        # at the END. tests/test_node_contract.py pins this list.
        #
        # There is deliberately NO identity_reference input here, and no reference_still one.
        # ADR 0003: each subject's identity anchor is rebuilt from their own matched crops in
        # the TARGET CLIP, which the released tracker already does by itself once a face_pick
        # names the subject per shot and identity_reference is left unwired. Feeding the
        # subject reference into it would anchor the track on a different image domain - the
        # measured 0.305-similarity failure mode (nodes.py:1104). The subject reference this
        # node outputs belongs in H3's ref_image slots and nowhere else.
        return {
            "required": {
                "plan": (H3FACEPLAN, {
                    "tooltip": "The subject plan, from H3 Multi-Face Plan: who the subjects "
                               "are, which face in the target clip each one is, and which "
                               "face in the reference still became their subject reference."
                               "\n\nIt records the match frame's boxes only - that is where "
                               "both plates were detected - so it cannot name this subject in "
                               "the other shots on its own. That is what face_pick is for."}),
                "face_pick": (H3FACEPICK, {
                    "tooltip": "This clip's own pick payload, from H3 Load Video + Face "
                               "Select - the same loader the images batch comes from."
                               "\n\nEvery frame's boxes, confidences and shot boundaries "
                               "travel in it, so the pass crops without detecting anything. "
                               "Its picks are replaced with this subject's; the boxes pass "
                               "through untouched, which is what lets the tracker skip its "
                               "detection pass entirely."
                               "\n\nWire this into H3 Face Track + Crop's face_pick input too. "
                               "Do not resize, crop or trim between the loader and this node: "
                               "every box would land in the wrong place."}),
            },
        }

    RETURN_TYPES = (H3FACEPICK, "IMAGE", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("face_pick", "subject_reference", "reference_exists", "subject_index",
                    "previous_subject", "report")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, plan, face_pick):
        if not isinstance(plan, dict) or "subjects" not in plan:
            raise ValueError(
                "plan is not a subject plan - wire it from H3 Multi-Face Plan's plan output, "
                "or read it out of the job folder's plan.json."
            )
        if not (plan.get("subjects") or []):
            raise ValueError(
                "this subject plan holds no subjects, so there is nobody to refine - every face "
                "landed in excluded_faces. Re-run H3 Multi-Face Plan with the faces you want kept."
            )
        job = job_dir_for_plan(plan)
        # Always the next subject that has no saved pass: the walk IS the job folder's pass
        # manifest, so queueing the same graph advances one subject per completed pass, and a queue
        # that died mid-pass is taken again rather than skipped (self-healing). There is no
        # subject/mode widget to drive it - press Reset run on the plan node to start over.
        if job:
            chosen = plan_store.next_subject(plan, job)
            if chosen is None:
                raise ValueError(
                    "every subject in this plan already has a completed pass, so there is nobody "
                    "left to refine. Press 'Reset run' on the H3 Multi-Face Plan node to clear the "
                    "pass bookkeeping and start the whole job over from the first subject."
                )
        else:
            # No job folder (job_name blank on an unqueued plan): completion cannot be read, so take
            # the first subject - the same first step a job folder would name for a fresh job.
            chosen = (plan.get("subjects") or [None])[0]
        current = int(chosen["index"])

        narrowed = narrow_pick_to_subject(plan, face_pick, chosen)
        reference, exists, reference_note = load_subject_reference(job, chosen)
        cursor = _subject_walk_note(plan, job, current)

        report = subject_report(plan, chosen, narrowed, job_dir=job,
                               reference_note=reference_note, cursor_note=cursor)
        print("[H3SubjectSelect] " + report.replace("\n", "\n[H3SubjectSelect] "))
        # The subject this pass must composite ONTO: the target clip (-1) for the first subject
        # in plan order, otherwise the previous subject. It comes from the plan, not a per-copy
        # widget, so a single reusable pass chain can serve any subject: wire this into H3 Pass
        # Save's `chained_from` and H3 Pass Load's `subject` and the chain configures itself for
        # whichever subject `next` resolves to. ADR 0006 - the composite checks the same rule.
        previous = int(_expected_chain_base(plan, int(chosen["index"])))
        return (narrowed, reference, int(exists), int(chosen["index"]), previous, report)


# ----------------------------------------------------------------------------
# H3 Pass Save / H3 Pass Load: one subject's pass, checkpointed in the job folder (#5)
# ----------------------------------------------------------------------------
#
# The pass itself stays the released pipeline (ADR 0005): tracker, H3, stitch, untouched. What
# is checkpointed is what a pass PRODUCES - the crops it was fed, the frames it composited
# back, and the crop positions that put them where they belong - so a resumed job never re-runs
# a detection pass, a track, or a generation for a subject that is already done.

# The same advice on every refusal, because the reason is the same every time: a job folder
# holds ONE pairing's passes, and mixing two pairings into one clip is what these checks
# exist to prevent.
PAIRING_ADVICE = (
    "Set job_name on H3 Multi-Face Plan to a new name for this pairing, or re-run that node "
    "deliberately with the SAME inputs to resume this job. If the inputs really are meant to "
    "change, rename or delete the job folder first: every pass already inside it was cut from "
    "the inputs above."
)


def checked_plan(plan, label: str = "plan") -> dict:
    if not isinstance(plan, dict) or "subjects" not in plan:
        raise ValueError(
            f"{label} is not a subject plan - wire it from H3 Multi-Face Plan's plan output, "
            f"or read it out of the job folder's plan.json.")
    return plan


def job_folder_for_plan(plan, node: str) -> str:
    job = job_dir_for_plan(plan)
    if not job:
        raise ValueError(
            f"the plan records no job folder, so {node} has nowhere to read or write: run H3 "
            f"Multi-Face Plan with job_name set.")
    return job


def checked_subject(plan, choice) -> int:
    """A subject number this plan actually has.

    The manifest keys and the folder names ARE this number, so a number the plan does not have
    would checkpoint a subject nobody planned and resume a job on a phantom.
    """
    try:
        index = int(choice)
    except (TypeError, ValueError):
        raise ValueError(f"subject must be a subject number, got {choice!r}") from None
    known = plan_store.plan_subject_indices(plan)
    if index not in known:
        raise ValueError(
            f"subject {index} is not in this plan - its subjects are "
            f"{', '.join(str(i) for i in known) or 'none'}. Use a number H3 Multi-Face Plan "
            f"printed in its report and on the pairing preview.")
    return index


def _expected_chain_base(plan, index):
    """What THIS plan's order says subject `index` composites onto: the target clip for the first
    subject, the previous subject for every later one. The composite checks the same rule from the
    manifest, so both ends of the chain quote one answer (ADR 0006)."""
    order = plan_store.plan_subject_indices(plan)
    if index not in order or order.index(index) == 0:
        return plan_store.CHAINED_TO_CLIP
    return order[order.index(index) - 1]


def checked_image_batch(images, label: str) -> tuple[int, int, int]:
    """An IMAGE batch, as (frames, width, height) - float 0-1, or say so plainly."""
    if images is None or int(images.dim()) != 4:
        raise ValueError(
            f"{label}: wire an IMAGE batch - the crops from H3 Face Track + Crop, or the "
            f"stitched frames from H3 Face Stitch.")
    count, height, width = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
    if count < 1:
        raise ValueError(f"{label} arrived with no frames")
    if str(images.dtype) not in ("torch.float32", "torch.float64", "torch.float16"):
        raise ValueError(
            f"{label} is a {images.dtype} batch. This pack's IMAGE batches are float 0-1, so "
            f"convert it before checkpointing - a uint8 batch would be written back as noise.")
    return count, width, height


def _checkpoint_report_lines(job, index, stage, written, complete, note: str = "") -> list[str]:
    """The lines both pass nodes print, from the manifest record they share."""
    folder = plan_store.stage_dir(job, index, stage)
    lines = [f"subject {index}   stage {stage}   {written.get('frames')} frame(s) at "
             f"{written.get('width')}x{written.get('height')}"]
    lines.append(f"images: {folder}")
    if written.get("first"):
        lines.append(f"          {written['first']} ... {written['last']}   ordinary PNGs: "
                     f"open them to see what this pass was actually fed")
    if written.get("positions"):
        lines.append(f"crop positions: {plan_store.positions_path(job, index)}")
    if note:
        lines.append(note)
    lines.append("pass COMPLETE - crops and stitched are both recorded" if complete
                 else "pass NOT complete yet: a subject only counts as done once its stitched "
                      "stage is recorded, so a half-finished pass is resumed, not skipped")
    lines.append(f"manifest: {plan_store.manifest_path(job)}")
    return lines



class H3PassSave:
    """Write one subject's pass into the job folder: the frames as an ordinary numbered image
    sequence, the crop positions as readable JSON, and the manifest entry saying how far this
    subject got.

    Only the **stitched** stage completes a pass. A `crops` entry means "this far got done", so
    a job that died after the crops keeps that work and still resumes on the same subject
    instead of declaring it finished.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # WIDGET ORDER IS PERMANENT (ADR 0002): ComfyUI stores widget values by position.
        # Anything new goes at the end. tests/test_node_contract.py pins this list.
        return {
            "required": {
                "plan": (H3FACEPLAN, {
                    "tooltip": "The subject plan, from H3 Multi-Face Plan. It carries the job "
                               "folder and the subject set, so this node needs no job_name of "
                               "its own - and it is checked against the plan.json already in "
                               "that folder before a single frame is written."}),
                "subject": ("INT", {
                    "default": 0, "min": 0, "max": MAX_SUBJECT_CHOICES - 1, "step": 1,
                    "tooltip": "Which subject's pass this is - the number H3 Subject Select "
                               "took for it. Wire that node's subject_index output in and the "
                               "two can never disagree. A number the plan does not have is "
                               "refused, rather than checkpointing a subject nobody planned."}),
                "stage": (list(plan_store.PASS_STAGES), {
                    "default": "crops",
                    "tooltip": "Which half of the pass is being checkpointed.\n\n"
                               "crops (default): the crops H3 was fed, plus their positions. "
                               "This does NOT finish a subject.\n\n"
                               "stitched: the frames composited back onto the clip. Recording "
                               "this is what completes a subject's pass and lets the job "
                               "resume past them."}),
                "images": ("IMAGE", {
                    "tooltip": "The frames to write: the crop batch for stage=crops, the "
                               "stitched frames for stage=stitched. Written as PNGs in frame "
                               "order - frame_0000.png onward - so you can open them and see "
                               "exactly what H3 was given. The names never change: re-running "
                               "a pass overwrites its own frames instead of leaving a second "
                               "copy nobody can line up with frame 7."}),
            },
            "optional": {
                "transform": ("H3FACEXFORM", {
                    "tooltip": "The crop positions, from H3 Face Track + Crop. Required for "
                               "stage=crops - without them the crops cannot be put back where "
                               "they were cut. For stage=stitched there is nothing new to "
                               "record, so leave it unwired."}),
                # chained_from (#11 review) is appended AFTER transform for the same reason the
                # required widgets are appended at the end: ComfyUI replays saved graphs by
                # position (ADR 0002), so nothing that already existed moves.
                "chained_from": ("INT", {
                    "default": plan_store.CHAINED_TO_CLIP,
                    "min": plan_store.CHAINED_TO_CLIP, "max": MAX_SUBJECT_CHOICES - 1, "step": 1,
                    "tooltip": "Which plate THIS pass was composited onto: -1 means the target "
                               "clip; any other number is the subject whose stitched result this "
                               "pass was composited onto.\n\n"
                               "H3 Multi-Face Composite returns the LAST subject's stitched "
                               "frames on the claim that every earlier pass is already carried "
                               "inside them. That is only true if each pass was composited onto "
                               "the previous pass's result, which no node in the graph can see - "
                               "so it is declared here and checked there.\n\n"
                               "For the first pass in the plan's order, leave it at -1. For every "
                               "later pass, set it to the subject before it: subject 1 says 0, "
                               "subject 2 says 1. Wire every pass onto the target clip instead "
                               "and each pass throws the previous subject's work away, so the "
                               "finished clip ends up with exactly one refined face in it. The "
                               "composite refuses when what is declared here contradicts the "
                               "plan's order."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "INT")
    RETURN_NAMES = ("images", "report", "subject_index", "pass_complete")
    FUNCTION = "run"
    # The report and pass_complete are meant to be read off the node at the end of a pass;
    # `images` passes straight through, so a save can sit in the middle of the chain too.
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def run(self, plan, subject, stage, images, transform=None,
            chained_from=plan_store.CHAINED_TO_CLIP):
        plan = checked_plan(plan)
        job = job_folder_for_plan(plan, "H3 Pass Save")
        # Refuse before two pairings mix: the plan in the graph against the plan.json already
        # in this folder, naming every input that differs.
        plan_store.check_pairing_inputs(plan_store.read_plan(job), plan, job, PAIRING_ADVICE)
        plan_store.check_manifest_subjects(plan, job)

        index = checked_subject(plan, subject)
        name = plan_store.checked_stage(stage)
        count, width, height = checked_image_batch(images, "images")
        # #11 review: this is the node that knows what it was composited onto, so this is where
        # the chain is DECLARED. H3 Multi-Face Composite reads it back and refuses if it
        # contradicts the plan's own order, instead of asserting the chain it cannot see.
        base = plan_store.checked_chained_from(plan, index, chained_from)

        files = plan_store.stage_files(job, index, name, count)
        os.makedirs(os.path.dirname(files[0]), exist_ok=True)
        # The same PNG writer the subject references use: float 0-1 in, 8-bit PNG out.
        for path, frame in zip(files, images):
            _write_reference_png(path, frame, (0, 0, width, height))

        written = {"frames": count, "width": width, "height": height,
                   "first": os.path.basename(files[0]),
                   "last": os.path.basename(files[-1]),
                   # The declared chain, recorded with the stage it describes so the composite
                   # can check the claim against the plan order instead of asserting it.
                   "chained_from": base}
        note = ""
        if name == "crops":
            if transform is None:
                raise ValueError(
                    "transform: the crops stage records where each crop sat in its frame, "
                    "which is the only thing that lets the stitched result be put back. Wire "
                    "H3 Face Track + Crop's transform into this input, or save "
                    "stage=stitched.")
            written["positions"] = os.path.basename(
                plan_store.write_positions(job, index, transform))
        elif transform is not None:
            note = ("transform was wired but nothing new to record: a subject's crop positions "
                    "belong to its crops stage, and rewriting them from a stitched pass would "
                    "put those frames back somewhere other than where they were cut")
        else:
            note = ("transform left unwired, and there is nothing new to record: a stitched "
                    "pass's crop positions are the ones its crops stage saved, and they stay "
                    "exactly as they were written")

        plan_store.record_pass_stage(job, index, name, written)
        complete = 1 if plan_store.pass_is_complete(job, index) else 0
        lines = _checkpoint_report_lines(job, index, name, written, complete, note)
        # The declared chain, echoed back so a wrong widget is visible at the node that took it
        # and not only at the composite that refuses on it 20 minutes of generation later.
        expected = plan_store.chain_label(_expected_chain_base(plan, index))
        lines.append(_line("chained onto", plan_store.chain_label(base))
                     + ("" if base == expected else
                        f"   NOTE: this plan composites subject {index} onto {expected}, so H3 "
                        f"Multi-Face Composite will refuse on that unless the passes really were "
                        f"chained differently"))
        # #6: the shot account, now measured against what this pass actually produced. Derived
        # from the payload already wired - the tracker's transform for stage=crops, otherwise the
        # crops.json this pass checkpointed earlier - so no widget is added and no saved graph
        # changes meaning. A planned absence says the clip's pixels were kept on purpose; a lost
        # track says which shot to look at and which widgets would fix it.
        shot_lines, _account = shot_account_lines(plan, index, transform=transform, job_dir=job)
        lines.extend(shot_lines)
        if complete:
            coming = plan_store.next_subject(plan, job)
            nxt = coming["index"] if coming is not None else "none - this job is finished"
            lines.append(f"next subject: {nxt}")
        else:
            lines.append(f"next subject still up: {index}   H3 Subject Select's next takes "
                         f"it, and this subject again until its stitched stage is recorded")
        report = "\n".join(lines)
        print("[H3PassSave] " + report.replace("\n", "\n[H3PassSave] "))
        return (images, report, index, complete)



class H3PassLoad:
    """Read one subject's pass back out of the job folder.

    `stream = crops` returns the crops H3 was fed plus their crop positions, so a pass can be
    stitched without the tracker ever running again. `stream = stitched` returns the frames that
    pass composited back onto the clip - the result of that subject's work, ready to be the base
    the next subject's pass starts from. Nothing here re-detects, re-tracks or re-generates.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # WIDGET ORDER IS PERMANENT (ADR 0002): ComfyUI stores widget values by position.
        # Anything new goes at the end. tests/test_node_contract.py pins this list.
        return {
            "required": {
                "plan": (H3FACEPLAN, {
                    "tooltip": "The same subject plan the pass was saved with. It is checked "
                               "against the job folder's plan.json first, so a pass cannot be "
                               "read into a job planned from a different pairing."}),
                "subject": ("INT", {
                    "default": 0, "min": plan_store.CHAINED_TO_CLIP,
                    "max": MAX_SUBJECT_CHOICES - 1, "step": 1,
                    "tooltip": "Which subject's pass to read - the number H3 Pass Save wrote "
                               "it under. Wire H3 Subject Select's `previous_subject` here: it is "
                               "the subject before the one being refined, or -1 for the FIRST "
                               "subject (no previous pass) - then this node hands back whatever "
                               "is wired into `no_previous` (the target clip) instead of reading "
                               "a pass, so one reusable chain can serve every subject."}),
                "stream": (list(plan_store.PASS_STAGES), {
                    "default": "crops",
                    "tooltip": "crops (default): the crops that were fed to H3, plus their "
                               "crop positions - feed these to H3 Face Stitch to composite a "
                               "checkpointed pass without re-tracking.\n\n"
                               "stitched: the frames that pass composited back onto the clip. "
                               "Wire this into the next subject's base_images."}),
            },
            "optional": {
                # no_previous is appended at the END of the optional block: widget/socket order is
                # permanent (ADR 0002) and a new socket only ever goes last.
                "no_previous": ("IMAGE", {
                    "tooltip": "The base to hand back when `subject` is -1 - i.e. this is the "
                               "first subject in plan order and there is no previous stitched "
                               "pass to load. Wire the TARGET CLIP here so a single reusable "
                               "pass chain composites the first face straight onto the clip. "
                               "Ignored for any other subject."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "H3FACEXFORM", "STRING")
    RETURN_NAMES = ("images", "transform", "report")
    FUNCTION = "run"
    CATEGORY = CATEGORY


    def run(self, plan, subject, stream, no_previous=None):
        import torch

        plan = checked_plan(plan)
        job = job_folder_for_plan(plan, "H3 Pass Load")
        plan_store.check_pairing_inputs(plan_store.read_plan(job), plan, job, PAIRING_ADVICE)
        plan_store.check_manifest_subjects(plan, job)

        # subject == -1 (H3 Subject Select's previous_subject for the FIRST subject in plan
        # order): there is no previous stitched pass to load, so hand back the target clip from
        # `no_previous` as this pass's base. That is what lets ONE reusable pass chain cover any
        # subject: the first pass composites onto the clip, every later pass onto the previous
        # pass's stitched result.
        if int(subject) == plan_store.CHAINED_TO_CLIP:
            if no_previous is None:
                raise ValueError(
                    "H3 Pass Load got subject -1 (the first subject, no previous pass) but "
                    "`no_previous` is unwired. Wire the target clip into `no_previous` so this "
                    "first pass has a base to composite its refined face onto.")
            report = ("no previous subject: H3 Pass Load returned the target clip from "
                      "`no_previous` as base_images (first subject in plan order).")
            print("[H3PassLoad] " + report)
            return (no_previous, None, report)

        index = checked_subject(plan, subject)
        stage = plan_store.checked_stage(stream)
        folder = plan_store.stage_dir(job, index, stage)
        record = plan_store.manifest_stage_record(job, index, stage)
        if not isinstance(record, dict) or not record.get("frames"):
            raise FileNotFoundError(
                f"subject {index} has no {stage} pass recorded in "
                f"{plan_store.manifest_path(job)}, and {folder} holds nothing to read instead. "
                f"Run this subject's pass and save it with H3 Pass Save stage={stage}, or read "
                f"a subject the manifest does record.")
        count = int(record["frames"])
        files = plan_store.stage_files(job, index, stage, count)
        gaps = [path for path in files if not os.path.isfile(path)]
        if gaps:
            raise FileNotFoundError(
                f"subject {index}'s {stage} pass is incomplete: {len(gaps)} of {count} frame(s) "
                f"are missing from {folder}, starting with {gaps[0]}. The manifest records the "
                f"pass, so the images went after it was saved. Re-run this pass and H3 Pass "
                f"Save stage={stage} for subject {index}, or delete that manifest entry so the "
                f"subject counts as unfinished again.")

        # One PNG per frame, back in the order the pass was saved. Same converter the subject
        # references use, so a checkpoint reads exactly as it was written.
        images = torch.cat([_read_reference_png(path) for path in files])

        # A pass read from disk has to span the SAME frames the current clip does, or the chain
        # breaks downstream: H3 Face Stitch pastes each refined crop onto the base frame its
        # tracker recorded it from, so a shorter saved pass makes it index a base frame that is
        # not there (`index N out of bounds for dimension 0 with size N`). The plan's recorded
        # clip length is the authority. Catch it HERE, where the count is known, with the real
        # reason - rather than letting the frozen stitch node fail with an opaque IndexError.
        clip_frames = (plan.get("run") or {}).get("frames")
        try:
            clip_frames = int(clip_frames) if clip_frames else None
        except (TypeError, ValueError):
            clip_frames = None
        if clip_frames is not None and int(images.shape[0]) != clip_frames:
            raise ValueError(
                f"subject {index}'s {stage} pass has {int(images.shape[0])} frame(s), but this "
                f"job's clip is {clip_frames} frame(s). The pass chain composites each subject "
                f"onto the previous subject's stitched result and stitches each crop onto the "
                f"frame it came from, so a pass of a different length than the clip cannot be "
                f"chained with this run - it makes H3 Face Stitch fail with an IndexError. This "
                f"happens when the clip changed length (or a subject was rendered from a "
                f"differently-cut clip) and the older passes were left in the job folder. Press "
                f"'Reset run' on the H3 Multi-Face Plan node and re-queue from the FIRST subject "
                f"so every pass is rendered against the current clip, or press 'Process Plan' if "
                f"the clip itself changed and re-queue the whole job.")

        if stage == "crops":
            # A crops stream without its positions is useless, so this raises rather than
            # handing back frames nothing can be stitched from.
            positions = plan_store.read_positions(job, index)
        else:
            positions = (plan_store.read_positions(job, index)
                         if os.path.isfile(plan_store.positions_path(job, index)) else None)

        complete = 1 if plan_store.pass_is_complete(job, index) else 0
        lines = _checkpoint_report_lines(job, index, stage, record, complete)
        if positions is not None:
            lines.append(f"crop positions: {plan_store.positions_path(job, index)}   "
                         f"{len(positions.get('boxes') or [])} frame(s); canvas, src_size and "
                         f"segments come back as tuples, the payload the tracker built")
        else:
            lines.append("no crop positions for this subject, so the transform output is "
                         "empty: these frames are already composited, and their positions were "
                         "never saved")
        report = "\n".join(lines)
        print("[H3PassLoad] " + report.replace("\n", "\n[H3PassLoad] "))
        return (images, positions, report)


def _order_line(order) -> str:
    """Subjects as the composite names them: plan order, left to right at the match frame."""
    return " -> ".join(f"subject {index}" for index in order)


def _percent(value) -> str:
    return f"{float(value) * 100.0:.1f}%"


class H3MultiFaceComposite:
    """The finished clip: every subject's pass in plan order, plus the proof they are all there.

    Passes **chain**. Subject 0's `H3 Face Stitch` composited onto the target clip; subject 1's
    composited onto subject 0's saved stitched result (Pass Load with ``stream = stitched`` wired
    into the stitch node's ``base_images``); and so on down the plan. So the LAST subject's
    stitched frames already carry every earlier pass inside them, and assembling the job means
    picking those frames up - not blending anything again. There is deliberately no
    ``base_images`` and no ``transform`` on this node: a second feather laid over a feathered edge
    is how a freshly refined face goes soft.

    What this node adds is the gate. A job queued once per subject and interrupted between
    subjects would otherwise hand back a clip where one person is sharp and the other is still
    blurry, looking finished. So every subject's ``stitched`` stage has to be recorded in the
    manifest, and its frame count and frame size have to agree with the target clip wired in
    here. Anything missing and the node refuses, naming every subject that never landed.

    Audio is not this node's to handle: it carries image batches, so the finished clip's audio is
    the target clip's own and every pass's generated audio was discarded. The report says so,
    because a silent surprise at the end of a long job is its own kind of bug.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # WIDGET ORDER IS PERMANENT. ComfyUI stores widget values by position, so inserting or
        # reordering one silently reinterprets a saved workflow (ADR 0002). Anything new goes
        # at the END. tests/test_node_contract.py pins this list.
        return {
            "required": {
                "plan": (H3FACEPLAN, {
                    "tooltip": "The subject plan, from H3 Multi-Face Plan - the same one every "
                               "pass in this job ran on.\n\nIts subject order IS the composite "
                               "order: subjects are numbered left to right at the match frame, so "
                               "the plan says who was composited first with no widget to set "
                               "here.\n\nThe plan is also checked against the plan.json already in "
                               "the job folder. If the pairing inputs differ, the passes in that "
                               "folder were cut from a different pairing, and this refuses rather "
                               "than stitching two ideas of who is who into one clip."}),
                "images": ("IMAGE", {
                    "tooltip": "The target clip - the same IMAGE batch that fed the first pass's "
                               "H3 Face Stitch, from the same loader, unresized and untrimmed."
                               "\n\nNothing is composited onto it here. It is wired in so this "
                               "node can PROVE the chain is complete: every recorded stitched pass "
                               "has to hold this many frames at this size, or the frames in the "
                               "job folder are not this clip's."
                               "\n\nThis node carries image batches only and touches no audio at all, so each pass's generated audio was "
                               "discarded by never being wired anywhere - whatever audio the "
                                "finished video has comes from this clip, where you assemble it."}),
                "overlap_threshold": ("FLOAT", {
                    "default": plan_store.DEFAULT_OVERLAP_THRESHOLD, "min": 0.0, "max": 1.0,
                    "step": 0.01, "round": 0.001,
                    "tooltip": "How much of their combined area two subjects' crop regions may "
                               "share at the match frame before the report warns about them."
                               "\n\n0.25 (the default) warns past a quarter; 0 warns about any "
                               "overlap at all; 1 never warns. Measured on the crop regions H3 "
                               "Face Track + Crop would cut, not on the faces inside them."
                               "\n\nA warning, never an error: two people standing close together "
                               "are still two subjects and the clip is still the right answer - "
                               "but one pass's feathered edge lands on the other's refined "
                               "pixels, and a soft seam across a face is easier to live with once "
                               "you know where it is."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "INT")
    RETURN_NAMES = ("images", "report", "final_subject", "subject_count")
    FUNCTION = "run"
    # The report is the gate's verdict, and it is read off the node at the end of a job.
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def run(self, plan, images, overlap_threshold):
        import torch  # deferred: the schema has to build with only numpy installed

        plan = checked_plan(plan)
        job = job_folder_for_plan(plan, "H3 Multi-Face Composite")
        # The pass nodes' two refusals, before anything in the job folder is read.
        plan_store.check_pairing_inputs(plan_store.read_plan(job), plan, job, PAIRING_ADVICE)
        plan_store.check_manifest_subjects(plan, job)

        frames, width, height = checked_image_batch(images, "images")
        order = plan_store.plan_subject_indices(plan)
        if not order:
            raise ValueError(
                "this plan has no subjects, so there is nothing to composite: H3 Multi-Face Plan "
                f"found no faces, or every face was excluded. Sort the plan in {job} out first.")

        # #11 review: what each pass was composited onto is an INTENT the pass declares on H3 Pass
        # Save, not a fact this node can see - the wiring lives in the user's graph. A pass wired
        # onto the target clip instead of the previous pass's result silently drops every earlier
        # subject from the finished clip, so a declaration that contradicts the plan's own order
        # is refused here instead of being reported afterwards as though it had been honoured.
        chain = plan_store.chain_check(plan, job)
        if chain["wrong"]:
            raise ValueError(f"H3 Multi-Face Composite refuses: {plan_store.chain_error(chain)}")

        stage = "stitched"   # the stage that completes a pass, and the only one that is composited
        recorded = {}
        absent = []
        for index in order:
            entry = plan_store.manifest_stage_record(job, index, stage)
            if not isinstance(entry, dict) or not entry.get("frames"):
                absent.append((index, "no stitched pass is recorded in the manifest, so this "
                                      "subject's refined face is nowhere in the job folder"))
                continue
            files = plan_store.stage_files(job, index, stage, int(entry["frames"]))
            gaps = [path for path in files if not os.path.isfile(path)]
            if gaps:
                absent.append((index, f"its stitched pass is recorded but {len(gaps)} of "
                                      f"{len(files)} frame(s) are not in "
                                      f"{plan_store.stage_dir(job, index, stage)}, starting with "
                                      f"{os.path.basename(gaps[0])}"))
                continue
            recorded[index] = {"frames": len(files), "files": files,
                               "width": entry.get("width"), "height": entry.get("height")}

        if absent:
            coming = plan_store.next_subject(plan, job)
            nxt = (coming["index"] if coming is not None
                   else "none - the manifest says every subject in this plan is done")
            lines = [f"{len(absent)} of {len(order)} pass(es) are not in {job}, in plan order "
                     f"{_order_line(order)}:"]
            lines.extend(f"  - subject {index}: {why}" for index, why in absent)
            raise ValueError(
                "\n".join(lines)
                + f"\nnext subject: {nxt}   H3 Subject Select's next goes to that subject: run "
                  f"its pass through to H3 Face Stitch, then save it with H3 Pass Save "
                  f"stage={stage}."
                  + "\nThis composite refuses: a half-composited clip is worse than an error. It "
                    "would look finished while the faces of the subjects named above kept the "
                    "pixels the target clip arrived with.")

        clashes = []
        for index in order:
            entry = recorded[index]
            if entry["frames"] != frames:
                clashes.append(f"  - subject {index}: its stitched pass holds "
                               f"{entry['frames']} frame(s), the target clip wired here holds "
                               f"{frames} frame(s)")
            if entry["width"] is not None and (int(entry["width"]) != width
                                               or int(entry["height"]) != height):
                clashes.append(f"  - subject {index}: its stitched pass is "
                               f"{int(entry['width'])}x{int(entry['height'])}px, while the target "
                               f"clip wired here is {width}x{height}px")
        if clashes:
            raise ValueError(
                f"the passes recorded in {job} are not this clip:\n" + "\n".join(clashes)
                + "\nEvery pass composites onto the target clip, so its frame count and its frame "
                  "size ARE the clip's. Wire the clip those passes were cut from, or re-run the "
                  "subjects named above against this clip and save them again.")

        # The assembly, in one line: the last subject's frames, because every earlier pass is
        # already inside them. Frames come off disk in the order the manifest counted, through the
        # same PNG reader the subject references and Pass Load use, so the same job folder read
        # twice gives back identical tensors.
        final = order[-1]
        finished = torch.cat([_read_reference_png(path) for path in recorded[final]["files"]])

        warnings = plan_store.overlapping_subject_pairs(plan, overlap_threshold)
        frame = (plan or {}).get("match_frame")
        lines = [f"job folder: {job}",
                 f"order: {_order_line(order)}   plan order = left to right at match frame {frame}"]
        # The chain in the voice of what was actually recorded: what each pass DECLARED it was
        # composited onto on H3 Pass Save, or the gap where a pass declared nothing. This node
        # cannot see the graph's wiring, so it never says the passes were chained on its own.
        lines.extend(plan_store.chain_lines(chain))
        for index in order:
            entry = recorded[index]
            size = (f"{int(entry['width'])}x{int(entry['height'])}px"
                    if entry["width"] is not None else "size not recorded in the manifest")
            lines.append(f"subject {index}   {stage} {entry['frames']} frame(s) at {size}   "
                         f"{plan_store.stage_dir(job, index, stage)}")
        if len(order) > 1:
            carried = ", ".join(f"subject {i}" for i in order[:-1])
            lines.append(f"final result: subject {final}'s stitched frames   this node hands back "
                         f"those {frames} frame(s) and blends nothing: they carry {carried} in "
                         f"them only if each of those passes was composited onto the one before "
                         f"it, which the chain line above says - or says nobody declared")
        else:
            lines.append(f"final result: subject {final}'s stitched frames   this job has one "
                         f"pass, so those {frames} frame(s) are it on the target clip")
        for pair in warnings:
            lines.append(
                f"overlap warning: subject {pair['a']} and subject {pair['b']} crop regions share "
                f"{_percent(pair['overlap'])} of their combined area at match frame {frame}, past "
                f"the {float(overlap_threshold):g} set here - if they are two people, one pass's "
                f"feathered edge lands on the other's refined pixels")
        if not warnings:
            lines.append(f"no overlapping subject crop regions at match frame {frame} "
                         f"(threshold {float(overlap_threshold):g})")
        lines.append("audio: this node carries images only and touches no audio at all. Whatever "
                     "audio the finished clip has comes from the target clip, at the point where "
                     "you assemble the video - and each pass's generated audio is discarded by "
                     "never being wired anywhere.")

        report = "\n".join(lines)
        print("[H3MultiFaceComposite] " + report.replace("\n", "\n[H3MultiFaceComposite] "))
        return (finished, report, final, len(order))


NODE_CLASS_MAPPINGS = {"H3MultiFacePlan": H3MultiFacePlan,
                       "H3SubjectSelect": H3SubjectSelect,
                       "H3PassSave": H3PassSave,
                       "H3PassLoad": H3PassLoad,
                       "H3MultiFaceComposite": H3MultiFaceComposite}
NODE_DISPLAY_NAME_MAPPINGS = {"H3MultiFacePlan": "H3 Multi-Face Plan",
                              "H3SubjectSelect": "H3 Subject Select",
                              "H3PassSave": "H3 Pass Save",
                              "H3PassLoad": "H3 Pass Load",
                              "H3MultiFaceComposite": "H3 Multi-Face Composite"}


