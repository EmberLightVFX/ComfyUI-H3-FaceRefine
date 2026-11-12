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


CATEGORY = "MiniMax H3/Face Refine"

# The pack's own tracker defaults, quoted so the report can name the cost of a job before
# any pass has picked a size. `H3FaceTrackCrop` ships crop_factor 2.5 and a 768x768 canvas.
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
    for face in plan["excluded_faces"]:
        clip_colour[int(face)] = _EXCLUDED_COLOUR

    excluded = {int(i) for i in plan["excluded_faces"]}
    for plate, boxes, colours, scale in ((clip_plate, clip_boxes, clip_colour, 1.0),
                                         (still_plate, still_boxes, still_colour,
                                          float(plan["scale"]))):
        for i, box in enumerate(boxes):
            x0, y0, x1, y1 = (float(v) * scale for v in box[:4])
            colour = colours.get(i, _UNSEEDED_COLOUR)
            _draw_rect(plate, x0, y0, x1, y1, colour, thick * (2 if i in colours else 1))
            if i in excluded:
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
    tw, th = plan["inputs"]["target_size"]
    sw, sh = plan["inputs"]["still_size"]
    scale = float(plan["scale"])
    lines = [
        _line("job", plan["run"]["job_name"]),
        _line("target clip", f"{tw}x{th}   {frames} frame(s)"
                             f"   match frame {plan['match_frame']}"),
        _line("reference still", f"{sw}x{sh}   still -> clip scale {scale:.3f}"
                                 f"   (1 clip px = {1.0 / scale:.3f} still px)"),
        _line("cost", f"{plan['subject_count']} subject(s) x {frames} frame(s)"
                      f" @ {canvas_w}x{canvas_h}"
                      f"   = {plan['subject_count'] * frames} frame(s) to generate"),
        _line("", f"(canvas size is H3 Face Track + Crop's;"
                  f" crops are {DEFAULT_CROP_FACTOR}x face height)"),
        _line("pairing", f"{plan['inputs']['subject_source']}"
                         f"   tolerance {plan['inputs']['match_tolerance']:.3f}"
                         f" = {plan['tolerance_px']:.0f}px of the {tw}x{th} diagonal"
                         f"   {len(plan['pairs'])} of {len(plan['target_boxes'])}"
                         f" clip face(s) paired"),
    ]
    if plan["excluded_faces"]:
        lines.append(_line(
            "excluded",
            f"clip face(s) {', '.join(str(i) for i in plan['excluded_faces'])}"
            f" - switched off, so they keep their original pixels"))
    else:
        lines.append(_line("excluded", "none"))
    if plan["inputs"]["max_subjects"]:
        lines.append(_line("max_subjects", f"{plan['inputs']['max_subjects']}"
                                           f" (recorded in the plan; the passes act"
                                           f" on it)"))

    for subject in plan["subjects"]:
        where = (f"clip face {subject['clip_face']}" if subject["clip_face"] is not None
                 else "not seen in the clip on this frame")
        if subject["has_reference"]:
            crop = subject["ref_crop"]
            ref = (f"reference from still face {subject['still_face']},"
                   f" {crop[2]}x{crop[3]}px at the still's native resolution"
                   f" -> {subject['ref_file']}")
        else:
            ref = ("NO subject reference - refine it without one, or fix the pairing"
                   " and re-run this node")
        lines.append(_line(f"subject {subject['index']}", where + "   " + ref))

    lines.append(_line("written", job_dir))
    lines.append(_line("", f"{plan_store.PLAN_FILENAME} + {refs} subject reference"
                           f" crop(s)"))
    if notes:
        lines.append(_line("notes", ""))
        lines.extend(_line("", f"- {note}") for note in notes)
    return "\n".join(lines)


class H3MultiFacePlan:
    """Decide who the subjects of this job are.

    Detects the target clip at the match frame and the reference still, pairs the two
    positionally, and writes the subject plan plus every subject reference crop into the job
    folder - so no later pass needs the still loaded again.
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
                    "tooltip": "Clip face numbers to switch off, comma-separated, exactly as "
                               "numbered on this node's preview: 0, 3. An excluded face never "
                               "becomes a subject and keeps its original pixels. A text "
                               "widget, so it survives a reload with no front end."}),
                "max_subjects": ("INT", {
                    "default": 0, "min": 0, "max": 64, "step": 1,
                    "tooltip": "0 = no cap. The cap is recorded in the plan for the passes to "
                               "act on."}),
                "job_name": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "The job folder under ComfyUI's output folder, holding the plan "
                               "and every subject reference crop. Left blank it is named "
                               "after the clip. Keep the same name to resume a job."}),
            },
        }



    RETURN_TYPES = (H3FACEPLAN, "IMAGE", "STRING", "INT")
    RETURN_NAMES = ("plan", "preview", "report", "subject_count")
    FUNCTION = "run"
    # The preview and the report exist to be read BEFORE anything is queued, so this is an
    # output node: no preview node has to be wired on to check a pairing. The plan output
    # still feeds downstream.
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def run(self, images, reference_still, detector, confidence, match_frame,
            subject_source, match_tolerance, excluded_faces, max_subjects, job_name):
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

        clip_boxes, clip_confs = _detect(images[frame])
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
            plan_store.parse_excluded_faces(excluded_faces), int(max_subjects))
        if not plan["subjects"]:
            if plan["excluded_faces"]:
                raise ValueError(
                    f"nothing to plan: every face found on frame {frame} is named in "
                    f"excluded_faces ({', '.join(str(i) for i in plan['excluded_faces'])}). "
                    f"Clear the numbers that are not bystanders.")
            raise ValueError(
                f"nothing to plan: subject_source={subject_source} seeds its subjects from a "
                f"plate holding no faces on frame {frame}. Use union, or lower confidence.")



        name = _derived_job_name(images, job_name)
        job = plan_store.job_dir_for(folder_paths.get_output_directory(), name)
        os.makedirs(job, exist_ok=True)

        # The crops are cut and written BEFORE the plan is written, so the plan records the
        # file each subject's reference lives in. That is what lets a later pass resume a job
        # with neither plate loaded again.
        refs = 0
        for subject in plan["subjects"]:
            if subject["ref_crop"]:
                file_name = f"subject_{int(subject['index'])}.png"
                _write_reference_png(os.path.join(job, file_name), still,
                                     subject["ref_crop"])
                subject["ref_file"] = file_name
                refs += 1
        plan["run"] = {"job_name": name, "frames": frames, "detector": str(detector),
                       "confidence": float(confidence),
                       "clip_source": (str(getattr(images, "h3_video_name", "") or "")
                                       or None)}
        written = plan_store.write_plan(job, plan)

        unreference = [int(s["index"]) for s in plan["subjects"] if not s["has_reference"]]
        if unreference:
            notes.append(
                f"subject(s) {', '.join(str(i) for i in unreference)} have no face in the "
                f"reference still, so no subject reference: a pass for them either runs "
                f"without a reference or they keep their original pixels")
        unseeded = [int(j) for j in plan["unpaired_still"]
                    if not any(s["still_face"] == j for s in plan["subjects"])]
        if unseeded:
            notes.append(
                f"reference still face(s) {', '.join(str(j) for j in unseeded)} are not "
                f"subjects while subject_source={subject_source}")

        preview = _pairing_preview(images[frame], still, clip_boxes, still_boxes, plan)
        report = _format_report(plan, frames, job, refs, notes + [f"plan: {written}"])
        return (plan, preview, report, int(plan["subject_count"]))


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
    ref = (f"still face {still_face}" if still_face is not None
           else "no subject reference")
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


def _left_to_right_order(boxes) -> list[int]:
    """Face indices counted left to right, as the pairing preview and Pick faces number them.

    Mirrors the released tracker's own ``left_most`` ranking (nodes.py:866): centre x first,
    ties settled on the box corner, so two faces at the same height never swap numbers. The
    tracker scores ``left_most`` as ``-centre_x`` and sorts on the negated score, which lands
    on ascending centre x - the same order, spelled out here rather than doubled-negated.
    """
    return sorted(
        range(len(boxes)),
        key=lambda k: ((float(boxes[k][0]) + float(boxes[k][2])) / 2.0,
                       float(boxes[k][0]), float(boxes[k][1])))


def _iou(a, b) -> float:
    """Overlap of two boxes - the only honest way to say "the face in the plan is that face
    in the payload", since the plan's detection pass and the picker's may have returned them
    in a different order."""
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    inter = (max(0.0, min(ax2, bx2) - max(ax1, bx1))
             * max(0.0, min(ay2, by2) - max(ay1, by1)))
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _planned_face_index(plan_box, on_frame, clip_face, match_frame) -> int:
    """Which of the payload's boxes on the match frame is the face the plan measured."""
    best, best_iou = None, 0.0
    for i, box in enumerate(on_frame):
        score = _iou(plan_box, box)
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
            return frame, _left_to_right_order(all_boxes[frame])[face_number]
    return None


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

    match_frame = int((plan or {}).get(
        "match_frame", ((plan or {}).get("inputs") or {}).get("match_frame", -1)))
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
    face_number = _left_to_right_order(on_match).index(face_index)

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
    # Informational, for whoever reads a payload while debugging a pass: the tracker itself
    # reads boxes, confs, segments, picks, detector and confidence, and nothing else.
    narrowed["select"] = "subject_plan"
    narrowed["select_index"] = face_number
    narrowed["rank_by"] = "left_most"
    narrowed["subject_index"] = int(subject["index"])
    narrowed["plan_match_frame"] = match_frame
    return narrowed


def absent_shots(narrowed) -> list:
    """The shot numbers this subject is not in, 1-based, as the report names them."""
    return [k + 1 for k, pick in enumerate(narrowed.get("picks") or []) if pick.get("absent")]


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
    match_frame = narrowed.get("plan_match_frame")
    paired = subject.get("still_face") is not None
    on_match = 0
    try:
        on_match = len((narrowed.get("boxes") or [])[int(match_frame)])
    except (IndexError, TypeError, ValueError):
        on_match = 0

    pairing = (f"still face {subject['still_face']} (paired)" if paired
               else "no subject reference (unpaired)")
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


def load_subject_reference(job_dir, subject):
    """``(IMAGE or None, 1/0, what the report says about it)``.

    A subject with no subject reference is a normal state of a plan, not an error: that pass
    runs with nothing wired into H3's reference slots. ``None`` rather than a black
    placeholder, so a graph that insists on a reference fails loudly instead of generating
    from mush dressed up as detail.
    """
    name = (subject or {}).get("ref_file")
    index = (subject or {}).get("index")
    if not name:
        return None, 0, (f"subject {index} has no subject reference: no face in the "
                         f"reference still paired with this clip face, so this pass runs "
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
    return (reference, 1,
            f"{name}   {int(reference.shape[2])}x{int(reference.shape[1])}px, read from the "
            f"job folder - the reference still is not loaded again")


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
                "subject": (["next"] + [str(i) for i in range(MAX_SUBJECT_CHOICES)], {
                    "default": NEXT_SUBJECT,
                    "tooltip": "Which subject this pass refines.\n\n"
                               "next (default): the first subject in plan order with no "
                               "completed pass recorded in the job folder. Queue the graph "
                               "once per subject and change nothing in between - the "
                               "bookkeeping lives in the job folder, and H3 Pass Save is what "
                               "writes that record.\n\n"
                               "a number: that subject, by the number H3 Multi-Face Plan "
                               "prints in its report and on the pairing preview. Use it to "
                               "pick a specific subject, or to re-run one.\n\n"
                               "The list cannot show the plan's own names: ComfyUI asks for "
                               "the options before anything is wired, so the plan is not "
                               "readable yet. The report names every subject in the plan, and "
                               "an answer that is not one of them lists them."}),
            },
        }

    RETURN_TYPES = (H3FACEPICK, "IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("face_pick", "subject_reference", "reference_exists", "subject_index",
                    "report")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, plan, face_pick, subject=NEXT_SUBJECT):
        if not isinstance(plan, dict) or "subjects" not in plan:
            raise ValueError(
                "plan is not a subject plan - wire it from H3 Multi-Face Plan's plan output, "
                "or read it out of the job folder's plan.json."
            )

        job = job_dir_for_plan(plan)
        chosen = resolve_subject(plan, subject, job)
        narrowed = narrow_pick_to_subject(plan, face_pick, chosen)
        reference, exists, reference_note = load_subject_reference(job, chosen)

        cursor = ""
        if str(NEXT_SUBJECT if subject is None else subject).strip() == NEXT_SUBJECT and job:
            done = plan_store.completed_subjects(plan, job)
            cursor = (f"subject {chosen['index']} of {len(plan['subjects'])} is up"
                      + (f"   already recorded complete: subject(s) "
                         f"{', '.join(str(i) for i in done)}" if done
                         else "   no completed pass recorded in this job yet"))

        report = subject_report(plan, chosen, narrowed, job_dir=job,
                               reference_note=reference_note, cursor_note=cursor)
        print("[H3SubjectSelect] " + report.replace("\n", "\n[H3SubjectSelect] "))
        return (narrowed, reference, int(exists), int(chosen["index"]), report)


NODE_CLASS_MAPPINGS = {"H3MultiFacePlan": H3MultiFacePlan,
                       "H3SubjectSelect": H3SubjectSelect}
NODE_DISPLAY_NAME_MAPPINGS = {"H3MultiFacePlan": "H3 Multi-Face Plan",
                              "H3SubjectSelect": "H3 Subject Select"}

