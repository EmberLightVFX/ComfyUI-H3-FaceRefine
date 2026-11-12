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


NODE_CLASS_MAPPINGS = {"H3MultiFacePlan": H3MultiFacePlan}
NODE_DISPLAY_NAME_MAPPINGS = {"H3MultiFacePlan": "H3 Multi-Face Plan"}

