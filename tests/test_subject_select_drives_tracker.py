"""The seam that earns ticket #3: Subject Select's payload drives the RELEASED tracker.

ADR 0005 promises that a multi-face pass is a driver over the existing pipeline rather than a
fork of it. That is only worth believing if the untouched `H3FaceTrackCrop.run` accepts what
Subject Select emits and crops the face the plan named. So this test builds a synthetic clip
and a synthetic `H3FACEPICK`, narrows it through the real plan and the real narrowing code,
and calls the released node with its own released widget names.

The plan knows one frame's boxes; the payload carries every frame's, which is what lets
`nodes.py:1312` skip detection outright. `_load_detector` is patched to explode, so this test
fails loudly if any detection - or any identity/body model - is reached at all: no weights,
no download, no GPU.

Skips itself when `comfy` or `torch` is missing, which is CI's setup (numpy only). Locally it
runs on the ComfyUI interpreter with ComfyUI's root importable; `H3_COMFY_ROOT` overrides
where to find it.
"""

import importlib.util
import json
import os
import pathlib
import sys

import numpy as np
import pytest

import multi_face_nodes as m
import plan_store as ps

PACK_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_COMFY_ROOT = pathlib.Path(r"C:\Users\emberlight\ComfyUI Cuda12.8\ComfyUI")


def _import_or_skip(name, hint):
    try:
        return importlib.import_module(name)
    except Exception:
        pytest.skip(f"{name} is not importable - {hint}", allow_module_level=True)


_comfy_root = pathlib.Path(os.environ.get("H3_COMFY_ROOT") or str(DEFAULT_COMFY_ROOT))
if _comfy_root.is_dir() and str(_comfy_root) not in sys.path:
    # nodes.py imports `comfy.nested_tensor` and `folder_paths`; without ComfyUI's own root
    # on the path there is no released node to drive.
    sys.path.insert(0, str(_comfy_root))

torch = _import_or_skip("torch", "run this on the ComfyUI interpreter")
_import_or_skip("comfy.nested_tensor",
                f"set H3_COMFY_ROOT to a ComfyUI checkout (tried {_comfy_root})")
_import_or_skip("folder_paths",
                f"set H3_COMFY_ROOT to a ComfyUI checkout (tried {_comfy_root})")


def _load_released_nodes():
    """`nodes.py` exactly as released, loaded by path so the pack under test is the one in
    this worktree rather than the copy installed in ComfyUI's custom_nodes."""
    spec = importlib.util.spec_from_file_location("h3_released_nodes",
                                                 str(PACK_ROOT / "nodes.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


nodes = _load_released_nodes()


# --- a synthetic clip: two people, one cut between them ---------------------------------

W, H = 320, 256
FRAMES = 12
SEGMENTS = [(0, 6), (6, 12)]
MATCH_FRAME = 3
MIDDLE_X = W / 2.0

LEFT = [65.0, 100.0, 95.0, 140.0]      # cx 80,  cy 120, face height 40
RIGHT = [225.0, 100.0, 255.0, 140.0]   # cx 240, cy 120

CROP_FACTOR = 2.5
CANVAS = 128


def _boxes_at(frame, single_after_the_cut=False):
    """Shot 1 reports the faces left-then-right, shot 2 right-then-left: the cut renumbers
    them, so the plan's index cannot be carried across verbatim."""
    if single_after_the_cut and frame >= SEGMENTS[1][0]:
        return [list(LEFT)]
    return ([list(LEFT), list(RIGHT)] if frame < SEGMENTS[1][0]
            else [list(RIGHT), list(LEFT)])


def _payload(single_after_the_cut=False):
    boxes = [_boxes_at(f, single_after_the_cut) for f in range(FRAMES)]
    return {
        "version": 1,
        "boxes": boxes,
        "confs": [[0.9, 0.8] if len(b) == 2 else [0.9] for b in boxes],
        "segments": [[int(a), int(b)] for a, b in SEGMENTS],
        "picks": [{"segment": [int(a), int(b)], "frame": int(a), "box": 0, "index": 0,
                   "absent": False} for a, b in SEGMENTS],
        "frames": FRAMES,
        "src_size": (W, H),
        "detector": "face_yolov8m.pt",
        "confidence": 0.35,
        "select": "manual",
        "select_index": 0,
        "rank_by": "left_most",
    }


def _plan(payload):
    """The real plan builder, on a still of the same framing, so both faces pair and both
    become subjects - leftmost first, exactly as the subject plan orders them."""
    at = [list(b) for b in payload["boxes"][MATCH_FRAME]]
    return ps.build_subject_plan(at, [0.9, 0.8], at, [0.95, 0.9],
                                 (W, H), (W, H), MATCH_FRAME, "union", 0.05, (), 0)


def _clip():
    """Two colour-coded halves, so a crop says which half of the frame it was taken from.

    Left half red, right half blue: a crop's mean channels then name the person it is of,
    which is the outcome an editor would check by eye.
    """
    frames = torch.zeros((FRAMES, H, W, 3))
    frames[..., :W // 2, :] = torch.tensor([1.0, 0.15, 0.15])
    frames[..., W // 2:, :] = torch.tensor([0.15, 0.15, 1.0])
    return frames


def _no_detection(monkeypatch):
    """Any detector load fails the test: with a wired pick, `nodes.py:1312` skips the whole
    detection pass, and that is the acceptance criterion."""
    def _boom(*args, **kwargs):
        raise AssertionError("the tracker detected: the narrowed payload was not enough")

    monkeypatch.setattr(nodes, "_load_detector", _boom)
    monkeypatch.setattr(nodes, "_make_embedder", _boom)


def _track(monkeypatch, narrowed):
    _no_detection(monkeypatch)
    images = _clip()
    return nodes.H3FaceTrackCrop().run(
        images, "face_yolov8m.pt", 0.35, CROP_FACTOR, CANVAS, CANVAS, "manual",
        5, 5, "gaussian", "per_frame",
        select="largest_face", fallback_detector="none", fallback_head_frac=0.5,
        identity_reference=None, identity_threshold=0.28, identity_track=False,
        select_index=0, identity_model="insightface", identity_clip_vision=None,
        cut_detection="none", cut_threshold=3.0, absent_shots="off", X=0, Y=0,
        frame_index=0, face_pick=narrowed)


def _centres(transform):
    return [float(b[0]) + float(b[2]) / 2.0 for b in transform["boxes"]]


# --- the tests --------------------------------------------------------------------------

def test_the_released_tracker_crops_the_subject_the_plan_named(monkeypatch):
    """No detection, no identity model, no weights: the payload decides who is tracked, and
    the crops come back on the left-hand person's face."""
    payload = _payload()
    plan = _plan(payload)
    narrowed = m.narrow_pick_to_subject(plan, payload, plan["subjects"][0])

    crops, transform, preview, report, cw, ch, K = _track(monkeypatch, narrowed)

    assert K == FRAMES
    assert tuple(crops.shape) == (FRAMES, CANVAS, CANVAS, 3)
    assert (cw, ch) == (CANVAS, CANVAS)
    assert transform["source"] == list(range(FRAMES))

    # The released report says it out loud: the subject came from the payload, and the boxes
    # came with it - no second detection pass.
    assert "subject: from face_pick, one per shot" in report
    assert "pick: boxes from face_pick" in report
    assert "(0 resolved by face identity)" in report

    # Where the plan says: a crop side on each of the left person's face, 100px tall at
    # crop_factor 2.5, centred on cx 80.
    for centre in _centres(transform):
        assert abs(centre - 80.0) < 4.0
    assert all(abs(float(b[3]) - CROP_FACTOR * (LEFT[3] - LEFT[1])) < 4.0
               for b in transform["boxes"])

    # And what the crop is actually a picture OF: the red half of the frame.
    means = crops.mean(dim=(1, 2))
    assert bool((means[:, 0] > means[:, 2] + 0.3).all())


def test_the_other_subject_gets_the_other_face_from_the_same_payload(monkeypatch):
    """The multi-person case, driven end to end: two subjects, one clip, one payload, and the
    untouched tracker produces two different crops - one per person.

    Shot 2 reports its faces the other way round, so a pass that carried the plan's box index
    across the cut would crop the wrong half of the frame and this would fail.
    """
    payload = _payload()
    plan = _plan(payload)
    assert [s["clip_face"] for s in plan["subjects"]] == [0, 1]

    left_crops, left_xform = _track(monkeypatch, m.narrow_pick_to_subject(
        plan, payload, plan["subjects"][0]))[:2]
    right_crops, right_xform = _track(monkeypatch, m.narrow_pick_to_subject(
        plan, payload, plan["subjects"][1]))[:2]

    for centre in _centres(right_xform):
        assert abs(centre - 240.0) < 4.0
    left_means, right_means = left_crops.mean(dim=(1, 2)), right_crops.mean(dim=(1, 2))
    assert bool((left_means[:, 0] > left_means[:, 2]).all())
    assert bool((right_means[:, 2] > right_means[:, 0]).all())
    assert not torch.allclose(left_crops, right_crops)


def test_a_shot_the_subject_is_not_in_keeps_its_original_pixels(monkeypatch):
    """Absence travels in the payload shape, so the released node handles it: those frames are
    flagged, centred rather than tracked, and the stitch leaves them alone."""
    payload = _payload(single_after_the_cut=True)
    plan = _plan(payload)
    right = plan["subjects"][1]
    narrowed = m.narrow_pick_to_subject(plan, payload, right)

    crops, transform, preview, report, cw, ch, K = _track(monkeypatch, narrowed)

    assert transform["absent"] == [False] * 6 + [True] * 6
    live = _centres(transform)[:6]
    for centre in live:
        assert abs(centre - 240.0) < 4.0
    # The absent stretch is centred padding, not a crop of the person still on screen.
    for centre in _centres(transform)[6:]:
        assert abs(centre - MIDDLE_X) < 2.0
    live_means = crops[:6].mean(dim=(1, 2))
    assert bool((live_means[:, 2] > live_means[:, 0] + 0.3).all())


# --- the node itself: job folder in, narrowed payload out --------------------------------

def _job_with_references(tmp_path, monkeypatch, plan, payload, files):
    """Write the job folder the way ticket #2's node leaves it, then read the plan back off
    disk - which is what a pass in a later queue run actually gets handed."""
    from PIL import Image

    job = tmp_path / "output" / "group shot"
    os.makedirs(job, exist_ok=True)
    plan = dict(plan)
    plan["run"] = {"job_name": "group shot", "frames": FRAMES}
    for subject in plan["subjects"]:
        name = files.get(subject["index"])
        if name:
            # Distinctive per subject, so a crop of the wrong one cannot pass for the right one.
            tone = 40 + 60 * int(subject["index"])
            arr = np.zeros((64, 96, 3), np.uint8)
            arr[..., 0], arr[..., 1], arr[..., 2] = tone, 200 - tone, 128
            Image.fromarray(arr).save(os.path.join(str(job), name))
            subject["ref_file"] = name
    with open(os.path.join(str(job), "plan.json"), "w", encoding="utf-8") as fh:
        json.dump(plan, fh)
    # The node asks ComfyUI's own output folder, which this test has no business touching.
    monkeypatch.setattr(m, "_output_root", lambda: str(tmp_path / "output"))
    return ps.read_plan(job), job
    os.makedirs(job, exist_ok=True)
    plan = dict(plan)
    plan["run"] = {"job_name": "group shot", "frames": FRAMES}
    for subject in plan["subjects"]:
        name = files.get(subject["index"])
        if name:
            # Distinctive per subject, so a crop of the wrong one cannot pass for the right one.
            tone = 40 + 60 * int(subject["index"])
            arr = np.zeros((64, 96, 3), np.uint8)
            arr[..., 0], arr[..., 1], arr[..., 2] = tone, 200 - tone, 128
            Image.fromarray(arr).save(os.path.join(str(job), name))
            subject["ref_file"] = name
    with open(os.path.join(str(job), "plan.json"), "w", encoding="utf-8") as fh:
        json.dump(plan, fh)
    monkeypatch.setattr(folder_paths, "get_output_directory",
                        lambda: str(tmp_path / "output"))
    return ps.read_plan(job), job


def test_the_node_hands_over_the_narrowed_pick_and_its_subject_reference(tmp_path,
                                                                       monkeypatch):
    """`next` is the default, so a queued pass needs no widget change; the subject reference
    comes out of the job folder, with no reference still in sight."""
    payload = _payload()
    plan, job = _job_with_references(tmp_path, monkeypatch, _plan(payload), payload,
                                     {0: "subject_0.png", 1: "subject_1.png"})

    pick, reference, exists, index, report = m.H3SubjectSelect().run(plan, payload)

    assert index == 0 and exists == 1
    assert len(pick["picks"]) == len(SEGMENTS)
    # ADR 0005: the handed-over payload is the picker's key set, verbatim. Whose pass it is
    # rides on the `subject_index` OUTPUT above, never inside the payload.
    assert set(pick) == set(payload)
    assert "subject_index" not in pick and "plan_match_frame" not in pick
    assert tuple(reference.shape) == (1, 64, 96, 3)
    assert abs(float(reference[0, 0, 0, 0]) - 40.0 / 255.0) < 0.01
    assert "subject 0" in report
    assert str(job) in report


def test_next_moves_on_when_a_pass_is_recorded_complete(tmp_path, monkeypatch):
    """The bookkeeping the spec asks for: re-queueing takes the next subject, and a subject
    with no subject reference says so rather than inventing one."""
    payload = _payload()
    plan, job = _job_with_references(tmp_path, monkeypatch, _plan(payload), payload,
                                     {0: "subject_0.png"})
    with open(os.path.join(str(job), ps.MANIFEST_FILENAME), "w", encoding="utf-8") as fh:
        json.dump({"version": ps.MANIFEST_VERSION,
                   "passes": {"0": {"complete": True}}}, fh)

    pick, reference, exists, index, report = m.H3SubjectSelect().run(plan, payload)

    assert index == 1
    assert reference is None and exists == 0
    assert "no subject reference" in report
    assert "shot" in report
