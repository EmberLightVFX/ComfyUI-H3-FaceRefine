"""An absent shot keeps the target clip's original pixels, through the RELEASED stitch (#6).

#6's acceptance says the frames of an absent shot keep the clip's own pixels. That behaviour is
inherited, not new: the pick entry `{"absent": true, "segment": [start, end]}` (nodes.py:1334)
makes the tracker skip those frames and weight them at zero, and `H3FaceStitch` multiplies its
paste mask by that weight. Inherited has to be proven rather than assumed, so this test runs the
untouched released nodes end to end - `H3FaceTrackCrop.run` on a synthetic clip and payload, then
`H3FaceStitch.run` with crops that are unmistakably refined - and compares what came back, frame
by frame, against the original clip.

No weights, no GPU, no detector: the pick payload carries every box, so `_load_detector` and
`_make_embedder` are patched to explode. If either is reached the test fails loudly instead of
quietly loading a model. Nothing here touches a real ComfyUI folder: the released nodes are
tensor in, tensor out.

Skips itself when `comfy` or `torch` is missing, which is CI's setup (numpy only). Locally it runs
on the ComfyUI interpreter with ComfyUI's root importable; `H3_COMFY_ROOT` overrides where to
find it. Same pattern as `tests/test_subject_select_drives_tracker.py`.
"""

import importlib.util
import os
import pathlib
import sys

import numpy as np
import pytest

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
    # nodes.py imports `comfy.nested_tensor` and `folder_paths`; without ComfyUI's own root on
    # the path there is no released node to drive.
    sys.path.insert(0, str(_comfy_root))

torch = _import_or_skip("torch", "run this on the ComfyUI interpreter")
_import_or_skip("comfy.nested_tensor",
                f"set H3_COMFY_ROOT to a ComfyUI checkout (tried {_comfy_root})")
_import_or_skip("folder_paths",
                f"set H3_COMFY_ROOT to a ComfyUI checkout (tried {_comfy_root})")


def _load_released_nodes():
    """`nodes.py` exactly as released, loaded by path - the copy in this worktree, not the one
    installed in ComfyUI's custom_nodes."""
    spec = importlib.util.spec_from_file_location("h3_released_nodes_for_absence",
                                                  str(PACK_ROOT / "nodes.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


nodes = _load_released_nodes()

# --- a synthetic clip: one subject in shot 1, a different person in shot 2 ------------------

W, H = 320, 256
FRAMES = 12
SEGMENTS = [(0, 6), (6, 12)]
ABSENT_FROM = 6

CROP_FACTOR = 2.5
CANVAS = 128

SUBJECT_BOX = [225.0, 100.0, 255.0, 140.0]   # the subject, right half of shot 1
OTHER_BOX = [65.0, 100.0, 95.0, 140.0]       # someone else, entirely, in shot 2


def _payload():
    """One face per frame, and a pick that names the subject in shot 1 only.

    The absent entry is the pack's existing absence mechanism verbatim - this test does not
    invent a way to say "not in this shot", it uses the one the released tracker reads.
    """
    boxes = [[list(SUBJECT_BOX)] if f < ABSENT_FROM else [list(OTHER_BOX)]
             for f in range(FRAMES)]
    return {
        "version": 1,
        "boxes": boxes,
        "confs": [[0.9] for _ in boxes],
        "segments": [[int(a), int(b)] for a, b in SEGMENTS],
        "picks": [
            {"segment": [0, 6], "frame": 0, "box": 0, "index": 0, "absent": False},
            {"segment": [6, 12], "frame": -1, "box": -1, "index": 0, "absent": True},
        ],
        "frames": FRAMES,
        "src_size": (W, H),
        "detector": "face_yolov8m.pt",
        "confidence": 0.35,
        "select": "subject_plan",
        "select_index": 0,
        "rank_by": "left_most",
    }


def _clip():
    """Every frame a flat colour of its own, plus a darker patch over the face in it.

    Frame index over 16 is exact in float32, so "these pixels are the originals" can be asserted
    with `torch.equal` instead of a tolerance. A frame that got composited cannot hide inside a
    tolerance band.
    """
    frames = torch.zeros((FRAMES, H, W, 3))
    for i in range(FRAMES):
        tone = float(i + 1) / 16.0
        frames[i] = torch.tensor([tone, tone, tone])
        box = SUBJECT_BOX if i < ABSENT_FROM else OTHER_BOX
        x1, y1, x2, y2 = (int(v) for v in box)
        frames[i, y1:y2, x1:x2, :] = torch.tensor([tone * 0.25, tone, tone * 0.5])
    return frames


def _track(monkeypatch, payload):
    """The untouched released tracker, with detection and identity models booby-trapped."""
    def _boom(*args, **kwargs):
        raise AssertionError("a model was reached: this seam needs no detector and no weights")

    monkeypatch.setattr(nodes, "_load_detector", _boom)
    monkeypatch.setattr(nodes, "_make_embedder", _boom)
    return nodes.H3FaceTrackCrop().run(
        _clip(), "face_yolov8m.pt", 0.35, CROP_FACTOR, CANVAS, CANVAS, "manual",
        5, 5, "gaussian", "per_frame",
        select="largest_face", fallback_detector="none", fallback_head_frac=0.5,
        identity_reference=None, identity_threshold=0.28, identity_track=False,
        select_index=0, identity_model="insightface", identity_clip_vision=None,
        cut_detection="none", cut_threshold=3.0, absent_shots="off", X=0, Y=0,
        frame_index=0, face_pick=payload)


def _stitch(clip, refined, transform):
    """The untouched released stitch at its shipped defaults (undetected frames fade out)."""
    return nodes.H3FaceStitch().run(
        base_images=clip, refined_crops=refined, transform=transform,
        paste_region="full_crop", mask_dilation=0, feather=8, colour_match=0.0,
        blend=1.0)[0]


# --- the seam -----------------------------------------------------------------------------

def test_the_released_stitch_keeps_original_pixels_across_an_absent_shot(monkeypatch):
    """Frames 6-11 come out of the released pipeline exactly as they went in.

    `blend=1.0` with crops that are flat white: a frame the stitch touches cannot pass for an
    original one, so equality here is the acceptance criterion, not a proxy for it.
    """
    crops, transform, preview, report, cw, ch, kept = _track(monkeypatch, _payload())
    assert transform["absent"] == [False] * 6 + [True] * 6
    assert transform["weights"][ABSENT_FROM:] == [0.0] * (FRAMES - ABSENT_FROM)

    clip = _clip()
    out = _stitch(clip, torch.ones_like(crops), transform)
    assert tuple(out.shape) == tuple(clip.shape)

    for frame in range(ABSENT_FROM, FRAMES):
        assert torch.equal(out[frame], clip[frame]), \
            f"frame {frame} was composited - and this subject is not in this shot"
    # Not vacuous: same node, same crops, same transform - the frames holding the subject did
    # get refined, so the frames above were kept, not merely missed.
    for frame in range(0, ABSENT_FROM):
        assert not torch.equal(out[frame], clip[frame]), \
            f"frame {frame} never got refined, so this test would prove nothing"


def test_the_shot_account_names_exactly_the_frames_the_released_pipeline_left_alone(monkeypatch):
    """The report and the pixels agree: the frame range this pack derives from the crop positions
    is the frame range the released stitch left untouched - no off-by-one between what the user
    is told and what was kept."""
    crops, transform, preview, report, cw, ch, kept = _track(monkeypatch, _payload())
    clip = _clip()
    out = _stitch(clip, torch.ones_like(crops), transform)
    untouched = [f for f in range(FRAMES) if torch.equal(out[f], clip[f])]

    plan = {"version": ps.PLAN_VERSION, "match_frame": 0, "run": {"frames": FRAMES},
            "subjects": [{"index": 0, "clip_face": 0, "paired": True, "has_reference": True}]}
    account = ps.absence_account(plan, 0, transform=transform)
    assert account["absent_planned"] == [2]
    assert account["lost"] == []

    shot = account["shots"][1]
    assert [shot["start"], shot["end"]] == [ABSENT_FROM, FRAMES - 1]
    assert untouched == list(range(shot["start"], shot["end"] + 1))
    assert "subject 0 is not in shot 2 (frames 6-11)" in "\n".join(account["lines"])
