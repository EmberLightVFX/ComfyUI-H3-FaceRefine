"""**H3 Multi-Face Composite**: the finished clip, and the gate that proves the chain is whole.

Needs torch, so the numpy-only CI run skips it. The job folder lands under ``tmp_path`` -
``pack`` is the plan-node fixture, which redirects ComfyUI's output root there whether or not a
real ComfyUI tree is importable, so nothing here can write into a real output folder.

Passes **chain**: subject 1's stitch already composited onto subject 0's saved stitched result,
so the last subject's stitched frames ARE the finished clip. This node assembles and verifies;
it never re-blends, because a second feather over a feathered edge is a bug, not a feature.
"""

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# The plan-node fixture and the crop-position shape, reused rather than rebuilt.
from test_pass_save_load_roundtrip import _transform
from test_plan_node_run import _run, pack  # noqa: F401  (pack is the fixture)

FRAMES = 4


def _frames(offset: int, step: int):
    """Flat colour per frame, exact in 8-bit, so ordering and blending cannot hide."""
    arr = np.zeros((FRAMES, 6, 8, 3), dtype=np.uint8)
    for i in range(FRAMES):
        arr[i] = int(offset + i * step)
    return torch.from_numpy(arr.astype(np.float32) / 255.0)


CLIP = _frames(0, 64)            # the target clip: 0, 64, 128, 192
PASS_0 = _frames(10, 40)         # subject 0 stitched: 10, 50, 90, 130
PASS_1 = _frames(200, -30)       # subject 1 stitched: 200, 170, 140, 110


def _job(tmp_path):
    return tmp_path / "output" / "group shot"


def _composite(pack):
    return pack.NODE_CLASS_MAPPINGS["H3MultiFaceComposite"]


def _save(pack, plan, subject, stitched, crops=None):
    """One subject's whole pass, checkpointed the way the pass nodes do it."""
    save = pack.NODE_CLASS_MAPPINGS["H3PassSave"]()
    frames = stitched if crops is None else crops
    save.run(plan, subject, "crops", frames, _transform(int(frames.shape[0])))
    save.run(plan, subject, "stitched", stitched)


def test_the_two_subject_job_composites_in_plan_order_and_returns_the_last_pass(pack, monkeypatch):
    plan, _, _, _ = _run(pack, monkeypatch, excluded="1")     # two subjects
    _save(pack, plan, 0, PASS_0)
    _save(pack, plan, 1, PASS_1)

    images, report, final, count = _composite(pack)().run(plan, CLIP, 0.25)

    assert count == 2
    assert final == 1
    # Nothing was re-blended: the finished clip is subject 1's stitched frames, frame for frame,
    # and those frames already carry subject 0 composited into them.
    assert images.shape == PASS_1.shape
    assert torch.equal(images, PASS_1)
    assert "subject 0 -> subject 1" in report
    assert "final result: subject 1" in report


def test_every_pass_is_reported_with_its_frame_count(pack, monkeypatch):
    plan, _, _, _ = _run(pack, monkeypatch, excluded="1")
    _save(pack, plan, 0, PASS_0)
    _save(pack, plan, 1, PASS_1)

    _, report, _, _ = _composite(pack)().run(plan, CLIP, 0.25)

    assert f"subject 0   stitched {FRAMES} frame(s)" in report
    assert f"subject 1   stitched {FRAMES} frame(s)" in report
    assert "order: subject 0 -> subject 1" in report
    # The chain, stated plainly, so nobody goes looking for a blend node.
    assert "chained" in report


def test_the_audio_note_names_the_one_audio_that_survives(pack, monkeypatch):
    """This node only ever carries image batches, so it says what happens to the sound."""
    plan, _, _, _ = _run(pack, monkeypatch, excluded="1")
    _save(pack, plan, 0, PASS_0)
    _save(pack, plan, 1, PASS_1)

    _, report, _, _ = _composite(pack)().run(plan, CLIP, 0.25)

    assert "audio" in report
    assert "the target clip's audio" in report
    assert "discarded" in report


def test_a_missing_middle_pass_refuses_naming_it_and_where_next_goes(pack, monkeypatch):
    """Half a composited clip is worse than an error: a crops-only pass is not a pass."""
    plan, _, _, _ = _run(pack, monkeypatch)                   # three subjects
    _save(pack, plan, 0, PASS_0)
    save = pack.NODE_CLASS_MAPPINGS["H3PassSave"]()
    save.run(plan, 1, "crops", PASS_1, _transform(FRAMES))    # got this far, no stitched
    _save(pack, plan, 2, PASS_1)

    with pytest.raises(Exception) as exc:
        _composite(pack)().run(plan, CLIP, 0.25)
    message = str(exc.value)

    assert "subject 1" in message
    assert "stitched" in message
    assert "next subject: 1" in message
    assert "half-composited" in message


def test_the_same_plan_and_job_give_the_identical_tensors_twice(pack, monkeypatch):
    """Determinism: the composite reads the checkpoint, it never generates anything."""
    plan, _, _, _ = _run(pack, monkeypatch, excluded="1")
    _save(pack, plan, 0, PASS_0)
    _save(pack, plan, 1, PASS_1)

    node = _composite(pack)()
    first, first_report, _, _ = node.run(plan, CLIP, 0.25)
    second, second_report, _, _ = node.run(plan, CLIP, 0.25)

    assert torch.equal(first, second)
    assert first_report == second_report


def test_a_pass_whose_frames_disagree_with_the_target_clip_refuses(pack, monkeypatch):
    """The clip is in the graph precisely so this can be proved instead of assumed."""
    plan, _, _, _ = _run(pack, monkeypatch, excluded="1")
    _save(pack, plan, 0, PASS_0)
    save = pack.NODE_CLASS_MAPPINGS["H3PassSave"]()
    three = PASS_1[:3]
    save.run(plan, 1, "crops", three, _transform(3))
    save.run(plan, 1, "stitched", three)

    with pytest.raises(Exception) as exc:
        _composite(pack)().run(plan, CLIP, 0.25)
    message = str(exc.value)
    assert "subject 1" in message
    assert "3 frame(s)" in message and "4 frame(s)" in message


def test_overlapping_subjects_warn_in_the_report_and_still_finish(pack, monkeypatch):
    """Two people standing close: their crops overlap, the clip is still the right answer."""
    plan, _, _, _ = _run(pack, monkeypatch, excluded="1")
    _save(pack, plan, 0, PASS_0)
    _save(pack, plan, 1, PASS_1)

    # clip_box is not part of the pairing fingerprint, so moving a box is a legal way to put
    # two subjects' crops on top of each other without replanning the job.
    crowded = copy.deepcopy(plan)
    crowded["subjects"][1]["clip_box"] = [250.0, 100.0, 350.0, 200.0]

    images, report, final, _ = _composite(pack)().run(crowded, CLIP, 0.2)

    assert torch.equal(images, PASS_1) and final == 1
    assert "overlap warning" in report
    assert "no overlapping" not in report


def test_a_plan_whose_subjects_stay_apart_says_so(pack, monkeypatch):
    plan, _, _, _ = _run(pack, monkeypatch, excluded="1")
    _save(pack, plan, 0, PASS_0)
    _save(pack, plan, 1, PASS_1)

    _, report, _, _ = _composite(pack)().run(plan, CLIP, 0.25)

    assert "no overlapping subject crop regions" in report
    assert "overlap warning" not in report

    assert f"subject 1   stitched {FRAMES} frame(s)" in report
    assert "order: subject 0 -> subject 1" in report
    # The chain, stated plainly, so nobody goes looking for a blend node.
    assert "chained" in report
