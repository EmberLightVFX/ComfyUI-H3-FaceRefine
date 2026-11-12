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
import plan_store as ps

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


def _save(pack, plan, subject, stitched, crops=None, chained_from=None):
    """One subject's whole pass, checkpointed the way the pass nodes do it.

    A pass DECLARES which plate it was composited onto (H3 Pass Save's `chained_from`): -1 is the
    target clip, any other number the subject whose stitched result it took as its base. Left unset
    here it is the previous subject in plan order, which is the chain these tests describe; pass
    `chained_from` to test a pass that declares something else.
    """
    save = pack.NODE_CLASS_MAPPINGS["H3PassSave"]()
    frames = stitched if crops is None else crops
    save.run(plan, subject, "crops", frames, _transform(int(frames.shape[0])))
    base = ps.CHAINED_TO_CLIP if subject <= 0 else subject - 1
    save.run(plan, subject, "stitched", stitched,
             chained_from=base if chained_from is None else chained_from)


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
    # The chain, as declared at H3 Pass Save, so nobody goes looking for a blend node.
    assert "chain as declared by H3 Pass Save" in report


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
    plan, _, _, _ = _run(pack, monkeypatch, mode="no_reference")   # three subjects
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
    save.run(plan, 1, "stitched", three, chained_from=0)

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
    # The chain is reported as DECLARED, never as assumed, so nobody goes looking for a blend
    # node - and nobody is told the chain was proved when nothing said so.
    assert ("chain as declared by H3 Pass Save: subject 0 onto the target clip; "
            "subject 1 onto subject 0") in report
    assert "every pass named the plate it was composited onto" in report


def test_a_correctly_declared_chain_is_the_finished_clip(pack, monkeypatch):
    """Subject 0 onto the target clip, subject 1 onto subject 0: the plan's own order, declared
    by both passes, so returning subject 1's stitched frames is honest."""
    plan, _, _, _ = _run(pack, monkeypatch, excluded="1")
    _save(pack, plan, 0, PASS_0, chained_from=-1)
    _save(pack, plan, 1, PASS_1, chained_from=0)

    images, report, final, _ = _composite(pack)().run(plan, CLIP, 0.25)

    assert torch.equal(images, PASS_1) and final == 1
    assert "chain as declared by H3 Pass Save" in report
    assert "recorded no chained_from" not in report


def test_a_pass_that_declares_the_wrong_plate_refuses_naming_it(pack, monkeypatch):
    """Every pass wired onto the target clip instead of the previous pass: subject 0 silently
    reverts while the report claims a chain. That is a refusal, and it names the subject."""
    plan, _, _, _ = _run(pack, monkeypatch, excluded="1")
    _save(pack, plan, 0, PASS_0, chained_from=-1)
    _save(pack, plan, 1, PASS_1, chained_from=-1)     # onto the clip, not onto subject 0

    with pytest.raises(Exception) as exc:
        _composite(pack)().run(plan, CLIP, 0.25)
    message = str(exc.value)
    assert "subject 1" in message
    assert "chained_from" in message
    assert "composites it onto subject 0" in message


def test_a_job_that_declared_no_chain_warns_rather_than_refusing(pack, monkeypatch, tmp_path):
    """A job folder saved before chained_from existed still composites. The claim is simply
    unproven, and the report says which subjects it cannot vouch for."""
    plan, _, _, _ = _run(pack, monkeypatch, excluded="1")
    _save(pack, plan, 0, PASS_0)
    _save(pack, plan, 1, PASS_1)
    # Take the declarations back out, the way an older job folder has none of them.
    job = str(_job(tmp_path))
    manifest = ps.read_manifest(job)
    for entry in (manifest.get("passes") or {}).values():
        for stage in (entry.get("stages") or {}).values():
            stage.pop("chained_from", None)
    ps.write_manifest(job, manifest)

    images, report, final, _ = _composite(pack)().run(plan, CLIP, 0.25)

    assert torch.equal(images, PASS_1) and final == 1
    assert "subject 1 recorded no chained_from" in report
    assert "unproven" in report
    assert "every pass named the plate" not in report
