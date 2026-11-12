"""**H3 Pass Save** and **H3 Pass Load** round-tripping one subject's pass through the job.

Needs torch, so the numpy-only CI run skips it. The job folder lands under ``tmp_path``:
``pack`` is the fixture from the plan-node test, which redirects ComfyUI's output root there
whether or not a real ComfyUI tree happens to be importable - nothing in this file can write
into a real ComfyUI output folder.

What is under test is the checkpoint itself: the frames and crop positions that go in are the
ones that come back out, in the same order; a crops-only pass does not make a subject
complete; and a job whose pairing inputs no longer match refuses rather than mixing two
pairings into one clip.
"""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# The plan-node fixture, reused rather than rebuilt: it loads the pack as ComfyUI loads it,
# stubs ComfyUI/folder_paths when needed, and redirects the output root to tmp_path either
# way (the bug that once leaked job folders into the real output tree).
from test_plan_node_run import _run, pack  # noqa: F401  (pack is the fixture)

FRAMES = 4


def _frames(count: int = FRAMES):
    """Distinct per frame, and exact in 8-bit: what comes back can be compared, not
    apologised for. Each frame is a flat colour, so an ordering bug cannot hide."""
    arr = np.zeros((count, 6, 8, 3), dtype=np.uint8)
    for i in range(count):
        arr[i] = i * 64
    return torch.from_numpy(arr.astype(np.float32) / 255.0)


def _transform(count: int = FRAMES):
    """A crop-position payload in the shape `H3FaceTrackCrop` hands the stitch node."""
    return {
        "boxes": [[10.0 + 10 * i, 20.0, 100.0, 120.0] for i in range(count)],
        "canvas": (768, 768),
        "src_size": (1000, 1000),
        "frames": count,
        "source": list(range(count)),
        "source_frames": count,
        "segments": [(0, count)],
        "absent": [i == count - 1 for i in range(count)],
        "weights": [1.0, 0.5, 1.0, 0.25][:count],
        "detected": [True, True, True, False][:count],
        "face_rect": [(1.0 + i, 2.0, 3.0, 4.0) for i in range(count)],
        "crop_factor": 2.5,
    }


def _job(tmp_path):
    return tmp_path / "output" / "group shot"


def test_a_saved_crops_pass_loads_back_as_the_same_frames_in_the_same_order(pack, monkeypatch,
                                                                           tmp_path):
    plan, _, _, _ = _run(pack, monkeypatch)
    crops = _frames()
    transform = _transform()

    save = pack.NODE_CLASS_MAPPINGS["H3PassSave"]()
    passthrough, report, subject_index, complete = save.run(plan, 0, "crops", crops,
                                                            transform)
    assert subject_index == 0
    assert complete == 0                      # crops alone is not a finished pass
    assert torch.equal(passthrough, crops)
    assert "subject 0" in report

    job = _job(tmp_path)
    folder = job / "passes" / "subject_0000" / "crops"
    names = sorted(p.name for p in folder.iterdir())
    assert names == ["frame_0000.png", "frame_0001.png", "frame_0002.png", "frame_0003.png"]
    assert (job / "passes" / "subject_0000" / "crops.json").is_file()
    assert str(folder) in report

    load = pack.NODE_CLASS_MAPPINGS["H3PassLoad"]()
    images, positions, load_report = load.run(plan, 0, "crops")
    assert images.shape == crops.shape
    assert torch.equal(images, crops)          # frame for frame, in the same order
    assert positions["boxes"] == transform["boxes"]
    assert positions["canvas"] == (768, 768)   # tuples, exactly as the tracker built them
    assert positions["src_size"] == (1000, 1000)
    assert positions["segments"] == [(0, FRAMES)]
    assert positions["absent"] == [False, False, False, True]
    assert positions["weights"] == [1.0, 0.5, 1.0, 0.25]
    assert positions["detected"] == [True, True, True, False]
    assert positions["source"] == list(range(FRAMES))
    assert positions["face_rect"] == transform["face_rect"]   # the stitch node's paste rects
    assert positions["crop_factor"] == 2.5
    assert str(folder) in load_report



def test_a_stitched_pass_completes_the_subject_and_the_job_resumes_past_it(pack, monkeypatch,
                                                                          tmp_path):
    import plan_store as ps

    plan, _, _, _ = _run(pack, monkeypatch)
    save = pack.NODE_CLASS_MAPPINGS["H3PassSave"]()
    job = str(_job(tmp_path))

    save.run(plan, 0, "crops", _frames(), _transform())
    with open(job + "/manifest.json", encoding="utf-8") as fh:
        written = json.load(fh)
    assert written["passes"]["0"]["complete"] is False
    assert ps.next_subject(plan, job)["index"] == 0

    # A stitched stream has nothing new to record, so it is legal with transform unwired.
    _, stitched_report, _, complete = save.run(plan, 0, "stitched", _frames())
    assert complete == 1
    assert "nothing new to record" in stitched_report
    with open(job + "/manifest.json", encoding="utf-8") as fh:
        written = json.load(fh)
    assert written["passes"]["0"]["complete"] is True
    assert sorted(written["passes"]["0"]["stages"]) == ["crops", "stitched"]

    load = pack.NODE_CLASS_MAPPINGS["H3PassLoad"]()
    composited, _, report = load.run(plan, 0, "stitched")
    assert torch.equal(composited, _frames())
    assert "stitched" in report
    # The cursor has moved on: subject 1 is next, which is the whole point of the manifest.
    assert ps.next_subject(plan, job)["index"] == 1
    assert ps.completed_subjects(plan, job) == [0]


def test_a_missing_pass_names_the_subject_the_stage_and_the_path(pack, monkeypatch, tmp_path):
    plan, _, _, _ = _run(pack, monkeypatch)
    load = pack.NODE_CLASS_MAPPINGS["H3PassLoad"]()

    with pytest.raises(Exception) as exc:
        load.run(plan, 1, "crops")
    message = str(exc.value)
    job = _job(tmp_path)
    assert "subject 1" in message
    assert "crops" in message
    assert str(job / "passes" / "subject_0001") in message

    # And half a pass: a frame deleted after it was written.
    save = pack.NODE_CLASS_MAPPINGS["H3PassSave"]()
    save.run(plan, 0, "crops", _frames(), _transform())
    folder = job / "passes" / "subject_0000" / "crops"
    (folder / "frame_0003.png").unlink()
    with pytest.raises(Exception) as exc:
        load.run(plan, 0, "crops")
    message = str(exc.value)
    assert "subject 0" in message and "crops" in message
    assert "frame_0003.png" in message


def test_pass_save_refuses_a_plan_whose_pairing_inputs_differ_from_the_job(pack, monkeypatch,
                                                                          tmp_path):
    import copy

    plan, _, _, _ = _run(pack, monkeypatch)
    save = pack.NODE_CLASS_MAPPINGS["H3PassSave"]()
    moved = copy.deepcopy(plan)
    moved["inputs"]["match_frame"] = 3
    moved["inputs"]["excluded_faces"] = [1]

    with pytest.raises(ValueError) as exc:
        save.run(moved, 0, "crops", _frames(), _transform())
    message = str(exc.value)
    for wanted in ("match_frame", "excluded_faces", "job_name", str(_job(tmp_path))):
        assert wanted in message

    load = pack.NODE_CLASS_MAPPINGS["H3PassLoad"]()
    with pytest.raises(ValueError) as exc:
        load.run(moved, 0, "crops")
    assert "match_frame" in str(exc.value)


def test_the_plan_node_refuses_to_overwrite_a_job_planned_from_other_inputs(pack, monkeypatch,
                                                                           tmp_path):
    """Same job_name, different widgets: the passes already in the folder were cut from the
    old pairing, so re-planning there would mix two pairings into one clip."""
    _run(pack, monkeypatch)
    with pytest.raises(ValueError) as exc:
        _run(pack, monkeypatch, source="target_clip")
    message = str(exc.value)
    assert "subject_source" in message
    assert "job_name" in message
    assert str(_job(tmp_path)) in message


def test_the_plan_node_re_plans_the_same_pairing_and_the_saved_pass_survives(pack, monkeypatch,
                                                                            tmp_path):
    """Re-running the plan is the normal case of resuming a job: nothing may be refused, and
    nothing already checkpointed may be touched."""
    import plan_store as ps

    plan, _, _, _ = _run(pack, monkeypatch)
    save = pack.NODE_CLASS_MAPPINGS["H3PassSave"]()
    save.run(plan, 0, "crops", _frames(), _transform())
    save.run(plan, 0, "stitched", _frames())

    again, _, _, _ = _run(pack, monkeypatch)
    job = str(_job(tmp_path))
    assert ps.completed_subjects(again, job) == [0]
    assert ps.next_subject(again, job)["index"] == 1
    assert (_job(tmp_path) / "passes" / "subject_0000" / "crops"
            / "frame_0002.png").is_file()

    load = pack.NODE_CLASS_MAPPINGS["H3PassLoad"]()
    images, _, _ = load.run(again, 0, "crops")
    assert torch.equal(images, _frames())
