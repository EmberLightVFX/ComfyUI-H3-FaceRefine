"""The report half of #6: the shot account has to reach the user, not just the data.

No torch and no ComfyUI: Subject Select's report builder and the shot-account lines are
reachable with numpy alone (the same import boundary the contract test pins), so the sentences a
user reads are pinned here. The one test that runs H3 Pass Save itself imports torch inside the
test, so this module still collects on a numpy-only run.

Wording is the deliverable: an absence and a lost track have to be tellible apart from the report
alone, each naming its shot, its frame range, and what to do about it.
"""

import pytest

import multi_face_nodes as m
import plan_store as ps


FRAMES = 12
SHOTS = [(0, 6), (6, 12)]


def _plan(count: int = 1, frames: int = FRAMES, job_name: str = "") -> dict:
    plan = {
        "version": ps.PLAN_VERSION,
        "match_frame": 2,
        "inputs": {"target_size": [128, 128]},
        "subjects": [
            {"index": i, "clip_face": i, "clip_box": [10.0 * i, 10.0, 40.0 + 10.0 * i, 50.0],
             "still_face": i, "paired": True, "has_reference": True, "ref_file": None}
            for i in range(count)
        ],
        "subject_count": count,
    }
    if job_name:
        plan["run"] = {"job_name": job_name, "frames": frames}
    return plan


def _narrowed(absent_flags, segments=SHOTS, frames: int = FRAMES) -> dict:
    """A narrowed pick payload for one subject: what Subject Select itself emits."""
    return {
        "version": 1,
        "boxes": [[[10.0, 10.0, 40.0, 50.0], [70.0, 10.0, 100.0, 50.0]] for _ in range(frames)],
        "confs": [[0.9, 0.8] for _ in range(frames)],
        "segments": [[int(a), int(b)] for a, b in segments],
        "picks": [{"segment": [a, b], "frame": (-1 if flag else a), "box": (-1 if flag else 0),
                   "index": 0, "absent": bool(flag)}
                  for (a, b), flag in zip(segments, absent_flags)],
        "frames": frames,
        "src_size": (128, 128),
        "detector": "face_yolov8m.pt",
        "confidence": 0.35,
        "select": "subject_plan",
        "select_index": 0,
    }


def _positions(absent, detected, segments=SHOTS, frames: int = FRAMES) -> dict:
    """A crop-position record in the shape the tracker writes (nodes.py:1908)."""
    kept = len(absent)
    return {
        "boxes": [(0, 0, 64, 64)] * kept,
        "canvas": (64, 64), "src_size": (128, 128),
        "frames": kept, "source": list(range(kept)), "source_frames": frames,
        "segments": [tuple(s) for s in segments],
        "absent": [bool(a) for a in absent], "detected": [bool(d) for d in detected],
        "weights": [0.0 if a else 1.0 for a in absent],
        "face_rect": [(1.0, 1.0, 10.0, 10.0)] * kept, "crop_factor": 2.5,
    }


# --- Subject Select's report ---------------------------------------------------------------

def test_subject_select_names_the_shots_this_subject_is_absent_from():
    report = m.subject_report(_plan(), _plan()["subjects"][0], _narrowed([False, True]))
    assert "subject 0 is not in shot 2 (frames 6-11)" in report
    assert "original pixels are kept" in report
    # An expected absence must not read like a failure.
    assert "was lost" not in report


def test_subject_select_says_the_tracker_half_needs_the_crop_positions():
    """Before the tracker has run for this subject, a clean plan is all that can be claimed."""
    narrowed = _narrowed([False], segments=[(0, FRAMES)])
    report = m.subject_report(_plan(), _plan()["subjects"][0], narrowed)
    assert "read from the crop positions" in report


def test_a_lost_shot_shows_up_once_this_subject_has_crop_positions(tmp_path):
    """Subject Select reads back what a previous run of this subject left in the job folder, so
    the same node that planned the pass is the one that reports the tracker losing them."""
    job = str(tmp_path / "job")
    ps.write_positions(job, 0, _positions([False] * 6 + [False] * 6, [True] * 6 + [False] * 6))

    lines, account = m.shot_account_lines(_plan(), 0, job_dir=job, pick=_narrowed([False, False]))
    text = "\n".join(lines)
    assert "subject 0 was lost in shot 2 (frames 6-11)" in text
    assert "identity_track" in text and "confidence" in text
    assert account["lost"] == [2]


def test_the_lost_line_says_shot_frames_cause_and_fix_once_each(tmp_path):
    """#6 left this debt: the lost line chained three clauses and read heavier than its sibling.

    One clause of cause and one of fix, the fix joined by a semicolon - and nothing given up:
    which shot, which frames, how many of them, why, and which widgets would change it. The
    absence line beside it has always read in two clauses, so the two now weigh the same.
    """
    job = str(tmp_path / "job")
    ps.write_positions(job, 0, _positions([False] * 6 + [False] * 6, [True] * 6 + [False] * 6))
    lines, _account = m.shot_account_lines(_plan(), 0, job_dir=job, pick=_narrowed([False, False]))
    line = next(text for text in lines if "was lost" in text)
    assert line.endswith(
        "subject 0 was lost in shot 2 (frames 6-11): 6 of 6 frame(s) produced nobody"
        " - faces were detected in this shot but none of them matched this subject's identity;"
        " check identity_track and identity_threshold, or lower confidence,"
        " on H3 Face Track + Crop")
    # The doubled diagnosis is what made the sentence heavy: `lost` was said three times over.
    assert "the tracker stopped finding them there" not in line


def test_planned_absence_and_a_lost_track_are_tellible_apart_in_one_pass():
    """The acceptance criterion in one report: both kinds of empty shot, same subject, same pass,
    and nothing ambiguous about which is which."""
    transform = _positions([False] * 4 + [True] * 4 + [False] * 4,
                           [True] * 4 + [False] * 4 + [False] * 4,
                           segments=[(0, 4), (4, 8), (8, 12)])
    lines, account = m.shot_account_lines(
        _plan(), 0, transform=transform,
        pick=_narrowed([False, True, False], segments=[(0, 4), (4, 8), (8, 12)]))
    text = "\n".join(lines)
    assert "subject 0 is not in shot 2 (frames 4-7)" in text
    assert "subject 0 was lost in shot 3 (frames 8-11)" in text
    assert account["absent_planned"] == [2] and account["lost"] == [3]


def test_the_shot_account_needs_no_new_widget_on_either_node():
    """Nothing was added to ask the user with: the payload each node already has is enough.
    (`chained_from` is #11 review's widget and answers a different question - the shot account
    still needs nothing of its own.)"""
    save = m.H3PassSave.INPUT_TYPES()
    assert list(save["required"]) == ["plan", "subject", "stage", "images"]
    assert list(save["optional"]) == ["transform", "chained_from"]
    select = m.H3SubjectSelect.INPUT_TYPES()
    assert list(select["required"]) == ["plan", "face_pick"]
    assert not select.get("optional")


def test_a_job_folder_that_has_never_been_saved_reports_nothing_rather_than_failing(tmp_path):
    lines, account = m.shot_account_lines(_plan(), 3, job_dir=str(tmp_path / "no-such-job"),
                                          pick=_narrowed([False], segments=[(0, FRAMES)]))
    assert account["lost"] == [] and account["absent_planned"] == []
    assert any("crop positions" in line for line in lines)


def test_pass_save_prints_the_same_account_from_the_payload_it_was_handed(tmp_path, monkeypatch):
    """No new widget on H3 Pass Save: the account is derived from the crop positions it is
    already given, and the pass report gains the lines. torch only because the node writes PNGs.

    stage=crops has the transform wired; stage=stitched has nothing wired and reads the same
    record back out of the pass folder - which is why the two reports can never disagree.
    """
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(m, "_output_root", lambda: str(tmp_path / "output"))
    plan = _plan(frames=4, job_name="shot account")
    ps.write_plan(ps.job_dir_for(str(tmp_path / "output"), "shot account"), plan)

    # Two shots; the tracker reached the second with this subject selected and produced nobody.
    transform = _positions([False, False, False, False], [True, True, False, False],
                           segments=[(0, 2), (2, 4)], frames=4)
    images = torch.zeros((4, 6, 8, 3))

    _, crops_report, index, complete = m.H3PassSave().run(plan, 0, "crops", images, transform)
    assert index == 0 and complete == 0
    assert "subject 0 was lost in shot 2 (frames 2-3)" in crops_report
    assert "identity_track" in crops_report

    _, stitched_report, _, complete = m.H3PassSave().run(plan, 0, "stitched", images)
    assert complete == 1
    assert "subject 0 was lost in shot 2 (frames 2-3)" in stitched_report
