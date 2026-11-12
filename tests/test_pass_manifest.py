"""The pass manifest's WRITE side, the resume cursor, and the pairing fingerprint.

Pure `plan_store` again (numpy, json, os only) - no ComfyUI, no torch, no GPU, and every job
folder here is under ``tmp_path``, never a real ComfyUI output folder.

The rule this file exists to pin: **a subject's pass counts as complete only once its
stitched stage is recorded.** A `crops` entry means "this far got done" and must never make
`next_subject` skip a subject - skipping one is how a person ends up unrefined in the final
clip with nothing saying so.
"""

import json
import os

import numpy as np
import pytest

import plan_store as ps


def _plan(*indices) -> dict:
    """A plan holding only what the manifest bookkeeping reads: the subject indices."""
    return {"version": ps.PLAN_VERSION,
            "subjects": [{"index": i, "clip_face": i, "still_face": i}
                         for i in indices],
            "subject_count": len(indices)}


def test_a_recorded_crops_stage_is_partial_and_a_stitched_one_completes_the_pass(tmp_path):
    job = str(tmp_path / "job")

    crops = ps.record_pass_stage(job, 0, "crops", {"frames": np.int64(4)})
    assert crops["passes"]["0"]["complete"] is False
    assert crops["passes"]["0"]["stages"]["crops"]["frames"] == 4

    plan = _plan(0, 1)
    # The crops stage is on disk, and still nothing counts as finished.
    assert ps.completed_subjects(plan, job) == []
    assert ps.next_subject(plan, job)["index"] == 0

    done = ps.record_pass_stage(job, 0, "stitched", {"frames": 4})
    assert done["passes"]["0"]["complete"] is True
    assert sorted(done["passes"]["0"]["stages"]) == ["crops", "stitched"]
    assert ps.completed_subjects(plan, job) == [0]
    assert ps.next_subject(plan, job)["index"] == 1


def test_the_manifest_round_trips_through_disk_with_the_read_side_unchanged(tmp_path):
    job = str(tmp_path / "job")
    ps.record_pass_stage(job, 2, "crops", {"frames": 3, "files": ["frame_0000.png"]})

    with open(ps.manifest_path(job), encoding="utf-8") as fh:
        raw = json.load(fh)
    assert raw["version"] == ps.MANIFEST_VERSION
    # Keys are strings because JSON object keys are, and the read side compares strings.
    assert list(raw["passes"]) == ["2"]
    assert raw["passes"]["2"]["complete"] is False
    assert raw == ps.read_manifest(job)

    assert ps.manifest_stage_record(job, 2, "crops")["frames"] == 3
    assert ps.manifest_stage_record(job, 2, "stitched") is None
    assert ps.pass_is_complete(job, 2) is False

    # A second record for the same subject joins the first rather than replacing it.
    again = ps.record_pass_stage(job, 2, "stitched", {"frames": 3})
    assert again["passes"]["2"]["stages"]["crops"]["frames"] == 3
    assert again["passes"]["2"]["complete"] is True
    assert ps.pass_is_complete(job, 2) is True


def test_resume_takes_the_first_subject_with_no_stitched_stage_recorded(tmp_path):
    job = str(tmp_path / "job")
    plan = _plan(0, 1, 2)

    # No manifest at all: nothing is finished, which is the normal state of a fresh job.
    assert not os.path.isfile(ps.manifest_path(job))
    assert ps.next_subject(plan, job)["index"] == 0

    for index in (0, 2):
        ps.record_pass_stage(job, index, "crops", {"frames": 2})
        ps.record_pass_stage(job, index, "stitched", {"frames": 2})
    # Subject 1 got its crops and never got stitched, so the job resumes there.
    ps.record_pass_stage(job, 1, "crops", {"frames": 2})

    assert ps.completed_subjects(plan, job) == [0, 2]
    assert ps.next_subject(plan, job)["index"] == 1

    ps.record_pass_stage(job, 1, "stitched", {"frames": 2})
    assert ps.completed_subjects(plan, job) == [0, 1, 2]
    assert ps.next_subject(plan, job) is None


def test_a_manifest_naming_a_subject_this_plan_does_not_have_raises(tmp_path):
    job = str(tmp_path / "job")
    ps.record_pass_stage(job, 7, "stitched", {"frames": 2})

    with pytest.raises(ValueError) as exc:
        ps.check_manifest_subjects(_plan(0, 1), job)
    message = str(exc.value)
    assert ps.manifest_path(job) in message
    assert "7" in message
    # The plan's own subjects are named too, so the fix is obvious without opening the file.
    assert "0, 1" in message


def test_pass_paths_are_deterministic_and_zero_padded(tmp_path):
    job = str(tmp_path / "job")
    assert ps.pass_dir(job, 3) == os.path.join(str(tmp_path / "job"), "passes",
                                               "subject_0003")
    assert ps.stage_dir(job, 3, "crops").endswith(os.path.join("subject_0003", "crops"))
    names = [os.path.basename(p) for p in ps.stage_files(job, 3, "stitched", 12_005)]
    assert names[:2] == ["frame_0000.png", "frame_0001.png"]
    assert names[-1] == "frame_12004.png"
    assert ps.positions_path(job, 3).endswith(os.path.join("subject_0003", "crops.json"))
    with pytest.raises(ValueError) as exc:
        ps.stage_dir(job, 3, "frames")
    assert "crops | stitched" in str(exc.value)


def _built_plan(match_frame=2, tolerance=0.05, source="union", excluded=(),
                still_size=(2000, 2000)):
    return ps.build_subject_plan(
        [[100, 100, 200, 220], [400, 400, 500, 500]], [0.9, 0.8],
        [[200, 200, 400, 440], [800, 800, 1000, 1000]], [0.95, 0.6],
        (1000, 1000), still_size, match_frame, source, tolerance, excluded)


def test_the_fingerprint_survives_the_plan_written_to_and_read_from_disk(tmp_path):
    job = str(tmp_path / "job")
    plan = _built_plan()
    ps.write_plan(job, plan)
    stored = ps.read_plan(job)

    assert ps.pairing_fingerprint(stored) == ps.pairing_fingerprint(plan)
    assert ps.fingerprint_difference(stored, plan) == []
    # The same pairing re-planned into the same folder is a resume, not a mix.
    assert ps.check_pairing_inputs(stored, plan, job, "re-run H3 Multi-Face Plan") is None


def test_the_fingerprint_names_every_pairing_input_that_differs(tmp_path):
    job = str(tmp_path / "job")
    one = _built_plan()

    moved = _built_plan(match_frame=5)
    only = ps.fingerprint_difference(one, moved)
    assert len(only) == 1
    assert "match_frame" in only[0]
    with pytest.raises(ValueError) as moved_exc:
        ps.check_pairing_inputs(one, moved, job, "pick a new job_name")
    assert "match_frame" in str(moved_exc.value)

    several = _built_plan(match_frame=5, tolerance=0.2, source="target_clip",
                          excluded=(1,), still_size=(3000, 3000))
    lines = ps.fingerprint_difference(one, several)
    named = "\n".join(lines)
    for field in ("match_frame", "match_tolerance", "subject_source", "excluded_faces",
                  "still_size"):
        assert field in named, field
    assert len(lines) >= 5

    with pytest.raises(ValueError) as exc:
        ps.check_pairing_inputs(one, several, job, "pick a new job_name")
    message = "\n".join(str(exc.value).split("\n"))
    assert job in message
    assert "job_name" in message
    assert "match_tolerance" in message

    # A plan that paired someone differently is a different pairing even at equal widgets.
    repaired = _built_plan()
    repaired["subjects"][0]["still_face"] = None
    assert any("subject" in line for line in ps.fingerprint_difference(_built_plan(),
                                                                      repaired))


# --- the chain each pass DECLARES it composited onto (#11 review, P3) -----------------

def test_the_chain_is_read_back_from_what_each_pass_declared(tmp_path):
    """The composite hands back the last subject stitched frames on the claim that each pass
    composited onto the previous pass's result. Pass Save records that claim, so the composite
    can check it instead of asserting it."""
    job = ps.job_dir_for(tmp_path, "group shot")
    plan = _plan(0, 1, 2)
    ps.write_plan(job, plan)
    for subject, chained in ((0, -1), (1, 0), (2, 1)):
        ps.record_pass_stage(job, subject, "stitched",
                             {"frames": 4, "size": [8, 6], "chained_from": chained})

    check = ps.chain_check(plan, job)
    assert check["declared"] == {0: -1, 1: 0, 2: 1}
    assert check["wrong"] == [] and check["missing"] == []


def test_a_pass_that_declares_the_wrong_predecessor_is_named(tmp_path):
    """Every pass wired onto the target clip instead of the previous pass: subject 0 silently
    reverts in the finished clip while the report claims a chain. Refused, naming the subject."""
    job = ps.job_dir_for(tmp_path, "group shot")
    plan = _plan(0, 1, 2)
    ps.write_plan(job, plan)
    for subject, chained in ((0, -1), (1, -1), (2, 1)):
        ps.record_pass_stage(job, subject, "stitched", {"chained_from": chained})

    check = ps.chain_check(plan, job)
    assert check["wrong"] == [(1, -1, 0)]
    assert check["missing"] == []


def test_an_older_job_that_declared_nothing_reports_the_gap(tmp_path):
    """A job saved before chained_from existed is not refused for its age - the composite warns
    that the chain is unproven, and names which subjects it cannot vouch for."""
    job = ps.job_dir_for(tmp_path, "group shot")
    plan = _plan(0, 1, 2)
    ps.write_plan(job, plan)
    for subject in (0, 1, 2):
        ps.record_pass_stage(job, subject, "stitched", {"frames": 4})

    check = ps.chain_check(plan, job)
    # Subject 0 sits on the target clip by definition, so its silence costs nothing.
    assert check["missing"] == [1, 2]
    assert check["wrong"] == []


def test_the_crop_positions_round_trip_with_the_tracker_s_shapes_back(tmp_path):
    """`crops.json` is readable JSON that a resumed pass can stitch from - including the
    tuples the tracker built, which JSON would otherwise turn back into lists."""
    job = str(tmp_path / "job")
    transform = {
        "boxes": [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        "canvas": (768, 768),
        "src_size": (1000, 1000),
        "frames": np.int64(2),
        "source": [0, 2],
        "source_frames": 3,
        "segments": [(0, 1), (1, 2)],
        "absent": [False, True],
        "weights": [np.float64(1.0), 0.0],
        "detected": [True, False],
        "face_rect": [(1.5, 2.5, 3.5, 4.5), (0.0, 0.0, 1.0, 1.0)],
        "crop_factor": np.float64(2.5),
    }
    path = ps.write_positions(job, 4, transform)
    assert path == ps.positions_path(job, 4)

    back = ps.read_positions(job, 4)
    assert back["canvas"] == (768, 768)
    assert back["src_size"] == (1000, 1000)
    assert back["segments"] == [(0, 1), (1, 2)]
    assert back["face_rect"] == [(1.5, 2.5, 3.5, 4.5), (0.0, 0.0, 1.0, 1.0)]
    assert back["frames"] == 2
    assert back["boxes"] == [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    assert back["weights"] == [1.0, 0.0]
    assert back["absent"] == [False, True]
    assert back["crop_factor"] == 2.5

    with pytest.raises(FileNotFoundError) as exc:
        ps.read_positions(job, 9)
    assert ps.positions_path(job, 9) in str(exc.value)
