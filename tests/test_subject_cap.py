"""The subject cap, on the pure plan module: no ComfyUI, no torch, no GPU, no weights.

#10: `max_subjects` truncates the subject list AFTER the left-to-right ordering at the match
frame, and the truncation is loud - every dropped face is named with its position, the report
still states what the uncapped job would have cost, and a dropped face can never be selected
for a pass, by number or by `next`.

Same fixture plates as test_plan_store.py: three clip faces left to right (centres 150, 450,
850), the still found the outer two at twice the clip's scale.
"""

import json

import multi_face_nodes as m
import plan_store as ps
import pytest


CLIP = (1000, 1000)
STILL = (2000, 2000)
CLIP_BOXES = [[100, 100, 200, 220], [400, 400, 500, 500], [800, 120, 900, 240]]
CLIP_CONFS = [0.9, 0.8, 0.7]
STILL_BOXES = [[1600, 240, 1800, 480], [200, 200, 400, 440]]
STILL_CONFS = [0.6, 0.95]


def _plan(**over):
    args = dict(target_boxes=CLIP_BOXES, target_confs=CLIP_CONFS,
                still_boxes=STILL_BOXES, still_confs=STILL_CONFS,
                target_size=CLIP, still_size=STILL, match_frame=12,
                subject_source="union", tolerance=0.05, excluded=[], max_subjects=0,
                unmatched_faces="no_reference")
    args.update(over)
    return ps.build_subject_plan(**args)


# --- the cap itself --------------------------------------------------------------------------

def test_no_cap_keeps_every_subject_and_records_the_cap_as_zero():
    plan = _plan()

    assert plan["subject_count"] == 3
    assert plan["inputs"]["max_subjects"] == 0
    assert plan["capped_out"] == []


def test_a_cap_at_or_above_the_subject_count_is_a_no_op():
    for cap in (3, 4, 64):
        plan = _plan(max_subjects=cap)

        assert plan["subject_count"] == 3, cap
        assert [s["clip_face"] for s in plan["subjects"]] == [0, 1, 2], cap
        assert plan["capped_out"] == [], cap
        assert ps.capped_lines(plan) == [], cap


def test_the_cap_drops_the_tail_of_the_left_to_right_order():
    """Cap 1 of 3: the leftmost face is the subject, the other two are dropped in order."""
    plan = _plan(max_subjects=1)

    assert plan["subject_count"] == 1
    assert [(s["index"], s["clip_face"]) for s in plan["subjects"]] == [(0, 0)]
    assert [s["clip_face"] for s in plan["capped_out"]] == [1, 2]
    assert plan["inputs"]["max_subjects"] == 1


def test_a_capped_face_carries_no_index_and_keeps_its_position():
    """Rule #8's shape, applied to the cap: `index: None` means it can never be numbered,
    so it can never be selected, never reach `next`, never be a pass folder."""
    plan = _plan(max_subjects=1)

    assert [s["index"] for s in plan["capped_out"]] == [None, None]
    # Where each dropped face stood on the match frame, in the clip's own pixels.
    assert [(s["match_x"], s["match_y"]) for s in plan["capped_out"]] == [(450.0, 450.0),
                                                                         (850.0, 180.0)]
    assert plan["capped_out"][0]["capped_by"] == 1


def test_a_capped_still_only_face_is_named_by_where_it_would_have_stood():
    """A face the clip never saw still had a slot in the order, so the report can still say
    which one went - its position is the still's box in the clip's pixels."""
    clip = [[250, 100, 350, 220]]                 # centre x 300 in the clip
    still = [[1400, 200, 1600, 440]]              # clip-space centre x 750, never paired

    plan = ps.build_subject_plan(clip, [0.9], still, [0.9], CLIP, STILL, 12, "union",
                                 0.05, [], 1, unmatched_faces="no_reference")

    assert [s["clip_face"] for s in plan["subjects"]] == [0]
    assert [s["still_face"] for s in plan["capped_out"]] == [0]
    assert plan["capped_out"][0]["clip_box"] is None
    assert plan["capped_out"][0]["match_x"] == pytest.approx(750.0)
    assert "still face 0" in "\n".join(ps.capped_lines(plan))


def test_a_face_already_left_out_is_not_also_counted_as_capped():
    """`leave` took the unpaired face out first; the cap then cuts the tail of what is left,
    and nobody is named twice."""
    uncapped = _plan(unmatched_faces="leave")
    plan = _plan(max_subjects=1, unmatched_faces="leave")

    assert [s["clip_face"] for s in uncapped["subjects"]] == [0, 2]
    assert [s["clip_face"] for s in uncapped["left_out"]] == [1]
    assert [s["clip_face"] for s in plan["subjects"]] == [0]
    assert [s["clip_face"] for s in plan["capped_out"]] == [2]
    assert [s["clip_face"] for s in plan["left_out"]] == [1]
    assert plan["subject_count"] + len(plan["capped_out"]) + len(plan["left_out"]) == 3


def test_the_survivors_stay_contiguously_numbered_under_a_cap():
    plan = _plan(max_subjects=2)

    assert [s["index"] for s in plan["subjects"]] == [0, 1]
    assert ps.plan_subject_indices(plan) == [0, 1]


# --- nobody dropped can ever reach a pass ----------------------------------------------------

def test_next_can_never_surface_a_capped_face(tmp_path):
    """Subject Select has no new widget and no change: `next` walks the plan's own subject
    list, and a capped face is not in it. Once the capped-in subjects are done the job is
    done - it does not reach for the face the cap removed."""
    plan = _plan(max_subjects=2)
    job = ps.job_dir_for(tmp_path, "capped job")
    ps.write_plan(job, plan)

    assert ps.next_subject(plan, job)["index"] == 0
    with open(ps.manifest_path(job), "w", encoding="utf-8") as fh:
        json.dump({"version": ps.MANIFEST_VERSION,
                   "passes": {"0": {"complete": True}, "1": {"complete": True}}}, fh)

    assert ps.next_subject(plan, job) is None
    assert ps.completed_subjects(plan, job) == [0, 1]


def test_a_number_the_cap_took_out_is_refused_by_name():
    """H3 Subject Select and H3 Pass Save both resolve a number through the plan's subject
    set, so a capped face has nothing to be indexed into."""
    plan = _plan(max_subjects=2)

    assert m.checked_subject(plan, 1) == 1
    with pytest.raises(ValueError) as exc:
        m.checked_subject(plan, 2)
    assert "subject 2 is not in this plan" in str(exc.value)
    assert "0, 1" in str(exc.value)


def test_the_subject_value_is_a_plain_int_the_plan_cannot_be_read_from_it():
    """No new widget reads the plan: `subject` is a bare int and `mode` is a fixed direction, so
    which face the plan actually holds is settled by the plan, not the widget - a capped face stays
    unselectable because it is simply not in plan_subject_indices."""
    schema = m.H3SubjectSelect.INPUT_TYPES()["required"]
    assert list(schema) == ["plan", "face_pick", "subject", "mode"]
    assert schema["subject"][0] == "INT"
    plan = _plan(max_subjects=2)
    assert ps.plan_subject_indices(plan) == [0, 1]
    assert 2 not in ps.plan_subject_indices(plan)


# --- what the report says --------------------------------------------------------------------

def test_the_truncation_line_names_every_dropped_face_and_how_to_fix_it():
    text = "\n".join(ps.capped_lines(_plan(max_subjects=1)))

    assert "subject cap 1: 2 face(s) capped" in text
    # Both faces named, with the positions a user reads off the pairing preview.
    assert "clip face 1 (at 450, 450 on frame 12)" in text
    assert "clip face 2 (at 850, 180 on frame 12)" in text
    assert "Raise max_subjects" in text
    assert "excluded_faces" in text


def test_no_truncation_line_when_nobody_was_dropped():
    assert ps.capped_lines(_plan()) == []
    assert ps.capped_lines(_plan(max_subjects=9)) == []


def test_the_report_still_states_the_full_uncapped_cost():
    """The cap's whole point is generations saved, so the number it saved has to sit next to
    the number it spends."""
    assert ps.uncapped_cost_line(_plan(max_subjects=2), 22) == (
        "2 subject(s) of 3 to generate; uncapped cost would be 3 subject(s) x 22 frame(s)")


def test_a_capped_face_survives_the_job_folder(tmp_path):
    plan = _plan(max_subjects=1)
    job = ps.job_dir_for(tmp_path, "capped job")
    ps.write_plan(job, plan)

    back = ps.read_plan(job)
    assert back == plan
    assert [s["clip_face"] for s in back["capped_out"]] == [1, 2]
    assert [s["index"] for s in back["capped_out"]] == [None, None]
    assert ps.capped_lines(back) == ps.capped_lines(plan)


# --- the fingerprint: a different cap is a different job -------------------------------------

def test_a_cap_change_against_an_existing_job_folder_refuses_naming_the_difference(tmp_path):
    job = ps.job_dir_for(tmp_path, "group shot")
    ps.write_plan(job, _plan(max_subjects=1))

    lines = ps.fingerprint_difference(ps.read_plan(job), _plan(max_subjects=2))
    assert lines[0].startswith("max_subjects: ")
    assert "the job folder has 1" in lines[0] and "this plan has 2" in lines[0]

    with pytest.raises(ValueError) as exc:
        ps.check_pairing_inputs(ps.read_plan(job), _plan(max_subjects=2), job, "act on it")
    assert "max_subjects" in str(exc.value)


def test_the_same_cap_into_the_same_job_folder_is_a_resume(tmp_path):
    job = ps.job_dir_for(tmp_path, "group shot")
    ps.write_plan(job, _plan(max_subjects=2))

    assert ps.fingerprint_difference(ps.read_plan(job), _plan(max_subjects=2)) == []
    ps.check_pairing_inputs(ps.read_plan(job), _plan(max_subjects=2), job, "act on it")


def test_no_cap_into_a_no_cap_folder_is_a_resume(tmp_path):
    job = ps.job_dir_for(tmp_path, "group shot")
    ps.write_plan(job, _plan())

    assert ps.fingerprint_difference(ps.read_plan(job), _plan()) == []


# --- the plan format version -----------------------------------------------------------------

def test_a_new_plan_is_written_at_the_current_version():
    assert ps.PLAN_VERSION == 4
    assert _plan()["version"] == ps.PLAN_VERSION
    # Every version a job folder might have been written with stays readable, so an
    # interrupted job survives a pack update rather than being refused for its version number.
    assert ps.PLAN_READABLE_VERSIONS == (1, 2, 3, ps.PLAN_VERSION)


def test_a_folder_written_with_the_previous_version_is_read_not_rejected(tmp_path):
    """The honest rule: an old version is understood, not migrated. Version 1 already recorded
    max_subjects and version 2 already cut its references at the tracker's shipped default, so
    the fingerprint - not the version number - is what refuses a genuinely different job."""
    stored = _plan()
    stored["version"] = 2
    del stored["inputs"]["crop_factor"]       # version 2 recorded no crop factor at all
    job = ps.job_dir_for(tmp_path, "group shot")
    ps.write_plan(job, stored)

    back = ps.read_plan(job)
    assert back["version"] == 2
    note = ps.plan_version_note(back)
    assert note and "version 2" in note and f"version {ps.PLAN_VERSION}" in note
    # The factor a version 2 folder never recorded is read back as the default it used.
    assert ps.plan_crop_factor(back) == ps.DEFAULT_CROP_FACTOR


def test_the_current_version_gets_no_version_note(tmp_path):
    job = ps.job_dir_for(tmp_path, "group shot")
    ps.write_plan(job, _plan())

    assert ps.plan_version_note(ps.read_plan(job)) == ""


def test_a_folder_from_a_version_this_pack_cannot_read_refuses_by_name(tmp_path):
    stored = _plan()
    stored["version"] = 99
    job = ps.job_dir_for(tmp_path, "group shot")
    ps.write_plan(job, stored)

    with pytest.raises(ValueError) as exc:
        ps.read_plan(job)
    assert "99" in str(exc.value)
    assert "2" in str(exc.value)


def test_a_version_1_folder_is_a_resume_when_the_cap_matches(tmp_path):
    """An old folder is not rejected for being old; it is refused only when its recorded cap
    or its subject set actually differs."""
    stored = _plan(max_subjects=2)
    stored["version"] = 1
    job = ps.job_dir_for(tmp_path, "group shot")
    ps.write_plan(job, stored)

    assert ps.fingerprint_difference(ps.read_plan(job), _plan(max_subjects=2)) == []

    lines = ps.fingerprint_difference(ps.read_plan(job), _plan(max_subjects=1))
    assert any(line.startswith("max_subjects: ") for line in lines)
