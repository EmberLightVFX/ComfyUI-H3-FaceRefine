"""What happens to a face that found no partner in the reference still (#8).

The decision is the PLAN's, not the user's per pass: `unmatched_faces` shapes the subject set,
and the subject set is what every downstream node is gated on - the composite needs a stitched
pass for every subject, and the pairing fingerprint compares the subject set to stop two
pairings mixing into one job folder. Deciding it once, here, means the answer is in `plan.json`
and every pass reads the same one.

Pure: numpy and the standard library only (the import boundary `test_node_contract.py` pins), so
the three modes and what they do to the subject set are testable with no ComfyUI, no torch, no
weights and no GPU.

Fixture is the one `test_plan_store.py` uses: three clip faces left to right, the still found
the outer two at twice the clip's scale, so the middle face (clip face 1, centre 450,450) is the
one with no partner.
"""

import pytest

import plan_store as ps


CLIP = (1000, 1000)
STILL = (2000, 2000)
CLIP_BOXES = [[100, 100, 200, 220], [400, 400, 500, 500], [800, 120, 900, 240]]
CLIP_CONFS = [0.9, 0.8, 0.7]
STILL_BOXES = [[1600, 240, 1800, 480], [200, 200, 400, 440]]
STILL_CONFS = [0.6, 0.95]


def _plan(mode=ps.DEFAULT_UNMATCHED_FACES, **over):
    args = dict(target_boxes=CLIP_BOXES, target_confs=CLIP_CONFS,
                still_boxes=STILL_BOXES, still_confs=STILL_CONFS,
                target_size=CLIP, still_size=STILL, match_frame=12,
                subject_source="union", tolerance=0.05, excluded=[], max_subjects=0,
                unmatched_faces=mode)
    args.update(over)
    return ps.build_subject_plan(**args)


def _only_unpaired(plan) -> dict:
    """The one subject this fixture makes that has no partner in the still."""
    rows = [s for s in plan["subjects"] if s["clip_face"] == 1]
    assert len(rows) == 1, [s["clip_face"] for s in plan["subjects"]]
    return rows[0]


# --- the setting ---------------------------------------------------------------------------

def test_the_mode_vocabulary_lives_in_the_plan_module():
    """One place decides the words, so the node's combo and the plan cannot drift apart."""
    assert ps.UNMATCHED_FACES == ("leave", "no_reference", "with_still")
    assert ps.DEFAULT_UNMATCHED_FACES == "leave"
    assert ps.REF_KINDS == ("crop", "still", "none")


def test_a_mode_that_is_not_one_of_the_three_is_refused():
    with pytest.raises(ValueError) as exc:
        _plan(mode="everything")
    text = str(exc.value)
    assert "unmatched_faces" in text
    for option in ps.UNMATCHED_FACES:
        assert option in text


def test_the_mode_is_recorded_in_the_inputs_the_fingerprint_reads():
    assert _plan(mode="with_still")["inputs"]["unmatched_faces"] == "with_still"
    assert _plan()["inputs"]["unmatched_faces"] == "leave"


# --- leave: the face is not made a subject at all -------------------------------------------

def test_leave_does_not_make_a_subject_of_an_unpaired_face():
    plan = _plan()

    assert plan["subject_count"] == 2 == len(plan["subjects"])
    assert [s["clip_face"] for s in plan["subjects"]] == [0, 2]
    assert [s["index"] for s in plan["subjects"]] == [0, 1]     # the set stays contiguous
    assert [s["has_reference"] for s in plan["subjects"]] == [True, True]
    # The face that was left is recorded, not forgotten: the report has to name it.
    assert [s["clip_face"] for s in plan["left_out"]] == [1]
    assert plan["left_out"][0]["has_reference"] is False


def test_leave_names_every_face_it_left_and_where_it_sat_at_the_match_frame():
    plan = _plan()
    text = "\n".join(ps.unmatched_lines(plan))

    assert "unmatched_faces=leave" in text
    assert "clip face 1" in text
    assert "450" in text                                   # its position on frame 12
    assert "original pixels" in text
    # Leaving is the safe default, not an error, and the report has to say so: a wrong pairing
    # is worse than an unrefined face.
    assert "not a failure" in text
    assert "safe default" in text


def test_leave_drops_only_faces_that_found_no_partner():
    """Every clip face has a still partner here, so `leave` leaves nobody out."""
    plan = _plan(still_boxes=[[200, 200, 400, 440], [800, 800, 1000, 1000],
                              [1600, 240, 1800, 480]])

    assert plan["subject_count"] == 3
    assert plan["left_out"] == []
    assert ps.unmatched_lines(plan) == []


def test_leave_never_drops_a_paired_face_or_repeats_an_excluded_one():
    """Exclusion and leaving are two different decisions with two different records: an excluded
    face is excluded whether or not `leave` is on, and is not reported as left out."""
    plan = _plan(excluded=[0])

    assert plan["excluded_faces"] == [0]
    assert plan["inputs"]["excluded_faces"] == [0]
    assert [s["clip_face"] for s in plan["subjects"]] == [2]
    assert [s["clip_face"] for s in plan["left_out"]] == [1]
    text = "\n".join(ps.unmatched_lines(plan))
    assert "clip face 1" in text and "clip face 0" not in text


def test_a_face_only_the_still_saw_keeps_its_own_crop_in_every_mode():
    """It found no partner, but it has a subject reference cut from the still's own face, so
    `leave` has nothing to leave out."""
    clip = [[100, 100, 200, 220], [300, 300, 400, 400]]
    still = [[100, 100, 200, 220], [600, 600, 700, 720]]

    for mode in ps.UNMATCHED_FACES:
        plan = ps.build_subject_plan(clip, [0.9, 0.8], still, [0.9, 0.9], (1000, 1000),
                                     (1000, 1000), 0, "reference_still", 0.05, [], 0, mode)
        assert [s["has_reference"] for s in plan["subjects"]] == [True, True], mode
        assert [s["ref_kind"] for s in plan["subjects"]] == ["crop", "crop"], mode
        assert plan["left_out"] == [], mode


# --- no_reference: a subject, with nothing wired in -----------------------------------------

def test_no_reference_makes_a_subject_of_it_with_no_subject_reference():
    plan = _plan(mode="no_reference")
    subject = _only_unpaired(plan)

    assert plan["subject_count"] == 3
    assert subject["has_reference"] is False
    assert subject["ref_kind"] == "none"
    assert subject["ref_crop"] is None
    assert subject["ref_file"] is None
    assert plan["left_out"] == []


def test_no_reference_says_the_pass_gets_nothing_in_the_reference_slots():
    text = "\n".join(ps.unmatched_lines(_plan(mode="no_reference")))

    assert "unmatched_faces=no_reference" in text
    assert "no subject reference" in text
    assert "reference slots" in text
    assert "weaker" in text                    # weaker than a paired subject, said plainly


# --- with_still: the whole still becomes the subject reference ------------------------------

def test_with_still_gives_the_unpaired_subject_the_whole_still():
    plan = _plan(mode="with_still")
    subject = _only_unpaired(plan)

    assert subject["has_reference"] is True
    assert subject["ref_kind"] == "still"
    # The whole still, at its own pixels. `crop_rect_for_reference` is for cut references, and
    # this is deliberately not one.
    assert subject["ref_crop"] == [0, 0, 2000, 2000]
    assert subject["ref_file"] is None         # the node fills it in once it has written it


def test_with_still_warns_that_conditioning_is_whole_frame():
    text = "\n".join(ps.unmatched_lines(_plan(mode="with_still")))

    assert "unmatched_faces=with_still" in text
    assert "whole reference still" in text
    assert "2000x2000" in text
    assert "whole-frame" in text
    assert "not a guarantee" in text           # H3 chooses which face it applies to


# --- a paired subject is untouched by the setting -------------------------------------------

def test_a_paired_subject_is_identical_in_all_three_modes():
    plans = [_plan(mode=mode) for mode in ps.UNMATCHED_FACES]
    rows = [[{k: v for k, v in subject.items() if k != "index"}
             for subject in plan["subjects"] if subject["paired"]] for plan in plans]

    assert rows[0] == rows[1] == rows[2]
    for plan in plans:
        paired = [s for s in plan["subjects"] if s["paired"]]
        assert [s["ref_kind"] for s in paired] == ["crop"] * len(paired)
        assert [s["clip_face"] for s in paired] == [0, 2]


# --- the counts a report reads off ----------------------------------------------------------

def test_the_plan_counts_its_subjects_by_kind():
    assert ps.subject_kind_counts(_plan()) == {
        "paired": 2, "unpaired_no_reference": 0, "unpaired_whole_still": 0, "left_out": 1}
    assert ps.subject_kind_counts(_plan(mode="no_reference")) == {
        "paired": 2, "unpaired_no_reference": 1, "unpaired_whole_still": 0, "left_out": 0}
    assert ps.subject_kind_counts(_plan(mode="with_still")) == {
        "paired": 2, "unpaired_no_reference": 0, "unpaired_whole_still": 1, "left_out": 0}


def test_every_subject_is_counted_exactly_once():
    for mode in ps.UNMATCHED_FACES:
        plan = _plan(mode=mode)
        counts = ps.subject_kind_counts(plan)
        assert sum(counts.values()) == plan["subject_count"] + counts["left_out"]
        kinds = [ps.subject_kind(s) for s in plan["subjects"]]
        assert "left_out" not in kinds


# --- the fingerprint sees a mode change as a different job ----------------------------------

def test_the_same_faces_in_a_different_mode_are_a_different_job(tmp_path):
    """The mode changes who the subjects are, so a job folder planned on one mode has to refuse
    a plan made on another - same faces, same plates, same everything else."""
    job = ps.job_dir_for(tmp_path, "group shot")
    ps.write_plan(job, _plan(mode="no_reference"))

    lines = ps.fingerprint_difference(ps.read_plan(job), _plan(mode="with_still"))
    assert [line.split(":")[0] for line in lines] == ["unmatched_faces"]

    with pytest.raises(ValueError) as exc:
        ps.check_pairing_inputs(ps.read_plan(job), _plan(mode="with_still"), job, "act on it")
    assert "unmatched_faces" in str(exc.value)
    assert "no_reference" in str(exc.value) and "with_still" in str(exc.value)


def test_the_same_mode_into_the_same_job_folder_is_a_resume(tmp_path):
    job = ps.job_dir_for(tmp_path, "group shot")
    ps.write_plan(job, _plan(mode="with_still"))

    assert ps.fingerprint_difference(ps.read_plan(job), _plan(mode="with_still")) == []
    ps.check_pairing_inputs(ps.read_plan(job), _plan(mode="with_still"), job, "act on it")


def test_the_mode_and_what_was_left_out_survive_the_job_folder(tmp_path):
    plan = _plan()
    job = ps.job_dir_for(tmp_path, "group shot")
    ps.write_plan(job, plan)

    back = ps.read_plan(job)
    assert back == plan
    assert back["inputs"]["unmatched_faces"] == "leave"
    assert [s["clip_face"] for s in back["left_out"]] == [1]
    assert ps.unmatched_lines(back) == ps.unmatched_lines(plan)
    assert ps.subject_kind_counts(back) == ps.subject_kind_counts(plan)
