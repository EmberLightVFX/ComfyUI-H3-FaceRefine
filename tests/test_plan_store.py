"""Pairing, tested with synthetic boxes: no ComfyUI, no torch, no GPU, no weights.

Every test names an outcome an editor would recognise - "the still's second face is the
clip's leftmost face's subject reference", "the bystander at index 2 is excluded" - and
never asserts how a helper does its arithmetic.
"""

import json
import os

import pytest

import plan_store as ps


# --- pairing: positional, measured at the match frame -------------------------------

# --- replacing a plan: clear the plan node's own artifacts, keep pass stages -----------------

def _seed_job(tmp_path, first_plan="OLD"):
    job = os.path.join(str(tmp_path), "job")
    os.makedirs(os.path.join(job, "passes", "subject_0000"), exist_ok=True)

    def touch(*rel, content="x"):
        path = os.path.join(job, *rel)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return path

    touch(ps.PLAN_FILENAME, content=first_plan)
    touch(ps.PREVIEW_FILENAME)
    touch("subject_0.png")
    touch("subject_1.png")
    touch("clip_0.png")
    touch("manifest.json", content="MANIFEST")
    touch("passes", "subject_0000", "crops.json", content="CROPS")
    touch("subject_notes.txt", content="keep")   # starts with subject_ but is not a crop
    return job


def test_replacing_a_plan_backs_it_up_and_removes_only_the_plan_node_artifacts(tmp_path):
    job = _seed_job(tmp_path)

    res = ps.clear_plan_artifacts_for_replace(job)

    assert res["existed"] is True
    assert res["prev"] == ps.PLAN_PREV_FILENAME
    assert sorted(res["cleared"]) == ["clip_0.png", ps.PREVIEW_FILENAME,
                                      "subject_0.png", "subject_1.png"]
    # The old plan is kept under the backup name, so a replacement is recoverable.
    with open(os.path.join(job, ps.PLAN_PREV_FILENAME), encoding="utf-8") as fh:
        assert fh.read() == "OLD"
    # plan.json is gone, so the next plan-node run into this name writes fresh (nothing to guard).
    assert not os.path.isfile(os.path.join(job, ps.PLAN_FILENAME))
    # Pass stages are left alone: their own nodes' guards handle them, not the replace.
    assert os.path.isfile(os.path.join(job, "manifest.json"))
    with open(os.path.join(job, "passes", "subject_0000", "crops.json"), encoding="utf-8") as fh:
        assert fh.read() == "CROPS"
    # A file that merely starts with subject_ but is not an <int>.png crop survives.
    assert os.path.isfile(os.path.join(job, "subject_notes.txt"))


def test_replacing_a_plan_twice_keeps_only_the_one_previous_plan(tmp_path):
    job = os.path.join(str(tmp_path), "job")
    os.makedirs(job)
    with open(os.path.join(job, ps.PLAN_FILENAME), "w", encoding="utf-8") as fh:
        fh.write("A")
    ps.clear_plan_artifacts_for_replace(job)          # A -> plan_prev
    with open(os.path.join(job, ps.PLAN_FILENAME), "w", encoding="utf-8") as fh:
        fh.write("B")

    res = ps.clear_plan_artifacts_for_replace(job)    # B -> plan_prev, dropping A

    assert res["prev"] == ps.PLAN_PREV_FILENAME
    with open(os.path.join(job, ps.PLAN_PREV_FILENAME), encoding="utf-8") as fh:
        assert fh.read() == "B"


def test_clear_plan_artifacts_reports_a_missing_job_folder(tmp_path):
    assert ps.clear_plan_artifacts_for_replace(os.path.join(str(tmp_path), "nope")) == {
        "existed": False, "cleared": [], "prev": None}


# --- re-running the plan node: the reuse flag and the unapplied-change summary ----------------

def _two_plans(excluded=()):
    clip = [[100, 100, 200, 220], [400, 120, 520, 260]]
    return ps.build_subject_plan(clip, [0.9, 0.9], clip, [0.9, 0.9],
                                 (1000, 1000), (1000, 1000), excluded=excluded)


def test_resume_flag_round_trips_and_names_unapplied_changes(tmp_path):
    job = os.path.join(str(tmp_path), "job")
    os.makedirs(job)
    assert ps.read_resume_state(job) == {} and ps.plan_is_resumed(job) is False

    ps.mark_resumed(job, ["excluded_faces: the node now has [3], the saved plan used []"])
    assert ps.plan_is_resumed(job) is True
    state = ps.read_resume_state(job)
    assert state["reused"] is True and len(state["changed"]) == 1
    assert "excluded_faces" in state["changed"][0]

    ps.clear_resumed_flag(job)
    assert ps.read_resume_state(job) == {} and ps.plan_is_resumed(job) is False


def test_changed_widget_lines_names_only_widget_inputs_a_run_did_not_apply():
    saved = _two_plans()
    now = _two_plans(excluded=(0,))
    lines = ps.changed_widget_lines(saved, now)
    assert any("excluded_faces" in line for line in lines)
    # The saved plan's subject set had 2, the new exclusions would make fewer - said as a count.
    assert any("subject set" in line for line in lines)


def test_changed_widget_lines_is_empty_when_the_run_matches_the_saved_plan():
    assert ps.changed_widget_lines(_two_plans(), _two_plans()) == []


def test_identical_plates_pair_by_position():
    """Two plates framed identically: each face finds the face in the same place."""
    clip = [[100, 100, 200, 220], [400, 120, 520, 260]]
    still = [[100, 100, 200, 220], [400, 120, 520, 260]]

    got = ps.pair_faces(clip, still, (1000, 1000), (1000, 1000), 0.05)

    assert [(p["target"], p["still"]) for p in got["pairs"]] == [(0, 0), (1, 1)]
    assert got["unpaired_target"] == []
    assert got["unpaired_still"] == []
    assert got["scale"] == pytest.approx(1.0)


def test_still_at_a_different_size_pairs_and_reports_the_factor():
    """A 4K still against a 1080p clip: boxes scale proportionally, factor reported."""
    clip = [[100, 100, 200, 220], [400, 120, 520, 260]]
    still = [[200, 200, 400, 440], [800, 240, 1040, 520]]

    got = ps.pair_faces(clip, still, (1000, 1000), (2000, 2000), 0.05)

    assert [(p["target"], p["still"]) for p in got["pairs"]] == [(0, 0), (1, 1)]
    assert got["unpaired_target"] == [] and got["unpaired_still"] == []
    # still -> clip: a still coordinate halved lands on the clip's frame.
    assert got["scale"] == pytest.approx(0.5)


def test_a_larger_tolerance_is_widened_against_the_frame_not_the_box():
    """match_tolerance is a fraction of the TARGET diagonal, so it means the same thing
    at every resolution."""
    clip = [[100, 100, 200, 200]]
    still = [[128, 128, 228, 228]]        # centres 39.6px apart

    tight = ps.pair_faces(clip, still, (1000, 1000), (1000, 1000), 0.02)   # 28.3px
    loose = ps.pair_faces(clip, still, (1000, 1000), (1000, 1000), 0.05)   # 70.7px

    assert tight["pairs"] == []
    assert tight["unpaired_target"] == [0] and tight["unpaired_still"] == [0]
    assert [(p["target"], p["still"]) for p in loose["pairs"]] == [(0, 0)]
    assert loose["tolerance_px"] == pytest.approx(0.05 * (1000 ** 2 + 1000 ** 2) ** 0.5)


def test_faces_far_apart_in_a_small_frame_stay_unpaired():
    """A bystander whose nearest still face is out of reach is not paired to it."""
    clip = [[10, 10, 70, 70], [200, 10, 260, 70]]
    still = [[10, 10, 70, 70]]

    got = ps.pair_faces(clip, still, (300, 300), (300, 300), 0.2)

    assert [(p["target"], p["still"]) for p in got["pairs"]] == [(0, 0)]
    assert got["unpaired_target"] == [1]


def test_centres_within_reach_but_no_overlap_are_not_a_pair():
    """The tolerance alone is not enough: two faces have to be the same face's two
    detections, which means their boxes have to touch."""
    clip = [[0, 0, 200, 200], [300, 0, 500, 200]]      # disjoint, centres 300px apart
    still = [[0, 0, 200, 200], [600, 0, 800, 200]]     # centre 300px from clip face 1

    got = ps.pair_faces(clip, still, (1000, 1000), (1000, 1000), 0.25)   # reach 353.6px

    assert [(p["target"], p["still"]) for p in got["pairs"]] == [(0, 0)]
    assert got["unpaired_target"] == [1] and got["unpaired_still"] == [1]


def test_each_face_pairs_with_its_own_nearest_face_not_whichever_comes_first():
    """The still detected the two faces in the opposite order; pairing still puts each
    person with themselves rather than with whoever shares their index."""
    clip = [[80, 80, 160, 180], [180, 80, 260, 180]]           # centres 120, 220
    still = [[175, 80, 255, 180], [85, 80, 165, 180]]          # centres 215, 125

    got = ps.pair_faces(clip, still, (400, 400), (400, 400), 0.2)

    assert [(p["target"], p["still"]) for p in got["pairs"]] == [(0, 1), (1, 0)]
    assert got["unpaired_target"] == [] and got["unpaired_still"] == []


def test_different_aspect_ratios_refuse_to_pair():
    """Pairing is positional, so a differently-framed plate would put every box in the
    wrong place. It raises rather than guessing."""
    clip = [[100, 100, 200, 200]]
    still = [[100, 100, 200, 200]]

    with pytest.raises(ValueError) as exc:
        ps.pair_faces(clip, still, (1920, 1080), (1024, 768), 0.05)

    message = str(exc.value)
    assert "aspect" in message
    assert "1920x1080" in message and "1024x768" in message
    assert "1.778" in message and "1.333" in message


def test_a_plate_with_no_faces_pairs_with_nothing():
    """No faces is a legitimate detection result at one frame, not a crash: the plan
    decides what to do about it."""
    got = ps.pair_faces([[10, 10, 50, 60]], [], (1000, 1000), (1000, 1000), 0.05)

    assert got["pairs"] == []
    assert got["unpaired_target"] == [0] and got["unpaired_still"] == []


# --- the rectangle a subject reference is cut with ---------------------------------

def test_the_reference_crop_uses_the_rectangle_the_tracker_would_use():
    """A subject reference has to be framed the way H3 will be asked to generate, or the
    detail it carries lands somewhere else in the crop."""
    # A 100px tall face at crop_factor 2.5 gives a 250px square, centred on the face.
    assert ps.tracker_rect([400, 400, 500, 500], (1000, 1000)) == [325.0, 325.0,
                                                                  250.0, 250.0]


def test_a_face_near_the_edge_slides_the_crop_inside_the_frame():
    """The crop is never allowed to hang off the plate, exactly as the tracker clamps it."""
    assert ps.tracker_rect([20, 20, 120, 120], (1000, 1000)) == [0.0, 0.0, 250.0, 250.0]


def test_a_face_filling_the_frame_cannot_produce_a_crop_bigger_than_the_frame():
    rect = ps.tracker_rect([0, 0, 900, 900], (1000, 1000), crop_factor=4.0)
    assert rect == [0.0, 0.0, 1000.0, 1000.0]


def test_the_reference_crop_is_scaled_into_the_still_at_native_resolution():
    """A 4K still against a 2K clip: the same scene region is twice as many still pixels,
    and that is what gets cut - no resampling."""
    assert ps.crop_rect_for_reference([325.0, 325.0, 250.0, 250.0],
                                      (2000, 2000), (1000, 1000)) == [650, 650, 500, 500]


def test_a_smaller_still_gives_a_smaller_crop_and_invents_no_pixels():
    """A soft, small still cannot be asked for the clip's pixel count: that would dress
    mush up as detail. The cut stays at the still's own pixels."""
    crop = ps.crop_rect_for_reference([100.0, 100.0, 320.0, 320.0], (500, 500),
                                      (1000, 1000))
    assert crop == [50, 50, 160, 160]
    assert crop[2] < 320 and crop[3] < 320      # not the clip-sized rect
    assert crop[2] <= 500 and crop[3] <= 500    # never more than the still holds


def test_a_rect_wider_than_the_still_is_cut_down_to_the_still():
    """A crop asking for more than the still has can only be answered by inventing pixels,
    so it is clamped to the still instead."""
    assert ps.crop_rect_for_reference([0.0, 0.0, 1500.0, 1500.0], (500, 500),
                                      (1000, 1000)) == [0, 0, 500, 500]


def test_the_reference_crop_stays_inside_the_still():
    x, y, w, h = ps.crop_rect_for_reference([900.0, 900.0, 300.0, 300.0], (500, 500),
                                            (1000, 1000))
    assert (x, y, w, h) == (350, 350, 150, 150)
    assert x >= 0 and y >= 0 and x + w <= 500 and y + h <= 500


def test_the_reference_crop_refuses_a_differently_framed_still():
    with pytest.raises(ValueError) as exc:
        ps.crop_rect_for_reference([0.0, 0.0, 100.0, 100.0], (1024, 768), (1920, 1080))
    assert "aspect" in str(exc.value)



# --- the subject plan --------------------------------------------------------------

CLIP = (1000, 1000)
STILL = (2000, 2000)
# Three faces on the clip, left to right; the soft-faced middle one has no partner.
CLIP_BOXES = [[100, 100, 200, 220], [400, 400, 500, 500], [800, 120, 900, 240]]
CLIP_CONFS = [0.9, 0.8, 0.7]
# The still found the outer two, in its own detection order, at twice the clip's scale.
STILL_BOXES = [[1600, 240, 1800, 480], [200, 200, 400, 440]]
STILL_CONFS = [0.6, 0.95]


def _plan(**over):
    args = dict(target_boxes=CLIP_BOXES, target_confs=CLIP_CONFS,
                still_boxes=STILL_BOXES, still_confs=STILL_CONFS,
                target_size=CLIP, still_size=STILL, match_frame=12,
                subject_source="union", tolerance=0.05, excluded=[], max_subjects=0,
                unmatched_faces=ps.DEFAULT_UNMATCHED_FACES)
    args.update(over)
    return ps.build_subject_plan(**args)


def test_union_makes_a_subject_of_every_face_on_either_plate():
    """Nothing that was detected is dropped silently, and every subject says whether it has a
    subject reference. `unmatched_faces=no_reference` here because the shipped default leaves
    an unpaired face out of the subject set entirely (#8, test_unmatched_faces.py)."""
    plan = _plan(unmatched_faces="no_reference")

    assert plan["subject_count"] == 3
    assert [s["clip_face"] for s in plan["subjects"]] == [0, 1, 2]
    # The still found the left and right faces; the middle one is on its own.
    assert [s["has_reference"] for s in plan["subjects"]] == [True, False, True]
    assert [s["still_face"] for s in plan["subjects"]] == [1, None, 0]
    assert plan["unpaired_target"] == [1]
    assert plan["unpaired_still"] == []


def test_subjects_are_ordered_left_to_right_at_the_match_frame():
    """Passes composite in this order, so it is the order the user reads off the preview."""
    plan = _plan(target_boxes=list(reversed(CLIP_BOXES)),
                 target_confs=list(reversed(CLIP_CONFS)), unmatched_faces="no_reference")

    # clip_face is the detection order; the SUBJECTS themselves run left to right.
    assert [s["clip_face"] for s in plan["subjects"]] == [2, 1, 0]
    assert [s["index"] for s in plan["subjects"]] == [0, 1, 2]


def test_target_clip_source_takes_subjects_from_the_clip_only():
    """When the clip detected better, a face only the still saw does not become a
    subject - there is nothing in the clip to hold it."""
    plan = _plan(subject_source="target_clip", unmatched_faces="no_reference")

    assert [s["clip_face"] for s in plan["subjects"]] == [0, 1, 2]
    assert [s["has_reference"] for s in plan["subjects"]] == [True, False, True]


def test_reference_still_source_takes_subjects_from_the_still():
    """When the still detected better, its faces seed the subjects; the still-only face
    still becomes one, with no clip box to track yet."""
    clip = [[100, 100, 200, 220], [300, 300, 400, 400]]
    still = [[100, 100, 200, 220], [600, 600, 700, 720]]

    plan = ps.build_subject_plan(clip, [0.9, 0.8], still, [0.9, 0.9], (1000, 1000),
                                 (1000, 1000), 0, "reference_still", 0.05, [], 0)

    assert [s["still_face"] for s in plan["subjects"]] == [0, 1]
    assert plan["subjects"][0]["clip_face"] == 0
    assert plan["subjects"][1]["clip_box"] is None
    assert [s["has_reference"] for s in plan["subjects"]] == [True, True]
    # The clip face the still never saw is not a subject in this mode.
    assert plan["unpaired_target"] == [1]


def test_the_default_subject_source_is_union():
    import inspect
    signature = inspect.signature(ps.build_subject_plan).parameters
    assert signature["subject_source"].default == "union"



def test_excluded_faces_never_become_subjects():
    """The bystander at index 1 is switched off, and keeps its original pixels."""
    plan = _plan(excluded=[1])

    assert [s["clip_face"] for s in plan["subjects"]] == [0, 2]
    assert plan["excluded_faces"] == [1]
    assert plan["inputs"]["excluded_faces"] == [1]


def test_excluding_a_face_drops_the_still_face_paired_to_it():
    """Switching a person off switches off their subject reference too, whichever plate the
    subjects are seeded from."""
    plan = _plan(subject_source="reference_still", excluded=[2])

    assert [s["still_face"] for s in plan["subjects"]] == [1]


def test_an_exclusion_that_names_no_face_found_is_an_error():
    """A stale exclusion is worse than an error: it looks like the bystander was switched
    off while the numbering has since moved."""
    with pytest.raises(ValueError) as exc:
        _plan(excluded=[7])
    assert "7" in str(exc.value) and "3" in str(exc.value)


def test_the_excluded_faces_widget_is_read_as_a_typed_list():
    """excluded_faces is a STRING widget so it needs no front end; the numbers are the ones
    printed on the pairing preview."""
    assert ps.parse_excluded_faces("2, 5") == [2, 5]
    assert ps.parse_excluded_faces("") == []
    assert ps.parse_excluded_faces(None) == []
    with pytest.raises(ValueError) as exc:
        ps.parse_excluded_faces("2, bystander")
    assert "bystander" in str(exc.value)


# --- an exclusion can name a face on EITHER plate (#11 review, P4; spec story 6) ------

def test_an_s_index_is_a_reference_still_face_and_a_bare_number_is_a_clip_face():
    """Two plates, two numberings, and one widget. `s2` is the still's face 2 as numbered on the
    pairing preview; a bare 2 stays the clip's face 2, exactly as it always was."""
    assert ps.parse_excluded_faces("2, s3, s1, 0") == [2, "s3", "s1", 0]
    assert ps.parse_excluded_faces("S1") == ["s1"]
    assert ps.parse_excluded_faces(["s0", 1]) == ["s0", 1]
    with pytest.raises(ValueError) as exc:
        ps.parse_excluded_faces("ss2")
    assert "ss2" in str(exc.value)


def test_an_s_index_switches_off_a_face_only_the_reference_still_saw():
    """Spec story 6: `union` seeds every face on either plate, so a bystander who is in the photo
    and never in the clip became a subject that could not be switched off. `s2` switches it off."""
    still = STILL_BOXES + [[1800, 1800, 1900, 1900]]      # a face the clip never saw

    kept = _plan(still_boxes=still, unmatched_faces="no_reference")
    off = _plan(still_boxes=still, unmatched_faces="no_reference", excluded=["s2"])

    assert 2 in [s["still_face"] for s in kept["subjects"]]
    assert 2 not in [s["still_face"] for s in off["subjects"]]
    assert off["subject_count"] == kept["subject_count"] - 1
    # Both kinds land in the plan and in the fingerprint, and the still one stays a still one.
    assert off["excluded_faces"] == ["s2"]
    assert off["inputs"]["excluded_faces"] == ["s2"]


def test_an_s_index_on_a_paired_face_drops_the_pair_it_belongs_to():
    """A switched-off detection cannot pair, so the person it was matched to falls to whatever
    `unmatched_faces` says - which is what switching a face off has to mean."""
    plan = _plan(excluded=["s0"], unmatched_faces="no_reference")

    assert [(s["clip_face"], s["still_face"]) for s in plan["subjects"]] == [(0, 1), (1, None),
                                                                            (2, None)]
    assert plan["subjects"][2]["ref_kind"] == ps.REF_NONE


def test_an_out_of_range_index_names_the_plate_it_was_checked_against():
    """`s9` against a two-face still and a bare 9 against a three-face clip are different
    mistakes; each message says which plate was counted and how to name the other one."""
    with pytest.raises(ValueError) as exc:
        _plan(excluded=["s2"])
    message = str(exc.value)
    assert "reference still face 2" in message and "0-1" in message

    with pytest.raises(ValueError) as exc:
        _plan(excluded=["s-1"])
    assert "reference still face -1" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        _plan(excluded=[9])
    assert "target clip face 9" in str(exc.value)
    assert "s9" in str(exc.value)


def test_a_bare_index_still_means_the_target_clip_exactly_as_before():
    """Nothing about the released behaviour moves: the same exclusions, the same plan."""
    plan = _plan(excluded=[1])

    assert [s["clip_face"] for s in plan["subjects"]] == [0, 2]
    assert plan["excluded_faces"] == [1] and plan["inputs"]["excluded_faces"] == [1]
    assert _plan(excluded=[1])["subjects"] == plan["subjects"]


def test_max_subjects_is_recorded_and_applied():
    """#10 reversed this test. The cap is not decoration: it truncates the ordered subject set
    here, so every downstream stage - the subject combo, `next`, the manifest, the composite
    gate - sees the same job rather than three different ones. Cap 2 of 3 keeps the two leftmost
    faces and drops the third by name; `no_reference` keeps the unpaired face in the set so the
    order is the plain left-to-right one."""
    plan = _plan(max_subjects=2, unmatched_faces="no_reference")

    assert plan["inputs"]["max_subjects"] == 2
    assert plan["subject_count"] == 2
    assert [s["clip_face"] for s in plan["subjects"]] == [0, 1]
    assert [s["clip_face"] for s in plan["capped_out"]] == [2]
    assert [s["index"] for s in plan["capped_out"]] == [None]


def test_an_unknown_subject_source_is_an_error():
    with pytest.raises(ValueError) as exc:
        _plan(subject_source="both")
    assert "union" in str(exc.value) and "reference_still" in str(exc.value)


def test_a_subject_reference_is_cut_where_the_clip_crop_would_land():
    """The tracker's framing, in the still's own pixels: for a 2x still that is a 600px cut
    sitting where the clip's 300px crop sits. `no_reference` keeps the unpaired face in the
    subject set so this test can see that it has nothing to cut."""
    plan = _plan(unmatched_faces="no_reference")

    assert plan["subjects"][0]["ref_crop"] == [0, 20, 600, 600]
    assert plan["subjects"][2]["ref_crop"] == [1400, 60, 600, 600]
    # No partner means no subject reference, so there is nothing to cut.
    assert plan["subjects"][1]["ref_crop"] is None


def test_the_plan_carries_a_version_and_the_pairing_inputs():
    plan = _plan(excluded=[1], max_subjects=4)

    assert plan["version"] == ps.PLAN_VERSION
    assert plan["inputs"] == {"target_size": [1000, 1000], "still_size": [2000, 2000],
                              "match_frame": 12, "subject_source": "union",
                              "match_tolerance": 0.05, "excluded_faces": [1],
                              "max_subjects": 4,
                              "unmatched_faces": ps.DEFAULT_UNMATCHED_FACES,
                              # The rectangle the subject references were cut with (#11 review).
                              "crop_factor": ps.DEFAULT_CROP_FACTOR,
                              # Manual pairings from the pairing panel (#12); empty by default.
                              "pair_overrides": []}
    assert plan["scale"] == pytest.approx(0.5)
    # The match-frame measurements stay in the plan so a later pass can rebuild one
    # subject's pick without re-detecting anything.
    assert plan["target_boxes"] == [[float(v) for v in b] for b in CLIP_BOXES]
    assert plan["still_boxes"] == [[float(v) for v in b] for b in STILL_BOXES]
    assert plan["pairs"] == [{"target": 0, "still": 1, "distance": 0.0},
                             {"target": 2, "still": 0, "distance": 0.0}]


def test_every_subject_carries_what_a_pass_needs():
    plan = _plan(unmatched_faces="no_reference")
    want = {"index", "clip_face", "clip_box", "still_face", "still_box",
            "has_reference", "ref_crop", "ref_kind", "ref_file", "paired"}
    for subject in plan["subjects"]:
        assert want <= set(subject)
    assert plan["subject_count"] == len(plan["subjects"])
    # What kind of reference each subject ended up with, so a pass never has to guess (#8).
    assert [s["ref_kind"] for s in plan["subjects"]] == ["crop", "none", "crop"]



def test_a_still_only_subject_is_ordered_by_where_it_sits_in_the_clip():
    """A face only the still saw joins the left-to-right order at its position in the
    clip, not at its pixel position in a still of a different size."""
    clip = [[250, 100, 350, 220]]          # centre x 300 in the clip
    still = [[300, 100, 400, 220]]         # pixel centre x 350, but clip-space centre 175

    plan = ps.build_subject_plan(clip, [0.9], still, [0.9], (1000, 1000), (2000, 2000),
                                 0, "union", 0.05, [], 0, unmatched_faces="no_reference")

    assert [s["clip_face"] for s in plan["subjects"]] == [None, 0]
    assert [s["still_face"] for s in plan["subjects"]] == [0, None]


# --- the job folder ----------------------------------------------------------------

def test_the_job_folder_lives_under_the_output_folder(tmp_path):
    assert ps.job_dir_for(tmp_path, "group shot") == str(tmp_path / "group shot")


def test_a_job_name_cannot_escape_the_output_folder(tmp_path):
    """job_name is a text widget, so it is not allowed to walk out of the output tree."""
    for name in ("../../outside", "..", "/abs/path"):
        made = ps.job_dir_for(tmp_path, name)
        assert os.path.dirname(made) == str(tmp_path)
        assert "outside" not in os.path.dirname(made)


def test_a_blank_job_name_is_refused(tmp_path):
    with pytest.raises(ValueError):
        ps.job_dir_for(tmp_path, "   ")


def test_the_plan_round_trips_through_the_job_folder(tmp_path):
    """Whatever the plan decided survives being written and read back, unchanged, by a
    later pass that has neither plate in hand."""
    plan = _plan(excluded=[1])
    job = ps.job_dir_for(tmp_path, "group shot")

    written = ps.write_plan(job, plan)
    assert os.path.basename(written) == ps.PLAN_FILENAME
    assert ps.read_plan(job) == plan

def test_an_exclusion_of_either_plate_round_trips_and_stays_its_own_job(tmp_path):
    """`s` tokens are JSON like any other, and the fingerprint tells an `s2` job from a 2 job -
    otherwise a resume would read a still-side switch-off as a clip-side one."""
    mixed = _plan(excluded=["s2", 1], still_boxes=STILL_BOXES + [[1800, 1800, 1900, 1900]],
                  unmatched_faces="no_reference")
    job = ps.job_dir_for(tmp_path, "bystanders")
    ps.write_plan(job, mixed)
    back = ps.read_plan(job)

    assert back == mixed
    assert back["excluded_faces"] == [1, "s2"] == back["inputs"]["excluded_faces"]
    assert ps.pairing_fingerprint(back) == ps.pairing_fingerprint(mixed)
    assert ps.fingerprint_difference(back, _plan(excluded=[2])) != []



def test_a_written_plan_carries_a_version_even_when_nobody_mentioned_one(tmp_path):
    path = ps.write_plan(str(tmp_path), {"subjects": []})
    with open(path, encoding="utf-8") as fh:
        assert json.load(fh)["version"] == ps.PLAN_VERSION


def test_a_missing_plan_names_the_job_that_was_looked_for(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        ps.read_plan(str(tmp_path / "group shot"))
    assert "group shot" in str(exc.value)


def test_a_plan_folder_is_not_forgotten_when_the_inputs_change(tmp_path):
    """The plan is the record a later pass compares against, so the pairing inputs have to
    be readable straight off the file."""
    plan = _plan(tolerance=0.11, subject_source="target_clip", max_subjects=3)
    job = ps.job_dir_for(tmp_path, "job")
    ps.write_plan(job, plan)

    back = ps.read_plan(job)
    assert back["inputs"]["match_tolerance"] == 0.11
    assert back["inputs"]["subject_source"] == "target_clip"
    assert back["inputs"]["max_subjects"] == 3


# --- the fingerprint covers the pairing EVIDENCE, not only the widgets (#11 review, P2) --

def _with_run(plan, detector="face_yolov8m.pt", confidence=0.35):
    """What the plan node writes into `run`, which is where the detector lives."""
    plan["run"] = dict(plan.get("run") or {}, job_name="group shot", frames=4,
                       detector=detector, confidence=confidence)
    return plan


def test_a_different_still_of_the_same_size_refuses_rather_than_resumes(tmp_path):
    """A nudged still keeps the SAME subject indices and still pairs everyone, so comparing the
    indices alone would call this the same job. It is not: the passes were cut from a pairing
    that measured different boxes."""
    job = ps.job_dir_for(tmp_path, "group shot")
    ps.write_plan(job, _with_run(_plan()))
    moved = _with_run(_plan(still_boxes=[[1600, 240, 1800, 480], [210, 210, 410, 450]]))

    # Same subjects, same widgets - only the measured boxes moved.
    assert ([s["index"] for s in moved["subjects"]]
            == [s["index"] for s in ps.read_plan(job)["subjects"]])
    lines = ps.fingerprint_difference(ps.read_plan(job), moved)
    assert any(line.startswith("still_boxes: ") for line in lines)
    assert "first difference at face 1" in "\n".join(lines)
    # The box that moved is named; the boxes that did not are not dumped at the reader.
    assert "[210.0, 210.0, 410.0, 450.0]" in "\n".join(lines)
    assert "[1600.0, 240.0, 1800.0, 480.0]" not in "\n".join(lines)


def test_a_different_detector_or_confidence_refuses_naming_it(tmp_path):
    """Another detector can pair different faces into the same indices, so the run record is
    part of the fingerprint and not a detail of the report."""
    job = ps.job_dir_for(tmp_path, "group shot")
    ps.write_plan(job, _with_run(_plan()))

    lines = ps.fingerprint_difference(ps.read_plan(job),
                                      _with_run(_plan(), detector="face_yolov8s.pt"))
    assert any(line.startswith("detector: ") for line in lines)

    lines = ps.fingerprint_difference(ps.read_plan(job),
                                      _with_run(_plan(), confidence=0.25))
    assert any(line.startswith("confidence: ") for line in lines)


def test_an_identical_pairing_resumes_clean(tmp_path):
    """Same plates, same boxes, same detector, same cap: re-running the plan into its own job
    folder is a resume, and has to stay possible or no job could ever be re-run."""
    stored = _with_run(_plan(excluded=[1], max_subjects=2))
    job = ps.job_dir_for(tmp_path, "group shot")
    ps.write_plan(job, stored)

    current = _with_run(_plan(excluded=[1], max_subjects=2))
    assert ps.fingerprint_difference(ps.read_plan(job), current) == []
    ps.check_pairing_inputs(ps.read_plan(job), current, job, "act on it")


def test_the_crop_factor_the_references_were_cut_at_is_part_of_the_fingerprint(tmp_path):
    """A job folder holds subject references cut at one factor; a plan at another factor would
    write new references beside passes cut from the old ones."""
    job = ps.job_dir_for(tmp_path, "group shot")
    ps.write_plan(job, _plan())

    assert ps.fingerprint_difference(ps.read_plan(job), _plan()) == []
    lines = ps.fingerprint_difference(ps.read_plan(job), _plan(crop_factor=1.5))
    assert any(line.startswith("crop_factor: ") for line in lines)

    # A version 2 folder recorded no factor at all, and used the shipped default - which is what
    # an untouched plan today also uses, so that folder is still the same job.
    old = _plan()
    old["version"] = 2
    del old["inputs"]["crop_factor"]
    ps.write_plan(job, old)
    assert ps.fingerprint_difference(ps.read_plan(job), _plan()) == []


# --- the tracker's crop_factor, asked of the plan (#11 review, P1) --------------------

def test_the_plan_cuts_subject_references_at_the_crop_factor_it_is_given():
    """The tracker takes `crop_factor` from a widget, so the plan cannot assume 2.5: a subject
    reference framed at the wrong factor puts the detail it carries somewhere else in the crop."""
    default = _plan()
    tight = _plan(crop_factor=1.5)

    # clip face 0 is 120px tall: at 1.5 that is an 180px square centred on (150, 160), where
    # the shipped default gives a 300px square pushed up against the frame's top-left.
    assert ps.subject_crop_rect(default, default["subjects"][0]) == [0.0, 10.0, 300.0, 300.0]
    assert ps.subject_crop_rect(tight, tight["subjects"][0]) == [60.0, 70.0, 180.0, 180.0]
    assert default["subjects"][0]["ref_crop"] == [0, 20, 600, 600]
    assert tight["subjects"][0]["ref_crop"] == [120, 140, 360, 360]
    # The factor is recorded in the plan, so the passes and the preview read the rectangle
    # back out of the job rather than re-guessing it.
    assert tight["inputs"]["crop_factor"] == 1.5
    assert default["inputs"]["crop_factor"] == ps.DEFAULT_CROP_FACTOR


def test_a_plan_without_the_new_argument_is_todays_plan_byte_for_byte():
    """The default is the tracker's shipped default, so nothing that exists today moves."""
    assert json.dumps(_plan(), sort_keys=True, default=str) == json.dumps(
        _plan(crop_factor=ps.DEFAULT_CROP_FACTOR), sort_keys=True, default=str)
    assert _plan()["subjects"][0]["ref_crop"] == [0, 20, 600, 600]


def test_a_crop_factor_that_cannot_crop_anything_is_refused():
    """A factor below 1 is smaller than the face, which is not a crop the tracker could make."""
    with pytest.raises(ValueError) as exc:
        _plan(crop_factor=0.0)
    assert "crop_factor" in str(exc.value)


# --- manual pairing: the pairing panel's re-pairing editor (#12) ----------------------------

def test_pair_overrides_repoint_one_faces_partner_to_another():
    """Two faces per plate, auto-paired 0-0 and 1-1. Pointing clip face 0's reference at still
    face 1 moves that still face under it and frees the face it used to share, so clip face 1
    now has no partner and falls to whatever unmatched_faces says."""
    clip = [[100, 100, 200, 220], [400, 100, 500, 220]]
    still = [[100, 100, 200, 220], [400, 100, 500, 220]]
    auto = ps.pair_faces(clip, still, (1000, 1000), (1000, 1000), 0.05)
    assert [(p["target"], p["still"]) for p in auto["pairs"]] == [(0, 0), (1, 1)]

    fixed = ps.apply_pair_overrides(auto, ps.parse_pair_overrides("0=s1"), 2, 2, 0)

    assert [(p["target"], p["still"]) for p in fixed["pairs"]] == [(0, 1)]
    assert fixed["unpaired_target"] == [1] and fixed["unpaired_still"] == [0]
    assert fixed["pairs"][0]["manual"] is True


def test_pair_overrides_switch_off_keeps_the_face_but_drops_its_partner():
    """`0=none` is not an exclusion: the face stays a candidate, only its partner is cleared so
    it falls to unmatched_faces, while every other pair is left exactly as the detector made it."""
    clip = [[100, 100, 200, 220], [400, 100, 500, 220]]
    still = [[100, 100, 200, 220], [400, 100, 500, 220]]
    auto = ps.pair_faces(clip, still, (1000, 1000), (1000, 1000), 0.05)
    fixed = ps.apply_pair_overrides(auto, ps.parse_pair_overrides("0=none"), 2, 2, 0)
    assert [(p["target"], p["still"]) for p in fixed["pairs"]] == [(1, 1)]
    assert fixed["unpaired_target"] == [0] and fixed["unpaired_still"] == [0]


def test_pair_overrides_canonicalise_so_either_direction_compares_equal():
    """`0=s1` and `s1=0` are one pairing written from either plate; the fingerprint must see one
    answer so a re-plan is not refused for spelling."""
    assert ps.pair_override_tokens(ps.parse_pair_overrides("0=s1")) == ["0=s1"]
    assert ps.pair_override_tokens(ps.parse_pair_overrides("s1=0")) == ["0=s1"]
    assert ps.pair_override_tokens(ps.parse_pair_overrides("s1=none")) == ["s1=none"]
    assert ps.pair_override_tokens(ps.parse_pair_overrides("2=none")) == ["2=none"]


def test_pair_overrides_survive_the_plan_and_the_fingerprint_calls_a_change_a_different_job(tmp_path):
    """A corrected pairing is remembered in plan.json, and a folder planned without it cannot be
    resumed with it - that would composite two pairings into one clip."""
    job = os.path.join(str(tmp_path), "job")
    ps.write_plan(job, _plan())
    assert _plan()["inputs"]["pair_overrides"] == []
    assert _plan(pair_overrides="0=s1")["inputs"]["pair_overrides"] == ["0=s1"]

    lines = ps.fingerprint_difference(ps.read_plan(job), _plan(pair_overrides="0=s1"))
    assert any(line.startswith("pair_overrides:") for line in lines)

    ps.write_plan(job, _plan(pair_overrides="0=s1"))
    assert ps.fingerprint_difference(ps.read_plan(job), _plan(pair_overrides="0=s1")) == []


def test_an_out_of_range_or_illegal_pair_override_is_refused():
    clip = [[100, 100, 200, 220]]
    still = [[100, 100, 200, 220]]
    auto = ps.pair_faces(clip, still, (1000, 1000), (1000, 1000), 0.05)
    with pytest.raises(ValueError) as exc:
        ps.apply_pair_overrides(auto, ps.parse_pair_overrides("0=s5"), 1, 1, 0)
    assert "reference still" in str(exc.value)
    with pytest.raises(ValueError):
        ps.parse_pair_overrides("0=1")           # two faces on the target clip
    with pytest.raises(ValueError):
        ps.parse_pair_overrides("oops")          # not a pairing at all


def test_no_pair_overrides_leave_the_pairing_untouched():
    """The default path must stay byte-identical to before the feature, or every existing job
    would silently change. Empty overrides return the auto pairing exactly as it came."""
    clip = [[100, 100, 200, 220], [400, 100, 500, 220]]
    still = [[100, 100, 200, 220], [400, 100, 500, 220]]
    auto = ps.pair_faces(clip, still, (1000, 1000), (1000, 1000), 0.05)
    assert ps.apply_pair_overrides(auto, ps.parse_pair_overrides(""), 2, 2, 0) == auto


# --- geometry the plan module owns --------------------------------------------------
#
# `overlapping_subject_pairs` and the subject-select node used to measure the same overlap and
# count the same faces in two places. These pin the one measure now that it has one home: the
# score between two face boxes, and the number a face carries on a frame.

def test_a_face_overlaps_the_box_that_covers_it():
    """The overlap score is how this module says "that detection is this face": fully on itself,
    a quarter on a box inside it, nothing on a box that merely touches."""
    face = [100, 100, 200, 220]
    assert ps.box_iou(face, face) == 1.0
    assert ps.box_iou(face, [100, 100, 150, 160]) == pytest.approx(0.25)
    assert ps.box_iou(face, [200, 220, 300, 320]) == 0.0


def test_faces_on_a_frame_are_numbered_the_way_the_picker_numbers_them():
    """A subject's face NUMBER is the box counted left to right on that frame, whatever order the
    detector returned them in - and a centre-x tie is settled on the box corner, which is the
    released tracker's rule and deliberately not the rule that orders the plan's subjects."""
    boxes = [[400, 120, 520, 260], [100, 100, 200, 220], [250, 150, 350, 300]]
    assert ps.left_to_right_face_order(boxes) == [1, 2, 0]

    tied = [[-40, 100, 60, 220], [0, 100, 20, 108]]      # both centred at x = 10
    assert ps.left_to_right_face_order(tied) == [0, 1]

