"""Subject Select: one subject's pick payload, in the shape the released tracker takes.

ADR 0005 says every pass reuses `H3FACEPICK` rather than inventing a second pick type, so the
narrowing is worth pinning where it is written: a plan dict and a payload in, a narrowed
payload out. Nothing here needs ComfyUI, torch, weights or a GPU.

The payload alone does not prove ADR 0005 - `tests/test_subject_select_drives_tracker.py`
hands this exact output to the released `H3FaceTrackCrop` and checks the crops come back
where the plan says. What these tests hold is the part that decides WHICH face every shot
locks onto, and the bookkeeping (`next`) that picks the subject at all.
"""

import copy
import json
import os

import pytest

import multi_face_nodes as m
import plan_store as ps

W, H = 640, 400
MIDDLE_X = W / 2.0
FRAMES = 20
SEGMENTS = [(0, 10), (10, 20)]
MATCH_FRAME = 6

# Two people, far enough apart that a crop cannot be about both of them.
LEFT = [60.0, 100.0, 140.0, 210.0]
RIGHT = [500.0, 120.0, 580.0, 230.0]


def _shift(box, dx):
    return [box[0] + dx, box[1], box[2] + dx, box[3]]


def _boxes_at(frame, single_after_the_cut=False):
    """The detector's answer on one frame.

    Before the cut it comes back left-then-right, after it right-then-left: a cut renumbers
    everyone, which is exactly why a face number measured at the match frame has to be
    translated into a box index per shot instead of carried across verbatim.
    """
    dx = (frame % 3) * 2.0
    left, right = _shift(LEFT, dx), _shift(RIGHT, -dx)
    if single_after_the_cut and frame >= SEGMENTS[1][0]:
        return [left]
    return [left, right] if frame < SEGMENTS[1][0] else [right, left]


def _payload(single_after_the_cut=False):
    """A synthetic `H3FACEPICK`, in the shape `H3FaceSelect` emits (nodes.py:3353)."""
    boxes = [_boxes_at(f, single_after_the_cut) for f in range(FRAMES)]
    confs = [[0.9, 0.8] if len(b) == 2 else [0.9] for b in boxes]
    picks = [{"segment": [int(a), int(b)], "frame": int(a), "box": 0, "index": 0,
              "absent": False} for a, b in SEGMENTS]
    return {
        "version": 1,
        "boxes": boxes,
        "confs": confs,
        "segments": [[int(a), int(b)] for a, b in SEGMENTS],
        "picks": picks,
        "frames": FRAMES,
        "src_size": (W, H),
        "detector": "face_yolov8m.pt",
        "confidence": 0.35,
        "select": "manual",
        "select_index": 0,
        "rank_by": "left_most",
        "identity_model": None,
        "identity_threshold": None,
        "identity_scores": [None, None],
    }


def _plan(payload, still_extra=(), subject_source="union", reversed_detection=False):
    """The subject plan for that clip, measured by the real plan builder.

    A reference still of the same framing at the same size pairs every face, so each subject
    gets a subject reference; `still_extra` adds a head the clip never saw.
    `reversed_detection` stands in for the plan node's detector returning the match frame's
    faces in the other order, which the narrowing has to survive.
    """
    at = [list(b) for b in payload["boxes"][MATCH_FRAME]]
    if reversed_detection:
        at = list(reversed(at))
    still = [list(b) for b in at] + [list(b) for b in still_extra]
    return ps.build_subject_plan(at, [0.9] * len(at), still, [0.95] * len(still),
                                 (W, H), (W, H), MATCH_FRAME, subject_source, 0.05, (), 0)


def _centre(payload, frame, box_index):
    box = payload["boxes"][frame][box_index]
    return (float(box[0]) + float(box[2])) / 2.0


# --- the narrowed payload ---------------------------------------------------------------

def test_the_narrowed_payload_carries_every_frame_of_the_clip():
    """The plan knows one frame; the clip's boxes carry the rest, untouched.

    The tracker raises if `frames` or `src_size` disagree with the batch that arrives, and it
    reads every frame's boxes, so a narrowed payload that trimmed or reordered them would
    fail the pass for a reason the report cannot explain.
    """
    payload = _payload()
    plan = _plan(payload)
    narrowed = m.narrow_pick_to_subject(plan, payload, plan["subjects"][0])

    assert narrowed is not payload
    assert narrowed["boxes"] == payload["boxes"]
    assert narrowed["confs"] == payload["confs"]
    assert narrowed["segments"] == payload["segments"]
    assert narrowed["frames"] == FRAMES
    assert narrowed["src_size"] == (W, H)
    assert narrowed["detector"] == payload["detector"]
    assert narrowed["confidence"] == payload["confidence"]
    assert narrowed["version"] == payload["version"]
    assert narrowed["identity_scores"] == payload["identity_scores"]


def test_narrowing_adds_no_key_the_picker_does_not_emit_and_removes_none():
    """ADR 0005 in one sentence: the narrowed payload has the picker's key set, verbatim.

    `H3FACEPICK` is a promise about a shape, not a label. `H3FaceSelect` writes this set
    (nodes.py:3353) and `H3 Face Track + Crop` reads frames, src_size, segments, picks,
    boxes, detector and confidence out of it (nodes.py:1315-1551), so a key the picker never
    writes makes the narrowed payload a different type wearing the old name - and the tracker
    would not read it even then. Whatever a pass needs to know about itself therefore rides
    on the plan or in the report, which is where these values are reachable from now.
    """
    payload = _payload()
    plan = _plan(payload)
    subject = plan["subjects"][1]
    narrowed = m.narrow_pick_to_subject(plan, payload, subject)

    assert set(narrowed) == set(payload)
    assert "subject_index" not in narrowed and "plan_match_frame" not in narrowed

    # The pick ENTRIES are shape-checked too: a per-shot entry the released producer never
    # writes would be just as foreign to the tracker as a top-level key it has never seen.
    assert [sorted(p) for p in narrowed["picks"]] == [sorted(p) for p in payload["picks"]]

    # A payload arriving with fewer keys leaves just as thin as it came, so nothing here
    # quietly "completes" it into a shape the loader did not produce.
    thin = {k: v for k, v in payload.items() if k not in ("identity_scores", "rank_by")}
    assert set(m.narrow_pick_to_subject(plan, thin, subject)) == set(thin)

    # And the two values that used to ride along are still reachable, plus still readable:
    # whose pass this is comes from the plan (it is also the node's `subject_index` output),
    # and the frame the plan measured comes from the plan's own `match_frame`.
    assert subject["index"] == 1
    assert plan["match_frame"] == MATCH_FRAME
    report = m.subject_report(plan, subject, narrowed)
    assert "subject 1 of 2" in report
    assert f"measured on match frame {MATCH_FRAME}" in report


def test_the_picks_are_one_per_shot_for_this_subject_only():
    payload = _payload()
    plan = _plan(payload)
    narrowed = m.narrow_pick_to_subject(plan, payload, plan["subjects"][0])

    assert len(narrowed["picks"]) == len(SEGMENTS)
    for pick, (a, b) in zip(narrowed["picks"], SEGMENTS):
        assert [int(pick["segment"][0]), int(pick["segment"][1])] == [a, b]
        assert pick["absent"] is False
        assert a <= int(pick["frame"]) < b
        assert 0 <= int(pick["box"]) < len(payload["boxes"][int(pick["frame"])])


def test_the_shot_at_the_match_frame_locks_on_the_frame_the_plan_measured():
    """The plan's whole claim is about the match frame, so that is the frame the tracker
    locks on - no measurement is invented for it."""
    payload = _payload()
    plan = _plan(payload)
    narrowed = m.narrow_pick_to_subject(plan, payload, plan["subjects"][0])

    assert narrowed["picks"][0]["frame"] == MATCH_FRAME
    assert narrowed["picks"][0]["box"] == plan["subjects"][0]["clip_face"]


def test_each_subject_gets_its_own_face():
    """Two subjects, two different faces - the point of a per-subject pass.

    In the editor's terms: subject 0's locks are all on the left-hand person, subject 1's on
    the right-hand one.
    """
    payload = _payload()
    plan = _plan(payload)
    left = m.narrow_pick_to_subject(plan, payload, plan["subjects"][0])
    right = m.narrow_pick_to_subject(plan, payload, plan["subjects"][1])

    assert left["picks"] != right["picks"]
    for pick in left["picks"]:
        assert _centre(payload, pick["frame"], pick["box"]) < MIDDLE_X
    for pick in right["picks"]:
        assert _centre(payload, pick["frame"], pick["box"]) > MIDDLE_X


def test_a_shot_that_renumbered_its_faces_still_gets_the_same_person():
    """After the cut the detector returns the two faces the other way round.

    A pick names a box index on one frame, so carrying the plan's index across the cut would
    track the other person. The plan's subject keeps their face NUMBER - counted left to
    right, as the pairing preview and Pick faces number them - which resolves to the other
    index in the renumbered shot.
    """
    payload = _payload()
    plan = _plan(payload)
    left = m.narrow_pick_to_subject(plan, payload, plan["subjects"][0])
    right = m.narrow_pick_to_subject(plan, payload, plan["subjects"][1])

    # At the match frame the left person is box 0; in shot 2 they are box 1.
    assert left["picks"][0]["box"] == 0 and left["picks"][1]["box"] == 1
    assert right["picks"][0]["box"] == 1 and right["picks"][1]["box"] == 0
    for pick in left["picks"][1:]:
        assert _centre(payload, pick["frame"], pick["box"]) < MIDDLE_X
    for pick in right["picks"][1:]:
        assert _centre(payload, pick["frame"], pick["box"]) > MIDDLE_X


def test_the_plan_and_the_pick_may_disagree_about_detection_order():
    """The plan measured its own detection pass; the payload carries another one.

    Matching the plan's recorded box to the payload's box at the match frame - rather than
    trusting the index - is what keeps a subject on their own face when the two passes
    returned the faces in a different order.
    """
    payload = _payload()
    plan = _plan(payload, reversed_detection=True)
    subject = next(s for s in plan["subjects"] if s["index"] == 1)

    narrowed = m.narrow_pick_to_subject(plan, payload, subject)

    for pick in narrowed["picks"]:
        assert _centre(payload, pick["frame"], pick["box"]) > MIDDLE_X


def test_a_subject_not_in_a_shot_is_marked_absent_for_that_shot():
    """Absence is already in the released payload shape: `absent: true` over a segment, so
    those frames keep their original pixels instead of the tracker locking onto whoever is
    nearest (nodes.py:1334)."""
    payload = _payload(single_after_the_cut=True)
    plan = _plan(payload)
    right = next(s for s in plan["subjects"] if s["index"] == 1)

    narrowed = m.narrow_pick_to_subject(plan, payload, right)

    assert narrowed["picks"][0]["absent"] is False
    gone = narrowed["picks"][1]
    assert gone["absent"] is True
    assert [int(gone["segment"][0]), int(gone["segment"][1])] == list(SEGMENTS[1])
    assert gone["frame"] == -1 and gone["box"] == -1
    assert narrowed["boxes"] == payload["boxes"]     # the boxes stay, whoever is in them


def test_the_report_names_the_subject_its_pairing_and_the_shots_it_was_absent_from():
    payload = _payload(single_after_the_cut=True)
    plan = _plan(payload)
    right = next(s for s in plan["subjects"] if s["index"] == 1)
    narrowed = m.narrow_pick_to_subject(plan, payload, right)

    report = m.subject_report(plan, right, narrowed)

    assert "subject 1" in report
    assert "clip face %d" % right["clip_face"] in report
    assert "still face %d" % right["still_face"] in report
    assert "shot 2" in report and "absent" in report
    assert "no detection" in report           # the boxes ride in on the payload
    assert "identity_reference" in report     # ADR 0003: the tracker anchors the subject


# --- the payload is an input, not a workspace -------------------------------------------

def test_the_input_payload_is_never_touched():
    """Two passes in one graph read the same payload. A narrowed pick that edited it in place
    would leave the second pass tracking the first subject's face."""
    payload = _payload(single_after_the_cut=True)
    plan = _plan(payload)
    before = copy.deepcopy(payload)
    narrowed = m.narrow_pick_to_subject(plan, payload, plan["subjects"][1])

    narrowed["picks"][0]["box"] = 99
    narrowed["picks"].append({"segment": [0, 1], "frame": 0, "box": 0, "index": 0,
                              "absent": True})

    assert payload == before


# --- a plan that cannot drive a pass says so --------------------------------------------

def test_a_subject_the_clip_never_saw_refuses_rather_than_tracking_a_stranger():
    """`union` makes a subject of a head only the still can see. There is no box for it in
    the clip, so no shot can be named and the pass would refine a stranger."""
    payload = _payload()
    plan = _plan(payload, still_extra=[[300.0, 300.0, 380.0, 410.0]])
    ghost = next(s for s in plan["subjects"] if s["clip_face"] is None)

    with pytest.raises(ValueError) as exc:
        m.narrow_pick_to_subject(plan, payload, ghost)

    assert "clip face" in str(exc.value)


def test_a_pick_from_a_differently_sized_clip_refuses():
    """Every box in the payload would land in the wrong place, and the tracker's own check
    would only catch it after the plan had been read."""
    payload = _payload()
    payload["src_size"] = (1280, 800)
    plan = _plan(_payload())

    with pytest.raises(ValueError) as exc:
        m.narrow_pick_to_subject(plan, payload, plan["subjects"][0])
    assert "src_size" in str(exc.value)


def test_a_pick_shorter_than_the_match_frame_refuses():
    payload = _payload()
    plan = _plan(payload)
    payload["frames"] = 3
    payload["boxes"] = payload["boxes"][:3]
    payload["confs"] = payload["confs"][:3]

    with pytest.raises(ValueError) as exc:
        m.narrow_pick_to_subject(plan, payload, plan["subjects"][0])
    assert "match frame" in str(exc.value)


# --- the `next` cursor (plan_store reads the manifest; ticket #5 writes it) --------------

def _write_manifest(job_dir, passes):
    with open(os.path.join(str(job_dir), ps.MANIFEST_FILENAME), "w", encoding="utf-8") as fh:
        json.dump({"version": ps.MANIFEST_VERSION, "passes": passes}, fh)


def test_a_missing_manifest_reads_as_nothing_completed_yet(tmp_path):
    plan = _plan(_payload())

    assert ps.read_manifest(tmp_path) == {"version": ps.MANIFEST_VERSION, "passes": {}}
    assert ps.completed_subjects(plan, tmp_path) == []


def test_next_takes_the_first_subject_when_nothing_is_recorded(tmp_path):
    plan = _plan(_payload())

    assert ps.next_subject(plan, tmp_path)["index"] == 0


def test_next_skips_the_subjects_whose_pass_is_recorded_complete(tmp_path):
    plan = _plan(_payload())
    _write_manifest(tmp_path, {"0": {"complete": True}, "1": {"complete": False}})

    assert ps.completed_subjects(plan, tmp_path) == [0]
    assert ps.next_subject(plan, tmp_path)["index"] == 1


def test_next_says_when_every_pass_is_already_recorded(tmp_path):
    plan = _plan(_payload())
    _write_manifest(tmp_path, {str(s["index"]): {"complete": True}
                               for s in plan["subjects"]})

    assert ps.next_subject(plan, tmp_path) is None


def test_next_only_counts_subjects_this_plan_holds(tmp_path):
    """A manifest written for another plan must not answer for this one: plan order, not file
    order, decides who is up next."""
    plan = _plan(_payload())
    _write_manifest(tmp_path, {"7": {"complete": True}})

    assert ps.completed_subjects(plan, tmp_path) == []
    assert ps.next_subject(plan, tmp_path)["index"] == 0


# --- the widget's answer ----------------------------------------------------------------

def test_the_subject_widget_understands_next_a_number_and_the_label(tmp_path):
    plan = _plan(_payload())

    assert m.resolve_subject(plan, "next", tmp_path)["index"] == 0
    assert m.resolve_subject(plan, "1", tmp_path)["index"] == 1
    assert m.resolve_subject(plan, m.subject_label(plan["subjects"][1]),
                             tmp_path)["index"] == 1


def test_next_without_a_job_folder_says_what_is_missing():
    plan = _plan(_payload())

    with pytest.raises(ValueError) as exc:
        m.resolve_subject(plan, "next", None)
    assert "job" in str(exc.value)


def test_an_unknown_subject_names_the_choices():
    plan = _plan(_payload())

    with pytest.raises(ValueError) as exc:
        m.resolve_subject(plan, "who?", None)
    assert "next" in str(exc.value)
    assert m.subject_label(plan["subjects"][0]) in str(exc.value)


# --- the subject walk: always the next subject with no saved pass, read from the job ---------
#
# There is no subject/mode widget. run() reads the job folder's pass manifest and takes the first
# subject with no completed pass, so queueing the same graph advances one subject per SAVED pass,
# and a queue that crashes mid-pass is taken again rather than skipped (the self-heal that used to
# deadlock on Pass Load). narrow, the reference load and the report builder are stubbed here so only
# the walk is under test; the output root is redirected into tmp_path so nothing writes to a real
# ComfyUI output folder.

def _walk_plan(job_name="walk job"):
    return {"version": ps.PLAN_VERSION, "match_frame": 0,
            "subjects": [{"index": 0}, {"index": 1}, {"index": 2}],
            "run": {"job_name": job_name}}


def _walk_select(monkeypatch, tmp_path):
    """H3SubjectSelect with the pick-specific machinery stubbed: only the subject WALK (which
    subject comes up next from the pass manifest, and the walk note) is under test, so narrowing,
    the reference load and the full report builder are replaced with no-ops, and the output root is
    redirected into tmp_path so nothing ever writes to a real ComfyUI output folder."""
    monkeypatch.setattr(m, "_output_root", lambda: str(tmp_path / "output"))
    monkeypatch.setattr(m, "narrow_pick_to_subject",
                        lambda plan, face_pick, chosen: face_pick)
    monkeypatch.setattr(m, "load_subject_reference",
                        lambda job, chosen: (None, 0, ""))
    # Keep the walk note: the real report builder folds cursor_note into the report, so the stub
    # returns it and the report assertions below still see the walk line.
    monkeypatch.setattr(m, "subject_report",
                        lambda *a, **k: k.get("cursor_note", ""))
    return m.H3SubjectSelect()


def _job(tmp_path, job_name="walk job"):
    return ps.job_dir_for(str(tmp_path / "output"), job_name)


def test_next_is_the_first_subject_with_no_saved_pass(monkeypatch, tmp_path):
    """Nothing saved yet: the walk sits on subject 0. Only a queue that SAVES a pass (records it
    complete in the manifest) moves it on - one queue per subject, with no widget to drive it."""
    plan = _walk_plan()
    node = _walk_select(monkeypatch, tmp_path)
    job = _job(tmp_path)
    os.makedirs(str(job), exist_ok=True)

    assert [node.run(plan, "PICK")[3] for _ in range(3)] == [0, 0, 0]

    _write_manifest(job, {"0": {"complete": True}})
    assert node.run(plan, "PICK")[3] == 1
    _write_manifest(job, {"0": {"complete": True}, "1": {"complete": True}})
    assert node.run(plan, "PICK")[3] == 2


def test_the_walk_goes_back_to_the_earliest_subject_with_no_pass(monkeypatch, tmp_path):
    """Subject 1's pass is on record but subject 0's is not - an out-of-order or crashed job. The
    chain cannot honour that, so the walk returns to the earliest subject with no pass."""
    plan = _walk_plan()
    node = _walk_select(monkeypatch, tmp_path)
    job = _job(tmp_path)
    os.makedirs(str(job), exist_ok=True)
    _write_manifest(job, {"1": {"complete": True}})

    assert node.run(plan, "PICK")[3] == 0
    assert node.run(plan, "PICK")[3] == 0


def test_the_report_shows_the_current_id_and_the_remaining_queues(monkeypatch, tmp_path):
    plan = _walk_plan()
    node = _walk_select(monkeypatch, tmp_path)
    job = _job(tmp_path)
    os.makedirs(str(job), exist_ok=True)

    first = node.run(plan, "PICK")[5]
    assert "you are on subject 0 of 3" in first
    assert "2 still to run after this one" in first
    _write_manifest(job, {"0": {"complete": True}})
    second = node.run(plan, "PICK")[5]
    assert "you are on subject 1 of 3" in second
    assert "1 still to run after this one" in second


def test_an_empty_plan_is_refused(monkeypatch, tmp_path):
    plan = {"version": ps.PLAN_VERSION, "match_frame": 0, "subjects": [],
            "run": {"job_name": "walk job"}}
    node = _walk_select(monkeypatch, tmp_path)
    with pytest.raises(ValueError) as exc:
        node.run(plan, "PICK")
    assert "no subjects" in str(exc.value)


def test_when_every_pass_is_saved_the_walk_points_at_reset_run(monkeypatch, tmp_path):
    """Every subject done: rather than invent a generation, it refuses and names Reset run."""
    plan = _walk_plan()
    node = _walk_select(monkeypatch, tmp_path)
    job = _job(tmp_path)
    os.makedirs(str(job), exist_ok=True)
    _write_manifest(job, {str(i): {"complete": True} for i in range(3)})

    with pytest.raises(ValueError) as exc:
        node.run(plan, "PICK")
    assert "already has a completed pass" in str(exc.value)
    assert "Reset run" in str(exc.value)

