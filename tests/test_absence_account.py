"""Per-subject, per-shot absence, derived from what a pass already leaves behind (#6).

Pure: numpy and the standard library only (the same import boundary `test_node_contract.py`
pins for `plan_store`), so the difference between a subject who is genuinely NOT in a shot and
a tracker that LOST them mid-shot is testable with no ComfyUI, no torch, no weights, no GPU.

Absence itself is not invented here. The released payload says "this person is not in this
shot" with a pick entry `{"absent": true, "segment": [start, end]}` (nodes.py:1334), and the
released tracker and stitch already honour it. What this module adds is the READ-OUT: which
shot dropped which subject, over which frames, and whether that was planned or a failure.

The evidence is the crop-position record (`H3FACEXFORM`, nodes.py:1908): per KEPT frame it
carries `absent` (inside a shot the pick marked absent), `detected` (the tracker resolved this
subject's face on that frame) and `source` (which source frame each crop came from). Frames
the tracker dropped out of the batch (nodes.py:1792 - it only ever drops frames already marked
absent) leave a gap in `source`, so a gap is absence, never a lost subject.
"""

import pytest

import plan_store as ps


FRAMES = 12
SHOTS = [(0, 6), (6, 12)]


def _plan(count: int = 1, frames: int = FRAMES) -> dict:
    """A plan with `count` subjects - only the numbers and pairing matter here."""
    return {
        "version": ps.PLAN_VERSION,
        "match_frame": 2,
        "run": {"job_name": "group shot", "frames": frames},
        "subjects": [
            {"index": i, "clip_face": i, "clip_box": [10.0 * i, 10.0, 40.0 + 10.0 * i, 50.0],
             "still_face": i, "paired": True, "has_reference": True}
            for i in range(count)
        ],
        "subject_count": count,
    }


def _transform(absent, detected, source=None, segments=None, frames: int = FRAMES) -> dict:
    """A crop-position payload in the shape `H3FaceTrackCrop` hands over.

    `absent` / `detected` are per KEPT frame in crop order, exactly as the tracker writes
    them; `segments` are CROP-space shot bounds, which is what makes the clip's own frame
    ranges a derivation rather than something the payload states outright.
    """
    kept = len(absent)
    return {
        "boxes": [(0, 0, 64, 64)] * kept,
        "canvas": (64, 64),
        "src_size": (128, 128),
        "frames": kept,
        "source": list(range(kept)) if source is None else list(source),
        "source_frames": frames,
        "segments": [(0, kept)] if segments is None else [tuple(s) for s in segments],
        "absent": [bool(v) for v in absent],
        "detected": [bool(v) for v in detected],
        "weights": [0.0 if a else 1.0 for a in absent],
        "face_rect": [(1.0, 1.0, 10.0, 10.0)] * kept,
        "crop_factor": 2.5,
    }


def _pick(absent_flags, faces_per_frame: int = 1, segments=SHOTS, frames: int = FRAMES) -> dict:
    """The narrowed pick payload Subject Select hands the tracker, for the same clip.

    `faces_per_frame` is how many faces the detector reported on each frame, which is what
    tells "nobody was ever detected in this shot" apart from "faces were there, none of them
    was this subject".
    """
    picks = []
    for (a, b), flag in zip(segments, absent_flags):
        picks.append({"segment": [a, b], "frame": (-1 if flag else a),
                      "box": (-1 if flag else 0), "index": 0, "absent": bool(flag)})
    return {
        "boxes": [[([10.0, 10.0, 40.0, 50.0] if faces_per_frame >= 1 else [])]
                  * faces_per_frame for _ in range(frames)],
        "confs": [[0.9] * faces_per_frame for _ in range(frames)],
        "segments": [[int(a), int(b)] for a, b in segments],
        "picks": picks,
        "frames": frames,
        "src_size": (128, 128),
        "select_index": 0,
    }


# --- the three states ---------------------------------------------------------------------

def test_a_shot_the_plan_marked_absent_is_reported_as_planned_absence():
    transform = _transform([False] * 6 + [True] * 6, [True] * 6 + [False] * 6,
                           segments=[(0, 6), (6, 12)])
    account = ps.absence_account(_plan(), 0, transform=transform, pick=_pick([False, True]))

    by_shot = {s["shot"]: s for s in account["shots"]}
    assert by_shot[1]["state"] == ps.SHOT_PAIRED
    assert by_shot[2]["state"] == ps.SHOT_ABSENT_PLANNED
    assert by_shot[2]["start"] == 6 and by_shot[2]["end"] == 11
    assert account["absent_planned"] == [2]
    assert account["lost"] == []
    assert account["present"] == [1]


def test_a_shot_the_tracker_reached_but_never_found_them_in_is_reported_lost():
    """The pick named this subject for the shot, so the shot is NOT absent - the tracker got
    there with the subject selected and produced nobody. That is a failure, and the report has
    to say so in words the user can act on."""
    transform = _transform([False] * 12, [True] * 6 + [False] * 6, segments=[(0, 6), (6, 12)])
    account = ps.absence_account(_plan(), 0, transform=transform, pick=_pick([False, False]))

    by_shot = {s["shot"]: s for s in account["shots"]}
    assert by_shot[1]["state"] == ps.SHOT_PAIRED
    assert by_shot[2]["state"] == ps.SHOT_LOST
    assert by_shot[2]["start"] == 6 and by_shot[2]["end"] == 11
    assert account["lost"] == [2]
    assert account["absent_planned"] == []


def test_a_subject_absent_from_one_shot_is_still_present_in_the_others():
    """Three shots, absent in the middle one only: the shots either side of it are unaffected,
    which is what the released pipeline does with a per-shot absent entry."""
    segments = [(0, 4), (4, 8), (8, 12)]
    transform = _transform([False] * 4 + [True] * 4 + [False] * 4,
                           [True] * 4 + [False] * 4 + [True] * 4, segments=segments)
    account = ps.absence_account(_plan(), 0, transform=transform,
                                 pick=_pick([False, True, False],
                                            segments=[(0, 4), (4, 8), (8, 12)]))

    by_shot = {s["shot"]: s for s in account["shots"]}
    assert [by_shot[k]["state"] for k in (1, 2, 3)] == [ps.SHOT_PAIRED, ps.SHOT_ABSENT_PLANNED,
                                                        ps.SHOT_PAIRED]
    assert account["present"] == [1, 3]
    assert account["absent_planned"] == [2]
    assert (by_shot[1]["start"], by_shot[1]["end"]) == (0, 3)
    assert (by_shot[3]["start"], by_shot[3]["end"]) == (8, 11)


def test_a_partial_loss_inside_a_shot_names_the_frames_that_produced_nobody():
    transform = _transform([False] * 12, [True] * 6 + [False] * 4 + [True] * 2,
                           segments=[(0, 6), (6, 12)])
    account = ps.absence_account(_plan(), 0, transform=transform, pick=_pick([False, False]))
    lost = account["shots"][1]
    assert lost["state"] == ps.SHOT_LOST
    assert lost["lost_start"] == 6 and lost["lost_end"] == 9
    assert lost["frames_lost"] == 4


# --- ranges are the clip's frames, not the crop's -------------------------------------------

def test_ranges_are_the_clip_s_frames_when_the_batch_dropped_frames():
    """Frames the subject is absent from are dropped out of the batch (nodes.py:1792) and kept
    back only as padding to reach H3's 17k+5 grid, so the payload's crop-space segment numbers
    are not frame numbers. The account names the clip's frames anyway."""
    source = list(range(6)) + [6, 7]
    transform = _transform([False] * 6 + [True] * 2, [True] * 6 + [False] * 2,
                           source=source, segments=[(0, 6), (6, 8)])
    account = ps.absence_account(_plan(), 0, transform=transform)

    by_shot = {s["shot"]: s for s in account["shots"]}
    assert by_shot[2]["state"] == ps.SHOT_ABSENT_PLANNED
    # frames 8-11 never reached the batch at all; they are still reported, because the user
    # has to be told this subject was not refined there.
    assert (by_shot[2]["start"], by_shot[2]["end"]) == (6, 11)
    assert by_shot[2]["frames_dropped"] == 4


def test_frames_dropped_out_of_the_batch_are_absence_and_never_a_lost_subject():
    transform = _transform([False] * 6, [True] * 6, source=list(range(6)), segments=[(0, 6)])
    account = ps.absence_account(_plan(), 0, transform=transform)
    assert account["lost"] == []
    assert account["present"] == [1]


# --- the reason says which kind of non-appearance it was ------------------------------------

def test_a_shot_with_no_faces_at_all_is_reported_as_never_detected():
    transform = _transform([False] * 12, [True] * 6 + [False] * 6, segments=[(0, 6), (6, 12)])
    account = ps.absence_account(_plan(), 0, transform=transform,
                                 pick=_pick([False, False], faces_per_frame=0))
    assert "no face" in account["shots"][1]["reason"]


def test_faces_present_but_unmatched_is_reported_as_the_identity_falling_away():
    transform = _transform([False] * 12, [True] * 6 + [False] * 6, segments=[(0, 6), (6, 12)])
    account = ps.absence_account(_plan(), 0, transform=transform,
                                 pick=_pick([False, False], faces_per_frame=2))
    assert "identity" in account["shots"][1]["reason"]


# --- the wording a non-developer can act on -------------------------------------------------

def test_a_planned_absence_says_which_shot_and_that_original_pixels_are_kept():
    transform = _transform([False] * 6 + [True] * 6, [True] * 6 + [False] * 6,
                           segments=[(0, 6), (6, 12)])
    account = ps.absence_account(_plan(), 1, transform=transform, pick=_pick([False, True]))
    lines = account["lines"]
    assert any("subject 1 is not in shot 2 (frames 6-11)" in line for line in lines)
    text = "\n".join(lines)
    assert "original pixels are kept" in text
    assert "was lost" not in text


def test_a_lost_subject_says_which_shot_and_what_to_change():
    transform = _transform([False] * 12, [True] * 6 + [False] * 6, segments=[(0, 6), (6, 12)])
    account = ps.absence_account(_plan(), 1, transform=transform, pick=_pick([False, False]))
    lines = account["lines"]
    assert any("subject 1 was lost in shot 2 (frames 6-11)" in line for line in lines)
    text = "\n".join(lines)
    assert "identity_track" in text and "confidence" in text
    assert "is not in" not in text


def test_the_two_kinds_of_non_appearance_are_not_confusable_in_one_report():
    """One shot planned absent, another lost, in the same pass: the report alone has to let
    the user tell which is which, because only one of them is worth fixing."""
    segments = [(0, 4), (4, 8), (8, 12)]
    transform = _transform([False] * 4 + [True] * 4 + [False] * 4,
                           [True] * 4 + [False] * 4 + [False] * 4, segments=segments)
    account = ps.absence_account(_plan(), 0, transform=transform,
                                 pick=_pick([False, True, False],
                                            segments=[(0, 4), (4, 8), (8, 12)]))
    text = "\n".join(account["lines"])
    assert "is not in shot 2 (frames 4-7)" in text
    assert "was lost in shot 3 (frames 8-11)" in text


def test_a_pass_that_planned_nothing_and_lost_nobody_says_so_plainly():
    transform = _transform([False] * 12, [True] * 12, segments=[(0, 12)])
    account = ps.absence_account(_plan(), 0, transform=transform,
                                 pick=_pick([False], segments=[(0, 12)]))
    assert account["lost"] == [] and account["absent_planned"] == []
    assert account["present"] == [1]
    assert account["lines"] == [] or "every shot" in "\n".join(account["lines"])


def test_the_account_carries_the_subject_it_was_asked_about():
    account = ps.absence_account(_plan(), 1, transform=_transform([False] * 12, [True] * 12))
    assert account["subject"] == 1
    assert all(s["shot"] for s in account["shots"])


def test_a_subject_handed_as_a_plan_entry_is_the_same_account():
    plan = _plan(2)
    by_number = ps.absence_account(plan, 1, transform=_transform([False] * 12, [True] * 12))
    by_entry = ps.absence_account(plan, plan["subjects"][1],
                                  transform=_transform([False] * 12, [True] * 12))
    assert by_number["subject"] == by_entry["subject"] == 1
    assert [s["state"] for s in by_number["shots"]] == [s["state"] for s in by_entry["shots"]]


def test_a_payload_that_is_not_crop_positions_is_refused_rather_than_guessed():
    with pytest.raises(ValueError) as exc:
        ps.shot_account(_plan(), 0, transform=["not", "a", "payload"])
    assert "crop positions" in str(exc.value)


def test_a_subject_with_no_subject_reference_is_worded_for_that_not_for_a_missing_partner():
    """#8: an unpaired subject has no subject reference, and that is a reason of its own for a
    shot holding nobody. The account must say so instead of talking about pairing partners:
    there is no partner to find, so that wording sends the user off to fix something that is
    already settled."""
    plan = _plan()
    subject = plan["subjects"][0]
    subject["still_face"] = None
    subject["paired"] = False
    subject["has_reference"] = False
    subject["ref_kind"] = "none"

    transform = _transform([False] * 6 + [True] * 6, [True] * 6 + [False] * 6,
                           segments=[(0, 6), (6, 12)])
    account = ps.absence_account(plan, 0, transform=transform, pick=_pick([False, True]))

    assert account["absent_planned"] == [2]
    reason = account["shots"][1]["reason"]
    assert "no subject reference" in reason
    text = "\n".join(account["lines"])
    assert "partner" not in text
    # And it stays actionable: the widget that would change it is named.
    assert "unmatched_faces" in text
