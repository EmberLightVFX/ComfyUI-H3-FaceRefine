"""Pairing, tested with synthetic boxes: no ComfyUI, no torch, no GPU, no weights.

Every test names an outcome an editor would recognise - "the still's second face is the
clip's leftmost face's subject reference", "the bystander at index 2 is excluded" - and
never asserts how a helper does its arithmetic.
"""

import pytest

import plan_store as ps


# --- pairing: positional, measured at the match frame -------------------------------

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
