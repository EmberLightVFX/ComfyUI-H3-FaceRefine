"""How much two subjects' crop regions overlap at the match frame - pure, no ComfyUI, no torch.

H3 Face Stitch feathers each pass back onto the clip. When two subjects' crop regions sit on
top of each other, two feathers land on the same pixels and the second pass softens the first
one's work. That is a thing to WARN about, never to refuse: the pair may be two people standing
close together, and the composite is still the right answer for them.

The measure is the intersection over union of the two crop rectangles the tracker would cut,
both in target-clip pixels, at the match frame - the same rect `tracker_rect` records the plan
was built on.
"""

import pytest

import plan_store as ps

SIZE = (1000, 1000)
FACE = [100.0, 100.0, 200.0, 200.0]          # centre (150, 150), 100px tall -> 250px crop


def _shifted(dx: int) -> list[float]:
    """The same face, `dx` clip pixels to the right."""
    return [FACE[0] + dx, FACE[1], FACE[2] + dx, FACE[3]]


def _plan(boxes, still_only=()) -> dict:
    """A plan in the shape `build_subject_plan` writes, enough for the crop rect alone.

    `still_only` holds subject indices whose person the clip never saw, so their framing comes
    from the still's own face.
    """
    subjects = []
    for index, box in enumerate(boxes):
        seen = index not in still_only
        subjects.append({
            "index": index,
            "clip_face": index if seen else None,
            "clip_box": [float(v) for v in box] if seen else None,
            "still_face": index,
            "still_box": [float(v) for v in box],
            "paired": seen,
        })
    return {
        "version": ps.PLAN_VERSION,
        "match_frame": 2,
        "scale": 1.0,
        "inputs": {"target_size": list(SIZE), "still_size": list(SIZE), "match_frame": 2},
        "target_boxes": [[float(v) for v in b] for b in boxes],
        "subjects": subjects,
        "subject_count": len(subjects),
    }


def test_the_default_threshold_is_a_fraction_and_crops_touching_edges_are_no_overlap():
    assert ps.DEFAULT_OVERLAP_THRESHOLD > 0.0
    assert ps.DEFAULT_OVERLAP_THRESHOLD < 1.0

    # 250px crops, 250px apart: they share an edge, which is no overlap at all.
    pairs = ps.overlapping_subject_pairs(_plan([FACE, _shifted(250)]))
    assert pairs == []


def test_a_pair_sitting_exactly_on_the_threshold_is_not_a_warning():
    """Borderline: 100px of a 250px crop in common is an IoU of exactly 0.25.

    Past the threshold warns, ON it does not - otherwise every plan planned at a tidy round
    number would print a warning nobody can act on.
    """
    plan = _plan([FACE, _shifted(150)])

    assert ps.overlapping_subject_pairs(plan, 0.25) == []

    just_past = ps.overlapping_subject_pairs(plan, 0.24)
    assert len(just_past) == 1
    pair = just_past[0]
    assert (pair["a"], pair["b"]) == (0, 1)
    assert pair["overlap"] == pytest.approx(0.25)


def test_past_the_threshold_names_both_subjects_in_plan_order():
    plan = _plan([FACE, _shifted(100)])          # IoU 3/7

    pairs = ps.overlapping_subject_pairs(plan, 0.25)
    assert len(pairs) == 1
    assert (pairs[0]["a"], pairs[0]["b"]) == (0, 1)
    assert pairs[0]["overlap"] == pytest.approx(3.0 / 7.0)
    for rect in (pairs[0]["rect_a"], pairs[0]["rect_b"]):
        assert len(rect) == 4


def test_only_the_pairs_past_the_threshold_appear_and_never_a_subject_against_itself():
    plan = _plan([FACE, _shifted(150), _shifted(650)])   # 0-1 overlap, 2 is elsewhere

    pairs = ps.overlapping_subject_pairs(plan, 0.2)
    assert [(p["a"], p["b"]) for p in pairs] == [(0, 1)]
    pairs = ps.overlapping_subject_pairs(plan, 0.0)
    assert [(p["a"], p["b"]) for p in pairs] == [(0, 1)]


def test_a_single_subject_plan_has_no_pairs_to_warn_about():
    assert ps.overlapping_subject_pairs(_plan([FACE])) == []
    assert ps.overlapping_subject_pairs({}) == []


def test_a_subject_the_clip_never_saw_is_measured_from_the_still_face():
    """No clip box, so the framing comes from the still's face at the same crop factor."""
    plan = _plan([FACE, _shifted(150)], still_only=(1,))

    pairs = ps.overlapping_subject_pairs(plan, 0.24)
    assert [(p["a"], p["b"]) for p in pairs] == [(0, 1)]
    assert pairs[0]["overlap"] == pytest.approx(0.25)


def test_the_measure_is_the_same_rect_the_plan_cut_the_subject_reference_with():
    """A real plan, not a hand-written one: the crop region is `tracker_rect`'s output."""
    plan = ps.build_subject_plan([FACE, _shifted(150)], [0.9, 0.8], [], [], SIZE, SIZE,
                                 match_frame=2, subject_source="target_clip")
    assert ps.plan_subject_indices(plan) == [0, 1]

    expected = ps.overlapping_subject_pairs(_plan([FACE, _shifted(150)]), 0.24)
    assert [(p["a"], p["b"], round(p["overlap"], 6)) for p in
            ps.overlapping_subject_pairs(plan, 0.24)] == [
        (p["a"], p["b"], round(p["overlap"], 6)) for p in expected]


def test_a_nonsense_threshold_refuses_rather_than_warning_about_everything():
    plan = _plan([FACE, _shifted(150)])
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError) as exc:
            ps.overlapping_subject_pairs(plan, bad)
        assert "overlap_threshold" in str(exc.value)
