"""The plan node run end to end, with the detector stood in for.

Needs torch, so CI (numpy only) skips it and it runs on the ComfyUI interpreter. The
detector is the one part a unit test cannot honestly fake, so it is replaced with a stub
returning the boxes a run would have measured: everything downstream of detection - the
plan, the crops, the job folder, the preview, the report - is the pack's own code.

No weights, no GPU. `nodes.py` is imported as it is released, with only ComfyUI's own two
imports stubbed when ComfyUI is not on the path.
"""

import importlib.util
import json
import pathlib
import sys
import types

import pytest

torch = pytest.importorskip("torch")

PACK_ROOT = pathlib.Path(__file__).resolve().parents[1]

CLIP_BOXES = [[100, 100, 200, 220], [400, 400, 500, 500], [800, 120, 900, 240]]
CLIP_CONFS = [0.9, 0.8, 0.7]
# Twice the clip's scale, and its own detection order: the still saw the outer faces.
STILL_BOXES = [[200, 200, 400, 440], [1600, 240, 1800, 480]]
STILL_CONFS = [0.95, 0.6]


class _Boxes:
    """The `.boxes` of one ultralytics result: sized, with xyxy and conf."""

    def __init__(self, boxes, confs):
        self._n = len(boxes)
        self.xyxy = types.SimpleNamespace(tolist=lambda: [list(b) for b in boxes])
        self.conf = types.SimpleNamespace(tolist=lambda: list(confs))

    def __len__(self):
        return self._n


class _Found:
    """The shape of one ultralytics result, as `nodes.py` and `run()` read it."""

    def __init__(self, boxes, confs):
        self.boxes = _Boxes(boxes, confs)


class _FakeModel:
    """Returns the given detections once each: clip plate first, then still plate."""

    def __init__(self, found):
        self._found = list(found)
        self.calls = 0

    def predict(self, image, conf=None, verbose=False):
        result = self._found[self.calls]
        self.calls += 1
        return [result]


@pytest.fixture()
def pack(tmp_path, monkeypatch):
    """The pack as ComfyUI loads it - a package - without needing ComfyUI installed."""
    if importlib.util.find_spec("comfy") is None:
        comfy = types.ModuleType("comfy")
        nested = types.ModuleType("comfy.nested_tensor")
        nested.NestedTensor = type("NestedTensor", (), {})
        comfy.nested_tensor = nested
        monkeypatch.setitem(sys.modules, "comfy", comfy)
        monkeypatch.setitem(sys.modules, "comfy.nested_tensor", nested)
    # The plan node asks for the output root, so it must never resolve to a real ComfyUI
    # output folder. Importing the tracker test puts ComfyUI's own tree on sys.path, which
    # makes the real `folder_paths` importable mid-collection - so redirect the output root
    # whether or not a stub had to be invented for it. Model lookup stays real, since this
    # test drives the detector through a fake model, not through model discovery.
    if importlib.util.find_spec("folder_paths") is None:
        paths = types.ModuleType("folder_paths")
        paths.get_output_directory = lambda: str(tmp_path / "output")
        paths.models_dir = str(tmp_path / "models")

        def _no_models(key):
            raise RuntimeError("no model folders in this test")

        paths.get_filename_list = _no_models
        paths.get_full_path = lambda key, name: None
        monkeypatch.setitem(sys.modules, "folder_paths", paths)
    else:
        import folder_paths as paths
        monkeypatch.setattr(paths, "get_output_directory",
                            lambda: str(tmp_path / "output"), raising=False)

    name = "h3_facerefine_under_test"
    spec = importlib.util.spec_from_file_location(
        name, str(PACK_ROOT / "__init__.py"),
        submodule_search_locations=[str(PACK_ROOT)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        for loaded in [k for k in list(sys.modules)
                       if k == name or k.startswith(name + ".")]:
            sys.modules.pop(loaded, None)


def _run(pack, monkeypatch, still_boxes=STILL_BOXES, still_size=(2000, 2000),
         excluded="", job_name="group shot", source="union", mode="leave", max_subjects=0,
         crop_factor=None, overrides=""):
    fake = _FakeModel([_Found(CLIP_BOXES, CLIP_CONFS), _Found(still_boxes, STILL_CONFS)])
    monkeypatch.setattr(pack.nodes, "_load_detector", lambda name: fake)
    node = pack.NODE_CLASS_MAPPINGS["H3MultiFacePlan"]()
    images = torch.zeros((4, 1000, 1000, 3))
    images.h3_video_name = "shot_07.mp4"
    still = torch.zeros((1, int(still_size[1]), int(still_size[0]), 3))
    # crop_factor is the LAST required widget (ADR 0002), so it goes on the end of the call.
    # Left None means "the widget was never touched", which is what a saved graph does.
    # crop_factor and pair_overrides are the last two widgets, so they only reach the call when
    # given. An override forces crop_factor on too (it sits before it in the widget order).
    tail = [mode]
    if crop_factor is not None or overrides:
        tail.append(crop_factor if crop_factor is not None else 2.5)
        if overrides:
            tail.append(overrides)
    return node.run(images, still, "face_yolov8m.pt", 0.35, 2, source, 0.05, excluded,
                    max_subjects, job_name, *tail)


def test_process_plan_replaces_the_plan_after_clearing_stale_artifacts(pack, monkeypatch, tmp_path):
    """Re-running the plan node into an existing job REUSES the saved plan - it never re-cuts, even
    when a widget changed - so a changed pairing cannot mix into the folder. To write a fresh plan
    the pairing panel's Process Plan first clears the plan node's OWN artifacts (previous plan kept
    as plan_prev.json); the same node re-run then plans afresh into that job_name."""
    import plan_store as ps

    _run(pack, monkeypatch, mode="no_reference", job_name="replace me")
    job = pathlib.Path(ps.job_dir_for(str(tmp_path / "output"), "replace me"))
    assert json.loads((job / ps.PLAN_FILENAME).read_text(encoding="utf-8"))[
        "inputs"]["unmatched_faces"] == "no_reference"

    # A bare re-run with a changed pairing reuses the saved plan and warns: it neither raises nor
    # overwrites, so the saved subject set (and the passes cut from it) stay valid.
    reused, _, report, _ = _run(pack, monkeypatch, mode="with_still", job_name="replace me")
    assert reused["inputs"]["unmatched_faces"] == "no_reference"        # the saved plan, not this run
    assert "reused the saved plan" in report
    assert "NOTICE" in report and "unmatched_faces" in report
    assert ps.plan_is_resumed(str(job))
    assert json.loads((job / ps.PLAN_FILENAME).read_text(encoding="utf-8"))[
        "inputs"]["unmatched_faces"] == "no_reference"                  # plan.json left untouched

    # Process Plan clears the plan node's artifacts first, so the node re-run writes a fresh plan.
    res = ps.clear_plan_artifacts_for_replace(str(job))
    assert res["existed"] is True and res["prev"] == ps.PLAN_PREV_FILENAME

    plan, _, _, count = _run(pack, monkeypatch, mode="with_still", job_name="replace me")
    assert count == plan["subject_count"]
    assert plan["inputs"]["unmatched_faces"] == "with_still"           # freshly planned now
    assert not ps.plan_is_resumed(str(job))                            # a real plan, not a reuse
    prev = job / ps.PLAN_PREV_FILENAME
    assert prev.is_file()
    assert json.loads(prev.read_text(encoding="utf-8"))["inputs"]["unmatched_faces"] == "no_reference"


def test_a_manual_pair_override_repairs_a_wrong_pair_and_makes_it_a_subject(pack, monkeypatch):
    """The middle clip face has no partner in the still, so with the default `leave` it is not a
    subject at all. The pairing panel's re-pairing editor would fix exactly this: a manual `1=s0`
    pairs it with a still face anyway (the positional guess need not agree), and the subject set -
    the thing the panel shows and every pass is cut against - carries the correction."""
    base, _, _, _ = _run(pack, monkeypatch, job_name="manual base")
    assert not [s for s in base["subjects"] if s["clip_face"] == 1]   # left out by `leave`

    plan, _, report, _ = _run(pack, monkeypatch, job_name="manual fixed", overrides="1=s0")

    assert (1, 0) in sorted((p["target"], p["still"]) for p in plan["pairs"])
    made = [s for s in plan["subjects"] if s["clip_face"] == 1]
    assert made and made[0]["still_face"] == 0 and made[0]["ref_kind"] == "crop"
    assert made[0]["has_reference"] is True
    assert "manual pairing" in report
    assert plan["inputs"]["pair_overrides"] == ["1=s0"]


def test_the_plan_node_plans_pairs_and_reports(pack, monkeypatch):
    """The shipped default is `leave`, so the middle face - detected on the clip, never found in
    the still - is not made a subject at all, and says so in the report."""
    plan, preview, report, count = _run(pack, monkeypatch)

    assert count == 2 == plan["subject_count"]
    assert [(s["clip_face"], s["still_face"]) for s in plan["subjects"]] == [(0, 0), (2, 1)]
    assert [s["index"] for s in plan["subjects"]] == [0, 1]
    assert [[s["clip_face"] for s in plan["left_out"]]] == [[1]]
    # The three numbers that decide whether the job is worth queuing.
    assert "2 subject(s) x 4 frame(s) @ 768x768" in report
    assert "still -> clip scale 0.500" in report
    assert "match frame 2" in report
    # Leaving is said plainly, and named, rather than left to be noticed in the finished clip.
    assert "unmatched_faces=leave" in report
    assert "clip face 1" in report
    assert "not a failure" in report
    # The preview is both plates side by side, one image, both at the clip's height.
    assert preview.dim() == 4 and preview.shape[0] == 1 and preview.shape[1] == 1000
    assert preview.shape[2] > 2000


def test_the_cap_reports_loudly_at_the_node(pack, monkeypatch):
    """#10's acceptance is that a tail cut cannot be missed. Pure tests pin the sentence; this
    pins that the node actually prints it, with the dropped faces named and the number the cap
    saved sitting next to the number it spends."""
    plan, _, report, count = _run(pack, monkeypatch, mode="no_reference", max_subjects=1)

    assert count == 1 == plan["subject_count"]
    assert [s["clip_face"] for s in plan["capped_out"]] == [1, 2]
    assert "subject cap 1: 2 face(s) capped" in report
    assert "clip face 1 (at 450, 450 on frame 2)" in report
    assert "clip face 2 (at 850, 180 on frame 2)" in report
    assert "Raise max_subjects" in report
    assert "1 subject(s) of 3 to generate; uncapped cost would be 3 subject(s) x 4 frame(s)" \
        in report


def test_the_plan_and_the_subject_references_land_in_the_job_folder(pack, monkeypatch,
                                                                    tmp_path):
    plan, _, report, _ = _run(pack, monkeypatch)
    job = tmp_path / "output" / "group shot"

    assert (job / "plan.json").is_file()
    assert (job / "subject_0.png").is_file()
    assert (job / "subject_1.png").is_file()
    # The left-out face is not a subject, so nothing is numbered after it: the subject set is
    # contiguous, which is what the passes, the manifest and the composite all count on.
    assert not (job / "subject_2.png").exists()
    assert [s["ref_file"] for s in plan["subjects"]] == ["subject_0.png", "subject_1.png"]
    # The clip face beside it (#9): a subject with a clip face on the match frame gets a clip
    # crop recorded in the plan and written out, so the panel can show both faces at once.
    assert (job / "clip_0.png").is_file()
    assert (job / "clip_1.png").is_file()
    assert [s["clip_file"] for s in plan["subjects"]] == ["clip_0.png", "clip_1.png"]

    with open(job / "plan.json", encoding="utf-8") as fh:
        written = json.load(fh)
    assert written == plan
    assert written["version"]
    assert written["run"]["job_name"] == "group shot"
    assert written["inputs"]["unmatched_faces"] == "leave"
    assert str(job) in report

    # Cut at the still's own pixels - 2x the clip's crop, not resampled to it.
    from PIL import Image
    with Image.open(job / "subject_0.png") as crop:
        assert crop.size == (600, 600)


def test_a_non_default_crop_factor_moves_every_subject_reference(pack, monkeypatch, tmp_path):
    """The tracker derives its crop rect from its own `crop_factor` widget, so a plan that
    assumes 2.5 mis-frames EVERY subject reference the moment anyone changes that widget."""
    default, _, _, _ = _run(pack, monkeypatch, job_name="factor untouched")
    tight, _, report, _ = _run(pack, monkeypatch, job_name="factor 1.5", crop_factor=1.5)

    assert tight["inputs"]["crop_factor"] == 1.5
    ps = pack.plan_store
    assert ps.subject_crop_rect(default, default["subjects"][0]) == [0.0, 10.0, 300.0, 300.0]
    assert ps.subject_crop_rect(tight, tight["subjects"][0]) == [60.0, 70.0, 180.0, 180.0]
    assert default["subjects"][0]["ref_crop"] == [0, 20, 600, 600]
    assert tight["subjects"][0]["ref_crop"] == [120, 140, 360, 360]

    from PIL import Image
    with Image.open(tmp_path / "output" / "factor 1.5" / "subject_0.png") as crop:
        assert crop.size == (360, 360)
    with Image.open(tmp_path / "output" / "factor untouched" / "subject_0.png") as crop:
        assert crop.size == (600, 600)
    # The cost line cannot claim a canvas this node never set, and the report has to say which
    # rectangle the references were actually cut with.
    assert "crop_factor 1.5" in report
    assert "estimate" in report


def test_the_untouched_default_writes_todays_bytes(pack, monkeypatch, tmp_path):
    """Touching nothing moves nothing: the new widget's default is the tracker's default, and
    an untouched graph writes the same subject reference it wrote before the widget existed."""
    _run(pack, monkeypatch, job_name="untouched")
    _run(pack, monkeypatch, job_name="explicit default", crop_factor=2.5)
    untouched = (tmp_path / "output" / "untouched" / "subject_0.png").read_bytes()
    explicit = (tmp_path / "output" / "explicit default" / "subject_0.png").read_bytes()

    assert untouched == explicit
    with open(tmp_path / "output" / "untouched" / "plan.json", encoding="utf-8") as fh:
        written = json.load(fh)
    assert written["inputs"]["crop_factor"] == pack.plan_store.DEFAULT_CROP_FACTOR
    assert written["subjects"][0]["ref_crop"] == [0, 20, 600, 600]


def test_a_different_still_into_the_same_job_folder_reuses_the_saved_plan(pack, monkeypatch):
    """Spec story 26, restated for a reuse plan node: two pairings never share one clip. Re-running
    the same job_name with a different still reuses the saved plan (the first still's pairing) and
    never re-cuts, so the folder still holds exactly one pairing - the new still is not applied."""
    first, _, _, _ = _run(pack, monkeypatch)                    # first plan, into "group shot"

    reused, _, report, _ = _run(
        pack, monkeypatch, still_boxes=[[210, 210, 410, 450], [1600, 240, 1800, 480]])

    # The saved pairing is what comes back - the different still was never written over it.
    assert [(s["clip_face"], s["still_face"]) for s in reused["subjects"]] == \
           [(s["clip_face"], s["still_face"]) for s in first["subjects"]]
    assert "reused the saved plan" in report


def test_excluded_faces_come_off_the_subjects_and_into_the_report(pack, monkeypatch):
    plan, _, report, count = _run(pack, monkeypatch, excluded="1")

    assert count == 2
    assert plan["excluded_faces"] == [1]
    assert "clip face(s) 1" in report
    assert "keep their original pixels" in report


def test_a_differently_framed_still_refuses_rather_than_pairs(pack, monkeypatch):
    """A 4:3 still against a square clip: pairing would put every face in the wrong place,
    so the node raises instead of planning on a guess."""
    with pytest.raises(ValueError) as exc:
        _run(pack, monkeypatch, still_size=(1500, 1125))
    assert "aspect" in str(exc.value)



def test_a_still_of_the_same_size_as_the_clip_pairs_and_previews_without_resizing(pack,
                                                                                 monkeypatch):
    """A still cut from the clip's own frame: factor 1, and the two plates simply sit side
    by side - the branch of the preview that needs no resize."""
    plan, preview, report, count = _run(pack, monkeypatch, still_boxes=CLIP_BOXES,
                                        still_size=(1000, 1000), job_name="same size")

    assert count == 3 == plan["subject_count"]
    assert [s["still_face"] for s in plan["subjects"]] == [0, 1, 2]
    assert plan["scale"] == pytest.approx(1.0)
    assert "still -> clip scale 1.000" in report
    assert preview.dim() == 4 and preview.shape[1] == 1000


# --- unmatched_faces: the three ways to handle a face with no partner (#8) ------------------

def test_with_still_writes_that_subject_s_reference_as_the_whole_still(pack, monkeypatch,
                                                                       tmp_path):
    """No crop for a face the still never resolved: the whole still goes in as its reference,
    written the same way a paired crop is and at the still's full size, not a crop's."""
    plan, _, report, count = _run(pack, monkeypatch, mode="with_still", job_name="whole still")
    job = tmp_path / "output" / "whole still"

    assert count == 3
    subject = [s for s in plan["subjects"] if s["clip_face"] == 1][0]
    assert subject["ref_kind"] == "still"
    assert subject["ref_crop"] == [0, 0, 2000, 2000]
    assert subject["has_reference"] is True
    assert (job / "subject_1.png").is_file()
    assert subject["ref_file"] == "subject_1.png"

    from PIL import Image
    with Image.open(job / "subject_1.png") as written:
        # The full still, at its own pixels - not the crop that could not be cut.
        assert written.size == (2000, 2000)
    # The paired subject is untouched by the mode: still a 600x600 crop out of the still.
    with Image.open(job / "subject_0.png") as crop:
        assert crop.size == (600, 600)

    # And the report warns about what whole-frame conditioning costs.
    assert "whole reference still" in report
    assert "whole-frame" in report
    assert "not a guarantee" in report


def test_no_reference_makes_a_subject_of_it_and_writes_no_reference_file(pack, monkeypatch,
                                                                        tmp_path):
    plan, _, report, count = _run(pack, monkeypatch, mode="no_reference", job_name="no ref")
    job = tmp_path / "output" / "no ref"

    assert count == 3
    subject = [s for s in plan["subjects"] if s["clip_face"] == 1][0]
    assert subject["has_reference"] is False
    assert subject["ref_kind"] == "none"
    assert subject["ref_file"] is None
    assert not (job / "subject_1.png").exists()
    assert "nothing wired into H3's reference slots" in report
    assert "weaker" in report


def test_leave_with_nobody_paired_refuses_and_names_the_widget_that_fixes_it(pack, monkeypatch):
    """Every face is unpaired, so `leave` would plan an empty job. That is a refusal, not a
    silently empty clip: the message names the two modes that would make somebody of them."""
    with pytest.raises(ValueError) as exc:
        _run(pack, monkeypatch, still_boxes=[], job_name="no partners")

    text = str(exc.value)
    assert "unmatched_faces" in text
    assert "no_reference" in text and "with_still" in text


def test_changing_the_mode_against_an_existing_job_folder_reuses_and_warns(pack, monkeypatch):
    """Same plates, same faces, same everything but the mode. Rather than orphan the passes already
    in that folder, the plan node reuses the saved plan and says the mode change was NOT applied,
    pointing at Process Plan to apply it."""
    _run(pack, monkeypatch, mode="no_reference", job_name="mode switch")
    plan, _, report, _ = _run(pack, monkeypatch, mode="with_still", job_name="mode switch")

    assert plan["inputs"]["unmatched_faces"] == "no_reference"   # the saved plan, reused
    assert "reused the saved plan" in report
    assert "NOTICE" in report
    assert "unmatched_faces" in report and "no_reference" in report and "with_still" in report


def test_the_reference_reader_hands_back_the_whole_still_for_a_whole_still_subject(pack,
                                                                                  monkeypatch,
                                                                                  tmp_path):
    """Subject Select's existing reference loading, unchanged and with no new widget: a whole
    still goes in, so a whole still comes back out, at its own size, and the note says what it
    is rather than describing a crop."""
    plan, _, _, _ = _run(pack, monkeypatch, mode="with_still", job_name="whole still sel")
    subject = [s for s in plan["subjects"] if s["ref_kind"] == "still"][0]

    reference, exists, note = pack.multi_face_nodes.load_subject_reference(
        str(tmp_path / "output" / "whole still sel"), subject)

    assert exists == 1 and reference is not None
    assert tuple(reference.shape) == (1, 2000, 2000, 3)
    assert "WHOLE reference still" in note
    assert "whole-frame" in note


def test_the_reference_reader_still_says_so_for_a_no_reference_subject(pack, monkeypatch,
                                                                      tmp_path):
    """`none` is still a normal state of a plan: nothing is invented, and the wording names the
    widget that decided it."""
    plan, _, _, _ = _run(pack, monkeypatch, mode="no_reference", job_name="no ref sel")
    subject = [s for s in plan["subjects"] if s["ref_kind"] == "none"][0]

    reference, exists, note = pack.multi_face_nodes.load_subject_reference(
        str(tmp_path / "output" / "no ref sel"), subject)

    assert reference is None and exists == 0
    assert "no subject reference" in note
    assert "unmatched_faces" in note


def test_a_tiny_saved_reference_is_lifted_to_h3s_floor(pack, tmp_path):
    """A job folder cut before the crop floor (or a still smaller than it) still reaches a pass
    as a too-small PNG. H3 crashes on a reference whose /16 latent is under its 7-wide patch
    conv, so the reader lifts the IMAGE to the floor and says so, rather than handing H3 a crop
    it cannot encode."""
    import os

    import numpy as np
    from PIL import Image

    job = tmp_path / "output" / "tiny ref"
    os.makedirs(str(job), exist_ok=True)
    # A 30x40 crop: its /16 latent (~2-3) is under H3's 7-wide patch conv.
    Image.fromarray((np.full((40, 30, 3), 128, np.uint8))).save(str(job / "subject_0.png"))
    subject = {"index": 0, "ref_file": "subject_0.png", "ref_kind": "crop"}

    reference, exists, note = pack.multi_face_nodes.load_subject_reference(str(job), subject)

    assert exists == 1 and reference is not None
    h, w = int(reference.shape[1]), int(reference.shape[2])
    edge = pack.plan_store.MIN_H3_REF_EDGE
    assert h >= edge and w >= edge and min(h, w) == edge
    assert "under H3's minimum reference size" in note

