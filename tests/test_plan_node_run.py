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
         excluded="", job_name="group shot", source="union", mode="leave"):
    fake = _FakeModel([_Found(CLIP_BOXES, CLIP_CONFS), _Found(still_boxes, STILL_CONFS)])
    monkeypatch.setattr(pack.nodes, "_load_detector", lambda name: fake)
    node = pack.NODE_CLASS_MAPPINGS["H3MultiFacePlan"]()
    images = torch.zeros((4, 1000, 1000, 3))
    images.h3_video_name = "shot_07.mp4"
    still = torch.zeros((1, int(still_size[1]), int(still_size[0]), 3))
    return node.run(images, still, "face_yolov8m.pt", 0.35, 2, source, 0.05, excluded,
                    0, job_name, mode)


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


def test_changing_the_mode_against_an_existing_job_folder_refuses(pack, monkeypatch):
    """Same plates, same faces, same everything but the mode - and the mode decided who the
    subjects are, so the passes already in that folder are a different job's."""
    _run(pack, monkeypatch, mode="no_reference", job_name="mode switch")
    with pytest.raises(ValueError) as exc:
        _run(pack, monkeypatch, mode="with_still", job_name="mode switch")

    text = str(exc.value)
    assert "unmatched_faces" in text
    assert "no_reference" in text and "with_still" in text


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

