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
    if importlib.util.find_spec("folder_paths") is None:
        paths = types.ModuleType("folder_paths")
        paths.get_output_directory = lambda: str(tmp_path / "output")
        paths.models_dir = str(tmp_path / "models")

        def _no_models(key):
            raise RuntimeError("no model folders in this test")

        paths.get_filename_list = _no_models
        paths.get_full_path = lambda key, name: None
        monkeypatch.setitem(sys.modules, "folder_paths", paths)

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
         excluded="", job_name="group shot", source="union"):
    fake = _FakeModel([_Found(CLIP_BOXES, CLIP_CONFS), _Found(still_boxes, STILL_CONFS)])
    monkeypatch.setattr(pack.nodes, "_load_detector", lambda name: fake)
    node = pack.NODE_CLASS_MAPPINGS["H3MultiFacePlan"]()
    images = torch.zeros((4, 1000, 1000, 3))
    images.h3_video_name = "shot_07.mp4"
    still = torch.zeros((1, int(still_size[1]), int(still_size[0]), 3))
    return node.run(images, still, "face_yolov8m.pt", 0.35, 2, source, 0.05, excluded,
                    0, job_name)


def test_the_plan_node_plans_pairs_and_reports(pack, monkeypatch):
    plan, preview, report, count = _run(pack, monkeypatch)

    assert count == 3 == plan["subject_count"]
    assert [(s["clip_face"], s["still_face"]) for s in plan["subjects"]] == [
        (0, 0), (1, None), (2, 1)]
    # The three numbers that decide whether the job is worth queuing.
    assert "3 subject(s) x 4 frame(s) @ 768x768" in report
    assert "still -> clip scale 0.500" in report
    assert "match frame 2" in report
    # The preview is both plates side by side, one image, both at the clip's height.
    assert preview.dim() == 4 and preview.shape[0] == 1 and preview.shape[1] == 1000
    assert preview.shape[2] > 2000


def test_the_plan_and_the_subject_references_land_in_the_job_folder(pack, monkeypatch,
                                                                    tmp_path):
    plan, _, report, _ = _run(pack, monkeypatch)
    job = tmp_path / "output" / "group shot"

    assert (job / "plan.json").is_file()
    assert (job / "subject_0.png").is_file()
    assert (job / "subject_2.png").is_file()
    # The middle face found no partner, so there is no crop of it to write.
    assert not (job / "subject_1.png").exists()
    assert [s["ref_file"] for s in plan["subjects"]] == ["subject_0.png", None,
                                                        "subject_2.png"]

    with open(job / "plan.json", encoding="utf-8") as fh:
        written = json.load(fh)
    assert written == plan
    assert written["version"]
    assert written["run"]["job_name"] == "group shot"
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

