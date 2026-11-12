"""The node contract: schema, return types, widget order, registration.

ComfyUI stores widget values **positionally** (ADR 0002), so a widget inserted in the middle
of a node's list silently reinterprets what people have already saved. That makes the order
of `INPUT_TYPES()` a fact to pin down rather than an implementation detail - and it is a
fact no amount of eyeballing catches.

These tests also pin the import boundary: `multi_face_nodes` and `plan_store` are imported
here with only numpy installed, because that is what CI has. Everything needing ComfyUI,
torch or the detector lives inside `run()`, not at module level.
"""

import ast
import inspect
import pathlib

import multi_face_nodes as m
import plan_store as ps

PACK_ROOT = pathlib.Path(__file__).resolve().parents[1]

WIDGET_ORDER = ["detector", "confidence", "match_frame", "subject_source",
                "match_tolerance", "excluded_faces", "max_subjects", "job_name"]

SUBJECT_WIDGET_ORDER = ["subject"]


def test_exactly_one_new_custom_type_lands_with_the_feature():
    """ADR 0005: one new type, the subject plan. Every pass reuses the existing shapes."""
    assert m.H3FACEPLAN == "H3FACEPLAN"
    assert m.CUSTOM_TYPES == (m.H3FACEPLAN,)


def test_the_plan_node_is_registered_in_its_own_module():
    assert m.NODE_CLASS_MAPPINGS["H3MultiFacePlan"] is m.H3MultiFacePlan
    assert m.NODE_DISPLAY_NAME_MAPPINGS["H3MultiFacePlan"] == "H3 Multi-Face Plan"


def test_the_pack_init_wires_the_new_node_and_the_new_type():
    """`__init__.py` cannot be imported without ComfyUI, so the wiring is checked where it
    is written: it has to merge the new module's mappings, not replace the released ones."""
    source = (PACK_ROOT / "__init__.py").read_text(encoding="utf-8")
    assert "from .multi_face_nodes import" in source
    assert "NODE_CLASS_MAPPINGS" in source and "NODE_DISPLAY_NAME_MAPPINGS" in source
    assert "CUSTOM_TYPES" in source
    assert "WEB_DIRECTORY" in source


def test_the_plan_node_s_returns_are_the_plan_a_preview_a_report_and_a_count():
    node = m.H3MultiFacePlan
    assert node.RETURN_TYPES == (m.H3FACEPLAN, "IMAGE", "STRING", "INT")
    assert node.RETURN_NAMES == ("plan", "preview", "report", "subject_count")
    assert len(node.RETURN_TYPES) == len(node.RETURN_NAMES)
    # The preview and the report are meant to be read off the node itself, before anything
    # is queued, so the node is an output node.
    assert node.OUTPUT_NODE is True
    assert node.FUNCTION == "run"
    assert node.CATEGORY == m.CATEGORY == "MiniMax H3/Face Refine"


def test_the_inputs_and_widgets_are_in_this_exact_order():
    """Permanent the moment anyone saves a workflow: ComfyUI matches widgets by position."""
    schema = m.H3MultiFacePlan.INPUT_TYPES()
    assert list(schema["required"]) == ["images", "reference_still"] + WIDGET_ORDER
    assert not schema.get("optional")


def test_the_plates_are_images_and_every_widget_is_a_typed_one():
    schema = m.H3MultiFacePlan.INPUT_TYPES()["required"]
    assert schema["images"][0] == "IMAGE"
    assert schema["reference_still"][0] == "IMAGE"
    for name in WIDGET_ORDER:
        choice = schema[name][0]
        if name in ("detector", "subject_source"):
            # A combo has to offer options, or the prompt fails validation on load.
            assert isinstance(choice, (list, tuple)) and len(choice) > 0, name
        else:
            assert choice in ("STRING", "INT", "FLOAT"), name
    assert schema["excluded_faces"][0] == "STRING"     # editable with no front end
    assert schema["job_name"][0] == "STRING"


def test_the_widget_defaults_are_what_the_feature_promises():
    schema = m.H3MultiFacePlan.INPUT_TYPES()["required"]
    assert list(schema["subject_source"][0]) == list(ps.SUBJECT_SOURCES)
    assert schema["subject_source"][1]["default"] == "union"
    assert schema["match_frame"][1]["default"] == 0
    assert schema["max_subjects"][1]["default"] == 0     # 0 = no cap
    assert schema["excluded_faces"][1]["default"] == ""
    assert schema["job_name"][1]["default"] == ""        # derived from the clip
    assert schema["confidence"][1]["default"] == 0.35    # the pack's detector floor
    assert schema["detector"][0]                        # a combo always needs an option


def test_the_subject_node_is_registered_in_its_own_module():
    assert m.NODE_CLASS_MAPPINGS["H3SubjectSelect"] is m.H3SubjectSelect
    assert m.NODE_DISPLAY_NAME_MAPPINGS["H3SubjectSelect"] == "H3 Subject Select"
    # ADR 0006: the new nodes ride in on the module's mappings, which `__init__.py` merges -
    # so H3FACEPLAN is registered once, by multi_face_nodes, never hand-written here.
    source = (PACK_ROOT / "__init__.py").read_text(encoding="utf-8")
    assert "from .multi_face_nodes import CUSTOM_TYPES" in source
    assert '"H3FACEPLAN":' not in source


def test_the_subject_node_emits_the_released_pick_type_not_a_new_one():
    """ADR 0005: the pass reuses `H3FACEPICK`, so the released tracker, denoise and stitch
    nodes run inside it. A second pick type here would fork the pipeline."""
    node = m.H3SubjectSelect
    assert m.H3FACEPICK == "H3FACEPICK"
    assert m.H3FACEPICK not in m.CUSTOM_TYPES     # only H3FACEPLAN is new
    assert node.RETURN_TYPES == (m.H3FACEPICK, "IMAGE", "INT", "INT", "STRING")
    assert node.RETURN_NAMES == ("face_pick", "subject_reference", "reference_exists",
                                 "subject_index", "report")
    assert len(node.RETURN_TYPES) == len(node.RETURN_NAMES)
    assert node.FUNCTION == "run"
    assert node.CATEGORY == m.CATEGORY


def test_the_subject_inputs_are_a_plan_and_the_clip_s_own_pick():
    """Detection stays once in the user's graph: the plan says WHO the subject is, the payload
    carries every frame's boxes. Neither is recomputed here."""
    schema = m.H3SubjectSelect.INPUT_TYPES()
    assert list(schema["required"]) == ["plan", "face_pick"] + SUBJECT_WIDGET_ORDER
    assert not schema.get("optional")
    assert schema["required"]["plan"][0] == m.H3FACEPLAN
    assert schema["required"]["face_pick"][0] == m.H3FACEPICK
    # ADR 0003: the reference still is not an input, and there is no identity input to wire
    # the subject reference into. The tracker anchors the subject from the target clip.
    for banned in ("identity_reference", "reference_still", "images", "still"):
        assert banned not in schema["required"]


def test_the_subject_widget_is_a_combo_with_next_first_and_as_the_default():
    schema = m.H3SubjectSelect.INPUT_TYPES()["required"]["subject"]
    choices = schema[0]
    assert isinstance(choices, (list, tuple)) and len(choices) > 1
    assert choices[0] == "next"
    assert schema[1]["default"] == "next"
    # The rest are subject numbers, matching the numbers the plan's report prints.
    assert list(choices[1:]) == [str(i) for i in range(len(choices) - 1)]


def test_the_subject_run_takes_its_inputs_in_schema_order():
    """ComfyUI calls FUNCTION with the required inputs positionally, so a reordered parameter
    reads a plan as a pick."""
    schema = m.H3SubjectSelect.INPUT_TYPES()
    args = [p for p in inspect.signature(m.H3SubjectSelect.run).parameters if p != "self"]
    assert args == list(schema["required"])


def _module_level_imports(path) -> set[str]:
    """Absolute top-level module names imported outside any function or class.

    Anything a call pulls in is fair game - only the import at module scope decides what a
    test needs installed.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()

    def walk(node) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(child, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in child.names)
            elif isinstance(child, ast.ImportFrom):
                if not child.level and child.module:
                    roots.add(child.module.split(".")[0])
            walk(child)

    walk(tree)
    return roots


def test_the_plan_module_stays_pure():
    """The whole reason the pairing is testable with no GPU, no weights and no ComfyUI.

    A stray `import folder_paths` in here would quietly take the geometry hostage to the
    node machinery.
    """
    roots = _module_level_imports(PACK_ROOT / "plan_store.py")
    assert roots <= {"__future__", "numpy", "json", "os"}, sorted(roots)


def test_the_node_module_defers_everything_that_needs_comfyui():
    """`INPUT_TYPES()` is callable with no ComfyUI installed, which is what lets the
    contract above run in CI."""
    roots = _module_level_imports(PACK_ROOT / "multi_face_nodes.py")
    forbidden = {"comfy", "folder_paths", "torch", "ultralytics", "nodes", "PIL"}
    assert not (roots & forbidden), sorted(roots & forbidden)

