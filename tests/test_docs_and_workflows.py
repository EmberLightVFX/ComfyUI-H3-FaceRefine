"""The release guard: README, example workflows and tooltips cannot drift apart from the nodes.

AGENTS.md states the rule - "a node change that doesn't update `README.md` is incomplete" - and
for five tickets that rule was honoured in prose only, which is how a node can ship undocumented
and pass review. Everything here is a fact that can be checked with no ComfyUI, no torch and no
weights: what is registered, what the README has a heading for, which node types the shipped
workflows ask for, and which inputs carry a tooltip.

The core node types cannot be checked from here: they belong to ComfyUI, to VideoHelperSuite and
to ComfyUI-GGUF, none of which is importable in this run. They are therefore listed and skipped
by name below, so the assertion that bites is the one about this pack's own nodes, and the skip is
stated rather than hidden.
"""

import ast
import json
import pathlib

import multi_face_nodes as m

PACK_ROOT = pathlib.Path(__file__).resolve().parents[1]
README = (PACK_ROOT / "README.md").read_text(encoding="utf-8")
EXAMPLE_DIR = PACK_ROOT / "example_workflows"

# The five nodes this feature added. Each one has to be documented, and every one of its inputs
# has to explain itself - they are the nodes a person meets for the first time in 1.2.0.
NEW_NODES = ("H3MultiFacePlan", "H3SubjectSelect", "H3PassSave", "H3PassLoad",
             "H3MultiFaceComposite")

# Node types that are not this pack's to check, and are named here rather than quietly allowed.
# ComfyUI core, the MiniMax H3 nodes ComfyUI ships, VideoHelperSuite and ComfyUI-GGUF: none of
# them is importable in a run with no ComfyUI installed, so their registrations cannot be read.
FOREIGN_NODE_TYPES = {
    # ComfyUI core
    "BasicGuider", "BasicScheduler", "CLIPLoader", "KSamplerSelect", "LoadImage",
    "LoraLoaderModelOnly", "Note", "RandomNoise", "SAMLoader", "SamplerCustomAdvanced",
    "UNETLoader", "VAEDecode", "VAELoader",
    # ComfyUI core, MiniMax H3
    "MiniMaxH3NativeAudioLock", "MiniMaxH3ReferenceToVideo",
    # VideoHelperSuite
    "VHS_LoadAudioUpload", "VHS_LoadVideoPath", "VHS_VideoCombine",
    # ComfyUI-GGUF
    "CLIPLoaderGGUF", "UnetLoaderGGUF",
}


def _registered(module_name: str) -> dict:
    """``{type: display name}`` for the nodes one module registers, read out of its source.

    Source, not import: `nodes.py` imports ComfyUI at module level, so it cannot be loaded in a
    run with no ComfyUI installed. The class mapping's values are the classes themselves, so only
    its keys can be read literally; the display mapping's values are plain strings, and the two
    sets of keys have to agree for either read to mean anything.
    """
    tree = ast.parse((PACK_ROOT / module_name).read_text(encoding="utf-8"))
    types = None
    display = None
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            continue
        if node.targets[0].id == "NODE_CLASS_MAPPINGS":
            types = [key.value for key in node.value.keys]
        elif node.targets[0].id == "NODE_DISPLAY_NAME_MAPPINGS":
            display = {ast.literal_eval(key): ast.literal_eval(value)
                       for key, value in zip(node.value.keys, node.value.values)}
    assert types is not None and display is not None, module_name
    assert set(types) == set(display), module_name
    return display


def registered_nodes() -> dict:
    """``{type: display name}`` for every node the pack registers, released and new."""
    released = _registered("nodes.py")
    new = _registered("multi_face_nodes.py")
    assert set(new) == set(m.NODE_CLASS_MAPPINGS)
    return {**released, **new}



def readme_section(display_name: str) -> str:
    """The README block belonging to one node, by its `## <display name>` heading."""
    heading = f"\n## {display_name}\n"
    start = README.find(heading)
    if start < 0:
        return ""
    start += 1
    end = README.find("\n## ", start + len(heading))
    return README[start:end if end > 0 else len(README)]


def test_every_registered_node_has_a_display_name():
    """A type with no display name shows up in the node menu as its class name."""
    for node_type, display in registered_nodes().items():
        assert display and display != node_type, node_type


def test_every_registered_node_has_its_own_readme_section():
    """The drift guard AGENTS.md asks for: add a node, forget the README, fail a test."""
    for node_type, display in registered_nodes().items():
        assert readme_section(display), f"{node_type} has no README section for {display!r}"


def test_every_widget_on_the_new_nodes_carries_a_tooltip():
    """Every input of the five new nodes explains itself in the node, not only in the README."""
    for node_type in NEW_NODES:
        schema = getattr(m, node_type).INPUT_TYPES()
        for group in ("required", "optional"):
            for name, spec in (schema.get(group) or {}).items():
                config = spec[1] if len(spec) > 1 else {}
                assert isinstance(config, dict) and config.get("tooltip"), \
                    f"{node_type}.{name} has no tooltip"


def test_every_input_of_the_new_nodes_is_documented_in_its_section():
    """A widget added to a new node with no README line is a test failure, not a review comment."""
    for node_type in NEW_NODES:
        display = registered_nodes()[node_type]
        section = readme_section(display)
        schema = getattr(m, node_type).INPUT_TYPES()
        for group in ("required", "optional"):
            for name in (schema.get(group) or {}):
                assert f"`{name}`" in section, f"{display}: `{name}` is not in the README"


def test_the_example_workflows_are_the_ones_the_readme_promises():
    """The README names its templates; the directory has to hold exactly those."""
    shipped = sorted(path.name for path in EXAMPLE_DIR.glob("*.json"))
    assert "H3_Face_Refine_Multi_Face.json" in shipped
    for name in shipped:
        assert name in README, f"{name} is shipped but never mentioned in the README"


def test_each_example_workflow_parses_and_uses_known_nodes():
    """A template that loads with red nodes is worse than no template."""
    for path in sorted(EXAMPLE_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as handle:
            workflow = json.load(handle)
        assert workflow.get("nodes"), path.name
        for node in workflow["nodes"]:
            node_type = node["type"]
            if node_type in FOREIGN_NODE_TYPES:
                continue                                    # not this pack's to check
            assert node_type in registered_nodes(), f"{path.name}: {node_type} is not registered"


def test_the_multi_face_example_wires_the_chain_the_readme_describes():
    """The template demonstrates the pass model, so it has to contain the nodes that make it."""
    with open(EXAMPLE_DIR / "H3_Face_Refine_Multi_Face.json", encoding="utf-8") as handle:
        workflow = json.load(handle)
    used = {node["type"] for node in workflow["nodes"]}
    for node_type in NEW_NODES + ("H3FaceSelect", "H3FaceTrackCrop", "H3FaceStitch"):
        assert node_type in used, node_type


def test_the_pack_version_has_a_whats_new_entry():
    """A version with no notes in the README is a version nobody can read their way into."""
    text = (PACK_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    version = [line.split("=")[1].strip().strip('"')
               for line in text.splitlines() if line.startswith("version =")][0]
    assert f"## What's new in {version}" in README, version
