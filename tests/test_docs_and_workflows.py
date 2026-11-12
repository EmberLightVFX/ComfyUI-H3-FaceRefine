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
import re

import multi_face_nodes as m

PACK_ROOT = pathlib.Path(__file__).resolve().parents[1]
README = (PACK_ROOT / "README.md").read_text(encoding="utf-8")
AGENTS = (PACK_ROOT / "AGENTS.md").read_text(encoding="utf-8")
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


# ------------------------------------------------------------------------ AGENTS.md
#
# The same drift that left a node undocumented left the agent-facing doc describing a pack that
# no longer exists - "no test suite" over a filling tests/, a node count that stopped being the
# pack's. These two check the doc against counts read from the repository, never against numbers
# written down by hand, so the guard itself cannot go stale.

def test_agents_md_admits_the_pack_has_a_test_suite():
    """A doc that denies its own suite is worse than no doc: an agent who believes it writes no
    test for the change they were just sent to make."""
    test_files = sorted(path.name for path in (PACK_ROOT / "tests").glob("test_*.py"))
    assert test_files, "tests/ holds no test files, so a doc saying so would be accurate"
    assert "no test suite" not in AGENTS.lower(), \
        f"AGENTS.md denies the suite while tests/ holds {len(test_files)} test files"


def test_agents_md_does_not_undercount_the_nodes_it_describes():
    """The pack registers more node classes than the frozen `nodes.py` holds, so a doc that
    describes the released count as the pack's count sends an agent looking in the wrong module."""
    total = len(registered_nodes())
    released = len(_registered("nodes.py"))
    assert total > released, f"nothing beyond the released {released} to guard against"
    assert f"{released} node classes" not in AGENTS, \
        f"AGENTS.md calls the released {released} the whole pack; it registers {total}"
    for claim in re.finditer(r"(\d+)\s+node classes?", AGENTS):
        assert int(claim.group(1)) == total, \
            f"AGENTS.md claims {claim.group(0)!r}; the pack registers {total}"


# --------------------------------------------------------------- template model paths
#
# A loader widget's stored value is resolved against the folder that loader searches, so a value
# like `MiniMax H3\file.safetensors` on a VAELoader asks for `models/vae/MiniMax H3/file.safetensors`.
# Every shipped template carried exactly that on four active loaders, for two releases: the files
# were installed correctly, in the right folder, under the right name, and the templates still
# refused to load. Nothing caught it, because the README already warns that you may have to
# re-pick loaders after opening a template - so a validation error reads as expected housekeeping
# rather than as a defect in the shipped files.
#
# Muted nodes are deliberately NOT checked (ComfyUI marks a muted node with a non-zero `mode`).
# The GGUF and SAM pairs ship muted on purpose, the README says so, and their stored values name
# quantisations the pack does not recommend - guarding them would force the README to document
# optional files it pointedly leaves out.

MODEL_LOADERS = {
    "UNETLoader": 0, "VAELoader": 0, "CLIPLoader": 0,
    "LoraLoaderModelOnly": 0, "CheckpointLoaderSimple": 0,
}


def _muted(node) -> bool:
    """ComfyUI's mute flag: absent or zero means the node runs."""
    return bool(node.get("mode", 0))


def _active_loader_paths(workflow) -> list:
    """`(stored value, node type)` for every model path a running loader node holds."""
    found = []
    for node in workflow.get("nodes") or []:
        index = MODEL_LOADERS.get(node.get("type"))
        if index is None or _muted(node):
            continue
        widgets = node.get("widgets_values") or []
        assert index < len(widgets), f"{node.get('type')}: no widget at {index}"
        value = widgets[index]
        assert isinstance(value, str), f"{node.get('type')}: {value!r} is not a model name"
        found.append((value, node["type"]))
    return found


def test_shipped_workflows_store_bare_model_names_the_readme_names():
    """A subfolder inside a stored model name asks for a folder no installer creates.

    The README's model table is flat - one folder per loader type, model names written bare. A
    template that files a model under a subfolder of that folder resolves to a path that cannot
    exist, which is the bug #12 is about.
    """
    checked = 0
    for path in sorted(EXAMPLE_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as handle:
            workflow = json.load(handle)
        for value, node_type in _active_loader_paths(workflow):
            checked += 1
            assert "\\" not in value and "/" not in value, (
                f"{path.name}: {node_type} stores {value!r}. A model name in a shipped template "
                f"is bare: the loader's own folder is the only folder it may carry.")
            assert value in README, (
                f"{path.name}: {node_type} stores {value!r}, which the README never names - so "
                f"the template asks for a file the install guide does not even tell you to fetch.")
    # A guard that has quietly stopped matching any node is worse than no guard at all.
    assert checked >= 12, f"only {checked} active loader paths found across the templates"


def test_the_templates_all_load_the_same_text_encoder():
    """The text encoder is the one stored path that was always right, so pin it.

    Without this, the guard above would still pass if one template drifted to a different encoder
    than the others, and a reader comparing templates would have no idea which one the pack means.
    """
    encoders = set()
    for path in sorted(EXAMPLE_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as handle:
            workflow = json.load(handle)
        for value, node_type in _active_loader_paths(workflow):
            if node_type == "CLIPLoader":
                encoders.add(value)
    assert encoders, "no active CLIPLoader in any shipped template"
    assert len(encoders) == 1, f"the shipped templates disagree about the text encoder: {encoders}"
    assert next(iter(encoders)) in README, encoders


def test_a_muted_node_is_not_checked():
    """The GGUF and SAM pairs ship muted on purpose, so their paths are not the pack's promise."""
    workflow = {"nodes": [
        {"type": "VAELoader", "mode": 2, "widgets_values": ["MiniMax H3\\gone.safetensors"]},
        {"type": "VAELoader", "widgets_values": ["minimax_h3_video_vae_fp16.safetensors"]},
    ]}
    assert _active_loader_paths(workflow) == [("minimax_h3_video_vae_fp16.safetensors",
                                               "VAELoader")]

