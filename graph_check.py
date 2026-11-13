"""Repo-side graph checker for H3 FaceRefine workflows.

The five faults found in the user's live multi-face graph are all *type-correct and
schema-legal*: ComfyUI's ``validate_workflow`` passes them, because they wire the right
socket types together. What makes them wrong is a **pack invariant** — the kind of model
that belongs in a given socket, which stage a Pass Save is recording, whether the three
detector widgets agree — that no generic validator knows. This module encodes those
invariants so a saved graph can be checked without ever loading ComfyUI, torch, or a
weight file.

Like ``plan_store.py`` this is a **pure module**: ``json``, ``re`` and the pure
``plan_store`` sibling, never ``comfy``. It reads the front-end (UI-export) graph format — the shape every
``example_workflows/*.json`` and every saved workflow under ``user/default/workflows`` is
stored in — where a node's wired sockets appear as ``inputs[].link`` and ``links`` are
``[link_id, origin_node, origin_slot, target_node, target_slot, type]``.

The single public seam is ``check_graph(workflow) -> list[Problem]``; callers observe the
returned problems (each with a ``code`` and the ``node_ids`` it names) and nothing about
how they were found.
"""

from collections import namedtuple
import json
import re

import plan_store  # pure sibling (numpy/json/os only) — the source of CHAINED_TO_CLIP

Problem = namedtuple("Problem", ["code", "message", "node_ids"])


def load_workflow(path):
    """Read one saved workflow file into the dict ``check_graph`` consumes."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


# --- graph substrate (not behaviour under test) -------------------------------
# These turn the front-end export into something the invariant checks can walk. They
# only mirror how ComfyUI itself stores a graph; they decide nothing about the pack.

def _nodes_by_id(workflow):
    return {node["id"]: node for node in (workflow.get("nodes") or [])}


def _links_by_id(workflow):
    return {link[0]: link for link in (workflow.get("links") or [])}


def _is_active(node):
    """ComfyUI's mute/bypass flag: absent or zero means the node runs."""
    return not node.get("mode", 0)


def _socket_link(node, socket_name):
    """The link id wired into ``socket_name``, or None (also None if no such input)."""
    for inp in node.get("inputs") or []:
        if inp["name"] == socket_name:
            return inp.get("link")
    return None


def _source_node(node, socket_name, nodes, links):
    """The node feeding ``socket_name``, or None when that socket is unwired."""
    link_id = _socket_link(node, socket_name)
    if link_id is None:
        return None
    link = links.get(link_id)
    if link is None:
        return None
    return nodes.get(link[1])


def _consumers(node, out_name, nodes, links):
    """`[(target_node, target_input_name)]` for every wire leaving ``out_name``."""
    found = []
    for out in node.get("outputs") or []:
        if out["name"] != out_name:
            continue
        for link_id in (out.get("links") or []):
            link = links.get(link_id)
            if link is None:
                continue
            target = nodes.get(link[3])
            if target is None:
                continue
            slot = link[4]
            inputs = target.get("inputs") or []
            in_name = inputs[slot]["name"] if 0 <= slot < len(inputs) else None
            found.append((target, in_name))
    return found


def _subject_number(node, nodes, links):
    """The subject a pass node acts on. Its ``subject`` socket is wired from H3SubjectSelect,
    so follow that link and read the select's own subject widget; fall back to this node's
    own subject widget (position 0 on both Pass Save and Pass Load)."""
    src = _source_node(node, "subject", nodes, links)
    if src is not None:
        widgets = src.get("widgets_values") or []
        if widgets:
            try:
                return int(str(widgets[0]))
            except ValueError:
                return None
    widgets = node.get("widgets_values") or []
    if widgets:
        try:
            return int(str(widgets[0]))
        except ValueError:
            return None
    return None


# --- invariants --------------------------------------------------------------

# (node type, socket name) pairs, split by the kind of VAE the pack expects to receive.
# A video-decoding socket wired to an audio VAE decodes video latents through an audio
# VAE — the graph runs, the output is garbage, and no schema check notices.
_VIDEO_VAE_SOCKETS = frozenset({
    ("VAEDecode", "vae"),
    ("H3InjectVideoLatent", "vae"),
    ("MiniMaxH3ReferenceToVideo", "vae"),
})
_AUDIO_VAE_SOCKETS = frozenset({
    ("MiniMaxH3ReferenceToVideo", "audio_vae"),
    ("MiniMaxH3NativeAudioLock", "audio_vae"),
})


def _vae_kind(model_name):
    """`'video'`, `'audio'`, or None — read from the file name the loader was given."""
    name = (model_name or "").lower()
    if "audio" in name:
        return "audio"
    if "video" in name:
        return "video"
    return None


def _check_vae_kind(workflow):
    nodes = _nodes_by_id(workflow)
    links = _links_by_id(workflow)
    problems = []
    for node in workflow.get("nodes") or []:
        if not _is_active(node):
            continue
        for inp in node.get("inputs") or []:
            socket = (node.get("type"), inp["name"])
            want = "video" if socket in _VIDEO_VAE_SOCKETS else (
                "audio" if socket in _AUDIO_VAE_SOCKETS else None)
            if want is None:
                continue
            src = _source_node(node, inp["name"], nodes, links)
            if src is None or src.get("type") != "VAELoader":
                continue
            model = (src.get("widgets_values") or [None])[0]
            kind = _vae_kind(model)
            if kind is not None and kind != want:
                problems.append(Problem(
                    "VAE_KIND",
                    f"{node['type']}.{inp['name']} is fed {model!r}, a {kind} VAE; this "
                    f"socket needs the {want} VAE.",
                    tuple(sorted([node["id"], src["id"]]))))
    return problems


# H3PassSave's `images` socket is fed by one of two places, and only one stage is ever
# correct for each. Crops come from the tracker, stitched frames come from the stitch.
# Wiring the stitched output into a Pass Save that still says `crops` is the fault: the
# pass never completes and its frames overwrite the crops the job resumes from.
_STAGE_FOR_FEEDER = {
    "H3FaceTrackCrop": "crops",
    "H3FaceStitch": "stitched",
}


def _check_pass_save_stage(workflow):
    nodes = _nodes_by_id(workflow)
    links = _links_by_id(workflow)
    problems = []
    for node in workflow.get("nodes") or []:
        if not _is_active(node) or node.get("type") != "H3PassSave":
            continue
        widgets = node.get("widgets_values") or []
        if len(widgets) < 2:
            continue
        # widgets_values for H3PassSave is [subject, stage]; skip when `stage` itself is
        # driven by a link, because then the stored widget is not what the node runs on.
        if _socket_link(node, "stage") is not None:
            continue
        stage = widgets[1]
        src = _source_node(node, "images", nodes, links)
        if src is None:
            continue
        want = _STAGE_FOR_FEEDER.get(src.get("type"))
        if want is not None and stage != want:
            problems.append(Problem(
                "PASS_SAVE_STAGE",
                f"H3PassSave (subject {widgets[0]}) records stage={stage!r} but its images "
                f"come from {src['type']}, which is always stage={want!r}.",
                tuple(sorted([node["id"], src["id"]]))))
    return problems


# The picker, the plan and the tracker each carry their own detector + confidence widgets
# at fixed positions. They detect on the same two plates, so they must run the same
# detector at the same score floor — otherwise "which faces exist" is answered three
# different ways and the picker offers a set the plan never agreed to.
# {node type: (detector widget index, confidence widget index)}.
_DETECTOR_WIDGET = {
    "H3FaceSelect": (1, 2),
    "H3FaceTrackCrop": (0, 1),
    "H3MultiFacePlan": (0, 1),
}


def _check_detector_consistency(workflow):
    by_value = {}   # detector name -> node ids using it
    by_conf = {}    # confidence value -> node ids using it
    for node in workflow.get("nodes") or []:
        if not _is_active(node):
            continue
        idx = _DETECTOR_WIDGET.get(node.get("type"))
        if idx is None:
            continue
        det_idx, conf_idx = idx
        widgets = node.get("widgets_values") or []
        if len(widgets) > det_idx and _socket_link(node, "detector") is None:
            by_value.setdefault(widgets[det_idx], []).append(node["id"])
        if len(widgets) > conf_idx and _socket_link(node, "confidence") is None:
            by_conf.setdefault(widgets[conf_idx], []).append(node["id"])

    problems = []
    if len(by_value) > 1:
        problems.append(Problem(
            "DETECTOR_CONSISTENCY",
            "the face nodes disagree about the detector: "
            + "; ".join(f"{d!r} on {sorted(ids)}" for d, ids in sorted(by_value.items(), key=lambda kv: str(kv[0])))
            + " — the plates are paired positionally, so one detector has to answer for all of them.",
            tuple(sorted(nid for ids in by_value.values() for nid in ids))))
    if len(by_conf) > 1:
        problems.append(Problem(
            "CONFIDENCE_CONSISTENCY",
            "the face nodes disagree about the detector confidence: "
            + "; ".join(f"{c!r} on {sorted(ids)}" for c, ids in sorted(by_conf.items(), key=lambda kv: str(kv[0])))
            + " — a different floor means a different face set at the match frame.",
            tuple(sorted(nid for ids in by_conf.values() for nid in ids))))
    return problems


# A turbo LoRA is trained for one step count; the BasicScheduler that walks the schedule has
# to use the same number. The LoRA's own file name is the independent source of truth ("the
# 8-step one"), so a scheduler that disagrees with the weights it loads is a real fault even
# though both are INT-legal widgets.
_STEP_IN_NAME = re.compile(r"(\d+)\s*[-_]?\s*steps?", re.IGNORECASE)


def _check_scheduler_steps(workflow):
    # The step count the loaded LoRAs declare. Enforced only when they agree on exactly one
    # number — two different declared counts, or none at all, leaves no single answer.
    declared = set()
    for node in workflow.get("nodes") or []:
        if not _is_active(node) or node.get("type") != "LoraLoaderModelOnly":
            continue
        model = (node.get("widgets_values") or [None])[0]
        match = _STEP_IN_NAME.search(model or "")
        if match:
            declared.add(int(match.group(1)))
    if len(declared) != 1:
        return []
    want = declared.pop()

    problems = []
    for node in workflow.get("nodes") or []:
        if not _is_active(node) or node.get("type") != "BasicScheduler":
            continue
        widgets = node.get("widgets_values") or []
        if len(widgets) > 1 and isinstance(widgets[1], int) and widgets[1] != want:
            problems.append(Problem(
                "SCHEDULER_STEPS",
                f"BasicScheduler runs {widgets[1]} steps, but the loaded LoRA names the "
                f"{want}-step one; the LoRA and the step count have to agree.",
                (node["id"],)))
    return problems


# Nodes that legitimately end a branch. Renderers and notes have no consumer by design; the
# pack's checkpoint nodes (Pass Save / Pass Load) and the plan/composite are OUTPUT_NODE, so
# they run even with nothing wired and refuse or write as their whole job — an unwired value
# output on those is not a mistake. Everything else that produces a value must be read.
_SINK_TYPES = frozenset({
    "Note", "MarkdownNote",
    "VHS_VideoCombine", "SaveImage", "SaveVideo", "SaveWEBM", "SaveAnimatedWEBP",
    "PreviewImage", "PreviewAny",
    "H3MultiFacePlan", "H3MultiFaceComposite", "H3PassSave", "H3PassLoad",
})


def _check_orphans(workflow):
    problems = []
    for node in workflow.get("nodes") or []:
        if not _is_active(node) or node.get("type") in _SINK_TYPES:
            continue
        outputs = node.get("outputs") or []
        if not outputs:
            continue
        if all(not (out.get("links") or []) for out in outputs):
            problems.append(Problem(
                "NO_ORPHANS",
                f"{node['type']} (id {node['id']}) produces output nothing reads, so it "
                f"never runs and its settings do nothing.",
                (node["id"],)))
    return problems


def _stitch_output_subject(stitch, nodes, links):
    """The subject whose pass a stitch belongs to: read it off the stitched Pass Save that
    consumes the stitch's frames."""
    for target, _ in _consumers(stitch, "images", nodes, links):
        widgets = target.get("widgets_values") or []
        if target.get("type") == "H3PassSave" and len(widgets) >= 2 and widgets[1] == "stitched":
            return _subject_number(target, nodes, links)
    return None


def _check_pass_chain_order(workflow):
    nodes = _nodes_by_id(workflow)
    links = _links_by_id(workflow)
    problems = []
    for node in workflow.get("nodes") or []:
        if not _is_active(node):
            continue

        if node.get("type") == "H3PassLoad":
            loaded = _subject_number(node, nodes, links)
            for stitch, in_name in _consumers(node, "images", nodes, links):
                if stitch.get("type") != "H3FaceStitch" or in_name != "base_images":
                    continue
                fed = _stitch_output_subject(stitch, nodes, links)
                if loaded is None or fed is None:
                    continue
                if fed == 0:
                    problems.append(Problem(
                        "PASS_CHAIN_ORDER",
                        f"H3PassLoad (subject {loaded}) feeds subject 0's stitch, but "
                        f"subject 0 composites onto the target clip — there is no previous "
                        f"pass to load.",
                        tuple(sorted([node["id"], stitch["id"]]))))
                elif loaded != fed - 1:
                    problems.append(Problem(
                        "PASS_CHAIN_ORDER",
                        f"H3PassLoad loads subject {loaded} but feeds subject {fed}'s stitch; "
                        f"a pass must load subject {fed - 1}, the one before it in plan order.",
                        (node["id"],)))

        elif node.get("type") == "H3PassSave":
            # chained_from is an optional widget after [subject, stage]; when it is set it
            # must chain onto the previous subject, and the first subject onto the clip.
            # When the widget is absent (the shipped default), it is left alone.
            widgets = node.get("widgets_values") or []
            if len(widgets) >= 3 and _socket_link(node, "chained_from") is None:
                subject = _subject_number(node, nodes, links)
                if subject is not None:
                    want = plan_store.CHAINED_TO_CLIP if subject == 0 else subject - 1
                    if widgets[2] != want:
                        problems.append(Problem(
                            "PASS_CHAIN_ORDER",
                            f"H3PassSave subject {subject} declares chained_from={widgets[2]!r}; "
                            f"expected {want!r}.",
                            (node["id"],)))
    return problems


def check_graph(workflow):
    """Every pack invariant this workflow breaks. Empty list means it is internally sound."""
    problems = []
    problems += _check_vae_kind(workflow)
    problems += _check_pass_save_stage(workflow)
    problems += _check_detector_consistency(workflow)
    problems += _check_scheduler_steps(workflow)
    problems += _check_orphans(workflow)
    problems += _check_pass_chain_order(workflow)
    return problems
