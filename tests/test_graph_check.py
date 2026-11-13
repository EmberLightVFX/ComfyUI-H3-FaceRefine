"""The repo-side graph checker's behaviour, at its one public seam: check_graph.

Every fault these tests encode is type-correct and schema-legal — ComfyUI's own
validate_workflow accepts them — so the only way to catch them is to assert this pack's
invariants directly against a graph. The correct fixture is the shipped
``H3_Face_Refine_Multi_Face.json`` (ground-truth wiring, extracted by probe, not
invented); each negative test mutates exactly one socket/widget of that known-good graph,
so an assertion can only disagree with reality if the checker is wrong.

Pure module, no ComfyUI, no torch, no weights.
"""

import copy
import json
import pathlib

import graph_check as gc

PACK_ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLE_DIR = PACK_ROOT / "example_workflows"


def _load(name):
    return json.loads((EXAMPLE_DIR / name).read_text(encoding="utf-8"))


# A single node's stored widget value, mutated by finding a widget value rather than a
# node id — the ids differ per template, the model names and widget meanings do not.
def _rename_widget(workflow, node_type, old_value, new_value):
    for node in workflow["nodes"]:
        if node.get("type") != node_type:
            continue
        widgets = node.get("widgets_values") or []
        for i, value in enumerate(widgets):
            if value == old_value:
                widgets[i] = new_value
                return node
    raise AssertionError(f"no {node_type} widget equals {old_value!r}")


# --- every shipped template is internally sound -----------------------------
# This is the reference: if a check fires on a template the pack itself ships, the check
# (not the template) is wrong — except where a mismatch is a real finding we chose to
# keep out of scope (plan.crop_factor inherits the tracker's, asserted below).

def test_shipped_templates_raise_no_problems():
    for path in sorted(EXAMPLE_DIR.glob("*.json")):
        problems = gc.check_graph(json.loads(path.read_text(encoding="utf-8")))
        assert problems == [], (
            path.name, [(p.code, p.node_ids, p.message) for p in problems])


# --- VAE_KIND: the right kind of VAE in the right socket (fault #1) ----------

def test_multi_face_wires_the_video_vae_into_the_video_sockets():
    # Green: the shipped graph decodes with the video VAE and conditions audio with the
    # audio VAE. This asserts the fixture itself, so a template that swaps a VAE would
    # break here before any mutation test means anything.
    codes = {p.code for p in gc.check_graph(_load("H3_Face_Refine_Multi_Face.json"))}
    assert "VAE_KIND" not in codes


def test_audio_vae_into_the_video_decode_socket_is_caught():
    # Fault #1 verbatim: the decode / inject / reference video sockets get a VAE whose
    # name says "audio" (the graph the user actually had: LTX23_audio_vae in VAEDecode).
    workflow = _load("H3_Face_Refine_Multi_Face.json")
    _rename_widget(workflow, "VAELoader",
                   "minimax_h3_video_vae_fp16.safetensors",
                   "LTX23_audio_vae_bf16.safetensors")
    problems = [p for p in gc.check_graph(workflow) if p.code == "VAE_KIND"]
    assert problems, "an audio VAE feeding the video-decode socket was not caught"
    # VAEDecode.vae is the socket named in the fault; it must be among the offenders.
    offending_nodes = {nid for p in problems for nid in p.node_ids}
    vadedecode_ids = {n["id"] for n in workflow["nodes"] if n["type"] == "VAEDecode"}
    assert offending_nodes & vadedecode_ids


def _set_pass_save_stage(node, value):
    node["widgets_values"][1] = value


def _pass_save_fed_by(workflow, source_type):
    """The H3PassSave nodes whose `images` socket is fed by a given source type."""
    nodes = {n["id"]: n for n in workflow["nodes"]}
    links = {l[0]: l for l in workflow["links"]}
    found = []
    for node in workflow["nodes"]:
        if node["type"] != "H3PassSave":
            continue
        for inp in node.get("inputs") or []:
            if inp["name"] == "images" and inp.get("link") is not None:
                if nodes[links[inp["link"]][1]]["type"] == source_type:
                    found.append(node)
    return found


# --- PASS_SAVE_STAGE: the stage matches what feeds the Pass Save (fault #2) ---

def test_crops_and_stitched_passes_are_recorded_at_the_right_stage():
    workflow = _load("H3_Face_Refine_Multi_Face.json")
    crops_saves = _pass_save_fed_by(workflow, "H3FaceTrackCrop")
    stitched_saves = _pass_save_fed_by(workflow, "H3FaceStitch")
    assert crops_saves and stitched_saves, "fixture changed shape"
    assert all(n["widgets_values"][1] == "crops" for n in crops_saves)
    assert all(n["widgets_values"][1] == "stitched" for n in stitched_saves)


def test_stitched_pass_saved_as_crops_is_caught():
    # Fault #2 verbatim: the Pass Save fed by H3 Face Stitch is labelled `crops`, so the
    # pass never completes and its frames overwrite the crops on disk.
    workflow = _load("H3_Face_Refine_Multi_Face.json")
    stitched_saves = _pass_save_fed_by(workflow, "H3FaceStitch")
    _set_pass_save_stage(stitched_saves[0], "crops")
    problems = [p for p in gc.check_graph(workflow) if p.code == "PASS_SAVE_STAGE"]
    assert problems, "a stitched feed recorded as `crops` was not caught"
    assert stitched_saves[0]["id"] in problems[0].node_ids


# --- DETECTOR / CONFIDENCE: the three detector widgets must agree (fault #4) --

def test_plan_and_trackcrop_agree_on_detector_and_confidence():
    workflow = _load("H3_Face_Refine_Multi_Face.json")
    codes = {p.code for p in gc.check_graph(workflow)}
    assert "DETECTOR_CONSISTENCY" not in codes
    assert "CONFIDENCE_CONSISTENCY" not in codes


def test_picker_confidence_below_the_plan_and_tracker_is_caught():
    # Fault #4 verbatim: H3FaceSelect runs at 0.2 while the plan and the tracker both run
    # at 0.35, so the picker offers a different face set than the plan decided on.
    workflow = _load("H3_Face_Refine_Multi_Face.json")
    _rename_widget(workflow, "H3FaceSelect", 0.35, 0.2)
    codes = {p.code for p in gc.check_graph(workflow)}
    assert "CONFIDENCE_CONSISTENCY" in codes


def test_a_different_detector_on_the_plan_is_caught():
    # The plan pairs the two plates positionally; a plan on a different detector from the
    # picker/tracker means two detectors disagree about which box belongs to whom.
    workflow = _load("H3_Face_Refine_Multi_Face.json")
    _rename_widget(workflow, "H3MultiFacePlan", "bbox\\face_yolov8m.pt", "bbox\\face_yolov8s.pt")
    codes = {p.code for p in gc.check_graph(workflow)}
    assert "DETECTOR_CONSISTENCY" in codes


# --- SCHEDULER_STEPS: the step count matches the loaded turbo LoRA (fault #5) -

def test_schedulers_match_the_8_step_lora_they_load():
    workflow = _load("H3_Face_Refine_Multi_Face.json")
    codes = {p.code for p in gc.check_graph(workflow)}
    assert "SCHEDULER_STEPS" not in codes


def test_twenty_steps_against_the_8_step_lora_is_caught():
    # Fault #5 verbatim: BasicScheduler.steps = 20 against the turbo_8step LoRA. The LoRA
    # and the step count have to agree or the sampler walks a schedule the weights were
    # not trained for.
    workflow = _load("H3_Face_Refine_Multi_Face.json")
    for node in workflow["nodes"]:
        if node["type"] == "BasicScheduler":
            node["widgets_values"][1] = 20
    problems = [p for p in gc.check_graph(workflow) if p.code == "SCHEDULER_STEPS"]
    assert problems, "20 steps against an 8-step LoRA was not caught"
    sched_ids = {n["id"] for n in workflow["nodes"] if n["type"] == "BasicScheduler"}
    assert set(problems[0].node_ids) & sched_ids


def test_a_lora_that_does_not_name_its_step_count_is_not_enforced():
    # Nothing in the name says how many steps, so the checker has no independent source to
    # hold the scheduler to — it must stay silent rather than invent a number.
    workflow = _load("H3_Face_Refine_Multi_Face.json")
    _rename_widget(workflow, "LoraLoaderModelOnly",
                   "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
                   "minimax_h3_fl2v_lora_comfyui_bf16.safetensors")
    codes = {p.code for p in gc.check_graph(workflow)}
    assert "SCHEDULER_STEPS" not in codes


# --- NO_ORPHANS: nothing that produces a value is left unread (fault #5b) ----

def _add_orphan(workflow, node_type, widgets, node_id=9999):
    workflow["nodes"].append({
        "id": node_id, "type": node_type, "mode": 0,
        "inputs": [], "outputs": [{"name": "SIGMAS", "type": "SIGMAS", "links": None}],
        "widgets_values": widgets,
    })
    return node_id


def test_shipped_multi_face_has_no_orphan_nodes():
    codes = {p.code for p in gc.check_graph(_load("H3_Face_Refine_Multi_Face.json"))}
    assert "NO_ORPHANS" not in codes


def test_a_scheduler_that_feeds_nothing_is_flagged():
    # Fault #5b verbatim: the graph carries extra BasicScheduler nodes wired to no sampler.
    # ComfyUI only runs nodes an output depends on, so an orphan never executes and its
    # settings quietly do nothing.
    workflow = _load("H3_Face_Refine_Multi_Face.json")
    orphan_id = _add_orphan(workflow, "BasicScheduler", ["simple", 8, 0.4], node_id=9999)
    problems = [p for p in gc.check_graph(workflow) if p.code == "NO_ORPHANS"]
    assert problems, "a scheduler wired to nothing was not flagged"
    assert orphan_id in problems[0].node_ids


# --- PASS_CHAIN_ORDER: each pass chains onto the pass before it ---------------

def test_pass_load_chains_onto_the_previous_subject():
    workflow = _load("H3_Face_Refine_Multi_Face.json")
    codes = {p.code for p in gc.check_graph(workflow)}
    assert "PASS_CHAIN_ORDER" not in codes


def _node(workflow, node_id):
    return next(n for n in workflow["nodes"] if n["id"] == node_id)


def _unlink(workflow, node, socket_name):
    """Drop a link off a node's socket so graph_check reads the node's own WIDGET instead."""
    for inp in node.get("inputs", []):
        if inp["name"] == socket_name:
            lid = inp.get("link")
            inp["link"] = None
            if lid is not None:
                workflow["links"] = [l for l in workflow["links"] if l[0] != lid]
            return


def test_pass_load_must_chain_onto_the_previous_subject_in_plan_order():
    # In the single-chain template the H3PassLoad subject comes from H3SubjectSelect's run-time
    # walk, so it is not knowable statically and the checker leaves it alone. The chaining rule is
    # pinned on a minimal graph where both subjects ARE static numbers: H3PassLoad feeds subject 1's
    # stitch (base_images), so it must load subject 0 - anything else is a pass that did not build
    # on the pass before it.
    def graph(load_value):
        return {
            "nodes": [
                {"id": 2, "type": "H3PassLoad", "mode": 0, "widgets_values": [load_value, "stitched"],
                 "inputs": [{"name": "plan"}, {"name": "images"}],
                 "outputs": [{"name": "images", "links": [12]}]},
                {"id": 3, "type": "H3FaceStitch", "mode": 0,
                 "inputs": [{"name": "base_images", "link": 12}],
                 "outputs": [{"name": "images", "links": [13]}]},
                {"id": 4, "type": "H3PassSave", "mode": 0, "widgets_values": [1, "stitched"],
                 "inputs": [{"name": "images", "link": 13}, {"name": "plan"}, {"name": "subject"}],
                 "outputs": [{"name": "images", "links": []}]},
            ],
            "links": [
                [12, 2, 0, 3, 0, "IMAGE"],
                [13, 3, 0, 4, 0, "IMAGE"],
            ],
        }

    problems = [p for p in gc.check_graph(graph(5)) if p.code == "PASS_CHAIN_ORDER"]
    assert problems, "a pass that did not chain onto the previous subject was not caught"
    assert 2 in problems[0].node_ids
    # subject 0 is the one subject 1 must build on, so a graph loading 0 is clean.
    assert not [p for p in gc.check_graph(graph(0)) if p.code == "PASS_CHAIN_ORDER"]


def test_a_fixed_subject_pass_save_with_a_wrong_chained_from_is_caught():
    # The template drives a Pass Save's subject + chained_from from H3SubjectSelect, so their widgets
    # are ignored. Unlink both sockets on #39 so the checker reads the widgets, pin subject 1, and
    # declare chained_from=5 - neither the target clip (-1) nor the previous subject (0): a plate this
    # pass did not build on.
    workflow = _load("H3_Face_Refine_Multi_Face.json")
    n39 = _node(workflow, 39)
    _unlink(workflow, n39, "subject")
    _unlink(workflow, n39, "chained_from")
    n39["widgets_values"] = [1, "stitched", 5]
    problems = [p for p in gc.check_graph(workflow) if p.code == "PASS_CHAIN_ORDER"]
    assert problems and 39 in problems[0].node_ids


def test_a_later_subject_chained_from_the_clip_is_caught():
    # chained_from = -1 (the target clip) is only legitimate for the FIRST subject. A later subject
    # building on the clip drops every earlier face from the finished clip - the composite hands back
    # the LAST pass's frames, which carry the earlier subjects only through the chain - so it is
    # refused, naming the subject.
    workflow = _load("H3_Face_Refine_Multi_Face.json")
    n39 = _node(workflow, 39)
    _unlink(workflow, n39, "subject")
    _unlink(workflow, n39, "chained_from")
    n39["widgets_values"] = [1, "stitched", -1]
    problems = [p for p in gc.check_graph(workflow) if p.code == "PASS_CHAIN_ORDER"]
    assert problems and 39 in problems[0].node_ids


def test_a_correct_explicit_chained_from_is_accepted():
    workflow = _load("H3_Face_Refine_Multi_Face.json")
    n39 = _node(workflow, 39)
    _unlink(workflow, n39, "subject")
    _unlink(workflow, n39, "chained_from")
    n39["widgets_values"] = [1, "stitched", 0]    # subject 1 onto the previous subject (0): the chain
    problems = [p for p in gc.check_graph(workflow) if p.code == "PASS_CHAIN_ORDER"]
    assert not problems






