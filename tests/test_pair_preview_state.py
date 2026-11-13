"""The data behind the read-only pairing preview panel (ticket #9).

The panel itself is a front end, and this pack has no JS test runner, so what is under
test here is the data layer behind it: `preview_state(job_name)`, the one function the
route calls, and the file it reads. That is honest coverage - the pixels in the panel come
from `pairing_preview.png`, which the plan node already draws, so the node's side of the
contract is pinned here too: the file lands in the job folder and the report names it.

Nothing here re-detects or re-pairs anything. The pairing was decided by the plan node, and
the panel reads that decision back off disk - which is the whole reason it can be read-only.

Needs torch (the plan node fixture), so CI with numpy only skips it. No GPU, no weights, and
the output root is redirected by the `pack` fixture whether or not a real ComfyUI
`folder_paths` is importable, so no test writes into a real ComfyUI folder.
"""

import os

import pytest

torch = pytest.importorskip("torch")

import plan_store as ps
# The plan-node fixture and its run helper, reused rather than rebuilt: this file drives a
# real plan node into a real tmp job folder and reads the folder back.
from test_plan_node_run import pack, _run  # noqa: F401  (`pack` is the fixture)


def _job(tmp_path, name):
    return tmp_path / "output" / name


# ----------------------------------------------------------------- the happy path

def test_the_pairing_preview_lands_in_the_job_folder(pack, monkeypatch, tmp_path):
    """The node already builds the preview tensor for its image output; #9 writes those same
    bytes out, so the panel has something to show without drawing anything itself."""
    _, _, report, _ = _run(pack, monkeypatch)
    job = _job(tmp_path, "group shot")

    assert (job / "pairing_preview.png").is_file()
    assert ps.PREVIEW_FILENAME == "pairing_preview.png"

    from PIL import Image
    with Image.open(job / "pairing_preview.png") as written:
        # Both plates side by side at the clip's height, as the node's preview output is.
        assert written.size[0] > 2000 and written.size[1] == 1000
    # The report says where to look, so a user can find the file without a front end.
    assert "pairing_preview.png" in report


def test_the_state_names_every_subject_and_the_reference_file_behind_it(pack, monkeypatch,
                                                                       tmp_path):
    """The row the panel draws per subject: number, clip face, paired still face, and the
    subject reference crop as it sits on disk."""
    _run(pack, monkeypatch)
    state = pack.pair_preview_api.preview_state("group shot")

    assert state["ready"] is True and state["ok"] is True
    assert state["status"] == "ready"
    assert state["job_name"] == "group shot"
    assert state["job_dir"] == str(_job(tmp_path, "group shot"))
    assert state["has_plan"] is True
    assert state["subject_count"] == 2

    subjects = state["subjects"]
    assert [s["index"] for s in subjects] == [0, 1]
    assert [s["clip_face"] for s in subjects] == [0, 2]
    assert [s["still_face"] for s in subjects] == [0, 1]
    assert [s["ref_kind"] for s in subjects] == [ps.REF_CROP, ps.REF_CROP]
    assert [s["ref_file"] for s in subjects] == ["subject_0.png", "subject_1.png"]
    assert [s["ref_exists"] for s in subjects] == [True, True]
    assert [s["has_reference"] for s in subjects] == [True, True]
    assert [s["pairing"] for s in subjects] == ["clip face 0 -> still face 0",
                                                "clip face 2 -> still face 1"]


def test_the_state_points_at_comfyui_s_own_view_route_for_the_images(pack, monkeypatch,
                                                                     tmp_path):
    """The panel is served by one JSON route and nothing else: the images come back as
    ComfyUI's own /view URLs into the output folder, so no static file serving is invented."""
    _run(pack, monkeypatch)
    state = pack.pair_preview_api.preview_state("group shot")

    url = state["preview_url"]
    assert url.startswith("/view?")
    assert "type=output" in url
    assert "group%20shot" in url and "pairing_preview.png" in url
    assert os.sep not in url and "\\" not in url

    refs = [s["ref_url"] for s in state["subjects"]]
    assert all(u.startswith("/view?") and "type=output" in u for u in refs)
    assert "subject_0.png" in refs[0]
    # The clip face beside each reference (#9): a /view URL into the clip crop the node wrote.
    clips = [s["clip_url"] for s in state["subjects"]]
    assert all(u and u.startswith("/view?") and "subfolder=" in u for u in clips)
    assert "clip_0.png" in clips[0] and "clip_1.png" in clips[1]
    assert state["preview_exists"] is True


def test_the_state_carries_the_pairing_inputs_the_panel_shows(pack, monkeypatch):
    """Match frame, source, tolerance, unmatched_faces, max_subjects: read back out of the
    plan, not re-read off the node's widgets, so the panel shows what the job actually was."""
    _run(pack, monkeypatch, excluded="1")
    state = pack.pair_preview_api.preview_state("group shot")

    inputs = state["inputs"]
    assert inputs["match_frame"] == 2
    assert inputs["subject_source"] == "union"
    assert inputs["match_tolerance"] == pytest.approx(0.05)
    assert inputs["unmatched_faces"] == "leave"
    assert inputs["max_subjects"] == 0
    assert state["excluded_faces"] == [1]
    assert state["match_frame"] == 2


# ---------------------------------------------------------- the not-ready states

def test_a_missing_job_folder_is_a_state_not_a_stack_trace(pack, monkeypatch, tmp_path):
    """The panel has to say "no plan yet for this job_name" - which is the normal state of a
    node nobody has queued yet - so the data layer returns it instead of raising."""
    assert not _job(tmp_path, "never run").exists()
    state = pack.pair_preview_api.preview_state("never run")

    assert state["ready"] is False and state["ok"] is False
    assert state["status"] == "no_folder"
    assert state["has_plan"] is False
    assert state["subjects"] == [] and state["subject_count"] == 0
    assert state["preview_exists"] is False and state["preview_url"] is None
    assert "H3 Multi-Face Plan" in state["message"]
    assert "never run" in state["message"]


def test_a_folder_with_no_plan_is_a_state_not_a_stack_trace(pack, monkeypatch, tmp_path):
    """A folder can exist for other reasons (a queued pass, a half-deleted job) without ever
    having held a subject plan."""
    job = _job(tmp_path, "no plan inside")
    os.makedirs(str(job), exist_ok=True)
    state = pack.pair_preview_api.preview_state("no plan inside")

    assert state["ready"] is False
    assert state["status"] == "no_plan"
    assert state["has_plan"] is False
    assert "H3 Multi-Face Plan" in state["message"]


def test_a_blank_job_name_says_what_to_type(pack, monkeypatch):
    """The plan node derives a job_name from the clip when its widget is blank, so a blank
    query is the panel asking for a folder that was never named."""
    state = pack.pair_preview_api.preview_state("")

    assert state["ready"] is False
    assert state["status"] == "no_job_name"
    assert "job_name" in state["message"]
    assert state["job_dir"] is None


def test_a_plan_this_pack_cannot_read_is_named_not_half_read(pack, monkeypatch, tmp_path):
    """`read_plan` refuses a plan format it does not understand; the panel has to pass that
    refusal through as a state rather than showing a confident half-plan."""
    import json

    job = _job(tmp_path, "from the future")
    os.makedirs(str(job), exist_ok=True)
    with open(str(job / ps.PLAN_FILENAME), "w", encoding="utf-8") as fh:
        json.dump({"version": 99}, fh)

    state = pack.pair_preview_api.preview_state("from the future")

    assert state["ready"] is False
    assert state["status"] == "bad_plan"
    assert state["has_plan"] is False
    assert "version" in state["message"]


# ----------------------------------------------- the faces that are not subjects

def test_a_left_out_face_shows_up_and_is_not_a_subject(pack, monkeypatch):
    """`unmatched_faces=leave` is the default, so an unpaired face keeps its original pixels.
    The panel has to show that face as left out, and never give it a subject number."""
    _run(pack, monkeypatch, mode="leave")
    state = pack.pair_preview_api.preview_state("group shot")

    assert [s["index"] for s in state["subjects"]] == [0, 1]
    assert [f["clip_face"] for f in state["left_out"]] == [1]
    assert [f["index"] for f in state["left_out"]] == [None]
    assert state["left_out"][0]["pairing"] == "clip face 1 -> no partner in the still"
    assert state["capped_out"] == []


def test_the_cap_shows_who_it_dropped(pack, monkeypatch):
    """#10's cap truncates loudly in the report; the panel is where the same faces have to be
    findable afterwards, named by the numbers the preview draws."""
    _run(pack, monkeypatch, mode="no_reference", max_subjects=1, job_name="capped job")
    state = pack.pair_preview_api.preview_state("capped job")

    assert state["subject_count"] == 1
    assert [s["index"] for s in state["subjects"]] == [0]
    assert [f["clip_face"] for f in state["capped_out"]] == [1, 2]
    assert [f["index"] for f in state["capped_out"]] == [None, None]
    assert state["inputs"]["max_subjects"] == 1
    assert state["inputs"]["unmatched_faces"] == "no_reference"


def test_a_no_reference_subject_says_so_and_shows_no_crop(pack, monkeypatch):
    """ref_kind none: a subject with nothing wired into H3's reference slots, so the row has
    to read "no subject reference" rather than show a broken image."""
    _run(pack, monkeypatch, mode="no_reference", job_name="no ref")
    state = pack.pair_preview_api.preview_state("no ref")

    subject = [s for s in state["subjects"] if s["clip_face"] == 1][0]
    assert subject["ref_kind"] == ps.REF_NONE
    assert subject["has_reference"] is False
    assert subject["ref_file"] is None
    assert subject["ref_exists"] is False
    assert subject["ref_url"] is None
    assert subject["still_face"] is None


def test_a_whole_still_subject_is_reported_as_the_whole_still(pack, monkeypatch):
    """ref_kind still: `ref_crop` is the full frame, so the panel must read the kind and not
    guess at whether a crop exists."""
    _run(pack, monkeypatch, mode="with_still", job_name="whole still")
    state = pack.pair_preview_api.preview_state("whole still")

    subject = [s for s in state["subjects"] if s["clip_face"] == 1][0]
    assert subject["ref_kind"] == ps.REF_STILL
    assert subject["ref_file"] == "subject_1.png"
    assert subject["ref_exists"] is True
    assert "whole" in subject["pairing"]


def test_each_face_row_carries_the_widget_token_that_switches_it_off(pack, monkeypatch):
    """Every face row carries the `excluded_faces` token that toggles it: the clip number for a
    clip face, `s<number>` for a face seen only in the reference still. That token is exactly what
    the panel's Ignore/Restore toggle writes to the node's widget, so the node - not the panel -
    still owns the plan, and a low-confidence face matched twice can be switched off once."""
    _run(pack, monkeypatch)
    state = pack.pair_preview_api.preview_state("group shot")

    for row in list(state["subjects"]) + list(state["left_out"]) + list(state["capped_out"]):
        if row["clip_face"] is not None:
            assert row["exclude_token"] == str(row["clip_face"])
        elif row["still_face"] is not None:
            assert row["exclude_token"] == "s" + str(row["still_face"])
        else:
            assert row["exclude_token"] is None

    subject_tokens = [s["exclude_token"] for s in state["subjects"]]
    assert subject_tokens and all(isinstance(t, str) and t for t in subject_tokens)
    # A clip-face subject is addressed by its bare clip number - exactly what Ignore writes.
    clip_subjects = [s for s in state["subjects"] if s["clip_face"] is not None]
    assert [s["exclude_token"] for s in clip_subjects] == [str(s["clip_face"]) for s in clip_subjects]


def test_the_state_offers_every_face_for_repairing_and_names_the_overrides_in_force(pack, monkeypatch):
    """The re-pairing grid needs, per plate, every found face with its CURRENT partner and a
    thumbnail only where the plan node actually cut one - and the panel must learn which manual
    pairings are already in force, so it can show the user what they already corrected."""
    _run(pack, monkeypatch, job_name="re-pair")
    state = pack.pair_preview_api.preview_state("re-pair")

    clip, still = state["clip_faces"], state["still_faces"]
    assert clip and still
    assert [c["face"] for c in clip] == list(range(len(clip)))
    assert [s["face"] for s in still] == list(range(len(still)))
    assert all(c["token"] == str(c["face"]) for c in clip)
    assert all(s["token"] == "s" + str(s["face"]) for s in still)
    # Nothing corrected yet, and a face `leave` left unpaired has no partner to show.
    assert state["pair_overrides"] == []
    assert [c for c in clip if c["face"] == 1][0]["partner_still"] is None

    # A manual pairing now in force: the override is named, and the face that gained a partner
    # shows it - which is exactly what the grid marks as the current choice.
    _run(pack, monkeypatch, job_name="re-paired", overrides="1=s0")
    st2 = pack.pair_preview_api.preview_state("re-paired")
    assert st2["pair_overrides"] == ["1=s0"]
    gained = [c for c in st2["clip_faces"] if c["face"] == 1][0]
    assert gained["partner_still"] == 0 and gained["partner_token"] == "s0"


def test_the_state_surfaces_the_last_runs_reuse_and_what_it_did_not_apply(pack, monkeypatch):
    """A fresh plan run is not a reuse; after the node reuses a saved plan the panel says so and
    lists the settings that reuse did NOT apply, so the user knows to press Process Plan."""
    _run(pack, monkeypatch, job_name="group shot")
    state = pack.pair_preview_api.preview_state("group shot")
    assert state["reused"] is False and state["unapplied"] == []

    ps.mark_resumed(state["job_dir"], ["excluded_faces: the node now has [3], the saved plan used []"])
    reused = pack.pair_preview_api.preview_state("group shot")
    assert reused["reused"] is True
    assert any("excluded_faces" in line for line in reused["unapplied"])


# ---------------------------------------------------------------- wiring, busy

def test_the_pack_init_wires_the_preview_module_next_to_the_picker():
    """The routes have to be registered for the panel to have anything to ask, and a front end
    has to exist to ask it - checked where they are written, since `__init__.py` needs ComfyUI
    to import. The picker stays wired, and `WEB_DIRECTORY` still points at `web`."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source = open(os.path.join(root, "__init__.py"), encoding="utf-8").read()
    assert "from . import pair_preview_api" in source
    assert "pair_preview_api.register()" in source
    assert "from . import picker_api" in source
    # The picker comes first, and both registrations share one kind of try/except, so the
    # nodes still load if the routes fail to register.
    assert source.index("from . import picker_api") < source.index("from . import pair_preview_api")
    assert 'WEB_DIRECTORY = "./web"' in source

    js = os.path.join(root, "web", "h3_pair_preview.js")
    assert os.path.isfile(js)
    text = open(js, encoding="utf-8").read()
    assert "H3MultiFacePlan" in text
    assert "/h3_facerefine/pair_preview" in text
    # A button holds no value: serialising it would shift every saved widget after it.
    assert "serialize = false" in text


def test_the_preview_script_stays_a_separate_front_end_from_the_picker():
    """Acceptance: the panel leaves the existing picker's routes and script untouched. That is
    a file-level fact, so it is pinned at the file level."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    text = open(os.path.join(root, "web", "h3_pair_preview.js"), encoding="utf-8").read()
    assert "h3_face_select" not in text
    assert "app.registerExtension" in text
    # Read-only: nothing in the panel writes to the graph.
    assert "setValue" not in text


def test_the_busy_gate_refuses_while_a_prompt_is_running(pack, monkeypatch):
    """The panel refuses to open or refresh mid-render, by the mechanism the picker already
    uses: ask the prompt queue, refuse if anything is running. With no server at all the gate
    says nothing is running, which is what lets the pack load outside ComfyUI."""
    api = pack.pair_preview_api

    assert api.render_running() is False
    assert api.BUSY_MESSAGE and "render" in api.BUSY_MESSAGE

    import sys
    import types

    server = types.ModuleType("server")

    class _Queue:
        def get_current_queue(self):
            return ([{"prompt": [1, 2, 3]}], [])

    class _Server:
        instance = types.SimpleNamespace(prompt_queue=_Queue())

    server.PromptServer = _Server
    monkeypatch.setitem(sys.modules, "server", server)
    assert api.render_running() is True


def test_register_survives_no_server_and_no_routes(pack):
    """`__init__.py` calls this on import. It must not raise when ComfyUI's server is absent,
    exactly as the picker's does."""
    pack.pair_preview_api.register()

