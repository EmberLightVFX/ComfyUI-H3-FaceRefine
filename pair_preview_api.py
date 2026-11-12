"""The read-only pairing preview: the state the panel shows, and its one route (#9).

Pairing is the failure mode this feature invents, and a render on a bad pairing costs N
generations, so the pairing has to be checkable BEFORE anything is queued. The plan node already
decides it and already draws both plates with every face numbered; this module reads that
decision back out of the job folder - `plan.json`, `pairing_preview.png`, and each subject's
`subject_N.png` - and hands it to the front end as plain data.

Read-only is the whole design, and it is what keeps this safe:

* nothing here detects, crops, pairs or re-decides anything. The panel cannot disagree with the
  graph, because it is the graph's own output;
* nothing here writes. There is no route that takes a pairing back in - exclusions stay the
  typed `excluded_faces` widget on the plan node, and the interactive pair editor is a later
  ticket;
* the images come back as ComfyUI's own ``/view`` URLs into the output folder. No new static
  file serving is invented here.

One route, ``/h3_facerefine/pair_preview``, and one data function behind it, `preview_state`,
which is importable and testable without a server - and which returns a state instead of raising
when there is no plan, because "no plan yet for this job_name" is the normal state of a node
nobody has queued and the panel has to say so rather than show a stack trace.

The one busy rule the panel keeps is the picker's: while a prompt is executing, it refuses. The
reason is not the cost of reading a JSON file - it is that a panel open over a half-written job
folder shows a pairing that the render in front of it may be about to invalidate.
"""

import os
import urllib.parse

try:  # loaded as part of the pack
    from . import plan_store
    from . import multi_face_nodes as _M
except ImportError:  # imported by name from the pack root, as the unit tests do
    import plan_store
    import multi_face_nodes as _M

# The route namespace the picker already uses; the panel gets its own path inside it, and the
# picker's routes and script are untouched.
PREVIEW_ROUTE = "/h3_facerefine/pair_preview"

# Same rule the face picker enforces, same reason: while the queue is running, refuse.
BUSY_MESSAGE = (
    "A render is running, so the pairing panel will not open or refresh. The job folder it "
    "reads is the one that render is writing into, so what it shows could be half-written. "
    "Try again once the queue is clear.")


def render_running() -> bool:
    """True while ComfyUI is executing a prompt.

    The picker asks the same question the same way (`picker_api._render_running`), so the two
    panels refuse on the same terms. With no server at all the answer is False - that is what
    lets the pack load outside ComfyUI, and what the tests pin.
    """
    try:
        from server import PromptServer
        running, _pending = PromptServer.instance.prompt_queue.get_current_queue()
        return bool(running)
    except Exception:
        return False


def _output_root() -> str:
    """ComfyUI's output folder, asked for lazily.

    Lazy on purpose: this module is imported when the pack loads, and the tests redirect the
    output root per test. Nothing here may resolve a real ComfyUI folder in a test run.
    """
    import folder_paths
    return str(folder_paths.get_output_directory())


def _view_url(root: str, path: str) -> str:
    """A ``/view`` URL for a file under ComfyUI's output folder.

    ComfyUI serves that folder itself; a job folder sits one level inside it, so the file name
    ``/view`` wants is the path relative to the output root, with forward slashes.
    """
    rel = os.path.relpath(str(path), str(root))
    rel = rel.replace(os.sep, "/").replace("\\", "/")
    return "/view?type=output&filename=" + urllib.parse.quote(rel)


def _face_number(value):
    """A face number, or None. `leave` and the cap both give a dropped face `index: None`, and
    that is exactly what the panel has to show: a face with no subject number."""
    return None if value is None else int(value)


def _pairing_text(face) -> str:
    """How one face reads in the panel: which clip face it is, and which face of the reference
    still its subject reference came from.

    Read `ref_kind` (`crop | still | none`, #8), never whether a crop happens to exist - a
    whole-still subject's `ref_crop` is the full frame.
    """
    clip_face, still_face = face.get("clip_face"), face.get("still_face")
    where = (f"clip face {int(clip_face)}" if clip_face is not None
             else "not seen in the clip on the match frame")
    if still_face is not None:
        partner = f"still face {int(still_face)}"
    elif face.get("ref_kind") == plan_store.REF_STILL:
        partner = "no partner in the still - conditioned on the whole reference still"
    else:
        partner = "no partner in the still"
    return f"{where} -> {partner}"



def _empty_state(job_name) -> dict:
    """The shape the panel always gets, ready or not.

    Every key is present in every state, so the front end renders one template and the panel can
    say "no plan yet" without a stack trace or a missing field.
    """
    return {
        "ok": False, "ready": False, "status": "", "message": "",
        "read_only": True, "busy": False,
        "job_name": str(job_name or ""), "job_dir": None, "has_plan": False,
        "plan_version": None,
        "subject_count": 0, "subjects": [],
        "excluded_faces": [], "left_out": [], "capped_out": [],
        "pairs": [], "unpaired_target": [], "unpaired_still": [],
        "match_frame": None, "scale": None, "tolerance_px": None,
        "inputs": {}, "counts": {},
        "run": {},
        "preview_file": plan_store.PREVIEW_FILENAME,
        "preview_exists": False, "preview_url": None,
    }


def _not_ready(state, status: str, message: str) -> dict:
    state["status"] = status
    state["message"] = message
    return state


def _subject_row(root, job, subject) -> dict:
    """One subject's row: its number, the two faces that make it, and the subject reference
    file that sits behind it - including whether that file is actually there."""
    ref_file = subject.get("ref_file")
    exists = bool(ref_file) and os.path.isfile(os.path.join(job, str(ref_file)))
    return {
        "index": _face_number(subject.get("index")),
        "clip_face": _face_number(subject.get("clip_face")),
        "still_face": _face_number(subject.get("still_face")),
        "ref_kind": subject.get("ref_kind") or plan_store.REF_NONE,
        "has_reference": bool(subject.get("has_reference")),
        "ref_file": ref_file,
        "ref_exists": exists,
        "ref_url": _view_url(root, os.path.join(job, str(ref_file))) if exists else None,
        "ref_crop": subject.get("ref_crop"),
        "pairing": _pairing_text(subject),
        # The pack's own wording for a subject, reused rather than reworded here.
        "label": _M.subject_label(subject),
    }


def _dropped_row(face) -> dict:
    """A face that is not a subject: left out by `unmatched_faces=leave` (#8) or dropped by
    `max_subjects` (#10). Never a subject number - those entries carry `index: None`, which is
    what keeps them out of every pass folder."""
    return {
        "index": _face_number(face.get("index")),
        "clip_face": _face_number(face.get("clip_face")),
        "still_face": _face_number(face.get("still_face")),
        "pairing": _pairing_text(face),
        "match_x": face.get("match_x"), "match_y": face.get("match_y"),
        "capped_by": face.get("capped_by"),
    }


def preview_state(job_name) -> dict:
    """Everything the pairing panel shows for one job, read off disk.

    Returns a state, never an exception, for every way a job can be absent: no name, no folder,
    a folder with no plan, a plan this pack cannot read. The plan, the preview image and the
    subject crops are the only sources - nothing is detected or paired again.
    """
    state = _empty_state(job_name)

    try:
        root = _output_root()
    except Exception as exc:
        return _not_ready(state, "no_output_root",
                          f"ComfyUI's output folder cannot be resolved: {exc}")

    try:
        job = plan_store.job_dir_for(root, job_name)
    except ValueError as exc:
        return _not_ready(
            state, "no_job_name",
            "There is no job folder to read: the panel names one with the plan node's job_name "
            f"widget, and that widget is blank. {exc}")

    state["job_dir"] = job
    if not os.path.isdir(job):
        return _not_ready(
            state, "no_folder",
            f"No plan yet for job_name \"{state['job_name']}\": there is no job folder at {job}. "
            f"Run H3 Multi-Face Plan with this job_name, then open this panel again.")

    if not os.path.isfile(plan_store.plan_path(job)):
        return _not_ready(
            state, "no_plan",
            f"No plan yet for job_name \"{state['job_name']}\": {job} holds no "
            f"{plan_store.PLAN_FILENAME}. Run H3 Multi-Face Plan with this job_name first - it "
            f"is the node that decides the pairing this panel shows.")

    try:
        plan = plan_store.read_plan(job)
    except Exception as exc:
        return _not_ready(state, "bad_plan", str(exc))

    inputs = dict(plan.get("inputs") or {})
    state["has_plan"] = True
    state["plan_version"] = plan.get("version")
    state["match_frame"] = plan.get("match_frame")
    state["scale"] = plan.get("scale")
    state["tolerance_px"] = plan.get("tolerance_px")
    state["pairs"] = plan.get("pairs") or []
    state["unpaired_target"] = plan.get("unpaired_target") or []
    state["unpaired_still"] = plan.get("unpaired_still") or []
    state["excluded_faces"] = [int(i) for i in (plan.get("excluded_faces") or [])]
    state["subject_count"] = int(plan.get("subject_count") or len(plan.get("subjects") or []))
    state["subjects"] = [_subject_row(root, job, s) for s in (plan.get("subjects") or [])]
    # The faces the plan took out of the subject set, named rather than counted: `leave` (#8)
    # and the cap (#10) both decide who does NOT get refined, and both have to stay findable
    # afterwards. Never index these entries - they carry `index: None`.
    state["left_out"] = [_dropped_row(s) for s in (plan.get("left_out") or [])]
    state["capped_out"] = [_dropped_row(s) for s in (plan.get("capped_out") or [])]
    state["counts"] = plan_store.subject_kind_counts(plan)
    state["run"] = dict(plan.get("run") or {})
    # The pairing inputs, read out of the plan rather than off the widgets: this is what the job
    # actually was, which is the only thing worth checking a pairing against.
    state["inputs"] = {
        "match_frame": inputs.get("match_frame"),
        "subject_source": inputs.get("subject_source"),
        "match_tolerance": inputs.get("match_tolerance"),
        "excluded_faces": inputs.get("excluded_faces") or [],
        "unmatched_faces": inputs.get("unmatched_faces") or plan_store.DEFAULT_UNMATCHED_FACES,
        "max_subjects": int(inputs.get("max_subjects") or 0),
        "target_size": inputs.get("target_size"),
        "still_size": inputs.get("still_size"),
    }

    preview = os.path.join(job, plan_store.PREVIEW_FILENAME)
    state["preview_exists"] = os.path.isfile(preview)
    state["preview_url"] = _view_url(root, preview) if state["preview_exists"] else None

    state["status"] = "ready"
    state["ok"] = state["ready"] = True
    state["message"] = (f"plan.json read from {job}" if state["preview_exists"] else
                        f"plan.json read from {job}, but {plan_store.PREVIEW_FILENAME} is "
                        f"missing - re-run H3 Multi-Face Plan to write it")
    return state


def register():
    """Register the panel's one route. Safe to call when ComfyUI's server is absent."""
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:
        return

    routes = getattr(getattr(PromptServer, "instance", None), "routes", None)
    if routes is None:
        return

    @routes.get(PREVIEW_ROUTE)
    async def _state_route(request):
        """The pairing state for one job_name, as JSON.

        Read-only by construction: the only input is a job name, and the only thing that comes
        back is what the plan node already wrote. A refusal because a render is running is a
        normal answer, not an HTTP error, so the panel can show it as a message.
        """
        name = request.query.get("job_name", "")
        if render_running():
            message = BUSY_MESSAGE
            return web.json_response({"ok": False, "ready": False, "status": "busy",
                                      "busy": True, "error": message, "message": message,
                                      "job_name": name, "read_only": True}, status=200)
        try:
            return web.json_response(preview_state(name))
        except Exception as exc:            # a state, never a stack trace in the browser
            message = str(exc)
            return web.json_response({"ok": False, "ready": False, "status": "error",
                                      "error": message, "message": message,
                                      "job_name": name, "read_only": True}, status=200)
