"""The read-only pairing preview: the state the panel shows, and its one route (#9).

Pairing is the failure mode this feature invents, and a render on a bad pairing costs N
generations, so the pairing has to be checkable BEFORE anything is queued. The plan node already
decides it and already draws both plates with every face numbered; this module reads that
decision back out of the job folder - `plan.json`, `pairing_preview.png`, and each subject's
`subject_N.png` - and hands it to the front end as plain data.

Read-only is the whole design, and it is what keeps this safe:

* nothing here detects, crops, pairs or re-decides anything. The panel cannot disagree with the
  graph, because it is the graph's own output;
* the panel never authors or re-decides a pairing. It can do three things to a job, and none of
  them makes it a source of truth: a per-face toggle edits only the plan node's own typed
  ``excluded_faces`` widget (this module only reports each face's widget token, `exclude_token`);
  a "replace" action (Process Plan) WIPES one job ENTIRELY - its plan, preview, crops, manifest,
  subject cursor and every saved pass - so the plan node's OWN re-run starts the job over from
  scratch (DESTRUCTIVE: it deletes completed renders; the front end confirms it, and one
  ``plan_prev.json`` safety copy is kept); and a "reset" action (Reset run) clears only one job's
  pass BOOKKEEPING (the manifest's pass records and the subject walk) so the walk starts over
  without touching any frame. None of them writes a plan, a crop, a pairing or a pass stage's
  frames - the plan node decides and writes the pairing, and the pass nodes write their own
  stages. See ``clear_job_for_replace`` and ``reset_run``. The interactive re-pairing editor is
  still a later ticket;
* the images come back as ComfyUI's own ``/view`` URLs into the output folder. No new static
  file serving is invented here.

Three routes share one namespace. ``/h3_facerefine/pair_preview`` (GET) is read-only and backed by
one data function, `preview_state`, importable and testable without a server, which returns a
state instead of raising when there is no plan, because "no plan yet for this job_name" is the
normal state of a node nobody has queued. ``/h3_facerefine/pair_preview/replace`` (POST - Process Plan) WIPES one job entirely - its plan
(kept once as ``plan_prev.json``), preview and crops, ``manifest.json``, the subject cursor and the
whole ``passes/`` tree - so the plan node re-runs into the same name as a true fresh start; it is
DESTRUCTIVE (it deletes completed renders) and the front end confirms it. ``/h3_facerefine/pair_preview/reset``
(POST) clears one job's pass BOOKKEEPING (the manifest's pass records and the subject-cursor file)
so the always-next walk starts over; it is non-destructive to plan.json, the references, the preview
and the saved pass frames. Both writes refuse while a render runs and neither authors a pairing.

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

# Process Plan's write action: wipe a job ENTIRELY (plan artifacts + manifest + subject cursor + the
# whole passes/ tree, keeping one plan_prev.json) so the plan node re-runs into the same job_name as a
# true fresh start. DESTRUCTIVE - deletes completed renders. Separate path so the read route is read-only.
REPLACE_ROUTE = PREVIEW_ROUTE + "/replace"

# The panel's second (and only other) write action: reset a job's run - clear the pass bookkeeping
# (the manifest's pass records and the subject cursor) so the walk starts over at the first subject.
# A separate path so the read route stays read-only; like replace it never touches plan.json.
RESET_ROUTE = PREVIEW_ROUTE + "/reset"

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

    ComfyUI serves that folder itself; a job folder sits one level inside it. ``/view`` wants the
    folder and the bare file name as SEPARATE query params - a subpath crammed inside ``filename``
    404s - so split at the last forward slash and quote each half. That also lets a job-name folder
    which itself contains a space resolve, since the space survives as ``%20`` in ``subfolder``.
    """
    rel = os.path.relpath(str(path), str(root))
    rel = rel.replace(os.sep, "/").replace("\\", "/")
    head, _, base = rel.rpartition("/")
    return ("/view?type=output&subfolder=" + urllib.parse.quote(head)
            + "&filename=" + urllib.parse.quote(base))


def _face_number(value):
    """A face number, or None. `leave` and the cap both give a face that is not a subject
    an `index: None`, and
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
        "clip_faces": [], "still_faces": [], "pair_overrides": [],
        # Re-running the plan node reuses the saved plan; the panel surfaces the last run's reuse
        # record and any settings it did NOT apply (defaults: not reused, nothing pending).
        "reused": False, "unapplied": [],
        "preview_file": plan_store.PREVIEW_FILENAME,
        "preview_exists": False, "preview_url": None,
    }


def _not_ready(state, status: str, message: str) -> dict:
    state["status"] = status
    state["message"] = message
    return state


def _exclude_token(face) -> "str | None":
    """The ``excluded_faces`` token that switches this face off, spelled the way the widget spells
    it: a bare clip number for a target-clip face, ``s<n>`` for a still-only face, ``None`` for a
    row with no face at all. The panel's include/ignore toggle writes exactly this token to the
    node's widget - the sanctioned control - so the node, never the panel, owns the decision."""
    clip_face = face.get("clip_face")
    if clip_face is not None:
        return str(int(clip_face))
    still_face = face.get("still_face")
    if still_face is not None:
        return "s" + str(int(still_face))
    return None


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
        "clip_file": subject.get("clip_file"),
        "clip_exists": bool(subject.get("clip_file")) and
                       os.path.isfile(os.path.join(job, str(subject.get("clip_file")))),
        "clip_url": (_view_url(root, os.path.join(job, str(subject["clip_file"])))
                     if (subject.get("clip_file") and
                         os.path.isfile(os.path.join(job, str(subject["clip_file"]))))
                     else None),
        "ref_crop": subject.get("ref_crop"),
        "pairing": _pairing_text(subject),
        "exclude_token": _exclude_token(subject),
        # The pack's own wording for a subject, reused rather than reworded here.
        "label": _M.subject_label(subject),
    }


def _not_subject_row(face) -> dict:
    """A face that is not a subject: left out by `unmatched_faces=leave` (#8) or capped by
    `max_subjects` (#10). Never a subject number - those entries carry `index: None`, which is
    what keeps them out of every pass folder."""
    return {
        "index": _face_number(face.get("index")),
        "clip_face": _face_number(face.get("clip_face")),
        "still_face": _face_number(face.get("still_face")),
        "pairing": _pairing_text(face),
        "match_x": face.get("match_x"), "match_y": face.get("match_y"),
        "capped_by": face.get("capped_by"),
        "exclude_token": _exclude_token(face),
    }


def _face_candidate_lists(plan, root, job):
    """Every face on each plate the re-pairing grid can pick from: number, current partner, and a
    thumbnail where one already exists (a subject's clip crop, or a subject's crop reference).

    This is hybrid by construction: faces that are already subjects come with the image the plan
    node cut for them, and a face that never became a subject has no crop on disk, so it renders as
    a numbered cell and the big preview PNG's numbers carry the rest. The partner shown is the
    CURRENT one - the pairing after any manual override already in the plan - because that is what
    the user sees being corrected.
    """
    n_clip = len(plan.get("target_boxes") or [])
    n_still = len(plan.get("still_boxes") or [])
    pairs = {(int(p["target"])): int(p["still"]) for p in (plan.get("pairs") or [])
             if p.get("target") is not None and p.get("still") is not None}
    clip_of_still = {s: c for c, s in pairs.items()}

    def thumb(file_name):
        if not file_name:
            return None
        full = os.path.join(job, str(file_name))
        return _view_url(root, full) if os.path.isfile(full) else None

    clip_thumb, still_thumb = {}, {}
    for subject in (plan.get("subjects") or []):
        cf, sf = subject.get("clip_face"), subject.get("still_face")
        if cf is not None and subject.get("clip_file"):
            clip_thumb[int(cf)] = thumb(subject["clip_file"])
        if (sf is not None and subject.get("ref_kind") == plan_store.REF_CROP
                and subject.get("ref_file")):
            still_thumb[int(sf)] = thumb(subject["ref_file"])

    clip_faces = []
    for i in range(n_clip):
        st = pairs.get(i)
        clip_faces.append({
            "face": i, "token": str(i),
            "partner_still": st, "partner_token": None if st is None else "s" + str(st),
            "thumb_url": clip_thumb.get(i),
        })
    still_faces = []
    for j in range(n_still):
        ct = clip_of_still.get(j)
        still_faces.append({
            "face": j, "token": "s" + str(j),
            "partner_clip": ct, "partner_token": None if ct is None else str(ct),
            "thumb_url": still_thumb.get(j),
        })
    return clip_faces, still_faces


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
    # Kept as the plan wrote them: a bare number is a target clip face, an `s` prefix a reference
    # still face. Coercing to int here would throw away which plate was named.
    state["excluded_faces"] = [str(i) if isinstance(i, str) else int(i)
                               for i in (plan.get("excluded_faces") or [])]
    state["subject_count"] = int(plan.get("subject_count") or len(plan.get("subjects") or []))
    state["subjects"] = [_subject_row(root, job, s) for s in (plan.get("subjects") or [])]
    # The faces the plan took out of the subject set, named rather than counted: `leave` (#8)
    # and the cap (#10) both decide who does NOT get refined, and both have to stay findable
    # afterwards. Never index these entries - they carry `index: None`.
    state["left_out"] = [_not_subject_row(s) for s in (plan.get("left_out") or [])]
    state["capped_out"] = [_not_subject_row(s) for s in (plan.get("capped_out") or [])]
    state["counts"] = plan_store.subject_kind_counts(plan)
    # What the re-pairing grid can offer: every face on each plate, its current partner, and a
    # thumbnail where the plan node already cut one. Plus the manual pairings in force, so the
    # panel can show which faces the user has already re-pointed.
    state["clip_faces"], state["still_faces"] = _face_candidate_lists(plan, root, job)
    state["pair_overrides"] = list(inputs.get("pair_overrides") or [])
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

    # The last plan-node run's reuse record (#re-running the plan node reuses the saved plan).
    # If the node reused this job's plan instead of re-planning, name it and list any settings the
    # node now sets differently that the run did NOT apply - so the panel can tell the user to
    # press Process Plan to apply them. Empty when the last run actually wrote a fresh plan.
    resume = plan_store.read_resume_state(job)
    state["reused"] = bool(resume.get("reused"))
    state["unapplied"] = [str(line) for line in (resume.get("changed") or [])]

    # The two run-state facts the panel should show: how far through the always-next walk the job
    # is (so the user knows how many times to queue), and whether every saved pass still spans the
    # clip's frame count (so a stale/short pass is visible before the chain throws an IndexError).
    state["queue"] = plan_store.run_counts(plan, job)
    state["frames"] = plan_store.frame_consistency(plan, job)
    state["frames_note"] = plan_store.frame_consistency_note(plan, job)

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

    @routes.post(REPLACE_ROUTE)
    async def _replace_route(request):
        """Clear one job ENTIRELY so a plan-node re-run starts it over from scratch.

        **Process Plan**'s write action. It deletes everything the pipeline wrote for this job - the
        plan (kept once as plan_prev.json), the preview, the subject/clip crops, `manifest.json`, the
        subject-cursor file, and the whole `passes/` tree - so re-pairing is a true fresh start and
        cannot leave stale or shorter passes to break the chain. This is DESTRUCTIVE (it removes
        completed renders); the front end confirms it first, and it keeps a single plan_prev.json
        safety copy. It still authors no pairing (the plan node re-decides on re-run). Refuses while
        a render is running, like the read route.
        """
        try:
            body = await request.json()
        except Exception:
            body = {}
        name = (body or {}).get("job_name", "")
        if render_running():
            message = BUSY_MESSAGE
            return web.json_response({"ok": False, "status": "busy", "busy": True,
                                      "error": message, "message": message, "job_name": name,
                                      "replaced": False}, status=200)
        try:
            job = plan_store.job_dir_for(_output_root(), name)
            res = plan_store.clear_job_for_replace(job)
        except Exception as exc:            # empty/illegal job_name, or a folder we cannot reach
            message = str(exc)
            return web.json_response({"ok": False, "status": "error", "error": message,
                                      "message": message, "job_name": name,
                                      "replaced": False}, status=200)
        return web.json_response({"ok": True, "status": "cleared", "job_name": name,
                                  "replaced": res["existed"], "cleared": res["cleared"],
                                  "prev": res["prev"],
                                  "passes_removed": res["passes_removed"],
                                  "manifest_removed": res["manifest_removed"],
                                  "cursor_removed": res["cursor_removed"],
                                  "job": job}, status=200)

    @routes.post(RESET_ROUTE)
    async def _reset_route(request):
        """Start one job's subject passes over: clear the pass bookkeeping so the walk restarts.

        The panel's other write action, and non-destructive: it drops only the manifest's pass
        records and the subject-cursor file (plan_store.reset_run), so every subject reads as
        not-yet-processed and the next queue takes the first one again. It never touches plan.json
        (the pairing), the references, the preview or the saved pass frames - re-running a subject
        overwrites its own frames. Refuses while a render is running, like the other routes.
        """
        try:
            body = await request.json()
        except Exception:
            body = {}
        name = (body or {}).get("job_name", "")
        if render_running():
            message = BUSY_MESSAGE
            return web.json_response({"ok": False, "status": "busy", "busy": True,
                                      "error": message, "message": message, "job_name": name,
                                      "reset": False}, status=200)
        try:
            job = plan_store.job_dir_for(_output_root(), name)
            res = plan_store.reset_run(job)
        except Exception as exc:            # empty/illegal job_name, or a folder we cannot reach
            message = str(exc)
            return web.json_response({"ok": False, "status": "error", "error": message,
                                      "message": message, "job_name": name,
                                      "reset": False}, status=200)
        return web.json_response({"ok": True, "status": "reset", "job_name": name,
                                  "reset": True, "cleared": res["cleared"],
                                  "existed": res["existed"], "job": job}, status=200)
