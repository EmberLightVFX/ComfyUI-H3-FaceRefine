import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// A read-only pairing panel on H3 Multi-Face Plan (#9).
//
// Pairing is the failure mode this feature invents, and a render on a bad pairing costs N
// generations, so the pairing has to be checkable before anything is queued. The plan node
// already decides it and already draws both plates with every face outlined and numbered; this
// panel reads that decision back out of the job folder over one JSON route and shows it.
//
// The panel is read off the job folder and never authors a pairing. The ONE thing it can change on
// the node is its own `excluded_faces` widget, through a per-face include/ignore toggle - the
// single sanctioned control for switching a face off. "Process Plan" then re-runs the plan node
// into job_name and REPLACES the plan: to let that overwrite past the pairing-fingerprint guard it
// first clears the plan node's OWN stale artifacts (plan.json -> plan_prev.json, plus the preview
// and the subject/clip crops) over one narrow, render-guarded route. The panel itself never writes
// a plan, crop, pairing or pass stage, and pass stages (manifest.json, passes/) are left for the
// pass/composite nodes' own guards. The panel can also re-point one face's partner to another face
// (manual pairing) by writing the node's `pair_overrides` widget - the node still re-decides the
// whole pairing when Process Plan runs, so the panel is never a second source of truth. The
// panel is its own front end, separate from the face picker's, so the picker stays untouched.
//
// It does keep the picker's one rule: while a prompt is running it refuses. The reason here is
// not GPU contention but that the job folder it reads is the one that render is writing into, so
// the panel could show a half-written pairing. Unlike the picker there is no "anyway" override:
// a half-read folder is not a judgement call about a card, and the wait is seconds, not a scan.

const PLAN = "H3MultiFacePlan";
const ROUTE = "/h3_facerefine/pair_preview";

function styleOnce() {
    if (document.getElementById("h3pp-style")) return;
    const s = document.createElement("style");
    s.id = "h3pp-style";
    s.textContent = [
        ".h3pp-back{position:fixed;inset:0;background:rgba(0,0,0,.72);z-index:10000;",
        "  display:flex;align-items:center;justify-content:center;font-family:sans-serif}",
        ".h3pp-box{background:#222;color:#ddd;border:1px solid #444;border-radius:8px;",
        "  max-width:min(1200px,94vw);max-height:92vh;display:flex;flex-direction:column}",
        ".h3pp-head{padding:12px 16px;border-bottom:1px solid #444;font-size:14px}",
        ".h3pp-body{padding:12px 16px;overflow:auto;flex:1}",
        ".h3pp-foot{padding:12px 16px;border-top:1px solid #444;display:flex;gap:8px;",
        "  justify-content:flex-end;align-items:center}",
        ".h3pp-msg{flex:1;font-size:12px;color:#999}",
        ".h3pp-note{font:12px sans-serif;color:#aaa;margin:0 0 10px;white-space:pre-wrap}",
        ".h3pp-inputs{font:12px sans-serif;color:#bbb;background:#1e1e1e;border:1px solid #3a3a3a;",
        "  border-radius:6px;padding:8px 10px;margin-bottom:12px}",
        ".h3pp-inputs b{color:#eee}",
        ".h3pp-sect{margin:0 0 6px;font-size:13px;font-weight:600;color:#bbb}",
        ".h3pp-imgwrap{margin-bottom:16px;line-height:0}",
        ".h3pp-imgwrap img{max-width:100%;border-radius:4px;display:block;background:#111}",
        ".h3pp-row{display:flex;gap:12px;align-items:flex-start;border:1px solid #3a3a3a;",
        "  border-radius:6px;padding:8px 10px;margin-bottom:8px}",
        ".h3pp-cell{flex:1;font:12px sans-serif;color:#ccc}",
        ".h3pp-cell b{color:#eee}",
        ".h3pp-cell small{display:block;color:#8f8f8f;margin-top:4px}",
        ".h3pp-crops{display:flex;gap:10px;align-items:flex-start}",
        ".h3pp-thumb{display:flex;flex-direction:column;gap:3px;align-items:center}",
        ".h3pp-cap{font:11px sans-serif;color:#8f8f8f;text-align:center;max-width:132px}",
        ".h3pp-crop{width:132px;max-height:132px;object-fit:contain;background:#111;",
        "  border:1px solid #555;border-radius:4px}",
        ".h3pp-nocrop{width:132px;height:80px;border:1px dashed #555;border-radius:4px;color:#888;",
        "  font:11px sans-serif;display:flex;align-items:center;justify-content:center;",
        "  text-align:center;padding:6px;box-sizing:border-box;background:#1e1e1e}",
        ".h3pp-chips{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px}",
        ".h3pp-chip{font:12px sans-serif;color:#ccc;background:#2b2b2b;border:1px solid #4a4a4a;",
        "  border-radius:4px;padding:3px 8px}",
        ".h3pp-chip.warn{color:#f0c674;border-color:#6a5a2a}",
        ".h3pp-chip.off{color:#e08a8a;border-color:#6a3a3a;background:rgba(224,138,138,.16)}",
        ".h3pp-toggle{font:12px sans-serif;color:#eee;background:#3a3a3a;border:1px solid #5a5a5a;",
        "  border-radius:4px;padding:3px 8px;cursor:pointer}",
        ".h3pp-chip + .h3pp-toggle{margin-left:4px}",
        ".h3pp-cell .h3pp-toggle{margin-top:5px;display:block}",
        ".h3pp-row.off{background:rgba(224,138,138,.14);border-color:#7a4040}",
        ".h3pp-row.off b{color:#e6b3b3}",
        ".h3pp-facegrid{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 14px}",
        ".h3pp-face{display:flex;flex-direction:column;gap:3px;align-items:center;border:1px solid #555;",
        "  border-radius:4px;padding:5px;cursor:pointer;background:#1e1e1e;min-width:96px}",
        ".h3pp-face:hover{border-color:#888}",
        ".h3pp-face img{width:88px;max-height:88px;object-fit:contain;background:#111;border-radius:3px}",
        ".h3pp-face span{font:11px sans-serif;color:#bbb}",
        ".h3pp-face.cur{border-color:#e0c060;background:#33301f}",
        ".h3pp-face.cur span{color:#f0c674}",
        ".h3pp-face.noneopt{color:#ccc;min-width:96px;align-items:center;justify-content:center;",
        "  font:12px sans-serif}",
        ".h3pp-thumb.clickable{cursor:pointer}",
        ".h3pp-thumb.clickable:hover .h3pp-crop{border-color:#e0c060}",
        ".h3pp-thumb.clickable:hover .h3pp-nocrop{border-color:#e0c060}",
        ".h3pp-hint{font:11px sans-serif;color:#7aa0c8;margin-top:2px;text-align:center}",
        ".h3pp-reused{color:#8fb4d8}",
        ".h3pp-warn{color:#e0c060;font-weight:600}",
        ".h3pp-sub{font:12px sans-serif;color:#e0c060;margin:2px 0 0 14px}",
    ].join("\n");
    document.head.appendChild(s);
}

function dialog(title) {
    styleOnce();
    const back = document.createElement("div");
    back.className = "h3pp-back";
    const box = document.createElement("div");
    box.className = "h3pp-box";
    const head = document.createElement("div");
    head.className = "h3pp-head";
    head.textContent = title;
    const body = document.createElement("div");
    body.className = "h3pp-body";
    const foot = document.createElement("div");
    foot.className = "h3pp-foot";
    const msg = document.createElement("span");
    msg.className = "h3pp-msg";
    foot.appendChild(msg);
    box.appendChild(head);
    box.appendChild(body);
    box.appendChild(foot);
    back.appendChild(box);
    document.body.appendChild(back);
    const close = () => back.remove();
    back.addEventListener("click", (e) => { if (e.target === back) close(); });
    return { body, foot, msg, head, close };
}

function button(label, cls, fn) {
    const b = document.createElement("button");
    b.textContent = label;
    if (cls) b.className = cls;
    b.onclick = fn;
    return b;
}

function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = String(text);
    return n;
}

const w = (node, name) => (node.widgets || []).find((x) => x.name === name);

/* ------------------------------------------------------------ render guard */

// The client-side half of the guard; the route holds the authority half. `app.queueSize` alone
// is not enough - a prompt can already be running before the front end has counted it - so the
// execution events carry the state too.
let executing = false;
api.addEventListener("execution_start", () => { executing = true; });
api.addEventListener("execution_success", () => { executing = false; });
api.addEventListener("execution_error", () => { executing = false; });
api.addEventListener("executing", ({ detail }) => { if (detail === null) executing = false; });

function renderRunning() {
    if (executing) return true;
    try { return Number(app.queueSize) > 0; } catch (e) { return false; }
}

function showWait(d, node) {
    d.head.textContent = "Waiting on the render";
    d.body.textContent =
        "A render is running, so the pairing panel will not read the job folder: that render is " +
        "writing into it, so anything shown would be half-written. It reads as soon as the queue " +
        "is clear.";
    d.msg.textContent = "";
    d.foot.appendChild(button("Try again", "", () => { d.close(); openPanel(node); }));
    d.foot.appendChild(button("Close", "", d.close));
}

/* ------------------------------------------------------------------ render */

function thumb(url, cap, missing, onPick, hint) {
    const col = el("div", "h3pp-thumb");
    if (onPick) col.classList.add("clickable");
    if (url) {
        const img = el("img", "h3pp-crop");
        img.src = api.apiURL(url);
        img.alt = cap;
        col.appendChild(img);
    } else {
        col.appendChild(el("div", "h3pp-nocrop", missing));
    }
    col.appendChild(el("small", "h3pp-cap", cap));
    if (onPick) {
        if (hint) col.appendChild(el("small", "h3pp-hint", hint));
        col.onclick = onPick;
    }
    return col;
}

function cropCell(s, onPickClip, onPickRef) {
    // Both faces of the subject side by side: the clip face on the match frame, and the
    // reference it is conditioned on. A still-only subject has no clip face, so it shows the
    // placeholder instead of an image. Either image is pressable where that side can be
    // re-pointed: click the clip face to pick a new clip face for the subject, click the
    // reference to pick a new reference still face. A side with nothing to re-point is not
    // clickable.
    const wrap = el("div", "h3pp-crops");
    wrap.appendChild(thumb(s.clip_url, "clip face", "no clip face\non the match frame",
        onPickClip, "click to pick clip face"));
    const why = !s.has_reference
        ? "no subject reference\n(unmatched_faces=no_reference)"
        : "reference file missing\nre-run H3 Multi-Face Plan";
    wrap.appendChild(thumb(s.ref_url, s.ref_file || "reference", why,
        onPickRef, "click to pick reference face"));
    return wrap;
}

function kindNote(s) {
    if (s.ref_kind === "still") {
        return "whole reference still as its subject reference: conditioning is whole-frame, so "
            + "H3 picks which face it applies";
    }
    if (s.ref_kind === "none") {
        return "no subject reference: this pass runs with nothing wired into H3's reference "
            + "slots, so it is weaker than a paired subject";
    }
    const crop = s.ref_crop ? " (" + s.ref_crop[2] + "x" + s.ref_crop[3] + "px)" : "";
    return "subject reference cut from the reference still" + crop;
}

function chips(list, cls, text) {
    const wrap = el("div", "h3pp-chips");
    for (const item of list) wrap.appendChild(el("span", "h3pp-chip " + cls, text(item)));
    return wrap;
}

// The include/ignore toggle. It reads and writes ONLY the node's `excluded_faces` widget - the
// one sanctioned control - so the panel never re-decides the pairing itself; the node re-decides
// when the user presses Process Plan (which serialises this widget). Tokens: a bare number is a
// clip face, an `s`-prefixed one a reference still face, exactly as the widget and preview number them.
function tokenLabel(t) {
    return (typeof t === "string" && t[0] === "s")
        ? "reference still face " + t.slice(1)
        : "clip face " + t;
}

function excludedTokens(node) {
    const wd = w(node, "excluded_faces");
    if (!wd) return null;
    const raw = (wd.value === null || wd.value === undefined) ? "" : String(wd.value);
    return { wd, tokens: raw.split(",").map((t) => t.trim()).filter((t) => t.length) };
}

function writeExcluded(node, tokens) {
    const info = excludedTokens(node);
    if (!info) return "";
    const seen = new Set();
    const out = [];
    for (const t of tokens) {
        const s = String(t);
        if (s && !seen.has(s)) { seen.add(s); out.push(s); }
    }
    info.wd.value = out.join(",");
    try { if (node.setDirtyValue) node.setDirtyValue(); } catch (e) {}
    try { if (node.dirtyWidget) node.dirtyWidget(info.wd); } catch (e) {}
    try { app.setDirtyCanvas(true, true); } catch (e) {}
    return info.wd.value;
}

function applyExclude(node, token, off) {
    const info = excludedTokens(node);
    if (!info) return "";
    let toks = info.tokens.slice();
    token = String(token);
    if (off) { if (!toks.includes(token)) toks.push(token); }
    else { toks = toks.filter((t) => t !== token); }
    return writeExcluded(node, toks);
}

// Manual re-pairing. Like the include/ignore toggle it edits ONLY a typed widget on the node - here
// `pair_overrides` - so the panel never authors the pairing itself: the plan node re-decides the
// whole thing when Process Plan runs. A token is `face=partner`: a face is a bare clip number or an
// `s`-prefixed reference-still number, exactly as excluded_faces and the preview number them, and
// the partner is another face on the other plate or the word `none`.
function pairOverrideTokens(node) {
    const wd = w(node, "pair_overrides");
    if (!wd) return null;
    const raw = (wd.value === null || wd.value === undefined) ? "" : String(wd.value);
    return { wd, tokens: raw.split(",").map((t) => t.trim()).filter((t) => t.length) };
}

function writePairOverrides(node, tokens) {
    const info = pairOverrideTokens(node);
    if (!info) return "";
    const seen = new Set();
    const out = [];
    for (const t of tokens) {
        const s = String(t);
        if (s && !seen.has(s)) { seen.add(s); out.push(s); }
    }
    info.wd.value = out.join(", ");
    try { if (node.setDirtyValue) node.setDirtyValue(); } catch (e) {}
    try { if (node.dirtyWidget) node.dirtyWidget(info.wd); } catch (e) {}
    try { app.setDirtyCanvas(true, true); } catch (e) {}
    return info.wd.value;
}

// Set faceToken's partner. Drop every entry that names faceToken on EITHER side first, so a face
// never ends up with two partners in the widget; then add the new one unless the partner is null.
function applyPairOverride(node, faceToken, partnerToken) {
    const info = pairOverrideTokens(node);
    if (!info) return "";
    faceToken = String(faceToken);
    const kept = info.tokens.filter((t) => {
        const eq = t.split("=");
        return (eq[0] || "").trim() !== faceToken && (eq[1] || "").trim() !== faceToken;
    });
    if (partnerToken !== null && partnerToken !== undefined) kept.push(faceToken + "=" + partnerToken);
    return writePairOverrides(node, kept);
}

function faceCell(face, currentToken, onPick) {
    const cell = el("div", "h3pp-face");
    if (face.thumb_url) {
        const img = el("img");
        img.src = api.apiURL(face.thumb_url);
        img.alt = tokenLabel(face.token);
        cell.appendChild(img);
    }
    cell.appendChild(el("span", null, tokenLabel(face.token)));
    if (face.token === currentToken) cell.classList.add("cur");
    cell.onclick = onPick;
    return cell;
}

// The face chooser: press one of a subject's face images and choose which face that side should be
// matched to. `candidates` is every face on the relevant plate; the cell matching the current
// choice is marked; a "no partner" button switches that side's partner off. A pick writes the
// node's pair_overrides widget - the plan node re-pairs the whole job when Process Plan runs.
// `apply(faceToken, pickedToken)` is the caller's handler; faceToken says which face is being
// re-pointed and pickedToken is the chosen face's token or the word none.
function openFaceChooser(node, data, faceToken, candidates, currentToken, title, apply) {
    const g = dialog(title);
    g.body.appendChild(el("p", "h3pp-note",
        "Pick which face to use for this side, or no partner. This writes the node's pair_overrides "
        + "widget - press Process Plan to apply it, and note it re-pairs the whole job, so its "
        + "passes must be re-run."));
    const grid = el("div", "h3pp-facegrid");
    for (const f of (candidates || [])) {
        grid.appendChild(faceCell(f, currentToken, () => { g.close(); apply(faceToken, f.token); }));
    }
    const none = el("button", "h3pp-face noneopt", "no partner");
    none.onclick = () => { g.close(); apply(faceToken, "none"); };
    grid.appendChild(none);
    g.body.appendChild(grid);
    g.foot.appendChild(button("Close", "", g.close));
}

function render(d, data, node, name) {
    d.body.innerHTML = "";
    d.foot.innerHTML = "";
    d.foot.appendChild(el("span", "h3pp-msg",
        "toggles edit the node's excluded_faces widget; click a face image to re-pair it (writes "
        + "pair_overrides). Process Plan re-runs the node into job_name and REPLACES its plan (the "
        + "previous plan.json is kept as plan_prev.json)"));
    const refresh = button("Refresh", "", () => { d.close(); openPanel(node, name); });

    if (!data.ready) {
        d.head.textContent = "Pairing preview - " + (data.job_name || "(no job_name)")
            + ": no plan yet";
        d.body.appendChild(el("p", "h3pp-note",
            data.message || data.error || "This job has no plan to show yet."));
        d.foot.appendChild(refresh);
        d.foot.appendChild(button("Close", "", d.close));
        return;
    }

    d.head.textContent = "Pairing preview - " + data.job_name + ": " + data.subject_count
        + " subject(s), match frame " + data.match_frame;

    // Run progress + frame-consistency: how many queues are left for the always-next walk, and
    // whether every saved pass still spans the clip's frame count (a stale/short pass is exactly
    // what makes H3 Face Stitch throw `index N out of bounds`). Both read off the job folder.
    const q = data.queue;
    if (q) {
        let line = "queues: " + q.total + " total (one per subject)";
        if (q.complete) line += ", " + q.complete + " done";
        if (q.remaining) {
            line += ", " + q.remaining + " still to run";
            if (q.next_index !== null && q.next_index !== undefined)
                line += " - next queue takes subject " + q.next_index;
        } else {
            line += " - all complete (press Reset run to start the passes over)";
        }
        d.body.appendChild(el("p", "h3pp-status", line));
    }
    if (data.frames && !data.frames.consistent) {
        d.body.appendChild(el("p", "h3pp-note h3pp-warn",
            data.frames_note || ("WARNING - saved passes do not all match the clip's "
                + data.frames.expected + " frame(s): " + JSON.stringify(data.frames.counts))));
        d.body.appendChild(button("Reset run (clear pass bookkeeping)", "h3pp-reset",
            () => { d.close(); resetRun(node); }));
    }

    if (data.reused) {
        const banner = el("p", "h3pp-note h3pp-reused",
            "This job's plan was REUSED on the last H3 Multi-Face Plan run - the node only plans "
            + "when there is no saved plan yet, or after Process Plan clears this folder, so running "
            + "the flow again never re-cuts or overwrites it.");
        d.body.appendChild(banner);
        const pending = data.unapplied || [];
        if (pending.length) {
            d.body.appendChild(el("p", "h3pp-note h3pp-warn",
                "Your node settings differ from the saved plan, so they were NOT applied to that "
                + "run (it reused the saved subject set to keep the existing passes valid). Press "
                + "Process Plan to apply them (the passes must then be re-run), or use a new "
                + "job_name. What differs:"));
            for (const line of pending) d.body.appendChild(el("p", "h3pp-sub h3pp-warn", line));
        }
    }

    const inputs = data.inputs || {};
    const box = el("div", "h3pp-inputs");
    // Every value here came out of plan.json, so none of it is ever spliced into markup: the
    // words around each widget setting are text nodes and each setting goes into its own <b>
    // by textContent, the way every other line in this panel is built.
    const setting = (value) => box.appendChild(el("b", "", value));
    setting("pairing as planned");
    box.appendChild(document.createTextNode(" - match frame "));
    setting(inputs.match_frame);
    box.appendChild(document.createTextNode(" - source "));
    setting(inputs.subject_source || "?");
    box.appendChild(document.createTextNode(" - tolerance "));
    setting(Number(inputs.match_tolerance).toFixed(3));
    box.appendChild(document.createTextNode(" of the clip diagonal ("));
    box.appendChild(document.createTextNode(Math.round(data.tolerance_px) + "px"));
    box.appendChild(document.createTextNode(") - unmatched_faces "));
    setting(inputs.unmatched_faces);
    box.appendChild(document.createTextNode(" - max_subjects "));
    setting(Number(inputs.max_subjects) > 0 ? inputs.max_subjects : "off");
    d.body.appendChild(box);

    if (data.preview_url) {
        const wrap = el("div", "h3pp-imgwrap");
        const img = el("img");
        img.src = api.apiURL(data.preview_url);
        img.alt = "both plates with every face outlined and numbered";
        wrap.appendChild(img);
        d.body.appendChild(wrap);
        d.body.appendChild(el("p", "h3pp-note",
            "Target clip on the left, reference still on the right. Face numbers are the numbers "
            + "the excluded_faces widget takes; a pair shares a colour; a switched-off face is "
            + "struck through."));
    } else {
        d.body.appendChild(el("p", "h3pp-note",
            "No pairing_preview.png in this job folder - re-run H3 Multi-Face Plan to write it."));
    }

    const widgetTokens = ((excludedTokens(node) || {}).tokens) || [];
    const tokenSet = new Set(widgetTokens);

    d.body.appendChild(el("h4", "h3pp-sect",
        "subjects (Ignore switches a face off; click a face image to re-pair that side)"));
    for (const s of (data.subjects || [])) {
        const row = el("div", "h3pp-row");
        const cell = el("div", "h3pp-cell");
        cell.appendChild(el("b", null, "subject " + s.index + " - " + s.pairing));
        cell.appendChild(el("small", null, kindNote(s)));
        if (s.has_reference && !s.ref_exists) {
            cell.appendChild(el("small", null, "reference file not on disk: " + s.ref_file));
        }
        if (s.exclude_token) {
            const tok = String(s.exclude_token);
            const off = tokenSet.has(tok);
            if (off) row.classList.add("off");
            cell.appendChild(button(off ? "Include this face again" : "Ignore this face",
                "h3pp-toggle", () => {
                    const now = applyExclude(node, tok, !off);
                    render(d, data, node, name);
                    d.msg.textContent = "excluded_faces on the node is now '" + now +
                        "' - press Process Plan to apply (this panel still shows the last plan).";
                }));
        }
        // Click either face image to re-pick that side. The clip image re-points the clip side of
        // the still face this subject tracks (face token s<still>); the reference image re-points
        // the reference of the clip face this subject tracks (face token <clip>). A side with no
        // partner on the other plate to re-point is left non-clickable.
        const applyPick = (faceTok, tok) => {
            const now = applyPairOverride(node, faceTok, tok);
            render(d, data, node, name);
            d.msg.textContent = "pair_overrides on the node is now '" + now +
                "' - press Process Plan to apply (this panel still shows the last plan).";
        };
        const canClip = (s.still_face !== null && s.still_face !== undefined);
        const canRef = (s.clip_face !== null && s.clip_face !== undefined);
        const curClip = (s.clip_face === null || s.clip_face === undefined) ? null : String(s.clip_face);
        const curRef = (s.still_face === null || s.still_face === undefined) ? null : "s" + s.still_face;
        const onPickClip = canClip ? () => openFaceChooser(
            node, data, "s" + s.still_face, data.clip_faces, curClip,
            "Subject " + s.index + ": pick the clip face", applyPick) : null;
        const onPickRef = canRef ? () => openFaceChooser(
            node, data, String(s.clip_face), data.still_faces, curRef,
            "Subject " + s.index + ": pick the reference face", applyPick) : null;
        row.appendChild(cell);
        row.appendChild(cropCell(s, onPickClip, onPickRef));
        d.body.appendChild(row);
    }

    // The switched-off list is shown from the UNION of the plan that was made and the widget as it
    // is now, so a pending Ignore that has not been re-planned yet is still visible (and reversible)
    // without digging into the widget. A Restore button pulls the token back out of the widget.
    const offTokens = [];
    {
        const seen = new Set();
        for (const t of (data.excluded_faces || [])) {
            const s = String(t);
            if (!seen.has(s)) { seen.add(s); offTokens.push(s); }
        }
        for (const t of widgetTokens) {
            if (!seen.has(t)) { seen.add(t); offTokens.push(t); }
        }
    }
    if (offTokens.length) {
        d.body.appendChild(el("h4", "h3pp-sect",
            "switched off in the excluded_faces widget - they keep their original pixels " +
            "(Process Plan applies the widget to a fresh plan)"));
        const wrap = el("div", "h3pp-chips");
        for (const t of offTokens) {
            wrap.appendChild(el("span", "h3pp-chip off", tokenLabel(t)));
            wrap.appendChild(button("Restore", "h3pp-toggle", () => {
                const now = applyExclude(node, t, false);
                render(d, data, node, name);
                d.msg.textContent = "excluded_faces on the node is now '" + now +
                    "' - press Process Plan to apply (this panel still shows the last plan).";
            }));
        }
        d.body.appendChild(wrap);
    }
    if ((data.left_out || []).length) {
        d.body.appendChild(el("h4", "h3pp-sect",
            "left out by unmatched_faces=leave - not a failure, and not refined"));
        d.body.appendChild(chips(data.left_out, "warn", (f) => f.pairing));
    }
    if ((data.capped_out || []).length) {
        d.body.appendChild(el("h4", "h3pp-sect", "capped by max_subjects"));
        d.body.appendChild(chips(data.capped_out, "warn", (f) => f.pairing));
    }

    d.foot.appendChild(refresh);
    d.foot.appendChild(button("Close", "", d.close));
}

/* ------------------------------------------------------------------- open */

async function openPanel(node, nameOverride) {
    // Refuse before anything is read, then again on the answer: the route is the one that can
    // actually see the queue, so its refusal is the authority and this one is only faster.
    if (renderRunning()) {
        const d = dialog("Pairing preview");
        showWait(d, node);
        return;
    }
    let name;
    if (nameOverride !== undefined && nameOverride !== null) {
        name = String(nameOverride);
    } else {
        const wd = w(node, "job_name");
        name = wd ? String(wd.value === null || wd.value === undefined ? "" : wd.value) : "";
    }
    const d = dialog("Pairing preview - " + (name || "(no job_name)"));
    d.msg.textContent = "reading the job folder...";

    let data;
    try {
        const r = await fetch(api.apiURL(ROUTE + "?job_name=" + encodeURIComponent(name)));
        data = await r.json();
    } catch (e) {
        d.head.textContent = "Pairing preview - request failed";
        d.body.textContent = "Could not reach the pairing route: " + e;
        d.msg.textContent = "";
        d.foot.appendChild(button("Close", "", d.close));
        return;
    }
    if (data && data.busy) { showWait(d, node); return; }
    d.msg.textContent = "";
    render(d, data || { ready: false, message: "The pairing route returned nothing." }, node, name);
}

/* ---------------------------------------------------------- process the plan */

// Process Plan runs the REAL plan node into its real job_name and REPLACES the plan. The plan node
// guards its folder against mixing two pairings, so before the run we ask the narrow replace route
// to clear the plan node's OWN stale artifacts (plan.json -> plan_prev.json, the preview, the
// subject/clip crops). The node then writes a fresh plan with nothing to refuse against. Pass
// stages are left for the pass/composite nodes' own guards, so an old pass cannot mix with the new
// plan - replacing a plan means its passes have to be re-run.

// The plan node is an output node, so a prompt of just it plus the nodes that FEED it runs it
// and stops there - no tracker, sampler or render. Walk the plan's link inputs upstream.
function ancestorSubgraph(prompt, rootId) {
    const keep = new Set();
    const stack = [String(rootId)];
    while (stack.length) {
        const id = stack.pop();
        if (keep.has(id) || !prompt[id]) continue;
        keep.add(id);
        for (const value of Object.values(prompt[id].inputs || {})) {
            if (Array.isArray(value)) stack.push(String(value[0]));
        }
    }
    const sub = {};
    for (const id of keep) sub[id] = prompt[id];
    return sub;
}

async function waitForPrompt(promptId, d) {
    const t0 = Date.now();
    for (;;) {
        await new Promise((res) => setTimeout(res, 500));
        let hist;
        // The frontend's api object has no .fetch() - the whole pack uses global fetch on
        // api.apiURL(...) (as openPanel and the face picker already do). A 404 while the job is still
        // queued is normal, so fall back to null rather than throwing.
        try {
            const hr = await fetch(api.apiURL("/history/" + encodeURIComponent(promptId)));
            hist = hr.ok ? await hr.json() : null;
        } catch (e) { hist = null; }
        const entry = hist && hist[promptId];
        const status = entry && entry.status;
        if (status && status.status_str === "error") {
            const msgs = (entry.messages || [])
                .map((m) => (Array.isArray(m) && m[1] ? JSON.stringify(m[1]) : "")).filter(Boolean);
            d.head.textContent = "Process Plan - the re-plan failed";
            d.body.textContent = msgs.join("\n") || status.summary ||
                "The plan node reported an error (it may have refused: nothing to plan, or a " +
                "pairing that seeds from a plate holding no faces).";
            d.foot.appendChild(button("Close", "", d.close));
            return false;
        }
        if (status && (status.completed || status.status_str === "success" || entry.outputs)) {
            return true;
        }
        if (Date.now() - t0 > 300000) {
            d.head.textContent = "Process Plan - timed out";
            d.body.textContent = "The re-plan run did not finish within 5 minutes.";
            d.foot.appendChild(button("Close", "", d.close));
            return false;
        }
    }
}

async function processPlan(node) {
    if (renderRunning()) {
        const d = dialog("Process Plan");
        showWait(d, node);
        return;
    }
    const wd = w(node, "job_name");
    const name = wd ? String(wd.value === null || wd.value === undefined ? "" : wd.value).trim() : "";
    const d = dialog("Process Plan" + (name ? " - " + name : ""));
    d.msg.textContent = "building a plan-only prompt from the current settingsâ€¦";

    let built;
    try {
        built = await app.graphToPrompt();
    } catch (e) {
        d.head.textContent = "Process Plan - could not read the graph";
        d.body.textContent = "The front end could not serialize the workflow: " + e;
        d.foot.appendChild(button("Close", "", d.close));
        return;
    }
    const prompt = built.prompt || built.output || {};
    const planId = String(node.id);
    const sub = ancestorSubgraph(prompt, planId);
    if (!sub[planId]) {
        d.head.textContent = "Process Plan - plan node not in the serialized graph";
        d.body.textContent = "The plan node had no serializable inputs to run on.";
        d.foot.appendChild(button("Close", "", d.close));
        return;
    }
    // Run the plan node into the REAL job_name. Clear its stale plan artifacts first (name-bearing
    // clicks only) so the fingerprint guard has nothing to refuse on and the node writes fresh.
    // With the widget blank the node derives a name we cannot see from the browser, so we cannot
    // pre-clear it: run anyway and surface the node's own refusal if the pairing changed.
    sub[planId].inputs = sub[planId].inputs || {};
    sub[planId].inputs["job_name"] = name;
    if (name) {
        try {
            await fetch(api.apiURL(ROUTE + "/replace"), {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ job_name: name }),
            });
        } catch (e) {
            d.head.textContent = "Process Plan - could not clear the old plan";
            d.body.textContent = "The replace route was unreachable: " + e;
            d.foot.appendChild(button("Close", "", d.close));
            return;
        }
    }

    // client_id is optional on /prompt; the frontend keeps it on api.clientId (there is no
    // api.getClientId()). Send it when present, so execution events route back, but never require it.
    const payload = { prompt: sub };
    if (api.clientId) payload.client_id = api.clientId;
    let promptId;
    try {
        const r = await fetch(api.apiURL("/prompt"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const resp = await r.json().catch(() => ({}));
        // A bad prompt comes back non-2xx with error / node_errors / error_details in the body;
        // surface that rather than treating a missing prompt_id as success.
        const errs = resp.node_errors || resp.error_details || resp.error;
        if (!r.ok || resp.error || (resp.node_errors && Object.keys(resp.node_errors).length)) {
            throw new Error(typeof errs === "string" ? errs : JSON.stringify(errs || ("HTTP " + r.status)));
        }
        promptId = resp.prompt_id;
    } catch (e) {
        d.head.textContent = "Process Plan - run rejected";
        d.body.textContent = String(e);
        d.foot.appendChild(button("Close", "", d.close));
        return;
    }
    d.msg.textContent = "detecting and re-pairing on the clip + reference (no render)â€¦";
    const ok = await waitForPrompt(promptId, d);
    if (!ok) return;   // waitForPrompt already wrote the reason into the dialog
    if (!name) {
        d.head.textContent = "Process Plan - plan replaced (derived job_name)";
        d.body.textContent = "The plan node wrote its derived job folder. Set job_name on the node, "
            + "then reopen the pairing preview to see the new plan.";
        d.foot.appendChild(button("Close", "", d.close));
        return;
    }
    d.close();
    await openPanel(node, name);
}

/* ------------------------------------------------------------------ reset run */

// Reset run clears a job's pass bookkeeping (the manifest's pass records + the subject cursor) so
// the always-next walk starts over at the first subject. It is non-destructive: plan.json, the
// references, the preview and the saved pass frames stay on disk - a re-run simply overwrites its
// own frames. The route answers how many completed pass records it dropped, which we report.
async function resetRun(node) {
    if (renderRunning()) {
        const d = dialog("Reset run");
        showWait(d, node);
        return;
    }
    const wd = w(node, "job_name");
    const name = wd ? String(wd.value === null || wd.value === undefined ? "" : wd.value).trim() : "";
    if (!name) {
        const d = dialog("Reset run - no job_name");
        d.body.textContent = "This plan node has no job_name, so there is no run to reset. Set "
            + "job_name on the node (or press Process Plan to write a job folder) first.";
        d.foot.appendChild(button("Close", "", d.close));
        return;
    }
    const d = dialog("Reset run - " + name);
    d.msg.textContent = "clearing the pass bookkeeping...";
    let data;
    try {
        const r = await fetch(api.apiURL(ROUTE + "/reset"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ job_name: name }),
        });
        data = await r.json().catch(() => ({}));
    } catch (e) {
        d.head.textContent = "Reset run - request failed";
        d.body.textContent = "The reset route was unreachable: " + e;
        d.foot.appendChild(button("Close", "", d.close));
        return;
    }
    if (data && data.busy) { showWait(d, node); return; }
    if (!data || !data.ok) {
        d.head.textContent = "Reset run - refused";
        d.body.textContent = (data && data.message) || "The reset route returned nothing.";
        d.foot.appendChild(button("Close", "", d.close));
        return;
    }
    const n = Number(data.cleared || 0);
    d.head.textContent = "Reset run - done";
    d.body.textContent = "Run reset. "
        + (n ? "Cleared " + n + " completed pass record(s). "
             : "No completed passes were recorded anyway. ")
        + "Every subject now reads as not-yet-processed, so the next queue starts again at the "
        + "first subject. The plan, the references and the saved pass frames on disk are untouched "
        + "- re-running a subject simply overwrites its own frames. Queue the graph again to start.";
    d.foot.appendChild(button("Close", "", d.close));
}

/* --------------------------------------------------------------- register */

app.registerExtension({
    name: "h3.facerefine.pair_preview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== PLAN) return;

        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = created ? created.apply(this, arguments) : undefined;
            const replan = this.addWidget("button", "Process Plan", null,
                                          () => processPlan(this));
            const reset = this.addWidget("button", "Reset run", null,
                                         () => resetRun(this));
            const btn = this.addWidget("button", "Pairing preview", null,
                                       () => openPanel(this));
            // A button holds no value, so it must not be serialised: a slot in widgets_values
            // would reinterpret every saved widget after it (ADR 0002 - the plan node's widget
            // order is permanent, and this adds no widget to it). Process Plan then Reset run are
            // added first so they render above Pairing preview on the node.
            for (const b of [replan, reset, btn]) {
                if (b) {
                    b.serialize = false;
                    b.serializeValue = () => undefined;
                }
            }
            return r;
        };
    },
});


