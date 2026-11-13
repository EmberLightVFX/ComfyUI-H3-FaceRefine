import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// A read-only pairing panel on H3 Multi-Face Plan (#9).
//
// Pairing is the failure mode this feature invents, and a render on a bad pairing costs N
// generations, so the pairing has to be checkable before anything is queued. The plan node
// already decides it and already draws both plates with every face outlined and numbered; this
// panel reads that decision back out of the job folder over one JSON route and shows it.
//
// READ-ONLY, on purpose and by construction: nothing here writes. No widget value is set, no
// route is posted to, no file is asked for. Exclusions stay the typed `excluded_faces` widget on
// the plan node - that is a scope line from the spec - and the interactive pair editor is a
// later ticket. The panel is also its own front end, separate from the face picker's, so the
// picker's behaviour stays untouched.
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
        ".h3pp-crop{width:132px;max-height:132px;object-fit:contain;background:#111;",
        "  border:1px solid #555;border-radius:4px}",
        ".h3pp-nocrop{width:132px;height:80px;border:1px dashed #555;border-radius:4px;color:#888;",
        "  font:11px sans-serif;display:flex;align-items:center;justify-content:center;",
        "  text-align:center;padding:6px;box-sizing:border-box;background:#1e1e1e}",
        ".h3pp-chips{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px}",
        ".h3pp-chip{font:12px sans-serif;color:#ccc;background:#2b2b2b;border:1px solid #4a4a4a;",
        "  border-radius:4px;padding:3px 8px}",
        ".h3pp-chip.warn{color:#f0c674;border-color:#6a5a2a}",
        ".h3pp-chip.off{color:#e08a8a;border-color:#6a3a3a}",
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

function cropCell(s) {
    const cell = el("div");
    if (s.ref_url) {
        const img = el("img", "h3pp-crop");
        img.src = api.apiURL(s.ref_url);
        img.alt = s.label || ("subject " + s.index);
        cell.appendChild(img);
        cell.appendChild(el("small", "h3pp-cell", s.ref_file));
        return cell;
    }
    const why = !s.has_reference
        ? "no subject reference\n(unmatched_faces=no_reference)"
        : "subject reference file missing\nre-run H3 Multi-Face Plan";
    cell.appendChild(el("div", "h3pp-nocrop", why));
    return cell;
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

function render(d, data, node) {
    d.body.innerHTML = "";
    d.foot.innerHTML = "";
    d.foot.appendChild(el("span", "h3pp-msg",
        "read-only - nothing here writes to the graph or to disk"));
    const refresh = button("Refresh", "", () => { d.close(); openPanel(node); });

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

    d.body.appendChild(el("h4", "h3pp-sect", "subjects"));
    for (const s of (data.subjects || [])) {
        const row = el("div", "h3pp-row");
        const cell = el("div", "h3pp-cell");
        cell.appendChild(el("b", null, "subject " + s.index + " - " + s.pairing));
        cell.appendChild(el("small", null, kindNote(s)));
        if (s.has_reference && !s.ref_exists) {
            cell.appendChild(el("small", null, "reference file not on disk: " + s.ref_file));
        }
        row.appendChild(cell);
        row.appendChild(cropCell(s));
        d.body.appendChild(row);
    }

    if ((data.excluded_faces || []).length) {
        d.body.appendChild(el("h4", "h3pp-sect",
            "excluded - switched off, so they keep their original pixels"));
        d.body.appendChild(chips(data.excluded_faces, "off",
            (i) => (typeof i === "string" && i[0] === "s"
                ? "reference still face " + i.slice(1)
                : "clip face " + i) + " (the excluded_faces widget)"));
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

async function openPanel(node) {
    // Refuse before anything is read, then again on the answer: the route is the one that can
    // actually see the queue, so its refusal is the authority and this one is only faster.
    if (renderRunning()) {
        const d = dialog("Pairing preview");
        showWait(d, node);
        return;
    }
    const wd = w(node, "job_name");
    const name = wd ? String(wd.value === null || wd.value === undefined ? "" : wd.value) : "";
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
    render(d, data || { ready: false, message: "The pairing route returned nothing." }, node);
}

/* --------------------------------------------------------------- register */

app.registerExtension({
    name: "h3.facerefine.pair_preview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== PLAN) return;

        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = created ? created.apply(this, arguments) : undefined;
            const btn = this.addWidget("button", "Pairing preview", null,
                                       () => openPanel(this));
            // A button holds no value, so it must not be serialised: a slot in widgets_values
            // would reinterpret every saved widget after it (ADR 0002 - the plan node's widget
            // order is permanent, and this adds no widget to it).
            if (btn) {
                btn.serialize = false;
                btn.serializeValue = () => undefined;
            }
            return r;
        };
    },
});


