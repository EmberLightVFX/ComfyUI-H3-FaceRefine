# ComfyUI-H3-FaceRefine

**A ComfyUI custom node set to refine and improve the quality of small faces in MiniMax H3 video.**

MiniMax H3 renders faces poorly when the head occupies a small fraction of the frame. This is a property
of head-size-in-frame, not of output resolution, so it persists at 720p and above. These nodes
detect the face on every frame, crop to it so it fills a canvas, let H3 re-generate it, and
composite the result back into the original video.

Modelled on [Impact Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)'s **FaceDetailer**,
adapted from stills to video.

---

## Results

Source on the left, refined on the right:

![source vs refined, side by side](screenshots/COMPARISON.gif)

Single frame at full resolution:

| Source | Refined |
|---|---|
| ![source frame](screenshots/INPUT_00001.png) | ![refined frame](screenshots/REFINED_00001.png) |

The reason it works is what H3 is handed. Instead of a distant head a few dozen pixels tall, it
gets the face tracked and normalized to fill the canvas:

| Crop in: what H3 sees | Crop out: what H3 returns |
|---|---|
| ![input crop](screenshots/CROPS_INPUT_00001.gif) | ![refined crop](screenshots/CROPS_00001.gif) |
| [full-res still](screenshots/CROPS_INPUT_00001.png) | [full-res still](screenshots/CROPS_00001.png) |

These two are what to watch for temporal behaviour: the box has to sit still on a moving subject,
or the refined face boils.

---

## What's new in 1.2.0

**Several people in one clip is now one job, not a stack of copies of the graph.** Five nodes run a
multi-face clip as one **pass** per subject — the released pipeline repeated, once per person, with
the bookkeeping in a **job folder** instead of in your clipboard:
[H3 Multi-Face Plan](#h3-multi-face-plan) decides who the subjects are and cuts each one's
**subject reference** out of the **reference still**, [H3 Subject Select](#h3-subject-select) hands
one subject to the released tracker, [H3 Pass Save](#h3-pass-save) and
[H3 Pass Load](#h3-pass-load) checkpoint each pass, and
[H3 Multi-Face Composite](#h3-multi-face-composite) refuses until every subject has landed.
The seven nodes already in the pack are byte-for-byte unchanged; a single-subject graph loads and
behaves exactly as it did (ADR 0002).

**The cost, stated up front: one generation per subject.** A job with three subjects costs three
H3 renders of the whole clip, not one. The plan node prints that number before you queue —
`2 subject(s) x 4 frame(s) @ 768x768 = 8 frame(s) to generate` — so the cost is a decision you make
looking at the graph, not a surprise an hour in. The size on that line is an **estimate**: the crop
canvas belongs to [H3 Face Track + Crop](#h3-face-track--crop), which sets it on its own widgets and
can switch it to `auto`. The one size this node does decide is the **subject reference** crop, and it
says which `crop_factor` it cut those references with.

**Why one pass per subject:** H3's conditioning is **whole-frame**, not per face. A subject
reference does not travel with the face it came from; it conditions the whole generated frame.
Hand H3 two subject references at once, or generate both people in one render, and the reference
for one face gets applied to whoever is nearest it — a face that shares a frame with another
inherits the other person's features. So each pass carries exactly one subject's picture, and the
several passes are recombined in the clip rather than in the latent. See
[docs/adr/0001](docs/adr/0001-one-generation-per-subject.md).

**Composite order is the plan's order, and it is deliberately not a widget.** Subjects are
numbered left to right at the **match frame**, and passes are composited in that order. A widget
would let the order drift from the numbering the reports and the **pairing preview** use, and then
"subject 1" would mean two different things in one job. What each pass composites onto is the
previous pass's result, so the last subject's stitched frames *are* the finished clip — see
[The chaining recipe](#h3-multi-face-composite).

**A job resumes where it stopped, and knows which subject that was.** Every pass writes its stages
into `<output>/<job_name>/`; `manifest.json` is the **pass manifest**, and `next` on H3 Subject
Select is the resume cursor. A subject only counts as done once its `stitched` stage is recorded —
a pass that died after the crops is re-run, not skipped.

**A face that never pairs gets an explicit answer.** `unmatched_faces` on the plan node chooses
between leaving it alone, refining it with no reference, or conditioning it on the whole reference
still. The default is `leave`, and the report says out loud that leaving is not a failure — see
[unmatched_faces](#h3-multi-face-plan).

**The pairing is checkable before anything is queued.** The plan node now draws both plates with
every face numbered (`pairing_preview.png` in the job folder) and carries a read-only panel that
shows that pairing and each subject's reference. A bad pairing costs N renders, so it has to be
visible at the node that made it — see [the pairing preview panel](#h3-multi-face-plan).

**Two pairings can never end up in one clip.** The plan node, both pass nodes and the composite
compare the plan in the graph against the `plan.json` already in the job folder — the
**pairing fingerprint** — and name every input that differs. This is the failure mode of
run-per-subject workflows: a job folder holding passes cut from two different ideas of who is who
produces a clip that quietly swaps faces, and it looks finished.

A third template ships with this: **H3 Face Refine Multi Face**
(`example_workflows/H3_Face_Refine_Multi_Face.json`), a two-subject job wired end to end.

---

## What's new in 1.1.2

**[H3 Face Track + Crop](#h3-face-track--crop) no longer jumps to another person when it loses the
subject.** When the subject's face stopped being detected - a head turned away, a spin, a face
covered - while anyone else was in shot, the tracker moved onto the nearest other face and stayed
there, and the report still counted every frame as found. It now only carries the subject onto a
face within reach of where they just were. Frames with no such face are left unresolved: the crop is
interpolated across them and the composite fades out, so they keep their original pixels. The
subject is picked up again when their face reappears near where it was lost. Frames before a shot's
lock-on are filled back under the same rule, and shots that never hold more than one face track
exactly as before. See [When the subject is lost](#choosing-which-face).

**The report says when the subject was lost.** A new `lost:` line counts those frames and lists
their ranges, and `face=` and `dropout runs` include them, instead of reporting every frame as found.
In `preview` those frames are drawn red - see [Reading `preview`](#h3-face-track--crop).

**`fallback_detector` uses the body nearest where the subject's head should be**, rather than the
largest body in frame.

Nothing needs rewiring, and saved workflows load unchanged.

---

## What's new in 1.1.0

**[H3 Load Video + Face Select](#h3-load-video--face-select) is new.** It loads the video, finds
every face and every hard cut in one pass, and settles which face is the subject *before* the graph
runs, so the tracker never detects the same video twice. It brings two interactive surfaces with it:
[Pick faces](#pick-faces), for choosing the subject per shot by eye, and
[Preview coordinates](#preview-coordinates), for checking where an `X`, `Y` lands.

**That node adds one dependency: [PySceneDetect](https://github.com/Breakthrough/PySceneDetect)**
(`scenedetect>=0.7`), which installs automatically with the pack. It backs
`cut_detection = auto (pyscenedetect)` on both the new node and `H3 Face Track + Crop`, finding
the **hard cuts** in a clip. Cuts matter because a cut renumbers every face and breaks continuity:
with them known, the subject is chosen once per shot and the smoothing, interpolation and
composite fade all run per shot instead of being dragged across the join. It is fed frame by frame
from the decode pass that is already running, so it costs no second decode of the video.
`cut_threshold` is PySceneDetect's own adaptive default of `3.0`.

It is a soft dependency in practice: if it is missing or fails to import, the run does not stop.
The video is treated as a single shot and the report says so — `cut detection unavailable,
treating the video as one shot`.

**[H3 Per-Frame Denoise](#h3-per-frame-denoise) now takes and returns `MODEL`.** A per-frame mask
needs two changes to the model rather than the latent, so the node must sit in the model path and
its `model` output has to reach the guider. Its two strength widgets were renamed
`denoise_multiplier_small_face` and `denoise_multiplier_large_face`, which is what they always were.

> **This breaks workflows saved against 1.0.0.** `model` is a *required* input, so an old graph fails validation with `Required input is missing: model` until you route the model through the node and take its `model` output onward to the guider. Nothing else in this release requires rewiring.

**[H3 Face Track + Crop](#h3-face-track--crop) gained** a `frame_count` output to wire into the H3
node's `length`, a `face_pick` input for the node above, `absent_shots` for skipping shots the
subject is not in, `identity_model` for choosing how faces are compared, cut detection, and the
expanded `select` list below.

**`select` went from two modes to nine.** 1.0 offered `largest` and `most_central`; there are now
`largest_face`, `smallest_face`, `left_most`, `right_most`, `top_most`, `bottom_most`, `centre_most`,
`closest_to_xy` and `detector_score`, with direction in the name rather than a separate order widget.
Saved workflows migrate on load: `largest` becomes `largest_face`, `most_central` becomes
`centre_most`.

**`canvas_width` / `canvas_height` now default to `768`**, H3's native short edge, rather than `512`.

**The auto canvas modes now clamp up to a minimum of 512×512.** They size the canvas from the
largest crop, and the crop is bounded by the source frame, so on a small face in a low-resolution
clip the canvas used to track the crop down to a couple of hundred pixels — handing H3 the same
small face it renders badly. `manual` is unaffected and still takes whatever you type.

---

## Installation

Clone into `ComfyUI/custom_nodes/`:

```bash
git clone https://github.com/Carasibana/ComfyUI-H3-FaceRefine.git
```

Restart ComfyUI. The nodes appear under **MiniMax H3/Face Refine**.

### Requirements

**For the nodes themselves:**

| | |
|---|---|
| ComfyUI with MiniMax H3 support | H3's own nodes are **core** (`comfy_extras/nodes_minimax_h3.py`), not an add-on, you just need a build recent enough to have them |
| a face detector | e.g. `face_yolov8m.pt` in `models/ultralytics/bbox/`. The one thing you must supply yourself. For anime, use an anime face model instead. The **nodes** find it there on their own; the **example workflows** additionally need the folder registration described below, because their stored detector values carry its `bbox\` prefix |

Python packages (`ultralytics`, `scipy`, `insightface`, `scenedetect`) install automatically
from `requirements.txt` / `pyproject.toml`. `scenedetect` is PySceneDetect, used only by
`cut_detection`; without it cut detection is skipped and the clip is treated as one shot.

> **A note on onnxruntime.** `insightface` needs it, and this pack deliberately does not pin
> a variant. If you install `onnxruntime-gpu` *alongside* an existing `onnxruntime`, the
> CPU-only package shadows it, `CUDAExecutionProvider` disappears, and identity matching
> silently runs on CPU. Install one or the other, not both. Check with:
> `python -c "import onnxruntime; print(onnxruntime.get_available_providers())"`

**Additionally, to run the example workflows.** The first three are needed to queue either template
as shipped; the last two back a node pair that ships **muted**:

| | |
|---|---|
| [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) | **Both templates**, for the save (`VHS_VideoCombine`). **Auto Select** also loads the source with it (`VHS_LoadVideoPath`); **Manual Select** loads video with this pack's own `H3 Load Video + Face Select` instead, so it needs VideoHelperSuite only to save |
| [ComfyUI-H3-NativeAudioLock](https://github.com/Shrek3OnVH5/MiniMax-H3-NativeAudio-MusicVideo-Workflow) | **Both templates.** Supplies `MiniMaxH3NativeAudioLock`, which drives lipsync. It ships inside that repository under `custom_nodes/ComfyUI-H3-NativeAudioLock`, so copy that folder into your own `ComfyUI/custom_nodes/` |
| [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) | **Both templates.** Its subpack registers the `ultralytics_bbox` / `ultralytics` model folders, and the templates' detector values are stored in that registration's naming — `bbox\face_yolov8m.pt` and `segm\person_yolov8m-seg.pt`. Without it those values are not in the dropdown and the prompt fails validation on load. It also supplies `SAMLoader` if you unmute the SAM mask pair |
| [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) | only if you unmute the GGUF loader pair. Both workflows ship it muted in favour of the stock loaders |
| a SAM model in `models/sams/` | only if you unmute the SAM mask pair, alongside Impact Pack above |

Model lookups go through ComfyUI's `folder_paths`, so anything registered in
`extra_model_paths.yaml` is found automatically.

### Models

| Model | Goes in | Source |
|---|---|---|
| `face_yolov8m.pt` | `models/ultralytics/bbox/` | [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer/blob/main/face_yolov8m.pt) |
| `person_yolov8m-seg.pt` *(optional for the nodes, but both templates ship it wired as `fallback_detector`)* | `models/ultralytics/segm/` | [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer/blob/main/person_yolov8m-seg.pt) |
| an anime face detector *(optional, anime and other illustration)* | `models/ultralytics/bbox/` | [deepghs/anime_face_detection](https://huggingface.co/deepghs/anime_face_detection), or Anzhc's face-seg models. `face_yolov8m.pt` may not find anime faces — see [Anime and other non-photographic material](#anime-and-other-non-photographic-material) |
| a CLIP Vision model *(optional, only for `identity_model = clip_vision`)* | `models/clip_vision/` | any CLIP Vision model, wired through a `CLIPVisionLoader`. You probably already have one for IPAdapter or Redux |
| `minimax_h3_fl2va_pruned_int8_convrot.safetensors` *(the frame+last-frame-to-video model, int8 — this is what the example workflows load)* | `models/diffusion_models/` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` *(Qwen3-VL text encoder, nvfp4_awq — what the example workflows load)* | `models/text_encoders/` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) |
| `minimax_h3_video_vae_fp16.safetensors` | `models/vae/` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae) |
| `minimax_h3_audio_vae_fp32.safetensors` | `models/vae/` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae) |
| `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` *(optional turbo LoRA, big speed-up)* | `models/loras/` | [Kijai/MiniMax-H3_comfy](https://huggingface.co/Kijai/MiniMax-H3_comfy/tree/main/loras), distilled by [lightx2v](https://huggingface.co/lightx2v/Minimax-h3-Turbo). The example workflows ship this **8-step v1.0** at strength `1.0`, with `BasicScheduler` set to 8 steps to match — the step count must follow whichever turbo LoRA you load |
| `sam_vit_b_01ec64.pth` *(optional, only if you unmute the SAM mask pair)* | `models/sams/` | [Meta segment-anything](https://github.com/facebookresearch/segment-anything#model-checkpoints) |
| InsightFace `buffalo_l` *(optional, crowd tracking on photographed faces)* | `models/insightface/` | downloaded automatically on first use. Backs `identity_model = insightface`; on illustration use `clip_vision` or `ccip` instead |

H3's original weights are [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3); the
Comfy-Org repository above is the repackaged form ComfyUI expects. The example workflows load the
model and text encoder through the stock `UNETLoader` and `CLIPLoader`. Their stored values name the
specific quantisations the examples were built against, so unless you happen to have those exact
files you will need to re-pick both loaders after opening a template. Each stored value is a **bare
filename with no subfolder**, because that is where the table above says the file goes: a loader
resolves its value inside its own folder, so a name carrying a subfolder asks for a path no
installer creates. A test asserts every active loader in every shipped template follows that rule.
A **GGUF** pair ([ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)) sits beside them, muted, with a note:
unmute it and mute the stock pair to fit H3 in 12 GB.

**If a template fails validation on load, check which of the two it is**, because the fix is
different and the error looks the same. Either a *dependency* is missing, so a stored name is not in
any dropdown — the usual cause is [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)
not being installed, which is what registers `models/ultralytics/bbox/` and makes
`bbox\face_yolov8m.pt` resolvable — or you simply hold *different files*, in which case re-pick the
loader to whatever you do have. Neither means the template itself is broken.

Only the face detector is genuinely required by the nodes themselves. Everything else is needed
because the pipeline runs H3, so if you are already generating H3 video you will have it already.

---

### Reading videos from outside ComfyUI

`H3 Load Video + Face Select` takes a path to any video on your machine, because source footage
usually lives somewhere other than ComfyUI's `input` folder. Two things follow from that.

**Browse uploads; pasting does not.** The **Browse…** button copies the file you pick into
ComfyUI's `input` folder and refers to it by name from then on, exactly like the stock image
loaders. Pasting a full path instead leaves the file where it is and reads it in place — which is
what you want for a 4 GB clip you would rather not duplicate.

**Previews of pasted paths are served by this pack.** ComfyUI's own file server only reaches its
`input` folder, so a pasted path needs a route of its own. That route reads any file with a video
extension that ComfyUI itself can read, named by the request. ComfyUI listens on localhost only
unless you have deliberately changed that, so on a normal single-machine setup this is the same
access you already have.

If ComfyUI is reachable from anywhere else — `--listen`, a tunnel, a shared box — set this to
confine those routes to ComfyUI's own folder:

| variable | effect |
|---|---|
| `H3_FACEREFINE_STRICT_PATHS` | Unset (default): paths anywhere are read. Set to any non-empty value: only paths inside ComfyUI's own folder are read; everything else is refused with a message saying so. |

With it set, use **Browse…** to bring a video in rather than pasting a path to it.

Setting it, in the launcher you already use:

```bat
:: run_nvidia_gpu.bat, before the python line
set H3_FACEREFINE_STRICT_PATHS=1
```

```powershell
# PowerShell, for the session
$env:H3_FACEREFINE_STRICT_PATHS = "1"
```

```bash
# Linux / macOS
export H3_FACEREFINE_STRICT_PATHS=1
```

It is read per request, so it takes effect as soon as ComfyUI restarts with it set.

---

## Quick start

Three example workflows are included, all annotated in-graph. The first two differ in **how the
subject is chosen**; everything downstream is identical. The third runs that same chain once per
**subject** for a clip with more than one person in it:

| Template | File | Which face gets refined | Video loaded by |
|---|---|---|---|
| **H3 Face Refine Auto Select** | `H3_Face_Refine_Auto_Select.json` | a ranking rule, `largest_face` as shipped | your own video loader |
| **H3 Face Refine Manual Select** | `H3_Face_Refine_Manual_Select.json` | you, in [Pick faces](#pick-faces), one per shot | `H3 Load Video + Face Select` |
| **H3 Face Refine Multi Face** | `H3_Face_Refine_Multi_Face.json` | the **subject plan**: every face the pairing finds, one **pass** each | `H3 Load Video + Face Select` |

The first two ship the SAM mask pair present but **muted**, so neither needs a SAM model to run. Unmute the
pair to swap the rect paste mask for a face-shaped one, and drop `feather` to 4-8 when you do.
Impact Pack itself is still needed either way — see [Requirements](#requirements) — because the
templates' detector values use the model-folder naming its subpack registers.

All three are installed as ComfyUI **workflow templates**. Open them from
**Workflow → Browse Templates** once the pack is installed, no file hunting required.
They also live in [`example_workflows/`](example_workflows) if you would rather load the JSON
directly.

### Common to both

1. **Ref image(s)** — the same character references you generated the video with. `Ref 1` is the
   identity; `Ref 2` is optional, for wardrobe or a second look.
2. **Prompt** — the video's own prompt.

That is all either template needs beyond the video. `length` on the H3 node is already wired from
the track node's `frame_count`, so it follows whatever clip you load, and `width`/`height` are wired
from `canvas_w`/`canvas_h` so the crop and the latent cannot disagree.

Audio comes from the source clip. If you have an isolated vocal track for lipsync, unmute the
separate audio loader and wire it in place of the clip's own audio.

### Auto Select — a rule picks the face

Point `VHS_LoadVideoPath` at your video and queue. The tracker detects every face and takes the one
`select` names — `largest_face` as shipped, with `select_index` taking the 2nd, 3rd and so on out of
that ranking. See [Choosing which face](#choosing-which-face) for the full vocabulary.

The subject is chosen **once per shot**, not per frame, and continuity follows that same face from
there. If the clip has hard cuts, set `cut_detection` to `auto (pyscenedetect)` so each shot chooses again — a cut
renumbers everyone, so a rule can otherwise land on a different person after the join.

Use this when one rule describes your subject for the whole clip: they are the biggest face, or the
only face, or reliably centre frame.

### Manual Select — you pick the face

Here `H3 Load Video + Face Select` replaces the video loader and does the detection itself, handing
the tracker its boxes, its cuts and your choice on `face_pick`. The tracker never decodes or detects
the clip a second time.

1. **Point it at your video** — **Browse…** copies a file into ComfyUI's `input`, or paste any
   absolute path, since source footage usually lives elsewhere.
2. **Set `cut_detection` to `auto (pyscenedetect)`** if the clip has hard cuts, so it is split into shots.
3. **Click `Pick faces`.** The dialogue scans once, then shows a frame per shot with every face
   outlined and numbered. Choose your subject in each one and press **Use these** — see
   [Pick faces](#pick-faces).
4. **Queue.**

With `select` on `manual` the node will not run until a pick exists, so an unanswered graph fails
fast and says so rather than silently refining whoever happens to rank first. The other `select`
modes work here too if you would rather it choose: `identity_reference` matches a wired reference
image per shot, and the ranking rules behave exactly as they do on the tracker.

Use this when a rule cannot describe your subject: a crowd, a face that is not the biggest, or a
person who has to be followed across cuts that renumber everyone.

### How it fits together

```
FRONT END, one of the two:
  Auto Select    VHS_LoadVideoPath           -> images, audio
  Manual Select  H3 Load Video + Face Select -> images, audio, face_pick

  images ─────────────────────────────────────────────────────────────┐
     │                                            (face_pick, Manual) │
     ▼                                                                │
  H3 Face Track + Crop                                                │
     ├── canvas_w/h, frame_count ──► MiniMaxH3ReferenceToVideo        │
     │                               (refs + prompt)                  │
     │                                      │ av_latent               │
     ├── crops ──────────────────► H3 Inject Video Latent             │
     │                                      │                         │
     │        UNETLoader ► turbo LoRA ─┐    │                         │
     │                                 ▼    ▼                         │
     │                      MiniMaxH3NativeAudioLock ◄── audio        │
     │                                 │ model + av_latent            │
     ├── transform ──────────► H3 Per-Frame Denoise                   │
     │                            │ model        │ av_latent          │
     │                            ▼              ▼                    │
     │                   BasicGuider     SamplerCustomAdvanced        │
     │                   BasicScheduler           │                   │
     │                                            ▼                   │
     │                                        VAEDecode               │
     │                                            │                   │
     └── transform ──────────► H3 Face Stitch Back ◄──────────────────┘
                                       │
                          original audio ──► save
```

**`H3 Per-Frame Denoise` sits in the model path, not beside it.** It takes `MODEL` in and returns a
patched `MODEL` out, and that output must reach `BasicGuider` and `BasicScheduler`. Route the
unpatched model straight past it and the mask still applies but the held frames come back nearly
clean instead of at the sampler's current sigma — with no error to tell you.

The `transform` output is the spine of the whole thing: it records where every crop came from, so
the stitch can put each refined face back exactly where it belongs.

### Multi Face — the plan picks the subjects

`H3_Face_Refine_Multi_Face.json` is the Manual Select front end plus the plan, wired as **one
reusable pass chain** that covers any number of subjects: **H3 Subject Select** in Increment mode walks the
plan, and **H3 Pass Load** chains each pass onto the previous subject's stitched result (its
`no_previous` clip carries the first subject). You queue it once per subject rather than copying a
branch per person.

1. **Point `H3 Load Video + Face Select` at your clip** and set `cut_detection` to
   `auto (pyscenedetect)` if the clip has hard cuts.
2. **Load the reference still** into `Load Image`, and set `match_frame` on **H3 Multi-Face Plan**
   to a frame every subject is clearly in — that one frame is where the pairing happens.
3. **Run the plan node first, on its own.** Read the cost line and the pairing line, and open
   `pairing_preview.png` or press **View pairing preview** to check who was paired with whom. A
   wrong pairing costs one render per subject, so this is the cheapest moment to fix it.
4. **Queue once per subject.** The single chain refines one subject per queue: **H3 Pass Load**
   bases this subject's stitch on the previous subject's stitched pass (`previous_subject` from
   Subject Select), and for the first subject it returns the target clip from its `no_previous`
   input. The Composite refuses until every subject's stitched stage has landed, so it raises on
   every queue but the last — the completion gate, not a failure: the pass it waited on is already
   saved, so the next queue resumes where you left off, and the final queue writes the clip.
5. **Interrupted?** Queue again with nothing changed: `subject = next` takes the first subject with
   no completed **stitched** pass recorded in the job folder.

The template says which model files it expects on a **Note node in the graph** (`Model files this
template expects`) rather than burying them in this README, because they sit beside the loaders they
describe — and it carries a second note on the job folder and the resume rule. Swap the filenames for
what is in your models folder, and keep `steps` on the `BasicScheduler` node matching the turbo
LoRA's step count.

Use this when more than one face in the clip needs work. Accept the cost first: **N subjects is N
renders of the whole clip.**

### How it fits together with more than one subject

The chain above is one **pass**. A multi-face job runs it once per **subject** and keeps the state
in the **job folder** rather than in the graph:

```
FRONT END, once per job:
  H3 Load Video + Face Select  -> images, audio, face_pick
  images ─┬─ + reference_still ──► H3 Multi-Face Plan
          │                             │ writes the job folder, and the plan
          │                             │   plan.json · pairing_preview.png
          │                             │   subject_0.png · subject_1.png …
          │        plan ────────────┐   │
          │            face_pick ─┐ │   │
          ▼                       ▼ ▼   ▼
  (per subject, in plan order)  H3 Subject Select  -> face_pick, subject_reference
                                    │ narrowed pick
                                    ▼
                                H3 Face Track + Crop ── crops, transform ──► the released H3 chain
                                    │            │                                    │ refined crops
                                    │            └─► H3 Pass Save stage=crops ──► passes/subject_NNNN/
                                    │                    (+ crops.json)                   crops/ …
                                    ▼
                                H3 Face Stitch Back
                                    ▲ base_images = the previous pass's stitched frames
                                    │              (the first pass takes the target clip)
                                    ▼
                                H3 Pass Save stage=stitched ──► passes/subject_NNNN/stitched/ …
                                    │
                          … queue again, `subject = next` moves to the next subject …

  H3 Pass Load (subject = Subject Select's previous_subject = the one before; its `no_previous` carries the target clip for the FIRST subject) ──► that stitch node's base_images

  H3 Multi-Face Composite (plan, the target clip, overlap_threshold)
        │ refuses unless every subject's stitched stage is in manifest.json
        ▼
  the composited clip  +  the target clip's audio  ──► save
```

Two things to hold onto. **The job folder is the pipeline's memory**: a queued pass re-reads the
plan and the subject references from it, so nothing downstream needs the reference still loaded,
and `manifest.json` is what makes an interrupted job resumable. **The graph is not a loop**: you
queue the same graph again with `subject = next`, or you wire one branch per subject and chain them
with Pass Load. ComfyUI has no way to iterate over a plan, which is why the chaining is explicit
wiring and the counting is done by the manifest.

---

# Node reference

## H3 Load Video + Face Select

<img src="screenshots/H3%20Load%20Video%20%2B%20Face%20Select.png" alt="H3 Load Video + Face Select" width="380">

Loads a video, finds every face and every hard cut in one pass, and decides which face is the
subject **before the graph runs**. Optional: without it the tracker detects and ranks exactly as it
always has.

It replaces the video loader when used, so it emits the images and the audio as well as the
selection.

**Inputs**

| Input | Default | What it does |
|---|---|---|
| `video` | - | The video. **Browse** uploads one into ComfyUI's `input` folder, or paste any path - source footage usually lives elsewhere. Surrounding quotes are stripped, so a path copied from Explorer works as pasted. |
| `detector` | first found | Face detection model. This node owns detection when it is wired in — it detects once and hands the boxes to the tracker. |
| `confidence` | `0.35` | The **detector's** score floor - shown as `detector_confidence`. Lower catches more profiles and small faces, at the cost of false positives. Nothing to do with identity matching; that is `identity_threshold`. |
| `select` | `manual` | How the subject is decided. `manual` means you review the shots and choose; `identity_reference` matches a wired reference image; everything else is a rule that guesses - same vocabulary as the tracker, see [Choosing which face](#choosing-which-face). |
| `select_index` | `0` | Which face in that ranking is the subject. |
| `confirmed_pick` | - | Your reviewed answer while `select` is `manual`: **one face number per shot**, comma separated - `0,3,4`. Written by **Pick faces** and saved with the workflow. It is dropped automatically if you change the video, the detector, the frame range or the cut settings, because those change the shots it describes. |
| `cut_detection` *(opt)* | `none` | `auto (pyscenedetect)` finds hard cuts, so the subject is chosen per shot and the tracker does not smooth across a cut. |
| `cut_threshold` *(opt)* | `3.0` | How far a frame has to stand out from its neighbours to count as a cut. |
| `skip_first_frames` *(opt)* | `0` | Drop this many frames from the start. The audio is cut to match. |
| `frame_load_cap` *(opt)* | `0` | Stop after this many frames. 0 loads everything. |
| `select_every_nth` *(opt)* | `1` | Keep every nth frame. Above 1 the reported `fps` changes with it. |
| `identity_reference` *(opt)* | - | A picture of the person to refine. Each shot picks the face matching it, so the same person is followed across a cut without hand-picking. A frame of the clip works. Setting `select` to `identity_reference` without wiring one is a hard error. Wiring one while `select` is set to anything else is silent — the reference is simply never read. |
| `identity_clip_vision` *(opt)* | - | A `CLIP_VISION` model from a `CLIPVisionLoader`. Only for `identity_model = clip_vision`. |
| `identity_model` *(opt)* | `insightface` | Shown only when `select` is `identity_reference`. `insightface` for photographic faces; `clip_vision` or `ccip` for illustration and anime. See [Anime and other non-photographic material](#anime-and-other-non-photographic-material). |
| `identity_threshold` *(opt)* | `0.28` | Shown only when `select` is `identity_reference`. How close a match has to be to count as the same person. The scale depends on `identity_model` — **set `0` to use whichever default the chosen model recommends.** |
| `X` / `Y` *(opt)* | `0` | Shown only when `select` is `closest_to_xy`. The point to measure from, in pixels of the **source** frame, origin top-left. See [Preview coordinates](#preview-coordinates). |
| `frame_index` *(opt)* | `0` | Shown only when `select` is `closest_to_xy`. The frame the `X`, `Y` measurement is taken on, counting from `0` over the frames this node loaded — that is, after `skip_first_frames`, `frame_load_cap` and `select_every_nth`. |

**Outputs**

| Output | Goes to |
|---|---|
| `images` | `H3 Face Track + Crop` → `images`, and `H3 Face Stitch Back` → `base_images` |
| `audio` | The save node, and `MiniMax H3 Native Audio Lock` if you are driving lipsync |
| `face_pick` | `H3 Face Track + Crop` → `face_pick` |
| `preview` | Optional: one card per shot, every face outlined and numbered. This is how you find out which number to use |
| `report` | Text summary: shots found, and which frame each locked onto |
| `frame_count` | Wire to the H3 node's `length` so it follows the video |
| `fps` | Optional: the effective frame rate after `select_every_nth` |

**The subject is chosen once per shot, not once per video.** A cut renumbers everyone — the biggest
face before it need not be the biggest after — so `confirmed_pick` takes one index per shot. That is
also how you say *"these two differently-numbered faces are the same person"*: the ranking cannot
know that, so you tell it.

> **To see the numbers cheaply**, mute everything downstream of the tracker and queue. Only the
> detection pass runs — seconds, no H3.

### Pick faces

`select = manual` is answered here. **Pick faces** scans the video once, then shows a frame from
every shot with each face outlined and numbered, and writes your answer — one index per shot —
into `confirmed_pick`.

<img src="screenshots/Pick%20Faces%20UI.png" alt="The Pick faces dialogue" width="640">

Under each frame is a chip per detected face, plus **not in this shot**. The green chip is the
current pick, and the numbers on the chips are the numbers drawn on the boxes.

Those numbers run **left to right across the frame**, which is how the picker always numbers,
whatever `select` says. They are not `select_index` positions — in `manual` that widget is hidden
and the node ignores it entirely, because your pick already names the face.

**not in this shot** records `-1` for that shot: the subject genuinely is not present, so those
frames keep their original pixels instead of the tracker settling on somebody else. It is the
hand-picked equivalent of `absent_shots` on the tracker.

<img src="screenshots/Pick%20Faces%20UI%20-%20Multiple%20Faces.png" alt="Choosing between four faces" width="640">

**Across a cut the numbering is not stable.** A cut renumbers everyone, so the person who was face
`1` before it may be face `3` after. That is the whole reason the dialogue asks per shot rather than
once: it is where you say *"these differently-numbered faces are the same person"*, which no
ranking rule can work out for itself. The row along the top keeps your picks side by side so you
can check them against each other.

<img src="screenshots/Pick%20Faces%20UI%20-%20Multiple%20Scenes.png" alt="One pick per shot across three shots" width="640">

| Control | Does |
|---|---|
| **Use these** | Writes the picks into `confirmed_pick` and closes |
| **Cancel** | Closes and changes nothing |
| **Clear** | Empties every pick, to start over |

**Neither button will run while a render is going.** Both refuse rather than compete with the
sampler for the GPU. The scan dialogue offers a *Scan anyway* escape if you want it regardless;
**Preview coordinates** has none and simply declines until the queue is clear.

Scanning is cached: reopening the dialogue reports *reused the previous scan* and is immediate. The
scan is redone when the video, the detector, the confidence, the frame range or the cut settings
change, because every one of those changes the shots the picks describe. That cache belongs to the
dialogue alone — when the graph runs, the node detects the clip again for the render itself. What is
saved is the *tracker* repeating it, not the node. For the same reason
`confirmed_pick` is dropped automatically when they change, rather than being left pointing at
faces that have been renumbered underneath it.

Once picked, the button on the node names the choice, so the graph stays readable without
reopening the dialogue:

<img src="screenshots/H3%20Load%20Video%20%2B%20Face%20Select%20-%20Multiple%20Scenes.png" alt="Picks named on the node" width="340">

### Preview coordinates

`select = closest_to_xy` is typed rather than clicked, so the only real question is where the
numbers landed. **Preview coordinates** renders `frame_index` with a crosshair at your `X`, `Y`,
every face outlined, and the nearest one highlighted:

<img src="screenshots/H3%20Load%20Video%20%2B%20Face%20Select%20-%20XY%20Preview.png" alt="The X, Y coordinate preview" width="340">

The caption states what it resolved to — `frame 0 · 2 face(s) · nearest is face 0` — so the
point can be confirmed against the person you meant before anything is rendered.

The point does **not** have to land on a face. The nearest face *centre* wins at any distance, so
putting it roughly where the subject stands is enough. This preview sits on the node rather than in
a dialogue deliberately: the picture belongs beside the fields being edited, and a second dialogue would
read as a second way of choosing a face by hand, which `manual` already is.


## H3 Face Track + Crop

<img src="screenshots/H3%20Face%20Track%20%2B%20Crop.png" alt="H3 Face Track + Crop" width="380">

Detects the face on every frame, fills gaps where detection fails, smooths the trajectory, and
emits a constant-size batch of crops plus the `transform` needed to paste results back.

**Inputs**

| Input | Default | What it does |
|---|---|---|
| `images` | - | The video to refine. Connect your video loader. |
| `detector` | first found | Face detection model from `models/ultralytics/`. `face_yolov8m.pt` is a good default. |
| `confidence` | `0.35` | The **detector's** score floor - shown as `detector_confidence`. Lower catches more profiles and small faces, at the cost of false positives. Nothing to do with identity matching; that is `identity_threshold`. |
| `crop_factor` | `2.5` | Crop side as a multiple of face **height**. 2.5 puts the face at ~40% of the crop. Bigger gives more context so the seam lands in hair and background, but less magnification. **2.0-3.0 is the useful range.** |
| `canvas_width` / `canvas_height` | `768` | Resolution H3 generates at. In `manual` these are used **exactly as typed**, high or low. Ignored in the auto modes, which size the canvas from the crop instead. 768 is H3's native short edge; 512 costs 2.25× less in latent tokens. |
| `canvas_mode` | `manual` | `manual` uses the two values above, exactly as typed. `auto_no_downscale` sizes from the largest crop so no frame is ever downscaled. `auto_capped_768` does the same but clamps to 768, a sane VRAM ceiling and the best default. **Both auto modes clamp up to a minimum of 512×512**, whatever the crop: the crop is bounded by the source frame, so without that floor a small face in a low-resolution clip would size the canvas down to the crop and hand H3 the same small face it renders badly. `manual` is never clamped. |
| `smooth_window` | `21` | Frames of smoothing on the crop **centre**. 21 at 24 fps is ~0.9 s. Raise if the box shivers, lower if it lags fast head movement. |
| `size_smooth_window` | `51` | Frames of smoothing on the crop **size**. Deliberately larger than the centre window, because size jitter makes the crop breathe, which changes the resample factor every frame and reads as shimmer. |
| `smooth_method` | `gaussian` | `gaussian` rejects jitter best. `savgol` preserves the shape of a push-in at large windows. `moving_average` is a plain boxcar and leaves residual jitter. |
| `size_mode` | `per_frame` | `per_frame` holds the face at a constant fraction of every crop, which is correct for push-ins. `max_of_clip` uses one size throughout, only useful when the shot is static. |
| `identity_reference` *(opt)* | - | A clear face image of the person to track. Picks the subject by identity rather than size. **Read only while `identity_track` is on** — with it off the reference is never consulted and `select` decides instead, which the report warns about. |
| `identity_track` *(opt)* | `True` | Hold one subject through a crowd. Continuity decides most frames; the identity embedding is consulted only when two candidates are similarly plausible or their boxes overlap. |
| `identity_threshold` *(opt)* | `0.28` | Minimum score to accept a face as the reference person. Below it, the frame falls back to continuity, which is what carries tracking through profiles and occlusion. The scale depends on `identity_model` — **set `0` to use whichever default the chosen model recommends.** |
| `select` *(opt)* | `largest_face` | Which face is the subject. `largest_face`, `smallest_face`, `left_most`, `right_most`, `top_most`, `bottom_most`, `centre_most`, `closest_to_xy`, `detector_score`. See [Choosing which face](#choosing-which-face). |
| `fallback_detector` *(opt)* | `none` | Used only on frames where the face detector finds nothing. A person/body model gives a real head position from the top of the body box, which beats interpolating blindly. With several people in frame it uses the body whose head lands nearest where the subject is expected. |
| `fallback_head_frac` *(opt)* | `0.5` | Head centre as a multiple of face height below the top of the person box. 0.5 suits a head seen from behind. |
| `select_index` *(opt)* | `0` | Which face in that ranking to track. `0` is the first, `1` the second, and so on. |
| `identity_model` *(opt)* | `insightface` | Which model decides two faces are the same person. `insightface` for photographed faces, `clip_vision` or `ccip` for anime and other non-photographic material. See [Anime and other non-photographic material](#anime-and-other-non-photographic-material). |
| `cut_detection` *(opt)* | `none` | `auto (pyscenedetect)` finds hard cuts and treats each shot as its own window for the smoothing, interpolation and composite fade, so the crop is not dragged across a cut. See [Videos with cuts in them](#videos-with-cuts-in-them). |
| `cut_threshold` *(opt)* | `3.0` | How far a frame has to stand out from its neighbours to count as a cut. Only used when `cut_detection` is `auto (pyscenedetect)`. |
| `absent_shots` *(opt)* | `off` | Find shots the subject is not in, so they are never rendered. `by_identity` treats a shot where no sampled face matches the identity anchor above `identity_threshold` as not containing the subject: those frames keep their original pixels and drop out of the batch. Needs identity matching to be working — it does nothing without an anchor. Sampling is not exhaustive, so a brief appearance can be missed; `report` names every shot it drops and the score it saw. |
| `identity_clip_vision` *(opt)* | - | A `CLIP_VISION` model from a `CLIPVisionLoader`, used only when `identity_model` is `clip_vision`. |
| `X` / `Y` *(opt)* | `0` | Shown only when `select` is `closest_to_xy`. The point to measure from, in pixels of the **source** frame, origin top-left. |
| `frame_index` *(opt)* | `0` | Shown only when `select` is `closest_to_xy`. The frame the `X`, `Y` measurement is taken on, counting from `0` over the frames handed to this node. |
| `face_pick` *(opt)* | - | From `H3 Load Video + Face Select`. Carries the detected boxes, the shot boundaries and the chosen subject, so this node does not detect the video a second time. `select`, `select_index`, `cut_detection`, `cut_threshold`, `X`, `Y` and `frame_index` are all greyed out while it is connected. `detector` greys too unless a crop-based `identity_model` still needs it; `confidence` greys unless either that or a `fallback_detector` does. |

**Outputs**

| Output | Goes to |
|---|---|
| `crops` | `H3 Inject Video Latent` → `images`, and `H3 Face Mask (SAM)` → `crops` |
| `transform` | `H3 Face Stitch Back`, `H3 Per-Frame Denoise`, `H3 Face Mask (SAM)` |
| `preview` | Optional: the video with the crop box drawn on every frame, colour-coded by how the subject was placed. See **Reading `preview`** below |
| `report` | Text summary: detections, gaps, frames where the subject was lost, magnification warnings |
| `canvas_w` / `canvas_h` | **Must** be wired to the H3 node's `width` / `height` |
| `frame_count` | The number of frames actually **rendered**, which is fewer than the clip when `absent_shots` drops a shot. Wire to the H3 node's `length` so it follows the video |

> **Reading `preview`.** Every frame carries the crop box, coloured by how the tracker placed it:
>
> | Colour | Meaning |
> |---|---|
> | green | the subject's face was detected on this frame and followed |
> | yellow | no face was detected; the head was placed from the body box found by `fallback_detector` |
> | red | no position for the subject on this frame - the face was not detected, or the subject was lost - so the box is interpolated from the frames either side |
>
> The composite fades out across yellow and red stretches, so those frames keep their original
> pixels. With `select` = `closest_to_xy` and no `face_pick`, an amber crosshair also marks `X`, `Y`.
> Saving `preview` as a video is the quickest way to check a track before spending a render on it.

> **Wire the canvas, don't type it.** In the `auto_*` modes this node chooses the size. If the H3
> node's `width`/`height` disagree, the latent shapes differ and injection refuses.

> Watch `report` for `magnification < 1.0x`. That means the crop is being *downscaled* into the
> canvas and real detail is being discarded. Raise the canvas, or skip videos that are close-up
> throughout since they have nothing to gain.

### Choosing which face

`select` names **which face**, and `select_index` takes the *n*th out of that ranking. They
apply only when no `identity_reference` is connected.

| `select` | picks |
|---|---|
| `largest_face` | the biggest face by **height** — the measurement the tracker uses everywhere else |
| `smallest_face` | the smallest by height, for refining the distant person rather than the near one |
| `left_most` / `right_most` | furthest left / right, by the **centre** of the face box |
| `top_most` / `bottom_most` | highest / lowest in frame, by the centre of the box |
| `centre_most` | nearest the centre of the frame |
| `closest_to_xy` | nearest the `X`, `Y` you give, measured on `frame_index` |
| `detector_score` | the detection the detector is most confident about |
| `identity_reference` | the face matching a reference image — `H3 Load Video + Face Select` only |
| `manual` | the face you chose in the picker — `H3 Load Video + Face Select` only |

Direction lives in the name, so there is no separate order widget: `left_most` and `right_most`
are two modes rather than one metric read forwards and backwards.

`closest_to_xy` takes three values of its own, shown only when it is selected. `X` and `Y` are
**pixels of the source frame** measured from the **top-left corner** — X increasing to the right,
Y increasing downward — so on a 960×544 clip (0, 0) is the top-left, (960, 544) the bottom-right,
and (480, 272) the middle. The point does not have to land on a face: the nearest face centre wins
at any distance. `frame_index` is the frame the measurement is taken on, counting from `0`, over
the frames the node loaded — so with `skip_first_frames` set it counts from the first frame kept.

`select_index` then picks out of that ranking: `0` is the first, `1` the second. To see which
number is who, put an **`H3 Load Video + Face Select`** in front of the tracker and connect its
`preview` to a `PreviewImage` — it emits **one card per shot**, every detected face outlined and
numbered, the chosen one highlighted in green.

The numbering on those cards follows **that node's own `select`**, and while it sits at its default
`manual` the cards are numbered left to right across the frame. So a number read off the preview
only lines up with the tracker's `select_index` when both nodes are set to the same ranking.

Three things worth knowing:

- **A rank is a per-frame property, not an identity.** The pick is made **once**, on the first frame
  that actually contains the requested index, and continuity carries the subject from there. Ranking
  every frame independently would hop between people the moment two of them cross or change size.
  The report prints the frame it locked on at.
- **Anything before that frame is interpolated.** If you ask for index 2 and the video opens on a
  single face, those opening frames have no index 2 to lock onto, so they are filled in from the
  lock-on frame and faded out of the composite, exactly like a detection dropout. The report counts
  them, and warns if there are more than a dozen.
- **The identity anchor follows your pick.** With no reference image the node normally assumes the
  biggest face is the subject — which used to override `centre_most` in a crowd. Now, once you
  select a different subject, the anchor is rebuilt from *your* pick, so identity matching holds
  that person rather than dragging the crop back onto the dominant face.

**Which mechanism decides.** Three things can answer "who is the subject", and they have a
fixed order:

1. **A reviewed pick** - `select` = `manual` on `H3 Load Video + Face Select`, carried on
   `face_pick`. You chose the face for each shot, so nothing overrides it.
2. **A connected `identity_reference`** - decides on its own, and `select` / `select_index` are
   ignored. The report says so.
3. **The ranking rule** - `select`, `select_index`.

Whichever wins only decides *where each shot starts*. From there the subject is held by
continuity - the nearest box to where they were, penalised for size change - and the identity
embedding is consulted only when two candidates are equally plausible or their boxes overlap.
On a typical video that is a couple of frames out of hundreds, and the report counts them.

**When the subject is lost.** In a shot with more than one face, continuity only carries the subject
onto a face within about three quarters of their own face height of where they just were, a little
more after a gap. When their face stops being detected and no face is within that reach, those
frames are left unresolved rather than handed to whoever is nearest: the crop is interpolated across
them, `preview` draws them red, the composite fades out so they keep their original pixels, and the
report lists them on its `lost:` line. The subject is picked up again when a face reappears within
reach of where they were lost. A subject who moves further than that while their face is hidden -
walking behind someone and out the other side - is not followed to the new position.

So a reference and a reviewed pick are not in conflict: the pick chooses the person, and the
reference still anchors the tie-breaking.

**If the automatic choice grabs the wrong person,** set `select` to `manual` and pick the face
yourself. That is what it is for. Lock-on deliberately accepts the best available match at any
score, so that small, poorly detailed faces - exactly the ones worth refining - are not
rejected for embedding weakly; the cost is that it can occasionally settle on the wrong face.

If the video never shows as many faces at once as you asked for, `select_index` is clamped to what
exists and the report says so.

### Videos with cuts in them

The crop trajectory is smoothed over the whole video. Across a hard cut the smoothing kernel spans
two shots and drags the box toward where the subject stood in the other one, for about half a
smoothing window either side. A dropout spanning the cut is worse: it is filled along a line that
existed in neither shot.

Set `cut_detection` to `auto (pyscenedetect)` and each shot becomes its own window for the smoothing, the gap
interpolation and the composite fade. Continuity also re-locks at each cut, since "nearest box to
where the subject was" means nothing once the camera has changed shot.

Three things worth knowing:

- **The canvas is still sized once for the whole video.** H3 generates one width/height for the
  batch, so a video mixing a wide shot with a close-up still sizes off the close-up. `max_of_clip`
  stays video-wide for the same reason.
- **A shot shorter than the smoothing window gets less smoothing**, not none — the window is clamped
  to the frames available. The crop can shiver on brief shots. The report names the shortest shot
  and warns when it is under the window.
- **A false cut costs smoothing on both sides of it.** The report says how many shots were found, so
  check it against what you know is in the video before trusting the result. Raise `cut_threshold` if
  it splits a continuous shot; lower it if it misses a cut between similar frames.

`H3 Per-Frame Denoise` splits its strength curve at the same boundaries, using the shot boundaries
the tracker publishes on `transform`.


### Anime and other non-photographic material

`buffalo_l` is ArcFace trained on photographed faces, and the node reaches it through InsightFace's
**own** detector (SCRFD, trained on WIDER FACE). On illustration that detector may struggle to match, so
there are no candidates to compare and identity matching quietly degrades to continuity. Nothing
goes visibly wrong — you simply never get the crowd handling.

Swapping only the recogniser would not fix that, because the candidates still have to come from
SCRFD. So the two other backends embed **the boxes your own detector already found**:

| `identity_model` | needs | good for |
|---|---|---|
| `insightface` *(default)* | nothing extra | photographed human faces. Unchanged. |
| `clip_vision` | a `CLIPVisionLoader` wired to `identity_clip_vision` | any domain — anime, 3D, stylised. No install. |
| `ccip` | `pip install dghs-imgutils` | anime specifically. Purpose-built for *"is this the same character?"* |

`ccip` is [CCIP](https://github.com/deepghs/imgutils), the illustration counterpart of ArcFace, and
it is the most accurate of the three on anime. It is **deliberately not a declared dependency** of
this pack: it pins `numpy<2` and pulls `opencv-contrib-python`, which shadows the OpenCV build
ComfyUI ships in exactly the way `onnxruntime-gpu` shadows `onnxruntime`. Install it yourself if you
want it; nothing else changes if you do not. Its ONNX models download on first use.

`clip_vision` needs no install, works on anything, and reuses a model you probably already have for
IPAdapter or Redux. It describes *appearance* rather than identity, so two characters with a similar
palette can collide, and its similarities sit high and close together.

**Also swap the detector.** These backends only see what `detector` gives them, so pair them with an
anime face model in `models/ultralytics/bbox/` — the usual choices are
[deepghs/anime_face_detection](https://huggingface.co/deepghs/anime_face_detection) or Anzhc's
face-seg models. A photographic `face_yolov8m.pt` may not find anime faces either.

**Setting the threshold.** `identity_threshold` means a different thing per backend: `0.28` suits
InsightFace cosine, `clip_vision` wants roughly `0.80`, and `ccip` scores `0.5` at its own published
operating point. **Set it to `0` to use the chosen model's own default.** The report prints the
scores it actually saw, and warns when the threshold cleared every candidate and so filtered
nothing — set it from that evidence rather than guessing.

**Multiple people:** run the pipeline once per subject, each with that person's `identity_reference`
and their own refs on the H3 node, then chain them, feeding run 1's stitched output in as run 2's
`base_images`. The composites accumulate. Since 1.2.0 that bookkeeping has its own nodes —
[H3 Multi-Face Plan](#h3-multi-face-plan) picks the subjects and
[H3 Pass Load](#h3-pass-load) supplies the previous pass's stitched frames as the next pass's
`base_images` — so the chaining is wired once in the graph instead of carried by hand. This recipe
stays correct for a graph that uses no multi-face nodes.

---

## H3 Inject Video Latent (img2img)

<img src="screenshots/H3%20Inject%20Video%20Latent%20(img2img).png" alt="H3 Inject Video Latent (img2img)" width="380">

Encodes real frames into the **video** stream of H3's joint audio-video latent, leaving the audio
stream intact. This is the missing img2img path: H3's stock nodes always build a zeros latent, because
references are conditioning re-injected each step, never a starting point. Without this there
is no video-to-video.

**Inputs**

| Input | What to connect |
|---|---|
| `av_latent` | The `LATENT` output of `MiniMaxH3ReferenceToVideo` |
| `images` | `crops` from **H3 Face Track + Crop** |
| `vae` | The **video** VAE |

**Outputs:** `av_latent` (onwards to `MiniMaxH3NativeAudioLock`) and `report`.

It has no widgets. Strength is set downstream by `BasicScheduler`'s `denoise`, **not** by
`SplitSigmas`. See [Denoise](#denoise) below.

---

## H3 Per-Frame Denoise

<img src="screenshots/H3%20Per-Frame%20Denoise.png" alt="H3 Per-Frame Denoise" width="380">

Varies denoise strength along the temporal axis via the latent's noise mask, so one sampling pass
covers a shot that goes from distant to close.

A single denoise cannot serve a whole video: a tiny face has no detail to preserve and wants a
strong pass so the model *synthesizes*, while a large face has real detail and wants a gentle pass
so it is not rewritten. This node scales the base denoise per frame by measured face size.

**Inputs**

| Input | Default | What it does |
|---|---|---|
| `model` | - | The H3 model on its way to the sampler, from `MiniMaxH3NativeAudioLock`. **Required.** Making a per-frame mask behave takes two changes to the MODEL rather than to the latent: this node's video mask is kept out of H3's per-token timesteps, and the frames that mask holds back are re-noised to the step the sampler is on instead of being handed back nearly clean. |
| `av_latent` | - | From `MiniMaxH3NativeAudioLock` |
| `transform` | - | From **H3 Face Track + Crop**. This is where face sizes come from |
| `denoise_multiplier_small_face` | `1.0` | **Multiplier** on the denoise set on `BasicScheduler`, applied where the face is smallest. Not a denoise value in itself. Below `1.0` no frame in the video receives the full denoise you set, which makes the whole video gentler. |
| `denoise_multiplier_large_face` | `0.35` | **Multiplier** on the same value, applied where the face is largest. Lower preserves the real detail those frames already have, since a big face needs less rebuilding than a distant one. |
| `scale_mode` | `absolute_px` | `absolute_px` keys off real face size in source pixels, which is safe across a batch, since a video that never has a small face just sits at the baseline. `relative_to_clip` normalizes to that video's own min/max, so its smallest face always gets the full boost. Use the latter when tuning one video to its extremes. |
| `face_px_small` | `30.0` | Face height (source px) at or below which full `denoise_multiplier_small_face` applies. |
| `face_px_large` | `120.0` | Face height (source px) at or above which `denoise_multiplier_large_face` applies. |
| `gamma` | `1.0` | Curve on the interpolation. `>1` keeps strength high until the face is genuinely large; `<1` drops it off early. |
| `smooth_frames` | `9` | Smooths the strength curve over time. An abrupt denoise change between neighbouring frames shows up as a texture pop, so be generous. |

**Granularity is one latent frame, not one pixel frame.** H3 packs 17 pixel frames into 5 latents, so the finest step the ramp can take is ~3.4 frames. Values are averaged within each latent frame.

**Outputs**

| Output | Goes to |
|---|---|
| `av_latent` | `SamplerCustomAdvanced` → `latent_image` |
| `report` | Text summary of the strength curve |
| `model` | **`BasicGuider` → `model`.** The patched model. Wiring the unpatched model straight from `MiniMaxH3NativeAudioLock` to the guider costs you both changes above, without any error: the mask still applies, but the held frames come back nearly clean rather than at the sampler's current sigma. |

Base denoise and these multipliers are tuned **together**. The example workflows ship a base of
`0.40` which this node scales down on large-face frames. If you bypass this node, drop the base a
long way or every large face gets rewritten.

---

### Denoise

Denoise values do **not** transfer from SDXL-family models. H3 is flow matching with a large sigma
shift:

```
sigma = shift * t / (1 + (shift - 1) * t)
```

At the default shift of 12, `0.25`, an ordinary FaceDetailer value, lands at an effective sigma
of **0.800** and rewrites the frame.

| denoise | effective sigma (shift 12) |
|---|---|
| 0.02 | 0.197 |
| 0.05 | 0.387 |
| 0.15 | ~0.68 |
| 0.25 | 0.800 |

`steps` and `denoise` are **independent**: `BasicScheduler` builds a `steps/denoise`-long
full-range schedule and keeps the lowest `steps+1` sigmas, so 4 steps with a turbo LoRA is both
fast and gentle. Do **not** use `SplitSigmas`. On a 4-step schedule at shift 12 even the last
split point is already sigma 0.800.

Push the ceiling too high and the head drifts relative to the body, a content problem no mask can
hide.

---

## H3 Face Mask (SAM)

<img src="screenshots/H3%20Face%20Mask%20(SAM).png" alt="H3 Face Mask (SAM)" width="380">

Produces true face-shaped paste masks instead of a rectangle, temporally smoothed. Optional, and
requires Impact Pack for `SAMLoader`.

**Inputs**

| Input | Default | What it does |
|---|---|---|
| `crops` | - | **The tracker's `crops`, not the decoded result.** See the warning below. |
| `sam_model` | - | From Impact Pack's `SAMLoader` |
| `transform` | - | From **H3 Face Track + Crop** |
| `threshold` | `0.93` | SAM confidence threshold for accepting mask pixels. |
| `dilation` | `0` | Grow the mask. SAM masks are accurate, so they rarely need it. |
| `temporal_smooth` | `5` | Frames of averaging across the mask stack. `1` disables it, and the mask edge will shimmer. |

**Outputs:** `masks` (to `H3 Face Stitch Back` → `masks`) and `report`.

> **Mask the input, never the output.** Wire the tracker's crops here. This matches FaceDetailer,
> which computes its mask from the source image; generation never feeds back into the mask. Mask
> the *generated* result instead and, if the model nudges the face inward, the mask traces the new
> smaller silhouette while the original face pokes out past it, most visibly the nose on profile
> shots. It is also cheaper: no dependency on the sampler, so SAM need not be resident alongside
> the video model.

**Is it worth it?** Often not. A rect mask frequently beats SAM here. SAM traces the face tightly,
so any drift in the refined face lands right on the silhouette, whereas a slightly looser rect puts
the seam in hair and background where it reads far less. Try the rect workflow first.

> If you use SAM masks, drop `feather` on the stitch node to **4-8**. A rect needs far more.

---

## H3 Face Stitch Back

<img src="screenshots/H3%20Face%20Stitch%20Back.png" alt="H3 Face Stitch Back" width="380">

Warps each refined crop back onto the exact float box it came from, colour-matches it, feathers the
edge and composites. A single batched `grid_sample` does the warp, so a trajectory smoothed to
sub-pixel precision is not re-quantized on the way home.

**Inputs**

| Input | Default | What it does |
|---|---|---|
| `base_images` | - | The **original** frames, the same video fed to the tracker |
| `refined_crops` | - | `VAEDecode` output |
| `transform` | - | From **H3 Face Track + Crop** |
| `paste_region` | `face_only` | What actually composites. `face_only` / `face_ellipse` paste just the detected face box; `full_crop` pastes hair, shoulders and background too and risks a visible rectangle. |
| `mask_dilation` | `16` | Grow the face box before blurring, in canvas px, so the blur has room and the blend does not eat into the face. |
| `feather` | `6` | Gaussian blur radius on the paste mask, in **source pixels**, measured against the final frame, so the blend is the same physical width at any magnification. Use ~24 with a rect mask, 4-8 with SAM. |
| `colour_match` | `1.0` | Match the crop's per-channel mean/std to the region it replaces. The crop and the frame went through independent passes, so without this the face can come back subtly brighter and read as pasted on. |
| `blend` | `1.0` | Global opacity of the refined face. Below 1.0 mixes back toward the original, useful to dial back over-sharpening. |
| `undetected_frames` | `fade_out` | What to do where no face was found. **All** frames still go through H3 either way, which is what keeps it temporally consistent. This only controls pasting. `fade_out` ramps the composite to zero across the gap; `skip` hard-cuts to original pixels; `composite_anyway` risks H3 hallucinating a face onto the back of a head. |
| `feather_scales_with_crop` *(opt)* | `False` | Legacy: treat `feather` as canvas pixels so the blend narrows as the crop shrinks. Leave off. |
| `masks` *(opt)* | - | Per-frame masks from **H3 Face Mask (SAM)**. Overrides `paste_region`. |

**Outputs:** `images`, the finished frames. Send to your video save node.

> **Only the face region composites.** The wide crop exists to give the sampler *context*. It is
> not what gets pasted. Pasting the whole crop covers roughly 88% of the canvas versus 16% for the
> face box, and any change the model made to hair or background returns as a rectangle.

---

## H3 Face Transform Info

<img src="screenshots/H3%20Face%20Transform%20Info.png" alt="H3 Face Transform Info" width="380">

A debug node. Prints the per-frame boxes so you can sanity-check tracking before spending time on a
sampling pass.

| Input | Default | What it does |
|---|---|---|
| `transform` | - | From **H3 Face Track + Crop** |
| `max_rows` | `12` | How many frames to print. |

**Outputs:** `info` (a string). It is an output node, so it displays in the graph directly.

Use it when the stitch looks misaligned, or to confirm gap-filling behaved on a video where the
subject turns away.

---

## H3 Multi-Face Plan

Decides who the **subjects** of a job are, and decides it once. It detects the **target clip** on
one frame — the **match frame** — and the **reference still**, pairs the two sets of faces
positionally, cuts one **subject reference** per subject out of the still, and writes the
**subject plan**, the crops and a numbered preview of both plates into the **job folder**. Every
node downstream reads that folder instead of detecting anything again, so this is the only node in
the multi-face chain that ever needs the reference still wired in.

It is an **output node**: the `report` and the `preview` are meant to be read off it *before*
anything is queued. That is the whole point of it — a wrong pairing costs one H3 render per
subject, so it has to be visible at the node that made it.

**Inputs**

| Input | Default | What it does |
|---|---|---|
| `images` | - | The **target clip**, as an `IMAGE` batch — from any Load Video / VideoHelperSuite / Load-Images node. The node detects it (the match frame for the plan, every frame for the `face_pick` output), so no separate face-picker upstream is needed. This node does not decode video itself. |
| `reference_still` | - | One high-quality image of the same scene, framed the same way as the clip. Its faces supply the detail. Only the first image of a batch is used, and if the batch holds more than one the report says which was paired. |
| `detector` | `face_yolov8m.pt` | The detector used on **both** plates, so the two sets of boxes mean the same thing. Same list as the other nodes here. |
| `confidence` | `0.35` | The detector's score floor, applied to both plates. Lower finds small and turned faces on both. |
| `match_frame` | `0` | The frame of the target clip whose face positions line up with the still. Pairing happens only here, so pick a frame every subject is clearly in; it need not be frame 0. A value outside the clip is clamped and the report says so. |
| `subject_source` | `union` | Whose faces can become subjects. `union` takes every face found on either plate and flags whether it has a subject reference, so nothing detected is dropped silently; `target_clip` takes only the clip's faces, when the clip detected better; `reference_still` takes only the still's, when the soft clip missed faces the still can see. |
| `match_tolerance` | `0.05` | How far apart the two detections of one face may sit and still pair, as a **fraction of the clip's diagonal** so it means the same at any resolution (0.05 of a 1000×1000 clip is 71 px). The boxes must also overlap. Raise it if the still is framed tighter than the clip; lower it in a crowd. |
| `excluded_faces` | empty | Face numbers to switch off, comma-separated, exactly as numbered on the `preview`: `0, 3`. A bare number is a **target clip** face; `s2` is a **reference still** face — the only way to switch off a bystander the photo saw and the clip never did. An **excluded face** never becomes a subject and keeps its original pixels, and the face paired to it falls to `unmatched_faces`. A text widget, so the answer survives a reload with no front end. |
| `max_subjects` | `0` | `0` = no cap. A cap is a **loud** truncation: it applies after the faces are ordered and after `unmatched_faces` has had its say, every dropped face is named in the report with its position at the match frame, and the number the cap saves is printed beside the number it spends. It also joins the **pairing fingerprint**, because a cap changes who the subjects are. |
| `job_name` | empty | The **job folder** under ComfyUI's output folder. Left blank it is named after the clip. **Keep the same name to resume a job** — change it, or change an input the fingerprint tracks, and the pass nodes **refuse** rather than mixing two pairings into one clip, while **H3 Multi-Face Plan** *reuses* the saved plan and warns that the change was not applied (press **Process Plan** to apply it). |
| `unmatched_faces` | `leave` | What becomes of an **unpaired face** — a face on one plate with no partner on the other. It decides the **subject set**, so it lives here rather than on the pass nodes, and it is recorded in `plan.json`. See [`unmatched_faces`](#unmatched_faces-three-answers). |
| `crop_factor` | `2.5` | The crop factor **H3 Face Track + Crop** will use for this job. Set it to that node's `crop_factor`, and to nothing else: each **subject reference** is cut with the rectangle that factor gives — crop_factor × face height, centred on the face, mapped into the **reference still** at its native resolution. A number that differs from the tracker's mis-frames every reference in the job, so it is recorded in `plan.json` (the overlap warnings, the pairing preview and a resumed job all read it back from there) and it joins the **pairing fingerprint**. |
| `pair_overrides` | empty | Manual pairings, for when the automatic pairing matched the **wrong** two faces. Set by clicking a face image in the pairing panel, or by hand: comma-separated `face=partner`, where a face is a bare **target clip** number or an `s`-prefixed **reference still** number, and the partner is another face on the other plate or `none` — e.g. `0=s2` (clip face 0's reference is still face 2), `s1=0` the same fact from the other side, `2=none` to switch a face's partner off. A manual pairing over-rides the positional guess; editing one in the panel is **pending** (it writes this widget and marks the node dirty) until you press the panel's **Apply** button, which re-runs the plan node — renumbering the subject set and clearing the job's saved passes, so it confirms first. You can also edit this widget by hand, then press **Process Plan** to commit it. Recorded in `plan.json` and part of the **pairing fingerprint**. |

**Outputs**

| Output | Goes to |
|---|---|
| `plan` | the `H3FACEPLAN` output feeds **H3 Subject Select**, **H3 Pass Save**, **H3 Pass Load** and **H3 Multi-Face Composite**. It carries the job folder's name, which is why none of those nodes has a `job_name` widget of its own — and why two jobs in one output tree cannot read each other's references. |
| `preview` | both plates side by side at the match frame, every face numbered, subjects outlined in their palette colour, excluded faces slashed red, faces with no partner marked. Wire it to any image viewer, or read it off the node. |
| `report` | printed on the node: who is in the job, what it costs, what was written where, every decision the plan made about faces it could not pair, and a **passes** line for a job that has started — how many subjects (queues) are still to run and which subject the next queue takes. |
| `subject_count` | how many passes this job is — the number of queues a fresh job takes (found faces minus the ignored ones). Wire it anywhere you want the number, and check it against the cost line before queueing. |
| `face_pick` | the whole clip as an `H3FACEPICK` — every frame's face boxes and confidences, the whole sequence treated as one shot segment (this node does no cut detection). Because the plan node runs the detector anyway, it can carry the per-frame boxes onward, so **`H3 Subject Select` and `H3 Face Track + Crop` can take their `face_pick` from here instead of from `H3 Load Video + Face Select`** — drive the whole job from just `images` + a `Load Image` reference, with no picker node. Single-shot clips are the natural fit; a clip with cuts is still tracked across them, but the whole sequence is presented as one segment. |

**Plugging in directly (no `H3 Load Video + Face Select`):** wire your image sequence (a Load Video / VideoHelperSuite / Load-Images batch) into `images` and the reference frame into `reference_still`, set `match_frame` (default `0`), and wire this node's `face_pick` into `H3 Subject Select` (and the tracker). This is the same `H3FACEPICK` shape the picker emits, so every released pass node runs unchanged.

**Buttons on the node:** **Process Plan** wipes the whole job and re-plans it from scratch — it clears the plan (kept once as `plan_prev.json`), the preview and crops, `manifest.json`, the subject walk, and the **entire `passes/` tree**, so every subject is re-rendered against the new pairing. This is **destructive** (it deletes completed renders) and the panel asks you to confirm first. It is the right move after changing the pairing or the clip, since passes saved under a different subject set would otherwise break the chain. **Pairing preview** opens the read-only panel over the job folder. **Reset run** clears only the **pass bookkeeping** (the manifest's pass records and the subject walk) so the always-next walk starts over at the first subject — it pops a note saying how many completed passes it cleared, and never touches `plan.json`, the references, the preview or the saved pass frames (a re-run simply overwrites its own frames).

---

### What the report says, and the job folder it writes

A two-subject job — three faces found on the clip, two on the still, `max_subjects` at `2`,
`unmatched_faces` on `with_still` — prints this. Paths abridged, every sentence the node's own:

```
job             shot 07
target clip     1000x1000   4 frame(s)   match frame 2
reference still 2000x2000   still -> clip scale 0.500   (1 clip px = 2.000 still px)
cost (estimate) 2 subject(s) x 4 frame(s) @ 768x768   = 8 frame(s) to generate
                (768x768 is H3 Face Track + Crop's crop canvas at its shipped default - what this job generates at is that node's canvas_width/canvas_height, or its auto canvas)
crop factor     2.5   each subject reference was cut at crop_factor 2.5 x face height - it must match the crop_factor H3 Face Track + Crop uses for this job
pairing         union   tolerance 0.050 = 71px of the 1000x1000 diagonal   2 of 3 clip face(s) paired
excluded        none
max_subjects    2
capped          subject cap 2: 1 face(s) capped - clip face 2 (at 850, 180 on frame 2). Raise max_subjects to include them, or switch a face off with excluded_faces to spend those generations elsewhere on purpose.
                2 subject(s) of 3 to generate; uncapped cost would be 3 subject(s) x 4 frame(s)
unmatched       with_still   1 paired   0 subject(s) with no reference   1 subject(s) on the whole still   0 face(s) left out
subject 0       clip face 0   reference from still face 0, 600x600px at the still's native resolution -> subject_0.png
subject 1       clip face 1   reference is the WHOLE reference still, 2000x2000px -> subject_1.png   whole-frame conditioning: H3 picks which face in it to apply
written         <output>/shot 07
                plan.json + 2 subject reference file(s) + pairing_preview.png
notes
                - clip face 1 (at 450, 450 on frame 2) is subject 1 with the whole reference still as its subject reference (2000x2000px): unmatched_faces=with_still, so the conditioning is whole-frame and H3 chooses which face in the still it applies to - not a guarantee the right face is used.
                - plan: <output>/shot 07/plan.json
```

`cost (estimate)`, `crop factor`, `pairing` and the one line per **subject** are what to read before
queueing: how many
renders this job is, whether the pairing actually found the people you can see, and who each render
will be about. The `unmatched` line counts what the plan decided about faces that found no partner;
the `capped` line appears only when `max_subjects` cut someone.

The folder itself:

```
<output>/<job_name>/
    plan.json                      the subject plan, and the pairing fingerprint every later
                                   node checks itself against before it touches anything
    manifest.json                  the pass manifest: which subject has which pass stage on
                                   disk, and which passes are therefore complete
    pairing_preview.png            both plates at the match frame, every face numbered
    subject_0.png  subject_1.png   one subject reference each - a crop of the still, or, for a
                                   whole-still subject, all of it
    clip_0.png     clip_1.png      each subject's face cut from the match-frame clip, beside its
                                   reference - the panel shows the two together (a subject with no
                                   clip face on the match frame has no clip_N.png)
    passes/
        subject_0000/
            crops/frame_0000.png …     the crops that pass was fed, in frame order
            crops.json                 where each of those crops sat in its frame
            stitched/frame_0000.png …  that pass composited back onto the target clip
```

Subject folders are numbered `subject_NNNN` — four digits — while the reference PNGs are not, so a
job with more than ten subjects still sorts correctly on disk. `crops.json` exists **once per
subject**, never once per stage: a stitched pass must not rewrite the positions its crops were cut
with, and H3 Pass Save says so out loud if you wire `transform` into it anyway.

The crops are checkpointed for two reasons. A job that dies mid-subject keeps that work instead of
paying for another detection pass and another track; and the PNGs are ordinary files, so you can
open `passes/subject_0000/crops/` and see exactly what H3 was fed — the fastest way to explain a
result you did not want.

### `unmatched_faces`: four answers

A face found on one plate at the match frame with no partner on the other is an **unpaired face**.
What it becomes is decided here, once, because it changes the **subject set** — and the subject set
is what the composite is gated on and what the fingerprint compares. The four modes, in the
report's own words:

- **`leave`** (the default): the face is **not a subject at all**. No pass ever runs for it, so it
  keeps the pixels the clip arrived with. The report names every face it left out and ends the
  sentence with `That is not a failure: it is the safe default when the pairing might be wrong.`
  Leaving is the safe default because an unpaired face is often a detection artefact, or someone who
  is in the clip and not in the still — refining them from a stranger's face is worse than leaving
  them alone. `leave` also **renumbers the survivors contiguously**: switching off the middle face
  turns `subject_2.png` into `subject_1.png`, because pass folders, manifest keys, `next` and the
  composite order all count on there being no gaps in the numbering.
- **`no_reference`**: the face **is** a subject (`ref_kind: none`), with no reference file written.
  Its pass runs with nothing wired into H3's `ref_image` slots, which the slots allow — they
  autogrow from zero — and the report says what that costs: `the refinement is driven by identity
  and quality only and will be weaker than a paired subject`.
- **`with_still`**: the face **is** a subject whose **subject reference** is the **whole still**,
  written out as `subject_N.png` like any other. The conditioning is whole-frame, so *H3 chooses
  which face in the still it applies* — a face in the still that is not this subject can be the one
  whose detail gets transferred. That is a warning about how H3 works, not a bug to tune out.
- **`clip_region`**: the face **is** a subject whose **subject reference** is cut from the still at
  the **region where this clip face sits** — the same `crop` rectangle a paired subject would get,
  the clip face's tracker box mapped into the still at its native resolution, also written out as
  `subject_N.png`. It reads `ref_kind: crop` (it is not a detected still face). This is the right
  region when the still shares the clip's framing — e.g. the reference is a frame grabbed from the
  same shot, or the person sits at a similar relative spot. When the still is a **different
  composition** the clip-aligned rectangle lands on the wrong area, so prefer `with_still` there.
  Use it to refine a face once when a low-confidence detection would otherwise leave it unpaired.

A face that did find its partner is unaffected: `crop` references, pairing and passes are identical
in every mode. Read `ref_kind` (`crop | still | none`) rather than whether a reference crop
exists — a whole-still subject's crop is the full frame, and a `clip_region` subject's is the
clip face's own rectangle.

### The refusal sites: why two pairings never share a clip

The **pairing fingerprint** is the set of inputs and measurements that decide who is who: both plate
sizes, the match frame, `subject_source`, `match_tolerance`, the exclusions, the manual
`pair_overrides`, `unmatched_faces`,
`max_subjects`, the `crop_factor` the **subject references** were cut at, the boxes both plates
measured at the match frame and their scale, the `detector` and `confidence` that produced them, and
the subject set itself. It is compared at three sites — **H3 Multi-Face Plan** before it
overwrites a folder, and **H3 Pass Save**, **H3 Pass Load** and **H3 Multi-Face Composite** before
they read or write a pass. They exist because a job folder holding two pairings' passes does not
fail: it returns a clip with faces quietly swapped that looks finished.

The measured boxes are in the fingerprint because the subject set is only the *outcome* of a
pairing. A different still of the same size, or another detector, can pair different faces into the
same subject indices — and comparing indices alone would call that a resume.

The refusal names every input that disagrees, one line each, then what to do:

```
<output>/shot 07/plan.json was written from different pairing inputs, so this would mix two
pairings into one job folder:
  - match_frame: the job folder has 5, this plan has 2
Set job_name on H3 Multi-Face Plan to a new name for this pairing, or re-run that node
deliberately with the SAME inputs to resume this job. If the inputs really are meant to change,
rename or delete the job folder first: every pass already inside it was cut from the inputs above.
```

A re-run into an existing job is now the ordinary case, not an error: **H3 Multi-Face Plan only
decides a pairing when there is no saved plan yet** — a first run, or after the pairing panel's
**Process Plan** clears the folder. Run the flow again and the plan node **reuses** the saved
`plan.json`: it neither re-cuts nor overwrites, so re-running is never destructive and cannot mix
two pairings. If the node's widgets differ from the saved plan, that change is *not* applied to the
reused run — applying it would orphan the passes already in the folder — so the report says so and
names what differs, and the pairing panel shows the same note, pointing at **Process Plan**. A
subject-set change is named as a set difference in that warning:
`subject set: the saved plan has N subject(s), these settings would make M`.

The **pass nodes** keep a hard gate the plan node does not: they refuse a plan whose fingerprint
differs from the `plan.json` already in the folder, and refuse a manifest recording a subject this
plan does not have — which is what catches a folder left over from a plan you have since changed.
The example above (`pairing_preview`'s error box) is that pass-node refusal.

### The pairing preview panel

The plan node carries a panel — **Pairing preview** — showing `pairing_preview.png` and, per
subject, **both** faces side by side: the clip face cut from the match frame (`clip_N.png`) and the
reference it is conditioned on (`subject_N.png`). A still-only subject has no clip face on the match
frame, so it shows a placeholder in its place. A pairing can be checked beside the node that made
it instead of by opening the folder in a file browser.

The **Pairing preview** view reads `plan.json`, `pairing_preview.png`, `subject_N.png` and
`clip_N.png` out of the job folder and decides nothing of its own, so it cannot disagree with the
graph. Each face row carries an **Ignore this face** / **Include this face again** toggle, and each
switched-off face carries **Restore**; there is also a click-to-re-pair editor (**click the
clip-face image** to pick which target-clip face this subject tracks, **click the reference image**
to pick which reference-still face it is conditioned on — a grid of every face on that plate plus a
**no partner** choice). Each of these edits is **pending**: it writes the node's `excluded_faces` /
`pair_overrides` widget and marks the node dirty, but re-plans nothing on its own — the panel keeps
showing the last plan until you commit. An **Apply** button in the panel's foot commits the batch: it
re-runs the plan node (the same path as the node's **Process Plan** button), which wipes the job and
re-plans from scratch with your changes in it. The foot names how many changes are pending, so you
can toggle several faces and re-pair a few, then apply them all at once.

The panel still owns no pairing of its own: the plan node re-decides the subject set, so it never
becomes a second source of truth, and the passes are always cut against the node's decision. The
re-plan is cheap (one detection + pairing pass, **no H3 render**) but it renumbers the subject set,
so — exactly like **Process Plan** — it **clears every saved pass** (crops + stitched) and **asks you
to confirm first**, naming how many completed renders that deletes. Pending edits stay only on the
node's widgets, so simply **closing the panel without Apply discards them** (the saved plan is
untouched). To undo a change you did apply, toggle the face back or **Restore** it and Apply again; the
reopened panel keeps every switched-off face in its switched-off list with a **Restore** button.

The node's own **Process Plan** button is the same wipe-and-re-plan, for when nothing in the panel is
pending but an *input* changed (a new clip, a different `crop_factor`, `match_tolerance`,
`max_subjects`, …): it clears the whole job — the plan (kept once as `plan_prev.json`), the preview
and crops, `manifest.json`, the subject walk, and the entire `passes/` tree — then re-plans from
scratch. It is the one way to commit a changed input to the job you are about to queue, and it is
destructive, so it confirms too.

It **refuses to open or refresh while a render is running**, with no override, and the front end
and the route enforce that independently:

```
A render is running, so the pairing panel will not open or refresh. The job folder it reads is the
one that render is writing into, so what it shows could be half-written. Try again once the queue
is clear.
```

The reason is not the cost of reading a JSON file: it is that a panel held open over a job folder
that a render is writing shows a pairing that the render in front of it may be about to invalidate.

## H3 Subject Select

Takes **one subject** out of the subject plan and hands the released tracker a pick payload for
that person alone, in exactly the shape `H3 Face Track + Crop` already accepts (ADR 0005). Nothing
is detected here: the plan says who the subject is and what their **subject reference** is, the
clip's own pick payload carries a box for every frame, and the node narrows the second to the first.
Detection therefore stays exactly once in your graph — the loader did it, this node reads it, and
the tracker that follows runs no detection pass at all.

One branch of the graph per **pass** — but you build only ONE chain. **H3 Subject Select** always
takes the **next subject with no saved pass**, so you queue the same graph once per person and it
walks the plan for you: there is no `subject` value or `mode` to set. The walk *is* the job folder's
**pass manifest** — it moves on only once a subject's pass is saved, and a queue that crashes
mid-pass is taken again rather than skipped, so the chain never composites onto a subject that was
never run. When every subject is done the node refuses and points at **Reset run** on the **H3
Multi-Face Plan** node, which clears the pass bookkeeping (never the pairing or the saved frames) and
starts the whole job over from the first subject.

**Inputs**

| Input | Default | What it does |
|---|---|---|
| `plan` | - | The subject plan from **H3 Multi-Face Plan**: who the subjects are, which face in the clip each one is, which face in the still became their subject reference, and whether they were paired at all. It records the match frame's boxes only, so it cannot name this subject in the other shots on its own — that is what `face_pick` is for. |
| `face_pick` | - | The clip's own pick payload: every frame's boxes, confidences and shot boundaries. Wire it from **H3 Multi-Face Plan's `face_pick` output** (the whole sequence as one segment — no picker node needed) or from **H3 Load Video + Face Select** (which cuts shots). Its picks are replaced with this subject's; the boxes pass through untouched. **Do not resize, crop or trim between the source and this node** — every box would land in the wrong place. |

**Outputs**

| Output | Goes to |
|---|---|
| `face_pick` | **H3 Face Track + Crop → `face_pick`**, and on to **H3 Pass Save** if you checkpoint the crops. |
| `subject_reference` | H3's `ref_image_*` slots, and **nowhere else**. For a whole-still subject it is the entire reference still. A too-small crop is lifted to H3's minimum reference size (128px): H3 VAE-encodes a reference at /16 and a DiT patch conv needs a 7-wide latent, so a smaller one crashes the node; new plans cut real pixels up to that floor, and a crop already on disk is enlarged to it (the report says so). |
| `reference_exists` | `1` / `0`. `0` means this subject has no subject reference — a normal state of a plan under `unmatched_faces=no_reference`, not an error, and it means nothing is wired into the H3 node's reference slots. |
| `subject_index` | **H3 Pass Save → `subject`** — the subject this pass is refining, so the saved pass and the selection can never disagree. |
| `previous_subject` | **H3 Pass Load → `subject`** *and* **H3 Pass Save → `chained_from`**. The subject this pass composites ONTO: `-1` for the FIRST subject in plan order (so Pass Load returns its `no_previous` target clip), otherwise the previous subject. From the plan, not a per-copy widget — this is what lets one reusable chain serve any number of subjects. |
| `report` | who was chosen, which frame each shot locks onto, where the reference came from, and this subject's **shot account**. |

> **Leave `identity_reference` unwired.** Each subject's **identity anchor** is rebuilt from their
> own matched crops in the **target clip**, which the released tracker does by itself once a
> `face_pick` names the subject per shot. Wiring `subject_reference` into `identity_reference`
> anchors the track in a different image domain — the still against the clip — which is how a
> subject gets lost mid-shot. The subject reference is H3's picture input, not an identity input
> (ADR 0003).

### The shot account

Every pass reports itself **per shot**, and the two kinds of empty shot are worded differently on
purpose, because only one of them is worth fixing:

```
shots           subject 0 is not in shot 2 (frames 4-7): the plan put an absent entry on this shot - no frame of it shows this subject, so there is nobody here for the pass to refine - original pixels are kept there, which is the intended result for an absence - if this person IS in this shot, plan on a match frame they share with the reference still, or name them with Pick faces
shots           subject 0 was lost in shot 3 (frames 8-11): 4 of 4 frame(s) produced nobody - faces were detected in this shot but none of them matched this subject's identity; check identity_track and identity_threshold, or lower confidence, on H3 Face Track + Crop
```

A **planned absence** is the plan's decision and the clip's own pixels stay: expected, not a failure.
A **lost track** is the tracker arriving with the subject selected and producing nobody: check
`identity_track`, `identity_threshold` or `confidence` on the tracker. Both name the shot and its
**clip frame numbers**, so you can scrub straight to it. Neither asks for a new widget: the crop
positions come from the `transform` already wired into H3 Pass Save, or are read back out of the
pass's `crops.json` — and a pass nobody has run yet says so instead of claiming a clean bill of
health.

## H3 Pass Save

Writes one **pass stage** into the **job folder**: the frames as an ordinary numbered PNG sequence,
the crop positions as readable JSON, and the **pass manifest** entry saying how far this subject
got. It sits **inline** in the chain — `images` comes straight back out — so checkpointing a pass
costs nothing to wire.

**Only the `stitched` stage completes a subject.** A `crops` entry means "this far got done": a job
that died after the crops keeps that work and resumes on the same subject instead of declaring it
finished. The asymmetry is deliberate — crops are cheap to keep and worthless as a result.

**Inputs**

| Input | Default | What it does |
|---|---|---|
| `plan` | - | The subject plan from **H3 Multi-Face Plan**. It carries the job folder and the subject set, so this node needs no `job_name` of its own — and the plan is checked against the `plan.json` already in that folder before a single frame is written. |
| `subject` | `0` | Whose pass this is. Wire **H3 Subject Select → `subject_index`** in and the two can never disagree. A number the plan does not have is refused, rather than checkpointing a subject nobody planned. |
| `stage` | `crops` | Which half of the pass is being checkpointed. `crops` writes the crops H3 was fed plus their positions, and does **not** finish a subject. `stitched` writes the frames composited back onto the clip; recording it is what completes the pass and lets the job resume past them. |
| `images` | - | The crop batch for `stage=crops`, the stitched frames for `stage=stitched`. Written in frame order as `frame_0000.png` onward through the same PNG writer as the subject references, so you can open them and see exactly what H3 was given. The names never change: re-running a pass overwrites its own frames instead of leaving a second copy nobody can line up with frame 7. The batch has to be float 0-1 — a `uint8` batch is refused with a convert-it message rather than written back as noise. |
| `transform` *(opt)* | - | The crop positions from **H3 Face Track + Crop**. **Required for `stage=crops`** — without them the crops cannot be put back where they were cut. For `stage=stitched` there is nothing new to record, so leave it unwired: a subject's positions belong to its crops stage, and rewriting them from a stitched pass would put those frames back somewhere other than where they were cut. The report says which of the two happened. |
| `chained_from` *(opt)* | `-1` | **Which plate this pass was composited onto.** `-1` is the target clip; any other number is the subject whose stitched result this pass was composited onto. The first pass in the plan's order leaves it at `-1`; every later pass names the subject before it — subject 1 says `0`, subject 2 says `1`. [H3 Multi-Face Composite](#h3-multi-face-composite) returns the last subject's stitched frames on the claim that every earlier pass is already carried inside them, and that is only true if the passes were chained this way — which no node can see from the graph, so it is declared here, recorded in the pass manifest, and checked there. Wire every pass onto the target clip and each pass throws the previous subject's work away: the composite refuses rather than returning a clip with one refined face in it and calling it the chain. |

**Outputs:** `images` (the batch, unchanged, so the node can sit mid-chain), `report`,
`subject_index`, `pass_complete` (`1` once `stitched` is recorded). It is an output node, so the
report and the flag can be read off it at the end of a pass without a viewer attached.

The middle of a pass, in the manifest's own words:

```
subject 0   stage crops   4 frame(s) at 768x768
images: <output>/shot 07/passes/subject_0000/crops
          frame_0000.png ... frame_0003.png   ordinary PNGs: open them to see what this pass was actually fed
crop positions: <output>/shot 07/passes/subject_0000/crops.json
pass NOT complete yet: a subject only counts as done once its stitched stage is recorded, so a half-finished pass is resumed, not skipped
manifest: <output>/shot 07/manifest.json
chained onto    the target clip
shots           subject 0 was lost in shot 1 (frames 0-3): 1 of 4 frame(s) (frames 3-3) produced nobody - the tracker reached this shot with this subject selected and produced nobody for those frames - their identity fell below the threshold, or the face was never detected; check identity_track and identity_threshold, or lower confidence, on H3 Face Track + Crop
next subject still up: 0   H3 Subject Select's next takes it, and this subject again until its stitched stage is recorded
```

The **shot account** repeats here — measured against what this pass actually produced, from the
`transform` wired in for `stage=crops` or read back from `crops.json` for `stage=stitched`, which is
why the two reports can never disagree about the same subject.

The complete flag is **derived** inside the manifest writer from the stages actually on disk; no
caller can set it, which is why a job cannot be talked into believing it is finished.

## H3 Pass Load

Reads one subject's **pass stage** back out of the job folder. Nothing here re-detects, re-tracks or
re-generates — which is the point: a resumed job pays for neither the detector, nor the tracker, nor
a finished render.

**Inputs**

| Input | Default | What it does |
|---|---|---|
| `plan` | - | The same subject plan the pass was saved with. Checked against the folder's `plan.json` first, so a pass cannot be read into a job planned from a different pairing. |
| `subject` | `0` | Whose pass to read — the number **H3 Pass Save** wrote it under. Wire **H3 Subject Select → `previous_subject`** into it: the pass chains onto the subject **before** the one being refined. `previous_subject` is `-1` for the FIRST subject (there is no pass before it), and then this node returns `no_previous` instead of reading a pass. A number the plan does not have is refused. |
| `stream` | `crops` | `crops`: the crops that were fed to H3, with their crop positions — wire them into **H3 Face Stitch**'s `refined_crops` and `transform` to composite a checkpointed pass without re-tracking. `stitched`: the frames that pass composited back onto the clip — wire them into the next subject's `base_images`. |
| `no_previous` *(opt)* | - | The base to return when `subject` is `-1` — the first subject, which has no previous stitched pass to load. Wire the **target clip** here so a single reusable pass chain composites the first face straight onto the clip; ignored for any other subject. |

**A single reusable chain:** wire **H3 Subject Select → `previous_subject`** into `subject` and the **target clip** into `no_previous`, then `images → H3 Face Stitch → base_images`. Subject Select's `next` walks every subject in turn, so one chain covers any number of people — the first subject bases on the clip, each later subject on the previous subject's stitched result. (Wiring `subject_index` and a fixed subject instead is the hand-chained shape — one branch per subject, chained by hand.)

**A pass must match the clip's length.** Every pass in a chained job spans the **same frames as the clip**, because the chain composites each subject onto the previous subject's stitched result and stitches each refined crop back onto the frame it came from. If a saved pass was rendered from a clip of a **different length** — the clip changed, or a subject was rendered before a re-cut — this node refuses with both counts named, instead of letting **H3 Face Stitch** fail later with an opaque `index N out of bounds for dimension 0 with size N`. Recover with **Reset run** on the plan node (then re-queue from the first subject so every pass is rendered against the current clip), or **Process Plan** if the clip itself changed. The plan node's report and the **Pairing preview** window both surface this: a `passes:` line (total subjects = queues, how many remain, and the next subject) plus a frame-count warning when saved passes disagree with the clip. |

**Outputs:** `images`, `transform`, `report`. The `transform` comes back as the payload the tracker
built — `canvas`, `src_size` and `segments` as tuples — so a checkpoint feeds the released stitch
node directly; for `stream = stitched` it is returned too when the positions exist, which is what
lets a saved pass be re-stitched. A `crops` stream without its positions raises instead of handing
back frames nothing can be assembled from; a `stitched` stream whose positions were never saved
returns an empty `transform` and says so, because those frames are already composited.

A pass that is on disk:

```
subject 0   stage stitched   4 frame(s) at 1000x1000
images: <output>/shot 07/passes/subject_0000/stitched
          frame_0000.png ... frame_0003.png   ordinary PNGs: open them to see what this pass was actually fed
pass COMPLETE - crops and stitched are both recorded
manifest: <output>/shot 07/manifest.json
crop positions: <output>/shot 07/passes/subject_0000/crops.json   4 frame(s); canvas, src_size and segments come back as tuples, the payload the tracker built
```

The frame count comes from the **manifest record**, never from a tensor in the graph — so a
half-deleted frame sequence is a named refusal (`3 of 4 frame(s) are missing from …`) rather than a
short batch that quietly resizes the job.

## H3 Multi-Face Composite

The end of the job: the **composited clip**, plus the proof that every subject is actually in it.

**It re-blends nothing, deliberately.** Each pass was feathered once by **H3 Face Stitch**, and each
pass was meant to be composited onto the previous pass's result — so the last subject's stitched
frames already contain every earlier pass. This node picks those frames up off disk. There is no
`base_images` and no `transform` on it because a second feather laid over a feathered edge is how a
freshly refined face goes soft, which is the one thing this feature must not do to work that is
already good.

**The chain is an intent the passes declare, not a fact this node can see.** Which `base_images`
each stitch node was wired to lives in your graph, and nothing here can read it. So each pass
declares it — [`chained_from` on H3 Pass Save](#h3-pass-save), recorded in the **pass manifest** —
and the composite walks the plan order against those declarations:

- every pass declared its base and they match the plan's order → the report says so, and returning
  the last subject's frames is honest;
- a pass declared something that contradicts the plan's order (say every pass onto the target clip)
  → it **refuses, naming the subject**: that arrangement drops every earlier subject out of the
  finished clip while the report still lists them;
- an older job folder recorded nothing → it **warns** and names which subjects it cannot vouch
  for, because the passes may well be chained; nothing on disk says so.

### The chaining recipe

Said once, because it is the whole arrangement:

- subject 0's **H3 Face Stitch** takes the **target clip** as its `base_images`, and its
  **H3 Pass Save** leaves `chained_from` at `-1`;
- subject 1's takes **H3 Pass Load** with `stream = stitched`, `subject = 0`, wired into that stitch
  node's `base_images`, and its **H3 Pass Save** sets `chained_from` to `0`; subject 2's takes
  subject 1's and sets `1`, and so on down the plan;
- **H3 Multi-Face Composite** returns the last subject's stitched frames as the finished clip,
  having checked that every pass said the same thing.

**Composite order is plan order** — subjects numbered left to right at the match frame — and it is
deliberately **not a widget**. A widget would let the order drift away from the numbering the
reports, the pairing preview and the manifest all use, and then "subject 1" would mean two different
people inside one job. To change the order, change what is in the shot at the match frame.

**Inputs**

| Input | Default | What it does |
|---|---|---|
| `plan` | - | The subject plan every pass in this job ran on. Its subject order **is** the composite order, and it is checked against the `plan.json` already in the job folder: if the pairing inputs differ, the passes in that folder were cut from a different pairing, and this refuses rather than stitching two ideas of who is who into one clip. |
| `images` | - | The **target clip** — the same `IMAGE` batch that fed the first pass's H3 Face Stitch, from the same loader, unresized and untrimmed. Nothing is composited onto it here. It is wired in so the node can **prove the chain is complete**: every recorded stitched pass has to hold this many frames at this size, or the frames in the folder are not this clip's. It is also the clip the finished video's audio comes from, if you wire it there — see the audio note. |
| `overlap_threshold` | `0.25` | How much of their combined area two subjects' crop regions may share at the match frame before the report warns about them. `0` warns about any overlap, `1` never warns. Measured on the crop regions **H3 Face Track + Crop** would cut, not on the faces inside them. **A warning, never a refusal**: two people standing close are still two subjects and the clip is still the right answer — but one pass's feathered edge lands on the other's refined pixels, and a soft seam across a face is easier to live with once you know where it is. |

**Outputs:** `images` (the composited clip), `report`, `final_subject` (whose frames those are),
`subject_count`. It is an output node, so the report — the gate's verdict — reads off the node.

**The gate.** A job queued once per subject and interrupted between subjects would otherwise hand
back a clip where one person is sharp and the other is still blurry, looking finished. So every
subject's `stitched` stage has to be recorded in the manifest, and its frame count and frame size
have to agree with the clip wired in here. Anything missing and the node refuses, naming every
subject that never landed and pointing at `next`:

```
2 of 2 pass(es) are not in <output>/shot 07, in plan order subject 0 -> subject 1:
  - subject 0: no stitched pass is recorded in the manifest, so this subject's refined face is nowhere in the job folder
  - subject 1: no stitched pass is recorded in the manifest, so this subject's refined face is nowhere in the job folder
next subject: 0   H3 Subject Select's next goes to that subject: run its pass through to H3 Face Stitch, then save it with H3 Pass Save stage=stitched.
This composite refuses: a half-composited clip is worse than an error. It would look finished while the faces of the subjects named above kept the pixels the target clip arrived with.
```

**Audio.** These nodes carry image batches only — no node here carries audio, and none of them
touches it. Whatever audio the finished clip has comes from the **target clip**, at the point
where you assemble the video; each pass's generated audio is discarded by never being wired
anywhere. The report says so at the end of every successful run — a silent surprise at the end of
a long job is its own kind of bug — and the fix is the one the single-subject graph already uses:
wire the target clip's own audio into your save node.

A finished two-subject job:

```
job folder: <output>/shot 07
order: subject 0 -> subject 1   plan order = left to right at match frame 2
chain as declared by H3 Pass Save: subject 0 onto the target clip; subject 1 onto subject 0
every pass named the plate it was composited onto, so the last subject's stitched frames carry the earlier passes into the finished clip because each pass took the previous one's result as its base - nothing is re-blended here
subject 0   stitched 4 frame(s) at 1000x1000px   <output>/shot 07/passes/subject_0000/stitched
subject 1   stitched 4 frame(s) at 1000x1000px   <output>/shot 07/passes/subject_0001/stitched
final result: subject 1's stitched frames   this node hands back those 4 frame(s) and blends nothing: they carry subject 0 in them only if each of those passes was composited onto the one before it, which the chain line above says - or says nobody declared
no overlapping subject crop regions at match frame 2 (threshold 0.25)
audio: this node carries images only and touches no audio at all. Whatever audio the finished clip has comes from the target clip, at the point where you assemble the video - and each pass's generated audio is discarded by never being wired anywhere.
```

Two subjects whose crop regions overlap print this instead — and the job still runs:

```
overlap warning: subject 0 and subject 1 crop regions share 67.4% of their combined area at match frame 2, past the 0.25 set here - if they are two people, one pass's feathered edge lands on the other's refined pixels
```

---

## Lipsync

H3 is a **joint** audio-video model. `MiniMaxH3NativeAudioLock` encodes real audio into the audio
stream of the AV latent, sets `noise_mask` to ones for video and zeros for audio so only video
denoises, and the video branch cross-attends to that fixed audio. That is what shapes the mouth.

Feed it an isolated **vocals** track for a cleaner signal. The **original** audio goes separately
into the save node. Two distinct audio paths, easy to confuse.

---

## Gotchas

- **Frame count is snapped up to H3's 17k+5 grid** (5, 22, 39 … 175, 226, 362). Videos generated by
  H3 already sit on it. Anything else is rounded up: the extra frames are padded with reference
  content, refined, then discarded on stitch. Both the select and track nodes say so in their report
  and name the nearest grid value below. It costs sampling time and nothing else, but `frame_load_cap`
  will trim to a grid value if you would rather not pay it.
- **Cost is `canvas² × frames`.** Auto canvas sizing considers face size only, so on a long video it
  can pick a canvas that exceeds VRAM and falls back to streaming weights from system RAM, an
  order-of-magnitude slowdown rather than a clean error.
- **`SAMLoader`'s `AUTO` device mode leaves the model on CPU** until `prepare_device()` is called.
  This pack calls it; if you write your own SAM path, missing it makes mask passes 10-50× slower
  with the GPU idle at half power.

---

## Credits

The compositing approach is taken directly from
**[ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)** by
**[ltdrdata](https://github.com/ltdrdata)**, specifically `FaceDetailer` and its detailer paste.
The dilate-then-blur face-region mask, the crop-for-context-but-paste-only-the-face principle, and
the bbox+SAM masking path are all its design; this pack adapts them to a per-frame video pipeline.
If you find this useful, that project is why.

The lipsync path in the example workflows depends on `MiniMaxH3NativeAudioLock`, from
**[MiniMax-H3-NativeAudio-MusicVideo-Workflow](https://github.com/Shrek3OnVH5/MiniMax-H3-NativeAudio-MusicVideo-Workflow)**
by **[Shrek3OnVH5](https://github.com/Shrek3OnVH5)**. It is not redistributed here, so install it
from that repository.

Contributions:

- **[fatemark](https://github.com/fatemark)** — rank-and-index face selection, the numbered face
  preview, and the pluggable identity backends that make crowd tracking work on illustration
  ([#10](https://github.com/Carasibana/ComfyUI-H3-FaceRefine/pull/10))
- **[Gourieff](https://github.com/Gourieff)** — fixed a `SyntaxError` that stopped `nodes.py`
  importing on Python older than 3.12
  ([#12](https://github.com/Carasibana/ComfyUI-H3-FaceRefine/pull/12))

Also builds on:

- **[MiniMax H3](https://github.com/MiniMax-AI)**, the joint audio-video model being refined
- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** by comfyanonymous
- **[ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)** by Kosinkadink
- **[ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)** by city96
- **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)** for detection
- **[PySceneDetect](https://github.com/Breakthrough/PySceneDetect)** for hard-cut detection
- **[InsightFace](https://github.com/deepinsight/insightface)** for identity embeddings

---

## Made with these nodes

[![More Than Words Can Ever Steal](https://img.youtube.com/vi/18iiffk-QWE/maxresdefault.jpg)](https://www.youtube.com/watch?v=18iiffk-QWE)

**[More Than Words Can Ever Steal](https://www.youtube.com/watch?v=18iiffk-QWE)**, a music video by
[Carasibana](https://www.youtube.com/@Carasibana-Music), generated with MiniMax H3. Every shot
where the subject sits at any distance from camera was face-refined with this pack.

---

## Licence

MIT. See [LICENSE](LICENSE).

---

<sub>Built with [Claude Code](https://claude.com/claude-code).</sub>
