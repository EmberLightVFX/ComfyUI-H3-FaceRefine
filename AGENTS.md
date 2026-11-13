# AGENTS.md

ComfyUI node pack for refining small/blurry faces in MiniMax H3 video: per-frame face
tracking, normalised crops, AV-latent injection, and feathered stitch-back.

## Agent skills

### Issue tracker

Issues live in GitHub Issues on `EmberLightVFX/ComfyUI-H3-FaceRefine` (this fork, not
upstream `Carasibana/...`), driven by the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary — `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`,
`wontfix`. All five exist on the fork. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.

## Working rules

- **Run one subagent at a time. Never spawn two concurrently.** This machine has no GPU
  headroom for two agents working at once. Spawn one, wait for its report, then spawn the
  next. Same rule applies to any two GPU-touching processes (a sample run, a model load).
- **Check every saved graph with `graph_check.py` before you call it done.** It is pure
  (`json`/`re`/`plan_store`, no `comfy`), GPU-free, and enforces the pack invariants that
  ComfyUI's own `validate_workflow` cannot (see Repository facts). Run
  `gc.check_graph(gc.load_workflow(path))` on any hand-built or user workflow and expect
  `[]`. The shipped `example_workflows/*.json` are its green fixtures, pinned by
  `tests/test_graph_check.py`, which `.github/workflows/tests.yml` already runs — that is
  the review gate for graph work.

## Repository facts worth knowing before you touch anything

- **Layout**: the seven released nodes live in `nodes.py` (~3.5k lines) and are **frozen** —
  ADR 0002 and ADR 0006 forbid changing them. The multi-face feature (1.2.0) lives beside them in
  `plan_store.py` (the pure plan module: numpy/json/os only, no `comfy` import, report sentences
  pinned by tests), `multi_face_nodes.py` (5 more node classes), `pair_preview_api.py` (the
  pairing-preview routes) and `web/h3_pair_preview.js`. `picker_api.py` and
  `web/h3_face_select.js` are the released face-picker and are also frozen. `__init__.py` merges
  the mappings.
- **Three custom ComfyUI types carry the pipeline's state**: `H3FACEPICK` (from
  `H3FaceSelect`, and narrowed per subject by `H3SubjectSelect`), `H3FACEXFORM` (from
  `H3FaceTrackCrop`, the crop-trajectory record that `H3FaceStitch` uses to paste refined faces
  back), and `H3FACEPLAN` (the subject plan, from `H3Multi-Face Plan`). Changes to the shape of
  these are changes to the whole graph.
- **Multiple people** are handled by the multi-face nodes: one plan, one pass per subject, passes
  chained through a job folder and composited in plan order. The older hand-chained recipe in the
  `identity_reference` tooltip and the "FOR MULTIPLE PEOPLE" note in `README.md` still works for
  graphs with no multi-face nodes.
- **The interpreter with torch installed** is
  `C:\Users\emberlight\ComfyUI Cuda12.8\ComfyUI\.venv\Scripts\python.exe`
  (Python 3.13, torch cu128, CUDA available). The `miniconda3` and `standalone-env`
  pythons on PATH have no torch and are the wrong interpreter for anything here.
- **There is a test suite**: `tests/` (~17 files) runs on that interpreter with
  `-m pytest`; the pure tests need no ComfyUI, no GPU and no weights, and
  `.github/workflows/tests.yml` runs them on plain pypi. Tests that drive the released tracker or
  stitch need ComfyUI importable and skip themselves in CI. Never let a test write into the real
  ComfyUI `output` folder — redirect the output root to a tmp dir.
- **The real output root is**
  `C:\Users\emberlight\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output` — this is a
  Comfy **Desktop** install launched with `--output-directory`, so `ComfyUI\output` holds only
  a placeholder. Any code or test that reads/writes run output must use the Desktop root (and
  tests must still redirect to a tmp dir, never there).
- **`graph_check.py` is the repo-side graph checker** (pure: `json`/`re`/`plan_store`, no
  `comfy`, GPU-free — the same shape as `plan_store.py`). It enforces pack invariants a saved
  graph can satisfy while staying type-correct and schema-legal, i.e. invisible to
  `validate_workflow`: the right kind of VAE in each loader socket (video VAE into
  `VAEDecode`/`H3InjectVideoLatent`/`MiniMaxH3ReferenceToVideo.vae`, audio VAE into the
  `audio_vae` sockets), `H3PassSave.stage` matching what feeds its `images` socket
  (tracker→`crops`, stitch→`stitched`), detector + confidence agreeing across
  `H3FaceSelect`/`H3MultiFacePlan`/`H3FaceTrackCrop`, `BasicScheduler.steps` matching the step
  count the loaded turbo LoRA names, no orphan nodes, and `H3PassLoad` chaining onto the
  previous subject in plan order (`chained_from`, when set explicitly, likewise). Read it in
  via `gc.check_graph(gc.load_workflow(path))`; `tests/test_graph_check.py` pins the shipped
  templates as clean and each fault as caught.
- `README.md` is the node reference and is kept in step with the tooltips in `nodes.py`;
  a node change that doesn't update `README.md` is incomplete.
