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

## Repository facts worth knowing before you touch anything

- **Layout**: everything lives in one module — `nodes.py` (~3.5k lines, 7 node classes),
  with `picker_api.py` (the face-picker HTTP routes) and `web/h3_face_select.js` (the picker
  front end, registered via `WEB_DIRECTORY` in `__init__.py`).
- **Two custom ComfyUI types carry the pipeline's state**: `H3FACEPICK` (from
  `H3FaceSelect`) and `H3FACEXFORM` (from `H3FaceTrackCrop`, the crop-trajectory record that
  `H3FaceStitch` uses to paste refined faces back). Changes to the shape of these are
  changes to the whole graph.
- **Multiple people** are currently handled by running the pipeline once per subject and
  chaining stitched outputs through `base_images` — see the `identity_reference` tooltip in
  `nodes.py` and the "FOR MULTIPLE PEOPLE" note in `README.md`.
- **The interpreter with torch installed** is
  `C:\Users\emberlight\ComfyUI Cuda12.8\ComfyUI\.venv\Scripts\python.exe`
  (Python 3.13, torch cu128, CUDA available). The `miniconda3` and `standalone-env`
  pythons on PATH have no torch and are the wrong interpreter for anything here.
- **No test suite** in this pack, and `pytest` is not installed in that venv.
- `README.md` is the node reference and is kept in step with the tooltips in `nodes.py`;
  a node change that doesn't update `README.md` is incomplete.
