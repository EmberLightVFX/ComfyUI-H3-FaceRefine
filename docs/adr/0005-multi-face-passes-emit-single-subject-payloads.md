# Multi-face passes emit the single-subject payload shapes

A multi-face feature that emitted its own multi-subject pick and transform types would need a
second stitcher, a second per-frame mask path and a second tracker, all of which then drift from
the originals. Instead **H3 Subject Select synthesises one subject's payload in exactly the shape
`H3FACEPICK` already has**, so `H3FaceTrackCrop`, `H3FaceStitch` and `H3PerFrameDenoise` run
unchanged inside every pass. There is exactly one new custom type — `H3FACEPLAN`, the subject
plan — and multi-face support is a *driver over* the existing pipeline rather than a fork of it.

## Consequences

Subject iteration happens above the pipeline (one queue run per subject, ADR 0001), not inside
it. Per-subject absence, backfill and smoothing are inherited for free, and any fix to the
single-subject path lands on the multi-subject path automatically.
