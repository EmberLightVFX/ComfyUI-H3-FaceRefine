"""The pure half of multi-face refinement: pairing, the subject plan, the job folder.

Import boundary is the whole point of this module (ADR 0006): numpy plus the standard
library, never ComfyUI, never torch, never `folder_paths`. `nodes.py` stays untouched
(ADR 0002), and the geometry below stays testable with synthetic boxes on a machine with
no GPU, no weights and no ComfyUI on the path. Anything that needs the ComfyUI world -
the output folder, the detector, pixel tensors - belongs in `multi_face_nodes.py`.

Coordinate conventions, used everywhere here:

* a FACE box is ``[x1, y1, x2, y2]``, the same shape `nodes.py` detects and
  `H3FACEPICK` carries;
* a RECT is ``[x, y, w, h]``, the same shape the tracker's crop box has;
* plate sizes are ``[width, height]`` in pixels.
"""

from __future__ import annotations

import json
import os

import numpy as np

# Bumped when the plan's shape changes, so a job folder written by an older pack is
# recognised rather than half-read.
PLAN_VERSION = 1
PLAN_FILENAME = "plan.json"

SUBJECT_SOURCES = ("target_clip", "reference_still", "union")

# How far the two plates' aspect ratios may drift before pairing calls it a different
# framing. Rounding of one true aspect (1080 against 1088) sits well inside this;
# 16:9 against 4:3 does not.
ASPECT_TOLERANCE = 0.01

# The tracker's defaults, mirrored so the plan can record the rectangle the tracker
# WOULD use before any pass has run. `crop_factor` and `canvas_mode="manual"` 768x768
# are `H3FaceTrackCrop`'s shipped defaults (nodes.py:1031, nodes.py:1043).
DEFAULT_CROP_FACTOR = 2.5
DEFAULT_CANVAS_ASPECT = 1.0
