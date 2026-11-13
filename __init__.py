"""ComfyUI-H3-FaceRefine

Per-frame face tracking, normalised cropping, AV-latent injection and feathered
stitch-back, so MiniMax H3 can re-generate small/distant faces at a useful scale
and be composited back into the source video.

Why this exists
---------------
H3 renders faces poorly when the head occupies a small fraction of the frame.
That is a property of head-size-in-frame, not of output resolution, so no
post-process upscaler fixes it. The workaround is to crop to the face, hand H3
a sequence where the face is large, refine at low denoise so the result stays
frame-aligned, then paste back.

The crop must be PER FRAME. A single fixed crop degenerates to a wide shot on a
push-in, which hands H3 exactly the small face it fails on.
"""

import os

from .nodes import NODE_CLASS_MAPPINGS as _RELEASED_CLASS_MAPPINGS
from .nodes import NODE_DISPLAY_NAME_MAPPINGS as _RELEASED_DISPLAY_NAME_MAPPINGS

# The multi-face nodes live in their own modules and are wired up here (ADR 0006). The
# released seven keep their names and their input schemas exactly as they were (ADR 0002),
# so saved workflows keep meaning what they meant - which is also why the merged mappings
# are new dicts rather than an in-place update of `nodes.py`'s.
from .multi_face_nodes import CUSTOM_TYPES as CUSTOM_TYPES
from .multi_face_nodes import NODE_CLASS_MAPPINGS as _MULTI_FACE_CLASS_MAPPINGS
from .multi_face_nodes import NODE_DISPLAY_NAME_MAPPINGS as _MULTI_FACE_DISPLAY_NAME_MAPPINGS

# The one new custom type: H3FACEPLAN, the subject plan (ADR 0005). Every pass reuses the
# existing H3FACEPICK / H3FACEXFORM shapes.
NODE_CLASS_MAPPINGS = {**_RELEASED_CLASS_MAPPINGS, **_MULTI_FACE_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**_RELEASED_DISPLAY_NAME_MAPPINGS,
                              **_MULTI_FACE_DISPLAY_NAME_MAPPINGS}

# Face picker: a button on H3 Load Video + Face Select that shows the detected faces
# and writes the chosen index per shot into confirmed_pick.
WEB_DIRECTORY = "./web"

try:
    from . import picker_api

    picker_api.register()
except Exception as _exc:  # the nodes must load even if the picker route does not
    print("[H3FaceRefine] face picker unavailable: %s" % _exc)

# The pairing preview (#9): a panel on H3 Multi-Face Plan showing the plan node's own pairing
# preview image and, per subject, BOTH faces side by side - the clip face on the match frame
# (clip_N.png) and the reference crop (subject_N.png) - read out of the job folder. The preview
# route has one read-only route and its own front-end script, and writes nothing to disk - the
# panel's per-face toggle edits only the node's typed `excluded_faces` widget, which the node
# re-applies on the next Process Plan. A second button, Process Plan, previews new settings
# without tripping the fingerprint guard by running the plan node itself into a throwaway
# `<job_name>__preview_*` folder (non-destructive: the real job is never touched). Registered the
# same way as the picker, so the nodes still load if the route does not register.
try:
    from . import pair_preview_api

    pair_preview_api.register()
except Exception as _exc:  # the nodes must load even if the preview route does not
    print("[H3FaceRefine] pairing preview unavailable: %s" % _exc)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "CUSTOM_TYPES",
           "WEB_DIRECTORY"]
