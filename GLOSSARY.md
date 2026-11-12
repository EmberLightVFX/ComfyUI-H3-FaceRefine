# ComfyUI-H3-FaceRefine

Re-generates small or blurry faces in MiniMax H3 video at a scale H3 renders well, then
composites them back. This is the language the nodes, tooltips and reports use.

## Language

**Target clip**:
The footage being fixed — the blurry, low-quality or small-faced video.
_Avoid_: source video, input video, bad plate, to-plate

**Reference still**:
A single high-quality image of the same scene as the target clip at one frame, whose faces supply the quality to be transferred.
_Avoid_: reference image (collides with H3's `ref_image_*`), good plate, from-plate

**Subject reference**:
The crop of one face lifted out of the reference still and handed to H3 as that subject's picture.
_Avoid_: face reference, crop ref

**Subject**:
One person tracked through the target clip, named by a face and its identity anchor.
_Avoid_: person, face, track, target

**Match frame**:
The single frame of the target clip whose face positions line up with the reference still, where subjects and subject references get paired.
_Avoid_: first frame (it need not be frame 0), anchor frame

**Identity anchor**:
A subject's identity signature taken from a crop of the target clip itself, used to hold that subject across cuts.
_Avoid_: reference embedding, identity reference

**Pass**:
One refinement run of the pipeline for exactly one subject, ending with that subject composited back into the target clip.
_Avoid_: iteration, loop, render

**Shot**:
A run of consecutive frames with no cut; face numbering and subject pairing are per shot.
_Avoid_: scene, segment (in user-facing copy)

**Absence**:
A subject genuinely not present in a given shot, so those frames keep their original pixels.
_Avoid_: missing, gap, dropped

**Pairing**:
Matching a face in the target clip at the match frame to a face in the reference still, so a subject gains a subject reference.
_Avoid_: mapping, linking, detection match

**Subject plan**:
The settled list of subjects for a job — who is in the clip, what each is paired with, and which faces are excluded.
_Avoid_: face list, track list, plan file (that's where it lives, not what it is)

**Excluded face**:
A face found at the match frame that the user has switched off, so it never becomes a subject and keeps its original pixels.
_Avoid_: disabled face, ignored face, skipped track

## Distinguish these three "references"

- **Reference still** — the high-quality image of the whole scene.
- **Subject reference** — one face's crop out of it, what H3 generates from.
- **`identity_reference`** — an existing node input meaning *who to track*, not *what to generate from*.
