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

**Pass stage**:
One of the two halves of a pass that get checkpointed into the job folder: `crops` (what H3 was fed, plus their crop positions) and `stitched` (that pass composited back onto the target clip). A pass counts as complete only once its **stitched** stage is recorded.
_Avoid_: step, phase, checkpoint (that is the act, not the half)

**Pass manifest**:
The job folder's `manifest.json`: which subjects have which pass stages on disk, and which passes are therefore complete. It is what `next` reads to resume an interrupted job.
_Avoid_: todo list, state file, progress file

**Pairing fingerprint**:
The inputs that decide a job's pairing - both plate sizes, the match frame, `subject_source`, the match tolerance, the exclusions, and the subject set itself - compared before any pass reads or writes a job folder, so two pairings never mix into one clip.
_Avoid_: hash, checksum, signature

**Composite order**:
The order the passes get composited in: the subject plan's own order, left to right at the match frame. Fixed by the plan, never by a widget, and named in the composite's report.
_Avoid_: render order, queue order, pass order (a pass belongs to a subject, this names the subjects)

**Composited clip**:
The finished clip - the last subject's stitched frames, which already carry every earlier pass composited into them, because each pass took the previous pass's result as its base.
_Avoid_: merged clip, blended output (nothing is blended at the end), combined clip

**Overlap warning**:
A note in the composite's report that two subjects' crop regions cover the same pixels at the match frame by more than the threshold, so one pass's feathered edge lands on another's refined pixels. A warning, never a refusal.
_Avoid_: collision, conflict, clash


**Shot**:
A run of consecutive frames with no cut; face numbering and subject pairing are per shot.
_Avoid_: scene, segment (in user-facing copy)

**Absence**:
A subject genuinely not present in a given shot, so those frames keep their original pixels.
_Avoid_: missing, gap, dropped

**Planned absence**:
An absence the subject plan chose — a shot with no partner for that subject, marked absent in its pick entry, so the clip's own pixels are kept there. Expected, never a failure.
_Avoid_: skip, dropout, missing shot

**Lost track**:
A shot where the tracker was handed the subject and produced nobody for them — identity below threshold, or the face never detected. A failure worth telling the user about, and never to be called an absence.
_Avoid_: absence, dropout, lost face

**Shot account**:
One subject's per-shot record of a pass: present, absent (planned), or lost, each with the clip frames it covers. What Subject Select and Pass Save print, so a real absence and a tracking failure cannot be confused.
_Avoid_: status list, per-shot summary, coverage report

**Pairing**:
Matching a face in the target clip at the match frame to a face in the reference still, so a subject gains a subject reference.
_Avoid_: mapping, linking, detection match

**Subject plan**:
The settled list of subjects for a job — who is in the clip, what each is paired with, and which faces are excluded.
_Avoid_: face list, track list, plan file (that's where it lives, not what it is)

**Excluded face**:
A face found at the match frame that the user has switched off, so it never becomes a subject and keeps its original pixels.
_Avoid_: disabled face, ignored face, skipped track

**Unpaired face**:
A face found on one plate at the match frame with no partner on the other. What becomes of it is the plan's `unmatched_faces` setting - leave, no reference, or the whole reference still - decided once, in the plan, because it shapes the subject set.
_Avoid_: orphan face, missed match, unmatched subject

**Left-out face**:
An unpaired face the plan did not make a subject of (`unmatched_faces=leave`), so no pass runs for it and it keeps its original pixels. Expected behaviour and the safe default, not a failure, and not an excluded face - nobody switched it off.
_Avoid_: excluded face, skipped subject, dropped face

**Whole-still reference**:
A subject reference that is the entire reference still instead of a crop of it (`unmatched_faces=with_still`), used for a subject with no paired face in the still. Conditioning is whole-frame, so which face in the still H3 applies is not guaranteed.
_Avoid_: full-frame reference, fallback crop, whole-image reference

## Distinguish these three "references"

- **Reference still** — the high-quality image of the whole scene.
- **Subject reference** — one face's crop out of it, what H3 generates from.
- **`identity_reference`** — an existing node input meaning *who to track*, not *what to generate from*.
