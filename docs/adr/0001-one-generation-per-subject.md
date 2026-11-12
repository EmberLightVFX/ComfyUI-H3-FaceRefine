# One generation per subject

H3 conditions on reference pictures **whole-frame** — `MiniMaxH3ReferenceToVideo` has no mask input, so putting every subject's reference into one generation leaves H3 to guess which face each picture applies to, and similar faces in frame is where identity swap comes from. Multi-face support therefore refines **one subject per pass**, each cropped to that subject alone, and composites the passes in a fixed order.

## Considered Options

- One generation carrying all subject references in its `ref_image_*` slots — cheaper, rejected on identity-swap risk and on losing the per-subject magnification the whole pack exists for.
- Multi-character RefMod conditioning (`academiasd`) — rejected, see ADR 0004.

## Consequences

N subjects cost N generations at the same canvas size, each paying H3's 17-step frame grid.
