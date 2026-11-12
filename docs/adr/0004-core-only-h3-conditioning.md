# Core-only H3 conditioning

Multi-character reference binding exists in this install as `academiasd`'s `AcademiaSD_H3RefModCharacter` / `AcademiaSD_H3RefModEncode`, but it needs weights under `models/refmods` (not present), is a format we don't own, and would make the pack's behaviour depend on a third party's release. H3 conditioning therefore goes through **core `MiniMaxH3*` nodes only** (`comfy_extras/nodes_minimax_h3.py`).

Per-subject isolation is achieved by running one pass per subject (ADR 0001) rather than by reaching for a multi-character conditioning mechanism.
