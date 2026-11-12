# New nodes live in new modules, `nodes.py` stays untouched

The obvious move is to add the multi-face nodes to `nodes.py` beside the other seven. We're not doing that: it is 3,477 lines of released code, and ADR 0002 promises it doesn't change. New code goes in `plan_store.py` (subject plan, job folder, crop-position serialisation — the pure and testable part), `multi_face_nodes.py` (the five new nodes), `pair_preview_api.py` and `web/h3_pair_preview.js`, all wired up from `__init__.py`.

This also puts the geometry and serialisation logic behind an import boundary that unit tests can reach without loading the node machinery. Splitting `nodes.py` itself is separate work, not part of this feature.
