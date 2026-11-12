# Identity anchored in the target clip, never in the reference still

The reference still is a different image domain from the clip being refined, and the pack already measured the cost: similarity between an external/stylised reference and the clip scored 0.305 where same-domain faces score 0.5–0.7 (`nodes.py:1104`). So each subject's **identity anchor is built from its matched crop in the target clip**; the reference still feeds H3's picture conditioning only.

Pairing by position holds only at the match frame — after a cut, positions mean nothing — so the anchor, not the still, is what carries a subject across shots.
