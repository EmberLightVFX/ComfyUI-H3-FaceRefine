# Multi-face support lands as new nodes only

ComfyUI stores widget values **positionally** (`nodes.py:3191`), and `H3FACEPICK` and `H3FACEXFORM` already sit in released workflows. Reshaping them, or inserting widgets into the existing seven nodes' input lists, silently reinterprets what people have saved. Multi-face support therefore arrives as **new nodes and new custom types** that reuse the existing helpers; the seven released nodes stay byte-stable.

Existing nodes may gain behaviour only where it cannot change a saved graph's meaning. Anything else is a new node.
