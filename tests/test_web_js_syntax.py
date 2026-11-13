"""Every web/*.js must parse, so a front-end button cannot silently vanish.

ComfyUI loads each front-end script as an ES module; if one of them has a syntax
error the whole script is dead, so its extension never registers and the node's
buttons disappear from the graph while the nodes and routes still load fine. That
failure is invisible to the Python tests, so without this check a bad edit ships
quietly (a juxtaposed string literal - `"a" "b"` with no `+` - once killed the
pairing-preview extension and its two buttons).

`node --check` is used on a `.mjs` copy of each file, because these are ESM
(with `import` statements): a `.js` copy would be parsed as CommonJS and reject the
imports. The whole test skips when `node` is not on PATH (the numpy-only CI runner
has none), so it stays safe there and guards locally, where it matters.
"""

import pathlib
import shutil
import subprocess

import pytest

PACK_ROOT = pathlib.Path(__file__).resolve().parents[1]
WEB = PACK_ROOT / "web"

JS_FILES = sorted(p.name for p in WEB.glob("*.js")) if WEB.is_dir() else []


@pytest.mark.parametrize("name", JS_FILES)
def test_each_web_script_parses_as_es_module(name, tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not on PATH; skipping the JS parse check")

    # Copy to a .mjs so node parses it as an ES module (these files use import).
    copy = tmp_path / (pathlib.Path(name).stem + ".mjs")
    copy.write_bytes((WEB / name).read_bytes())

    proc = subprocess.run([node, "--check", str(copy)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, (
        f"{name} does not parse as an ES module:\n"
        f"{proc.stderr.strip() or proc.stdout.strip()}")


def test_there_are_web_scripts_to_check():
    # Guard the guard: if the web dir or its scripts vanish, the parametrized test
    # above would pass vacuously. The pairing preview and picker both ship scripts.
    assert "h3_pair_preview.js" in JS_FILES
    assert "h3_face_select.js" in JS_FILES
