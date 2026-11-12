"""pytest plugin: keep the pack root out of the collection tree as a *package*.

The pack root is a ComfyUI package - it has an ``__init__.py``, and that file imports
``nodes.py``, which imports ComfyUI itself. pytest turns every directory holding an
``__init__.py`` into a :class:`pytest.Package` and imports that file before running
anything inside it, so a test run with no ComfyUI installed - the CI run, and every
geometry test in ``tests/`` - could not even start.

Nothing in ``tests/`` needs the package: ``plan_store`` and ``multi_face_nodes`` are
imported by name off the pack root (``pythonpath`` in pyproject.toml). Collecting the pack
root as a plain directory keeps the node machinery out of the run, and changes nothing
about how ComfyUI loads the pack.

It has to be a plugin rather than a hook in ``conftest.py``: pytest asks for a directory's
collector through the hook proxy of that directory's *parent*, so a conftest inside the
pack root is not consulted for the pack root itself.
"""

from pathlib import Path
import os.path

import pytest

PACK_ROOT = Path(__file__).resolve().parent


def pytest_collect_directory(path, parent):
    """Collect the pack root as a plain directory instead of as a package."""
    directory = Path(path)
    if os.path.normcase(str(directory)) == os.path.normcase(str(PACK_ROOT)):
        return pytest.Dir.from_parent(parent, path=directory)
    return None
