# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Point the ttnn runtime at a matching tt-metal source tree.

On a typical dev host the `ttnn` Python package is editable-installed
from somewhere, but its runtime .so looks up kernel sources and
dispatch-kernel includes via `TT_METAL_HOME`. If that env var isn't set
the runtime may fail with "Kernel file ... doesn't exist" or a dispatch
build error (e.g. `'DEVICE_PRINT' was not declared`) when the version
under CWD / fallback paths doesn't match the installed runtime ABI.

Resolution order:
    1. If `TT_METAL_HOME` is already set in the environment, respect it.
    2. Otherwise, derive it from the importable `ttnn` package by
       walking up from its `__file__` until a directory that contains a
       `ttnn/cpp/ttnn/operations/matmul` subtree is found.
    3. If still not found, raise with clear instructions.

Must be called (via `install()`) BEFORE `import ttnn` anywhere else in
the process.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _detect_tt_metal_home() -> str | None:
    """Walk up from the importable ttnn package to find a tt-metal root."""
    try:
        import importlib.util

        spec = importlib.util.find_spec("ttnn")
        if spec is None or not spec.origin:
            return None
        p = Path(spec.origin).resolve()
    except Exception:
        return None
    for parent in [p, *p.parents]:
        marker = parent / "ttnn" / "cpp" / "ttnn" / "operations" / "matmul"
        if marker.exists():
            return str(parent)
    return None


def install() -> None:
    home = os.environ.get("TT_METAL_HOME") or _detect_tt_metal_home()
    if home is None:
        raise RuntimeError(
            "Could not locate tt-metal for ttnn kernel sources. Set "
            "TT_METAL_HOME to the tt-metal checkout that matches your "
            "installed ttnn runtime."
        )
    os.environ["TT_METAL_HOME"] = home
    for p in (home, f"{home}/ttnn", f"{home}/tools"):
        if p not in sys.path:
            sys.path.insert(0, p)
