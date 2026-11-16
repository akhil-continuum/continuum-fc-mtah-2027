"""Refuse the match where only bench roots are allowed.

Several bench tools take an explicit ``--root DIR`` (decode, the stage runner, measurement
scripts). Each of them should call ``require_bench_root(root)`` before it reads anything, so that a
mistyped or relative root can never run a bench-only code path (which may print rows) on the
match:

    from pipeline.io.roots import require_bench_root
    root = require_bench_root(root)          # returns the resolved path, or raises

A root is refused when, after resolving symlinks and relative parts,

- it is the match root (``artifacts/match`` of this checkout) or lies inside it;
- any path component pair is ``artifacts/match`` (another checkout, a rebuild copy);
- it lies inside the match data folder (``games.DATA_DIR``, ``MTAH_DATA_DIR``) or the download
  folder (``MTAH Data``);
- its ``meta.json`` says ``"source": "match"`` or names the competition game.

The error message names no path and no value: it is safe to print anywhere.
"""
import json
import os

from pipeline.io import sources

REFUSAL = "bench-only command refused: the root is the match or match data (use the match entry point)"


class NotABenchRoot(PermissionError):
    """Raised when a bench-only command is pointed at the match."""


def _inside(path, parent):
    parent = os.path.realpath(parent)
    return path == parent or path.startswith(parent.rstrip(os.sep) + os.sep)


def _forbidden_parents():
    import games
    return [sources.match_root(), games.DATA_DIR, os.path.join(sources.REPO, "MTAH Data")]


def is_match_root(root):
    """True when ``root`` is, or is inside, anything that holds the match (see module doc)."""
    real = os.path.realpath(os.path.abspath(str(root)))
    if any(_inside(real, parent) for parent in _forbidden_parents()):
        return True
    parts = real.split(os.sep)
    if any(a == "artifacts" and b == "match" for a, b in zip(parts, parts[1:])):
        return True
    if any(p.startswith("MTAH Data") for p in parts):
        return True
    meta = os.path.join(real, "meta.json")
    if os.path.isfile(meta):
        import games
        try:
            with open(meta) as handle:
                info = json.load(handle)
        except (OSError, ValueError):
            return False
        if isinstance(info, dict) and (info.get("source") == "match"
                                       or info.get("game_id") == games.GAME):
            return True
    return False


def require_bench_root(root):
    """The resolved root, or NotABenchRoot when it is the match (message names no path)."""
    if is_match_root(root):
        raise NotABenchRoot(REFUSAL)
    return os.path.realpath(os.path.abspath(str(root)))


def require_bench_roots(roots):
    return [require_bench_root(r) for r in roots]
