"""Where each input lives and where each stage writes, for the match and for bench games.

The match gets every file location from ``games.asset_paths()`` and every per-game fact (frame
count, halves, pitch) from games.py. Nothing here names a match file or a match frame.

Bench games are discovered under ``bench_data/<game_id>/``: a directory holding exactly one
EPTS metadata XML (``*metadata*.xml`` or, failing that, a single ``*.xml``) and one tracking
``*.txt`` (``*tracking*.txt`` preferred). The bench-builder owns bench_data/; this module only
reads it. ``--meta``/``--raw`` on the command line bypass discovery.
"""
import glob
import os
import sys
from dataclasses import dataclass, field

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
import games  # noqa: E402

from pipeline.io import frames  # noqa: E402

ARTIFACTS = os.path.join(REPO, "artifacts")
BENCH_DATA = os.path.join(REPO, "bench_data")


@dataclass
class GameInput:
    game_id: str
    source: str                   # "match" or "bench"
    meta_path: str
    raw_path: str
    out_dir: str
    game: str = None              # key in games.GAMES, match only
    n_frames: int = None
    halves: list = None
    pitch: tuple = None
    extra: dict = field(default_factory=dict)


def match_root():
    return os.path.join(ARTIFACTS, "match")


def bench_root(game_id):
    return os.path.join(ARTIFACTS, "bench", game_id)


def match_input(game=games.GAME):
    """The competition match (tactical source), entirely from games.py."""
    meta, raw = games.require(["atd_meta", "atd_raw"], game=game, source="tactical")
    return GameInput(game_id=game, source="match", meta_path=meta, raw_path=raw,
                     out_dir=match_root(), game=game,
                     n_frames=games.source_frames("tactical", game),
                     halves=frames.match_halves(game), pitch=tuple(games.GAMES[game]["pitch"]))


def _pick(paths, preferred):
    if len(paths) == 1:
        return paths[0]
    best = [p for p in paths if preferred in os.path.basename(p).lower()]
    return best[0] if len(best) == 1 else None


def find_pair(folder):
    """(meta_xml, raw_txt) in a folder, or None when not exactly one of each."""
    xmls = sorted(glob.glob(os.path.join(folder, "*.xml")))
    txts = sorted(glob.glob(os.path.join(folder, "*.txt")))
    meta = _pick(xmls, "metadata")
    raw = _pick(txts, "tracking")
    if meta and raw:
        return meta, raw
    return None


def bench_inputs(bench_dir=BENCH_DATA, only=None):
    """GameInput for every bench game folder with an EPTS pair, sorted by game id."""
    out = []
    if not os.path.isdir(bench_dir):
        return out
    for name in sorted(os.listdir(bench_dir)):
        folder = os.path.join(bench_dir, name)
        if not os.path.isdir(folder) or (only and name not in only):
            continue
        pair = find_pair(folder)
        if pair is None:
            continue
        out.append(GameInput(game_id=name, source="bench", meta_path=pair[0], raw_path=pair[1],
                             out_dir=bench_root(name)))
    return out


def explicit_input(meta_path, raw_path, out_dir, game_id=None):
    """A bench-style input from explicit paths (tests, fixtures, one-off games)."""
    game_id = game_id or os.path.basename(os.path.dirname(os.path.abspath(raw_path)))
    return GameInput(game_id=game_id, source="bench", meta_path=meta_path, raw_path=raw_path,
                     out_dir=out_dir)
