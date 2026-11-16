"""Version v0 on the bench: the organisers' baseline (starter/baseline.py) on the noisy input.

    uv run python -m bench.baseline [--level match] [--games ...] [--window 50]

Same two steps as starter/baseline.py: the team of the nearest tracked player wherever the ball
is seen (nobody where it is not), then a centred majority vote over +/- window frames, D where
the window has no ball. `smooth` is imported from the starter so the logic is the organisers'.
Reads the contract files players.parquet and ball_raw.parquet; writes
<root>/labels/v0.csv (window 50, the published floor) or v0_w<window>.csv.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from bench import metrica_io

sys.path.insert(0, os.path.join(metrica_io.ROOT, "starter"))
from baseline import AWAY, HOME, NOBODY, smooth  # noqa: E402  organisers' code, unchanged

ART = os.path.join(metrica_io.ROOT, "artifacts", "bench")
GAME_STEMS = ("metrica_g1", "metrica_g2", "metrica_g3")


def nearest_player_codes(root):
    ball = pd.read_parquet(os.path.join(root, "ball_raw.parquet"))
    players = pd.read_parquet(os.path.join(root, "players.parquet"))
    n = len(ball)
    codes = np.full(n, NOBODY, dtype=np.int8)
    seen = ball[ball["seen"]][["frame", "x_m", "y_m"]].rename(columns={"x_m": "bx", "y_m": "by"})
    m = players.merge(seen, on="frame", how="inner")
    m["d2"] = (m["x_m"] - m["bx"]) ** 2 + (m["y_m"] - m["by"]) ** 2
    best = m.loc[m.groupby("frame")["d2"].idxmin()]
    codes[best["frame"].to_numpy()] = np.where(best["team"].to_numpy() == 0, HOME, AWAY)
    return codes


def run(root, window=50, version=None):
    labels = smooth(nearest_player_codes(root), window)
    version = version or ("v0" if window == 50 else "v0_w{}".format(window))
    out = os.path.join(root, "labels", "{}.csv".format(version))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame({"frame": np.arange(len(labels)), "label": labels}).to_csv(out, index=False)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--level", default="match")
    ap.add_argument("--games", nargs="*", default=list(GAME_STEMS))
    ap.add_argument("--window", type=int, default=50)
    args = ap.parse_args(argv)
    for g in args.games:
        root = os.path.join(ART, "{}_{}".format(g, args.level))
        if os.path.isdir(root):
            print(run(root, args.window))


if __name__ == "__main__":
    main()
