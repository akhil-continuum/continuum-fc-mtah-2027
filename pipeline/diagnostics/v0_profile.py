"""Label-free profile of v0's output on any load-stage root (match or bench).

    uv run python -m pipeline.diagnostics.v0_profile match
    uv run python -m pipeline.diagnostics.v0_profile bench [--game metrica_g3_match ...]

v0 is the organisers' baseline (nearest tracked player where the ball is seen, centred majority
over +-50 frames, D where the window has no ball), computed by bench.baseline from the contract
files. Prints, through summary(): predicted possessions per minute, share of D, stoppages per
minute, and histograms of predicted possession and D-run lengths (seconds). Nothing is scored here
and nothing is written next to the match data.
"""
import argparse
import os

import numpy as np

from bench import baseline as v0mod
from pipeline.diagnostics import input_stats
from pipeline.io import sources
from pipeline.report import match_entry, quiet_warnings, summary

# Params
WINDOW = 50
LEN_EDGES_S = [0, 1.5, 4, 10, 20, 40, 80, 10 ** 6]


def profile(root):
    fr = input_stats.Frames(root)
    lab = np.asarray(v0mod.smooth(v0mod.nearest_player_codes(root), WINDOW)).astype(str)
    poss, dead = [], []
    for a, b in fr.spans:
        seg = lab[a:b]
        change = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        starts = np.r_[0, change]
        ends = np.r_[change, len(seg)]
        for st, en in zip(starts, ends):
            (dead if seg[st] == "D" else poss).append((en - st) / fr.fps)
    minutes = fr.graded.sum() / fr.fps / 60.0
    g = fr.graded
    out = {
        "graded_minutes": round(minutes, 2),
        "pred_poss_per_min": round(len(poss) / minutes, 3),
        "pred_stoppages_per_min": round(len(dead) / minutes, 3),
        "pred_share_D": round(float((lab[g] == "D").mean()), 4),
        "pred_share_H_of_alive": round(float((lab[g] == "H").sum() / max(1, (lab[g] != "D").sum())), 4),
        "len_edges_s": LEN_EDGES_S[:-1],
        "pred_poss_len_hist": [int(c) for c in np.histogram(poss, bins=LEN_EDGES_S)[0]],
        "pred_dead_len_hist": [int(c) for c in np.histogram(dead, bins=LEN_EDGES_S)[0]],
        "pred_poss_median_s": float(np.median(poss)) if poss else None,
    }
    return out


@match_entry
def run_match():
    quiet_warnings()
    summary(_name="v0_profile.match", **profile(sources.match_root()))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Label-free v0 output profile")
    ap.add_argument("target", choices=("match", "bench"))
    ap.add_argument("--game", nargs="*")
    args = ap.parse_args(argv)
    if args.target == "match":
        return run_match()
    for root in input_stats.bench_roots(args.game):
        summary(_name="v0_profile.bench.{}".format(os.path.basename(root)), **profile(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
