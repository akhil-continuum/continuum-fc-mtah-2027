"""Whole-half aggregates of the kick-off formation tests (no windows, no frames, no positions).

    uv run python -m pipeline.restarts.diagnose match
    uv run python -m pipeline.restarts.diagnose bench --game metrica_g3_match

Per half: histograms (at most 16 bins) of tracked players and of players in the opponents'
half, and the share of frames passing each component of the per-frame kick-off test. Used to
compare the match with bench games when kick-offs are missed; prints through summary() only.
"""
import argparse
import os
import sys

import numpy as np

from pipeline.io.sources import ARTIFACTS, match_root
from pipeline.report import match_entry, quiet_warnings, summary
from pipeline.restarts import kickoff
from pipeline.restarts.__main__ import load_root
from pipeline.restarts.formation import kickoff_frame_measures


def half_aggregates(m, half):
    sel = m["half"].to_numpy() == half
    x = m[sel]
    out = {}
    tracked = x["n_tracked"].to_numpy()
    out["n_tracked_hist_0_30_by2"] = np.histogram(np.clip(tracked, 0, 29), bins=range(0, 32, 2))[0].tolist()
    wrong = x["n_wrong_half"].to_numpy()
    out["n_wrong_half_hist_0_to_10plus"] = np.bincount(np.clip(wrong, 0, 10).astype(int), minlength=11).tolist()
    ball = x["ball_d_spot"].to_numpy()
    comp = {
        "enough_tracked": tracked >= kickoff.MIN_TRACKED,
        "own_halves": wrong <= kickoff.KO_MAX_WRONG_HALF,
        "ball_unseen_or_on_spot": ~np.isfinite(ball) | (ball <= kickoff.BALL_SPOT_M),
        "ball_seen": np.isfinite(ball),
        "taker_at_spot_any_team": (x["d_spot_0"].to_numpy() <= kickoff.SPOT_RADIUS_M)
        | (x["d_spot_1"].to_numpy() <= kickoff.SPOT_RADIUS_M),
        "circle_empty_one_team": (x["n_circle_0"].to_numpy() == 0) | (x["n_circle_1"].to_numpy() == 0),
        "circle_empty_both": (x["n_circle_0"].to_numpy() == 0) & (x["n_circle_1"].to_numpy() == 0),
        "ring_ge_min_both": (x["n_ring_0"].to_numpy() >= kickoff.DEF_RING_MIN)
        & (x["n_ring_1"].to_numpy() >= kickoff.DEF_RING_MIN),
    }
    holds, kick = kickoff.formation_frames(m)
    comp["formation_a_taker"] = (kick >= 0)[sel]
    comp["formation_holds"] = holds[sel]
    out["share_pass"] = {k: round(float(np.mean(v)), 4) for k, v in comp.items()}
    # how many frames pass everything except one component (what blocks the formation most)
    base = {k: comp[k] for k in ("enough_tracked", "own_halves", "ball_unseen_or_on_spot")}
    out["share_own_halves_and_taker"] = round(float(np.mean(comp["own_halves"]
                                                             & comp["taker_at_spot_any_team"])), 4)
    out["share_own_halves_and_ring_both"] = round(float(np.mean(comp["own_halves"]
                                                                 & comp["ring_ge_min_both"])), 4)
    out["share_base_ok"] = round(float(np.mean(np.logical_and.reduce(list(base.values())))), 4)
    return out


def report(root, name):
    players, meta, ball = load_root(root)
    m = kickoff_frame_measures(players, meta, ball)
    for half in range(1, len(meta["halves"]) + 1):
        summary(_name=name, **{"h{}_{}".format(half, k): v
                               for k, v in half_aggregates(m, half).items()})


@match_entry
def run_match():
    quiet_warnings()
    report(match_root(), "restarts_diagnose")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", choices=("match", "bench"))
    parser.add_argument("--game", nargs="*", default=[])
    args = parser.parse_args(argv)
    if args.target == "match":
        return run_match()
    for g in args.game:
        report(os.path.join(ARTIFACTS, "bench", g), "restarts_diagnose.bench.{}".format(g))
    return 0


if __name__ == "__main__":
    sys.exit(main())
