"""Ball smoothness: does a bench ball look as low-pass filtered as the match's? (T-33, P-17a)

    uv run python -m bench.smoothness check [--roots metrica_g1_match ...]
    uv run python -m bench.smoothness fit [--levels low match high] [--sigmas 6 7 8 ...]
                                          [--orders before_gaps after_gaps]

Shape statistics of the ball come from ``pipeline.diagnostics.touch_funnel.compute`` (the same
code that wrote reports/touch_funnel.match.json from the match). Only these input aggregates are
compared, never touch rates, labels or scores:

- ``raw_d2_lag1_autocorr``: lag-1 autocorrelation of the raw second difference (white jitter about
  -0.67, a smoothed track near +1). Compared as a relative difference of 1 - r.
- ``kink_width_median`` and ``kink_width_hist_share``: frames with dv at or above half the peak
  around each velocity-change peak above 4 m/s.
- ``kink_short_over_long_dv_median``: dv over 3 frames against dv over 7 at those peaks.
- ``ladder_peaks_per_min`` s3, s4, s6, s8: velocity-change peaks above 3, 4, 6, 8 m/s per minute
  of graded seen-ball time.

Smoothness distance (0 = identical): the mean of four groups, each in [0, 1]: autocorrelation,
kink width (mean of median relative difference and histogram total variation), kink ratio, and
ladder (mean of four relative differences). Relative difference: |a - b| / (a + b).

``check`` reads <root>/ball.parquet when present, else runs the ball stage in memory. ``fit``
rebuilds the v3 ball in memory (bench.noise.v3_ball, same seeds) for each sigma and order, with the
players of the <game>_legacy2<level> root (v2 and v3 players are identical), and writes
artifacts/bench/_smoothness/fit.json. ``check`` writes artifacts/bench/_smoothness/<root>.json.
Bench roots only (require_bench_root).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from bench import metrica_io, noise
from pipeline.ball import clean, contact as contact_mod
from pipeline.diagnostics import touch_funnel as TF
from pipeline.io.roots import require_bench_root

ART = os.path.join(metrica_io.ROOT, "artifacts", "bench")
OUT = os.path.join(ART, "_smoothness")
MATCH_REPORT = os.path.join(metrica_io.ROOT, "reports", "touch_funnel.match.json")
GAME_STEMS = ("metrica_g1", "metrica_g2", "metrica_g3")
LADDER = ("s3", "s4", "s6", "s8")
GROUPS = ("d2_autocorr", "kink_width", "kink_ratio", "ladder")


def _rel(a, b):
    if a is None or b is None:
        return 1.0
    a, b = float(a), float(b)
    return 0.0 if a + b == 0 else abs(a - b) / abs(a + b)


def _tv(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.sum() == 0 or b.sum() == 0:
        return 1.0
    return 0.5 * float(np.abs(a / a.sum() - b / b.sum()).sum())


def shape(st):
    """The smoothness subset of a touch_funnel result (no touch rates)."""
    return {"raw_d2_lag1_autocorr": st["raw_d2_lag1_autocorr"],
            "kink_width_median": st["kink_width_median"],
            "kink_width_hist_share": st["kink_width_hist_share"],
            "kink_short_over_long_dv_median": st["kink_short_over_long_dv_median"],
            "acc_raw_median": st["acc_raw_median"],
            "dv_median_valid": st["dv_median_valid"],
            **{"ladder_" + k: st["ladder_peaks_per_min"][k] for k in LADDER}}


def distance(m, b):
    """(distance, {group: value}) between two shape dicts; m is the match."""
    ac = None if b["raw_d2_lag1_autocorr"] is None else 1 - b["raw_d2_lag1_autocorr"]
    g = {"d2_autocorr": _rel(1 - m["raw_d2_lag1_autocorr"], ac),
         "kink_width": 0.5 * (_rel(m["kink_width_median"], b["kink_width_median"])
                              + _tv(m["kink_width_hist_share"], b["kink_width_hist_share"])),
         "kink_ratio": _rel(m["kink_short_over_long_dv_median"],
                            b["kink_short_over_long_dv_median"]),
         "ladder": float(np.mean([_rel(m["ladder_" + k], b["ladder_" + k]) for k in LADDER]))}
    return float(np.mean([g[k] for k in GROUPS])), g


def match_shape():
    with open(MATCH_REPORT) as fh:
        return shape(json.load(fh))


def ball_stage(raw, players, meta):
    """pipeline.ball.run on an in-memory ball_raw (nothing written)."""
    n = int(meta["n_frames"])
    bx = np.full(n, np.nan)
    by = np.full(n, np.nan)
    f = raw["frame"].to_numpy()
    bx[f] = raw["x_m"].to_numpy(dtype=float)
    by[f] = raw["y_m"].to_numpy(dtype=float)
    near = contact_mod.contact(players, bx, by, n).d_any
    ball, _ = clean.clean_ball(raw, meta, near_player_m=near)
    return ball


def root_shape(root):
    root = require_bench_root(root)
    with open(os.path.join(root, "meta.json")) as fh:
        meta = json.load(fh)
    raw = pd.read_parquet(os.path.join(root, "ball_raw.parquet"))
    players = pd.read_parquet(os.path.join(root, "players.parquet"),
                              columns=["frame", "team", "x_m", "y_m", "vx", "vy"])
    bp = os.path.join(root, "ball.parquet")
    ball = pd.read_parquet(bp) if os.path.isfile(bp) else ball_stage(raw, players, meta)
    return shape(TF.compute(meta, ball, raw, players))


def _raw_frame(ball, pitch):
    L, W = pitch
    df = pd.DataFrame({"frame": np.arange(len(ball), dtype=np.int32),
                       "x_m": (ball[:, 0] + L / 2.0).astype(np.float32),
                       "y_m": (W / 2.0 - ball[:, 1]).astype(np.float32)})
    df["seen"] = np.isfinite(df["x_m"].to_numpy())
    return df


def candidate_shape(game, game_id, level, sigma, order, players, meta):
    """Shape of the v3 ball at (sigma, order), built in memory with the root's seed."""
    src = os.path.join(ART, game_id)
    labels = pd.read_csv(os.path.join(src, "truth.csv"))["label"].astype(str).to_numpy()
    stops = pd.read_csv(os.path.join(src, "stoppages.csv"))
    rng = np.random.default_rng(noise._seed(game_id, level))
    b3, _, _ = noise.v3_ball(game, labels, stops, level, rng, sigma, order)
    raw = _raw_frame(b3, game.pitch)
    return shape(TF.compute(meta, ball_stage(raw, players, meta), raw, players))


def fit(levels, sigmas, orders, games=GAME_STEMS):
    m = match_shape()
    res = {}
    for gid in games:
        game = metrica_io.load_game(gid)
        for level in levels:
            # Players and meta of the v2 root (players are identical in v2 and v3).
            root = require_bench_root(os.path.join(ART, "{}_legacy2{}".format(gid, level)))
            with open(os.path.join(root, "meta.json")) as fh:
                meta = json.load(fh)
            players = pd.read_parquet(os.path.join(root, "players.parquet"),
                                      columns=["frame", "team", "x_m", "y_m", "vx", "vy"])
            for order in orders:
                for s in sigmas:
                    sh = candidate_shape(game, gid, level, s, order, players, meta)
                    d, grp = distance(m, sh)
                    key = "{}|{}|{}|{:g}".format(gid, level, order, s)
                    res[key] = {"distance": d, "groups": grp, "shape": sh}
                    print("{:<12}{:<7}{:<12}{:>5g}  dist {:.3f}  ".format(gid, level, order, s, d)
                          + "  ".join("{} {:.3f}".format(k, grp[k]) for k in GROUPS), flush=True)
    # Pooled: mean distance over games per (level, order, sigma).
    pooled = {}
    for level in levels:
        for order in orders:
            for s in sigmas:
                ds = [res["{}|{}|{}|{:g}".format(g, level, order, s)]["distance"] for g in games]
                pooled["{}|{}|{:g}".format(level, order, s)] = float(np.mean(ds))
    print("\npooled (mean over games)")
    for level in levels:
        for order in orders:
            row = {s: pooled["{}|{}|{:g}".format(level, order, s)] for s in sigmas}
            best = min(row, key=row.get)
            print("{:<7}{:<12}".format(level, order) + " ".join(
                "{:g}:{:.3f}".format(s, v) for s, v in row.items()) + "   best {:g}".format(best))
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "fit.json")
    old = json.load(open(path)) if os.path.isfile(path) else {}
    old.setdefault("per_root", {}).update(res)
    old.setdefault("pooled", {}).update(pooled)
    old["match"] = m
    with open(path, "w") as fh:
        json.dump(old, fh, indent=1)
    return res, pooled


def check(roots):
    m = match_shape()
    keys = ["raw_d2_lag1_autocorr", "kink_width_median", "kink_short_over_long_dv_median",
            "acc_raw_median", "dv_median_valid"] + ["ladder_" + k for k in LADDER]
    short = ["d2ac", "width", "3/7", "accMed", "dvMed", "s3", "s4", "s6", "s8"]
    print("{:<28}".format("root") + "".join("{:>8}".format(s) for s in short) + "{:>8}".format("dist"))
    print("{:<28}".format("MATCH") + "".join("{:>8.3f}".format(m[k]) for k in keys))
    os.makedirs(OUT, exist_ok=True)
    out = {}
    for r in roots:
        path = r if os.path.isabs(r) else os.path.join(ART, r)
        sh = root_shape(path)
        d, grp = distance(m, sh)
        name = os.path.basename(os.path.normpath(path))
        out[name] = {"distance": d, "groups": grp, "shape": sh}
        print("{:<28}".format(name) + "".join(
            "{:>8.3f}".format(sh[k]) if sh[k] is not None else "{:>8}".format("-") for k in keys)
            + "{:>8.3f}".format(d), flush=True)
        with open(os.path.join(OUT, name + ".json"), "w") as fh:
            json.dump(out[name], fh, indent=1)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--roots", nargs="*",
                   default=["{}_{}".format(g, l) for l in ("low", "match", "high")
                            for g in GAME_STEMS])
    f = sub.add_parser("fit")
    f.add_argument("--levels", nargs="*", default=["low", "match", "high"])
    f.add_argument("--sigmas", nargs="*", type=float, default=[6, 7, 7.5, 8, 8.5, 9, 10])
    f.add_argument("--orders", nargs="*", default=["before_gaps", "after_gaps"])
    f.add_argument("--games", nargs="*", default=list(GAME_STEMS))
    args = ap.parse_args(argv)
    if args.cmd == "check":
        check(args.roots)
    else:
        fit(args.levels, args.sigmas, args.orders, args.games)


if __name__ == "__main__":
    main()
