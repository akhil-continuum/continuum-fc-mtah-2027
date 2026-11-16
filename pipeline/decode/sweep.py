"""One-at-a-time dial sweeps on pooled bench games (bench only; never the match, never the
public window).

    uv run python -m pipeline.decode.sweep --version v2 --probs synth_noiseA --level low
                                           [--truth default] [--set c_switch=8 ...]
                                           [--dials c_switch min_poss] [--bootstrap 0]

For every dial in the version's grid (GRIDS), every other dial is held at the version's value
(plus --set overrides) and the dial takes each grid value. Every row is scored pooled over the
three games with bench.score.score_pair (the official metric code). Prints the sensitivity table
and writes artifacts/bench/_scores/sweep_<version>_<probs>_<level>_<truth>.json.
"""
import argparse
import json
import os
import re
import sys

import numpy as np

from bench import score as bscore
from pipeline.decode import decoders
from pipeline.decode.__main__ import BENCH, GAMES, bench_inputs
from pipeline.io.roots import require_bench_roots

GRIDS = {
    "v1": {"th_enter": [0.4, 0.5, 0.6, 0.7], "th_exit": [0.3, 0.4, 0.5, 0.6],
           "k_poss": [4, 8, 12, 19, 25, 37, 50, 75], "k_dead": [25, 50, 75, 100, 150]},
    "v2": {"c_switch": [3, 6, 10, 15, 22, 30, 45, 65], "c_stop": [4, 8, 12, 18, 27],
           "c_restart": [4, 8, 12, 18, 27], "min_poss": [1, 4, 8, 12, 19, 25, 37],
           "min_dead": [12, 25, 50, 75, 100, 150], "w_restart": [0.0, 1.0, 2.0]},
}
GRIDS["v2h"] = GRIDS["v2"]
GRIDS["v1r"] = GRIDS["v1"]
GRIDS["v2r"] = GRIDS["v2hr"] = GRIDS["v2"]
GRIDS["v2s"] = dict(GRIDS["v2"], b_restart=[0.0, 5.0, 10.0, 20.0, 40.0],
                    restart_width=[4, 8, 12, 25, 50])
GRIDS["v3a"] = {"c_base": [18.0, 22.0, 26.0, 30.0, 40.0, 55.0, 75.0],
                "b_ctrl": [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0],
                "g_two": [0.0, 0.5, 0.8, 1.0], "g_one": [0.0, 0.25, 0.5],
                "ctrl_width": [4, 8, 12, 25, 50], "c_switch": [10.0, 14.0, 18.0, 22.0, 30.0],
                "c_stop": [30.0, 45.0, 60.0], "n_full": [2, 3, 4, 5, 6, 8, 12]}
COLS = ("spell_f1_mean", "spell_bootstrap_low", "spell_bootstrap_high", "macro_f1",
        "short_recall_1_5", "short_recall_4", "false_splits_per_hour", "missed_stoppages",
        "invented_stoppages", "restart_team_acc", "pred_poss_per_min")


def _parse_set(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            out[k] = float(v) if "." in v else int(v)
        except ValueError:
            out[k] = v
    return out


def pooled(cache, version, dials, truth, bootstrap):
    gts, preds = [], []
    for root, inp, truths in cache:
        codes, _ = decoders.decode(version, inp, dials)
        gt = truths[truth]
        sep = np.array([bscore.pm.X], dtype=gt.dtype)
        gts += [gt, sep]
        preds += [codes.astype(gt.dtype), sep]
    return bscore.score_pair(np.concatenate(gts), np.concatenate(preds), bootstrap)


def truth_path(root, truth):
    """<root>/<truth file>, else artifacts/bench/metrica_g<k>/<truth file> for a root whose name
    contains g<k> (dev roots carry no truth of their own)."""
    name = bscore.TRUTH_FILES[truth]
    own = os.path.join(root, name)
    if os.path.isfile(own):
        return own
    m = re.search(r"g(\d)", os.path.basename(os.path.normpath(root)))
    if not m:
        raise FileNotFoundError("no truth for root {}".format(os.path.basename(root)))
    return os.path.join(BENCH, "metrica_g{}".format(m.group(1)), name)


def load_cache(probs, roots, truths, compute="kickoff"):
    cache = []
    for root in roots:
        t = {k: bscore.read_codes(truth_path(root, k)) for k in truths}
        cache.append((root, bench_inputs(root, probs, compute), t))
    return cache


def fmt_row(label, r):
    f = lambda v, s="{:.3f}": "-" if v is None else s.format(v)   # noqa: E731
    return "{:<22}{:>7} [{}-{}]{:>7}{:>7}{:>7}{:>7}{:>5}{:>5}{:>7}{:>6}".format(
        label, f(r["spell_f1_mean"]), f(r["spell_bootstrap_low"]), f(r["spell_bootstrap_high"]),
        f(r["macro_f1"]), f(r["short_recall_1_5"]), f(r["short_recall_4"]),
        f(r["false_splits_per_hour"], "{:.0f}"), r["missed_stoppages"], r["invented_stoppages"],
        f(r["restart_team_acc"]), f(r["pred_poss_per_min"], "{:.2f}"))


def main(argv=None):
    ap = argparse.ArgumentParser(description="decoder dial sweep on the bench")
    ap.add_argument("--version", required=True, choices=sorted(GRIDS))
    ap.add_argument("--probs", default="rule_v1")
    ap.add_argument("--level", default="low")
    ap.add_argument("--truth", nargs="*", default=["default"])
    ap.add_argument("--games", nargs="*", default=list(GAMES))
    ap.add_argument("--root", nargs="*", help="explicit bench roots instead of <game>_<level>")
    ap.add_argument("--set", nargs="*", help="dial=value overrides of the base point")
    ap.add_argument("--dials", nargs="*", help="sweep only these dials")
    ap.add_argument("--bootstrap", type=int, default=0)
    args = ap.parse_args(argv)

    base = dict(decoders.VERSIONS[args.version]["dials"])
    base.update(_parse_set(args.set))
    roots = args.root or [os.path.join(BENCH, "{}_{}".format(g, args.level)) for g in args.games]
    require_bench_roots(roots)
    cache = load_cache(args.probs, roots, args.truth,
                       decoders.VERSIONS[args.version].get("compute", "kickoff"))
    out = {"version": args.version, "probs": args.probs, "level": args.level, "base": base,
           "roots": [os.path.basename(os.path.normpath(r)) for r in roots],
           "rows": []}
    head = "{:<22}{:>7} {:^13}{:>7}{:>7}{:>7}{:>7}{:>5}{:>5}{:>7}{:>6}".format(
        "dial=value", "spF1", "[5-95]", "macro", "sh1.5", "sh4", "spl/h", "miss", "inv",
        "rstT", "p/min")
    for truth in args.truth:
        print("version {} probs {} level {} truth {} base {}".format(
            args.version, args.probs, args.level, truth, base))
        print(head)
        r = pooled(cache, args.version, base, truth, args.bootstrap)
        print(fmt_row("base", r))
        out["rows"].append({"truth": truth, "dial": "base", "value": None,
                            **{c: r[c] for c in COLS}})
        for dial, grid in GRIDS[args.version].items():
            if args.dials and dial not in args.dials:
                continue
            for v in grid:
                if v == base[dial]:
                    continue
                r = pooled(cache, args.version, dict(base, **{dial: v}), truth, args.bootstrap)
                print(fmt_row("{}={}".format(dial, v), r))
                out["rows"].append({"truth": truth, "dial": dial, "value": v,
                                    **{c: r[c] for c in COLS}})
            sys.stdout.flush()
    path = os.path.join(BENCH, "_scores", "sweep_{}_{}_{}_{}.json".format(
        args.version, args.probs, args.level, "-".join(args.truth)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
