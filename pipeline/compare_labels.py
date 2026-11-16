"""Compare two versions' label files: whole-file aggregates only (T-44, the match gates of v11g).

    uv run python -m pipeline.compare_labels match --a v6w --b v11g
    uv run python -m pipeline.compare_labels bench --a v6w --b v11g [--levels low match high]
                                             [--game metrica_g1 ...] [--root DIR ...]
    make compare-labels A=v6w B=v11g PART=match|bench [LEVELS="low match high"]

Reads ``<root>/labels/<a>.csv`` and ``<root>/labels/<b>.csv`` (the match root, or the bench level
roots) and the root's meta.json for the graded frames. Nothing is decoded or written except the
report. The same function computes the numbers on the bench and on the match.

Counting conventions are the ones check_labels and the decode stage's match report already use:
pipeline.decode.checks.statistics on the graded spans (inputs.graded_mask: meta ``graded_spans``,
else ``halves``). A possession is a maximal H or A run inside one graded span, a stoppage a
maximal D run; that is the official scorer's possession_spells / segments definition. On bench
roots the official scorer's own count (evaluation/possession_metrics on the truth's graded runs)
is printed next to it as a cross-check.

Printed, through summary() only: possessions per version and the ratio b/a, possessions per
minute, share of D, H and A per version and the D difference, frame agreement on graded frames
(and how the disagreements split: team flip, live against dead), stoppages per version. On the
match also each label file's provenance status, and whether the round-five gates hold (v11g's
possession count 1.11 to 1.26 times v6w's, share of D within .003). Nothing
per instance: no frames, intervals, shortest or longest spell.
"""
import argparse
import os
import sys

import numpy as np

from pipeline.io.roots import require_bench_root
from pipeline.io.sources import ARTIFACTS, match_root
from pipeline.report import match_entry, quiet_warnings, summary

GAMES = ("metrica_g1", "metrica_g2", "metrica_g3")
LEVELS = ("low", "match", "high")
H, A, D, X = 0, 1, 2, 3
# Round-five match gates (set by the judge): from bench behaviour, not the match.
GATE_RATIO = (1.11, 1.26)
GATE_SHARE_D_TOL = 0.003
ROUND = 4


def _r(value):
    return None if value is None else round(float(value), ROUND)


def counts(codes, spans):
    """Aggregates of one label array on the graded spans, by checks.statistics (no instances)."""
    from pipeline.decode import checks
    stats = checks.statistics(np.asarray(codes), spans)
    graded = np.zeros(len(codes), dtype=bool)
    for a, b in spans:
        graded[a:b] = True
    g = np.asarray(codes)[graded]
    n = max(len(g), 1)
    return {"graded_frames": int(len(g)), "graded_minutes": stats["graded_minutes"],
            "possessions": stats["possessions"], "stoppages": stats["stoppages"],
            "possessions_per_min": stats["possessions_per_min"],
            "share_h": float(np.sum(g == H) / n), "share_a": float(np.sum(g == A) / n),
            "share_d": stats["share_d"], "x_on_graded": int(np.sum(g == X))}


def compare(codes_a, codes_b, spans):
    """(stats a, stats b, joint) for two label arrays of equal length on the graded spans."""
    codes_a, codes_b = np.asarray(codes_a), np.asarray(codes_b)
    if len(codes_a) != len(codes_b):
        raise ValueError("label files differ in length")
    sa, sb = counts(codes_a, spans), counts(codes_b, spans)
    graded = np.zeros(len(codes_a), dtype=bool)
    for a, b in spans:
        graded[a:b] = True
    ga, gb = codes_a[graded], codes_b[graded]
    n = max(len(ga), 1)
    live_a, live_b = np.isin(ga, (H, A)), np.isin(gb, (H, A))
    joint = {"agree": int(np.sum(ga == gb)),
             "team_flip": int(np.sum(live_a & live_b & (ga != gb))),
             "live_vs_dead": int(np.sum(live_a != live_b)),
             "frames": int(len(ga))}
    joint["agreement"] = joint["agree"] / n
    return sa, sb, joint


def gate_verdict(sa, sb, ratio_range=GATE_RATIO, tol=GATE_SHARE_D_TOL):
    """(ratio b/a, share-D difference b - a, ratio ok, share-D ok)."""
    ratio = sb["possessions"] / sa["possessions"] if sa["possessions"] else None
    diff = None if sa["share_d"] is None or sb["share_d"] is None else sb["share_d"] - sa["share_d"]
    eps = 1e-9      # edges inclusive, without float noise (0.403 - 0.400 > 0.003 in binary)
    ok_ratio = ratio is not None and ratio_range[0] - eps <= ratio <= ratio_range[1] + eps
    ok_d = diff is not None and abs(diff) <= tol + eps
    return ratio, diff, ok_ratio, ok_d


def report_fields(name_a, name_b, sa, sb, joint):
    """The printed aggregates (keys short, values scalars)."""
    ratio, diff, ok_ratio, ok_d = gate_verdict(sa, sb)
    n = max(joint["frames"], 1)
    return {
        "version_a": name_a, "version_b": name_b,
        "graded_minutes": sa["graded_minutes"],
        "possessions_a": sa["possessions"], "possessions_b": sb["possessions"],
        "possession_ratio_b_over_a": _r(ratio),
        "possessions_per_min_a": _r(sa["possessions_per_min"]),
        "possessions_per_min_b": _r(sb["possessions_per_min"]),
        "share_d_a": _r(sa["share_d"]), "share_d_b": _r(sb["share_d"]),
        "share_d_diff_b_minus_a": _r(diff),
        "share_h_a": _r(sa["share_h"]), "share_h_b": _r(sb["share_h"]),
        "share_a_a": _r(sa["share_a"]), "share_a_b": _r(sb["share_a"]),
        "agreement_graded": _r(joint["agreement"]),
        "disagree_team_flip_share": _r(joint["team_flip"] / n),
        "disagree_live_vs_dead_share": _r(joint["live_vs_dead"] / n),
        "stoppages_a": sa["stoppages"], "stoppages_b": sb["stoppages"],
        "x_on_graded_a": sa["x_on_graded"], "x_on_graded_b": sb["x_on_graded"],
        "gate_ratio_range": list(GATE_RATIO), "gate_share_d_tol": GATE_SHARE_D_TOL,
        "gate_ratio_ok": ok_ratio, "gate_share_d_ok": ok_d,
    }


def pool(stats_list):
    """Pooled counts over several roots (sums; rates recomputed from the sums)."""
    out = {k: sum(s[k] for s in stats_list)
           for k in ("graded_frames", "possessions", "stoppages", "x_on_graded")}
    f = max(out["graded_frames"], 1)
    out["graded_minutes"] = round(out["graded_frames"] / 25.0 / 60.0, 2)
    out["possessions_per_min"] = out["possessions"] / (out["graded_frames"] / 25.0 / 60.0) \
        if out["graded_frames"] else None
    for k in ("share_h", "share_a", "share_d"):
        out[k] = sum(s[k] * s["graded_frames"] for s in stats_list) / f
    return out


def pool_joint(joints):
    out = {k: sum(j[k] for j in joints) for k in ("agree", "team_flip", "live_vs_dead", "frames")}
    out["agreement"] = out["agree"] / max(out["frames"], 1)
    return out


def load_pair(root, name_a, name_b):
    """(codes a, codes b, graded spans) for a root; errors name no path and no value."""
    from pipeline.check_labels import read_codes
    from pipeline.decode import inputs
    meta = inputs.read_meta(root)
    n = int(meta["n_frames"])
    codes = []
    for name in (name_a, name_b):
        path = os.path.join(root, "labels", "{}.csv".format(name))
        if not os.path.isfile(path):
            raise FileNotFoundError("labels file for version {} not found".format(name))
        c = read_codes(path)
        if len(c) != n:
            raise ValueError("labels file for version {} has the wrong row count".format(name))
        codes.append(c)
    _graded, spans = inputs.graded_mask(meta, n)
    return codes[0], codes[1], spans


@match_entry
def run_match(name_a, name_b):
    quiet_warnings()
    from pipeline.io import provenance
    from pipeline.submit import version_expect
    root = match_root()
    ca, cb, spans = load_pair(root, name_a, name_b)
    sa, sb, joint = compare(ca, cb, spans)
    out = report_fields(name_a, name_b, sa, sb, joint)
    for tag, name in (("a", name_a), ("b", name_b)):
        path = os.path.join(root, "labels", "{}.csv".format(name))
        try:
            _ok, reason = provenance.check_stamp(path, expect=version_expect(name))
        except Exception as error:                      # noqa: BLE001 - report, do not stop
            reason = "stamp check failed: {}".format(type(error).__name__)
        out["provenance_{}".format(tag)] = str(reason)[:80]
    summary(_name="compare_labels", **out)
    return 0


def official_possessions(root, codes):
    """Bench only: the official scorer's spell count, on the truth's graded runs (cross-check)."""
    from bench.score import read_codes as truth_read
    import possession_metrics as pm  # noqa: E402 - bench.score puts evaluation/ on the path
    gt = truth_read(os.path.join(root, "truth.csv"))
    runs = pm.valid_runs(gt)
    return len(pm.possession_spells(pm.fill_abstentions(np.asarray(codes), runs), runs))


def run_bench(name_a, name_b, levels, games=GAMES, roots=None):
    groups = [("roots", roots)] if roots else \
        [(lv, [os.path.join(ARTIFACTS, "bench", "{}_{}".format(g, lv)) for g in games])
         for lv in levels]
    every_a, every_b, every_j = [], [], []
    for label, group in groups:
        st_a, st_b, js = [], [], []
        off = [0, 0]
        for root in group:
            root = require_bench_root(root)
            ca, cb, spans = load_pair(root, name_a, name_b)
            sa, sb, joint = compare(ca, cb, spans)
            st_a.append(sa), st_b.append(sb), js.append(joint)
            tag = os.path.basename(os.path.normpath(root))
            fields = report_fields(name_a, name_b, sa, sb, joint)
            if os.path.isfile(os.path.join(root, "truth.csv")):
                oa, ob = official_possessions(root, ca), official_possessions(root, cb)
                off = [off[0] + oa, off[1] + ob]
                fields["official_possessions_a"], fields["official_possessions_b"] = oa, ob
            summary(_name="compare_labels.bench.{}".format(tag), **fields)
        pa, pb, pj = pool(st_a), pool(st_b), pool_joint(js)
        fields = report_fields(name_a, name_b, pa, pb, pj)
        fields.update(roots=len(group), official_possessions_a=off[0],
                      official_possessions_b=off[1])
        summary(_name="compare_labels.bench.{}".format(label), **fields)
        every_a += st_a
        every_b += st_b
        every_j += js
    if len(groups) > 1:
        fields = report_fields(name_a, name_b, pool(every_a), pool(every_b), pool_joint(every_j))
        summary(_name="compare_labels.bench.all", roots=len(every_a), **fields)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="compare two label files (aggregates only)")
    ap.add_argument("target", choices=("match", "bench"))
    ap.add_argument("--a", required=True, help="reference version (labels/<a>.csv)")
    ap.add_argument("--b", required=True, help="compared version (labels/<b>.csv)")
    ap.add_argument("--levels", nargs="+", default=list(LEVELS))
    ap.add_argument("--game", nargs="*", default=list(GAMES))
    ap.add_argument("--root", nargs="*", help="bench: explicit roots")
    args = ap.parse_args(argv)
    for name in (args.a, args.b):
        if not name or "/" in name or len(name) > 40:
            ap.error("versions are short names")
    if args.target == "match":
        if args.root:
            ap.error("the match takes only --a and --b")
        return run_match(args.a, args.b)
    return run_bench(args.a, args.b, args.levels, args.game, args.root)


if __name__ == "__main__":
    sys.exit(main())
