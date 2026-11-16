"""Bench-only report for T-32: score stored label files on the nine smoothed level roots.

    uv run python -m pipeline.models.t32_report --versions v11g v11e [--base v6w]
                                                [--compare eval1_S1g.json] [--out name.json]

For every level (low, match, high) and truth (labels, default, confirmed): pooled and per-game
spellF1 (official metric code via bench.score), the paired block bootstrap of the difference to
the base version (fit_t32.paired_diff: bench/paired.py scheme B), the judge's error
counts (paired.error_counts) and bench.score's extra columns (invented stoppages, restart-team
accuracy, possessions per minute, predicted and true). Reads labels/<v>.csv and truth*.csv of
artifacts/bench/metrica_g{1,2,3}_<level> only; never the match. Optional --compare checks the
labels/default pooled and per-game numbers against an artifacts/bench/_t32/eval*.json.
"""
import argparse
import json
import os

import numpy as np

from pipeline.io.roots import require_bench_roots
from pipeline.models import fit_t32 as F
from pipeline.models import fit_team as FT

TRUTHS3 = ("labels", "default", "confirmed")
ERR_KEYS = ("missed_short_1_5", "missed_short_4", "switch_absent_50", "late_switch_gt10",
            "false_split", "invented_stoppage", "missed_stoppage", "wrong_team_after_restart")


def label_codes(game, level, version):
    path = os.path.join(FT.root_of(game, level), "labels", "{}.csv".format(version))
    lab = FT.truth_codes.__globals__["pd"].read_csv(path, dtype={"label": str})["label"]
    lab = lab.to_numpy().astype(str)
    return np.select([lab == c for c in "HADX"], [F.H, F.A, F.D, F.X], -1).astype(np.int8)


def cell(preds, base, gts):
    from bench import score as bscore
    block = int(F.BLOCK_S * 25)
    s = FT.score_set(preds, gts)
    tabs = {g: F.PJ.game_table(gts[g].astype(np.int64), preds[g].astype(np.int64), block)
            for g in F.GAMES}
    errs = {g: F.PJ.error_counts(gts[g].astype(np.int64), tabs[g]) for g in F.GAMES}
    out = {"pooled": s["pooled"]["spell_f1_mean"], "games": {g: s[g] for g in F.GAMES},
           "errors": {k: int(sum(errs[g][k] for g in F.GAMES)) for k in ERR_KEYS}}
    gl, pl = [], []
    for g in F.GAMES:
        sep = np.array([bscore.pm.X], dtype=gts[g].dtype)
        gl += [gts[g], sep]
        pl += [preds[g].astype(gts[g].dtype), sep]
    ex = bscore.extra_metrics(np.concatenate(gl), np.concatenate(pl))
    out.update({k: (round(float(ex[k]), 4) if ex[k] is not None else None) for k in
                ("pred_poss_per_min", "gt_poss_per_min", "restart_team_acc",
                 "false_splits_per_hour", "short_recall_4", "short_recall_1_5")})
    if base is not None:
        btabs = {g: F.PJ.game_table(gts[g].astype(np.int64), base[g].astype(np.int64), block)
                 for g in F.GAMES}
        out["paired"] = F.paired_diff(tabs, btabs)
    return out


def report(versions, base="v6w", truths=TRUTHS3):
    require_bench_roots([FT.root_of(g, lv) for g in F.GAMES for lv in F.LEVELS])
    res = {}
    for lv in F.LEVELS:
        gts = {t: {g: FT.truth_codes(FT.root_of(g, lv), t) for g in F.GAMES} for t in truths}
        bp = {g: label_codes(g, lv, base) for g in F.GAMES}
        for v in [base] + list(versions):
            pr = {g: label_codes(g, lv, v) for g in F.GAMES}
            for t in truths:
                c = cell(pr, None if v == base else bp, gts[t])
                res.setdefault(v, {}).setdefault(lv, {})[t] = c
                txt = "{:<8}{:<6}{:<10}{:.4f} [{}]".format(
                    v, lv, t, c["pooled"], " ".join("%.3f" % c["games"][g] for g in F.GAMES))
                if "paired" in c:
                    p = c["paired"]
                    txt += " d {:+.4f} [{:+.4f},{:+.4f}] ({})".format(
                        p["diff"], p["lo"], p["hi"],
                        " ".join("%+.3f" % p["games"][g]["diff"] for g in F.GAMES))
                e = c["errors"]
                txt += " | pm {:.2f}/{:.2f} m1.5 {} m4 {} none2s {} late {} fs {} inv {} " \
                       "wrongR {} rst {:.3f}".format(
                           c["pred_poss_per_min"], c["gt_poss_per_min"], e["missed_short_1_5"],
                           e["missed_short_4"], e["switch_absent_50"], e["late_switch_gt10"],
                           e["false_split"], e["invented_stoppage"],
                           e["wrong_team_after_restart"], c["restart_team_acc"] or 0.0)
                print(txt, flush=True)
    return res


def compare(res, version, path):
    with open(os.path.join(F.CACHE, path)) as fh:
        ev = json.load(fh)
    worst = 0.0
    for lv in F.LEVELS:
        for t in ("labels", "default"):
            a, b = res[version][lv][t], ev[lv][t]
            worst = max(worst, abs(a["pooled"] - b["pooled"]),
                        *(abs(a["games"][g] - b["games"][g]) for g in F.GAMES))
            if t == "labels":
                for k in ("false_split", "missed_short_1_5", "missed_short_4",
                          "late_switch_gt10", "switch_absent_50"):
                    if a["errors"][k] != b["errors"][k]:
                        print("  differs: {} {} {} chain {} eval {}".format(
                            lv, t, k, a["errors"][k], b["errors"][k]))
                        worst = max(worst, 1.0)
    print("compare {} with {}: max abs difference {:.6f} -> {}".format(
        version, path, worst, "REPRODUCED" if worst < 1e-9 else "DIFFERS"))
    return worst


def main(argv=None):
    ap = argparse.ArgumentParser(description="T-32 bench report over stored label files")
    ap.add_argument("--versions", nargs="+", required=True)
    ap.add_argument("--base", default="v6w")
    ap.add_argument("--compare", nargs="*", default=[],
                    help="pairs version=eval file under artifacts/bench/_t32/")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    res = report(args.versions, args.base)
    for pair in args.compare:
        v, path = pair.split("=")
        compare(res, v, path)
    if args.out:
        F.save(args.out, res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
