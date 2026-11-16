"""Score a version on the bench with the official metric code.

    uv run python -m bench.score --version v0 [--level match]
                                 [--truth default|first_touch|confirmed|labels]
                                 [--games metrica_g1 ...] [--bootstrap 1000]

Reads <root>/labels/<version>.csv and <root>/truth*.csv for every bench root
artifacts/bench/<game>_<level>/. Every metric comes from evaluation/possession_metrics.py
(imported, not re-implemented); the extra columns of the team's report are built from its
functions (valid_runs, fill_abstentions, segments, possession_spells, spell_overlaps,
matched_gt_flags). Games are pooled by concatenating them with an X frame between games, so the
pooled spellF1 and its bootstrap resample possessions across all games.

Extra columns, defined here once:
- short_recall_1_5 / short_recall_4: share of true possessions shorter than 1.5 s / 4 s that are
  matched at IoU 0.5 (official greedy matching).
- false_splits_per_hour: predicted label changes strictly inside a true possession and more than
  25 frames from every true label change, per hour of graded time.
- missed_stoppages: true D runs with less than half their frames predicted D.
- invented_stoppages: predicted D runs with less than half their frames true D.
- restart_team_acc: over true stoppages followed by a possession, the share where the first non-D
  predicted label in [restart, restart + 50 frames) is the true restart team.
Writes artifacts/bench/_scores/<version>_<level>_<truth>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from bench import metrica_io

sys.path.insert(0, os.path.join(metrica_io.ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402  official scorer

ART = os.path.join(metrica_io.ROOT, "artifacts", "bench")
GAME_STEMS = ("metrica_g1", "metrica_g2", "metrica_g3")
TRUTH_FILES = {"default": "truth.csv", "first_touch": "truth_first_touch.csv",
               "confirmed": "truth_confirmed.csv",
               # organisers' label conventions C1-C9 (bench.truth_labels, T-26)
               "labels": "truth_labels.csv"}


def read_codes(path):
    return pm.label_codes(pm.read_label_csv(path))


def extra_metrics(gt, pred):
    """The team's report columns not in pm.evaluate. gt, pred: code arrays (pred may hold X)."""
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    graded_frames = int(sum(b - a for a, b in runs))
    hours = graded_frames / pm.FPS / 3600.0
    minutes = hours * 60.0

    gt_sp = pm.possession_spells(gt, runs)
    pr_sp = pm.possession_spells(filled, runs)
    flags = pm.matched_gt_flags(pm.spell_overlaps(gt_sp, pr_sp), len(gt_sp), 0.5)
    lens = np.array([b - a for a, b, _ in gt_sp])

    def recall_under(sec):
        sel = lens < sec * pm.FPS
        return (float(flags[sel].mean()) if sel.any() else None), int(sel.sum())

    r15, n15 = recall_under(1.5)
    r4, n4 = recall_under(4.0)

    gt_tr = np.array([f for f, _, _ in pm.transitions(gt, runs)])
    pr_tr = [f for f, _, _ in pm.transitions(filled, runs)]
    splits = 0
    for f in pr_tr:
        inside = any(a < f < b for a, b, _ in gt_sp)
        far = not len(gt_tr) or np.min(np.abs(gt_tr - f)) > 25
        splits += int(inside and far)

    gt_seg = pm.segments(gt, runs)
    pr_seg = pm.segments(filled, runs)
    gt_d = [(a, b) for a, b, k in gt_seg if k == pm.D]
    pr_d = [(a, b) for a, b, k in pr_seg if k == pm.D]
    missed = sum(1 for a, b in gt_d if (filled[a:b] == pm.D).mean() < 0.5)
    invented = sum(1 for a, b in pr_d if (gt[a:b] == pm.D).mean() < 0.5)

    correct = total = 0
    for a, b in gt_d:
        if b >= len(gt) or gt[b] not in (pm.H, pm.A):
            continue
        window = filled[b:b + 50]
        live = window[window != pm.D]
        total += 1
        correct += int(len(live) > 0 and live[0] == gt[b])

    def share_d(codes):
        g = gt != pm.X
        return float((codes[g] == pm.D).mean()) if g.any() else None

    return {
        "short_recall_1_5": r15, "n_short_1_5": n15,
        "short_recall_4": r4, "n_short_4": n4,
        "false_splits_per_hour": splits / hours if hours else None,
        "missed_stoppages": missed, "invented_stoppages": invented,
        "n_gt_stoppages": len(gt_d), "n_pred_stoppages": len(pr_d),
        "restart_team_acc": correct / total if total else None, "n_restarts": total,
        "pred_poss_per_min": len(pr_sp) / minutes if minutes else None,
        "gt_poss_per_min": len(gt_sp) / minutes if minutes else None,
        "pred_share_D": share_d(filled), "gt_share_D": share_d(gt),
        "graded_minutes": minutes,
    }


def score_pair(gt, pred, bootstrap_n):
    res = pm.evaluate(gt, pred, bootstrap_n=bootstrap_n)
    keep = ("spell_f1_mean", "spell_bootstrap_low", "spell_bootstrap_high", "macro_f1",
            "alive_dead_f1", "team_acc_gt_alive", "team_acc_both_alive", "n_gt_spells",
            "n_pred_spells", "fragmentation")
    out = {k: res[k] for k in keep}
    out.update(extra_metrics(gt, pred))
    return out


def score_version(version, level="match", truth="default", games=GAME_STEMS, bootstrap_n=1000):
    per_game, gts, preds = {}, [], []
    for g in games:
        root = os.path.join(ART, "{}_{}".format(g, level))
        lab = os.path.join(root, "labels", "{}.csv".format(version))
        if not os.path.exists(lab):
            continue
        gt = read_codes(os.path.join(root, TRUTH_FILES[truth]))
        pred = read_codes(lab)
        if len(gt) != len(pred):
            raise ValueError("{}: {} labels for {} truth frames".format(g, len(pred), len(gt)))
        per_game[g] = score_pair(gt, pred, bootstrap_n)
        sep = np.array([pm.X], dtype=gt.dtype)
        gts += [gt, sep]
        preds += [pred, sep]
    if not per_game:
        raise SystemExit("no labels/{}.csv found for level {}".format(version, level))
    pooled = score_pair(np.concatenate(gts), np.concatenate(preds), bootstrap_n)
    return {"version": version, "level": level, "truth": truth, "games": per_game,
            "pooled": pooled, "bootstrap_n": bootstrap_n}


def _fmt(v, spec="{:.3f}"):
    return "-" if v is None else spec.format(v)


def print_table(result):
    head = ("{:<14}{:>8}{:>17}{:>8}{:>8}{:>8}{:>8}{:>9}{:>8}{:>8}{:>8}{:>9}{:>9}{:>8}{:>8}".format(
        "game", "spellF1", "[5%-95%]", "macroF1", "aliveF1", "teamAcc", "shrt1.5", "shrt4",
        "splt/h", "missSt", "invSt", "rstTeam", "poss/m", "gt p/m", "predD"))
    print("version {}  level {}  truth {}".format(result["version"], result["level"], result["truth"]))
    print(head)
    rows = list(result["games"].items()) + [("POOLED", result["pooled"])]
    for name, r in rows:
        ci = "[{}-{}]".format(_fmt(r["spell_bootstrap_low"]), _fmt(r["spell_bootstrap_high"]))
        print("{:<14}{:>8}{:>17}{:>8}{:>8}{:>8}{:>8}{:>9}{:>8}{:>8}{:>8}{:>9}{:>9}{:>8}{:>8}".format(
            name, _fmt(r["spell_f1_mean"]), ci, _fmt(r["macro_f1"]), _fmt(r["alive_dead_f1"]),
            _fmt(r["team_acc_gt_alive"]), _fmt(r["short_recall_1_5"]), _fmt(r["short_recall_4"]),
            _fmt(r["false_splits_per_hour"], "{:.0f}"),
            "{}/{}".format(r["missed_stoppages"], r["n_gt_stoppages"]), str(r["invented_stoppages"]),
            _fmt(r["restart_team_acc"]), _fmt(r["pred_poss_per_min"], "{:.2f}"),
            _fmt(r["gt_poss_per_min"], "{:.2f}"), _fmt(r["pred_share_D"])))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", required=True)
    ap.add_argument("--level", default="match")
    ap.add_argument("--truth", default="default", choices=list(TRUTH_FILES))
    ap.add_argument("--games", nargs="*", default=list(GAME_STEMS))
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args(argv)
    result = score_version(args.version, args.level, args.truth, args.games, args.bootstrap)
    out_dir = os.path.join(ART, "_scores")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "{}_{}_{}.json".format(args.version, args.level, args.truth)),
              "w") as fh:
        json.dump(result, fh, indent=1, default=float)
    print_table(result)
    return result


if __name__ == "__main__":
    main()
