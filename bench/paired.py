"""Paired comparison of bench versions on shared resamples, plus error counts by type.

    uv run python -m bench.paired --versions v6w v11g [--levels low match high]
        [--truths default confirmed labels] [--pairs v11g:v6w]
        [--draws 2000] [--block-s 60] [--out artifacts/bench/_judge/paired.json]
    make judge VERSION=v11g BASE=v6w [SCORE=0] [DRAWS=2000]

(Moved from the team's experiment folder; otherwise unchanged. pipeline.models.fit_t32 uses its
game tables, error counts and seed.)

Bench only: reads artifacts/bench/<game>_<level>/labels/<v>.csv and the truth files. Every
spell, overlap and match comes from evaluation/possession_metrics.py (imported).

Two paired bootstraps, the same draws for every version:
  A  "official": resample true possessions (pooled over games, as bench.score does), predicted
     count held fixed, clamped as in pm.bootstrap_spell_f1. Reproduces bench.score's interval.
     It never resamples predictions, so it understates uncertainty for versions that differ
     mainly in how many possessions they predict.
  B  "block": stratified by game, resample contiguous time blocks (default 60 s) with
     replacement; true and predicted possessions are assigned to the block of their start
     frame, matching is done once on the full game. Resamples false positives too.
Error counts (definitions in error_counts below) are per game and pooled.
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "evaluation"))
import possession_metrics as pm  # noqa: E402

ART = os.path.join(REPO, "artifacts", "bench")
GAMES = ("metrica_g1", "metrica_g2", "metrica_g3")
TRUTH = {"default": "truth.csv", "first_touch": "truth_first_touch.csv",
         "confirmed": "truth_confirmed.csv", "labels": "truth_labels.csv"}
SEED = 20261113


def codes(path):
    return pm.label_codes(pm.read_label_csv(path))


def game_table(gt, pred, block):
    """Per-game matching once; flags per IoU, spell starts, block ids."""
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    gs = pm.possession_spells(gt, runs)
    ps = pm.possession_spells(filled, runs)
    ov = pm.spell_overlaps(gs, ps)
    flags = np.vstack([pm.matched_gt_flags(ov, len(gs), t) for t in pm.SPELL_IOUS])
    nblk = int(np.ceil(len(gt) / block))
    gb = np.array([a // block for a, _, _ in gs], dtype=int)
    pb = np.array([a // block for a, _, _ in ps], dtype=int)
    tp_blk = np.zeros((nblk, len(pm.SPELL_IOUS)))
    for k in range(len(pm.SPELL_IOUS)):
        np.add.at(tp_blk[:, k], gb, flags[k].astype(float))
    gt_blk = np.bincount(gb, minlength=nblk).astype(float)
    pr_blk = np.bincount(pb, minlength=nblk).astype(float)
    return {"runs": runs, "filled": filled, "gs": gs, "ps": ps, "ov": ov, "flags": flags,
            "tp_blk": tp_blk, "gt_blk": gt_blk, "pr_blk": pr_blk, "n_pred": len(ps)}


def f1_from(tp, ngt, npred):
    """tp: (..., n_iou); ngt, npred broadcast. mean over IoU of 2tp/(ngt+npred)."""
    den = ngt[..., None] + npred[..., None]
    f = np.where(tp > 0, 2.0 * tp / np.maximum(den, 1e-9), 0.0)
    return f.mean(axis=-1)


def error_counts(gt, t):
    """Counts by type for one game. t: game_table output."""
    runs, filled, gs, ps, ov = t["runs"], t["filled"], t["gs"], t["ps"], t["ov"]
    flags05 = pm.matched_gt_flags(ov, len(gs), 0.5)
    lens = np.array([b - a for a, b, _ in gs])
    out = {}
    out["missed_short_1_5"] = int(((lens < 1.5 * pm.FPS) & ~flags05).sum())
    out["missed_short_4"] = int(((lens < 4 * pm.FPS) & ~flags05).sum())
    out["missed_ge_4"] = int(((lens >= 4 * pm.FPS) & ~flags05).sum())
    # predicted possessions with no match at IoU 0.5 (spurious or mis-edged)
    matched_pred = set()
    used = np.zeros(len(gs), bool)
    for iou, i, j in ov:
        if iou < 0.5:
            break
        if used[i] or j in matched_pred:
            continue
        used[i] = True
        matched_pred.add(j)
    out["unmatched_pred"] = int(len(ps) - len(matched_pred))
    # false splits: as bench.score
    gt_tr = np.array([f for f, _, _ in pm.transitions(gt, runs)])
    pr_tr = pm.transitions(filled, runs)
    splits = 0
    for f, _, _ in pr_tr:
        inside = any(a < f < b for a, b, _ in gs)
        far = not len(gt_tr) or np.min(np.abs(gt_tr - f)) > 25
        splits += int(inside and far)
    out["false_split"] = splits
    # stoppages
    gseg = pm.segments(gt, runs)
    pseg = pm.segments(filled, runs)
    gd = [(a, b) for a, b, k in gseg if k == pm.D]
    pd_ = [(a, b) for a, b, k in pseg if k == pm.D]
    missed = merged = 0
    for a, b in gd:
        if (filled[a:b] == pm.D).mean() >= 0.5:
            continue
        missed += 1
        if a > 0 and b < len(gt) and gt[a - 1] in (pm.H, pm.A) and gt[a - 1] == gt[b]:
            w = filled[a - 1:b + 1]
            if (w == gt[a - 1]).all() or (not (w == pm.D).any() and len(set(w.tolist())) == 1):
                merged += 1
    out["missed_stoppage"] = missed
    out["merged_across_missed_stoppage"] = merged
    out["invented_stoppage"] = sum(1 for a, b in pd_ if (gt[a:b] == pm.D).mean() < 0.5)
    # wrong team after restart: as bench.score (first non-D in [restart, restart+50))
    wrong = total = 0
    for a, b in gd:
        if b >= len(gt) or gt[b] not in (pm.H, pm.A):
            continue
        live = filled[b:b + 50]
        live = live[live != pm.D]
        total += 1
        wrong += int(not (len(live) and live[0] == gt[b]))
    out["wrong_team_after_restart"] = wrong
    out["n_restarts"] = total
    # direct live team switches: nearest predicted switch into the new team within +-50 frames
    late = early = ontime = absent = 0
    pr_into = {pm.H: np.array([f for f, fr, to in pr_tr if to == pm.H]),
               pm.A: np.array([f for f, fr, to in pr_tr if to == pm.A])}
    for f, fr, to in pm.transitions(gt, runs):
        if fr not in (pm.H, pm.A) or to not in (pm.H, pm.A):
            continue
        cand = pr_into[to]
        cand = cand[np.abs(cand - f) <= 50] if len(cand) else cand
        if not len(cand):
            absent += 1
            continue
        d = int(cand[np.argmin(np.abs(cand - f))] - f)
        if d > 10:
            late += 1
        elif d < -10:
            early += 1
        else:
            ontime += 1
    out.update({"late_switch_gt10": late, "early_switch_gt10": early,
                "switch_within10": ontime, "switch_absent_50": absent})
    out["n_gt_poss"] = len(gs)
    out["n_pred_poss"] = len(ps)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", nargs="+", default=["v0", "v1", "v2", "v2h"])
    ap.add_argument("--levels", nargs="+", default=["low", "match", "high"])
    ap.add_argument("--truths", nargs="+", default=["default", "confirmed"])
    ap.add_argument("--pairs", nargs="+",
                    default=["v1:v0", "v2:v0", "v2h:v0", "v2h:v2", "v1:v2", "v2h:v1"])
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--block-s", type=float, default=60.0)
    ap.add_argument("--out", default=os.path.join(REPO, "artifacts", "bench", "_judge",
                                                  "paired.json"))
    ap.add_argument("--no-errors", action="store_true")
    a = ap.parse_args(argv)
    block = int(a.block_s * pm.FPS)
    pairs = [tuple(p.split(":")) for p in a.pairs]
    result = {"versions": a.versions, "draws": a.draws, "block_s": a.block_s, "seed": SEED,
              "cells": {}}
    for level in a.levels:
        for truth in a.truths:
            key = "{}|{}".format(level, truth)
            tabs, gts = {}, {}
            for g in GAMES:
                root = os.path.join(ART, "{}_{}".format(g, level))
                gts[g] = codes(os.path.join(root, TRUTH[truth]))
                for v in a.versions:
                    pred = codes(os.path.join(root, "labels", "{}.csv".format(v)))
                    tabs[(g, v)] = game_table(gts[g], pred, block)
            # scheme A: pooled GT possessions resampled, same picks for all versions
            n_gt_g = {g: len(tabs[(g, a.versions[0])]["gs"]) for g in GAMES}
            n_tot = sum(n_gt_g.values())
            rng = np.random.RandomState(SEED)
            picksA = rng.randint(0, n_tot, (a.draws, n_tot))
            # scheme B: per-game block picks
            rng = np.random.RandomState(SEED + 1)
            nblk = {g: len(tabs[(g, a.versions[0])]["gt_blk"]) for g in GAMES}
            picksB = {g: rng.randint(0, nblk[g], (a.draws, nblk[g])) for g in GAMES}
            point, drawsA, drawsB, per_game = {}, {}, {}, {}
            for v in a.versions:
                fl = np.hstack([tabs[(g, v)]["flags"] for g in GAMES])  # (n_iou, n_tot)
                npred = float(sum(tabs[(g, v)]["n_pred"] for g in GAMES))
                tp = fl.sum(axis=1)
                point[v] = float(f1_from(tp[None], np.array([n_tot]), np.array([npred]))[0])
                # A
                tpA = np.stack([fl[k][picksA].sum(axis=1) for k in range(fl.shape[0])], -1)
                fpA = np.maximum(0, npred - tpA)
                fnA = n_tot - tpA
                fA = np.where(tpA > 0, 2 * tpA / (2 * tpA + fpA + fnA), 0.0).mean(-1)
                drawsA[v] = fA
                # B
                tpB = np.zeros((a.draws, len(pm.SPELL_IOUS)))
                ngB = np.zeros(a.draws)
                npB = np.zeros(a.draws)
                pg = {}
                for g in GAMES:
                    t = tabs[(g, v)]
                    P = picksB[g]
                    tpg = t["tp_blk"][P].sum(axis=1)
                    ngg = t["gt_blk"][P].sum(axis=1)
                    npg = t["pr_blk"][P].sum(axis=1)
                    tpB += tpg
                    ngB += ngg
                    npB += npg
                    pg[g] = {"point": float(f1_from(t["flags"].sum(axis=1)[None],
                                                    np.array([len(t["gs"])]),
                                                    np.array([t["n_pred"]]))[0]),
                             "draws": f1_from(tpg, ngg, npg)}
                drawsB[v] = f1_from(tpB, ngB, npB)
                per_game[v] = pg
            cell = {"point": point, "pairs": {}, "errors": {}}
            for c, b in pairs:
                if c not in point or b not in point:
                    continue
                dA = drawsA[c] - drawsA[b]
                dB = drawsB[c] - drawsB[b]
                games = {}
                for g in GAMES:
                    dg = per_game[c][g]["draws"] - per_game[b][g]["draws"]
                    games[g] = {"diff": per_game[c][g]["point"] - per_game[b][g]["point"],
                                "lo": float(np.percentile(dg, 5)),
                                "hi": float(np.percentile(dg, 95))}
                cell["pairs"]["{}-{}".format(c, b)] = {
                    "diff": point[c] - point[b],
                    "A_lo": float(np.percentile(dA, 5)), "A_hi": float(np.percentile(dA, 95)),
                    "B_lo": float(np.percentile(dB, 5)), "B_hi": float(np.percentile(dB, 95)),
                    "B_p_gt0": float((dB > 0).mean()), "games": games}
            cell["ci"] = {v: {"A": [float(np.percentile(drawsA[v], 5)),
                                    float(np.percentile(drawsA[v], 95))],
                              "B": [float(np.percentile(drawsB[v], 5)),
                                    float(np.percentile(drawsB[v], 95))]} for v in a.versions}
            if not a.no_errors:
                for v in a.versions:
                    per = {g: error_counts(gts[g], tabs[(g, v)]) for g in GAMES}
                    pooled = {k: sum(per[g][k] for g in GAMES) for k in per[GAMES[0]]}
                    cell["errors"][v] = {"pooled": pooled, "games": per}
            result["cells"][key] = cell
            print("== level {} truth {} (block {} s, {} draws)".format(level, truth, a.block_s,
                                                                       a.draws))
            for v in a.versions:
                ci = cell["ci"][v]
                print("  {:<5} {:.3f}  A[{:.3f}-{:.3f}]  B[{:.3f}-{:.3f}]".format(
                    v, point[v], *ci["A"], *ci["B"]))
            for name, p in cell["pairs"].items():
                gtxt = " ".join("{}:{:+.3f}[{:+.3f},{:+.3f}]".format(g[-2:], d["diff"], d["lo"],
                                                                     d["hi"])
                                for g, d in p["games"].items())
                print("  {:<8} {:+.3f}  A[{:+.3f},{:+.3f}]  B[{:+.3f},{:+.3f}] P>0 {:.3f} | {}".format(
                    name, p["diff"], p["A_lo"], p["A_hi"], p["B_lo"], p["B_hi"], p["B_p_gt0"],
                    gtxt))
            if not a.no_errors:
                ks = ["false_split", "missed_short_1_5", "missed_short_4", "missed_ge_4",
                      "unmatched_pred", "missed_stoppage", "merged_across_missed_stoppage",
                      "invented_stoppage", "late_switch_gt10", "early_switch_gt10",
                      "switch_absent_50", "wrong_team_after_restart", "n_pred_poss", "n_gt_poss"]
                print("  errors " + " ".join(k for k in ks))
                for v in a.versions:
                    e = cell["errors"][v]["pooled"]
                    print("  {:<5} ".format(v) + " ".join(str(e[k]) for k in ks))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(result, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
