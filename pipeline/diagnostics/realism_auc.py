"""Adversarial check: can a small classifier tell match windows from bench windows?

    uv run python -m pipeline.diagnostics.realism_auc --level match
    uv run python -m pipeline.diagnostics.realism_auc --level high --exclude jump_rate
    uv run python -m pipeline.diagnostics.realism_auc --roots DIR1 DIR2 ... --tag myvariant

Both sides are cut into non-overlapping 30 s windows inside graded spans; each window is described
by the aggregate features of ``input_stats.window_features`` (missing rate, gap counts and lengths,
ball speed and jitter, tracked players, crowd-conditional missing rates). Cross-validated out-of-
fold AUC with folds made of contiguous 5-minute blocks (so neighbouring windows never sit on both
sides of a split), for a gradient-boosted tree model and a logistic regression.

Prints only, through ``summary()``: the two AUCs, window counts, and the names of the ten features
with the largest held-out permutation importance (names only, no values from the match).
"""
import argparse
import os

import numpy as np
import pandas as pd

from pipeline.diagnostics import input_stats
from pipeline.io import sources
from pipeline.report import match_entry, quiet_warnings, summary

# Params
BLOCK_WINDOWS = 10       # 10 x 30 s = 5-minute blocks as CV groups
N_FOLDS = 5
SEED = 0
TOP_K = 10


def _blocks(df, offset, per_block=BLOCK_WINDOWS):
    return offset + np.arange(len(df)) // per_block


def _auc_and_importance(X, y, groups, names):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_gb = np.zeros(len(y))
    oof_lr = np.zeros(len(y))
    imp = np.zeros(X.shape[1])
    for tr, te in cv.split(X, y, groups):
        gb = make_pipeline(SimpleImputer(strategy="median"),
                           HistGradientBoostingClassifier(max_depth=3, max_iter=150,
                                                          learning_rate=0.1,
                                                          class_weight="balanced",
                                                          random_state=SEED))
        gb.fit(X[tr], y[tr])
        oof_gb[te] = gb.predict_proba(X[te])[:, 1]
        lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                           LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
        lr.fit(X[tr], y[tr])
        oof_lr[te] = lr.predict_proba(X[te])[:, 1]
        if len(np.unique(y[te])) == 2:
            pi = permutation_importance(gb, X[te], y[te], scoring="roc_auc", n_repeats=5,
                                        random_state=SEED)
            imp += pi.importances_mean
    order = np.argsort(-imp)[:TOP_K]
    return (float(roc_auc_score(y, oof_gb)), float(roc_auc_score(y, oof_lr)),
            [names[i] for i in order])


@match_entry
def main(argv=None):
    quiet_warnings()
    ap = argparse.ArgumentParser(description="Bench-versus-match adversarial AUC")
    ap.add_argument("--level", default="match", help="bench noise level (low, match, high)")
    ap.add_argument("--roots", nargs="*", help="explicit bench roots instead of --level")
    ap.add_argument("--tag", help="report name suffix (default: the level)")
    ap.add_argument("--window", type=int, default=input_stats.WINDOW_FRAMES,
                    help="window length in frames (default 750 = 30 s)")
    ap.add_argument("--exclude", nargs="*", default=[], help="feature names to leave out")
    args = ap.parse_args(argv)

    names = [f for f in input_stats.FEATURES if f not in set(args.exclude)]
    match_w = input_stats.window_features(input_stats.Frames(sources.match_root()), args.window)[names]
    if args.roots:
        roots = args.roots
    else:
        roots = [r for r in input_stats.bench_roots() if r.endswith("_" + args.level)]
    per_block = max(1, (BLOCK_WINDOWS * input_stats.WINDOW_FRAMES) // args.window)
    bench_parts, offset = [], 10 ** 6
    for root in roots:
        w = input_stats.window_features(input_stats.Frames(root), args.window)[names]
        w["_g"] = _blocks(w, offset, per_block)
        offset += 10 ** 6
        bench_parts.append(w)
    bench_w = pd.concat(bench_parts, ignore_index=True)
    match_w = match_w.assign(_g=_blocks(match_w, 0, per_block))

    X = np.vstack([match_w[names].to_numpy(float), bench_w[names].to_numpy(float)])
    y = np.concatenate([np.ones(len(match_w), int), np.zeros(len(bench_w), int)])
    groups = np.concatenate([match_w["_g"].to_numpy(), bench_w["_g"].to_numpy()])
    auc_gb, auc_lr, top = _auc_and_importance(X, y, groups, names)
    tag = args.tag or args.level
    out = {"auc_gbm": round(auc_gb, 4), "auc_logreg": round(auc_lr, 4),
           "n_match_windows": int(len(match_w)), "n_bench_windows": int(len(bench_w)),
           "n_bench_roots": len(roots), "n_features": len(names), "window_frames": args.window}
    out.update({"top{}".format(i + 1): name for i, name in enumerate(top)})
    if args.exclude:
        out["excluded"] = ",".join(args.exclude)[:80]
    summary(_name="realism_auc.{}".format(tag), **out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
