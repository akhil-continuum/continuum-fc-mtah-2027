"""Bench-only refit of rule_v1 and rule_v2 on the smoothed bench with w15s touches (T-35, P-17c).

    uv run python -m pipeline.models.refit_w eval  [--truths default labels] [--levels ...]
    uv run python -m pipeline.models.refit_w write --truth-w1 <t> --truth-w2 <t>
        -> pipeline/models/team_w1_params.json (rule_w1), team_w2_params.json (rule_w2)
    ... write [--touch-set dcv --suffix d]  (T-38: the same recipe on roots built with another
        touch set; writes rule_w1<suffix> / rule_w2<suffix>, never overwrites rule_w1 / rule_w2)

Reads the bench level roots artifacts/bench/metrica_g{1,2,3}_<level> only (require_bench_roots),
which must hold stage outputs built with the w15s touch set (a ``make candidate`` run of a
version declaring ``"touch_set": "w15s"``). Never reads the match or the public window.

The protocol is the one rule_v1 and rule_v2 were fitted with, unchanged:
- P(D): rule_v1's dead terms (rule_params.DEAD_DEFAULTS, shapes rule_params.SHAPE), bounded
  logistic fit of tune._fit: signs from rule_params.SIGNS, FROZEN held, L2 0.002 towards the
  physical defaults, bias held at the flat prior log(1/2), every 5th graded frame of the
  match-level roots. D frames are the same under every truth, so the dead fit has no truth.
- rule_w1's team part: rule_v1's team terms, the same bounded fit on truly live frames.
- rule_w2: rule_v2's columns and fit (fit_team.fit, variant shape_a1: alpha 1, C 1, team-flip
  augmentation, no intercept, lbfgs, seed 0).
The fit truth of each team part is an argument (``default`` = first touch, ``labels`` =
organisers' conventions); ``eval`` measures both.

eval: leave one game out for everything (dead weights, team weights): fit on the match level of
two games, predict the held-out game at every level, decode with v2s's dials (c_switch 18),
score under both truths. Controls on the same roots: rule_v1 (stored weights, fitted in-sample
on the old bench) and rule_v2 (stored leave-one-game-out weights), i.e. the no-refit ablation.
Writes artifacts/bench/_t35/eval_<tag>.json and a compact table on stdout (bench only).

write: dead weights, rule_w1 team weights and rule_w2 weights on all three games ("all", used
on the match) and on each pair ("logo", used on the held-out game's bench roots). Both files
carry the same "dead" block. Deterministic (one thread, no randomness besides seed 0).
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
import time

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np                              # noqa: E402

from pipeline.decode import inputs              # noqa: E402
from pipeline.io.roots import require_bench_roots  # noqa: E402
from pipeline.models import fit_team as FT      # noqa: E402
from pipeline.models import rule_params as RP   # noqa: E402
from pipeline.models import rule_scorer as RS   # noqa: E402
from pipeline.models import team_model as TMOD  # noqa: E402
from pipeline.models import tune                # noqa: E402

GAMES = FT.GAMES
LEVELS = FT.LEVELS
FIT_LEVEL = "match"
STEP = 5                     # as tune.STEP and fit_team.STEP
L2 = 0.002                   # as rule_v1 (tune --l2 default)
W2_VARIANT = "shape_a1"      # rule_v2's columns, alpha, C
C_SWITCH = 18.0              # v2s's dial (frozen, P-19)
TOUCH_SET = "w15s"
H, A, D = FT.H, FT.A, FT.D
CACHE = os.path.join(RS.ARTIFACTS, "bench", "_t35")
TRUTHS = ("default", "labels")


def _digest(root):
    h = hashlib.sha256()
    for f in ("events.parquet", "features.parquet", "ball.parquet"):
        with open(os.path.join(root, f), "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:16]


def load_root(game, level):
    """Terms, features, truths and decode inputs of a level root; cached by the content of its
    stage files (so a rebuild with another touch set never reads a stale cache)."""
    root = FT.root_of(game, level)
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "root_{}_{}_{}.pkl".format(game, level, _digest(root)))
    if os.path.isfile(path):
        with open(path, "rb") as fh:
            out = pickle.load(fh)
    else:
        t = time.time()
        data = RS.read_root(root)
        meta = data["meta"]
        n = int(meta["n_frames"])
        dead, team, _info = RS.compute_terms(data)
        probs_v1 = RS.combine(dead, team, n, meta["halves"])
        inp = inputs.build(root, probs_v1, compute="full")
        out = {"root": root, "meta": meta, "dead": dead, "team": team, "probs_v1": probs_v1,
               "inp": inp, "features": data["features"]}
        with open(path, "wb") as fh:
            pickle.dump(out, fh)
        print("cached {} {} in {:.0f} s".format(game, level, time.time() - t), flush=True)
    out["truth"] = {k: FT.truth_codes(root, k) for k in FT.TRUTHS}
    return out


# --- fits -----------------------------------------------------------------------------------

def _sel(r):
    sel = np.zeros(len(r["truth"]["default"]), dtype=bool)
    sel[::STEP] = True
    return sel


def fit_dead(rows):
    Xs, ys = [], []
    for r in rows:
        gt = r["truth"]["default"]
        g = _sel(r) & np.isin(gt, [H, A, D])
        Xs.append(np.stack([r["dead"][k] for k in RP.DEAD_DEFAULTS], 1)[g])
        ys.append(gt[g] == D)
    dw, _ = tune._fit(np.vstack(Xs), np.concatenate(ys), list(RP.DEAD_DEFAULTS),
                      RP.DEAD_DEFAULTS, RP.SIGNS, RP.FROZEN, L2, bias=None,
                      offset=RP.DEAD_BIAS_DEFAULT)
    return dw


def fit_team_terms(rows, truth):
    Xs, ys = [], []
    for r in rows:
        gt = r["truth"][truth]
        g = _sel(r) & ((gt == H) | (gt == A))
        Xs.append(np.stack([r["team"][k] for k in RP.TEAM_DEFAULTS], 1)[g])
        ys.append(gt[g] == H)
    tw, _ = tune._fit(np.vstack(Xs), np.concatenate(ys), list(RP.TEAM_DEFAULTS),
                      RP.TEAM_DEFAULTS, RP.SIGNS, RP.FROZEN, L2)
    return tw


def fit_w2(rows, truth):
    columns, alpha, C = FT.VARIANTS[W2_VARIANT]
    return FT.fit(rows, columns, alpha, C, truth)


# --- predictions (the same code path as the stage, team_model.score_root_models) -------------

def probs_of(r, columns, weights, dead_w):
    fps = float(r["meta"]["fps"])
    n = int(r["meta"]["n_frames"])
    spec = {"columns": columns, "weights": weights}
    q = TMOD.p_home(TMOD.design(columns, r["team"], r["features"], fps), spec)
    base = r["probs_v1"] if dead_w is None else RS.combine(
        r["dead"], r["team"], n, r["meta"]["halves"], dead_w=dead_w,
        dead_bias=RP.DEAD_BIAS_DEFAULT)
    return TMOD.compose(base, q, r["meta"]["halves"]), q


def dead_metrics(probs, gt):
    g = np.isin(gt, [H, A, D])
    p = np.clip(probs[g, 2].astype(float), 1e-6, 1 - 1e-6)
    y = gt[g] == D
    return {"ll_dead": round(float(-np.mean(np.where(y, np.log(p), np.log(1 - p)))), 4),
            "f1_dead": round(float(tune.RS_f1(p > 0.5, y)), 4)}


def evaluate(models, levels, cache):
    """models: name -> per-game (columns, team weights, dead weights or None)."""
    res = {}
    for name, per_game in models.items():
        res[name] = {}
        for lv in levels:
            preds, qs, cell = {}, {}, {"frame": {}, "dead": {}}
            for g in GAMES:
                r = cache[(g, lv)]
                cols, tw, dw = per_game[g]
                probs, q = probs_of(r, cols, tw, dw)
                qs[g] = q
                preds[g] = FT.decode_with(r["inp"], probs, C_SWITCH)
                cell["dead"][g] = dead_metrics(probs, r["truth"]["default"])
            for t in TRUTHS:
                fr = {}
                for g in GAMES:
                    r = cache[(g, lv)]
                    seen = r["features"]["ball_seen"].to_numpy(float) > 0.5
                    fr[g] = FT.frame_metrics(qs[g], r["truth"][t], seen)
                pooled = {}
                for ctx in ("all", "within_1s_of_change", "elsewhere"):
                    n_ = sum(fr[g][ctx]["n"] for g in GAMES)
                    pooled[ctx] = {k: round(sum(fr[g][ctx][k] * fr[g][ctx]["n"] for g in GAMES)
                                            / max(n_, 1), 4) for k in ("acc", "ll")}
                cell["frame"][t] = pooled
                cell[t] = FT.score_set(preds, {g: cache[(g, lv)]["truth"][t] for g in GAMES})
            res[name][lv] = cell
    return res


def stored_controls():
    v1 = (FT.TERMS, dict(RP.TEAM_W), None)
    p2 = TMOD.load_model("rule_v2")
    return {"rule_v1_stored": {g: v1 for g in GAMES},
            "rule_v2_stored": {g: (p2["columns"], p2["weights"]["logo"][g], None)
                               for g in GAMES}}


def refit_models(cache, truths):
    cols2 = FT.VARIANTS[W2_VARIANT][0]
    out = {}
    dws = {g: fit_dead([cache[(o, FIT_LEVEL)] for o in GAMES if o != g]) for g in GAMES}
    for t in truths:
        tr = {g: [cache[(o, FIT_LEVEL)] for o in GAMES if o != g] for g in GAMES}
        out["w1_" + t] = {g: (FT.TERMS, fit_team_terms(tr[g], t), dws[g]) for g in GAMES}
        out["w2_" + t] = {g: (cols2, fit_w2(tr[g], t), dws[g]) for g in GAMES}
    # P(D) refit alone, stored team parts: separates the dead refit from the team refit
    p2 = TMOD.load_model("rule_v2")
    out["v1team_wdead"] = {g: (FT.TERMS, dict(RP.TEAM_W), dws[g]) for g in GAMES}
    out["v2team_wdead"] = {g: (p2["columns"], p2["weights"]["logo"][g], dws[g]) for g in GAMES}
    return out, dws


def print_eval(res):
    for name, by_lv in res.items():
        for lv, cell in by_lv.items():
            lb, df = cell["labels"]["pooled"], cell["default"]["pooled"]
            fl, fd = cell["frame"]["labels"], cell["frame"]["default"]
            dl = np.mean([cell["dead"][g]["ll_dead"] for g in GAMES])
            print("{:<16}{:<6} spell lab {:.4f} [{}] def {:.4f} | acc lab {:.4f} near1s {:.4f} "
                  "ll {:.4f} | acc def {:.4f} near1s {:.4f} | ll_dead {:.4f} | pm {:.2f} sh4 "
                  "{:.3f} fs/h {:.1f}".format(
                      name, lv, lb["spell_f1_mean"],
                      " ".join("%.3f" % cell["labels"][g] for g in GAMES), df["spell_f1_mean"],
                      fl["all"]["acc"], fl["within_1s_of_change"]["acc"], fl["all"]["ll"],
                      fd["all"]["acc"], fd["within_1s_of_change"]["acc"], dl,
                      lb["pred_poss_per_min"], lb["short_recall_4"],
                      lb["false_splits_per_hour"]), flush=True)


def write_params(cache, truth_w1, truth_w2, touch_set=TOUCH_SET, suffix=""):
    rows = [cache[(g, FIT_LEVEL)] for g in GAMES]
    pairs = {g: [cache[(o, FIT_LEVEL)] for o in GAMES if o != g] for g in GAMES}
    dead = {"bias": RP.DEAD_BIAS_DEFAULT, "terms": list(RP.DEAD_DEFAULTS),
            "fit": {"l2": L2, "step": STEP, "fit_level": FIT_LEVEL, "truth": "D frames (same "
                    "in every truth)", "bias": "held at log(1/2)", "objective": "tune._fit"},
            "weights": {"all": fit_dead(rows), "logo": {g: fit_dead(pairs[g]) for g in GAMES}}}
    common = {"touch_set": touch_set, "bench": "smoothed (noise model v3, T-33)",
              "task": "T-35" if not suffix else "T-38 (T-35 recipe)"}
    w1 = {"model": "rule_w1" + suffix, "variant": "rule_v1_terms", "columns": list(FT.TERMS),
          "fit": dict(common, objective="tune._fit (bounded, signs, L2 towards defaults)",
                      l2=L2, step=STEP, fit_level=FIT_LEVEL, truth=FT.TRUTHS[truth_w1],
                      intercept=False),
          "weights": {"all": fit_team_terms(rows, truth_w1),
                      "logo": {g: fit_team_terms(pairs[g], truth_w1) for g in GAMES}},
          "dead": dead}
    columns, alpha, C = FT.VARIANTS[W2_VARIANT]
    w2 = {"model": "rule_w2" + suffix, "variant": W2_VARIANT, "columns": list(columns),
          "fit": dict(common, alpha=alpha, C=C, step=FT.STEP, fit_level=FIT_LEVEL,
                      truth=FT.TRUTHS[truth_w2], augmentation="team flip", intercept=False,
                      solver="lbfgs", seed=FT.SEED),
          "weights": {"all": fit_w2(rows, truth_w2),
                      "logo": {g: fit_w2(pairs[g], truth_w2) for g in GAMES}},
          "dead": dead}
    for params in (w1, w2):
        path = TMOD.params_path(params["model"])
        with open(path, "w") as fh:
            json.dump(params, fh, indent=1, sort_keys=True)
            fh.write("\n")
        print("wrote", os.path.relpath(path, RS.REPO))


def main(argv=None):
    ap = argparse.ArgumentParser(description="bench-only refit of rule_v1 / rule_v2 (T-35)")
    ap.add_argument("action", choices=("eval", "write"))
    ap.add_argument("--levels", nargs="+", default=list(LEVELS))
    ap.add_argument("--truths", nargs="+", default=list(TRUTHS), choices=TRUTHS)
    ap.add_argument("--truth-w1", choices=TRUTHS)
    ap.add_argument("--truth-w2", choices=TRUTHS)
    ap.add_argument("--tag", default="")
    ap.add_argument("--touch-set", default=TOUCH_SET,
                    help="touch set the roots' events were built with (recorded in the params)")
    ap.add_argument("--suffix", default="", choices=("", "d"),
                    help="model name suffix: rule_w1<suffix>, rule_w2<suffix> (T-38: d = dcv)")
    args = ap.parse_args(argv)
    if (args.touch_set == TOUCH_SET) != (args.suffix == ""):
        ap.error("another touch set needs its own model names (--suffix), and back")
    levels = list(args.levels) if args.action == "eval" else []
    require_bench_roots([FT.root_of(g, lv) for g in GAMES for lv in set(levels) | {FIT_LEVEL}])
    cache = {(g, lv): load_root(g, lv) for g in GAMES for lv in set(levels) | {FIT_LEVEL}}
    if args.action == "write":
        if not (args.truth_w1 and args.truth_w2):
            ap.error("write needs --truth-w1 and --truth-w2")
        write_params(cache, args.truth_w1, args.truth_w2, args.touch_set, args.suffix)
        return 0
    models = stored_controls()
    refits, dws = refit_models(cache, args.truths)
    models.update(refits)
    res = evaluate(models, levels, cache)
    print_eval(res)
    with open(os.path.join(CACHE, "eval{}.json".format(args.tag)), "w") as fh:
        json.dump({"results": res, "dead_logo": dws}, fh, indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
