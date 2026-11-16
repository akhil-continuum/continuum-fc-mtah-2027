"""Bench-only fitter and evaluator for the team model rule_v2 (pipeline/models/team_model.py).

    uv run python -m pipeline.models.fit_team eval  [--variants ...] [--levels low match high]
    uv run python -m pipeline.models.fit_team write --variant <name>   # -> team_v2_params.json

Bench level roots artifacts/bench/metrica_g{1,2,3}_<level> only (require_bench_roots). Never
reads the match or the public window.

Fit (per variant): logistic regression, no intercept, L2 (C), on every STEP-th truly live frame
of the match-level roots of the training games (truth.csv, the default truth). Each training row
is also entered mirrored (all columns negated, label flipped: team-flip augmentation), so the
fit has no lean to home or away. Possession-balanced sample weights 1 / L**alpha, L = length of
the true possession the frame is in (P-10; alpha 0 = frame log-loss). Columns are standardised
by their RMS on the training rows for the fit; stored weights are in raw units. lbfgs, one
thread, deterministic.

eval: leave one game out (fit on the other two games, predict the held-out game at every level),
then per level and truth: frame-level team accuracy and log-loss on live frames, accuracy by
context, a 10-bin reliability table, and spellF1 after the v2s decoder at a c_switch grid
(pooled and per game, with false splits per hour, short-possession recall, possessions per
minute). rule_v1 is evaluated the same way as the control. Results go to
artifacts/bench/_team_v2/eval_<tag>.json and a compact table on stdout.

write: fits the variant on all three games (weights "all") and on each pair (weights "logo"),
and writes pipeline/models/team_v2_params.json (or, with --model rule_vN, that model's file).

T-28: TRUTHS also has "labels" (truth_labels.csv, the organisers' conventions). Variants named
lab_* are fitted on it (FIT_TRUTH); rule_v3 = lab_shape_a1, rule_v4 = lab_lb_a1 (plus the
loose-ball columns). eval reports frame metrics under default and labels truth and spellF1 under
all three truths. --temperature: logits divided by a single temperature per held-out game,
fitted on cross-fitted logits of the two training games only (logo_temperatures).
"""
import argparse
import copy
import json
import os
import pickle
import sys
import time

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")          # one thread: deterministic fits (as the Makefile)

import numpy as np                          # noqa: E402
import pandas as pd                         # noqa: E402
from sklearn.linear_model import LogisticRegression# noqa: E402

from bench import score as bscore           # noqa: E402
from pipeline.decode import decoders, inputs# noqa: E402
from pipeline.io.roots import require_bench_roots# noqa: E402
from pipeline.models import rule_scorer as RS# noqa: E402
from pipeline.models import team_model as TMOD# noqa: E402

GAMES = ("metrica_g1", "metrica_g2", "metrica_g3")
LEVELS = ("low", "match", "high")
TRUTHS = {"default": "truth.csv", "confirmed": "truth_confirmed.csv",
          "labels": "truth_labels.csv"}
FIT_LEVEL = "match"
STEP = 5
SEED = 0
H, A, D, X = 0, 1, 2, 3
FPS = 25
CACHE = os.path.join(RS.ARTIFACTS, "bench", "_team_v2")
C_SWITCH_GRID = (18.0, 24.0, 30.0, 36.0, 45.0)

TERMS = ["ctrl_last", "ctrl_next", "carry", "pass", "touch_last", "near", "territory", "spread"]
W4 = ("c05", "c1", "c2", "c4")
PLAYERS = ["shape_{}_diff_{}".format(s, w) for s in ("fwd_vel", "speed") for w in W4] + \
          ["shape_{}_rel_{}".format(s, w) for s in ("depth", "width") for w in W4]
PLAYERS_RAW = ["shape_{}_diff_{}".format(s, w) for s in ("fwd_vel", "speed", "depth", "width")
               for w in W4]
PRESS = ["shape_{}_diff_{}".format(s, w) for s in ("press", "approach") for w in W4]
VARIANTS = {
    # name: (columns, alpha, C)
    "terms_a0": (TERMS, 0.0, 1.0),
    "terms_a1": (TERMS, 1.0, 1.0),
    "players_a1": (TERMS + PLAYERS, 1.0, 1.0),
    "playersraw_a1": (TERMS + PLAYERS_RAW, 1.0, 1.0),
    "shape_a0": (TERMS + PLAYERS + PRESS, 0.0, 1.0),
    "shape_a05": (TERMS + PLAYERS + PRESS, 0.5, 1.0),
    "shape_a1": (TERMS + PLAYERS + PRESS, 1.0, 1.0),
    "shape_ctx_a1": (TERMS + PLAYERS + PRESS + [c + "*unseen" for c in TERMS + PLAYERS],
                     1.0, 1.0),
}

# T-28: the same method fitted on the labels truth (truth_labels.csv, the organisers' conventions
# C1 to C9), then loose-ball features (P-14, features/loose_ball.py) and contact-type terms from
# the events table (P-15, team_model.event_terms). FIT_TRUTH names the truth each variant is
# fitted on (default: truth.csv, as rule_v2).
W3 = ("c0", "c05", "c1")
LB = ["lb_{}_diff_{}".format(s, w) for s in ("tti", "pfirst", "clear") for w in W3]
LB_P = ["lb_pfirst_diff_{}".format(w) for w in W3]
EVT = ["evt_redirect", "evt_header", "evt_defl_collect"]
SHAPE = TERMS + PLAYERS + PRESS
VARIANTS.update({
    "lab_shape_a1": (SHAPE, 1.0, 1.0),
    "lab_shape_a05": (SHAPE, 0.5, 1.0),
    "lab_shape_a0": (SHAPE, 0.0, 1.0),
    "lab_lb_a1": (SHAPE + LB, 1.0, 1.0),
    "lab_lbp_a1": (SHAPE + LB_P, 1.0, 1.0),
    "lab_lb_a05": (SHAPE + LB, 0.5, 1.0),
    "lab_lb_a0": (SHAPE + LB, 0.0, 1.0),
    "lab_lbevt_a1": (SHAPE + LB + EVT, 1.0, 1.0),
    "lab_lbg_a1": (SHAPE + LB + [c + "*loose" for c in LB], 1.0, 1.0),
})
FIT_TRUTH = {v: "labels" for v in VARIANTS if v.startswith("lab_")}


# Gradient-boosted comparison (eval only; not storable as tracked text in this module):
# name: (antisymmetric columns, symmetric context columns, alpha). Trained on rows and their
# mirrors (antisymmetric columns negated, context unchanged, label flipped); the prediction is
# symmetrised, q = (f(x) + 1 - f(mirror x)) / 2, so it has no lean either.
SYM = ["ctx:unseen", "shape_valid_c1"]
GBM_VARIANTS = {
    "gbm_terms_a1": (TERMS, SYM, 1.0),
    "gbm_shape_a1": (TERMS + PLAYERS + PRESS, SYM, 1.0),
}
GBM_PARAMS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
                  random_state=SEED)


def gbm_design(r, anti, sym):
    Xa = TMOD.design(anti, r["team"], r["features"], FPS)
    ctx = TMOD.context(r["features"], FPS)
    Xs = np.stack([ctx[c[4:]] if c.startswith("ctx:") else
                   np.nan_to_num(r["features"][c].to_numpy(float), nan=0.0) for c in sym], axis=1)
    return Xa, Xs


def fit_truth(variant):
    return FIT_TRUTH.get(variant, "default")


def fit_gbm(rows, anti, sym, alpha):
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xs_, ys, ws = [], [], []
    for r in rows:
        gt = r["truth"]["default"]
        Xa, Xs = gbm_design(r, anti, sym)
        live = np.flatnonzero((gt == H) | (gt == A))[::STEP]
        y = (gt[live] == H).astype(int)
        w = poss_weights(gt, alpha)[live]
        Xs_ += [np.hstack([Xa[live], Xs[live]]), np.hstack([-Xa[live], Xs[live]])]
        ys += [y, 1 - y]
        ws += [w, w]
    w = np.concatenate(ws)
    m = HistGradientBoostingClassifier(**GBM_PARAMS)
    m.fit(np.vstack(Xs_), np.concatenate(ys), sample_weight=w / w.mean())
    return m


def q_gbm(m, r, anti, sym):
    Xa, Xs = gbm_design(r, anti, sym)
    f = m.predict_proba(np.hstack([Xa, Xs]))[:, 1]
    fm = m.predict_proba(np.hstack([-Xa, Xs]))[:, 1]
    return 0.5 * (f + 1.0 - fm)


def root_of(game, level):
    return os.path.join(RS.ARTIFACTS, "bench", "{}_{}".format(game, level))


def truth_codes(root, which="default"):
    lab = pd.read_csv(os.path.join(root, TRUTHS[which]), dtype={"label": str})["label"]
    lab = lab.to_numpy().astype(str)
    return np.select([lab == c for c in "HADX"], [H, A, D, X], -1).astype(np.int8)


def runs_of(codes):
    br = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    b = np.concatenate(([0], br, [len(codes)]))
    return [(int(b[i]), int(b[i + 1]), int(codes[b[i]])) for i in range(len(b) - 1)]


# --- cached per-root inputs (bench only, under artifacts/bench/_team_v2/) --------------------

def load_root(game, level):
    """dict: team terms, features (needed columns), probs_v1, meta, DecodeInputs, truths."""
    os.makedirs(CACHE, exist_ok=True)
    root = root_of(game, level)
    path = os.path.join(CACHE, "root_{}_{}.pkl".format(game, level))
    feats = pd.read_parquet(os.path.join(root, "features.parquet"))
    if os.path.isfile(path):
        with open(path, "rb") as fh:
            out = pickle.load(fh)
    else:
        t = time.time()
        data = RS.read_root(root)
        n = int(data["meta"]["n_frames"])
        dead, team, info = RS.compute_terms(data)
        probs_v1 = RS.combine(dead, team, n, data["meta"]["halves"])
        inp = inputs.build(root, probs_v1, compute="full")
        out = {"team": team, "probs_v1": probs_v1, "meta": data["meta"], "inp": inp}
        with open(path, "wb") as fh:
            pickle.dump(out, fh)
        print("cached {} {} in {:.0f} s".format(game, level, time.time() - t), flush=True)
    out["features"] = feats
    out["truth"] = {k: truth_codes(root, k) for k in TRUTHS}
    ev = pd.read_parquet(os.path.join(root, "events.parquet"))
    out["team"] = dict(out["team"], **TMOD.event_terms(ev, len(feats), FPS))
    return out


def poss_weights(gt, alpha):
    w = np.zeros(len(gt))
    for a, b, k in runs_of(gt):
        if k in (H, A):
            w[a:b] = 1.0 / float(b - a) ** alpha
    return w


# --- fit ---------------------------------------------------------------------------------------

def fit(rows, columns, alpha, C, truth="default"):
    """rows: list of loaded roots (training games, fit level). Returns raw-unit weights."""
    Xs, ys, ws = [], [], []
    for r in rows:
        gt = r["truth"][truth]
        Xm = TMOD.design(columns, r["team"], r["features"], FPS)
        live = np.flatnonzero((gt == H) | (gt == A))[::STEP]
        Xs.append(Xm[live])
        ys.append((gt[live] == H).astype(int))
        ws.append(poss_weights(gt, alpha)[live])
    X_, y_, w_ = np.vstack(Xs), np.concatenate(ys), np.concatenate(ws)
    scale = np.sqrt(np.mean(X_ ** 2, axis=0))
    scale[scale <= 0] = 1.0
    Z = X_ / scale
    Z = np.vstack([Z, -Z])                               # team-flip augmentation
    y_ = np.concatenate([y_, 1 - y_])
    w_ = np.concatenate([w_, w_])
    w_ = w_ / w_.mean()
    m = LogisticRegression(fit_intercept=False, C=C, max_iter=5000, tol=1e-8,
                           random_state=SEED)
    m.fit(Z, y_, sample_weight=w_)
    coef = m.coef_[0] / scale
    return {c: round(float(v), 6) for c, v in zip(columns, coef)}


# --- evaluation --------------------------------------------------------------------------------

def contexts(gt, seen):
    """Boolean masks over frames (live truth only is applied by the caller)."""
    n = len(gt)
    near_change = np.zeros(n, bool)
    after_switch = np.zeros(n, bool)
    short = np.zeros(n, bool)
    rr = runs_of(gt)
    for i, (a, b, k) in enumerate(rr):
        if i > 0:
            near_change[max(0, a - FPS):min(n, a + FPS)] = True
            pk = rr[i - 1][2]
            if k in (H, A) and pk in (H, A) and pk != k:
                after_switch[a:min(b, a + FPS)] = True
        if k in (H, A) and (b - a) < 4 * FPS:
            short[a:b] = True
    return {"all": np.ones(n, bool), "within_1s_of_change": near_change,
            "elsewhere": ~near_change, "first_1s_after_switch": after_switch,
            "possessions_under_4s": short, "ball_seen": seen, "ball_unseen": ~seen}


def frame_metrics(q, gt, seen):
    live = (gt == H) | (gt == A)
    home = gt == H
    qc = np.clip(q, 1e-6, 1 - 1e-6)
    ok = (q >= 0.5) == home
    ll = -np.where(home, np.log(qc), np.log(1 - qc))
    out = {}
    for name, m in contexts(gt, seen).items():
        s = live & m
        out[name] = {"n": int(s.sum()), "acc": round(float(ok[s].mean()), 4) if s.any() else None,
                     "ll": round(float(ll[s].mean()), 4) if s.any() else None}
    return out


def reliability(qs, gts):
    q = np.concatenate([x[(g == H) | (g == A)] for x, g in zip(qs, gts)])
    y = np.concatenate([(g[(g == H) | (g == A)] == H).astype(float) for g in gts])
    edges = np.linspace(0, 1, 11)
    k = np.clip(np.digitize(q, edges[1:-1]), 0, 9)
    rows = []
    for b in range(10):
        s = k == b
        rows.append({"bin": "{:.1f}-{:.1f}".format(edges[b], edges[b + 1]), "n": int(s.sum()),
                     "pred": round(float(q[s].mean()), 4) if s.any() else None,
                     "obs": round(float(y[s].mean()), 4) if s.any() else None})
    n = sum(r["n"] for r in rows)
    ece = sum(r["n"] * abs(r["pred"] - r["obs"]) for r in rows if r["n"]) / max(n, 1)
    return rows, round(float(ece), 4)


def decode_with(inp, probs, c_switch):
    new = copy.copy(inp)
    new.probs = inputs.clean_probs(probs)[0]
    codes, _ = decoders.decode("v2s", new, {"c_switch": c_switch})
    return codes


def score_set(preds, gts):
    out, gl, pl = {}, [], []
    for g in preds:
        r = bscore.score_pair(gts[g], preds[g].astype(gts[g].dtype), 0)
        out[g] = round(float(r["spell_f1_mean"]), 4)
        sep = np.array([bscore.pm.X], dtype=gts[g].dtype)
        gl += [gts[g], sep]
        pl += [preds[g].astype(gts[g].dtype), sep]
    pooled = bscore.score_pair(np.concatenate(gl), np.concatenate(pl), 0)
    out["pooled"] = {k: (round(float(pooled[k]), 4) if isinstance(pooled[k], float)
                         else pooled[k]) for k in (
        "spell_f1_mean", "team_acc_gt_alive", "short_recall_1_5", "short_recall_4",
        "false_splits_per_hour", "missed_stoppages", "invented_stoppages", "restart_team_acc",
        "pred_poss_per_min", "gt_poss_per_min")}
    return out


def q_of(r, columns, weights):
    Xm = TMOD.design(columns, r["team"], r["features"], FPS)
    return TMOD.p_home(Xm, {"columns": columns, "weights": weights})


def q_rule_v1(r):
    p = r["probs_v1"].astype(float)
    return p[:, 0] / np.maximum(p[:, 0] + p[:, 1], 1e-9)


def fit_temperature(z, y):
    """Single temperature T minimising the log-loss of sigmoid(z / T) (grid, deterministic)."""
    grid = np.round(np.arange(0.40, 2.001, 0.01), 2)
    best, best_ll = 1.0, np.inf
    for t in grid:
        q = np.clip(1.0 / (1.0 + np.exp(-np.clip(z / t, -30, 30))), 1e-6, 1 - 1e-6)
        ll = -np.mean(np.where(y == 1, np.log(q), np.log(1 - q)))
        if ll < best_ll:
            best, best_ll = float(t), ll
    return best


def logo_temperatures(variant, cache, truth):
    """Per held-out game g: T fitted on cross-fitted logits of the two training games (fit on
    one, predict the other, both ways; live frames of `truth`, fit level). Nothing of g is used."""
    columns, alpha, C = VARIANTS[variant]
    out = {}
    for g in GAMES:
        a, b = [o for o in GAMES if o != g]
        zs, ys = [], []
        for tr, te in ((a, b), (b, a)):
            w = fit([cache[(tr, FIT_LEVEL)]], columns, alpha, C, truth)
            r = cache[(te, FIT_LEVEL)]
            gt = r["truth"][truth]
            live = (gt == H) | (gt == A)
            Xm = TMOD.design(columns, r["team"], r["features"], FPS)
            zs.append((Xm @ np.array([w[c] for c in columns]))[live])
            ys.append((gt[live] == H).astype(int))
        out[g] = fit_temperature(np.concatenate(zs), np.concatenate(ys))
    return out


def evaluate(variant, levels, cs_grid, cache, temperature=False):
    """LOGO evaluation of one variant ('rule_v1' = the stored scorer, the control). With
    temperature=True the logits are divided by a per-held-out-game temperature fitted without
    that game (logo_temperatures, on the variant's fit truth)."""
    weights, models, temps = {}, {}, {}
    if variant in VARIANTS:
        columns, alpha, C = VARIANTS[variant]
        for g in GAMES:
            weights[g] = fit([cache[(o, FIT_LEVEL)] for o in GAMES if o != g], columns, alpha, C,
                             fit_truth(variant))
        if temperature:
            temps = logo_temperatures(variant, cache, fit_truth(variant))
    elif variant in GBM_VARIANTS:
        anti, sym, alpha = GBM_VARIANTS[variant]
        for g in GAMES:
            models[g] = fit_gbm([cache[(o, FIT_LEVEL)] for o in GAMES if o != g], anti, sym, alpha)
    res = {"variant": variant, "levels": {}}
    for lv in levels:
        qs = {}
        for g in GAMES:
            r = cache[(g, lv)]
            if variant == "rule_v1":
                qs[g] = q_rule_v1(r)
            elif variant in GBM_VARIANTS:
                qs[g] = q_gbm(models[g], r, *GBM_VARIANTS[variant][:2])
            else:
                qs[g] = q_of(r, VARIANTS[variant][0], weights[g])
            if temps:
                z = np.log(np.clip(qs[g], 1e-12, 1)) - np.log(np.clip(1 - qs[g], 1e-12, 1))
                qs[g] = 1.0 / (1.0 + np.exp(-np.clip(z / temps[g], -30, 30)))
        cell = {"frame": {}, "spell": {}}
        for t, suf in (("default", ""), ("labels", "_labels")):
            cell["frame" + suf] = {}
            for g in GAMES:
                r = cache[(g, lv)]
                seen = r["features"]["ball_seen"].to_numpy(float) > 0.5
                cell["frame" + suf][g] = frame_metrics(qs[g], r["truth"][t], seen)
            rel, ece = reliability([qs[g] for g in GAMES],
                                   [cache[(g, lv)]["truth"][t] for g in GAMES])
            cell["reliability" + suf], cell["ece" + suf] = rel, ece
            pooled_ctx = {}
            fr = cell["frame" + suf]
            for ctx in fr[GAMES[0]]:
                n = sum(fr[g][ctx]["n"] for g in GAMES)
                acc = sum(fr[g][ctx]["acc"] * fr[g][ctx]["n"] for g in GAMES
                          if fr[g][ctx]["n"]) / max(n, 1)
                ll = sum(fr[g][ctx]["ll"] * fr[g][ctx]["n"] for g in GAMES
                         if fr[g][ctx]["n"]) / max(n, 1)
                pooled_ctx[ctx] = {"n": n, "acc": round(acc, 4), "ll": round(ll, 4)}
            cell["frame_pooled" + suf] = pooled_ctx
        for cs in cs_grid:
            preds = {}
            for g in GAMES:
                r = cache[(g, lv)]
                pd_ = r["probs_v1"][:, 2].astype(float)
                probs = np.stack([(1 - pd_) * qs[g], (1 - pd_) * (1 - qs[g]), pd_], axis=1)
                preds[g] = decode_with(r["inp"], probs, cs)
            cell["spell"]["cs{:g}".format(cs)] = {
                t: score_set(preds, {g: cache[(g, lv)]["truth"][t] for g in GAMES})
                for t in TRUTHS}
        res["levels"][lv] = cell
    res["weights"] = weights
    res["temperature"] = temps
    return res


def print_eval(res):
    if res.get("temperature"):
        print("{:<14} temperature {}".format(res["variant"], res["temperature"]))
    for lv, cell in res["levels"].items():
        for suf in ("_labels", ""):
            fp = cell["frame_pooled" + suf]
            print("{:<14} {:<6} {:<7} acc {:.4f} ll {:.4f} ece {:.4f} | near {:.3f} sw1 {:.3f} "
                  "short {:.3f} unseen {:.3f}".format(
                      res["variant"], lv, suf.strip("_") or "default", fp["all"]["acc"],
                      fp["all"]["ll"], cell["ece" + suf], fp["within_1s_of_change"]["acc"],
                      fp["first_1s_after_switch"]["acc"], fp["possessions_under_4s"]["acc"],
                      fp["ball_unseen"]["acc"]))
        for cs, by_truth in cell["spell"].items():
            d, lb = by_truth["default"], by_truth["labels"]
            print("    {:<5} labels {:.4f} [{}] default {:.4f} [{}] | pm {:.2f}/{:.2f} sh4 {:.3f} "
                  "sh1.5 {:.3f} fs/h {:.1f}".format(
                      cs, lb["pooled"]["spell_f1_mean"], " ".join("%.3f" % lb[g] for g in GAMES),
                      d["pooled"]["spell_f1_mean"], " ".join("%.3f" % d[g] for g in GAMES),
                      lb["pooled"]["pred_poss_per_min"], lb["pooled"]["gt_poss_per_min"],
                      lb["pooled"]["short_recall_4"], lb["pooled"]["short_recall_1_5"],
                      lb["pooled"]["false_splits_per_hour"]), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="bench-only fit / eval of the team model rule_v2")
    ap.add_argument("action", choices=("eval", "write"))
    ap.add_argument("--variants", nargs="+", default=["rule_v1"] + list(VARIANTS))
    ap.add_argument("--variant", help="write: the variant to fit and store")
    ap.add_argument("--levels", nargs="+", default=list(LEVELS))
    ap.add_argument("--cs", nargs="+", type=float, default=list(C_SWITCH_GRID))
    ap.add_argument("--tag", default="")
    ap.add_argument("--model", default=TMOD.MODEL_NAME, choices=sorted(TMOD.MODELS),
                    help="write: the model whose params file is written (default rule_v2)")
    ap.add_argument("--temperature", action="store_true",
                    help="eval: also divide logits by a leave-one-game-out temperature; write: "
                         "store the temperatures (all games and per held-out game)")
    args = ap.parse_args(argv)
    require_bench_roots([root_of(g, lv) for g in GAMES for lv in set(args.levels) | {FIT_LEVEL}])
    if args.action == "write":
        if args.variant not in VARIANTS:
            ap.error("--variant must be one of " + ", ".join(VARIANTS))
        cache = {(g, FIT_LEVEL): load_root(g, FIT_LEVEL) for g in GAMES}
        columns, alpha, C = VARIANTS[args.variant]
        ft = fit_truth(args.variant)
        rows = [cache[(g, FIT_LEVEL)] for g in GAMES]
        params = {"model": args.model, "variant": args.variant, "columns": columns,
                  "fit": {"alpha": alpha, "C": C, "step": STEP, "fit_level": FIT_LEVEL,
                          "truth": TRUTHS[ft], "augmentation": "team flip",
                          "intercept": False, "solver": "lbfgs", "seed": SEED},
                  "weights": {"all": fit(rows, columns, alpha, C, ft),
                              "logo": {g: fit([cache[(o, FIT_LEVEL)] for o in GAMES if o != g],
                                              columns, alpha, C, ft) for g in GAMES}}}
        if args.temperature:
            # "all": fitted on the cross-fitted logits of all three games (fit on two, predict
            # the third); "logo": per held-out game, without that game (logo_temperatures).
            zs, ys = [], []
            for g in GAMES:
                r = cache[(g, FIT_LEVEL)]
                gt = r["truth"][ft]
                live = (gt == H) | (gt == A)
                w = params["weights"]["logo"][g]
                Xm = TMOD.design(columns, r["team"], r["features"], FPS)
                zs.append((Xm @ np.array([w[c] for c in columns]))[live])
                ys.append((gt[live] == H).astype(int))
            params["temperature"] = {
                "all": fit_temperature(np.concatenate(zs), np.concatenate(ys)),
                "logo": logo_temperatures(args.variant, cache, ft)}
        path = TMOD.params_path(args.model)
        with open(path, "w") as fh:
            json.dump(params, fh, indent=1, sort_keys=True)
            fh.write("\n")
        print("wrote", os.path.relpath(path, RS.REPO))
        return 0
    cache = {(g, lv): load_root(g, lv) for g in GAMES for lv in set(args.levels) | {FIT_LEVEL}}
    os.makedirs(CACHE, exist_ok=True)
    for v in args.variants:
        res = evaluate(v, args.levels, args.cs, cache, args.temperature)
        print_eval(res)
        tag = args.tag + ("_T" if args.temperature else "")
        with open(os.path.join(CACHE, "eval_{}{}.json".format(v, tag)), "w") as fh:
            json.dump(res, fh, indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
