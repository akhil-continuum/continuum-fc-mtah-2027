"""Bench-only fitter and evaluator for T-32 (per-player, play-level trial), steps 0 and 1.

    uv run python -m pipeline.models.fit_t32 check                 # in-memory v6w == labels/v6w
    uv run python -m pipeline.models.fit_t32 eval0 [--variants S0w S0a S0b]
    uv run python -m pipeline.models.fit_t32 write0 --variant <S0x>   # -> team_w3_params.json
    uv run python -m pipeline.models.fit_t32 gate                   # busy-gate measurement
    uv run python -m pipeline.models.fit_t32 eval1 [--variants S1e S1g S1x] [--gate G]
    uv run python -m pipeline.models.fit_t32 write1 --variant <S1x> [--gate G]
    uv run python -m pipeline.models.fit_t32 sweep --model rule_p1g    # decoder dial sweep

Reads the nine smoothed level roots artifacts/bench/metrica_g{1,2,3}_{low,match,high} only
(require_bench_roots), which must hold v6w's stage outputs (w15s events) plus the T-32 feature
columns. Never reads the match or the public window.

Base: rule_w2 (team_w2_params.json; its leave-one-game-out weights and its own P(D)) decoded with
v2s's dials, which reproduces labels/v6w.csv (``check``).

Step 0 (linear): rule_w2's protocol unchanged (fit_team.fit: logistic, no intercept, alpha 1, C 1,
team-flip augmentation, every 5th live frame of the match-level roots of the training games,
lbfgs, seed 0), more columns. Leave one game out for every score.

Step 1 (trees): pipeline/models/player_model.py's design and learner, trained on all three levels
of the two training games, scored on the held-out game. See that module.

Scores: spellF1 under the labels and default truths after the v2s decoder (c_switch 18), pooled
and per game, a paired block bootstrap against the in-memory v6w (bench/paired.py's
scheme B: 60 s blocks, stratified by game, 2000 draws, seed 20261113), and the judge's error
counts. Results in artifacts/bench/_t32/.
"""
import argparse
import copy
import hashlib
import json
import os
import pickle
import sys
import time

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np                              # noqa: E402
import pandas as pd                             # noqa: E402

from pipeline.decode import decoders, inputs    # noqa: E402
from pipeline.io.roots import require_bench_roots  # noqa: E402
from pipeline.models import fit_team as FT      # noqa: E402
from pipeline.models import rule_scorer as RS   # noqa: E402
from pipeline.models import team_model as TMOD  # noqa: E402

GAMES = FT.GAMES
LEVELS = FT.LEVELS
FIT_LEVEL = "match"
C_SWITCH = 18.0
H, A, D, X = FT.H, FT.A, FT.D, FT.X
CACHE = os.path.join(RS.ARTIFACTS, "bench", "_t32")
SCORE_TRUTHS = ("labels", "default")
DRAWS, BLOCK_S, BOOT_SEED = 2000, 60.0, 20261113

from bench import paired as PJ                  # noqa: E402

# --- step 0 variants (declared before scoring) -------------------------------------------------
W2_COLS = list(FT.VARIANTS["shape_a1"][0])
WIDE = ["shape_{}_diff_{}".format(s, w) for s in ("fwd_vel", "speed", "press", "approach")
        for w in ("c6", "c8")] + ["shape_{}_rel_{}".format(s, w) for s in ("depth", "width")
                                  for w in ("c6", "c8")]
LBLONG = ["{}_pfirst_diff_{}".format(p, w) for p in ("lb2", "lb3") for w in ("c05", "c1")]
TOUCH = ["pc_touch_diff_{}{}".format(s, w) for s in ("b", "a") for w in (1, 2, 4)]
PLAYCTX = ["pc_near{}_diff_c1".format(r) for r in (2, 5, 10)] + \
    ["pc_duel_lead_diff_c05", "pc_gap_last_near_diff", "pc_gap_next_near_diff",
     "pc_gap_next_n5_diff", "pc_line_rel_c1", "pc_line_rel_c4", "pc_linevel_diff_c1",
     "pc_linevel_diff_c2"]
VARIANTS0 = {"S0w": (W2_COLS + WIDE + LBLONG + TOUCH, "default"),
             "S0a": (W2_COLS + WIDE + LBLONG + TOUCH + PLAYCTX, "default"),
             "S0b": (W2_COLS + WIDE + LBLONG + TOUCH + PLAYCTX, "labels")}


# --- loading -----------------------------------------------------------------------------------

def _digest(root):
    h = hashlib.sha256()
    for f in ("events.parquet", "ball.parquet", "players.parquet"):
        with open(os.path.join(root, f), "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:16]


def load_root(game, level):
    """Terms, rule_v1 probs, decode inputs (cached by the stage files' content), features, truths."""
    root = FT.root_of(game, level)
    os.makedirs(os.path.join(CACHE, "cache"), exist_ok=True)
    path = os.path.join(CACHE, "cache", "root_{}_{}_{}.pkl".format(game, level, _digest(root)))
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
        out = {"meta": meta, "dead": dead, "team": team, "probs_v1": probs_v1, "inp": inp}
        with open(path, "wb") as fh:
            pickle.dump(out, fh)
        print("cached {} {} in {:.0f} s".format(game, level, time.time() - t), flush=True)
    out["root"] = root
    out["game"] = game
    out["level"] = level
    out["features"] = pd.read_parquet(os.path.join(root, "features.parquet"))
    out["truth"] = {k: FT.truth_codes(root, k) for k in FT.TRUTHS}
    return out


def load_all(levels=LEVELS):
    need = sorted(set(levels) | {FIT_LEVEL})
    require_bench_roots([FT.root_of(g, lv) for g in GAMES for lv in need])
    return {(g, lv): load_root(g, lv) for g in GAMES for lv in need}


# --- base model (rule_w2) and its P(D) ---------------------------------------------------------

def w2_params():
    return TMOD.load_model("rule_w2")


def base_probs(r, params, weights):
    """(probs float32 (n, 3), q) of a linear team model with rule_w2's P(D) block."""
    fps = float(r["meta"]["fps"])
    n = int(r["meta"]["n_frames"])
    cols = list(params["columns"]) if set(params["columns"]) == set(weights) else list(weights)
    q = TMOD.p_home(TMOD.design(cols, r["team"], r["features"], fps),
                    {"columns": cols, "weights": weights})
    dead = params["dead"]
    base = RS.combine(r["dead"], r["team"], n, r["meta"]["halves"],
                      dead_w=dead["weights"]["logo"][r["game"]], dead_bias=float(dead["bias"]))
    return TMOD.compose(base, q, r["meta"]["halves"]), q


def w2_probs(r, params=None):
    params = params or w2_params()
    return base_probs(r, params, params["weights"]["logo"][r["game"]])


def decode(r, probs, c_switch=C_SWITCH):
    return FT.decode_with(r["inp"], probs, c_switch)


# --- scoring -----------------------------------------------------------------------------------

def score(preds, cache, level, base_preds=None):
    """Pooled and per-game spellF1 per truth, error counts (labels truth), and the paired block
    bootstrap of the difference to base_preds (if given). preds: {game: codes}."""
    out = {}
    block = int(BLOCK_S * 25)
    for t in SCORE_TRUTHS:
        gts = {g: cache[(g, level)]["truth"][t] for g in GAMES}
        s = FT.score_set(preds, gts)
        cell = {"pooled": s["pooled"]["spell_f1_mean"], "games": {g: s[g] for g in GAMES},
                "poss_per_min": s["pooled"]["pred_poss_per_min"]}
        tabs = {g: PJ.game_table(gts[g].astype(np.int64), preds[g].astype(np.int64), block)
                for g in GAMES}
        if t == "labels":
            per = {g: PJ.error_counts(gts[g].astype(np.int64), tabs[g]) for g in GAMES}
            cell["errors"] = {k: int(sum(per[g][k] for g in GAMES)) for k in per[GAMES[0]]}
        if base_preds is not None:
            btabs = {g: PJ.game_table(gts[g].astype(np.int64), base_preds[g].astype(np.int64),
                                      block) for g in GAMES}
            cell["paired"] = paired_diff(tabs, btabs)
        out[t] = cell
    return out


def paired_diff(tabs, btabs):
    """paired.py scheme B for one pair: point difference, 5% and 95%, per game."""
    rng = np.random.RandomState(PJ.SEED + 1)
    picks = {g: rng.randint(0, len(tabs[g]["gt_blk"]), (DRAWS, len(tabs[g]["gt_blk"])))
             for g in GAMES}

    def draws(tb):
        tp = np.zeros((DRAWS, len(PJ.pm.SPELL_IOUS)))
        ng = np.zeros(DRAWS)
        npr = np.zeros(DRAWS)
        pg = {}
        for g in GAMES:
            t, P = tb[g], picks[g]
            tpg, ngg, npg = t["tp_blk"][P].sum(1), t["gt_blk"][P].sum(1), t["pr_blk"][P].sum(1)
            tp, ng, npr = tp + tpg, ng + ngg, npr + npg
            pg[g] = (PJ.f1_from(t["flags"].sum(1)[None], np.array([len(t["gs"])]),
                                np.array([t["n_pred"]]))[0], PJ.f1_from(tpg, ngg, npg))
        return PJ.f1_from(tp, ng, npr), pg

    def point(tb):
        fl = np.hstack([tb[g]["flags"] for g in GAMES])
        return float(PJ.f1_from(fl.sum(1)[None], np.array([fl.shape[1]]),
                                np.array([float(sum(tb[g]["n_pred"] for g in GAMES))]))[0])

    dc, pgc = draws(tabs)
    db, pgb = draws(btabs)
    d = dc - db
    games = {g: {"diff": round(float(pgc[g][0] - pgb[g][0]), 4),
                 "lo": round(float(np.percentile(pgc[g][1] - pgb[g][1], 5)), 4),
                 "hi": round(float(np.percentile(pgc[g][1] - pgb[g][1], 95)), 4)} for g in GAMES}
    return {"diff": round(point(tabs) - point(btabs), 4),
            "lo": round(float(np.percentile(d, 5)), 4),
            "hi": round(float(np.percentile(d, 95)), 4), "games": games}


def print_cell(name, level, cell):
    lb, df = cell["labels"], cell["default"]
    e = lb["errors"]
    txt = "{:<6}{:<6} lab {:.4f} [{}] def {:.4f}".format(
        name, level, lb["pooled"], " ".join("%.3f" % lb["games"][g] for g in GAMES), df["pooled"])
    if "paired" in lb:
        p, q = lb["paired"], df["paired"]
        txt += " | dlab {:+.4f} [{:+.4f},{:+.4f}] ({}) ddef {:+.4f} [{:+.4f},{:+.4f}]".format(
            p["diff"], p["lo"], p["hi"],
            " ".join("%+.3f" % p["games"][g]["diff"] for g in GAMES), q["diff"], q["lo"], q["hi"])
    txt += " | pm {:.2f} fs {} m1.5 {} m4 {} late {} none2s {}".format(
        lb["poss_per_min"], e["false_split"], e["missed_short_1_5"], e["missed_short_4"],
        e["late_switch_gt10"], e["switch_absent_50"])
    print(txt, flush=True)


def save(name, obj):
    os.makedirs(CACHE, exist_ok=True)
    with open(os.path.join(CACHE, name), "w") as fh:
        json.dump(obj, fh, indent=1, default=float)


# --- actions -----------------------------------------------------------------------------------

def base_preds(cache, levels, model="rule_w2"):
    """The base linear model decoded with v2s's dials (rule_w2: v6w; rule_w2d: v6wd)."""
    p2 = TMOD.load_model(model)
    return {lv: {g: decode(cache[(g, lv)], w2_probs(cache[(g, lv)], p2)[0]) for g in GAMES}
            for lv in levels}


def act_check(cache):
    bp = base_preds(cache, LEVELS)
    for lv in LEVELS:
        for g in GAMES:
            lab = FT.truth_codes.__globals__["pd"].read_csv(
                os.path.join(FT.root_of(g, lv), "labels", "v6w.csv"), dtype={"label": str})
            codes = np.select([lab["label"].to_numpy().astype(str) == c for c in "HADX"],
                              [H, A, D, X], -1)
            print("check {} {} equal to labels/v6w.csv: {}".format(
                g, lv, bool(np.array_equal(codes, bp[lv][g]))), flush=True)
    res = {lv: score(bp[lv], cache, lv) for lv in LEVELS}
    for lv in LEVELS:
        print_cell("v6w", lv, res[lv])
    save("eval_v6w.json", res)
    return 0


def fit0(cache, cols, truth, games):
    return FT.fit([cache[(g, FIT_LEVEL)] for g in games], cols, 1.0, 1.0, truth)


def act_eval0(cache, variants):
    bp = base_preds(cache, LEVELS)
    p2 = w2_params()
    res = {}
    for v in variants:
        cols, truth = VARIANTS0[v]
        w = {g: fit0(cache, cols, truth, [o for o in GAMES if o != g]) for g in GAMES}
        res[v] = {}
        for lv in LEVELS:
            preds = {g: decode(cache[(g, lv)], base_probs(cache[(g, lv)], p2, w[g])[0])
                     for g in GAMES}
            res[v][lv] = score(preds, cache, lv, bp[lv])
            print_cell(v, lv, res[v][lv])
        res[v]["mean_labels"] = round(float(np.mean([res[v][lv]["labels"]["pooled"]
                                                     for lv in LEVELS])), 4)
        print("{} mean labels spellF1 over levels {:.4f}".format(v, res[v]["mean_labels"]))
        save("eval0_{}.json".format(v), res[v])
    return 0


def act_write0(cache, variant, model="rule_w3"):
    cols, truth = VARIANTS0[variant]
    p2 = w2_params()
    columns, alpha, C = cols, 1.0, 1.0
    params = {"model": model, "variant": "T-32 " + variant, "columns": list(columns),
              "fit": {"alpha": alpha, "C": C, "step": FT.STEP, "fit_level": FIT_LEVEL,
                      "truth": FT.TRUTHS[truth], "augmentation": "team flip", "intercept": False,
                      "solver": "lbfgs", "seed": FT.SEED, "touch_set": "w15s",
                      "bench": "smoothed (noise model v3, T-33)", "task": "T-32 step 0"},
              "weights": {"all": fit0(cache, cols, truth, GAMES),
                          "logo": {g: fit0(cache, cols, truth, [o for o in GAMES if o != g])
                                   for g in GAMES}},
              "dead": p2["dead"]}
    path = TMOD.params_path(model)
    with open(path, "w") as fh:
        json.dump(params, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print("wrote", os.path.relpath(path, RS.REPO))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="T-32 bench-only fits and evaluation")
    ap.add_argument("action", choices=("check", "eval0", "write0", "gate", "eval1", "write1",
                                       "sweep", "refit_check"))
    ap.add_argument("--variants", nargs="+")
    ap.add_argument("--variant")
    ap.add_argument("--gate")
    ap.add_argument("--model", help="write1: the model name whose params file is written")
    ap.add_argument("--cs", nargs="+", type=float, help="eval1: also decode at these c_switch")
    args = ap.parse_args(argv)
    if args.action in ("check",):
        return act_check(load_all())
    if args.action == "eval0":
        return act_eval0(load_all(), args.variants or list(VARIANTS0))
    if args.action == "write0":
        return act_write0(load_all((FIT_LEVEL,)), args.variant)
    from pipeline.models import fit_t32_step1 as S1
    return S1.main(args, load_all)


if __name__ == "__main__":
    sys.exit(main())
