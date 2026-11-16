"""T-32 step 1 (bench only): gate measurement, leave-one-game-out evaluation and the params file of
the per-player tree model (pipeline/models/player_model.py). Called through
``python -m pipeline.models.fit_t32 gate|eval1|write1``; see that module for the scoring.

Protocol (declared before any score):
- Rows: every STEP-th truly live frame (labels truth) of all three levels of the training games;
  both views of every row (team-flip augmentation); possession-balanced weights 1 / L (alpha 1).
- rule_w2 input: for a training game g' inside the fold that holds out g, rule_w2's columns
  refitted (fit_team.fit, its protocol) on the third game alone, so nothing of g reaches the
  training inputs; for the held-out game, rule_w2's stored leave-one-game-out weights. For the
  "all" forest (the match), each game's rule_w2 input is rule_w2's stored logo weights of that
  game, and the match uses rule_w2's "all" weights.
- Learner: HistGradientBoostingClassifier(**GBT_PARAMS), one thread, seed 0.
- Variants: S1e everywhere; S1g trained on gated rows only and used only inside the gate; S1x the
  S1e forest used only inside the gate.
"""
import json
import os
import time

import numpy as np

from pipeline.models import fit_t32 as F
from pipeline.models import fit_team as FT
from pipeline.models import player_model as PM
from pipeline.models import team_model as TMOD

STEP = 5
ALPHA = 1.0
FIT_TRUTH = "labels"
GBT_PARAMS = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100,
                  l2_regularization=1.0, early_stopping=False, random_state=0)
VARIANTS1 = {"S1e": {"train_gated": False, "use_gate": False},
             "S1g": {"train_gated": True, "use_gate": True},
             "S1x": {"train_gated": False, "use_gate": True},
             # ablations of S1g (session 2, honesty checks; never versions): trees without the
             # per-player columns (own*/oth*), and trees without rule_w2's logit (z_w2@*)
             "S1g_np": {"train_gated": True, "use_gate": True, "drop": ("own", "oth")},
             "S1g_nz": {"train_gated": True, "use_gate": True, "drop": ("z_w2@",)}}
BASE = "rule_w2"


def kept_cols(spec):
    """Indices of PM.feature_names() a variant's forest uses (None: all)."""
    drop = spec.get("drop")
    if not drop:
        return None
    return np.array([i for i, c in enumerate(PM.feature_names()) if not c.startswith(drop)])
MAX_FLAGGED = 0.40
FPS = 25.0


def w2_cols(base=BASE):
    return TMOD.load_model(base)["columns"]


def z_of(r, weights, base=BASE):
    cols = w2_cols(base)                # the stage's order (team_model.tree_probs)
    X = TMOD.design(cols, r["team"], r["features"], FPS)
    return X @ np.array([weights[c] for c in cols])


def nested_w2(cache, fold, base=BASE):
    """{training game: base (rule_w2) weights fitted on the other training game only}."""
    tr = [g for g in FT.GAMES if g != fold]
    cols = w2_cols(base)
    return {g: FT.fit([cache[(o, F.FIT_LEVEL)] for o in tr if o != g], cols, 1.0, 1.0, "default")
            for g in tr}


def feats_of(r, ball=None, quants=PM.QUANTS):
    """The features the trees and the gate read: the stored table. (``ball`` named T-41's
    deconvolved-ball variant, which was removed; it must be empty.)"""
    if ball:
        raise ValueError("ball variants (T-41) are not in this checkout")
    return r["features"]


def arrays_of(r, z, quants=PM.QUANTS, ball=None):
    feats = feats_of(r, ball, quants)
    ctx = TMOD.context(feats, FPS)["unseen"]
    return PM.arrays(feats, z, ctx, quants)


def events_of(r):
    import pandas as pd
    return pd.read_parquet(os.path.join(r["root"], "events.parquet"))


def gate_of(r, z, name, ball=None):
    return PM.gate_flags(feats_of(r, ball), events_of(r), PM.sigmoid(z), FPS)[name]


# --- gate measurement ----------------------------------------------------------------------------

def switch_frames(gt, pred):
    """True live H<->A switch frames of gt, split by v6w's outcome (absent within 50 f, late > 10)."""
    pm = F.PJ.pm
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    pr = pm.transitions(filled, runs)
    into = {pm.H: np.array([f for f, _, to in pr if to == pm.H]),
            pm.A: np.array([f for f, _, to in pr if to == pm.A])}
    out = {"absent": [], "late": [], "ok": []}
    for f, fr, to in pm.transitions(gt, runs):
        if fr not in (pm.H, pm.A) or to not in (pm.H, pm.A):
            continue
        c = into[to]
        c = c[np.abs(c - f) <= 50] if len(c) else c
        if not len(c):
            out["absent"].append(f)
        elif c[np.argmin(np.abs(c - f))] - f > 10:
            out["late"].append(f)
        else:
            out["ok"].append(f)
    return {k: np.array(v, dtype=np.int64) for k, v in out.items()}


def act_gate(cache):
    p2 = TMOD.load_model("rule_w2")
    bp = F.base_preds(cache, F.LEVELS)
    res = {}
    for lv in F.LEVELS:
        tot = {}
        for g in FT.GAMES:
            r = cache[(g, lv)]
            z = z_of(r, p2["weights"]["logo"][g])
            flags = PM.gate_flags(r["features"], events_of(r), PM.sigmoid(z), FPS)
            gt = r["truth"]["labels"]
            live = (gt == F.H) | (gt == F.A)
            sw = switch_frames(gt.astype(np.int64), bp[lv][g].astype(np.int64))
            for name, fl in flags.items():
                t = tot.setdefault(name, {"live": 0, "flag": 0, "absent": 0, "absent_in": 0,
                                          "late": 0, "late_in": 0, "ok": 0, "ok_in": 0})
                t["live"] += int(live.sum())
                t["flag"] += int((fl & live).sum())
                for k in ("absent", "late", "ok"):
                    t[k] += len(sw[k])
                    t[k + "_in"] += int(fl[sw[k]].sum()) if len(sw[k]) else 0
        for name, t in tot.items():
            t["share_flagged"] = round(t["flag"] / t["live"], 4)
            t["share_errors_in"] = round((t["absent_in"] + t["late_in"]) /
                                         max(1, t["absent"] + t["late"]), 4)
            t["share_ok_in"] = round(t["ok_in"] / max(1, t["ok"]), 4)
            t["score"] = round(t["share_errors_in"] - t["share_flagged"], 4)
            print("gate {:<3} {:<6} flagged {:.3f} errors_in {:.3f} (absent {}/{} late {}/{}) "
                  "ok_in {:.3f} score {:+.3f}".format(
                      name, lv, t["share_flagged"], t["share_errors_in"], t["absent_in"],
                      t["absent"], t["late_in"], t["late"], t["share_ok_in"], t["score"]),
                  flush=True)
        res[lv] = tot
    ok = {k: v for k, v in res[F.FIT_LEVEL].items() if v["share_flagged"] <= MAX_FLAGGED}
    best = max(ok, key=lambda k: ok[k]["score"]) if ok else None
    res["chosen"] = best
    print("chosen gate (declared criterion, match level):", best)
    F.save("gate.json", res)
    return 0


# --- training ------------------------------------------------------------------------------------

def rows_of(r, z, gate_name=None, quants=PM.QUANTS, ball=None):
    gt = r["truth"][FIT_TRUTH]
    idx = np.flatnonzero((gt == F.H) | (gt == F.A))[::STEP]
    if gate_name:
        idx = idx[gate_of(r, z, gate_name, ball)[idx]]
    ar = arrays_of(r, z, quants, ball)
    half = PM.half_ids(r["meta"]["halves"], len(gt))
    Xh, Xa = PM.views(ar, half, idx, quants)
    y = (gt[idx] == F.H).astype(int)
    w = FT.poss_weights(gt, ALPHA)[idx]
    return Xh, Xa, y, w


def fit_forest(blocks, cols=None):
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xs, ys, ws = [], [], []
    for Xh, Xa, y, w in blocks:
        if cols is not None:
            Xh, Xa = Xh[:, cols], Xa[:, cols]
        Xs += [Xh, Xa]
        ys += [y, 1 - y]
        ws += [w, w]
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    w = np.concatenate(ws)
    t = time.time()
    m = HistGradientBoostingClassifier(**GBT_PARAMS)
    m.fit(X, y, sample_weight=w / w.mean())
    print("  fitted {} rows x {} in {:.0f} s".format(X.shape[0], X.shape[1], time.time() - t),
          flush=True)
    return m


def train_fold(cache, fold, gate_name, cols=None, quants=PM.QUANTS, base=BASE, ball=None):
    zs = nested_w2(cache, fold, base)
    blocks = []
    for g in FT.GAMES:
        if g == fold:
            continue
        for lv in F.LEVELS:
            blocks.append(rows_of(cache[(g, lv)], z_of(cache[(g, lv)], zs[g], base), gate_name,
                                  quants, ball))
    return fit_forest(blocks, cols)


def predict(r, forest, w2_weights, gate_name, p2, cols=None, quants=PM.QUANTS, base=BASE,
            ball=None):
    """probs (n, 3) float32: trees' q (inside the gate if any), rule_w2's q elsewhere and P(D)."""
    z = z_of(r, w2_weights, base)
    base_p, q2 = F.base_probs(r, p2, w2_weights)
    ar = arrays_of(r, z, quants, ball)
    n = len(z)
    half = PM.half_ids(r["meta"]["halves"], n)
    if gate_name:
        g = PM.gate_weight(gate_of(r, z, gate_name, ball), FPS)
        idx = np.flatnonzero(g > 0)
        q = q2.copy()
        qt = PM.q_home(forest, ar, half, idx, cols, quants)
        q[idx] = g[idx] * qt + (1.0 - g[idx]) * q2[idx]
    else:
        q = PM.q_home(forest, ar, half, None, cols, quants)
    pd_ = base_p[:, 2].astype(np.float64)
    out = np.stack([(1 - pd_) * q, (1 - pd_) * (1 - q), pd_], axis=1).astype(np.float32)
    out[:, 2] = base_p[:, 2]
    inside = half > 0
    out[~inside] = np.float32(1.0 / 3.0)
    return out


def check_export(model, forest, r, z, cols=None, quants=PM.QUANTS, ball=None):
    ar = arrays_of(r, z, quants, ball)
    half = PM.half_ids(r["meta"]["halves"], len(z))
    idx = np.arange(0, len(z), 97)
    Xh, _ = PM.views(ar, half, idx, quants)
    if cols is not None:
        Xh = Xh[:, cols]
    a = model.predict_proba(Xh)[:, 1]
    b = PM.sigmoid(PM.forest_raw(forest, Xh))
    return float(np.max(np.abs(a - b)))


def act_eval1(cache, variants, gate_name, cs_grid=None):
    bps = {}
    forests = {}
    keys = sorted({(VARIANTS1[v]["train_gated"], VARIANTS1[v].get("drop", ()),
                    tuple(VARIANTS1[v].get("quants", PM.QUANTS)), VARIANTS1[v].get("base", BASE),
                    VARIANTS1[v].get("ball") or "") for v in variants})
    for train_gated, drop, quants, base, ball in keys:
        p2 = TMOD.load_model(base)
        cols = kept_cols({"drop": drop})
        for fold in FT.GAMES:
            print("fold {} trained {} drop {} quants {} base {} ball {}".format(
                fold, "gated" if train_gated else "everywhere", drop, len(quants), base,
                ball or "stored"), flush=True)
            m = train_fold(cache, fold, gate_name if train_gated else None, cols, quants, base,
                           ball or None)
            forests[(train_gated, drop, quants, base, ball, fold)] = PM.export(m)
            r = cache[(fold, F.FIT_LEVEL)]
            print("  export max |sklearn - numpy| = {:.2e}".format(check_export(
                m, forests[(train_gated, drop, quants, base, ball, fold)], r,
                z_of(r, p2["weights"]["logo"][fold], base), cols, quants, ball or None)))
    res = {}
    for v in variants:
        spec = VARIANTS1[v]
        base, ball = spec.get("base", BASE), spec.get("ball") or ""
        p2 = TMOD.load_model(base)
        if base not in bps:             # the paired baseline: the base model decoded (v6w / v6wd)
            bps[base] = F.base_preds(cache, F.LEVELS, base)
        bp = bps[base]
        res[v] = {"base": base, "ball": ball or "stored"}
        plateau = {}
        for lv in F.LEVELS:
            preds = {}
            for g in FT.GAMES:
                r = cache[(g, lv)]
                qs = tuple(spec.get("quants", PM.QUANTS))
                probs = predict(r, forests[(spec["train_gated"], spec.get("drop", ()), qs, base,
                                            ball, g)],
                                p2["weights"]["logo"][g],
                                gate_name if spec["use_gate"] else None, p2, kept_cols(spec), qs,
                                base, ball or None)
                preds[g] = F.decode(r, probs)
                for cs in cs_grid or ():
                    plateau.setdefault((lv, cs), {})[g] = F.decode(r, probs, cs)
            res[v][lv] = F.score(preds, cache, lv, bp[lv])
            F.print_cell(v, lv, res[v][lv])
            for cs in cs_grid or ():
                sc = FT.score_set(plateau[(lv, cs)],
                                  {g: cache[(g, lv)]["truth"]["labels"] for g in FT.GAMES})
                res[v].setdefault("c_switch", {}).setdefault(lv, {})[str(cs)] = \
                    sc["pooled"]["spell_f1_mean"]
                print("  {} {} c_switch {:g}: labels {:.4f}".format(
                    v, lv, cs, sc["pooled"]["spell_f1_mean"]), flush=True)
        res[v]["mean_labels"] = round(float(np.mean([res[v][lv]["labels"]["pooled"]
                                                     for lv in F.LEVELS])), 4)
        F.save("eval1_{}.json".format(v), res[v])
    return 0


def act_write1(cache, variant, gate_name, model, check=False):
    """Forests on all three games ("all", for the match) and per held-out game ("logo")."""
    spec = VARIANTS1[variant]
    base, ball = spec.get("base", BASE), spec.get("ball")
    p2 = TMOD.load_model(base)
    gname = gate_name if spec["train_gated"] else None
    quants = tuple(spec.get("quants", PM.QUANTS))
    logo = {fold: PM.export(train_fold(cache, fold, gname, None, quants, base, ball))
            for fold in FT.GAMES}
    blocks = []
    for g in FT.GAMES:
        for lv in F.LEVELS:
            r = cache[(g, lv)]
            blocks.append(rows_of(r, z_of(r, p2["weights"]["logo"][g], base), gname, quants,
                                  ball))
    allf = PM.export(fit_forest(blocks))
    params = {"model": model, "kind": "trees", "variant": "T-32 " + variant,
              "base": base, "gate": gate_name if spec["use_gate"] else None,
              "features": PM.feature_names(quants),
              "fit": {"learner": "sklearn HistGradientBoostingClassifier", "params": GBT_PARAMS,
                      "step": STEP, "alpha": ALPHA, "truth": FT.TRUTHS[FIT_TRUTH],
                      "levels": list(F.LEVELS), "augmentation": "team flip (two views)",
                      "symmetrised": True, "touch_set": "w15s", "task": "T-32 step 1",
                      "bench": "smoothed (noise model v3, T-33)",
                      "w2_input": "nested rule_w2 refits in logo folds; stored logo for all"},
              "forests": {"all": allf, "logo": logo}}
    if quants != tuple(PM.QUANTS):          # only S1g2's file carries the key
        params["quants"] = list(quants)
    path = TMOD.params_path(model)
    text = json.dumps(params, sort_keys=True, separators=(",", ":")) + "\n"
    if check:                      # refit determinism: compare with the tracked file, no write
        import hashlib
        with open(path) as fh:
            old = fh.read()
        same = old == text
        print("refit {} ({}): {} bytes, sha256 {} vs tracked {} -> {}".format(
            model, variant, len(text), hashlib.sha256(text.encode()).hexdigest()[:12],
            hashlib.sha256(old.encode()).hexdigest()[:12],
            "BYTE-IDENTICAL" if same else "DIFFERS"), flush=True)
        if not same:               # keep the refit for a structural diff (bench artefact)
            with open(os.path.join(F.CACHE, "refit_{}.json".format(model)), "w") as fh:
                fh.write(text)
        return 0 if same else 1
    with open(path, "w") as fh:
        fh.write(text)
    print("wrote", os.path.relpath(path, F.RS.REPO), "({:.1f} MB)".format(
        os.path.getsize(path) / 1e6))
    return 0


SWEEP_CS = (18.0, 22.0, 30.0, 38.0, 45.0)


def act_sweep(cache, model, grid=SWEEP_CS):
    """Decoder dial sweep for a stored team model (bench only): the stage's own probabilities
    (team_model.score_root_models with the tracked params file: leave-one-game-out forests on each
    bench root), decoded with v2s's decoder at c_switch in ``grid`` with c_stop = c_restart either
    fixed at v2s's value or scaled with c_switch (v2s's ratio, as v8t). Pooled spellF1 under the
    labels, default and confirmed truths, all three levels."""
    from pipeline.decode import decoders
    base = decoders.VERSIONS["v2s"]["dials"]
    ratio = float(base["c_stop"]) / float(base["c_switch"])
    params = TMOD.load_model(model)
    res = {}
    for lv in F.LEVELS:
        probs = {}
        for g in FT.GAMES:
            out, _p1, _m, _i = TMOD.score_root_models(cache[(g, lv)]["root"], {model: params})
            probs[g] = out[model]
        for mode in ("fixed", "scaled"):
            for cs in grid:
                st = float(base["c_stop"]) if mode == "fixed" else ratio * cs
                preds = {}
                for g in FT.GAMES:
                    new = F.copy.copy(cache[(g, lv)]["inp"])
                    new.probs = F.inputs.clean_probs(probs[g])[0]
                    preds[g] = decoders.decode("v2s", new, {"c_switch": cs, "c_stop": st,
                                                            "c_restart": st})[0]
                row = {}
                for t in ("labels", "default", "confirmed"):
                    sc = FT.score_set(preds, {g: cache[(g, lv)]["truth"][t] for g in FT.GAMES})
                    row[t] = sc["pooled"]["spell_f1_mean"]
                    row[t + "_games"] = {g: sc[g] for g in FT.GAMES}
                key = "{}|{:g}|{:g}".format(mode, cs, st)
                res.setdefault(lv, {})[key] = row
                print("sweep {} {:<6} c_switch {:>4g} c_stop=c_restart {:>6g}: labels {:.4f} "
                      "default {:.4f} confirmed {:.4f}".format(model, lv, cs, st, row["labels"],
                                                               row["default"], row["confirmed"]),
                      flush=True)
    F.save("sweep_{}.json".format(model), res)
    return 0


def main(args, load_all):
    if args.action == "sweep":
        return act_sweep(load_all(), args.model)
    if args.action == "gate":
        return act_gate(load_all())
    gate = args.gate
    if args.action == "eval1":
        return act_eval1(load_all(), args.variants or list(VARIANTS1), gate, args.cs)
    if args.action == "write1":
        return act_write1(load_all(), args.variant, gate, args.model)
    if args.action == "refit_check":
        return act_write1(load_all(), args.variant, gate, args.model, check=True)
    raise SystemExit("unknown action")
