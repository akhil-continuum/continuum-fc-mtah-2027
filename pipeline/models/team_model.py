"""Team models rule_v2 to rule_v5: rule_v1's P(D) unchanged, P(H | live) from a logistic model on
rule_v1's team terms plus the team-shape features (proposal P-9, possession-balanced fit P-10).

    uv run python -m pipeline.models.team_model match [--models rule_v2 ...]   # aggregates only
    uv run python -m pipeline.models.team_model bench [--models ...] [--root DIR ...] [--level L]

Writes ``<root>/probs/<model>.npy`` float32 (n_frames, 3) = P(H), P(A), P(D) (CONTRACTS.md) for
each model asked for (default rule_v2 only). Never writes probs/rule_v1.npy.

    rule_v2  team_v2_params.json  terms + shape, fitted on truth.csv (first touch; T-23)
    rule_v5  team_v5_params.json  contact-type terms from the events table (P-15;
                                  ``event_terms`` below), used only if its params file exists
                                  (none is tracked)
    rule_w1  team_w1_params.json  T-35: rule_v1 refitted on the smoothed bench, w15s touches
                                  (team columns = rule_v1's terms, own P(D) weights)
    rule_w2  team_w2_params.json  T-35: rule_v2's columns refitted there, P(D) = rule_w1's

A params file may carry a "dead" block {"bias", "weights": {"all", "logo"}} over rule_v1's
dead terms (rule_params.DEAD_DEFAULTS); the model's P(D) is then sigmoid(bias + sum w * term)
instead of rule_v1's (rule_w1, rule_w2; written by ``pipeline.models.refit_w``).

    P(D)        = rule_v1's P(D), exactly (same terms, same weights, same float32 column)
    P(H | live) = sigmoid(sum_k w_k * x_k)

Every input x_k is antisymmetric under swapping the teams and flipping the pitch (rule_v1's
team terms, ``*_diff_*`` / ``*_rel_*`` shape columns, and those times a symmetric context such as
the ball-unseen share), and there is no intercept, so the model has no built-in lean to home or
away: P(H | live) of the mirrored game is exactly 1 - P(H | live). Missing shape values (NaN,
too few tracked players, T7) contribute 0 (no evidence).

The weights w_k (raw units of each column) live in ``pipeline/models/team_v2_params.json`` (tracked text),
written by the bench-only fitter ``python -m pipeline.models.fit_team``. Nothing is fitted here,
and nothing is ever fitted on the competition match. The match and roots of unknown games use
the weights fitted on all three bench games; a bench root of game metrica_g<k> uses the weights
fitted without that game (leave-one-game-out), so bench scores of rule_v2 are held out.
"""
import argparse
import json
import os
import re
import sys

import numpy as np

from pipeline.io.roots import require_bench_roots
from pipeline.models import rule_scorer as RS
from pipeline.models import terms as TM
from pipeline.report import match_entry, quiet_warnings, summary

MODEL_NAME = "rule_v2"
HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(HERE, "team_v2_params.json")
MODELS = {"rule_v2": "team_v2_params.json", "rule_v5": "team_v5_params.json",
          # T-35 (P-17c): refits on the smoothed bench with w15s touch events
          # (pipeline/models/refit_w.py). Both carry their own P(D) ("dead" block, below):
          # rule_w1 = rule_v1's dead and team terms refitted (the v2s analogue);
          # rule_w2 = rule_v2's columns refitted, with rule_w1's P(D) (the v6s analogue).
          "rule_w1": "team_w1_params.json", "rule_w2": "team_w2_params.json",
          # T-32 step 1: per-player trees (player_model.py, fit_t32 write1), base rule_w2,
          # trained and used inside the busy gate G4 only (v11g).
          "rule_p1g": "team_p1g_params.json"}
SIZING_ONLY = ()
CONTEXT_HALF_S = 1.0        # half-width of the centred window for the ball-unseen share
LOOSE_D_M = 2.0             # "loose": nobody within a stride or two of the ball (P30)
LOOSE_HALF_S = 0.5          # half-width of the centred window for the loose-ball share

# Contact-type terms from the events table (P-15; rulebook P7, P17, P18, P21). Each is a signed
# step (+1 home, -1 away) over a short window around an event; the windows are physical choices.
EVT_REDIRECT_DV = 8.0       # m/s: a touch that changes the ball's velocity by at least this
EVT_PREV_TOUCH_S = 2.0      # ... soon after the other team's touch (a cut-out or block, P18)
EVT_AFTER_S = 1.0           # evidence lasts this long after the contact
EVT_BEFORE_S = 0.5          # header-like: starts this long before it (P17, P30 date it early)
EVT_FLIGHT_MIN_S = 1.0      # a flight at least this long ending at a touch: header-like (P17)
EVT_DEFL_COLLECT_S = 2.0    # deflection followed by the deflecting team's control within (P7)


def params_path(model):
    return os.path.join(HERE, MODELS[model])


def available_models():
    return [m for m in MODELS if m not in SIZING_ONLY and os.path.isfile(params_path(m))]


def event_terms(events, n, fps):
    """Signed contact-type columns (evt_redirect, evt_header, evt_defl_collect) from events."""
    out = {k: np.zeros(n) for k in ("evt_redirect", "evt_header", "evt_defl_collect")}
    if events is None or not len(events):
        return out
    after, before = int(round(EVT_AFTER_S * fps)), int(round(EVT_BEFORE_S * fps))

    def mark(name, a, b, team):
        a, b = max(0, int(a)), min(n, int(b))
        if b > a and team in (0, 1):
            out[name][a:b] = 1.0 if team == 0 else -1.0

    tch = events[events["kind"] == "touch"].sort_values("frame", kind="mergesort")
    ft = tch["frame"].to_numpy().astype(np.int64)
    tt = tch["team"].to_numpy().astype(np.int64)
    dv = tch["detail"].astype(str).str.extract(r"dv=([0-9.]+)")[0].astype(float).to_numpy()
    for i in range(1, len(ft)):
        if tt[i] in (0, 1) and tt[i - 1] in (0, 1) and tt[i] != tt[i - 1] and \
                ft[i] - ft[i - 1] <= EVT_PREV_TOUCH_S * fps and \
                np.nan_to_num(dv[i]) >= EVT_REDIRECT_DV:
            mark("evt_redirect", ft[i], ft[i] + after, tt[i])
    fl = events[events["kind"] == "flight"]
    for f0, f1 in zip(fl["frame"].to_numpy(), fl["end_frame"].to_numpy()):
        if f1 - f0 < EVT_FLIGHT_MIN_S * fps:
            continue
        k = np.flatnonzero((ft >= f1 - 2) & (ft <= f1 + 3) & ((tt == 0) | (tt == 1)))
        if len(k):
            mark("evt_header", ft[k[0]] - before, ft[k[0]] + after, tt[k[0]])
    ctrl = events[events["kind"] == "control"].sort_values("first_touch_frame", kind="mergesort")
    cf = ctrl["first_touch_frame"].to_numpy().astype(np.int64)
    ct = ctrl["team"].to_numpy().astype(np.int64)
    dfl = events[events["kind"] == "deflection"]
    for f, t in zip(dfl["frame"].to_numpy(), dfl["team"].to_numpy()):
        k = np.searchsorted(cf, f)
        if k < len(cf) and cf[k] - f <= EVT_DEFL_COLLECT_S * fps and ct[k] == t:
            mark("evt_defl_collect", f, cf[k] + 1, int(t))
    return out


# --- design ----------------------------------------------------------------------------------

def context(features, fps):
    """Symmetric per-frame context in [0, 1] used to gate columns (``<col>*<ctx>``)."""
    seen = np.nan_to_num(features["ball_seen"].to_numpy(float), nan=0.0)
    unseen = 1.0 - TM.centred_share(seen, int(round(CONTEXT_HALF_S * fps)))
    ctx = {"unseen": unseen}
    if "in_flight" in features and "near_d_h" in features:
        # loose ball (P30): ball seen and in flight or no player of either team within
        # LOOSE_D_M; centred share over +-LOOSE_HALF_S (symmetric under swapping the teams)
        dmin = np.fmin(features["near_d_h"].to_numpy(float), features["near_d_a"].to_numpy(float))
        with np.errstate(invalid="ignore"):
            raw = (seen > 0.5) & ((features["in_flight"].to_numpy(float) > 0.5) |
                                  (np.nan_to_num(dmin, nan=0.0) > LOOSE_D_M))
        ctx["loose"] = TM.centred_share(raw.astype(float), int(round(LOOSE_HALF_S * fps)))
    return ctx


def column(name, team, features, ctx):
    """One design column by name: a rule_v1 team term, a features.parquet column (NaN -> 0), or
    either times a context ('<base>*<context>')."""
    if "*" in name:
        base, c = name.split("*", 1)
        return column(base, team, features, ctx) * ctx[c]
    if name in team:
        return np.asarray(team[name], dtype=float)
    return np.nan_to_num(features[name].to_numpy(float), nan=0.0)


def design(columns, team, features, fps):
    ctx = context(features, fps)
    return np.stack([column(c, team, features, ctx) for c in columns], axis=1)


def p_home(X, spec):
    w = np.array([spec["weights"][c] for c in spec["columns"]])
    t = float(spec.get("temperature", 1.0))     # optional; absent (1.0) for rule_v2
    return RS.sigmoid(X @ w if t == 1.0 else (X @ w) / t)


def compose(probs_v1, q, halves):
    """rule_v1's P(D) column exactly, P(H) / P(A) from q; (1/3, 1/3, 1/3) outside the halves."""
    pd_ = probs_v1[:, 2].astype(np.float64)
    out = np.stack([(1 - pd_) * q, (1 - pd_) * (1 - q), pd_], axis=1).astype(np.float32)
    out[:, 2] = probs_v1[:, 2]
    n = len(out)
    inside = np.zeros(n, dtype=bool)
    for a, b in halves:
        inside[max(0, int(a)):min(n, int(b))] = True
    out[~inside] = np.float32(1.0 / 3.0)
    return out


# --- parameters ------------------------------------------------------------------------------

def load_params(path=PARAMS_PATH):
    with open(path) as handle:
        return json.load(handle)


def load_model(model):
    return load_params(params_path(model))


def needs_events(columns):
    return any(c.split("*")[0].startswith("evt_") for c in columns)


def spec_for(params, root=None):
    """Weights for a root: leave-one-game-out for a bench root of a fitted game, else 'all'."""
    w = params["weights"]["all"]
    if root is not None:
        m = re.search(r"(metrica_g\d+)", os.path.basename(os.path.normpath(root)))
        if m and m.group(1) in params["weights"].get("logo", {}):
            w = params["weights"]["logo"][m.group(1)]
    spec = {"columns": params["columns"], "weights": w}
    if "temperature" in params:                 # optional single temperature
        t = params["temperature"]
        spec["temperature"] = t["all"] if w is params["weights"]["all"] else \
            t["logo"][m.group(1)]
    return spec


def dead_spec_for(params, root=None):
    """(bias, dead weights) of a model's own P(D), or None for models that keep rule_v1's P(D)
    (every model without a "dead" block: rule_v2 to rule_v5). Same rule as spec_for: a bench
    root of a fitted game gets the weights fitted without that game, else "all"."""
    dead = params.get("dead")
    if not dead:
        return None
    w = dead["weights"]["all"]
    if root is not None:
        m = re.search(r"(metrica_g\d+)", os.path.basename(os.path.normpath(root)))
        if m and m.group(1) in dead["weights"].get("logo", {}):
            w = dead["weights"]["logo"][m.group(1)]
    return float(dead["bias"]), w


# --- stage -----------------------------------------------------------------------------------

def score_root(root, params=None, compute_dead=True):
    """(rule_v2 probs, rule_v1 probs, meta, info) for a root."""
    params = params or load_params()
    out, probs_v1, meta, info = score_root_models(root, {MODEL_NAME: params}, compute_dead)
    return out[MODEL_NAME], probs_v1, meta, info


def score_root_models(root, params_by_model, compute_dead=True):
    """({model: probs}, rule_v1 probs, meta, info) for a root; rule_v1's terms computed once."""
    data = RS.read_root(root, compute_dead=compute_dead)
    meta = data["meta"]
    n = int(meta["n_frames"])
    fps = float(meta["fps"])
    dead, team, info = RS.compute_terms(data)
    probs_v1 = RS.combine(dead, team, n, meta["halves"])
    info["dead_features"] = data["dead_source"]
    out, evt = {}, None
    for model, params in params_by_model.items():
        if params.get("kind") == "trees":       # T-32 step 1: pipeline/models/player_model.py
            out[model] = tree_probs(params, root, data, dead, team, n, fps, meta)
            m = re.search(r"(metrica_g\d+)", os.path.basename(os.path.normpath(root or "")))
            info["weights"] = "leave_one_out" if m and m.group(1) in \
                params["forests"].get("logo", {}) else "all"
            continue
        spec = spec_for(params, root)
        terms = team
        if needs_events(spec["columns"]):
            if evt is None:
                evt = event_terms(data["events"], n, fps)
            terms = dict(team, **evt)
        X = design(spec["columns"], terms, data["features"], fps)
        base = probs_v1
        ds = dead_spec_for(params, root)
        if ds is not None:          # the model's own P(D) (T-35); rule_v1's terms, own weights
            base = RS.combine(dead, team, n, meta["halves"], dead_w=ds[1], dead_bias=ds[0])
        out[model] = compose(base, p_home(X, spec), meta["halves"])
        info["weights"] = "all" if spec["weights"] is params["weights"]["all"] \
            else "leave_one_out"
    return out, probs_v1, meta, info


def tree_probs(params, root, data, dead, team, n, fps, meta):
    """T-32 per-player tree model: its base linear model's P(D) and q (leave-one-game-out on a
    bench root of a fitted game, else "all"), the stored forest's symmetrised q (inside the gate
    only, if the params name one), composed as compose() does."""
    from pipeline.models import player_model as PM
    bparams = load_model(params["base"])
    spec = spec_for(bparams, root)
    X = design(spec["columns"], team, data["features"], fps)
    z = X @ np.array([spec["weights"][c] for c in spec["columns"]])
    q2 = RS.sigmoid(z)
    ds = dead_spec_for(bparams, root)
    base = RS.combine(dead, team, n, meta["halves"], dead_w=ds[1], dead_bias=ds[0]) \
        if ds is not None else RS.combine(dead, team, n, meta["halves"])
    forest = params["forests"]["all"]
    if root is not None:
        m = re.search(r"(metrica_g\d+)", os.path.basename(os.path.normpath(root)))
        if m and m.group(1) in params["forests"].get("logo", {}):
            forest = params["forests"]["logo"][m.group(1)]
    quants = tuple(params.get("quants", PM.QUANTS))     # optional; S1g2 adds brake / turn
    feats = data["features"]
    ar = PM.arrays(feats, z, context(feats, fps)["unseen"], quants)
    half = PM.half_ids(meta["halves"], n)
    if params.get("gate"):
        flags = PM.gate_flags(feats, data["events"], q2, fps)[params["gate"]]
        g = PM.gate_weight(flags, fps)
        idx = np.flatnonzero(g > 0)
        q = np.asarray(q2, dtype=np.float64).copy()
        q[idx] = g[idx] * PM.q_home(forest, ar, half, idx, None, quants) + (1.0 - g[idx]) * q2[idx]
    else:
        q = PM.q_home(forest, ar, half, None, None, quants)
    return compose(base, q, meta["halves"])


def run_bench(roots, truth=None, write=True, models=(MODEL_NAME,)):
    params = {m: load_model(m) for m in models}
    for root in roots:
        out, probs_v1, meta, info = score_root_models(root, params)
        name = os.path.basename(os.path.normpath(root))
        for model, probs in out.items():
            if write:
                RS.write_probs(root, probs, model)
            stats = {"weights": info["weights"],
                     "p_dead_equal_rule_v1": bool(np.array_equal(probs[:, 2], probs_v1[:, 2]))}
            stats.update(RS.aggregates(probs, RS.graded_mask(meta, len(probs))))
            tp = RS.find_truth(root, truth)
            if tp:
                m = RS.bench_metrics(probs, tp)
                stats.update({k: v for k, v in RS.flat_metrics(m).items()
                              if k in ("team_acc_live", "logloss_team_live", "logloss_3class",
                                       "f1_dead", "ece_team")
                              or k.startswith("rel_team")})
            tag = name if model == MODEL_NAME else "{}.{}".format(name, model)
            summary(_name="team_model.bench.{}".format(tag), **stats)
    # the model count, not the joined names: with every tracked model the list passes the
    # report's 80-character limit for strings (rule_w1d / rule_w2d made it 83, 15 Nov)
    summary(_name="team_model.bench", bench_roots_run=len(roots), n_models=len(models))
    return 0


@match_entry
def run_match(models=(MODEL_NAME,)):
    quiet_warnings()
    root = os.path.join(RS.ARTIFACTS, "match")
    out, probs_v1, meta, info = score_root_models(
        root, {m: load_model(m) for m in models}, compute_dead=False)
    for model, probs in out.items():
        RS.write_probs(root, probs, model)
    mask = RS.graded_mask(meta, len(probs_v1))
    a1 = RS.aggregates(probs_v1, mask)
    for model in models:
        if model != MODEL_NAME:
            pm = out[model]
            am = RS.aggregates(pm, mask)
            lv = mask & (pm[:, 2] < 0.5)
            qm = pm[lv, 0] / np.maximum(pm[lv, 0] + pm[lv, 1], 1e-9)
            summary(_name="team_model.{}".format(model), model=model, weights=info["weights"],
                    p_dead_equal_rule_v1=bool(np.array_equal(pm[:, 2], probs_v1[:, 2])),
                    top_class_share=am["top_class_share"],
                    mean_p_home_given_live=round(float(qm.mean()), 4) if lv.any() else None,
                    share_live_top_h=round(float(np.mean(qm >= 0.5)), 4) if lv.any() else None,
                    flat_share=am["flat_share"])
    if any(m in ("rule_v4", "rule_v5") for m in models):
        feats = RS.read_root(root, compute_dead=False)["features"]
        if "lb_valid" in feats:
            lbv = feats["lb_valid"].to_numpy(float)[mask] > 0.5
            clr = feats["lb_clear_diff_c0"].to_numpy(float)[mask][lbv]
            summary(_name="team_model.loose_ball", lb_valid_share_graded=round(float(
                        lbv.mean()), 4),
                    lb_extrap_share_graded=round(float(
                        feats["lb_extrap"].to_numpy(float)[mask].mean()), 4),
                    lb_clear_share_of_valid=round(float(np.mean(np.abs(clr) > 0.5)), 4)
                    if lbv.any() else None,
                    lb_clear_home_share_of_clear=round(float(np.mean(clr[np.abs(clr) > 0.5]
                                                                     > 0)), 4)
                    if (np.abs(clr) > 0.5).any() else None)
    if MODEL_NAME not in out:
        return 0
    probs = out[MODEL_NAME]
    a2, a1 = RS.aggregates(probs, mask), RS.aggregates(probs_v1, mask)
    live = mask & (probs[:, 2] < 0.5)
    q2 = probs[live, 0] / np.maximum(probs[live, 0] + probs[live, 1], 1e-9)
    q1 = probs_v1[live, 0] / np.maximum(probs_v1[live, 0] + probs_v1[live, 1], 1e-9)
    feats = RS.read_root(root, compute_dead=False)["features"]
    cover = {"shape_valid_share_graded": round(float(
                 feats["shape_valid"].to_numpy(float)[mask].mean()), 4),
             "players_shape_finite_share_graded": round(float(np.isfinite(
                 feats["shape_fwd_vel_diff_c1"].to_numpy(float)[mask]).mean()), 4),
             "pressure_finite_share_graded": round(float(np.isfinite(
                 feats["shape_press_diff_c1"].to_numpy(float)[mask]).mean()), 4)}
    summary(_name="team_model", model=MODEL_NAME, weights=info["weights"],
            dead_features=info["dead_features"],
            p_dead_equal_rule_v1=bool(np.array_equal(probs[:, 2], probs_v1[:, 2])),
            top_class_share_rule_v2=a2["top_class_share"],
            top_class_share_rule_v1=a1["top_class_share"],
            mean_p_home_given_live_rule_v2=round(float(q2.mean()), 4) if live.any() else None,
            mean_p_home_given_live_rule_v1=round(float(q1.mean()), 4) if live.any() else None,
            share_live_top_h_rule_v2=round(float(np.mean(q2 >= 0.5)), 4) if live.any() else None,
            share_live_top_h_rule_v1=round(float(np.mean(q1 >= 0.5)), 4) if live.any() else None,
            agree_top_team_v1_v2=round(float(np.mean((q1 >= 0.5) == (q2 >= 0.5))), 4)
            if live.any() else None,
            flat_share=a2["flat_share"], **cover)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Team models: probs/<model>.npy (default rule_v2)")
    ap.add_argument("target", choices=("match", "bench"))
    ap.add_argument("--models", nargs="+", default=[MODEL_NAME], choices=sorted(MODELS),
                    help="team models to write (each needs its params file)")
    ap.add_argument("--root", nargs="*", help="explicit bench roots")
    ap.add_argument("--game", nargs="*", help="bench game ids under artifacts/bench/")
    ap.add_argument("--level", nargs="*", help="bench levels: roots metrica_g{1,2,3}_<level>")
    ap.add_argument("--truth", help="bench truth.csv (default: found from the root name)")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--version", help="pipeline version from make (accepted; the files are "
                                      "named by --models)")
    args = ap.parse_args(argv)
    missing = [m for m in args.models if not os.path.isfile(params_path(m))]
    if missing:
        ap.error("no params file for " + ", ".join(missing))
    if args.target == "match":
        if args.root or args.game or args.level or args.truth or args.no_write:
            ap.error("the match takes no options other than --models")
        return run_match(tuple(args.models))
    if args.root:
        roots = args.root
    elif args.level:
        roots = [os.path.join(RS.ARTIFACTS, "bench", "metrica_g{}_{}".format(k, lv))
                 for lv in args.level for k in (1, 2, 3)]
    else:
        roots = RS.bench_roots(args.game)
    require_bench_roots(roots)
    return run_bench(roots, args.truth, write=not args.no_write, models=tuple(args.models))


if __name__ == "__main__":
    sys.exit(main())
