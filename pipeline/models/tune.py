"""Choose the rule scorer's weights on bench roots (never the match, never the public 5 minutes).

    uv run python -m pipeline.models.tune --root DIR [DIR ...] [--check DIR ...]
        [--tau-ctrl 4 8 16 30] [--l2 0.01]

For each tuning root the terms are computed with the shapes in rule_params.SHAPE. The weights
are found by bounded maximum likelihood (L-BFGS-B), separately for the dead part (binary D vs
live on graded frames) and the team part (binary H vs A on truly live frames), with:
- the sign of each weight fixed by its rule's direction (rule_params.SIGNS);
- FROZEN weights held at their default;
- a small L2 pull towards the physically sensible defaults (``--l2`` per frame-averaged loss);
- every 5th frame used (frames are strongly autocorrelated; this changes nothing but speed).
Groups are whole games: the leave-one-game-out table fits on all roots but one game and scores
the held-out game, for every game. Roots whose names share ``g<k>`` belong to the same game.

Prints the chosen weights (paste into rule_params.DEAD_W / TEAM_W / DEAD_BIAS) and per-root
log-loss with defaults, with the pooled fit, and held out. Bench only: refuses artifacts/match.
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pipeline.models import rule_params as RP
from pipeline.io.roots import require_bench_root
from pipeline.models import rule_scorer as RS

STEP = 5
EPS = 1e-9


def _game(root):
    m = re.search(r"g(\d+)", os.path.basename(os.path.normpath(root)))
    return m.group(1) if m else os.path.basename(os.path.normpath(root))


def load(root, shape=None):
    require_bench_root(root)
    data = RS.read_root(root)
    dead, team, _info = RS.compute_terms(data, shape=shape)
    labels = pd.read_csv(RS.find_truth(root))["label"].to_numpy().astype(str)
    sel = np.zeros(len(labels), dtype=bool)
    sel[::STEP] = True
    g = sel & np.isin(labels, ["H", "A", "D"])
    D = np.stack([dead[k] for k in RP.DEAD_DEFAULTS], 1)[g]
    Tm = np.stack([team[k] for k in RP.TEAM_DEFAULTS], 1)[g]
    lab = labels[g]
    return {"root": root, "game": _game(root), "D": D, "T": Tm, "y_dead": (lab == "D"),
            "live": lab != "D", "y_home": lab == "H"}


def _fit(X, y, names, defaults, signs, frozen, l2, bias=None, offset=0.0):
    """Bounded logistic fit; returns (weights dict, bias or None)."""
    has_bias = bias is not None
    w0 = np.array([defaults[k] for k in names] + ([bias] if has_bias else []), dtype=float)
    bounds = []
    for k in names:
        if k in frozen:
            bounds.append((defaults[k], defaults[k]))
        elif signs.get(k, 0) > 0:
            bounds.append((0.0, 12.0))
        elif signs.get(k, 0) < 0:
            bounds.append((-12.0, 0.0))
        else:
            bounds.append((-12.0, 12.0))
    if has_bias:
        bounds.append((-6.0, 6.0))
    Xb = np.hstack([X, np.ones((len(X), 1))]) if has_bias else X
    yf = y.astype(float)

    def f(w):
        z = Xb @ w + offset
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        ll = -np.mean(yf * np.log(p + EPS) + (1 - yf) * np.log(1 - p + EPS))
        grad = Xb.T @ (p - yf) / len(yf)
        pen = l2 * np.sum((w - w0) ** 2)
        return ll + pen, grad + 2 * l2 * (w - w0)

    res = minimize(f, w0, jac=True, method="L-BFGS-B", bounds=bounds)
    w = res.x
    out = {k: round(float(v), 3) for k, v in zip(names, w[:len(names)])}
    return out, (round(float(w[-1]), 3) if has_bias else None)


def fit(sets, l2, free_bias=False):
    D = np.vstack([s["D"] for s in sets])
    yd = np.concatenate([s["y_dead"] for s in sets])
    T = np.vstack([s["T"][s["live"]] for s in sets])
    yh = np.concatenate([s["y_home"][s["live"]] for s in sets])
    if free_bias:
        dw, bias = _fit(D, yd, list(RP.DEAD_DEFAULTS), RP.DEAD_DEFAULTS, RP.SIGNS, RP.FROZEN,
                        l2, bias=RP.DEAD_BIAS_DEFAULT)
    else:   # flat prior kept: a frame with no evidence stays (1/3, 1/3, 1/3)
        Db = D.copy()
        dw, _ = _fit(Db, yd, list(RP.DEAD_DEFAULTS), RP.DEAD_DEFAULTS, RP.SIGNS, RP.FROZEN, l2,
                     bias=None, offset=RP.DEAD_BIAS_DEFAULT)
        bias = RP.DEAD_BIAS_DEFAULT
    tw, _ = _fit(T, yh, list(RP.TEAM_DEFAULTS), RP.TEAM_DEFAULTS, RP.SIGNS, RP.FROZEN, l2)
    return dw, tw, bias


def losses(s, dw, tw, bias):
    zd = bias + s["D"] @ np.array([dw[k] for k in RP.DEAD_DEFAULTS])
    pd_ = np.clip(RS.sigmoid(zd), 1e-6, 1 - 1e-6)
    zt = s["T"] @ np.array([tw[k] for k in RP.TEAM_DEFAULTS])
    ph = np.clip(RS.sigmoid(zt), 1e-6, 1 - 1e-6)
    y = s["y_dead"]
    ll_d = -np.mean(np.where(y, np.log(pd_), np.log(1 - pd_)))
    lv = s["live"]
    h = s["y_home"][lv]
    ll_t = -np.mean(np.where(h, np.log(ph[lv]), np.log(1 - ph[lv])))
    p3 = np.where(y, pd_, (1 - pd_) * np.where(s["y_home"], ph, 1 - ph))
    return {"ll3": round(float(-np.mean(np.log(np.clip(p3, 1e-6, 1)))), 4),
            "ll_dead": round(float(ll_d), 4), "ll_team": round(float(ll_t), 4),
            "f1_dead": round(float(RS_f1(pd_ > 0.5, y)), 4),
            "team_acc": round(float(np.mean((ph[lv] > 0.5) == h)), 4)}


def RS_f1(pred, true):
    tp = np.sum(pred & true)
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + np.sum(pred & ~true) + np.sum(~pred & true))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", nargs="+", required=True, help="tuning roots (bench)")
    ap.add_argument("--check", nargs="*", default=[], help="extra roots scored, not fitted")
    ap.add_argument("--tau-ctrl", nargs="*", type=float, default=None)
    ap.add_argument("--l2", type=float, default=0.002)
    ap.add_argument("--free-bias", action="store_true",
                    help="also fit DEAD_BIAS (default: held at log(1/2), the flat prior)")
    args = ap.parse_args(argv)
    taus = args.tau_ctrl or [RP.SHAPE["tau_ctrl_s"]]
    best = None
    for tau in taus:
        shape = {"tau_ctrl_s": tau}
        sets = [load(r, shape) for r in args.root]
        games = sorted({s["game"] for s in sets})
        dw, tw, bias = fit(sets, args.l2, args.free_bias)
        # leave one game out
        held = {}
        for g in games:
            tr = [s for s in sets if s["game"] != g]
            te = [s for s in sets if s["game"] == g]
            if not tr:
                continue
            d2, t2, b2 = fit(tr, args.l2, args.free_bias)
            for s in te:
                held[os.path.basename(os.path.normpath(s["root"]))] = losses(s, d2, t2, b2)
        mean_held = np.mean([v["ll3"] for v in held.values()]) if held else np.nan
        print("tau_ctrl_s={} held-out mean ll3={:.4f}".format(tau, mean_held))
        if best is None or mean_held < best[0]:
            best = (mean_held, tau, dw, tw, bias, sets, held)
    _, tau, dw, tw, bias, sets, held = best
    print("\nchosen tau_ctrl_s =", tau)
    print("DEAD_BIAS =", bias)
    print("DEAD_W =", dw)
    print("TEAM_W =", tw)
    dflt = (dict(RP.DEAD_DEFAULTS), dict(RP.TEAM_DEFAULTS), RP.DEAD_BIAS_DEFAULT)
    print("\nroot | defaults | pooled fit | held out (leave one game out)")
    for s in sets:
        name = os.path.basename(os.path.normpath(s["root"]))
        print(name, losses(s, *dflt), losses(s, dw, tw, bias), held.get(name))
    for r in args.check:
        s = load(r, {"tau_ctrl_s": tau})
        print("check", os.path.basename(os.path.normpath(r)), losses(s, *dflt),
              losses(s, dw, tw, bias))
    return 0


if __name__ == "__main__":
    sys.exit(main())
