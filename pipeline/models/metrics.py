"""Per-frame quality of a probs array against bench truth (bench only; never the match).

Graded frames are those whose truth label is H, A or D. Reported:
- alive/dead: F1 of the D class and of the live class at P(D) > 0.5, and their mean;
- team accuracy on truly live frames (argmax of P(H), P(A));
- log-loss: three-class on graded frames, binary dead on graded frames, binary team on live frames;
- reliability tables of 10 bins for P(D) (graded frames) and P(H | live) (live frames).
"""
import numpy as np

EPS = 1e-6
BINS = 10


def _f1(pred, true):
    tp = float(np.sum(pred & true))
    fp = float(np.sum(pred & ~true))
    fn = float(np.sum(~pred & true))
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def reliability(p, y, bins=BINS):
    """List of (bin low, bin high, n, mean predicted, observed rate) over equal-width bins."""
    edges = np.linspace(0, 1, bins + 1)
    k = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        s = k == b
        rows.append((round(edges[b], 2), round(edges[b + 1], 2), int(s.sum()),
                     round(float(p[s].mean()), 4) if s.any() else None,
                     round(float(y[s].mean()), 4) if s.any() else None))
    return rows


def ece(rows):
    n = sum(r[2] for r in rows)
    return float(sum(r[2] * abs(r[3] - r[4]) for r in rows if r[2]) / max(n, 1))


def evaluate(probs, labels):
    """probs (n, 3) = P(H), P(A), P(D); labels array of 'H', 'A', 'D', 'X'."""
    labels = np.asarray(labels).astype(str)
    g = np.isin(labels, ["H", "A", "D"])
    P = np.clip(probs[g].astype(float), EPS, 1.0)
    lab = labels[g]
    idx = np.select([lab == "H", lab == "A"], [0, 1], 2)
    ll3 = float(-np.mean(np.log(P[np.arange(len(lab)), idx])))
    dead_true = lab == "D"
    pd_ = np.clip(P[:, 2], EPS, 1 - EPS)
    dead_pred = pd_ > 0.5
    ll_dead = float(-np.mean(np.where(dead_true, np.log(pd_), np.log(1 - pd_))))
    live = ~dead_true
    ph = np.clip(P[live, 0] / np.maximum(P[live, 0] + P[live, 1], EPS), EPS, 1 - EPS)
    home = lab[live] == "H"
    ll_team = float(-np.mean(np.where(home, np.log(ph), np.log(1 - ph))))
    team_acc = float(np.mean((ph > 0.5) == home))
    rel_d = reliability(pd_, dead_true.astype(float))
    rel_h = reliability(ph, home.astype(float))
    argmax = np.argmax(P, axis=1)
    return {
        "graded_frames": int(g.sum()),
        "f1_dead": round(_f1(dead_pred, dead_true), 4),
        "f1_live": round(_f1(~dead_pred, ~dead_true), 4),
        "f1_alive_dead_mean": round((_f1(dead_pred, dead_true) + _f1(~dead_pred, ~dead_true)) / 2,
                                    4),
        "team_acc_live": round(team_acc, 4),
        "team_acc_live_confident_share": round(float(np.mean(np.abs(ph - 0.5) > 0.2)), 4),
        "frame_acc_3class": round(float(np.mean(argmax == idx)), 4),
        "logloss_3class": round(ll3, 4),
        "logloss_dead": round(ll_dead, 4),
        "logloss_team_live": round(ll_team, 4),
        "ece_dead": round(ece(rel_d), 4),
        "ece_team": round(ece(rel_h), 4),
        "reliability_dead": rel_d,
        "reliability_team": rel_h,
    }
