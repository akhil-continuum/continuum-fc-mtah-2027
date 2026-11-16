"""Per-player team model (T-32 step 1): gradient-boosted trees on the K nearest players of each
team to the ball, sampled at offsets either side of the frame, with rule_w2's output as an input.

Design (``views``). For a frame t the model sees, at each offset o in OFFSETS (frames; centred,
past and future), the snapshot columns of features/players_near_ball.py at t + o (NaN outside the
half): for the "own" team and the "other" team, their K_USE nearest players (distance, position
and velocity in the team's own attack frame, speed towards the ball), and the ball (seen, speed,
position and velocity in the own team's attack frame). Plus rule_w2's logit at a few offsets, the
touch-count and loose-ball differences, and symmetric context (duel, crowding, zone, box crowd,
ball-unseen share). The home view puts home in "own"; the away view puts away there, negates the
antisymmetric inputs and swaps the team blocks. Swapping the teams and turning the pitch turns one
view into the other exactly, so the model has no built-in lean to either team.

Learner: sklearn HistGradientBoostingClassifier (pinned in uv.lock), one thread, fixed seed, no
early stopping, trained on both views of every row (team-flip augmentation; label = own team has
the ball). Prediction, symmetrised: q_home = (f(home view) + 1 - f(away view)) / 2.

The fitted trees are exported to tracked JSON (``export``) and evaluated here in numpy
(``forest_raw``), so the stage never fits anything and needs no sklearn at run time; a test checks
the export against sklearn's own predictions.

Gate (optional, params["gate"]): a "busy" rule on stage inputs. Where the gate is on (smoothed over
+-GATE_SMOOTH_S), q = the trees' answer; elsewhere rule_w2's. P(D) is always rule_w2's.
"""
import numpy as np

OFFSETS = (-50, -25, -12, -5, 0, 5, 12, 25, 50)
K_USE = 3
QUANTS = ("d", "dx", "dy", "vto", "vatt")
QUANTS2 = QUANTS + ("brk", "trn")          # variant S1g2 (brake / turn, players_near_ball)
BALL_SYM = ("pnb_ball_ok", "pnb_ball_seen", "pnb_ball_speed")
BALL_ANTI = ("pnb_ball_xh", "pnb_ball_yh", "pnb_ball_vxh", "pnb_ball_vyh")
Z_OFFSETS = (-25, -12, 0, 12, 25)
ANTI = ("pc_touch_diff_b1", "pc_touch_diff_a1", "pc_touch_diff_b2", "pc_touch_diff_a2",
        "pc_touch_diff_b4", "pc_touch_diff_a4", "lb_pfirst_diff_c05", "lb2_pfirst_diff_c05",
        "pc_duel_lead_diff_c05", "pc_near5_diff_c1")
SYM = ("pc_duel", "pc_crowd5", "pc_box_crowd", "pc_zone", "pc_touch_n_c1", "ctx_unseen")
GATE_SMOOTH_S = 0.5
CHUNK = 20000


def feature_names(quants=QUANTS):
    names = []
    for o in OFFSETS:
        for side in ("own", "oth"):
            for k in range(1, K_USE + 1):
                names += ["{}{}_{}@{}".format(side, k, q, o) for q in quants]
        names += ["{}@{}".format(c, o) for c in BALL_SYM + BALL_ANTI]
    names += ["z_w2@{}".format(o) for o in Z_OFFSETS]
    names += list(ANTI) + list(SYM)
    return names


def needed_columns(quants=QUANTS):
    cols = ["pnb_{}{}_{}".format(t, k, q) for t in ("h", "a") for k in range(1, K_USE + 1)
            for q in quants]
    return cols + list(BALL_SYM) + list(BALL_ANTI) + list(ANTI) + \
        [c for c in SYM if c != "ctx_unseen"]


def arrays(features, z, ctx_unseen, quants=QUANTS):
    """Float64 arrays the design reads: feature columns, rule_w2 logit 'z', 'ctx_unseen'."""
    out = {c: features[c].to_numpy(np.float64) for c in needed_columns(quants)}
    out["z"] = np.asarray(z, dtype=np.float64)
    out["ctx_unseen"] = np.asarray(ctx_unseen, dtype=np.float64)
    return out


def half_ids(halves, n):
    half = np.zeros(n, dtype=np.int32)
    for h, (a, b) in enumerate(halves, 1):
        half[max(0, int(a)):min(n, int(b))] = h
    return half


def _shift(arr, idx, o, half):
    j = idx + o
    ok = (j >= 0) & (j < len(arr))
    jc = np.clip(j, 0, len(arr) - 1)
    ok &= half[jc] == half[idx]
    return np.where(ok, arr[jc], np.nan)


def views(ar, half, idx, quants=QUANTS):
    """(home view, away view) float64 (len(idx), n_features) at frames idx."""
    m = len(idx)
    nf = len(feature_names(quants))
    Xh = np.empty((m, nf))
    Xa = np.empty((m, nf))
    c = 0
    for o in OFFSETS:
        for own, oth, X in (("h", "a", Xh), ("a", "h", Xa)):
            cc = c
            for team in (own, oth):
                for k in range(1, K_USE + 1):
                    for q in quants:
                        X[:, cc] = _shift(ar["pnb_{}{}_{}".format(team, k, q)], idx, o, half)
                        cc += 1
        c += 2 * K_USE * len(quants)
        for col in BALL_SYM:
            v = _shift(ar[col], idx, o, half)
            Xh[:, c], Xa[:, c] = v, v
            c += 1
        for col in BALL_ANTI:
            v = _shift(ar[col], idx, o, half)
            Xh[:, c], Xa[:, c] = v, -v
            c += 1
    for o in Z_OFFSETS:
        v = _shift(ar["z"], idx, o, half)
        Xh[:, c], Xa[:, c] = v, -v
        c += 1
    for col in ANTI:
        v = ar[col][idx]
        Xh[:, c], Xa[:, c] = v, -v
        c += 1
    for col in SYM:
        v = ar[col][idx]
        Xh[:, c], Xa[:, c] = v, v
        c += 1
    assert c == nf
    return Xh, Xa


# --- trees: export and numpy inference -----------------------------------------------------------

def export(model):
    """A fitted HistGradientBoostingClassifier (binary) as plain lists (tracked JSON)."""
    trees = []
    for it in model._predictors:
        nodes = it[0].nodes
        if np.any(nodes["is_categorical"]):
            raise ValueError("categorical splits are not supported")
        trees.append({"f": nodes["feature_idx"].astype(int).tolist(),
                      "t": [float(v) for v in nodes["num_threshold"]],
                      "m": nodes["missing_go_to_left"].astype(int).tolist(),
                      "l": nodes["left"].astype(int).tolist(),
                      "r": nodes["right"].astype(int).tolist(),
                      "leaf": nodes["is_leaf"].astype(int).tolist(),
                      "v": [float(v) for v in nodes["value"]]})
    return {"baseline": float(np.ravel(model._baseline_prediction)[0]), "trees": trees}


def _tree_values(tree, X):
    f = np.asarray(tree["f"], dtype=np.int64)
    t = np.asarray(tree["t"], dtype=np.float64)
    m = np.asarray(tree["m"], dtype=bool)
    left = np.asarray(tree["l"], dtype=np.int64)
    right = np.asarray(tree["r"], dtype=np.int64)
    leaf = np.asarray(tree["leaf"], dtype=bool)
    val = np.asarray(tree["v"], dtype=np.float64)
    node = np.zeros(len(X), dtype=np.int64)
    active = ~leaf[node]
    rows = np.arange(len(X))
    while active.any():
        r = rows[active]
        nd = node[r]
        x = X[r, f[nd]]
        go_left = np.where(np.isnan(x), m[nd], x <= t[nd])
        node[r] = np.where(go_left, left[nd], right[nd])
        active[r] = ~leaf[node[r]]
    return val[node]


def forest_raw(forest, X):
    raw = np.full(len(X), float(forest["baseline"]))
    for tree in forest["trees"]:
        raw += _tree_values(tree, X)
    return raw


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def q_home(forest, ar, half, idx=None, cols=None, quants=QUANTS):
    """Symmetrised P(home | live) at frames idx (default: every frame), in chunks. ``cols``
    (bench ablations only): indices of feature_names() the forest was fitted on."""
    n = len(half)
    idx = np.arange(n) if idx is None else np.asarray(idx)
    out = np.empty(len(idx))
    for a in range(0, len(idx), CHUNK):
        sl = idx[a:a + CHUNK]
        Xh, Xa = views(ar, half, sl, quants)
        if cols is not None:
            Xh, Xa = Xh[:, cols], Xa[:, cols]
        out[a:a + CHUNK] = 0.5 * (sigmoid(forest_raw(forest, Xh)) + 1.0 -
                                  sigmoid(forest_raw(forest, Xa)))
    return out


# --- gate ----------------------------------------------------------------------------------------

def centred_any(flag, half_w):
    """1 where flag is set anywhere within +-half_w frames."""
    c = np.concatenate([[0], np.cumsum(flag.astype(np.int64))])
    i = np.arange(len(flag))
    lo, hi = np.maximum(i - half_w, 0), np.minimum(i + half_w + 1, len(flag))
    return (c[hi] - c[lo]) > 0


def centred_mean(a, half_w):
    c = np.concatenate([[0.0], np.cumsum(a.astype(np.float64))])
    i = np.arange(len(a))
    lo, hi = np.maximum(i - half_w, 0), np.minimum(i + half_w + 1, len(a))
    return (c[hi] - c[lo]) / (hi - lo)


def gate_flags(features, events, q_w2, fps):
    """Candidate busy rules (each a boolean per frame), from stage inputs only."""
    n = len(q_w2)
    h1, h05 = int(round(fps)), int(round(0.5 * fps))
    d = {t: np.stack([features["pnb_{}{}_d".format(t, k)].to_numpy(float) for k in range(1, 5)], 1)
         for t in ("h", "a")}
    with np.errstate(invalid="ignore"):
        crowd = (np.sum(d["h"] < 5.0, 1) >= 2) & (np.sum(d["a"] < 5.0, 1) >= 2)
    both = np.zeros(n, dtype=bool)
    if events is not None and len(events):
        tch = events[events["kind"] == "touch"]
        per = []
        for t in (0, 1):
            mark = np.zeros(n, dtype=bool)
            f = tch.loc[tch["team"] == t, "frame"].to_numpy().astype(np.int64)
            mark[f[(f >= 0) & (f < n)]] = True
            per.append(centred_any(mark, h1))
        both = per[0] & per[1]
    unsure = centred_any(np.abs(q_w2 - 0.5) < 0.2, h05)
    g = {"G1": centred_any(crowd, h1), "G2": both, "G3": unsure}
    g["G4"] = g["G1"] | g["G2"] | g["G3"]
    return g


def gate_weight(flag, fps):
    """Gate in [0, 1]: the busy flag smoothed over +-GATE_SMOOTH_S."""
    return centred_mean(flag.astype(np.float64), int(round(GATE_SMOOTH_S * fps)))
