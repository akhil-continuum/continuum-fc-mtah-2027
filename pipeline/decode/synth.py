"""Stand-in probabilities made by corrupting bench truth (bench only; refuses the match).

Used to build and stress the decoders before the rule scorer exists, and kept afterwards as a
decoder-only test bed. Never a source of dials for the real pipeline: once ``probs/rule_v1.npy``
exists the sweeps are rerun on it.

Each profile corrupts the truth in the ways a frame scorer fails, all seeded from the game id:

- boundary jitter: every true label change moved by up to ``jitter`` frames either way;
- weak evidence: per-run confidence of the true class drawn from ``conf``; possessions shorter
  than 2 s get ``short_conf`` (short possessions are the hard ones);
- late stoppages: the first U(0, ``late_dead``) frames of each stoppage keep the previous label;
- blips: wrong-team runs of ``blip_len`` frames at ``blip_rate`` per live minute;
- false stoppages: dead-looking runs of ``fdead_len`` frames at ``fdead_rate`` per live minute;
- gaps: flat live evidence (0.45, 0.45, 0.10) runs of ``gap_len`` at ``gap_rate`` per minute;
- frame noise: uniform jitter of +-``frame_noise`` on every probability, renormalised.

Restart hints: a restarts-like table (restart_frame, start_frame, p_home_restart) at true
restarts, present with ``hint_cover`` probability, naming the right team with ``hint_acc``.
"""
import os
import zlib

import numpy as np
import pandas as pd

H, A, D, X = 0, 1, 2, 3
FPS = 25

PROFILES = {
    "noiseA": dict(jitter=5, conf=(0.55, 0.85), short_conf=(0.45, 0.65), late_dead=40,
                   blip_rate=4.0, blip_len=(3, 12), fdead_rate=0.3, fdead_len=(20, 60),
                   gap_rate=1.5, gap_len=(25, 150), frame_noise=0.10,
                   hint_cover=0.7, hint_acc=0.85, hint_p=0.8),
    "noiseB": dict(jitter=10, conf=(0.45, 0.75), short_conf=(0.38, 0.55), late_dead=75,
                   blip_rate=8.0, blip_len=(3, 20), fdead_rate=0.8, fdead_len=(25, 75),
                   gap_rate=3.0, gap_len=(25, 250), frame_noise=0.15,
                   hint_cover=0.5, hint_acc=0.75, hint_p=0.7),
}


def _seed(game_id, profile):
    return zlib.crc32("{}|{}".format(game_id, profile).encode()) & 0x7FFFFFFF


def _runs(codes):
    br = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    bounds = np.concatenate(([0], br, [len(codes)]))
    return [(int(bounds[i]), int(bounds[i + 1]), int(codes[bounds[i]]))
            for i in range(len(bounds) - 1)]


def corrupt(truth, game_id, profile="noiseA"):
    """(probs float32 (n, 3), restarts-like DataFrame). truth: code array H0 A1 D2 X3."""
    prm = PROFILES[profile]
    rng = np.random.default_rng(_seed(game_id, profile))
    truth = np.asarray(truth, dtype=np.int8)
    n = len(truth)

    # boundary jitter
    lab = truth.copy()
    runs = _runs(truth)
    for i in range(1, len(runs)):
        a0, a1, ca = runs[i - 1]
        b0, b1, cb = runs[i]
        if X in (ca, cb):
            continue
        sh = int(rng.integers(-prm["jitter"], prm["jitter"] + 1))
        new = int(np.clip(b0 + sh, a0 + 1, b1 - 1)) if b1 - a0 > 2 else b0
        if new < b0:
            lab[new:b0] = cb
        elif new > b0:
            lab[b0:new] = ca

    p = np.full((n, 3), 1.0 / 3.0)
    runs = _runs(lab)
    prev = -1
    for a, b, c in runs:
        if c == X:
            prev = -1
            continue
        short = c in (H, A) and b - a < 2 * FPS
        lo, hi = prm["short_conf"] if short else prm["conf"]
        conf = rng.uniform(lo, hi)
        p[a:b] = (1.0 - conf) / 2.0
        p[a:b, c] = conf
        if c == D and prev in (H, A):          # late stoppage: previous label persists
            late = min(b - a - 1, int(rng.integers(0, prm["late_dead"] + 1)))
            p[a:a + late] = (1.0 - conf) / 2.0
            p[a:a + late, prev] = conf
        prev = c

    live = np.flatnonzero(np.isin(lab, (H, A)))
    live_min = len(live) / FPS / 60.0

    def inject(rate, length, fill):
        for _ in range(rng.poisson(rate * live_min)):
            f = int(live[rng.integers(len(live))])
            L = int(rng.integers(length[0], length[1] + 1))
            seg = slice(f, min(n, f + L))
            fill(seg, lab[f])

    def blip(seg, c):
        p[seg] = 0.15
        p[seg, 1 - c] = 0.7

    def fdead(seg, c):
        p[seg] = 0.2
        p[seg, D] = 0.6

    def gap(seg, c):
        p[seg] = (0.45, 0.45, 0.10)

    inject(prm["blip_rate"], prm["blip_len"], blip)
    inject(prm["fdead_rate"], prm["fdead_len"], fdead)
    inject(prm["gap_rate"], prm["gap_len"], gap)

    p = p + rng.uniform(-prm["frame_noise"], prm["frame_noise"], size=p.shape)
    p = np.clip(p, 0.01, None)
    p /= p.sum(axis=1, keepdims=True)
    p[truth == X] = 1.0 / 3.0

    # restart hints at true restarts (end of a D run followed by a team)
    rows = []
    for a, b, c in _runs(truth):
        if c != D or b >= n or truth[b] not in (H, A):
            continue
        if rng.random() > prm["hint_cover"]:
            continue
        right = rng.random() < prm["hint_acc"]
        home = (truth[b] == H) == right
        rows.append({"start_frame": -1, "restart_frame": int(b),
                     "p_home_restart": prm["hint_p"] if home else 1.0 - prm["hint_p"]})
    hints = pd.DataFrame(rows, columns=["start_frame", "restart_frame", "p_home_restart"])
    return p.astype(np.float32), hints


def for_root(root, truth_codes, profile):
    if os.path.normpath(root).endswith(os.path.join("artifacts", "match")):
        raise SystemExit("synthetic probabilities are bench-only")
    return corrupt(truth_codes, os.path.basename(os.path.normpath(root)), profile)
