"""Evidence-dependent switch cost read from the team model's own output (T-42, v12g).

Rulebook Part 2 table, H <-> A row: a switch is cheaper where the evidence for the new team is
strong. v3a (T-11) drove this from contact events and gained nothing, because the team model
already counted them. Here the evidence is the per-player trees' confidence, and the cost only
moves where the trees are in use (the busy gate G4 of rule_p1g). Elsewhere the decoder is v11g.

Per direction X in {H, A}, the cost of a switch into X with first frame t:

    c_X(t) = c_lo + (c_hi - c_lo) * G(t) * (1 - k_X(t))
    q_H    = P(H) / (P(H) + P(A)),  q_A = 1 - q_H          (the version's probabilities)
    m_X(t) = max of q_X(u) over |u - t| <= W                 (centred window, W frames)
    k_X(t) = clip((m_X(t) - theta) / (1 - theta), 0, 1)      (theta >= 1: k = 0)
    G(t)   = 1 where the version's team probability differs from its reference model's
             (rule_p1g vs rule_w2: the trees' gate weight is above zero), else 0

Monotone: more confidence for X nearby never makes a switch into X dearer, and the cost stays in
[c_lo, c_hi]. Centred windows only. The restart law is untouched: this only fills the H <-> A
arcs of pipeline.decode.viterbi.
"""
import numpy as np

W_DEFAULT = 12          # frames either side (0.5 s at 25 fps); declared, not swept
GATE_EPS = 1e-6         # team-probability difference that counts as "trees in use"


def team_q(probs):
    """q_H = P(H) / (P(H) + P(A)) per frame (probs (n, 3), already cleaned)."""
    p = np.asarray(probs, dtype=np.float64)
    return p[:, 0] / np.maximum(p[:, 0] + p[:, 1], 1e-12)


def centred_max(x, w):
    """max of x over [t - w, t + w] (truncated at the edges)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0 or w <= 0:
        return x.copy()
    pad = np.concatenate([np.full(w, -np.inf), x, np.full(w, -np.inf)])
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * w + 1)
    return win.max(axis=1)[:n]


def gate_from_reference(probs, ref_probs, eps=GATE_EPS):
    """bool (n,): frames where the team probability differs from the reference model's."""
    return np.abs(team_q(probs) - team_q(ref_probs)) > eps


def confidence(probs, theta, w=W_DEFAULT):
    """(n, 2) k_H, k_A in [0, 1]."""
    q = team_q(probs)
    m = np.stack([centred_max(q, w), centred_max(1.0 - q, w)], axis=1)
    if theta >= 1.0:
        return np.zeros_like(m)
    return np.clip((m - theta) / (1.0 - theta), 0.0, 1.0)


def switch_cost(probs, gate, c_lo, c_hi, theta, w=W_DEFAULT):
    """(n, 2) cost of a switch into H (column 0) and into A (column 1)."""
    k = confidence(probs, theta, w)
    g = np.asarray(gate, dtype=np.float64)[:, None]
    return float(c_lo) + (float(c_hi) - float(c_lo)) * g * (1.0 - k)
