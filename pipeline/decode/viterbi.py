"""Four-state Viterbi with the restart law as structure and exact minimum durations (v2).

Hidden states (rulebook Part 2, R0, S8):

    0 H     home possession
    1 A     away possession
    2 DH    dead ball, home restarts      written as D
    3 DA    dead ball, away restarts      written as D

Allowed transitions, and nothing else (the restart law is the absence of an arc, not a penalty):

    H <-> A          c_switch[t]
    H, A -> DH, DA   c_stop[t]
    DH -> H          c_restart[t]
    DA -> A          c_restart[t]

Every transition cost is a per-frame array indexed by the first frame of the new state, so
event-dependent costs (v3 onwards) plug in without touching the recursion; the constant dials of
v2 are just constant arrays.

Minimum durations: a run of state s must last at least k[s] frames (S9, S10). Implemented exactly
as a semi-Markov recursion instead of chained sub-states:

    E[t][s] = min_r M[t-1][r] + trans(r -> s, t)                 enter s at t
    M[t][s] = min(M[t-1][s] + e[t][s],                           stay (run already >= k[s])
                  E[t-k+1][s] + sum(e[t-k+1 .. t][s]))           run entered k[s]-1 frames ago

M is the best cost of being in s at t with the current run long enough to leave. The first run of
a segment and the last one may be shorter (the segment edges are not real boundaries). Ties keep
the current state. Pure Python on lists of floats: about one second per 70,000-frame half.
"""
import numpy as np

H, A, DH, DA = 0, 1, 2, 3
N_STATES = 4
STATE_NAMES = ("H", "A", "DH", "DA")
OUTPUT_CODE = np.array([0, 1, 2, 2], dtype=np.int8)     # H, A, D, D in possession_metrics codes

# (predecessor, kind) for every state; kind 0 switch into H, 3 switch into A, 1 stop, 2 restart.
PREDECESSORS = {
    H: ((A, 0), (DH, 2)),
    A: ((H, 3), (DA, 2)),
    DH: ((H, 1), (A, 1)),
    DA: ((H, 1), (A, 1)),
}
ALLOWED = {(r, s) for s, preds in PREDECESSORS.items() for r, _ in preds}
INF = float("inf")


def decode_segment(emis, c_switch, c_stop, c_restart, k):
    """Best hidden path for one segment.

    emis: (T, 4) float costs (-log evidence; large finite values for masked states).
    c_switch, c_stop, c_restart: (T,) per-frame transition costs (entry frame of the new state);
    c_switch may also be (T, 2): the cost of a switch into H (column 0) and into A (column 1).
    k: four minimum run lengths (>= 1).
    Returns (path int8 (T,), total cost).
    """
    emis = np.asarray(emis, dtype=np.float64)
    T = emis.shape[0]
    if T == 0:
        return np.zeros(0, dtype=np.int8), 0.0
    k = [max(1, int(v)) for v in k]
    cs = np.asarray(c_switch, dtype=np.float64)
    if cs.ndim == 2:                      # (T, 2): column 0 into H, column 1 into A (v3)
        sw_h, sw_a = cs[:, 0], cs[:, 1]
    else:
        sw_h = sw_a = cs
    trans = [np.broadcast_to(np.asarray(c, dtype=np.float64), (T,)).tolist()
             for c in (sw_h, c_stop, c_restart, sw_a)]
    e = emis.tolist()
    csum = np.vstack([np.zeros((1, N_STATES)), np.cumsum(emis, axis=0)]).tolist()  # csum[t+1]
    preds = [PREDECESSORS[s] for s in range(N_STATES)]

    M = [[INF] * N_STATES for _ in range(T)]
    E = [[INF] * N_STATES for _ in range(T)]
    Er = [[-1] * N_STATES for _ in range(T)]
    bp = [[-1] * N_STATES for _ in range(T)]      # -1 stay, else predecessor at t - k + 1 - 1
    M[0] = list(e[0])
    for t in range(1, T):
        Mp = M[t - 1]
        Et, Ert = E[t], Er[t]
        for s in range(N_STATES):
            best, arg = INF, -1
            for r, kind in preds[s]:
                v = Mp[r] + trans[kind][t]
                if v < best:
                    best, arg = v, r
            Et[s], Ert[s] = best, arg
        Mt, bpt, et = M[t], bp[t], e[t]
        ct1 = csum[t + 1]
        for s in range(N_STATES):
            stay = Mp[s] + et[s]
            tau = t - k[s] + 1
            if tau >= 1:
                enter = E[tau][s] + (ct1[s] - csum[tau][s])
                if enter < stay:
                    Mt[s], bpt[s] = enter, Er[tau][s]
                    continue
            Mt[s] = stay

    # end: a mature state, or a last run shorter than its minimum
    best, end_s, end_tau = INF, -1, -1
    for s in range(N_STATES):
        if M[T - 1][s] < best:
            best, end_s, end_tau = M[T - 1][s], s, -1
    cT = csum[T]
    for s in range(N_STATES):
        for tau in range(max(1, T - k[s] + 1), T):
            v = E[tau][s] + (cT[s] - csum[tau][s])
            if v < best:
                best, end_s, end_tau = v, s, tau

    path = np.empty(T, dtype=np.int8)
    t, s = T - 1, end_s
    if end_tau >= 0:
        path[end_tau:T] = s
        t, s = end_tau - 1, Er[end_tau][s]
    while t >= 0:
        r = bp[t][s]
        if r < 0 or t == 0:
            path[t] = s
            t -= 1
            continue
        tau = t - k[s] + 1
        path[tau:t + 1] = s
        t, s = tau - 1, r
    return path, float(best)


def path_violations(path):
    """Number of frame-to-frame changes not in the transition table (must be 0)."""
    p = np.asarray(path)
    if len(p) < 2:
        return 0
    a, b = p[:-1], p[1:]
    chg = a != b
    bad = 0
    for r, s in zip(a[chg].tolist(), b[chg].tolist()):
        bad += int((r, s) not in ALLOWED)
    return bad
