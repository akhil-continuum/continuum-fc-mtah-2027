"""Output checks shared by every decoder version (rulebook Part 2, "Output checks that block a
version"), plus the aggregate statistics printed on the match. Aggregates only.

Blocking (``failures``):
- row count equals n_frames; alphabet is H, A, D, X;
- X exactly on ungraded frames (S1);
- every S4-masked graded frame is D;
- hidden path (Viterbi versions) uses only allowed transitions (R0);
- no stoppage followed by a possession of the team the law forbids: where the restart evidence
  is law (inputs.hard_home / hard_away inside the stoppage), the first live label must agree.
Warnings (``warnings``): S11 ranges (share of D, stoppages per 90 min, possessions per minute)
and stoppages shorter than the S10 minimum. Warnings never block and are never tuned to.
"""
import numpy as np

from pipeline.decode import viterbi

H, A, D, X = 0, 1, 2, 3
FPS = 25.0
S11_SHARE_D = (0.35, 0.50)
S11_STOPPAGES_PER_90 = (90, 110)
S11_POSS_PER_MIN = (3.0, 5.0)
S10_MIN_DEAD_FRAMES = 50
HIST_EDGES_S = (0, 1, 1.5, 2, 4, 8, 15, 30, 60, 120, 1e9)     # 10 bins


def runs_in(codes, spans):
    """[(start, stop, code)] constant runs inside each span."""
    out = []
    for a, b in spans:
        w = codes[a:b]
        if not len(w):
            continue
        br = np.flatnonzero(w[1:] != w[:-1]) + 1
        bounds = np.concatenate(([0], br, [len(w)]))
        out += [(a + int(bounds[i]), a + int(bounds[i + 1]), int(w[bounds[i]]))
                for i in range(len(bounds) - 1)]
    return out


def statistics(codes, spans):
    runs = runs_in(codes, spans)
    graded = sum(b - a for a, b in spans)
    minutes = graded / FPS / 60.0
    poss = np.array([b - a for a, b, c in runs if c in (H, A)], dtype=float) / FPS
    dead = np.array([b - a for a, b, c in runs if c == D], dtype=float) / FPS
    share_d = float(sum(b - a for a, b, c in runs if c == D) / graded) if graded else None
    hist = np.histogram(poss, bins=HIST_EDGES_S)[0].tolist() if len(poss) else [0] * 10
    return {
        "graded_minutes": round(minutes, 2),
        "possessions": int(len(poss)),
        "possessions_per_min": float(len(poss) / minutes) if minutes else None,
        "share_d": share_d,
        "stoppages": int(len(dead)),
        "stoppages_per_90": float(len(dead) / minutes * 90) if minutes else None,
        "shortest_possession_s": float(poss.min()) if len(poss) else None,
        "median_possession_s": float(np.median(poss)) if len(poss) else None,
        "share_possessions_under_1_5s": float(np.mean(poss < 1.5)) if len(poss) else None,
        "share_possessions_under_4s": float(np.mean(poss < 4.0)) if len(poss) else None,
        "shortest_stoppage_s": float(dead.min()) if len(dead) else None,
        "median_stoppage_s": float(np.median(dead)) if len(dead) else None,
        "possession_length_hist": hist,
    }


def wrong_team_restarts(codes, spans, hard_home, hard_away):
    """Stoppages (D runs) whose first following live label contradicts law-level evidence
    inside the stoppage. Returns (violations, stoppages with law evidence)."""
    bad = checked = 0
    for a, b, c in runs_in(codes, spans):
        if c != D:
            continue
        home, away = bool(hard_home[a:b].any()), bool(hard_away[a:b].any())
        if home == away:
            continue
        # first live label after the stoppage, inside the same span
        span_end = next(s1 for s0, s1 in spans if s0 <= a < s1)
        if b >= span_end:
            continue
        checked += 1
        bad += int(codes[b] != (H if home else A))
    return bad, checked


def check(codes, inp, hidden=None):
    """(failures list of short strings, warnings list, aggregates dict)."""
    failures, warnings = [], []
    codes = np.asarray(codes)
    if len(codes) != inp.n:
        failures.append("row count {} != {}".format(len(codes), inp.n))
        return failures, warnings, {}
    if not np.isin(codes, (H, A, D, X)).all():
        failures.append("labels outside H, A, D, X")
    x_graded = int(np.sum((codes == X) & inp.graded))
    x_wrong = int(np.sum((codes != X) & ~inp.graded))
    if x_graded:
        failures.append("{} graded frames are X".format(x_graded))
    if x_wrong:
        failures.append("{} ungraded frames are not X".format(x_wrong))
    m = inp.mask_dead & inp.graded
    mask_bad = int(np.sum(codes[m] != D))
    if mask_bad:
        failures.append("{} S4-masked frames are not D".format(mask_bad))
    agg = {}
    if hidden is not None:
        viol = 0
        for a, b in inp.graded_spans:
            viol += viterbi.path_violations(hidden[a:b])
        agg["hidden_path_violations"] = viol
        if viol:
            failures.append("{} forbidden hidden transitions".format(viol))
    bad, checked = wrong_team_restarts(codes, inp.graded_spans, inp.hard_home, inp.hard_away)
    agg["law_restarts_checked"] = checked
    agg["law_restarts_wrong_team"] = bad
    if bad:
        failures.append("{} restarts to the wrong team under the law".format(bad))

    stats = statistics(codes, inp.graded_spans)
    agg.update(stats)
    if stats["share_d"] is not None and not S11_SHARE_D[0] <= stats["share_d"] <= S11_SHARE_D[1]:
        warnings.append("S11 share of D outside range")
    if stats["stoppages_per_90"] is not None and \
            not S11_STOPPAGES_PER_90[0] <= stats["stoppages_per_90"] <= S11_STOPPAGES_PER_90[1]:
        warnings.append("S11 stoppages per 90 outside range")
    if stats["possessions_per_min"] is not None and \
            not S11_POSS_PER_MIN[0] <= stats["possessions_per_min"] <= S11_POSS_PER_MIN[1]:
        warnings.append("S11 possessions per minute outside range")
    short_dead = sum(1 for a, b, c in runs_in(codes, inp.graded_spans)
                     if c == D and b - a < S10_MIN_DEAD_FRAMES and not
                     any(a == s0 or b == s1 for s0, s1 in inp.graded_spans))
    agg["stoppages_under_s10_min"] = short_dead
    if short_dead:
        warnings.append("{} stoppages shorter than the S10 minimum".format(short_dead))
    return failures, warnings, agg
