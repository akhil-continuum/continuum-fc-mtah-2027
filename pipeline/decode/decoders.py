"""v1 hysteresis yardstick and the v2 Viterbi wrapper. Both return code arrays H0 A1 D2 X3.

Both decode each segment from ``inputs.decode_segments`` separately (S3), force D on the S4 mask
and write X on every ungraded frame only (S1).
"""
import numpy as np

from pipeline.decode import viterbi
from pipeline.decode.inputs import decode_segments

H, A, D, X = 0, 1, 2, 3
BIG = 1.0e3          # emission cost of a masked state per frame (finite, so cumulative sums
                     # stay exact; far above any transition cost)


# ----------------------------------------------------------------------------- v1 hysteresis

def hysteresis_segment(p, mask_dead, th_enter, th_exit, k_poss, k_dead):
    """Switch to class c after k[c] consecutive frames that support c, dated back to the first.

    A frame supports c (not the current class s) when P(c) >= th_enter and P(s) < th_exit.
    The confirmed run becomes c from its first frame, so every run lasts at least k[c] frames.
    Masked frames count as P(D) = 1.
    """
    T = len(p)
    out = np.empty(T, dtype=np.int8)
    if T == 0:
        return out
    pl = p.tolist()
    md = mask_dead.tolist()
    k = (int(k_poss), int(k_poss), int(k_dead))
    first = (0.0, 0.0, 1.0) if md[0] else pl[0]
    s = int(np.argmax(first))
    cand, run0 = -1, 0
    for t in range(T):
        pt = (0.0, 0.0, 1.0) if md[t] else pl[t]
        c = 0 if pt[0] >= pt[1] else 1
        if pt[2] > pt[c]:
            c = 2
        if c != s and pt[c] >= th_enter and pt[s] < th_exit:
            if c != cand:
                cand, run0 = c, t
        else:
            cand = -1
        out[t] = s
        if cand >= 0 and t - run0 + 1 >= k[cand]:
            out[run0:t + 1] = cand
            s, cand = cand, -1
    return out


def decode_v1(inp, dials):
    codes = np.full(inp.n, X, dtype=np.int8)
    for a, b in decode_segments(inp):
        seg = hysteresis_segment(inp.probs[a:b], inp.mask_dead[a:b], dials["th_enter"],
                                 dials["th_exit"], dials["k_poss"], dials["k_dead"])
        seg[inp.mask_dead[a:b]] = D
        codes[a:b] = seg
    codes[~inp.graded] = X
    return codes, None


# ----------------------------------------------------------------------------- v2 Viterbi

def select_evidence(inp, mode):
    """(restart_llr, hard_home, hard_away) for an evidence mode ("all" or "kickoff")."""
    return inp.evidence.get(mode, (inp.restart_llr, inp.hard_home, inp.hard_away))


def restart_cost_array(inp, dials):
    """Per-frame dead -> live cost: c_restart, minus a bonus near each restart row's
    restart_frame (v2s): b_restart * conf * triangle(|t - restart_frame| / restart_width).
    Overlapping kernels take the largest bonus. Evidence, not a constraint."""
    base = np.full(inp.n, float(dials["c_restart"])) if inp.restart_cost is None \
        else np.asarray(inp.restart_cost, dtype=np.float64).copy()
    b = float(dials.get("b_restart", 0.0))
    if b <= 0 or inp.restart_points is None or not len(inp.restart_points[0]):
        return base
    width = max(1, int(dials.get("restart_width", 12)))
    bonus = np.zeros(inp.n)
    offs = np.arange(-width + 1, width)
    kern = 1.0 - np.abs(offs) / float(width)
    for r, c in zip(*inp.restart_points):
        idx = r + offs
        ok = (idx >= 0) & (idx < inp.n)
        np.maximum.at(bonus, idx[ok], b * float(c) * kern[ok])
    return base - bonus


def stop_cost_array(inp, dials):
    """Per-frame live -> dead cost (v4a): c_stop, minus b_stop * conf * triangle(|t - start| /
    stop_width) near each stoppage candidate's start frame (T-7 restarts rows). None when
    b_stop is 0 or there are no candidates (constant c_stop, v2s behaviour)."""
    b = float(dials.get("b_stop", 0.0))
    if b <= 0 or inp.stop_points is None or not len(inp.stop_points[0]):
        return inp.stop_cost
    base = np.full(inp.n, float(dials["c_stop"])) if inp.stop_cost is None \
        else np.asarray(inp.stop_cost, dtype=np.float64).copy()
    width = max(1, int(dials.get("stop_width", 25)))
    bonus = np.zeros(inp.n)
    offs = np.arange(-width + 1, width)
    kern = 1.0 - np.abs(offs) / float(width)
    for f, c in zip(*inp.stop_points):
        idx = f + offs
        ok = (idx >= 0) & (idx < inp.n)
        np.maximum.at(bonus, idx[ok], b * float(c) * kern[ok])
    return np.maximum(0.0, base - bonus)


def emissions(inp, a, b, w_restart=1.0, evidence="all"):
    """(T, 4) costs: -log P(H), -log P(A), and -log P(D) split by restart evidence; masks."""
    p = inp.probs[a:b]
    lp = -np.log(p)
    r_llr, r_hh, r_ha = select_evidence(inp, evidence)
    llr = w_restart * r_llr[a:b]
    e = np.empty((b - a, 4), dtype=np.float64)
    e[:, viterbi.H] = lp[:, 0]
    e[:, viterbi.A] = lp[:, 1]
    e[:, viterbi.DH] = lp[:, 2] + np.maximum(0.0, -llr)
    e[:, viterbi.DA] = lp[:, 2] + np.maximum(0.0, llr)
    m = inp.mask_dead[a:b]
    e[m, viterbi.H] += BIG
    e[m, viterbi.A] += BIG
    e[r_hh[a:b], viterbi.DA] += BIG
    e[r_ha[a:b], viterbi.DH] += BIG
    return e


def _cost(arr, const, a, b):
    if arr is None:
        return np.full(b - a, float(const))
    return np.asarray(arr[a:b], dtype=np.float64)


def switch_cost_array(inp, dials):
    """(n, 2) per-frame cost of a switch into H (column 0) and into A (column 1), v3
    (pipeline.decode.control):

        c_none(t) = c_switch + (c_base - c_switch) * seen_share(t)
        cost_into_X(t) = max(C_FLOOR, c_none(t) - b_ctrl * strength_X(t))

    With no events table on the root it returns None, so the decoder uses the constant c_switch
    (exactly v2s). Without ball.parquet, seen_share is taken as 1 where anchors exist."""
    from pipeline.decode import control
    if inp.anchors is None:
        return None
    lo = float(dials["c_switch"])
    hi = float(dials.get("c_base", lo))
    if inp.seen is not None:
        share = control.seen_share(inp.seen, int(dials.get("seen_win", control.SEEN_HALF_WIN)))
    else:
        share = np.ones(inp.n)
    c_none = lo + (hi - lo) * share
    s = control.strength(inp.n, inp.anchors, dials.get("g_two", 0.0),
                         dials.get("ctrl_width", 12), dials.get("anchor", "first"),
                         dials.get("lead"), dials.get("g_one", 0.0), dials.get("shift", 0),
                         dials.get("n_full"))
    cost = c_none[:, None] - float(dials.get("b_ctrl", 0.0)) * s
    return np.maximum(control.C_FLOOR, cost)


def tree_switch_cost_array(inp, dials):
    """(n, 2) T-42 switch cost from the team model's own confidence inside its gate
    (pipeline.decode.evidence_cost): c_switch outside the gate, between c_switch and c_hi
    inside it. Needs inp.gate (inputs.attach_gate); without it the version cannot run."""
    from pipeline.decode import evidence_cost
    if inp.gate is None:
        raise ValueError("tree_cost needs the gate (inputs.attach_gate with the version's "
                         "gate_ref model)")
    return evidence_cost.switch_cost(inp.probs, inp.gate, dials["c_switch"], dials["c_hi"],
                                     dials["theta"], int(dials.get("conf_win",
                                                                   evidence_cost.W_DEFAULT)))


def decode_v2(inp, dials):
    """Returns (codes, hidden path with -1 outside the decoded segments)."""
    codes = np.full(inp.n, X, dtype=np.int8)
    hidden = np.full(inp.n, -1, dtype=np.int8)
    k = (dials["min_poss"], dials["min_poss"], dials["min_dead"], dials["min_dead"])
    rcost = restart_cost_array(inp, dials)
    scost = switch_cost_array(inp, dials) if dials.get("switch_evidence") else None
    if dials.get("tree_cost"):
        scost = tree_switch_cost_array(inp, dials)
    if scost is None:
        scost = inp.switch_cost
    stcost = stop_cost_array(inp, dials)
    for a, b in decode_segments(inp):
        e = emissions(inp, a, b, dials.get("w_restart", 1.0), dials.get("evidence", "all"))
        path, _ = viterbi.decode_segment(
            e, _cost(scost, dials["c_switch"], a, b),
            _cost(stcost, dials["c_stop"], a, b), rcost[a:b], k)
        hidden[a:b] = path
        codes[a:b] = viterbi.OUTPUT_CODE[path]
    codes[~inp.graded] = X
    return codes, hidden


# ----------------------------------------------------------------------------- versions

# Dials. v1 and v2 values were chosen on the bench (pipeline.decode.sweep), never on the public
# window; the start values and the sweep tables are in the T-5 report.
#
# Chosen on rule_v1, pooled metrica_g1-3 at the match level, checked at low and high (T-5):
# - c_switch is the only v2 dial that matters alone; c_stop and c_restart must scale with it, or
#   the decoder routes team changes through invented stoppages (stop + restart < switch).
#   min_poss, min_dead, w_restart move pooled spellF1 by less than 0.01: start values (S9, S10).
# - c_switch 18 sits on the bench plateau 18-30 (sweep tables, judge round two); min_poss 8 and
#   k_poss 19 are flat in the bench sweeps. The short-possession survival test was originally
#   motivated by the published 5-minute window showing that possessions under one second exist
#   (rulebook S9, a definition fact); the dial values are supported by the bench sweeps alone.
# - v2h: the unconstrained bench optimum (flat for c_switch 65-90 at c_stop = c_restart 30 on all
#   three levels; low end kept). It wins on spellF1 by suppressing short possessions, so it
#   deliberately fails the short-possession survival test; a separate candidate for the lead and the judge.
VERSIONS = {
    "v1": {"decoder": decode_v1,
           "dials": {"th_enter": 0.4, "th_exit": 0.4, "k_poss": 19, "k_dead": 50}},
    "v2": {"decoder": decode_v2,
           "dials": {"c_switch": 18.0, "c_stop": 30.0, "c_restart": 30.0,
                     "min_poss": 8, "min_dead": 50, "w_restart": 1.0}},
    "v2h": {"decoder": decode_v2,
            "dials": {"c_switch": 65.0, "c_stop": 30.0, "c_restart": 30.0,
                      "min_poss": 8, "min_dead": 50, "w_restart": 1.0}},
}
for _v in VERSIONS.values():          # T-5 versions: rows as found, bench computes kick-offs only
    _v["compute"] = "kickoff"
    _v["dials"].setdefault("evidence", "all")

# T-5b, on the refitted rule_v1. Evidence mode fixed per version so bench and match agree:
# "kickoff" = kick-off rows + exit last touch (the T-5 bench evidence), "all" = every T-7 row.
# Bench roots without restarts files compute the whole restarts stage in memory ("full").
# Refitted rule_v1, joint grid c_switch x (c_stop = c_restart), pooled g1-3 (T-5b):
# ridge flat for c_switch 18-30 at c_stop = c_restart 45 on match, low and high levels;
# v2r = 18 (on the plateau), v2hr = 22 (best mean over
# the three levels, unconstrained; within 0.003 of v2r). v2s = v2r + every T-7 row + bonus:
# row evidence (~1.7 nats a row) only matters when scaled (w_restart 6-10 lifts restart-team
# accuracy 0.876 -> 0.89-0.90; spellF1 flat, falls from w_restart 20); bonus 40 over +-8 frames.
VERSIONS.update({
    "v1r": {"decoder": decode_v1, "compute": "full",
            "dials": {"th_enter": 0.4, "th_exit": 0.4, "k_poss": 19, "k_dead": 50,
                      "evidence": "kickoff"}},
    "v2r": {"decoder": decode_v2, "compute": "full",
            "dials": {"c_switch": 18.0, "c_stop": 45.0, "c_restart": 45.0, "min_poss": 8,
                      "min_dead": 50, "w_restart": 1.0, "evidence": "kickoff"}},
    "v2hr": {"decoder": decode_v2, "compute": "full",
             "dials": {"c_switch": 22.0, "c_stop": 45.0, "c_restart": 45.0, "min_poss": 8,
                       "min_dead": 50, "w_restart": 1.0, "evidence": "kickoff"}},
    "v2s": {"decoder": decode_v2, "compute": "full",
            "dials": {"c_switch": 18.0, "c_stop": 45.0, "c_restart": 45.0, "min_poss": 8,
                      "min_dead": 50, "w_restart": 6.0, "evidence": "all",
                      "b_restart": 40.0, "restart_width": 8}},
})


# T-11, v3a: v2s plus a per-frame, per-direction switch cost from control evidence
# (pipeline.decode.control; rulebook P1 to P5 cheap, P7 / P13 / P15 dear). Without an events
# table it is exactly v2s. Dials from the T-11 joint grid c_base x b_ctrl (bench, pooled g1-3,
# low / match / high, default and confirmed truth): a diagonal ridge where the cost at a control
# anchor stays near the v2s level (c_base - b_ctrl 15 to 25) and the cost where the ball is seen
# with no evidence rises; c_base 30, b_ctrl 15 is mid-ridge and the only point that keeps v2s's
# possessions per minute and short-possession recall (+0.003 to +0.006 over v2s in all six
# cells, under the noise band). Grading by contact count (n_full) was tried and was no better.
# g_two, g_one, ctrl_width: flat (one-at-a-time sweep); start values kept.
VERSIONS["v3a"] = {"decoder": decode_v2, "compute": "full",
                   "dials": dict(VERSIONS["v2s"]["dials"], switch_evidence=True, c_base=30.0,
                                 b_ctrl=15.0, g_two=0.8, g_one=0.0, ctrl_width=12,
                                 anchor="first", n_full=None)}


# T-23, v6s: v2s's dials unchanged, read with the team model rule_v2 (P-9 / P-10:
# pipeline/models/team_model.py; P(D) is rule_v1's). "probs" names the model the version is meant
# for; the decode stage reads it (T-24): without --probs it decodes rule_v2 and writes v6s.csv.
VERSIONS["v6s"] = {"decoder": decode_v2, "compute": "full", "probs": "rule_v2",
                   "dials": dict(VERSIONS["v2s"]["dials"])}

# T-35 (P-17c), on the smoothed bench (noise v3) with the retuned touch detector (w15s, T-34).
# Ablation, no refit: v2s15 / v6s15 = v2s / v6s with w15s events and their stored weights.
# Refit: v2w = v2s's dials on rule_w1 (rule_v1's dead and team terms refitted there);
# v6w = v2s's dials on rule_w2 (rule_v2's columns refitted there, P(D) = rule_w1's).
# Weights in pipeline/models/team_w1_params.json / team_w2_params.json (models.refit_w).
VERSIONS["v2s15"] = {"decoder": decode_v2, "compute": "full", "touch_set": "w15s",
                     "dials": dict(VERSIONS["v2s"]["dials"])}
VERSIONS["v6s15"] = {"decoder": decode_v2, "compute": "full", "probs": "rule_v2",
                     "touch_set": "w15s", "dials": dict(VERSIONS["v2s"]["dials"])}
VERSIONS["v2w"] = {"decoder": decode_v2, "compute": "full", "probs": "rule_w1",
                   "touch_set": "w15s", "dials": dict(VERSIONS["v2s"]["dials"])}
VERSIONS["v6w"] = {"decoder": decode_v2, "compute": "full", "probs": "rule_w2",
                   "touch_set": "w15s", "dials": dict(VERSIONS["v2s"]["dials"])}

# T-32 (per-player trees, step 1), smoothed bench, w15s events, v2s's dials.
# v11g = rule_p1g (per-player trees inside the busy gate G4, rule_w2 elsewhere). The submitted
# version; v6w (above) is the same chain without the trees.
VERSIONS["v11g"] = {"decoder": decode_v2, "compute": "full", "probs": "rule_p1g",
                    "touch_set": "w15s", "dials": dict(VERSIONS["v2s"]["dials"])}


def decode(version, inp, overrides=None):
    spec = VERSIONS[version]
    dials = dict(spec["dials"])
    dials.update(overrides or {})
    return spec["decoder"](inp, dials)
