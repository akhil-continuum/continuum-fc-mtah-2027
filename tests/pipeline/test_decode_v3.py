"""Synthetic tests for v3 (control-evidence switch cost): direction-aware switch costs, the restart
law with per-frame costs, degrade-to-v2s, the anchor builder. Scripted sequences only."""
import itertools

import numpy as np
import pandas as pd
import pytest

from pipeline.decode import control, decoders, inputs, viterbi
from pipeline.decode.decoders import A, H
from pipeline.events.base import event_frame

V3 = dict(decoders.VERSIONS["v3a"]["dials"])


def probs_from(labels, conf=0.8):
    p = np.full((len(labels), 3), (1 - conf) / 2)
    for i, c in enumerate(labels):
        if c == "L":
            p[i] = (0.45, 0.45, 0.10)
        else:
            p[i, "HAD".index(c)] = conf
    return p


def seq(*parts):
    return "".join(c * n for c, n in parts)


def runs(codes):
    return [(c, len(list(g))) for c, g in itertools.groupby(codes.tolist())]


def with_anchors(p, rows, seen=True):
    inp = inputs.from_arrays(p)
    ev = event_frame(rows)
    inp.anchors = control.anchors(ev, inp.halves)
    inp.seen = np.full(inp.n, bool(seen))
    return inp


def carry(f, team, length=15):
    return [{"frame": f, "kind": "touch", "team": team, "conf": 1.0},
            {"frame": f, "kind": "carry", "team": team, "conf": 1.0, "end_frame": f + length},
            {"frame": f, "kind": "control", "team": team, "conf": 1.0,
             "first_touch_frame": f, "confirmed_frame": f + 9}]


# --------------------------------------------------------------------- transition table


def test_viterbi_directional_switch_matches_brute_force():
    rng = np.random.default_rng(5)
    for _ in range(30):
        T = int(rng.integers(1, 7))
        e = rng.uniform(0, 3, (T, 4))
        cs = rng.uniform(0, 3, (T, 2))
        cst, cr = rng.uniform(0, 3, T), rng.uniform(0, 3, T)
        k = (1, 1, int(rng.integers(1, 3)), 2)
        k = (k[0], k[0], k[2], k[2])
        path, cost = viterbi.decode_segment(e, cs, cst, cr, k)
        best = np.inf
        for cand in itertools.product(range(4), repeat=T):
            if viterbi.path_violations(cand):
                continue
            rl = [(c, len(list(g))) for c, g in itertools.groupby(cand)]
            if any(ln < k[c] for c, ln in rl[1:-1]):
                continue
            tot = sum(e[t, cand[t]] for t in range(T))
            for t in range(1, T):
                r, s = cand[t - 1], cand[t]
                if r == s:
                    continue
                if {r, s} == {0, 1}:
                    tot += cs[t, s]
                else:
                    tot += cst[t] if s >= 2 else cr[t]
            best = min(best, tot)
        assert cost == pytest.approx(best)
        assert viterbi.path_violations(path) == 0


@pytest.mark.parametrize("seed", range(15))
def test_restart_law_holds_with_per_frame_switch_costs(seed):
    """Random evidence, random anchors (some at zero cost): a stoppage never exits to the wrong
    team and no forbidden hidden transition appears."""
    rng = np.random.default_rng(100 + seed)
    labels = "".join(rng.choice(list("HAD")) * int(rng.integers(1, 80)) for _ in range(40))
    n = len(labels)
    rows = []
    for f in rng.integers(0, n, 30):
        rows += carry(int(f), int(rng.integers(0, 2)))
    inp = with_anchors(probs_from(labels, float(rng.uniform(0.4, 0.95))), rows)
    inp.restart_llr = rng.normal(0, 3, n)
    inp.evidence = {}
    dials = dict(V3, b_ctrl=float(rng.uniform(0, 60)), c_base=float(rng.uniform(5, 60)))
    codes, hidden = decoders.decode_v2(inp, dials)
    assert viterbi.path_violations(hidden) == 0
    for t in np.flatnonzero((hidden[:-1] >= 2) & (hidden[1:] < 2)):
        assert hidden[t + 1] == (viterbi.H if hidden[t] == viterbi.DH else viterbi.A)


# --------------------------------------------------------------------- behaviour


def test_no_events_is_exactly_v2s():
    rng = np.random.default_rng(1)
    labels = "".join(rng.choice(list("HAD")) * int(rng.integers(5, 120)) for _ in range(50))
    inp = inputs.from_arrays(probs_from(labels, 0.6))
    c3, _ = decoders.decode("v3a", inp)
    c2, _ = decoders.decode("v2s", inp)
    assert (c3 == c2).all()


def test_ball_unseen_falls_back_to_the_v2s_cost():
    p = probs_from(seq(("H", 100)))
    inp = with_anchors(p, carry(50, A), seen=False)
    cost = decoders.switch_cost_array(inp, V3)
    far = np.r_[0:30, 80:100]
    assert np.allclose(cost[far], V3["c_switch"])
    inp.seen[:] = True
    cost = decoders.switch_cost_array(inp, V3)
    assert np.allclose(cost[far], V3["c_base"])


def test_discount_only_into_the_team_with_control():
    p = probs_from(seq(("H", 100)))
    inp = with_anchors(p, carry(50, A))
    cost = decoders.switch_cost_array(inp, V3)
    assert cost[50, 1] == pytest.approx(V3["c_base"] - V3["b_ctrl"])     # into A, cheap
    assert cost[50, 0] == pytest.approx(V3["c_base"])                    # into H, unchanged


def test_control_switch_survives_where_proximity_alone_does_not():
    """A weak 60-frame lean to A (60 * log(0.55 / 0.30) = 36 nats): kept when A shows a carry
    at its start (two discounted switches, 2 * 15), dropped without (30 + 15)."""
    labels = seq(("H", 300), ("A", 60), ("H", 300))
    p = probs_from(labels, 0.8)
    p[300:360] = (0.30, 0.55, 0.15)              # weak lean only (proximity-like evidence)
    rows_back = carry(360, H)
    with_ctrl = with_anchors(p, carry(300, A) + rows_back)
    without = with_anchors(p, rows_back)
    c1, _ = decoders.decode("v3a", with_ctrl)
    c0, _ = decoders.decode("v3a", without)
    assert [c for c, _ in runs(c1)] == [H, A, H]
    assert runs(c0) == [(H, 660)]


def test_switch_lands_near_the_control_first_touch():
    labels = seq(("H", 200), ("L", 30), ("A", 200))
    inp = with_anchors(probs_from(labels), carry(215, A))
    codes, _ = decoders.decode("v3a", inp)
    first_a = int(np.flatnonzero(codes == A)[0])
    assert abs(first_a - 215) <= 2


def test_deflection_gives_no_discount():
    p = probs_from(seq(("H", 100)))
    rows = [{"frame": 40, "kind": "touch", "team": H, "conf": 1.0},
            {"frame": 40, "kind": "deflection", "team": A, "conf": 0.9},
            {"frame": 40, "kind": "touch", "team": A, "conf": 1.0},
            {"frame": 45, "kind": "touch", "team": H, "conf": 1.0}]
    inp = with_anchors(p, rows)
    cost = decoders.switch_cost_array(inp, V3)
    assert np.allclose(cost[:, 1], V3["c_base"])


def test_anchor_grades():
    rows = (carry(10, H)
            + [{"frame": 200, "kind": "touch", "team": A, "conf": 1.0},
               {"frame": 210, "kind": "touch", "team": A, "conf": 1.0}]
            + [{"frame": 400, "kind": "touch", "team": H, "conf": 1.0}])
    anc = control.anchors(event_frame(rows), [(0, 1000)])
    got = {(int(f), int(t)): int(g) for f, t, g in zip(anc.frame, anc.team, anc.grade)}
    assert got == {(10, H): control.GRADE_CONTROL, (200, A): control.GRADE_TWO,
                   (400, H): control.GRADE_SINGLE}
    assert pd.api.types.is_integer_dtype(anc["n_contacts"])


# --------------------------------------------------------------------- v4a stop-cost hook


def test_stop_cost_hook_is_off_by_default_and_local():
    inp = inputs.from_arrays(probs_from(seq(("H", 300))))
    inp.stop_points = (np.array([100]), np.array([1.0]))
    dials = dict(decoders.VERSIONS["v2s"]["dials"])
    assert decoders.stop_cost_array(inp, dials) is None            # b_stop absent: v2s
    cost = decoders.stop_cost_array(inp, dict(dials, b_stop=30.0, stop_width=10))
    assert cost[100] == pytest.approx(dials["c_stop"] - 30.0)
    assert np.allclose(cost[:91], dials["c_stop"]) and np.allclose(cost[110:], dials["c_stop"])
