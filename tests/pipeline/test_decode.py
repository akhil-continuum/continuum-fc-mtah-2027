"""Synthetic tests for the decode stage: transition table, restart law, minimum durations, masks,
the output checks. No match data; sequences are scripted here."""
import itertools

import numpy as np
import pytest

from pipeline.decode import checks, decoders, inputs, viterbi
from pipeline.decode.decoders import A, D, H, X

V2 = dict(decoders.VERSIONS["v2"]["dials"])
V1 = dict(decoders.VERSIONS["v1"]["dials"])


def probs_from(labels, conf=0.8):
    """(n, 3) probabilities peaking at the scripted labels ('H', 'A', 'D' or '?' flat)."""
    p = np.full((len(labels), 3), (1 - conf) / 2)
    for i, c in enumerate(labels):
        if c == "?":
            p[i] = 1 / 3
        elif c == "L":                      # live, team unknown
            p[i] = (0.45, 0.45, 0.10)
        else:
            p[i, "HAD".index(c)] = conf
    return p


def seq(*parts):
    return "".join(c * n for c, n in parts)


def runs(codes):
    return [(c, len(list(g))) for c, g in itertools.groupby(codes.tolist())]


# --------------------------------------------------------------------- transition table


def test_transition_table_is_the_restart_law():
    allowed = viterbi.ALLOWED
    assert (viterbi.DH, viterbi.A) not in allowed
    assert (viterbi.DA, viterbi.H) not in allowed
    assert (viterbi.DH, viterbi.DA) not in allowed and (viterbi.DA, viterbi.DH) not in allowed
    assert {(viterbi.DH, viterbi.H), (viterbi.DA, viterbi.A)} <= allowed
    assert {(viterbi.H, viterbi.A), (viterbi.A, viterbi.H)} <= allowed
    for live in (viterbi.H, viterbi.A):
        for dead in (viterbi.DH, viterbi.DA):
            assert (live, dead) in allowed
    assert len(allowed) == 8


@pytest.mark.parametrize("seed", range(25))
def test_restart_law_never_violated_on_random_evidence(seed):
    """Adversarial evidence: random runs, restart evidence pointing the wrong way."""
    rng = np.random.default_rng(seed)
    labels = "".join(rng.choice(list("HAD?")) * int(rng.integers(1, 80)) for _ in range(40))
    n = len(labels)
    p = probs_from(labels, conf=float(rng.uniform(0.4, 0.95)))
    llr = rng.normal(0, 3, n)
    hh = rng.random(n) < 0.02
    ha = (rng.random(n) < 0.02) & ~hh
    inp = inputs.from_arrays(p, restart_llr=llr, hard_home=hh, hard_away=ha)
    dials = dict(V2, min_poss=int(rng.integers(1, 12)), min_dead=int(rng.integers(1, 60)),
                 c_switch=float(rng.uniform(0, 15)))
    codes, hidden = decoders.decode_v2(inp, dials)
    assert viterbi.path_violations(hidden) == 0
    # output: every dead run exits to the team of its hidden dead branch
    for t in np.flatnonzero((hidden[:-1] >= 2) & (hidden[1:] < 2)):
        assert hidden[t + 1] == (viterbi.H if hidden[t] == viterbi.DH else viterbi.A)
    assert set(np.unique(codes)) <= {H, A, D}


@pytest.mark.parametrize("version", ["v2", "v2h"])
def test_chosen_dials_obey_the_law_and_the_masks(version):
    rng = np.random.default_rng(11)
    labels = "".join(rng.choice(list("HAD")) * int(rng.integers(5, 200)) for _ in range(60))
    n = len(labels)
    mask = np.zeros(n, dtype=bool)
    mask[:150] = True
    hh = np.zeros(n, dtype=bool)
    hh[100:150] = True
    inp = inputs.from_arrays(probs_from(labels, 0.6), mask_dead=mask, hard_home=hh)
    codes, hidden = decoders.decode(version, inp)
    failures, _, agg = checks.check(codes, inp, hidden)
    assert failures == [] and agg["hidden_path_violations"] == 0
    assert (codes[:150] == D).all()


def test_restart_evidence_picks_the_team_after_the_stoppage():
    # live H, stoppage, then flat evidence: the restart hint alone decides who restarts
    labels = seq(("A", 200), ("D", 100), ("L", 300))
    p = probs_from(labels)
    for sign, team in ((1, H), (-1, A)):
        llr = np.zeros(len(labels))
        llr[250:300] = sign * np.log(0.9 / 0.1) / 50
        codes, hidden = decoders.decode_v2(inputs.from_arrays(p, restart_llr=llr), V2)
        d_end = np.flatnonzero(codes == D).max()
        assert codes[d_end + 1] == team and (codes[d_end + 1:] == team).all()


def test_law_mask_forces_the_restart_team_against_the_evidence():
    labels = seq(("H", 200), ("D", 100), ("A", 200))          # evidence says A restarts
    hh = np.zeros(len(labels), dtype=bool)
    hh[250:300] = True                                      # law: home restarts
    inp = inputs.from_arrays(probs_from(labels), hard_home=hh)
    codes, hidden = decoders.decode_v2(inp, V2)
    d_end = np.flatnonzero(codes == D).max()
    assert codes[d_end + 1] == H
    failures, _, agg = checks.check(codes, inp, hidden)
    assert agg["law_restarts_wrong_team"] == 0 and not failures


# --------------------------------------------------------------------- minimum durations


def test_synthetic_short_possession_between_two_long_ones_survives():
    # synthetic data: a clean short possession (20 frames, p = 0.8) must survive
    labels = seq(("H", 300), ("A", 20), ("H", 300))
    codes, _ = decoders.decode_v2(inputs.from_arrays(probs_from(labels)), V2)
    assert runs(codes) == [(H, 300), (A, 20), (H, 300)]


def test_synthetic_shorter_possession_is_possible():
    labels = seq(("H", 300), ("A", 19), ("H", 300))
    codes, _ = decoders.decode_v2(inputs.from_arrays(probs_from(labels, 0.9)), V2)
    assert [c for c, _ in runs(codes)] == [H, A, H]


def test_5_frame_blip_does_not_survive():
    labels = seq(("H", 300), ("A", 5), ("H", 300))
    codes, _ = decoders.decode_v2(inputs.from_arrays(probs_from(labels)), V2)
    assert runs(codes) == [(H, 605)]
    codes1, _ = decoders.decode_v1(inputs.from_arrays(probs_from(labels)), V1)
    assert runs(codes1) == [(H, 605)]


def test_v1_keeps_synthetic_short_possession_and_dates_switch_back():
    labels = seq(("H", 300), ("A", 20), ("H", 300))
    codes, _ = decoders.decode_v1(inputs.from_arrays(probs_from(labels)), V1)
    assert runs(codes) == [(H, 300), (A, 20), (H, 300)]


def test_minimum_durations_are_exact():
    rng = np.random.default_rng(7)
    labels = "".join(rng.choice(list("HAD")) * int(rng.integers(1, 30)) for _ in range(200))
    inp = inputs.from_arrays(probs_from(labels, 0.95))
    dials = dict(V2, c_switch=0.0, c_stop=0.0, c_restart=0.0, min_poss=8, min_dead=25)
    codes, hidden = decoders.decode_v2(inp, dials)
    lens = [(c, len(list(g))) for c, g in itertools.groupby(hidden.tolist())]
    for c, ln in lens[1:-1]:                         # inner runs; the edges may be shorter
        assert ln >= (8 if c < 2 else 25)


def test_viterbi_matches_brute_force_on_tiny_problems():
    rng = np.random.default_rng(3)
    for _ in range(30):
        T = int(rng.integers(1, 8))
        e = rng.uniform(0, 3, (T, 4))
        cs, cst, cr = (rng.uniform(0, 3, T) for _ in range(3))
        k = [int(v) for v in rng.integers(1, 4, 2)]
        k = (k[0], k[0], k[1], k[1])
        path, cost = viterbi.decode_segment(e, cs, cst, cr, k)
        best = np.inf
        for cand in itertools.product(range(4), repeat=T):
            if viterbi.path_violations(cand):
                continue
            rl = [(c, len(list(g))) for c, g in itertools.groupby(cand)]
            if any(ln < k[c] for c, ln in rl[1:-1]):
                continue
            if len(rl) > 1 and rl[-1][1] < 1:
                continue
            tot = sum(e[t, cand[t]] for t in range(T))
            for t in range(1, T):
                r, s = cand[t - 1], cand[t]
                if r != s:
                    tot += cs[t] if {r, s} == {0, 1} else (cst[t] if s >= 2 else cr[t])
            best = min(best, tot)
        assert cost == pytest.approx(best)
        assert viterbi.path_violations(path) == 0


# --------------------------------------------------------------------- masks and X


def test_masks_hold_and_x_only_on_ungraded():
    labels = seq(("?", 50), ("H", 200), ("A", 200), ("?", 30), ("A", 100), ("H", 200))
    n = len(labels)
    p = probs_from(labels)
    spans = [(50, 450), (480, n)]
    mask = np.zeros(n, dtype=bool)
    mask[480:560] = True                           # second half before its kick-off: D
    for name in ("v1", "v2"):
        inp = inputs.from_arrays(p, graded_spans=spans, mask_dead=mask)
        codes, hidden = decoders.decode(name, inp)
        assert (codes[:50] == X).all() and (codes[450:480] == X).all()
        assert (codes[50:450] != X).all() and (codes[480:] != X).all()
        assert (codes[480:560] == D).all()
        failures, _, _ = checks.check(codes, inp, hidden)
        assert failures == []


def test_halves_are_decoded_separately():
    labels = seq(("H", 300), ("H", 300))
    inp = inputs.from_arrays(probs_from(labels), graded_spans=[(0, 300), (300, 600)])
    mask = np.zeros(600, dtype=bool)
    mask[300:360] = True
    inp.mask_dead = mask
    codes, hidden = decoders.decode_v2(inp, V2)
    assert codes[299] == H and (codes[300:360] == D).all()


def test_segment_extends_back_over_masked_frames_before_graded_start():
    inp = inputs.from_arrays(probs_from(seq(("D", 100), ("H", 200))), graded_spans=[(100, 300)])
    inp.mask_dead = np.zeros(300, dtype=bool)
    inp.mask_dead[:100] = True
    inp.halves = [(0, 300)]
    assert inputs.decode_segments(inp) == [(0, 300)]


# --------------------------------------------------------------------- checks catch errors


def test_checks_catch_bad_outputs():
    labels = seq(("H", 100), ("D", 100), ("A", 100))
    n = len(labels)
    hh = np.zeros(n, dtype=bool)
    hh[150:200] = True
    mask = np.zeros(n, dtype=bool)
    mask[100:120] = True
    inp = inputs.from_arrays(probs_from(labels), graded_spans=[(0, 290)], mask_dead=mask,
                             hard_home=hh)
    codes = np.array([H] * 100 + [D] * 100 + [A] * 90 + [X] * 10, dtype=np.int8)
    f, _, agg = checks.check(codes, inp)
    assert agg["law_restarts_wrong_team"] == 1
    codes[105] = H
    codes[295] = H
    codes[5] = X
    f, _, _ = checks.check(codes, inp)
    joined = " ".join(f)
    assert "S4" in joined and "graded frames are X" in joined and "ungraded" in joined
    hidden = np.array([0] * 100 + [2] * 100 + [1] * 100, dtype=np.int8)
    f, _, agg = checks.check(np.where(np.arange(n) < 290, viterbi.OUTPUT_CODE[hidden], X)
                             .astype(np.int8), inp, hidden)
    assert agg["hidden_path_violations"] == 1


def test_restart_evidence_from_tables():
    import pandas as pd
    rs = pd.DataFrame({"start_frame": [-1, 10], "restart_frame": [100, 300],
                       "p_home_restart": [0.998, 0.3]})
    ev = pd.DataFrame({"frame": [500, 280], "kind": ["exit", "exit"], "team": [-1, -1],
                       "conf": [0.85, 0.85], "last_touch_team": [0, 1]})
    llr, hh, ha, counts = inputs.restart_evidence(1000, rs, ev)
    assert hh[50:100].all() and not hh[100:].any()
    assert llr[250:300].sum() == pytest.approx(np.log(0.3 / 0.7))
    assert llr[500:550].sum() == pytest.approx(np.log(0.2 / 0.8))    # H touched last: A
    assert counts["exit_covered_by_row"] == 1 and counts["exit_evidence_used"] == 1


# --------------------------------------------------------------------- T-5b: evidence modes, bonus


@pytest.mark.parametrize("version", ["v2r", "v2hr", "v2s"])
def test_new_versions_obey_the_law_and_the_masks(version):
    rng = np.random.default_rng(12)
    labels = "".join(rng.choice(list("HAD")) * int(rng.integers(5, 200)) for _ in range(60))
    n = len(labels)
    mask = np.zeros(n, dtype=bool)
    mask[:150] = True
    hh = np.zeros(n, dtype=bool)
    hh[100:150] = True
    inp = inputs.from_arrays(probs_from(labels, 0.6), mask_dead=mask, hard_home=hh)
    inp.restart_points = (np.array([400, 900]), np.array([0.9, 0.5]))
    codes, hidden = decoders.decode(version, inp)
    failures, _, agg = checks.check(codes, inp, hidden)
    assert failures == [] and agg["hidden_path_violations"] == 0
    assert (codes[:150] == D).all()


@pytest.mark.parametrize("version", ["v1r", "v2r", "v2s"])
def test_constrained_new_versions_keep_short_possessions(version):
    keep = seq(("H", 300), ("A", 20), ("H", 300))
    blip = seq(("H", 300), ("A", 5), ("H", 300))
    codes, _ = decoders.decode(version, inputs.from_arrays(probs_from(keep)))
    assert runs(codes) == [(H, 300), (A, 20), (H, 300)]
    codes, _ = decoders.decode(version, inputs.from_arrays(probs_from(blip)))
    assert runs(codes) == [(H, 605)]


def test_restart_bonus_is_local_and_capped_by_conf():
    inp = inputs.from_arrays(probs_from("H" * 400))
    inp.restart_points = (np.array([100, 300]), np.array([1.0, 0.5]))
    dials = {"c_restart": 30.0, "b_restart": 20.0, "restart_width": 10}
    c = decoders.restart_cost_array(inp, dials)
    assert c[100] == pytest.approx(10.0) and c[300] == pytest.approx(20.0)
    assert c[105] == pytest.approx(30.0 - 20.0 * 0.5)
    assert (c[:91] == 30.0).all() and (c[111:291] == 30.0).all() and (c[310:] == 30.0).all()
    assert (decoders.restart_cost_array(inp, {"c_restart": 30.0}) == 30.0).all()


def test_restart_bonus_moves_the_restart_to_the_restart_frame():
    # stoppage, then evidence of play that turns live only gradually: the bonus dates the restart
    labels = seq(("H", 300), ("D", 200), ("?", 40), ("H", 300))
    p = probs_from(labels, 0.9)
    base = dict(decoders.VERSIONS["v2s"]["dials"], b_restart=0.0)
    inp = inputs.from_arrays(p)
    inp.restart_points = (np.array([515]), np.array([1.0]))
    c0, _ = decoders.decode_v2(inp, base)
    c1, _ = decoders.decode_v2(inp, dict(base, b_restart=40.0, restart_width=8))
    first_live = lambda c: 300 + int(np.flatnonzero(c[300:] != D)[0])   # noqa: E731
    assert abs(first_live(c1) - 515) < abs(first_live(c0) - 515) or first_live(c1) == 515
    assert first_live(c1) == 515


def test_evidence_modes_and_law_only_for_kickoffs():
    import pandas as pd
    rs = pd.DataFrame({"start_frame": [0, 200, 600], "restart_frame": [100, 300, 700],
                       "restart_type": ["kick_off", "throw_in", "corner"],
                       "p_home_restart": [0.998, 0.995, 0.2], "conf": [0.95, 0.8, 0.6]})
    llr, hh, ha, counts = inputs.restart_evidence(1000, rs, None)
    assert hh[50:100].all()                         # kick-off at law level: mask
    assert not hh[250:300].any() and llr[250:300].sum() > 0     # throw-in 0.995: soft only
    k_llr, k_hh, _, _ = inputs.restart_evidence(1000, inputs.kickoff_rows(rs), None)
    assert k_hh[50:100].all() and k_llr[250:].sum() == 0
    fr, cf = inputs.restart_points(rs)
    assert list(fr) == [100, 300, 700] and list(cf) == [0.95, 0.8, 0.6]
