"""Synthetic tests for the T-42 evidence-dependent switch cost (pipeline.decode.evidence_cost):
confident evidence for the new team inside the gate -> the switch is allowed quickly; weak or
absent evidence -> held; outside the gate -> exactly the constant cost; the restart law holds."""
import itertools

import numpy as np
import pytest

from pipeline.decode import checks, decoders, evidence_cost, inputs, viterbi
from pipeline.decode.decoders import A, D, H

DIALS = dict(decoders.VERSIONS["v2s"]["dials"], tree_cost=True, c_hi=45.0, theta=0.75)


def runs(codes):
    return [(c, len(list(g))) for c, g in itertools.groupby(codes.tolist())]


def probs_q(q, p_dead=0.05):
    """(n, 3) with P(H) = q * (1 - p_dead)."""
    q = np.asarray(q, dtype=float)
    return np.stack([q * (1 - p_dead), (1 - q) * (1 - p_dead), np.full(len(q), p_dead)], 1)


def excursion(q_exc, length, before=300, after=300, q_home=0.95):
    q = np.concatenate([np.full(before, q_home), np.full(length, 1 - q_exc),
                        np.full(after, q_home)])
    return probs_q(q)


def with_gate(p, gate=True):
    inp = inputs.from_arrays(p)
    inp.gate = np.full(inp.n, bool(gate))
    return inp


# ------------------------------------------------------------------------- the cost itself

def test_cost_is_monotone_bounded_and_centred():
    q = np.full(101, 0.5)
    q[50] = 0.99                                          # one confident home frame
    p = inputs.clean_probs(probs_q(q))[0]
    c = evidence_cost.switch_cost(p, np.ones(101, bool), 18.0, 45.0, 0.75, w=12)
    assert c.min() >= 18.0 - 1e-9 and c.max() <= 45.0 + 1e-9
    # cheap into H within +-12 frames of the confident frame, on both sides (centred)
    assert c[38:63, 0].max() < 20.0 and c[37, 0] == pytest.approx(45.0)
    assert c[63, 0] == pytest.approx(45.0)
    # more confidence never costs more
    lo = evidence_cost.switch_cost(p, np.ones(101, bool), 18.0, 45.0, 0.6, w=12)
    assert (lo <= c + 1e-12).all()


def test_outside_the_gate_the_cost_is_the_constant():
    p = inputs.clean_probs(excursion(0.95, 20))[0]
    c = evidence_cost.switch_cost(p, np.zeros(len(p), bool), 18.0, 45.0, 0.75)
    assert np.all(c == 18.0)


def test_gate_from_reference_marks_only_differing_frames():
    a = probs_q(np.full(10, 0.7))
    b = a.copy()
    b[3:5] = probs_q(np.full(2, 0.4))
    g = evidence_cost.gate_from_reference(a, b)
    assert g.tolist() == [False] * 3 + [True] * 2 + [False] * 5


# ------------------------------------------------------------------------- decoding

def test_confident_short_play_by_the_other_team_is_allowed_quickly():
    # 15 frames at q_A 0.99: about 69 nats of evidence. Entry and exit each cost about 19
    # (confidence 0.96), well under 69; a flat 45 would need 90.
    p = excursion(0.99, 15)
    codes, _ = decoders.decode_v2(with_gate(p), DIALS)
    assert runs(codes) == [(H, 300), (A, 15), (H, 300)]


def test_weak_excursion_is_held_inside_the_gate_but_not_outside():
    # 40 frames at q_A 0.75: about 44 nats too, but never confident: entry costs c_hi 45.
    p = excursion(0.75, 40)
    codes, _ = decoders.decode_v2(with_gate(p, True), DIALS)
    assert runs(codes) == [(H, 640)]
    # outside the gate the same evidence switches, exactly as with the constant cost (v11g)
    codes_out, _ = decoders.decode_v2(with_gate(p, False), DIALS)
    const, _ = decoders.decode("v11g", inputs.from_arrays(p))
    assert runs(codes_out) == [(H, 300), (A, 40), (H, 300)]
    assert np.array_equal(codes_out, const)


def test_flat_high_cost_holds_the_confident_play_too():
    # theta 1.0 is the gate-only ablation: no evidence term, so the confident play is refused
    p = excursion(0.99, 15)
    codes, _ = decoders.decode_v2(with_gate(p), dict(DIALS, theta=1.0))
    assert runs(codes) == [(H, 615)]


def test_tree_cost_without_gate_refuses_to_run():
    with pytest.raises(ValueError):
        decoders.decode_v2(inputs.from_arrays(excursion(0.95, 15)), DIALS)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_restart_law_holds_with_the_tree_cost(seed):
    rng = np.random.default_rng(seed)
    n = 3000
    p = rng.dirichlet((1, 1, 1), n)
    llr = rng.normal(0, 0.5, n)
    hh = rng.random(n) < 0.02
    ha = (rng.random(n) < 0.02) & ~hh
    inp = inputs.from_arrays(p, restart_llr=llr, hard_home=hh, hard_away=ha)
    inp.gate = rng.random(n) < 0.5
    codes, hidden = decoders.decode_v2(inp, dict(DIALS, c_hi=float(rng.uniform(18, 60))))
    assert viterbi.path_violations(hidden) == 0
    for t in np.flatnonzero((hidden[:-1] >= 2) & (hidden[1:] < 2)):
        assert hidden[t + 1] == (viterbi.H if hidden[t] == viterbi.DH else viterbi.A)
    failures, _, _ = checks.check(codes, inp, hidden)
    assert failures == []
    assert set(np.unique(codes)) <= {H, A, D}
