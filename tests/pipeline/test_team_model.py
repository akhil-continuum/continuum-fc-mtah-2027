"""Team model rule_v2 (pipeline/models/team_model.py): no lean to either team, P(D) is rule_v1's
exactly, tracked parameters are well formed, the stage writes the contract file."""
import os
import re

import numpy as np
import pandas as pd

from pipeline import features
from pipeline.models import rule_params as RP
from pipeline.models import rule_scorer as RS
from pipeline.models import team_model as TMOD
from pipeline.scenes import library as lib

ANTISYM = re.compile(r"^(?:{}|shape_[a-z_]+_(?:diff|rel)_c\d+)(?:\*unseen)?$".format(
    "|".join(RP.TEAM_W)))


def test_params_file_columns_are_antisymmetric_and_weights_complete():
    params = TMOD.load_params()
    assert params["model"] == TMOD.MODEL_NAME
    assert params["fit"]["intercept"] is False
    for c in params["columns"]:
        assert ANTISYM.match(c), c
    sets = [params["weights"]["all"]] + list(params["weights"]["logo"].values())
    assert sorted(params["weights"]["logo"]) == ["metrica_g1", "metrica_g2", "metrica_g3"]
    for w in sets:
        assert set(w) == set(params["columns"])
        assert all(np.isfinite(v) for v in w.values())


def _random_inputs(n=300, seed=1):
    rng = np.random.default_rng(seed)
    params = TMOD.load_params()
    team = {k: rng.uniform(-1, 1, n) for k in RP.TEAM_W}
    cols = {c.split("*")[0] for c in params["columns"] if not c.split("*")[0] in team}
    feats = pd.DataFrame({c: rng.normal(0, 2, n) for c in sorted(cols)})
    feats.iloc[::7, 0] = np.nan                           # missing shape -> no evidence
    feats["ball_seen"] = (rng.uniform(size=n) > 0.3).astype(float)
    return params, team, feats


def test_mirrored_inputs_give_the_complement():
    params, team, feats = _random_inputs()
    spec = TMOD.spec_for(params)
    q = TMOD.p_home(TMOD.design(spec["columns"], team, feats, 25.0), spec)
    mteam = {k: -v for k, v in team.items()}
    mfeats = feats.copy()
    for c in mfeats.columns:
        if c != "ball_seen":
            mfeats[c] = -mfeats[c]
    qm = TMOD.p_home(TMOD.design(spec["columns"], mteam, mfeats, 25.0), spec)
    np.testing.assert_allclose(q + qm, 1.0, atol=1e-12)


def test_no_evidence_is_even():
    params, team, feats = _random_inputs()
    team = {k: np.zeros_like(v) for k, v in team.items()}
    for c in feats.columns:
        if c != "ball_seen":
            feats[c] = np.nan
    spec = TMOD.spec_for(params)
    q = TMOD.p_home(TMOD.design(spec["columns"], team, feats, 25.0), spec)
    np.testing.assert_allclose(q, 0.5)


def test_compose_keeps_rule_v1_p_dead_and_flat_outside_halves():
    rng = np.random.default_rng(0)
    p1 = rng.dirichlet([1, 1, 1], 100).astype(np.float32)
    out = TMOD.compose(p1, rng.uniform(size=100), [[0, 80]])
    assert out.dtype == np.float32
    assert np.array_equal(out[:80, 2], p1[:80, 2])
    assert np.allclose(out[80:], 1 / 3)
    assert np.allclose(out.sum(1), 1, atol=1e-6)


def test_leave_one_game_out_weights_by_root_name():
    params = TMOD.load_params()
    assert TMOD.spec_for(params, "/x/metrica_g2_low")["weights"] is \
        params["weights"]["logo"]["metrica_g2"]
    assert TMOD.spec_for(params, "/x/artifacts/match")["weights"] is params["weights"]["all"]


def test_stage_on_a_scene_root(tmp_path):
    sc = lib.interception_scene()
    root = str(tmp_path / "scene")
    sc.write(root)
    ball, ev = sc.run()
    ball.to_parquet(os.path.join(root, "ball.parquet"), index=False)
    ev.to_parquet(os.path.join(root, "events.parquet"), index=False)
    features.build(ball, sc.players_df(), ev, sc.meta()).to_parquet(
        os.path.join(root, "features.parquet"), index=False)
    probs, probs_v1, meta, info = TMOD.score_root(root, compute_dead=False)
    v1, _m, _i, _d, _t = RS.score_root(root, compute_dead=False)
    assert np.array_equal(probs[:, 2], v1[:, 2]) and np.array_equal(probs_v1, v1)
    path = RS.write_probs(root, probs, TMOD.MODEL_NAME)
    assert os.path.basename(path) == "rule_v2.npy"
    assert not os.path.exists(os.path.join(root, "probs", "rule_v1.npy"))
    arr = np.load(path)
    assert arr.dtype == np.float32 and arr.shape == (sc.n_frames, 3)
    assert np.allclose(arr.sum(1), 1, atol=1e-5) and np.all(arr >= 0)
    assert info["weights"] == "all"


# --- T-28: later team models (rule_v3 to rule_v5), event terms, temperature -----------------

ANTISYM_NEW = re.compile(r"^(?:{}|shape_[a-z_]+_(?:diff|rel)_c\d+|lb\d?_[a-z]+_diff_c\d+|"
                         r"pc_[a-z0-9_]*_(?:diff|rel)(?:_[abc]\d+)?|"   # T-32 play context
                         r"evt_[a-z_]+)(?:\*unseen)?$".format("|".join(RP.TEAM_W)))


def test_later_models_params_are_antisymmetric_and_complete():
    for model in TMOD.available_models():
        params = TMOD.load_model(model)
        assert params["model"] == model
        if params.get("kind") == "trees":       # T-32 tree models: tests/pipeline/test_t32_model.py
            continue
        assert params["fit"]["intercept"] is False
        for c in params["columns"]:
            assert ANTISYM_NEW.match(c), (model, c)
        assert sorted(params["weights"]["logo"]) == ["metrica_g1", "metrica_g2", "metrica_g3"]
        for w in [params["weights"]["all"]] + list(params["weights"]["logo"].values()):
            assert set(w) == set(params["columns"])
        if "temperature" in params:
            t = params["temperature"]
            assert t["all"] > 0 and all(v > 0 for v in t["logo"].values())


def test_rule_v2_file_untouched_by_the_registry():
    assert TMOD.params_path("rule_v2") == TMOD.PARAMS_PATH
    assert "temperature" not in TMOD.load_params()


def test_temperature_sharpens_and_keeps_symmetry():
    spec = {"columns": ["a"], "weights": {"a": 1.0}}
    X = np.array([[-2.0], [0.0], [2.0]])
    q1 = TMOD.p_home(X, spec)
    q2 = TMOD.p_home(X, dict(spec, temperature=0.5))
    assert q2[2] > q1[2] and q2[0] < q1[0] and q2[1] == 0.5
    np.testing.assert_allclose(q2[0] + q2[2], 1.0)


def _events(rows):
    cols = ["frame", "kind", "team", "end_frame", "first_touch_frame", "detail"]
    return pd.DataFrame(rows, columns=cols)


def test_event_terms_signs_and_team_swap():
    rows = [(100, "touch", 0, -1, -1, "dv=5.0"), (110, "touch", 1, -1, -1, "dv=12.0"),   # cut-out
            (200, "flight", 0, 240, -1, ""), (241, "touch", 1, -1, -1, "dv=9.0"),        # header
            (300, "deflection", 0, -1, -1, ""), (320, "control", 0, -1, 320, "carry"),   # P7
            (400, "touch", 0, -1, -1, "dv=3.0"), (410, "touch", 1, -1, -1, "dv=3.0")]    # weak
    ev = _events(rows)
    t = TMOD.event_terms(ev, 600, 25.0)
    assert t["evt_redirect"][115] == -1.0 and t["evt_redirect"][105] == 0.0
    assert t["evt_redirect"][415] == 0.0                     # small velocity change: nothing
    assert t["evt_header"][235] == -1.0 and t["evt_header"][250] == -1.0
    assert t["evt_defl_collect"][310] == 1.0 and t["evt_defl_collect"][330] == 0.0
    sw = ev.copy()
    sw["team"] = np.where(sw["team"] >= 0, 1 - sw["team"], sw["team"])
    ts = TMOD.event_terms(sw, 600, 25.0)
    for k in t:
        np.testing.assert_array_equal(t[k], -ts[k])
