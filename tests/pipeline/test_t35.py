"""T-35 (P-17c): models with their own P(D) (rule_w1, rule_w2), their params files, and the new
versions (v2s15, v6s15 ablations; v2w, v6w refits). Existing models keep rule_v1's P(D)."""
import os

import numpy as np

from pipeline import features
from pipeline.decode import decoders
from pipeline.models import rule_params as RP
from pipeline.models import rule_scorer as RS
from pipeline.models import team_model as TMOD
from pipeline.scenes import library as lib

GAMES = ["metrica_g1", "metrica_g2", "metrica_g3"]


def _dead_block(scale):
    w = {k: (v * scale if k not in RP.FROZEN else v) for k, v in RP.DEAD_W.items()}
    return {"bias": RP.DEAD_BIAS_DEFAULT,
            "weights": {"all": w, "logo": {g: dict(w) for g in GAMES}}}


def test_models_without_a_dead_block_keep_rule_v1():
    for model in ("rule_v2",):
        if os.path.isfile(TMOD.params_path(model)):
            assert TMOD.dead_spec_for(TMOD.load_model(model), "/x/metrica_g1_match") is None


def test_dead_spec_leave_one_game_out_by_root_name():
    block = _dead_block(1.0)
    block["weights"]["logo"]["metrica_g2"] = dict(block["weights"]["all"], miss=9.0)
    params = {"dead": block}
    assert TMOD.dead_spec_for(params, "/x/metrica_g2_low")[1]["miss"] == 9.0
    assert TMOD.dead_spec_for(params, "/x/metrica_g1_low")[1] is block["weights"]["logo"][
        "metrica_g1"]
    assert TMOD.dead_spec_for(params, "/x/artifacts/match")[1] is block["weights"]["all"]


def _scene_root(tmp_path):
    sc = lib.interception_scene()
    root = str(tmp_path / "scene")
    sc.write(root)
    ball, ev = sc.run()
    ball.to_parquet(os.path.join(root, "ball.parquet"), index=False)
    ev.to_parquet(os.path.join(root, "events.parquet"), index=False)
    features.build(ball, sc.players_df(), ev, sc.meta()).to_parquet(
        os.path.join(root, "features.parquet"), index=False)
    return root


def test_own_dead_block_sets_p_dead_and_leaves_others(tmp_path):
    root = _scene_root(tmp_path)
    p2 = TMOD.load_params()
    own = dict(p2, model="rule_x", dead=_dead_block(0.5))
    out, probs_v1, meta, _info = TMOD.score_root_models(
        root, {"rule_v2": p2, "rule_x": own}, compute_dead=False)
    assert np.array_equal(out["rule_v2"][:, 2], probs_v1[:, 2])
    data = RS.read_root(root, compute_dead=False)
    dead, team, _ = RS.compute_terms(data)
    n = int(meta["n_frames"])
    want = RS.combine(dead, team, n, meta["halves"], dead_w=own["dead"]["weights"]["all"],
                      dead_bias=RP.DEAD_BIAS_DEFAULT)
    assert np.array_equal(out["rule_x"][:, 2], want[:, 2])
    # the team part is the same model: P(H | live) unchanged
    q2 = out["rule_v2"][:, 0] / (out["rule_v2"][:, 0] + out["rule_v2"][:, 1])
    qx = out["rule_x"][:, 0] / (out["rule_x"][:, 0] + out["rule_x"][:, 1])
    np.testing.assert_allclose(q2, qx, atol=1e-5)
    assert np.allclose(out["rule_x"].sum(1), 1, atol=1e-5)


def test_refit_params_files_well_formed():
    files = [m for m in ("rule_w1", "rule_w2") if os.path.isfile(TMOD.params_path(m))]
    blocks = []
    for model in files:
        p = TMOD.load_model(model)
        assert p["model"] == model and p["fit"]["touch_set"] == "w15s"
        assert p["fit"]["intercept"] is False
        d = p["dead"]
        assert d["bias"] == RP.DEAD_BIAS_DEFAULT
        assert sorted(d["weights"]["logo"]) == GAMES
        for w in [d["weights"]["all"]] + list(d["weights"]["logo"].values()):
            assert set(w) == set(RP.DEAD_DEFAULTS)
            for k, v in w.items():
                if k in RP.FROZEN:
                    assert v == RP.DEAD_DEFAULTS[k]
                assert RP.SIGNS[k] * v >= 0
        blocks.append(d)
    if len(blocks) == 2:
        assert blocks[0] == blocks[1]                      # rule_w2's P(D) is rule_w1's
    if "rule_w1" in files:
        p = TMOD.load_model("rule_w1")
        assert p["columns"] == list(RP.TEAM_DEFAULTS)
        for w in [p["weights"]["all"]] + list(p["weights"]["logo"].values()):
            assert all(RP.SIGNS[k] * v >= 0 for k, v in w.items())


def test_new_versions_differ_only_where_declared():
    v2s, v6s = decoders.VERSIONS["v2s"], decoders.VERSIONS["v6s"]
    for name, model in (("v2s15", "rule_v1"), ("v6s15", "rule_v2"), ("v2w", "rule_w1"),
                        ("v6w", "rule_w2")):
        spec = decoders.VERSIONS[name]
        assert spec["touch_set"] == "w15s"
        assert spec.get("probs", "rule_v1") == model
        assert spec["dials"] == v2s["dials"] and spec["decoder"] is v2s["decoder"]
        assert spec["compute"] == v2s["compute"]
    assert "touch_set" not in v2s and "touch_set" not in v6s
    assert {k: v for k, v in decoders.VERSIONS["v6s15"].items() if k != "touch_set"} == v6s
    assert {k: v for k, v in decoders.VERSIONS["v2s15"].items() if k != "touch_set"} == v2s
