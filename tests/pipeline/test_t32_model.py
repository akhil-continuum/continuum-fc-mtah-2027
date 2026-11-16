"""T-32 per-player model (pipeline/models/player_model.py): the two views are exact mirrors, the
symmetrised q has no lean, the exported forest predicts what sklearn predicts, gate helpers."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np                                  # noqa: E402
import pandas as pd                                 # noqa: E402

from pipeline.models import player_model as PM     # noqa: E402


def _arrays(n=300, seed=0):
    rng = np.random.default_rng(seed)
    cols = {c: rng.normal(0, 3, n) for c in PM.needed_columns()}
    for c in cols:
        cols[c][rng.random(n) < 0.1] = np.nan
    df = pd.DataFrame(cols)
    return PM.arrays(df, rng.normal(0, 2, n), rng.random(n))


def _mirror(ar):
    out = {}
    for c, v in ar.items():
        if c.startswith("pnb_h") or c.startswith("pnb_a"):
            other = "pnb_" + ("a" if c[4] == "h" else "h") + c[5:]
            out[other] = v.copy()
        elif c in PM.BALL_ANTI or c in PM.ANTI or c == "z":
            out[c] = -v
        else:
            out[c] = v.copy()
    return out


def test_views_are_exact_mirrors():
    ar = _arrays()
    half = PM.half_ids([[0, 150], [150, 300]], 300)
    idx = np.arange(300)
    xh, xa = PM.views(ar, half, idx)
    mh, ma = PM.views(_mirror(ar), half, idx)
    np.testing.assert_array_equal(np.isnan(xh), np.isnan(ma))
    np.testing.assert_array_equal(np.nan_to_num(xh, nan=7.7), np.nan_to_num(ma, nan=7.7))
    np.testing.assert_array_equal(np.nan_to_num(xa, nan=7.7), np.nan_to_num(mh, nan=7.7))
    assert xh.shape[1] == len(PM.feature_names())
    # offsets never cross the half boundary
    k = PM.feature_names().index("own1_d@50")
    assert np.isnan(xh[120, k]) and np.isnan(xh[290, k])


def _fit(X, y):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=20, max_leaf_nodes=7, random_state=0,
                                          early_stopping=False).fit(X, y)


def test_export_matches_sklearn_and_q_is_symmetric():
    ar = _arrays(n=2000, seed=1)
    half = PM.half_ids([[0, 2000]], 2000)
    idx = np.arange(2000)
    xh, xa = PM.views(ar, half, idx)
    y = (np.nan_to_num(ar["z"]) + np.nan_to_num(ar["pnb_h1_d"]) > 0).astype(int)
    m = _fit(np.vstack([xh, xa]), np.concatenate([y, 1 - y]))
    forest = PM.export(m)
    np.testing.assert_allclose(PM.sigmoid(PM.forest_raw(forest, xh)), m.predict_proba(xh)[:, 1],
                               atol=1e-9)
    q = PM.q_home(forest, ar, half)
    qm = PM.q_home(forest, _mirror(ar), half)
    np.testing.assert_allclose(q, 1.0 - qm, atol=1e-12)


def test_gate_helpers():
    f = np.zeros(20, bool)
    f[10] = True
    a = PM.centred_any(f, 2)
    assert a[8:13].all() and not a[7] and not a[13]
    g = PM.gate_weight(np.ones(100, bool), 25.0)
    assert np.allclose(g, 1.0)


def test_stored_tree_params_are_complete():
    from pipeline.models import team_model as TMOD
    for model in TMOD.available_models():
        params = TMOD.load_model(model)
        if params.get("kind") != "trees":
            continue
        assert params["features"] == PM.feature_names(tuple(params.get("quants", PM.QUANTS)))
        assert params["base"] in TMOD.MODELS
        assert sorted(params["forests"]["logo"]) == ["metrica_g1", "metrica_g2", "metrica_g3"]
        nf = len(params["features"])
        for forest in [params["forests"]["all"]] + list(params["forests"]["logo"].values()):
            assert len(forest["trees"]) == params["fit"]["params"]["max_iter"]
            for tree in forest["trees"]:
                inner = [f for f, leaf in zip(tree["f"], tree["leaf"]) if not leaf]
                assert all(0 <= f < nf for f in inner)
        if params.get("gate"):
            assert params["gate"] in ("G1", "G2", "G3", "G4")
