"""Rule scorer (rule_v1) terms on synthetic scenes (pipeline/scenes), one test per term plus the
combination, the flat prior and the stage file on a scene root."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from pipeline import features
from pipeline.models import rule_params as RP
from pipeline.models import rule_scorer as RS
from pipeline.models import terms as TM
from pipeline.scenes import Scene
from pipeline.scenes import library as lib


def _terms(sc, dead=None, players=None):
    ball, ev = sc.run()
    players = sc.players_df() if players is None else players
    feat = features.build(ball, players, ev, sc.meta())
    d = TM.dead_terms(ball, players, ev, feat, sc.meta(), dead)
    t, _ = TM.team_terms(ball, players, ev, feat, sc.meta())
    return d, t, ball, ev


def _bystanders(sc, home_x=(30, 35, 40), away_x=(65, 70, 75), speed=0.0, n=None):
    """Extra players walking (or still) along y so collective speed is controlled."""
    n = n or sc.n_frames
    for x in home_x:
        sc.player(0, [(0, x, 10), (n - 1, x, 10 + speed * (n - 1) / 25)])
    for x in away_x:
        sc.player(1, [(0, x, 58), (n - 1, x, 58 - speed * (n - 1) / 25)])


# --- every term is bounded and zero outside the halves --------------------------------------

def test_terms_bounded_and_named():
    d, t, _, _ = _terms(lib.pass_scene())
    assert set(d) == set(RP.DEAD_DEFAULTS) and set(t) == set(RP.TEAM_DEFAULTS)
    for k, v in {**d, **t}.items():
        assert np.all(np.isfinite(v)) and np.all(np.abs(v) <= 1 + 1e-9), k


# --- dead terms ------------------------------------------------------------------------------

def test_miss_and_gap_length_d12_d13():
    short = lib.pass_scene()
    short.hide_ball(80, 104)                       # 1 s gap while the receiver carries
    long_ = Scene(900)
    p = long_.player(0, [(0, 40, 34), (899, 60, 34)])
    long_.player(1, [(0, 50, 20)])
    long_.ball_with(p, 0, 899)
    long_.hide_ball(100, 849)                      # 30 s gap
    ds, _, _, _ = _terms(short)
    dl, _, _, _ = _terms(long_)
    assert ds["miss"][90] == 1 and dl["miss"][400] == 1
    assert ds["miss"][60] == 0
    assert ds["miss_len"][90] < 0 < dl["miss_len"][400]


def test_miss_line_d3_and_flight_gap_trap():
    d, _, _, _ = _terms(lib.vanish_exit_scene())
    assert d["miss_line"][100] > 0.5
    f, _, _, _ = _terms(lib.flight_gap_scene(gap=20))
    assert f["miss_line"].max() == 0


def test_miss_crowd_p25():
    sc = Scene(300)
    c = sc.player(0, [(0, 50, 34)])
    for dx, dy in ((1.0, 0.5), (-1.0, 1.0), (0.5, -1.5)):
        sc.player(1, [(0, 50 + dx, 34 + dy)])
    sc.player(0, [(0, 51.5, 33)])
    sc.ball_with(c, 0, 299)
    sc.hide_ball(100, 250)
    d, _, _, _ = _terms(sc)
    assert d["miss_crowd"][150] >= 0.75
    assert d["miss_crowd"][50] == 0


def test_out_and_exit_d1_d2():
    d, _, ball, ev = _terms(lib.exit_scene("touch_y0"))
    e = int(ev.loc[ev["kind"] == "exit", "frame"].iloc[0])
    assert d["exit"][e + 20] > 0.5 and d["exit"][e - 5] == 0
    assert d["out"][e + 5] > 0.5 and d["out"][e - 20] == 0


def test_projection_trap_no_exit_a15():
    d, _, _, ev = _terms(lib.projection_trap_scene())
    assert d["exit"].max() == 0
    assert (ev["kind"] == "exit").sum() == 0


def test_pre_kickoff_mask_s4_and_formation_r6():
    sc = lib.carry_scene()
    n = sc.n_frames
    dead = pd.DataFrame({"frame": np.arange(n), "half": 1,
                         "form_kick_off": np.r_[np.ones(30), np.zeros(n - 30)],
                         "mask_pre_kickoff": np.r_[np.ones(40, bool), np.zeros(n - 40, bool)],
                         "p_restart_kick_off": 0.0})
    d, t, _, _ = _terms(sc, dead=dead)
    assert d["pre_kickoff"][10] == 1 and d["pre_kickoff"][60] == 0
    assert d["form_kickoff"][10] == 1 and d["form_kickoff"][60] == 0
    n_ = sc.n_frames
    probs = RS.combine(d, t, n_, sc.halves)
    assert probs[10, 2] > 0.9


def test_slow_d5_and_dip():
    # Everyone runs about 4 m/s, stands still for 14 s (a stoppage), then runs again.
    n = 1000
    sc = Scene(n)
    for k, (team, y) in enumerate([(0, 10), (0, 25), (0, 40), (1, 15), (1, 30), (1, 55)]):
        x0 = 20 + 10 * k
        sc.player(team, [(0, x0, y), (200, x0, y + 32), (400, x0, y), (500, x0, y),
                         (750, x0, y), (950, x0, y + 32), (999, x0, y + 24)])
    sc.ball([(0, 52, 34), (999, 52, 34)])
    d, _, _, _ = _terms(sc)
    assert d["slow"][600] > 0.5            # still for 10 s
    assert d["slow"][100] < 0              # running
    assert d["slow_dip"][450] > d["slow_dip"][100]


def test_still_ball_and_moving_ball():
    sc = Scene(200)
    sc.player(0, [(0, 30, 30)])
    sc.player(1, [(0, 70, 30)])
    sc.ball([(0, 40, 34), (99, 40, 34)])           # stationary ball
    sc.ball([(100, 40, 34), (199, 90, 34)])        # then rolling at ~12.6 m/s
    d, _, _, _ = _terms(sc)
    assert d["still"][50] > 0.9 and d["moving"][50] == 0
    assert d["moving"][150] > 0.9 and d["still"][150] == 0


def test_players_low_d10():
    sc = lib.carry_scene()
    d, _, _, _ = _terms(sc)                         # 6 players tracked: clearly few
    assert d["players_low"][50] == 1.0


# --- team terms ------------------------------------------------------------------------------

def test_carry_and_control_p1():
    _, t, _, ev = _terms(lib.carry_scene())
    assert t["carry"][60] == 1 and t["ctrl_last"][100] > 0
    assert t["near"][60] > 0


def test_interception_turns_control_p4():
    _, t, _, ev = _terms(lib.interception_scene())
    f = int(ev.loc[(ev["kind"] == "control") & (ev["team"] == 1), "first_touch_frame"].iloc[0])
    assert t["ctrl_last"][f + 30] < 0 < t["ctrl_last"][f - 10]
    assert t["ctrl_next"][f - 5] < 0               # the look-ahead twin sees it coming


def test_pass_span_p2():
    _, t, _, ev = _terms(lib.pass_scene())
    p = ev[(ev["kind"] == "pass") & (ev["team"] == 0)].iloc[0]
    mid = (int(p["frame"]) + int(p["end_frame"])) // 2
    assert t["pass"][mid] == 1


def test_dribble_push_nearest_opponent_is_weak_p13():
    _, t, _, _ = _terms(lib.dribble_push_scene())
    probs = RS.combine({k: np.zeros(125) for k in RP.DEAD_DEFAULTS}, t, 125, [[0, 125]])
    ph = probs[:, 0] / (probs[:, 0] + probs[:, 1])
    assert ph[30:60].min() > 0.5                   # never handed to the defender


def test_territory_and_spread_t1_t2():
    sc = Scene(100)
    # Home attacks +x; everyone in the away half; home wide, away compact.
    for x, y in ((60, 5), (62, 63), (75, 20), (78, 50), (85, 34)):
        sc.player(0, [(0, x, y)])
    for x, y in ((88, 30), (89, 36), (90, 33), (91, 38), (92, 31)):
        sc.player(1, [(0, x, y)])
    _, t, _, _ = _terms(sc)
    assert t["territory"][50] > 0.3
    assert t["spread"][50] > 0.5


def test_attack_direction_fallback_without_meta():
    sc = Scene(100, halves=[[0, 50], [50, 100]])
    sc.player(0, [(0, 20, 34)])                    # home deep at x = 20 ...
    sc.player(1, [(0, 80, 34)])
    meta = dict(sc.meta())
    meta.pop("attack_direction")
    d = TM.attack_home(meta, sc.players_df(), 100)
    assert d[10] == 1 and d[60] == 1


# --- combination ---------------------------------------------------------------------------

def test_no_evidence_is_flat():
    n = 50
    z = {k: np.zeros(n) for k in RP.DEAD_DEFAULTS}
    zt = {k: np.zeros(n) for k in RP.TEAM_DEFAULTS}
    p = RS.combine(z, zt, n, [[0, n]], dead_bias=RP.DEAD_BIAS_DEFAULT)
    assert np.allclose(p, 1 / 3, atol=1e-3)


def test_outside_halves_flat_and_rows_sum_to_one():
    d, t, _, _ = _terms(lib.pass_scene())
    p = RS.combine(d, t, 150, [[0, 100]])
    assert np.allclose(p.sum(1), 1, atol=1e-5)
    assert np.allclose(p[100:], 1 / 3)
    assert not np.allclose(p[:100], 1 / 3)


def test_stage_writes_contract_file(tmp_path):
    sc = lib.interception_scene()
    root = str(tmp_path / "scene")
    sc.write(root)
    ball, ev = sc.run()
    ball.to_parquet(os.path.join(root, "ball.parquet"), index=False)
    ev.to_parquet(os.path.join(root, "events.parquet"), index=False)
    features.build(ball, sc.players_df(), ev, sc.meta()).to_parquet(
        os.path.join(root, "features.parquet"), index=False)
    probs, meta, info, _, _ = RS.score_root(root, compute_dead=False)
    path = RS.write_probs(root, probs)
    arr = np.load(path)
    assert arr.dtype == np.float32 and arr.shape == (sc.n_frames, 3)
    assert np.allclose(arr.sum(1), 1, atol=1e-5) and np.all(arr >= 0)
    assert info["dead_features"] == "none"
