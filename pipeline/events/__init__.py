"""Events stage: ``python -m pipeline.events match|bench`` -> <root>/events.parquet.

Reads ball.parquet (ball stage), players.parquet and meta.json. Detectors, in order:

    touch, flight      pipeline/events/touch.py       B4, B5
    exit               pipeline/events/exits.py       D1, D2, D3 (+ R8 last touch team)
    carry, pass, deflection, control
                       pipeline/events/possession.py  P1, P2, P7, P4/P5 (Phase 1)

Each detector takes the shared ``Inputs`` (pipeline/events/base.py) and the rows found so far
and returns a list of row dicts; ``detect`` concatenates them into the events table. The table
layout is documented in pipeline/events/base.py. With the ball unseen there are no events.

The touch detector has named parameter sets (pipeline/events/touch.py PARAM_SETS). The default is
touch.DEFAULT_SET ("w7", the sharp-kick tuning every existing version was built with); another set
is chosen explicitly on the stage command line with ``--touch-set w15s``. The set used is reported
as ``touch_param_set``.
"""
import os

import numpy as np

from pipeline.ball.stage import read, read_meta
from pipeline.events import base, exits, possession, touch


LAST_INPUTS = {}


def detect(ball, players, meta, phase1=True, touch_set=None, inp=None, touch_kw=None):
    """Events DataFrame for one game (or scene).

    ``touch_set``: named touch parameter set; ``touch_kw``: overrides of it (bench sweeps only);
    ``inp``: prebuilt base.Inputs (sweeps).
    """
    inp = base.make_inputs(ball, players, meta) if inp is None else inp
    LAST_INPUTS["inp"] = inp
    t_rows = touch.touches(inp, touch_set=touch_set, **(touch_kw or {}))
    rows = list(t_rows)
    rows += touch.flights(inp, t_rows)
    rows += exits.exits(inp, t_rows)
    if phase1:
        rows += possession.detect(inp, t_rows, rows)
    return base.event_frame(rows)


def _graded_minutes(meta):
    return sum(b - a for a, b in meta["halves"]) / float(meta["fps"]) / 60.0


def run(root, touch_set=None):
    meta = read_meta(root)
    ball = read(root, "ball.parquet")
    players = read(root, "players.parquet")
    ev = detect(ball, players, meta, touch_set=touch_set)
    ev.to_parquet(os.path.join(root, "events.parquet"), index=False)
    minutes = max(_graded_minutes(meta), 1e-9)
    stats = {"minutes": round(minutes, 2),
             "events_per_kind": {k: int(v) for k, v in ev["kind"].value_counts().items()}}
    for kind in ("touch", "carry", "control", "pass", "deflection"):
        sub = ev[ev["kind"] == kind]
        stats["{}_per_min_by_team".format(kind)] = {
            str(t): round(float((sub["team"] == t).sum()) / minutes, 3) for t in (0, 1, -1)}
    ex = ev[ev["kind"] == "exit"]
    stats["exits_per_hour_by_line"] = {
        ln: round(float((ex["line"] == ln).sum()) / minutes * 60.0, 2)
        for ln in ("touch_y0", "touch_y1", "goal_x0", "goal_x1")}
    stats["exits_by_detail"] = {d: int((ex["detail"] == d).sum())
                                for d in ("persist", "vanish", "reappear", "extrapolated")}
    stats["exits_goal_mouth"] = int(ex["goal_mouth"].sum())
    stats["exit_last_touch_team_counts"] = {str(t): int((ex["last_touch_team"] == t).sum())
                                            for t in (0, 1, -1)}
    stats.update(touch.LAST_STATS)
    inp = LAST_INPUTS["inp"]
    stats.update(inp.align or {})
    d = inp.con.d_any[inp.seen]
    d = d[np.isfinite(d)]
    edges = [0, 0.5, 1, 1.5, 2, 3, 5, 10, 20]
    stats["ball_nearest_player_m_hist_edges"] = edges
    stats["ball_nearest_player_m_hist"] = [int(v) for v in np.histogram(d, edges + [1e9])[0]]
    stats["seen_share_player_within_1m"] = round(float((d <= 1).mean()), 4) if len(d) else None
    stats["seen_share_player_within_2m"] = round(float((d <= 2).mean()), 4) if len(d) else None
    fl = ev[ev["kind"] == "flight"]
    stats["flight_share_of_frames"] = round(
        float((fl["end_frame"] - fl["frame"] + 1).sum()) / max(int(meta["n_frames"]), 1), 4)
    return stats
