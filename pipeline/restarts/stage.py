"""One call for the whole restarts stage on one root: kick-offs (kickoff.py) plus stoppages
beyond kick-offs (stoppages.py), merged into restarts.parquet rows and features_dead columns."""
import json
import os

import numpy as np
import pandas as pd

from pipeline.restarts import kickoff, stoppages


def load_inputs(root):
    """Contract files of a root. ball.parquet and events.parquet are optional (kick-offs only
    without them)."""
    players = pd.read_parquet(os.path.join(root, "players.parquet"))
    with open(os.path.join(root, "meta.json")) as handle:
        meta = json.load(handle)
    def opt(name, cols=None):
        p = os.path.join(root, name)
        return pd.read_parquet(p, columns=cols) if os.path.isfile(p) else None
    return {"players": players, "meta": meta,
            "ball_raw": opt("ball_raw.parquet", ["frame", "x_m", "y_m"]),
            "ball": opt("ball.parquet"), "events": opt("events.parquet")}


def run(inputs, anchors=None, funnel=None):
    """Returns (restarts DataFrame, features_dead DataFrame)."""
    players, meta = inputs["players"], inputs["meta"]
    halves = [tuple(h) for h in meta["halves"]]
    n = int(meta["n_frames"])
    ko_funnel = {}
    kos, feats = kickoff.run(players, meta, inputs["ball_raw"], ko_funnel, anchors=anchors)
    if funnel is not None:
        funnel["kickoff"] = ko_funnel
    if inputs.get("ball") is None:
        return kos, feats
    m = stoppages.frame_measures(players, meta, inputs["ball"])
    st_funnel = {}
    stops = stoppages.detect_stoppages(m, inputs.get("events"), halves, funnel=st_funnel)
    if funnel is not None:
        funnel["stoppage_sources"] = st_funnel
    rows = stoppages.merge_kickoffs(stops, kos, halves)
    for name, col in stoppages.dead_features(m, rows, n).items():
        feats[name] = col
    # S4: the pre-kick-off mask has priority; nothing else to do here
    return rows, feats
