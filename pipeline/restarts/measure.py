"""Bench measurement of the kick-off detector against Metrica event data (bench games only).

    uv run python -m pipeline.restarts.measure [--game metrica_g3_match dev_g3_restarts ...]

Truth: every SET PIECE / KICK OFF event in the open Metrica sample games (bench_data/raw/metrica,
read-only), frame = event frame - 1 (the load stage's video-frame convention), team Home/Team A
= 0, Away/Team B = 1. Prints, per game and pooled: kick-offs in truth, found within
MATCH_TOL_FRAMES, false kick-offs, restart-frame error (signed and absolute), kicking-team
accuracy from the formation alone and after the kick-off law (S6) for half starts.

This never runs on the match: it refuses any root outside artifacts/bench/.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import pandas as pd

from pipeline.io.sources import ARTIFACTS, REPO
from pipeline.io.roots import require_bench_root
from pipeline.restarts import kickoff

MATCH_TOL_FRAMES = 50       # a detection within 2 s of the truth kick counts as found
RAW = os.path.join(REPO, "bench_data", "raw", "metrica")
BENCH = os.path.join(ARTIFACTS, "bench")


def game_number(game_id):
    """Metrica sample game number behind a bench root name, or None."""
    if game_id.startswith("dev_g3"):
        return 3
    for n in (1, 2, 3):
        if game_id.startswith("metrica_g{}".format(n)):
            return n
    return None


def truth_kickoffs(n):
    """DataFrame frame, team, period of every kick-off event in Metrica sample game n."""
    rows = []
    folder = os.path.join(RAW, "Sample_Game_{}".format(n))
    if n in (1, 2):
        path = os.path.join(folder, "Sample_Game_{}_RawEventsData.csv".format(n))
        with open(path) as handle:
            for r in csv.DictReader(handle):
                if r["Type"] == "SET PIECE" and r["Subtype"] == "KICK OFF":
                    rows.append({"frame": int(r["Start Frame"]) - 1,
                                 "team": 0 if r["Team"] == "Home" else 1,
                                 "period": int(r["Period"])})
    else:
        path = os.path.join(folder, "Sample_Game_3_events.json")
        with open(path) as handle:
            data = json.load(handle)["data"]
        for e in data:
            sub = e.get("subtypes")
            subs = sub if isinstance(sub, list) else [sub] if sub else []
            if e["type"]["name"] == "SET PIECE" and any(s["name"] == "KICK OFF" for s in subs):
                rows.append({"frame": int(e["start"]["frame"]) - 1,
                             "team": 0 if e["team"]["name"] == "Team A" else 1,
                             "period": int(e["period"])})
    return pd.DataFrame(rows, columns=["frame", "team", "period"])


def load_root(root):
    players = pd.read_parquet(os.path.join(root, "players.parquet"))
    with open(os.path.join(root, "meta.json")) as handle:
        meta = json.load(handle)
    ball_path = os.path.join(root, "ball_raw.parquet")
    ball = pd.read_parquet(ball_path) if os.path.isfile(ball_path) else None
    return players, meta, ball


def match_detections(det, truth, tol=MATCH_TOL_FRAMES):
    """Greedy one-to-one matching by absolute frame error. Returns list of (det_idx, truth_idx)."""
    pairs = []
    for i, d in det.iterrows():
        for j, t in truth.iterrows():
            err = abs(int(d["restart_frame"]) - int(t["frame"]))
            if err <= tol:
                pairs.append((err, i, j))
    pairs.sort()
    used_d, used_t, out = set(), set(), []
    for _err, i, j in pairs:
        if i in used_d or j in used_t:
            continue
        used_d.add(i)
        used_t.add(j)
        out.append((i, j))
    return out


def measure_game(game_id, root=None, use_ball=True):
    root = root or os.path.join(BENCH, game_id)
    require_bench_root(root)
    if not os.path.abspath(root).startswith(os.path.abspath(BENCH)):
        raise SystemExit("measure runs on bench roots only")
    n = game_number(game_id)
    truth = truth_kickoffs(n)
    players, meta, ball = load_root(root)
    det, _feats = kickoff.run(players, meta, ball if use_ball else None)
    pairs = match_detections(det, truth)
    res = {"game": game_id, "truth": len(truth), "detected": len(det), "found": len(pairs),
           "false": len(det) - len(pairs), "errors": [], "team_formation_ok": 0,
           "team_law_ok": 0, "half_start_found": 0, "half_start_truth": 0, "sources": {},
           "missed_frames": [], "team_formation_n": 0}
    half_start_truth = truth.groupby("period")["frame"].min()
    res["half_start_truth"] = len(half_start_truth)
    matched_t = set()
    for i, j in pairs:
        d, t = det.loc[i], truth.loc[j]
        matched_t.add(j)
        if not str(d["evidence"]).startswith("assumed"):
            res["errors"].append(int(d["restart_frame"]) - int(t["frame"]))
        res["sources"][d["evidence"]] = res["sources"].get(d["evidence"], 0) + 1
        if d["kick_team_llr"] != 0:
            formation_team = 0 if d["kick_team_llr"] > 0 else 1
            res["team_formation_ok"] += int(formation_team == t["team"])
            res["team_formation_n"] += 1
        law_team = 0 if d["p_home_restart"] > 0.5 else 1
        res["team_law_ok"] += int(law_team == t["team"])
        if d["start_kind"] == "half_start" and t["frame"] == half_start_truth.get(t["period"]):
            res["half_start_found"] += 1
    res["missed_frames"] = [int(truth.loc[j, "frame"]) for j in truth.index if j not in matched_t]
    res["false_frames"] = [int(det.loc[i, "restart_frame"]) for i in det.index
                           if i not in {p[0] for p in pairs}]
    res["hours"] = sum(b - a for a, b in meta["halves"]) / float(meta.get("fps", 25)) / 3600.0
    # S4 mask against truth: frames of each half before its first kick-off event
    mask = _feats["mask_pre_kickoff"].to_numpy()
    true_mask = np.zeros(len(mask), dtype=bool)
    for k, (h0, _h1) in enumerate(meta["halves"], 1):
        if k in half_start_truth.index:
            true_mask[int(h0):int(half_start_truth[k])] = True
    res["mask_tp"] = int(np.sum(mask & true_mask))
    res["mask_pred"] = int(np.sum(mask))
    res["mask_true"] = int(np.sum(true_mask))
    # half-start kicking team (S5/S6) and goal kick-off team (R13), law-adjusted
    res["hs_team_ok"] = res["hs_team_n"] = res["goal_team_ok"] = res["goal_team_n"] = 0
    for i, j in pairs:
        d, t = det.loc[i], truth.loc[j]
        ok = int((0 if d["p_home_restart"] > 0.5 else 1) == t["team"])
        key = "hs" if d["start_kind"] == "half_start" else "goal"
        res[key + "_team_ok"] += ok
        res[key + "_team_n"] += 1
    return res


def default_games():
    out = []
    for name in sorted(os.listdir(BENCH)) if os.path.isdir(BENCH) else []:
        root = os.path.join(BENCH, name)
        if game_number(name) and os.path.isfile(os.path.join(root, "players.parquet")):
            out.append(name)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", nargs="*")
    parser.add_argument("--no-ball", action="store_true", help="ignore the ball (formation only)")
    args = parser.parse_args(argv)
    games = args.game or default_games()
    pooled = {"truth": 0, "found": 0, "false": 0, "team_formation_ok": 0, "team_law_ok": 0,
              "team_formation_n": 0, "mask_tp": 0, "mask_pred": 0, "mask_true": 0,
              "hs_team_ok": 0, "hs_team_n": 0, "goal_team_ok": 0, "goal_team_n": 0,
              "half_start_found": 0, "half_start_truth": 0, "hours": 0.0}
    errors = []
    for g in games:
        r = measure_game(g, use_ball=not args.no_ball)
        e = np.array(r["errors"]) if r["errors"] else np.array([0])
        print("{:<20} truth {:>2} found {:>2} false {:>2} half-starts {}/{} team(formation) {}/{} "
              "team(law) {}/{} err median {:+.0f} |err| max {:.0f} sources {} missed {} false {}"
              .format(g, r["truth"], r["found"], r["false"], r["half_start_found"],
                      r["half_start_truth"], r["team_formation_ok"], r["team_formation_n"], r["team_law_ok"],
                      r["found"], float(np.median(e)), float(np.max(np.abs(e))), r["sources"],
                      r["missed_frames"], r["false_frames"]))
        for k in pooled:
            pooled[k] += r[k]
        errors += r["errors"]
    e = np.array(errors) if errors else np.array([0])
    print("POOLED games {} truth {} found {} recall {:.3f} false {} false/hour {:.3f} "
          "half-starts {}/{} team(formation) acc {:.3f} team(law) acc {:.3f}".format(
              len(games), pooled["truth"], pooled["found"],
              pooled["found"] / max(1, pooled["truth"]), pooled["false"],
              pooled["false"] / max(1e-9, pooled["hours"]), pooled["half_start_found"],
              pooled["half_start_truth"], pooled["team_formation_ok"] / max(1, pooled["team_formation_n"]),
              pooled["team_law_ok"] / max(1, pooled["found"])))
    print("S4 mask precision {:.4f} coverage {:.4f}; half-start team (law) {}/{}; goal kick-off "
          "team {}/{}".format(pooled["mask_tp"] / max(1, pooled["mask_pred"]),
                              pooled["mask_tp"] / max(1, pooled["mask_true"]),
                              pooled["hs_team_ok"], pooled["hs_team_n"], pooled["goal_team_ok"],
                              pooled["goal_team_n"]))
    print("restart-frame error (detected - truth, frames): median {:+.1f} mean {:+.1f} "
          "|err| median {:.1f} p90 {:.1f} max {:.0f} within 25: {}/{}".format(
              float(np.median(e)), float(np.mean(e)), float(np.median(np.abs(e))),
              float(np.percentile(np.abs(e), 90)), float(np.max(np.abs(e))),
              int(np.sum(np.abs(e) <= 25)), len(errors)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
