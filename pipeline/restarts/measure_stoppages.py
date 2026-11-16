"""Bench measurement of stoppage candidates against event-data stoppages (bench games only).

    uv run python -m pipeline.restarts.measure_stoppages --game dev_g3_clean dev_g3_noisy

Truth: artifacts/bench/metrica_g<n>/stoppages.csv (bench-builder; D from `start` to `restart`).
Half-start kick-offs are left out (the kick-off law handles them). A candidate matches a truth
stoppage when their spans overlap on at least MIN_OVERLAP of the shorter one (greedy, one to
one, largest overlap first). Prints precision by source, recall by truth type, start and
restart timing errors, restart-team accuracy by type, restart-type agreement, false per hour.
Refuses roots outside artifacts/bench/.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

from pipeline.io.sources import ARTIFACTS
from pipeline.io.roots import require_bench_root
from pipeline.restarts import stage
from pipeline.restarts.measure import game_number

BENCH = os.path.join(ARTIFACTS, "bench")
MIN_OVERLAP = 0.3
TYPE_MAP = {"THROW IN": "throw_in", "FREE KICK": "free_kick", "GOAL KICK": "goal_kick",
            "CORNER KICK": "corner", "KICK OFF": "kick_off", "UNLOGGED": "unlogged",
            "PENALTY": "penalty"}


def truth_for(game_id):
    n = game_number(game_id)
    t = pd.read_csv(os.path.join(BENCH, "metrica_g{}".format(n), "stoppages.csv"))
    t = t[t["cause"] != "half_start"].copy()
    t["type"] = t["restart_type"].map(TYPE_MAP).fillna("other")
    t["team"] = t["restart_team"].map({"H": 0, "A": 1})
    return t.reset_index(drop=True)


def match(cands, truth):
    pairs = []
    for i, c in cands.iterrows():
        s0, e0 = int(c["start_frame"]), int(c["restart_frame"])
        if s0 < 0:
            continue
        for j, t in truth.iterrows():
            s1, e1 = int(t["start"]), int(t["restart"])
            ov = min(e0, e1) - max(s0, s1)
            if ov <= 0:
                continue
            short = max(1, min(e0 - s0, e1 - s1))
            if ov / short >= MIN_OVERLAP:
                pairs.append((ov, i, j))
    pairs.sort(reverse=True)
    ud, ut, out = set(), set(), []
    for _ov, i, j in pairs:
        if i in ud or j in ut:
            continue
        ud.add(i)
        ut.add(j)
        out.append((i, j))
    return out


def measure_game(game_id):
    root = os.path.join(BENCH, game_id)
    require_bench_root(root)
    if not os.path.abspath(root).startswith(os.path.abspath(BENCH)):
        raise SystemExit("bench roots only")
    inputs = stage.load_inputs(root)
    if inputs["events"] is None and inputs["ball"] is not None:
        # bench-builder roots carry no events.parquet: run the signal-engineer's events stage in
        # memory (nothing is written into their root)
        from pipeline.events import detect
        inputs["events"] = detect(inputs["ball"], inputs["players"], inputs["meta"])
    rows, _ = stage.run(inputs)
    cands = rows[rows["start_kind"] != "half_start"].reset_index(drop=True)
    truth = truth_for(game_id)
    pairs = match(cands, truth)
    hours = sum(b - a for a, b in inputs["meta"]["halves"]) / 25.0 / 3600.0
    recs = []
    for i, j in pairs:
        c, t = cands.loc[i], truth.loc[j]
        recs.append({"source": c["start_kind"], "type_true": t["type"], "type_pred": c["restart_type"],
                     "start_err": int(c["start_frame"]) - int(t["start"]),
                     "end_err": int(c["restart_frame"]) - int(t["restart"]),
                     "team_true": t["team"], "p_home": float(c["p_home_restart"])})
    matched_c = {i for i, _ in pairs}
    matched_t = {j for _, j in pairs}
    return {"game": game_id, "cands": cands, "truth": truth, "recs": pd.DataFrame(recs),
            "matched_c": matched_c, "matched_t": matched_t, "hours": hours}


def report(results):
    allc = pd.concat([r["cands"].assign(_m=[i in r["matched_c"] for i in r["cands"].index],
                                        _g=r["game"]) for r in results], ignore_index=True)
    allt = pd.concat([r["truth"].assign(_m=[j in r["matched_t"] for j in r["truth"].index])
                      for r in results], ignore_index=True)
    recs = pd.concat([r["recs"] for r in results], ignore_index=True)
    hours = sum(r["hours"] for r in results)
    print("roots:", [r["game"] for r in results], "hours %.2f" % hours)
    print("candidates %d, truth %d, matched %d; precision %.3f recall %.3f; false/hour %.1f" % (
        len(allc), len(allt), int(allc["_m"].sum()), allc["_m"].mean(), allt["_m"].mean(),
        (~allc["_m"]).sum() / hours))
    print("precision by source:", {k: "%d/%d=%.2f" % (g["_m"].sum(), len(g), g["_m"].mean())
                                   for k, g in allc.groupby("start_kind")})
    print("recall by truth type:", {k: "%d/%d=%.2f" % (g["_m"].sum(), len(g), g["_m"].mean())
                                    for k, g in allt.groupby("type")})
    if len(recs):
        for col in ("start_err", "end_err"):
            e = recs[col].to_numpy()
            print("%s (cand - truth, frames): median %+.0f |err| median %.0f p90 %.0f" % (
                col, np.median(e), np.median(np.abs(e)), np.percentile(np.abs(e), 90)))
        print("end_err by type |median|/p90:", {k: "%.0f/%.0f" % (np.median(np.abs(g.end_err)), np.percentile(np.abs(g.end_err), 90))
                                         for k, g in recs.groupby("type_true")})
        dec = recs[recs["p_home"] != 0.5]
        ok = ((dec["p_home"] > 0.5).astype(int) == (1 - dec["team_true"])).astype(int)
        recs["team_ok"] = np.where(recs["p_home"] == 0.5, np.nan,
                                   ((recs["p_home"] > 0.5).astype(int) == (1 - recs["team_true"])).astype(float))
        print("restart team: decided %d/%d, accuracy on decided %.3f" % (len(dec), len(recs), ok.mean() if len(ok) else float("nan")))
        print("team accuracy by true type (decided):", {k: "%d/%d" % (np.nansum(g.team_ok), g.team_ok.notna().sum())
                                                         for k, g in recs.groupby("type_true")})
        print("type agreement by true type:", {k: "%d/%d" % ((g.type_pred == k).sum(), len(g))
                                               for k, g in recs.groupby("type_true")})
        print("predicted type counts (matched):", recs["type_pred"].value_counts().to_dict())
    print("false candidates by source:", allc[~allc["_m"]]["start_kind"].value_counts().to_dict())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", nargs="+", required=True)
    args = parser.parse_args(argv)
    report([measure_game(g) for g in args.game])
    return 0


if __name__ == "__main__":
    sys.exit(main())
