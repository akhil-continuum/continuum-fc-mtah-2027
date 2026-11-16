"""Is the ball position empty exactly while the ball is out of play? (bench-truth skill, step 1)

    uv run python -m bench.deadball_check [--games metrica_g1 ...]

Compares, per game, the frames where the corrected ball channel is NaN with the stoppages derived
from the event data only (bench.events.event_stoppages: from the end of the last live action to
the restart event). Prints rates in both directions, edge offsets, and the length profile of
missing-ball runs during live play. Writes artifacts/bench/<game>/deadball_check.json.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from bench import metrica_io
from bench.events import event_stoppages, half_end_frames, runs

ART = os.path.join(metrica_io.ROOT, "artifacts", "bench")
BINS_S = [0, 0.2, 0.5, 1, 2, 4, 8, 15, 30, 60, 1e9]


def check(game):
    halves = game.halves()
    all_stops = event_stoppages(game.events, halves)
    all_stops = all_stops[all_stops["restart"] > all_stops["start"]].reset_index(drop=True)
    ends = half_end_frames(game.events, halves)
    n = game.n_frames
    nan = np.isnan(game.ball[:, 0])

    # Stoppages whose restart is unlogged are found partly by guessing where play resumed; keep
    # them out of the headline rates (which must be independent of the ball channel) and report
    # them on their own. Their frames are excluded from both the dead and the live side.
    unlogged = all_stops[all_stops["restart_type"] == "UNLOGGED"]
    stops = all_stops[all_stops["restart_type"] != "UNLOGGED"].reset_index(drop=True)
    excluded = np.zeros(n, bool)
    for s in unlogged.itertuples():
        excluded[s.start:s.restart] = True
    u_share = np.array([nan[s.start:s.restart].mean() for s in unlogged.itertuples()])

    graded = np.zeros(n, bool)       # half start .. last live frame of the half
    dead_ev = np.zeros(n, bool)      # event-derived stoppages inside graded
    for (h0, h1), end in zip(halves, ends):
        graded[h0:end] = True
    graded &= ~excluded
    for s in stops.itertuples():
        dead_ev[s.start:s.restart] = True
    dead_ev &= graded
    live_ev = graded & ~dead_ev

    tail = np.zeros(n, bool)         # after the last event of a half (not graded here)
    for (h0, h1), end in zip(halves, ends):
        tail[end:h1] = True

    res = {"game": game.game_id, "n_frames": n, "n_graded": int(graded.sum()),
           "n_stoppages": int(len(stops)),
           "stoppage_causes": stops["cause"].value_counts().to_dict(),
           "restart_types": stops["restart_type"].value_counts().to_dict(),
           "n_unlogged_restart_stoppages": int(len(unlogged)),
           "unlogged_missing_share_hist": {
               "edges": [0, 0.25, 0.5, 0.75, 0.95, 1.0001],
               "counts": np.histogram(u_share, [0, 0.25, 0.5, 0.75, 0.95, 1.0001])[0].tolist()},
           "unlogged_frames_excluded": int(excluded.sum())}
    res["ball_missing_all_frames"] = float(nan.mean())
    res["ball_missing_graded"] = float(nan[graded].mean())
    res["P_missing_given_event_dead"] = float(nan[dead_ev].mean())
    res["P_missing_given_event_live"] = float(nan[live_ev].mean())
    res["P_event_dead_given_missing"] = float(dead_ev[nan & graded].mean())
    res["P_event_dead_given_present"] = float(dead_ev[~nan & graded].mean())
    res["P_missing_in_half_tails"] = float(nan[tail].mean()) if tail.any() else None
    res["share_dead_event"] = float(dead_ev[graded].mean())

    # Per stoppage: share missing; offsets of the overlapping missing run's edges.
    shares, start_off, end_off, n_zero, n_full = [], [], [], 0, 0
    for s in stops.itertuples():
        if s.restart <= s.start:
            continue
        seg = nan[s.start:s.restart]
        shares.append(seg.mean())
        n_zero += int(seg.mean() == 0)
        n_full += int(seg.all())
        if seg.any():
            first = s.start + int(np.argmax(seg))
            # extend the run containing the last missing frame inside the stoppage
            last_in = s.start + len(seg) - 1 - int(np.argmax(seg[::-1]))
            run_start = first
            while run_start > 0 and nan[run_start - 1]:
                run_start -= 1
            run_stop = last_in + 1
            while run_stop < n and nan[run_stop]:
                run_stop += 1
            start_off.append(run_start - s.start)
            end_off.append(run_stop - s.restart)
    shares = np.array(shares)
    so, eo = np.array(start_off), np.array(end_off)
    res["stoppage_missing_share"] = {"mean": float(shares.mean()), "median": float(np.median(shares)),
                                     "n_zero_missing": n_zero, "n_fully_missing": n_full,
                                     "n": int(len(shares))}

    def off_stats(a):
        return {"exact": int((a == 0).sum()), "within_1": int((np.abs(a) <= 1).sum()),
                "within_5": int((np.abs(a) <= 5).sum()), "within_25": int((np.abs(a) <= 25).sum()),
                "median": float(np.median(a)) if len(a) else None,
                "min": int(a.min()) if len(a) else None, "max": int(a.max()) if len(a) else None,
                "n": int(len(a))}
    res["missing_run_start_minus_stoppage_start"] = off_stats(so)
    res["missing_run_stop_minus_restart_frame"] = off_stats(eo)

    # Missing-ball runs that touch no event stoppage: live-play gaps in the corrected tracking.
    live_runs, dead_runs = [], []
    for a, b in runs(nan & graded):
        (dead_runs if dead_ev[a:b].any() else live_runs).append((a, b))
    lr = np.array([(b - a) / 25.0 for a, b in live_runs])
    res["missing_runs_touching_stoppage"] = len(dead_runs)
    res["missing_runs_live_only"] = len(live_runs)
    res["live_missing_run_length_hist_s"] = {
        "edges": BINS_S[:-1] + ["inf"],
        "counts": np.histogram(lr, BINS_S)[0].tolist() if len(lr) else []}
    res["live_missing_frames_in_runs_over_1s"] = int(sum(b - a for a, b in live_runs if b - a > 25))

    # Candidate rule "D = missing run that touches an event stoppage": frame agreement.
    d_rule = np.zeros(n, bool)
    for a, b in dead_runs:
        d_rule[a:b] = True
    d_rule &= graded
    inter = (d_rule & dead_ev).sum()
    res["rule_missing_runs_touching_stoppage"] = {
        "precision_vs_event_dead": float(inter / max(1, d_rule.sum())),
        "recall_vs_event_dead": float(inter / max(1, dead_ev.sum())),
    }
    return res, stops


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--games", nargs="*", default=None)
    args = ap.parse_args(argv)
    games = args.games or metrica_io.available_games()
    out = {}
    for gid in games:
        game = metrica_io.load_game(gid)
        res, stops = check(game)
        os.makedirs(os.path.join(ART, gid), exist_ok=True)
        with open(os.path.join(ART, gid, "deadball_check.json"), "w") as fh:
            json.dump(res, fh, indent=2)
        stops.to_csv(os.path.join(ART, gid, "event_stoppages.csv"), index=False)
        out[gid] = res
        print(json.dumps(res, indent=1))
    return out


if __name__ == "__main__":
    main()
