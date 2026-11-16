"""Rule scorer (rule_v1): per-frame P(H), P(A), P(D) from rules only, no learning.

    uv run python -m pipeline.models.rule_scorer match            # artifacts/match/, aggregates
    uv run python -m pipeline.models.rule_scorer bench [--root DIR ...] [--game ID ...]
    make match STAGE=probs   |   make bench STAGE=probs

Writes ``<root>/probs/rule_v1.npy``: float32 (n_frames, 3) = P(H), P(A), P(D) (CONTRACTS.md).

Two sums of log-odds with named weights (pipeline/models/rule_params.py, one rulebook ID per
term; the terms are in pipeline/models/terms.py):

    P(D)        = sigmoid(DEAD_BIAS + sum DEAD_W[k] * dead_term_k)
    P(H | live) = sigmoid(sum TEAM_W[k] * team_term_k)
    P(H) = (1 - P(D)) P(H | live),   P(A) = (1 - P(D)) (1 - P(H | live))

Frames outside the halves get (1/3, 1/3, 1/3); they are not graded. Bench runs score the
result against the game's truth.csv when one is found (``<root>/truth.csv``, else
``artifacts/bench/metrica_g<k>/truth.csv`` for a root whose name contains ``g<k>``) and print
per-frame quality; match runs print aggregates only (mean P(D), top-class shares, flat share).
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

from pipeline.models import rule_params as RP
from pipeline.models import terms as TM
from pipeline.io.roots import require_bench_roots
from pipeline.report import match_entry, quiet_warnings, summary

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACTS = os.path.join(REPO, "artifacts")
NEEDS = ("meta.json", "ball.parquet", "players.parquet", "events.parquet", "features.parquet")
FLAT_MAX = 0.40          # a frame is "flat" when no class has more than this probability
Z_CLIP = 30.0


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -Z_CLIP, Z_CLIP)))


# --- inputs ----------------------------------------------------------------------------------

def read_root(root, need_dead=True, compute_dead=True):
    """Contract files of a root as a dict. features_dead.parquet is read when present; on bench
    roots without it the restart stage's kick-off code is run in memory (nothing is written)."""
    missing = [f for f in NEEDS if not os.path.isfile(os.path.join(root, f))]
    if missing:
        raise FileNotFoundError("root lacks {} input file(s): {}".format(
            len(missing), ", ".join(missing)))
    with open(os.path.join(root, "meta.json")) as handle:
        meta = json.load(handle)
    rd = lambda name: pd.read_parquet(os.path.join(root, name))     # noqa: E731
    data = {"meta": meta, "ball": rd("ball.parquet"), "players": rd("players.parquet"),
            "events": rd("events.parquet"), "features": rd("features.parquet"),
            "dead": None, "dead_source": "none"}
    dpath = os.path.join(root, "features_dead.parquet")
    if os.path.isfile(dpath):
        data["dead"] = pd.read_parquet(dpath)
        data["dead_source"] = "file"
    elif need_dead and compute_dead:
        from pipeline.restarts import kickoff
        raw = os.path.join(root, "ball_raw.parquet")
        ball_raw = pd.read_parquet(raw, columns=["frame", "x_m", "y_m"]) \
            if os.path.isfile(raw) else None
        _k, feats = kickoff.run(data["players"], meta, ball_raw)
        data["dead"] = feats
        data["dead_source"] = "computed"
    return data


def compute_terms(data, shape=None):
    n = int(data["meta"]["n_frames"])
    ps = TM.player_summaries(data["players"], n)
    dead = TM.dead_terms(data["ball"], data["players"], data["events"], data["features"],
                         data["meta"], data["dead"], shape=shape, ps=ps)
    team, info = TM.team_terms(data["ball"], data["players"], data["events"], data["features"],
                               data["meta"], shape=shape, ps=ps)
    return dead, team, info


def combine(dead, team, n, halves, dead_w=None, team_w=None, dead_bias=None):
    dead_w = RP.DEAD_W if dead_w is None else dead_w
    team_w = RP.TEAM_W if team_w is None else team_w
    bias = RP.DEAD_BIAS if dead_bias is None else dead_bias
    zd = np.full(n, float(bias))
    for k, w in dead_w.items():
        if w:
            zd += w * dead[k]
    zt = np.zeros(n)
    for k, w in team_w.items():
        if w:
            zt += w * team[k]
    p_dead = sigmoid(zd)
    p_home = sigmoid(zt)
    probs = np.stack([(1 - p_dead) * p_home, (1 - p_dead) * (1 - p_home), p_dead], axis=1)
    inside = np.zeros(n, dtype=bool)
    for a, b in halves:
        inside[max(0, int(a)):min(n, int(b))] = True
    probs[~inside] = 1.0 / 3.0
    return probs.astype(np.float32)


def score_root(root, compute_dead=True):
    data = read_root(root, compute_dead=compute_dead)
    meta = data["meta"]
    n = int(meta["n_frames"])
    dead, team, info = compute_terms(data)
    probs = combine(dead, team, n, meta["halves"])
    info["dead_features"] = data["dead_source"]
    return probs, meta, info, dead, team


def write_probs(root, probs, name=RP.MODEL_NAME):
    out = os.path.join(root, "probs")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "{}.npy".format(name))
    np.save(path, probs.astype(np.float32))
    return path


def graded_mask(meta, n):
    spans = meta.get("graded_spans") or meta["halves"]
    g = np.zeros(n, dtype=bool)
    for a, b in spans:
        g[max(0, int(a)):min(n, int(b))] = True
    return g


def aggregates(probs, mask):
    """Aggregates only: mean P(D), top-class shares, flat share, max-probability histogram."""
    P = probs[mask].astype(float)
    top = np.argmax(P, axis=1)
    mx = P.max(axis=1)
    hist, _ = np.histogram(mx, bins=np.linspace(1 / 3, 1, 9))
    return {"frames_scored": int(mask.sum()),
            "mean_p_dead": round(float(P[:, 2].mean()), 4),
            "mean_p_home": round(float(P[:, 0].mean()), 4),
            "top_class_share": {"H": round(float(np.mean(top == 0)), 4),
                                "A": round(float(np.mean(top == 1)), 4),
                                "D": round(float(np.mean(top == 2)), 4)},
            "flat_share": round(float(np.mean(mx <= FLAT_MAX)), 4),
            "flat_definition_max_prob_le": FLAT_MAX,
            "max_prob_hist_edges_third_to_1": [round(float(e), 3)
                                               for e in np.linspace(1 / 3, 1, 9)],
            "max_prob_hist_share": [round(float(h) / max(1, len(mx)), 4) for h in hist]}


# --- bench -----------------------------------------------------------------------------------

def find_truth(root, override=None):
    if override:
        return override
    own = os.path.join(root, "truth.csv")
    if os.path.isfile(own):
        return own
    m = re.search(r"g(\d+)", os.path.basename(os.path.normpath(root)))
    if m:
        cand = os.path.join(ARTIFACTS, "bench", "metrica_g{}".format(m.group(1)), "truth.csv")
        if os.path.isfile(cand):
            return cand
    return None


def bench_metrics(probs, truth_path):
    from pipeline.models.metrics import evaluate
    labels = pd.read_csv(truth_path)["label"].to_numpy()
    if len(labels) != len(probs):
        raise ValueError("truth has {} rows, probs {}".format(len(labels), len(probs)))
    return evaluate(probs, labels)


def flat_metrics(m):
    """metrics.evaluate output -> summary()-safe dict (reliability as parallel lists)."""
    out = {k: v for k, v in m.items() if not k.startswith("reliability")}
    for key, short in (("reliability_dead", "rel_dead"), ("reliability_team", "rel_team")):
        rows = m[key]
        out[short + "_n"] = [r[2] for r in rows]
        out[short + "_pred"] = [r[3] for r in rows]
        out[short + "_obs"] = [r[4] for r in rows]
    return out


def bench_roots(games=None):
    found = sorted(os.path.dirname(p) for p in
                   glob.glob(os.path.join(ARTIFACTS, "bench", "*", "features.parquet")))
    if games:
        found = [r for r in found if os.path.basename(r) in games]
    return found


def run_bench(roots, truth=None, write=True):
    for root in roots:
        probs, meta, info, _d, _t = score_root(root)
        name = os.path.basename(os.path.normpath(root))
        if write:
            write_probs(root, probs)
        stats = dict(info)
        stats.update(aggregates(probs, graded_mask(meta, len(probs))))
        tp = find_truth(root, truth)
        stats["truth"] = (os.path.basename(os.path.dirname(tp)) + "/truth.csv") if tp \
            else "none found"
        if tp:
            stats.update(flat_metrics(bench_metrics(probs, tp)))
        summary(_name="rule_scorer.bench.{}".format(name), **stats)
    summary(_name="rule_scorer.bench", bench_roots_run=len(roots))
    return 0


# --- match -----------------------------------------------------------------------------------

@match_entry
def run_match():
    quiet_warnings()
    root = os.path.join(ARTIFACTS, "match")
    probs, meta, info, _d, _t = score_root(root, compute_dead=False)
    write_probs(root, probs)
    stats = {"model": RP.MODEL_NAME, "dead_features": info["dead_features"],
             "align_lag_frames": info["align_lag_frames"]}
    stats.update(aggregates(probs, graded_mask(meta, len(probs))))
    summary(_name="rule_scorer", **stats)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Rule scorer: probs/rule_v1.npy (CONTRACTS.md)")
    parser.add_argument("target", choices=("match", "bench"))
    parser.add_argument("--root", nargs="*", help="explicit bench roots")
    parser.add_argument("--game", nargs="*", help="bench game ids under artifacts/bench/")
    parser.add_argument("--truth", help="bench truth.csv (default: found from the root name)")
    parser.add_argument("--no-write", action="store_true", help="bench: score without writing")
    parser.add_argument("--version", help="pipeline version from make (accepted; the probs file "
                                          "is always probs/rule_v1.npy)")
    args = parser.parse_args(argv)
    if args.target == "match":
        if args.root or args.game or args.truth or args.no_write:
            parser.error("the match takes no options")
        return run_match()
    roots = args.root or bench_roots(args.game)
    require_bench_roots(roots)
    return run_bench(roots, args.truth, write=not args.no_write)


if __name__ == "__main__":
    sys.exit(main())
