"""Check-only mode: run the decoder's output checks on an existing labels file, without decoding.

    uv run python -m pipeline.check_labels match --version v2s     # artifacts/match/labels/v2s.csv
    uv run python -m pipeline.check_labels match --team <name>     # submissions/<name>.csv
    uv run python -m pipeline.check_labels bench --version v2s --level match [--game ...]
    uv run python -m pipeline.check_labels bench --version v2s --root DIR [DIR ...]
    make check-labels VERSION=v2s     |     make check-labels TEAM=<name>

The checks are pipeline.decode.checks.check (decoder-engineer's; used read-only): row count,
alphabet, X exactly on ungraded frames, S4-masked frames D, no restart to the team the law
forbids; plus the S10/S11 warnings and the aggregate statistics. The hidden-path check (R0)
needs the decoder's hidden states and is not available from a labels file; it is reported as
"not checked". The evidence (S4 mask, law rows) is read from the root as the decoder reads it;
no probabilities are needed (a flat array stands in; no check reads it).

Match runs print aggregates only, through summary(), and also report whether the labels'
provenance stamp is current (pipeline.io.provenance). Exit code 1 when any blocking check fails.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

from pipeline.io import provenance
from pipeline.io.roots import require_bench_root
from pipeline.io.sources import ARTIFACTS, REPO, match_root
from pipeline.report import match_entry, quiet_warnings, summary

GAMES = ("metrica_g1", "metrica_g2", "metrica_g3")
CODES = {"H": 0, "A": 1, "D": 2, "X": 3}
KEEP = ("possessions_per_min", "share_d", "stoppages", "stoppages_per_90",
        "median_possession_s", "share_possessions_under_1_5s",
        "share_possessions_under_4s", "median_stoppage_s",
        "possession_length_hist", "law_restarts_checked", "law_restarts_wrong_team",
        "stoppages_under_s10_min", "graded_minutes")
MAX_LISTED_FAILURES = 5


def read_codes(path):
    table = pd.read_csv(path, dtype={"label": str})
    if list(table.columns[:2]) != ["frame", "label"]:
        raise ValueError("labels file must have columns frame,label")
    if not np.array_equal(table["frame"].to_numpy(), np.arange(len(table))):
        raise ValueError("labels file frames are not 0..n-1 in order")
    labels = table["label"].fillna("").astype(str).str.strip().str.upper()
    return labels.map(CODES).fillna(-1).to_numpy().astype(np.int8)


def check_file(root, path, compute=None):
    """(failures, warnings, aggregates) for the labels file at path against root's evidence."""
    from pipeline.decode import checks, inputs
    meta = inputs.read_meta(root)
    n = int(meta["n_frames"])
    flat = np.full((n, 3), 1.0 / 3.0, dtype=np.float32)
    inp = inputs.build(root, flat, compute=compute)
    codes = read_codes(path)
    failures, warns, agg = checks.check(codes, inp, hidden=None)
    agg["hidden_path_violations"] = "not checked (labels file only)"
    return failures, warns, agg


def _report(name, failures, warns, agg, **head):
    out = dict(head)
    out["checks_failed"] = len(failures)
    out["warnings"] = len(warns)
    for i, msg in enumerate(failures[:MAX_LISTED_FAILURES], 1):
        out["failure_{}".format(i)] = msg[:80]       # messages carry counts only
    for i, msg in enumerate(warns[:MAX_LISTED_FAILURES], 1):
        out["warning_{}".format(i)] = msg[:80]
    out.update({k: agg.get(k) for k in KEEP + ("hidden_path_violations",) if k in agg})
    summary(_name=name, **out)


@match_entry
def run_match(version=None, team=None):
    quiet_warnings()
    root = match_root()
    if team:
        path = os.path.join(REPO, "submissions", "{}.csv".format(team))
        head = {"file": "submission", "team": team}
    else:
        path = os.path.join(root, "labels", "{}.csv".format(version))
        from pipeline.submit import version_expect
        ok, reason = provenance.check_stamp(path, expect=version_expect(version))
        head = {"file": "match labels", "version": version, "provenance": reason}
    if not os.path.isfile(path):
        raise FileNotFoundError("labels file not found")
    failures, warns, agg = check_file(root, path)
    _report("check_labels", failures, warns, agg, **head)
    return 1 if failures else 0


def run_bench(version, roots):
    from pipeline.decode import decoders
    compute = decoders.VERSIONS[version].get("compute", "kickoff") \
        if version in decoders.VERSIONS else "full"
    bad = 0
    for root in roots:
        root = require_bench_root(root)
        path = os.path.join(root, "labels", "{}.csv".format(version))
        failures, warns, agg = check_file(root, path, compute=compute)
        bad += bool(failures)
        _report("check_labels.bench.{}".format(os.path.basename(root)), failures, warns, agg,
                version=version)
    summary(_name="check_labels.bench", version=version, roots=len(roots), roots_failing=bad)
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="output checks on an existing labels file")
    ap.add_argument("target", choices=("match", "bench"))
    ap.add_argument("--version", help="labels/<version>.csv")
    ap.add_argument("--team", help="match: check submissions/<team>.csv instead")
    ap.add_argument("--level", default="match")
    ap.add_argument("--game", nargs="*", default=list(GAMES))
    ap.add_argument("--root", nargs="*")
    args = ap.parse_args(argv)
    if args.target == "match":
        if args.root or (bool(args.version) == bool(args.team)):
            ap.error("the match takes exactly one of --version or --team")
        return run_match(args.version, args.team)
    if not args.version:
        ap.error("bench needs --version")
    roots = args.root or [os.path.join(ARTIFACTS, "bench", "{}_{}".format(g, args.level))
                          for g in args.game]
    return run_bench(args.version, roots)


if __name__ == "__main__":
    sys.exit(main())
