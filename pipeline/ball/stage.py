"""Command line shared by the ball, events and features stages (CONTRACTS.md, "Stage command line").

    python -m pipeline.<stage> match                       # artifacts/match/, aggregates only
    python -m pipeline.<stage> bench [--game ID ...]       # artifacts/bench/<ID>/
    python -m pipeline.<stage> bench --root DIR [DIR ...]  # explicit roots (scenes, dev copies)

Each stage gives ``run(root) -> dict of aggregates``. Match runs go through
``pipeline.report.match_entry`` and print only through ``summary()``.
"""
import argparse
import glob
import json
import os
import sys

import pandas as pd

from pipeline.report import match_entry, quiet_warnings, summary
from pipeline.io.roots import require_bench_roots

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACTS = os.path.join(REPO, "artifacts")


def read_meta(root):
    with open(os.path.join(root, "meta.json")) as handle:
        return json.load(handle)


def read(root, name):
    return pd.read_parquet(os.path.join(root, name))


def bench_roots(games=None, needs="ball_raw.parquet"):
    found = sorted(os.path.dirname(p) for p in
                   glob.glob(os.path.join(ARTIFACTS, "bench", "*", needs)))
    if games:
        found = [r for r in found if os.path.basename(r) in games]
    return found


def main(stage, run, needs, argv=None):
    parser = argparse.ArgumentParser(description="{} stage (CONTRACTS.md)".format(stage))
    parser.add_argument("target", choices=("match", "bench"))
    parser.add_argument("--game", nargs="*", help="bench game ids (default: all with inputs)")
    parser.add_argument("--root", nargs="*", help="explicit bench roots")
    args = parser.parse_args(argv)

    if args.target == "match":
        if args.game or args.root:
            parser.error("the match root is fixed")
        return _run_match(stage, run)

    roots = args.root or bench_roots(args.game, needs)
    require_bench_roots(roots)
    for root in roots:
        stats = run(root)
        summary(_name="{}.bench.{}".format(stage, os.path.basename(os.path.normpath(root))),
                **stats)
    summary(_name="{}.bench".format(stage), bench_roots_run=len(roots))
    return 0


def _run_match(stage, run):
    @match_entry
    def go():
        quiet_warnings()
        stats = run(os.path.join(ARTIFACTS, "match"))
        summary(_name=stage, **stats)
        return 0
    return go()


if __name__ == "__main__":
    sys.exit(2)
