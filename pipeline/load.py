"""Load stage entry point: ``python -m pipeline.load match|bench``.

    uv run python -m pipeline.load match                     # artifacts/match/, aggregates only
    uv run python -m pipeline.load bench                     # every game under bench_data/
    uv run python -m pipeline.load bench --game sample_game_3
    uv run python -m pipeline.load bench --meta M.xml --raw T.txt --out DIR [--game-id ID]

Every stage follows the same command-line shape, so ``make match STAGE=<stage>`` and
``make bench STAGE=<stage>`` can call any of them. Match runs print through
pipeline.report.summary only and never show a traceback.
"""
import argparse
import sys

from pipeline.io import load, sources
from pipeline.report import match_entry, quiet_warnings, summary


@match_entry
def run_match():
    quiet_warnings()
    game = sources.match_input()
    stats = load.build(game.meta_path, game.raw_path, game.out_dir, game=game.game,
                       game_id=game.game_id, n_frames=game.n_frames, halves=game.halves,
                       pitch=game.pitch, source="match")
    summary(_name="load", **stats)
    return 0


def run_bench(inputs):
    if not inputs:
        summary(_name="load.bench", bench_games_found=0)
        return 0
    for game in inputs:
        stats = load.build(game.meta_path, game.raw_path, game.out_dir, game_id=game.game_id,
                           source="bench")
        summary(_name="load.bench.{}".format(game.game_id), **stats)
    summary(_name="load.bench", bench_games_found=len(inputs))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Load stage (CONTRACTS.md)")
    parser.add_argument("target", choices=("match", "bench"))
    parser.add_argument("--game", nargs="*", help="bench game ids to run (default: all)")
    parser.add_argument("--bench-dir", default=sources.BENCH_DATA)
    parser.add_argument("--meta", help="explicit EPTS metadata XML (bench only)")
    parser.add_argument("--raw", help="explicit EPTS raw tracking txt (bench only)")
    parser.add_argument("--out", help="output folder for --meta/--raw")
    parser.add_argument("--game-id", help="game id for --meta/--raw")
    args = parser.parse_args(argv)

    if args.target == "match":
        if args.meta or args.raw or args.out or args.game:
            parser.error("the match takes its paths from games.asset_paths() only")
        return run_match()

    if args.meta or args.raw:
        if not (args.meta and args.raw and args.out):
            parser.error("--meta, --raw and --out go together")
        return run_bench([sources.explicit_input(args.meta, args.raw, args.out, args.game_id)])
    return run_bench(sources.bench_inputs(args.bench_dir, only=args.game))


if __name__ == "__main__":
    sys.exit(main())
