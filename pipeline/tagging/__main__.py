"""Tagging stage: ``python -m pipeline.tagging match|bench`` (CONTRACTS.md, Tags row).

    uv run python -m pipeline.tagging match                                  # writes tags.parquet
    uv run python -m pipeline.tagging match --agreement --ref-version v6s    # G10 table, writes nothing
    uv run python -m pipeline.tagging match --agreement --ref pseudo         # once T-37a exists
    uv run python -m pipeline.tagging bench --root DIR [DIR ...]             # "no tags" on bench roots

Only the match has a Smart Tagging file (games.py). Every printed value goes through
``pipeline.report.summary``: counts, shares, per-code rows and histograms of at most 20 bins;
never an instance, an interval, a time or a frame number. Reports: ``reports/tagging.json`` and
``reports/tagging_agreement_<reference>.json``.
"""
import argparse
import functools
import sys

from pipeline.ball.stage import main
from pipeline.tagging import agreement, stage


def _split(argv):
    """Take the tagging-only options out of argv; the rest goes to the shared stage CLI."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--agreement", action="store_true",
                    help="print the per-code agreement table (G10) instead of writing tags")
    ap.add_argument("--ref", default="version", choices=("version", agreement.PSEUDO),
                    help="reference owner: decoded labels of --ref-version, or pseudo-labels")
    ap.add_argument("--ref-version", default=None,
                    help="decoded version whose labels/<version>.csv is the reference")
    known, rest = ap.parse_known_args(argv)
    return known, rest


def entry(argv):
    known, rest = _split(argv)
    if not known.agreement:
        if known.ref_version or known.ref != "version":
            sys.stderr.write("error: --ref and --ref-version need --agreement\n")
            return 2
        return main("tagging", stage.run, needs="meta.json", argv=rest)
    if known.ref == "version" and not known.ref_version:
        sys.stderr.write("error: --agreement needs --ref-version NAME or --ref pseudo\n")
        return 2
    if known.ref == agreement.PSEUDO and known.ref_version:
        sys.stderr.write("error: give --ref pseudo or --ref-version, not both\n")
        return 2
    name = agreement.PSEUDO if known.ref == agreement.PSEUDO else known.ref_version
    if not agreement._VERSION_NAME.match(name):
        sys.stderr.write("error: version name must be letters, digits, '_', '.', '-'\n")
        return 2
    run = functools.partial(agreement.run, ref=known.ref, ref_version=known.ref_version)
    return main("tagging_agreement_{}".format(name), run, needs="meta.json", argv=rest)


if __name__ == "__main__":
    sys.exit(entry(sys.argv[1:]))
