import argparse
import functools
import sys

from pipeline.ball.stage import main
from pipeline.events import run, touch


def _split(argv):
    """Take the events-only option --touch-set out of argv; the rest goes to the shared CLI."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--touch-set", default=None, choices=sorted(touch.PARAM_SETS),
                    help="touch detector parameter set (default {})".format(touch.DEFAULT_SET))
    known, rest = ap.parse_known_args(argv)
    return known.touch_set, rest


if __name__ == "__main__":
    touch_set, rest = _split(sys.argv[1:])
    sys.exit(main("events", functools.partial(run, touch_set=touch_set), needs="ball.parquet",
                  argv=rest))
