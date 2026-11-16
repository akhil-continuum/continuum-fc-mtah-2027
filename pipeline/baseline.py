"""Reproduce the organisers' nearest-player baseline through an approved entry point.

    uv run python -m pipeline.baseline          (or: make baseline)
    uv run python evaluation/run_evaluation.py  (scores everything in evaluation/results/)

Runs the starter logic unchanged (starter/baseline.py: nearest tracked player to the ball, centred
majority vote of +/-0, 25, 50, 125 frames) and writes ``evaluation/results/repro_<variant>.csv``.
The shipped ``baseline_<variant>.csv`` files are tracked in git and are left alone; this module
reports, per variant, whether the reproduction is byte-identical to the shipped file.

Prints through pipeline.report.summary only: label counts per variant and identity flags.
"""
import filecmp
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from pipeline.report import match_entry, quiet_warnings, summary  # noqa: E402

RESULTS = os.path.join(REPO, "evaluation", "results")
PREFIX = "repro_"
LABELS = os.path.join(REPO, "artifacts", "match", "labels")
V0_VARIANT = "baseline_smooth50"   # the published floor (starter/baseline.py docstring)


@match_entry
def main():
    quiet_warnings()
    import contextlib
    import io

    import numpy as np
    from starter import baseline as starter_baseline

    # starter.build() prints team names and a file name; swallow that, report through summary.
    with contextlib.redirect_stdout(io.StringIO()):
        codes = starter_baseline.build()
    out = {"ball_located_share": round(float(np.mean(codes != starter_baseline.NOBODY)), 4)}
    for name, window in starter_baseline.VARIANTS:
        labels = starter_baseline.smooth(codes, window)
        path = starter_baseline.write_submission(
            os.path.join(RESULTS, PREFIX + name + ".csv"), labels)
        shipped = os.path.join(RESULTS, name + ".csv")
        out[name + "_counts"] = {k: int(np.sum(labels == k)) for k in ("H", "A", "D")}
        out[name + "_identical_to_shipped"] = bool(
            os.path.isfile(shipped) and filecmp.cmp(path, shipped, shallow=False))
        if name == V0_VARIANT:
            # v0: the published floor, in the Labels contract location.
            starter_baseline.write_submission(os.path.join(LABELS, "v0.csv"), labels)
    out["v0_variant"] = V0_VARIANT
    summary(_name="baseline", **out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
