"""T-21: every bench-only entry point refuses a match root before reading anything.

The root is a temp dir whose meta.json says source=match; nothing real is created or read.
"""
import json

import pytest

from pipeline.io import roots


@pytest.fixture
def fake_match(tmp_path):
    d = tmp_path / "somewhere"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"source": "match"}))
    return str(d)


def _decode(r):
    from pipeline.decode import __main__ as m
    return m.main(["bench", "--version", "v2s", "--root", r])


def _sweep(r):
    from pipeline.decode import sweep, decoders
    return sweep.main(["--version", sorted(sweep.GRIDS)[0], "--root", r])


def _stage(r):
    from pipeline.ball import stage
    return stage.main("ball", lambda root: {}, "ball_raw.parquet", ["bench", "--root", r])


def _rule_scorer(r):
    from pipeline.models import rule_scorer
    return rule_scorer.main(["bench", "--root", r])


def _tune(r):
    from pipeline.models import tune
    return tune.main(["--root", r])


def _events_measure(r):
    from pipeline.events import measure
    return measure.measure(r, "unused.csv")


def _restarts_measure(r):
    from pipeline.restarts import measure
    return measure.measure_game("x", root=r)


def _stoppages_measure(r):
    from pipeline.restarts import measure_stoppages
    return measure_stoppages.measure_game(r)      # absolute id: join(BENCH, abs) is abs


def _input_stats(r):
    from pipeline.diagnostics import input_stats
    return input_stats.main(["bench", "--game", r])


@pytest.mark.parametrize("entry", [_decode, _sweep, _stage, _rule_scorer, _tune,
                                   _events_measure, _restarts_measure, _stoppages_measure,
                                   _input_stats])
def test_bench_entry_refuses_match_root(entry, fake_match):
    with pytest.raises(roots.NotABenchRoot):
        entry(fake_match)
