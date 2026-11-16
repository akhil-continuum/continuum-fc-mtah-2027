"""T-24: a version's own probabilities model, label naming, provenance stamp, candidate steps.

Synthetic inputs only; nothing here reads the match.
"""
import json

import pytest

from pipeline import candidate
from pipeline.decode import __main__ as dec
from pipeline.decode import decoders
from pipeline.io import provenance


def test_version_model_default_and_own():
    for v in ("v1r", "v2r", "v2hr", "v2s", "v3a"):
        assert dec.version_model(v) == "rule_v1"
    assert dec.version_model("v6s") == "rule_v2"


def test_label_names():
    assert dec.label_name("v2s", "rule_v1", explicit=False) == "v2s"
    assert dec.label_name("v6s", "rule_v2", explicit=False) == "v6s"
    # explicit --probs keeps the old naming
    assert dec.label_name("v2s", "rule_v1") == "v2s"
    assert dec.label_name("v6s", "rule_v2") == "v6s_rule_v2"
    assert dec.label_name("v2s", "synth_a") == "v2s_synth_a"


@pytest.fixture
def captured(monkeypatch):
    calls = []
    monkeypatch.setattr(dec, "run_match", lambda v, m, e=True: calls.append(("match", v, m, e)) or 0)
    monkeypatch.setattr(dec, "run_bench",
                        lambda v, m, r, e=True: calls.append(("bench", v, m, e)) or 0)
    monkeypatch.setattr(dec, "require_bench_roots", lambda roots: None)
    return calls


def test_main_resolves_probs(captured):
    dec.main(["match", "--version", "v6s"])
    dec.main(["match", "--version", "v2s"])
    dec.main(["match", "--version", "v6s", "--probs", "rule_v1"])
    dec.main(["bench", "--version", "v6s", "--probs", "rule_v2"])
    dec.main(["bench", "--version", "v6s"])
    assert captured == [("match", "v6s", "rule_v2", False), ("match", "v2s", "rule_v1", False),
                        ("match", "v6s", "rule_v1", True), ("bench", "v6s", "rule_v2", True),
                        ("bench", "v6s", "rule_v2", False)]


def test_candidate_probs_model_and_steps():
    assert candidate.probs_model("v2s") == "rule_v1"
    assert candidate.probs_model("v6s") == "rule_v2"
    names = [n for n, _ in candidate.match_steps("v6s")]
    assert names.index("probs2") == names.index("probs") + 1 < names.index("decode")
    steps = dict(candidate.match_steps("v6s"))
    assert steps["probs2"][:4] == ["pipeline.models.team_model", "match", "--models", "rule_v2"]
    bench = candidate.bench_stage_steps(["match"])
    assert [n for n, _ in bench][-2:] == ["bench probs", "bench probs2"]
    assert bench[-1][1][:3] == ["pipeline.models.team_model", "bench", "--root"]
    models = candidate.team_models()
    assert models[0] == "rule_v2"
    assert bench[-1][1][3:] == candidate.level_roots(["match"]) + ["--models"] + models


def test_stamp_records_probs_model(tmp_path):
    labels = tmp_path / "labels" / "v6s.csv"
    labels.parent.mkdir()
    labels.write_text("frame,label\n0,H\n")
    stamp = provenance.write_stamp(str(labels), "v6s", probs=candidate.probs_model("v6s"))
    assert stamp["probs"] == "rule_v2"
    with open(provenance.stamp_path(str(labels))) as handle:
        assert json.load(handle)["probs"] == "rule_v2"
    assert "probs" in decoders.VERSIONS["v6s"]
