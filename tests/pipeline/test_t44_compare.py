"""T-44: label-file comparison and gate-share diagnostics (aggregates only).

Synthetic label files and roots only; nothing here reads the match.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

from pipeline import compare_labels as CL
from pipeline import report
from pipeline.decode import checks
from pipeline.diagnostics import gate_share as GS
from pipeline.io import roots, sources

LET = np.array(["H", "A", "D", "X"])


def _codes(text):
    return np.array(["HADX".index(c) for c in text], dtype=np.int8)


def _root(tmp_path, labels, spans, name="metrica_g9_match"):
    """A synthetic bench-like root: meta.json and labels/<v>.csv for each {v: code string}."""
    root = tmp_path / name
    (root / "labels").mkdir(parents=True)
    n = len(next(iter(labels.values())))
    (root / "meta.json").write_text(json.dumps(
        {"n_frames": n, "fps": 25, "halves": [[0, n]], "graded_spans": [list(s) for s in spans],
         "source": "synthetic"}))
    for v, text in labels.items():
        c = _codes(text)
        pd.DataFrame({"frame": np.arange(n), "label": LET[c]}).to_csv(
            root / "labels" / "{}.csv".format(v), index=False, lineterminator="\n")
    return str(root)


# --- counting conventions --------------------------------------------------------------------

def test_counts_follow_checks_statistics():
    a = _codes("XHHHDDAAHHXXAAHX")
    spans = [(1, 10), (12, 15)]
    s = CL.counts(a, spans)
    ref = checks.statistics(a, spans)
    assert s["possessions"] == ref["possessions"] == 5      # HHH, AA, HH | AA, H
    assert s["stoppages"] == ref["stoppages"] == 1
    assert s["graded_frames"] == 12 and s["x_on_graded"] == 0
    assert s["share_d"] == pytest.approx(2 / 12)
    assert s["share_h"] == pytest.approx(6 / 12) and s["share_a"] == pytest.approx(4 / 12)


def test_counts_equal_official_spells():
    sys.path.insert(0, os.path.join(sources.REPO, "evaluation"))
    import possession_metrics as pm
    rng = np.random.default_rng(0)
    codes = np.repeat(rng.integers(0, 3, 300), rng.integers(1, 30, 300)).astype(np.int8)
    codes[:7] = 3
    codes[len(codes) // 2: len(codes) // 2 + 40] = 3
    runs = pm.valid_runs(codes)
    assert CL.counts(codes, runs)["possessions"] == len(pm.possession_spells(codes, runs))


def test_compare_ratio_agreement_and_gate():
    a = _codes("HHHHHDDDAAAAAAAA")
    b = _codes("HHAHHDDDAAAAHAAD")
    sa, sb, j = CL.compare(a, b, [(0, 16)])
    assert (sa["possessions"], sb["possessions"]) == (2, 6)
    assert j["agree"] == 13 and j["team_flip"] == 2 and j["live_vs_dead"] == 1
    ratio, diff, ok_r, ok_d = CL.gate_verdict(sa, sb)
    assert ratio == pytest.approx(3.0) and not ok_r
    assert diff == pytest.approx(1 / 16) and not ok_d
    with pytest.raises(ValueError):
        CL.compare(a, b[:-1], [(0, 15)])


def test_gate_edges():
    base = {"possessions": 100, "share_d": 0.400}
    assert CL.gate_verdict(base, {"possessions": 111, "share_d": 0.403})[2:] == (True, True)
    assert CL.gate_verdict(base, {"possessions": 126, "share_d": 0.397})[2:] == (True, True)
    assert CL.gate_verdict(base, {"possessions": 110, "share_d": 0.4031})[2:] == (False, False)
    assert CL.gate_verdict(base, {"possessions": 127, "share_d": 0.40})[2] is False
    assert CL.gate_verdict({"possessions": 0, "share_d": None}, base)[2:] == (False, False)


def test_pool_recomputes_rates_from_sums():
    s1 = CL.counts(_codes("HHHHDDDD"), [(0, 8)])
    s2 = CL.counts(_codes("AAHHHHHHHHHHHHHH"), [(0, 16)])
    p = CL.pool([s1, s2])
    assert p["possessions"] == 3 and p["graded_frames"] == 24
    assert p["share_d"] == pytest.approx(4 / 24)
    assert p["possessions_per_min"] == pytest.approx(3 / (24 / 25 / 60))


# --- what is printed -------------------------------------------------------------------------

def test_report_fields_are_aggregates_summary_accepts(tmp_path):
    a = _codes("HHHHHDDDAAAAAAAA" * 4)
    b = _codes("HHAHHDDDAAAAHAAD" * 4)
    sa, sb, j = CL.compare(a, b, [(0, 64)])
    fields = CL.report_fields("v6w", "v11g", sa, sb, j)
    for key, value in fields.items():
        assert not isinstance(value, (dict, np.ndarray)), key
        if isinstance(value, list):
            assert len(value) <= 2, key
    for banned in ("shortest", "longest", "first", "last", "frame_"):
        assert not any(banned in k for k in fields)
    out = report.summary(_name="t44_fields", _reports_dir=str(tmp_path), _quiet=True, **fields)
    assert out["possessions_a"] == 8 and out["possessions_b"] == 24


def test_summary_still_refuses_per_frame_values(tmp_path):
    a = _codes("HHAD")
    with pytest.raises(report.SummaryRefused):
        report.summary(_name="t44_bad", _reports_dir=str(tmp_path), labels=a)
    with pytest.raises(report.SummaryRefused):
        report.summary(_name="t44_bad", _reports_dir=str(tmp_path), frames=list(range(21)))
    assert not (tmp_path / "t44_bad.json").exists()


def test_run_bench_on_synthetic_roots(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(report, "REPORTS_DIR", tmp_path / "reports")
    r1 = _root(tmp_path, {"va": "XHHHHDDAAAX", "vb": "XHHAHDDAAAX"}, [(1, 10)], "metrica_g8_low")
    r2 = _root(tmp_path, {"va": "XAAAADDHHHX", "vb": "XAAAADDHHHX"}, [(1, 10)], "metrica_g7_low")
    assert CL.run_bench("va", "vb", ["low"], roots=[r1, r2]) == 0
    printed = capsys.readouterr().out
    assert "compare_labels.bench.roots: possessions_a = 4" in printed
    assert "compare_labels.bench.roots: possessions_b = 6" in printed
    pooled = json.loads((tmp_path / "reports" / "compare_labels.bench.roots.json").read_text())
    assert pooled["possession_ratio_b_over_a"] == 1.5
    assert pooled["agreement_graded"] == round(17 / 18, 4)


def test_run_bench_refuses_match_root():
    with pytest.raises(roots.NotABenchRoot):
        CL.run_bench("v6w", "v11g", ["match"], roots=[sources.match_root()])


def test_run_match_entry_on_stand_in_root(tmp_path, capsys, monkeypatch):
    """The match entry point, pointed at a synthetic root: aggregates only, no stamp -> reported;
    a missing file exits 1 with type and short message, no path."""
    monkeypatch.setattr(report, "REPORTS_DIR", tmp_path / "reports")
    r = _root(tmp_path, {"v6w": "XHHHHDDAAAAX", "v11g": "XHHAHDDAAHAX"}, [(1, 11)], "standin")
    monkeypatch.setattr(CL, "match_root", lambda: r)
    assert CL.run_match("v6w", "v11g") == 0
    out = json.loads((tmp_path / "reports" / "compare_labels.json").read_text())
    assert out["possessions_a"] == 2 and out["possessions_b"] == 6
    assert out["provenance_a"] == "no_stamp"
    capsys.readouterr()
    with pytest.raises(SystemExit) as ex:
        CL.run_match("v6w", "nosuch")
    assert ex.value.code == 1
    err = capsys.readouterr().err
    assert err.startswith("error: FileNotFoundError") and r not in err


# --- gate share ------------------------------------------------------------------------------

def test_gate_share_shares_from_counts(tmp_path):
    c = {"n_graded": 100, "n_label_live": 50, "flagged_graded": 40, "speak_graded": 55,
         "differ_graded": 52, "flagged_label_live": 20, "speak_label_live": 25,
         "differ_label_live": 25, "differ_outside_speak": 0}
    s = GS.shares(c)
    assert s == {"flagged_share_graded": 0.4, "speak_share_graded": 0.55,
                 "differ_share_graded": 0.52, "flagged_share_label_live": 0.4,
                 "speak_share_label_live": 0.5, "differ_share_label_live": 0.5}
    report.summary(_name="t44_gate", _reports_dir=str(tmp_path), _quiet=True, **s)
    assert GS.add({"a": 1}, {"a": 2, "b": 3}) == {"a": 3, "b": 3}


def test_gate_share_refuses_ungated_version():
    with pytest.raises(ValueError):
        GS.tree_params("v6w")
    model, params = GS.tree_params("v11g")
    assert model == "rule_p1g" and params["gate"] == "G4" and params["base"] == "rule_w2"


def test_gate_share_bench_refuses_match_root():
    with pytest.raises(roots.NotABenchRoot):
        GS.run_bench("v11g", ["match"], roots=[sources.match_root()])


def test_differ_flags_match_attach_gate(tmp_path):
    """differ = decode.inputs.attach_gate's gate (the T-42 gate_share_graded definition)."""
    from pipeline.decode import inputs
    r = _root(tmp_path, {"v": "HHHHHHHH"}, [(0, 8)], "metrica_g6_low")
    os.makedirs(os.path.join(r, "probs"))
    rng = np.random.default_rng(1)
    base = rng.dirichlet([1, 1, 1], 8).astype(np.float32)
    tree = base.copy()
    tree[[2, 5], 0] += 0.1
    np.save(os.path.join(r, "probs", "m_base.npy"), base)
    np.save(os.path.join(r, "probs", "m_tree.npy"), tree)
    flags = GS.differ_flags(r, "m_tree", "m_base", 8)
    assert np.flatnonzero(flags).tolist() == [2, 5]
    inp = inputs.from_arrays(np.load(os.path.join(r, "probs", "m_tree.npy")))
    inputs.attach_gate(inp, r, "m_base")
    assert np.array_equal(inp.gate, flags)
