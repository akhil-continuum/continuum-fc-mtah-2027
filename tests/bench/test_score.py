"""Scoring harness (bench/score.py): official metric, pooling, extra columns."""

import os

import numpy as np
import pandas as pd
import pytest

from bench import score
from bench.score import pm


def codes(s):
    return pm.label_codes(np.array(list(s)))


def write(path, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({"frame": np.arange(len(labels)), "label": list(labels)}).to_csv(path, index=False)


TRUTH = "X" + "H" * 60 + "A" * 30 + "D" * 50 + "A" * 40 + "H" * 20 + "X" * 5


def test_perfect_labels_score_one():
    gt = codes(TRUTH)
    pred = codes(TRUTH.replace("X", "D"))
    r = score.score_pair(gt, pred, bootstrap_n=0)
    assert r["spell_f1_mean"] == pytest.approx(1.0)
    assert r["missed_stoppages"] == 0 and r["invented_stoppages"] == 0
    assert r["restart_team_acc"] == pytest.approx(1.0)
    assert r["false_splits_per_hour"] == 0


def test_extra_metrics_detect_errors():
    gt = codes(TRUTH)
    # merge the stoppage into the neighbouring A possessions and split the first H possession
    bad = ("D" + "H" * 25 + "A" * 5 + "H" * 30 + "A" * 30 + "A" * 50 + "A" * 40 + "H" * 20
           + "D" * 5)
    r = score.score_pair(gt, codes(bad), bootstrap_n=0)
    assert r["missed_stoppages"] == 1
    assert r["false_splits_per_hour"] > 0
    assert r["spell_f1_mean"] < 1.0


def test_pooled_perfect_and_files(tmp_path, monkeypatch):
    monkeypatch.setattr(score, "ART", str(tmp_path))
    for g in ("ga", "gb"):
        root = tmp_path / "{}_match".format(g)
        write(str(root / "truth.csv"), TRUTH)
        write(str(root / "labels" / "perfect.csv"), TRUTH.replace("X", "H"))
    res = score.score_version("perfect", "match", "default", games=("ga", "gb"), bootstrap_n=50)
    assert set(res["games"]) == {"ga", "gb"}
    assert res["pooled"]["spell_f1_mean"] == pytest.approx(1.0)
    assert res["pooled"]["n_gt_spells"] == 2 * res["games"]["ga"]["n_gt_spells"]
    assert res["pooled"]["spell_bootstrap_low"] == pytest.approx(1.0)


def test_length_mismatch_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(score, "ART", str(tmp_path))
    root = tmp_path / "ga_match"
    write(str(root / "truth.csv"), TRUTH)
    write(str(root / "labels" / "short.csv"), TRUTH[:-1])
    with pytest.raises(ValueError):
        score.score_version("short", "match", "default", games=("ga",), bootstrap_n=0)
