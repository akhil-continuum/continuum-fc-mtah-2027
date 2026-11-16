"""P-20 gate in the agreement mode: mirror codes, kept codes and weights, weighted coverage and
agreement, edge lag. Synthetic root only.

Labels (25 fps, graded [100, 5900)): H [100, 1000), A [1000, 2000), H [2000, 3000),
A [3000, 4000), H [4000, 5900). Core (2 s) frames: H [150, 950), A [1050, 1950), H [2050, 2950),
A [3050, 3950), H [4050, 5850).
"""
import math
import os

import numpy as np
import pandas as pd
import pytest

from pipeline import report
from pipeline.tagging import agreement, votes
from tagging_synth import ALIASES, AWAY, HOME, inst, make_root, write_xml

N = 6000
TAGS = [inst("BUILD UP", 2010, 2900, HOME),                       # A 1.0
        inst("DEFENDING IN ATTACKING THIRD", 2010, 2900, AWAY),   # mirror of BUILD UP
        inst("CREATING CHANCES", 4100, 4500, HOME),               # A 400 / 500 = 0.8
        inst("CREATING CHANCES", 4500, 4600, AWAY),
        inst("BALL IN FINAL THIRD", 150, 950, HOME),              # A 900 / 1000 = 0.9
        inst("BALL IN FINAL THIRD", 2050, 2150, HOME),
        inst("BALL IN FINAL THIRD", 1100, 1200, HOME),
        inst("BALL IN THE BOX", 3060, 3840, AWAY),                # A 780 / 1000 = 0.78
        inst("BALL IN THE BOX", 5000, 5220, AWAY),
        inst("PLAYERS IN THE BOX", 3100, 3800, AWAY),             # A 700 / 1000 = 0.7: dropped
        inst("PLAYERS IN THE BOX", 5300, 5600, AWAY)]


def build(tmp_path):
    root = make_root(tmp_path / "root", N, [[100, 5900]])
    lab = np.array(["X"] * N, dtype=object)
    for a, b, v in ((100, 1000, "H"), (1000, 2000, "A"), (2000, 3000, "H"), (3000, 4000, "A"),
                    (4000, 5900, "H")):
        lab[a:b] = v
    os.makedirs(os.path.join(root, "labels"))
    pd.DataFrame({"frame": np.arange(N), "label": lab}).to_csv(
        os.path.join(root, "labels", "vt.csv"), index=False)
    return root, write_xml(tmp_path / "tags.xml", TAGS)


@pytest.fixture
def out(tmp_path, monkeypatch):
    monkeypatch.setattr(agreement, "LAG_MIN_MATCHED", 3)
    root, xml = build(tmp_path)
    result = agreement.run(root, ref_version="vt", xml_path=xml, aliases=ALIASES)
    report.summary(_name="gate_test", _reports_dir=str(tmp_path / "reports"), _quiet=True,
                   **result)
    return result


def test_mirror_codes_are_found_from_the_arrays(out):
    assert out["mirror_codes_dropped"] == {"defending_in_attacking_third": "build_up"}
    assert out["agreement_defending_in_attacking_third"]["agree_core"] == 1.0


def test_kept_codes_and_weights(out):
    assert out["kept_agree_a075"] == {"build_up": 1.0, "creating_chances": 0.8,
                                      "ball_in_final_third": 0.9, "ball_in_the_box": 0.78}
    assert out["kept_weight_a075"] == {
        "build_up": agreement.W_CAP, "creating_chances": round(0.5 * math.log(4), 4),
        "ball_in_final_third": round(0.5 * math.log(9), 4),
        "ball_in_the_box": round(0.5 * math.log(0.78 / 0.22), 4)}
    assert list(out["kept_agree_a080"]) == ["build_up", "creating_chances", "ball_in_final_third"]
    assert list(out["kept_agree_a085"]) == ["build_up", "ball_in_final_third"]


def test_weighted_gate_row(out):
    row = out["gate_a085"]
    assert row["kept_codes"] == 2 and row["live_frames"] == 5800
    assert row["coverage"] == round((800 + 100 + 890) / 5800, 4)
    assert (row["core_voting"], row["agree_core"]) == (1750, round(1650 / 1750, 4))
    assert (row["noncore_voting"], row["agree_noncore_uncertain"]) == (40, 1.0)
    assert row["gate_value"] == round(row["coverage"] * (row["agree_core"] - 0.70), 4)
    assert row["gate_pass"] is False
    assert out["gate_a075"]["kept_codes"] == 4 and out["gate_a080"]["kept_codes"] == 3


def test_weights_decide_conflicts():
    frames = pd.DataFrame({votes.vote_column("BUILD UP"): np.array([1] * 10, np.int8),
                           votes.vote_column("PROGRESSION"): np.array([-1] * 10, np.int8)})
    ref = agreement.Reference("r", np.zeros(10, np.int8), np.ones(10, bool), np.zeros(10, bool))
    graded = np.ones(10, bool)
    strong_h = [("BUILD UP", 1.0, 1.1), ("PROGRESSION", 0.8, 0.69)]
    assert agreement.gate_row(frames, ref, graded, strong_h, 0.75)["agree_core"] == 1.0
    strong_a = [("BUILD UP", 0.8, 0.69), ("PROGRESSION", 1.0, 1.1)]
    assert agreement.gate_row(frames, ref, graded, strong_a, 0.75)["agree_core"] == 0.0
    tie = [("BUILD UP", 0.8, 0.69), ("PROGRESSION", 0.8, 0.69)]
    assert agreement.gate_row(frames, ref, graded, tie, 0.75)["coverage"] == 0.0


def test_edge_lag(out):
    start = out["edge_lag_start_all_kept"]
    assert (start["matched"], start["unmatched"]) == (4, 4)      # offsets 10, 100, 50, 60
    assert (start["q25_frames"], start["median_frames"], start["q75_frames"]) == (40.0, 55.0, 70.0)
    end = out["edge_lag_end_all_kept"]
    assert (end["matched"], end["unmatched"]) == (2, 6)           # offsets -100, -50
    assert end["median_frames"] is None                           # below LAG_MIN_MATCHED
    bu = out["edge_lag_start_build_up"]
    assert (bu["matched"], bu["unmatched"], bu["median_frames"]) == (1, 0, None)
    assert out["edge_lag_gate"]["median_start_frames"] == 55.0
    assert out["edge_lag_gate"]["gate_pass"] is False
    assert "edge_lag_start_players_in_the_box" not in out           # not a kept code
    assert "edge_lag_start_defending_in_attacking_third" not in out  # mirror


def test_lag_summary_never_reports_quantiles_of_few_edges():
    assert agreement.LAG_MIN_MATCHED >= 5
    few = agreement.lag_summary([3, 4], 1)
    assert few == {"matched": 2, "unmatched": 1, "q25_frames": None, "median_frames": None,
                   "q75_frames": None}
    many = agreement.lag_summary([0, 10, 20, 30, 40], 0)
    assert many["median_frames"] == 20.0


def test_gate_with_pseudo_reference(tmp_path):
    root, xml = build(tmp_path)
    pseudo = np.full(N, -1, dtype=np.int8)
    pseudo[2000:3000] = 0
    result = agreement.run(root, ref="pseudo", xml_path=xml, aliases=ALIASES,
                           pseudo_fn=lambda r: pseudo)
    assert result["kept_agree_a075"] == {"build_up": 1.0, "ball_in_final_third": 1.0}
    assert result["gate_a075"]["live_frames"] == 1000
    assert result["gate_a075"]["coverage"] == round(890 / 1000, 4)
