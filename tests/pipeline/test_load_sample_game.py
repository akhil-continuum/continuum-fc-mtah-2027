"""The load stage on Metrica Sample Game 3 (same EPTS format as the match).

Skipped when the fixture is absent. Fetch it with:
    curl -sSL -o tests/fixtures/metrica/Sample_Game_3/<file> \
      https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_3/<file>
for <file> in Sample_Game_3_metadata.xml and Sample_Game_3_tracking.txt.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.diagnostics import atd_channels
from pipeline.io import epts, load

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "metrica" / "Sample_Game_3"
META = FIX / "Sample_Game_3_metadata.xml"
RAW = FIX / "Sample_Game_3_tracking.txt"

pytestmark = pytest.mark.skipif(not (META.is_file() and RAW.is_file()),
                                reason="Sample Game 3 fixture not downloaded")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("sg3")
    stats = load.build(str(META), str(RAW), str(out), game_id="sample_game_3")
    return out, stats


def test_counts(built):
    out, stats = built
    assert stats["n_records"] == 143761 and stats["n_frames"] == 143761
    assert stats["frame_ids_contiguous"] and stats["frame_ids_start_at_1"]
    assert stats["lines_bad_field_count"] == 0 and stats["lines_without_spec"] == 0
    assert stats["n_specs"] == 11
    assert 21.5 < stats["players_per_frame_in_halves_mean"] <= 22.0
    assert 0.6 < stats["ball_seen_share_all"] < 0.75


def test_meta(built):
    out, _ = built
    meta = json.loads((out / "meta.json").read_text())
    # provider parameters: first half 1..69661, second 69662..143761 (frame ids)
    assert meta["halves"] == [[0, 69661], [69661, 143761]]
    assert meta["pitch"] == [105.0, 68.0]
    # home goalkeeper starts at x ~ 0.85: home attacks towards -x in the first half
    assert meta["attack_direction"]["1"] == {"0": -1, "1": 1}
    assert meta["attack_direction"]["2"] == {"0": 1, "1": -1}


def test_first_frame_matches_raw_line(built):
    out, _ = built
    first = RAW.open().readline().strip().split(":")
    groups = [g.split(",") for g in first[1].split(";")]
    want = sorted((float(x) * 105.0, float(y) * 68.0) for x, y in groups if x != "NaN")
    p = pd.read_parquet(out / "players.parquet")
    got = sorted(zip(p.loc[p.frame == 0, "x_m"], p.loc[p.frame == 0, "y_m"]))
    assert len(got) == len(want)
    assert np.allclose(np.array(got), np.array(want), atol=1e-3)


def test_parser_handles_substitution_specs():
    meta = epts.read_metadata(str(META))
    orders = {tuple(leaf.ref for leaf in s.leaves if leaf.role == "player") for s in meta.specs}
    assert len(orders) > 1      # the group order really does change between specifications


def test_atd_channel_audit():
    result = atd_channels.audit(str(META), str(RAW))
    assert result["raw_ball_group_fields"] == {"2": 143761}
    assert set(result["raw_player_group_fields"]) == {"2"}
    assert result["player_channel_kinds"] == {"x": 35, "y": 35}
    assert not (result["flag_ball_state_channel"] or result["flag_ball_owner_channel"]
                or result["flag_height_channel"])
