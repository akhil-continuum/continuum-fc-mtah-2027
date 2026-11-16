"""The load stage on a small synthetic EPTS pair whose answers are known exactly."""
import json

import numpy as np
import pandas as pd
import pytest

from pipeline.io import epts, load

L, W, FPS = 100.0, 50.0, 25
N = 100
STEP = 0.001          # normalised x per frame for the moving player: 0.1 m/frame = 2.5 m/s


def _meta_xml(order_a, order_b):
    def spec(start, end, order):
        groups = "".join(
            '<SplitRegister separator=","><PlayerChannelRef playerChannelId="{0}_x"/>'
            '<PlayerChannelRef playerChannelId="{0}_y"/><PlayerChannelRef playerChannelId="{0}_s"/>'
            '</SplitRegister>'.format(p) for p in order)
        return ('<DataFormatSpecification startFrame="{}" endFrame="{}" separator=":">'
                '<StringRegister name="frameCount"/><SplitRegister separator=";">{}</SplitRegister>'
                '<SplitRegister separator=","><BallChannelRef channelId="x"/>'
                '<BallChannelRef channelId="y"/><BallChannelRef channelId="z"/></SplitRegister>'
                '</DataFormatSpecification>').format(start, end, groups)

    players = "".join('<Player id="{}" teamId="{}"><Name>{}</Name><ShirtNumber>{}</ShirtNumber>'
                      '</Player>'.format(p, t, p, i)
                      for i, (p, t) in enumerate([("p1", "TA"), ("p2", "TA"),
                                                  ("p3", "TB"), ("p4", "TB")], 1))
    pcs = "".join('<PlayerChannel id="{0}_{1}" playerId="{0}" channelId="{1}"/>'.format(p, c)
                  for p in ("p1", "p2", "p3", "p4") for c in ("x", "y", "s"))
    return """<?xml version="1.0" encoding="utf-8"?>
<main><Metadata>
<GlobalConfig><FrameRate>25</FrameRate><ProviderGlobalParameters>
<ProviderParameter><Name>first_half_start</Name><Value>1</Value></ProviderParameter>
<ProviderParameter><Name>first_half_end</Name><Value>50</Value></ProviderParameter>
<ProviderParameter><Name>second_half_start</Name><Value>51</Value></ProviderParameter>
<ProviderParameter><Name>second_half_end</Name><Value>100</Value></ProviderParameter>
</ProviderGlobalParameters></GlobalConfig>
<Sessions><Session id="s"><FieldSize><Width>100</Width><Height>50</Height></FieldSize></Session></Sessions>
<Teams><Team id="TA"><Name>Alpha</Name></Team><Team id="TB"><Name>Beta</Name></Team></Teams>
<Players>{players}</Players>
<Devices><Device id="d"><Sensors><Sensor id="position"><Channels>
<Channel id="x"><Name>position_x</Name><Unit>normalized</Unit></Channel>
<Channel id="y"><Name>position_y</Name><Unit>normalized</Unit></Channel>
<Channel id="s"><Name>speed</Name><Unit>m/s</Unit></Channel>
<Channel id="z"><Name>position_z</Name><Unit>m</Unit></Channel>
</Channels></Sensor></Sensors></Device></Devices>
<PlayerChannels>{pcs}</PlayerChannels>
</Metadata>
<DataFormatSpecifications>{a}{b}</DataFormatSpecifications></main>
""".format(players=players, pcs=pcs, a=spec(1, 50, order_a), b=spec(51, 100, order_b))


def truth(frame_id):
    """Normalised (x, y) per player at a frame id. Home (TA) left in half 1, right in half 2."""
    second = frame_id > 50
    pos = {
        "p1": (0.2 + STEP * frame_id, 0.3),
        "p2": (0.3, 0.7),
        "p3": (0.7, 0.4),
        "p4": (0.8, 0.6),
    }
    if 10 <= frame_id < 20:                       # p4 duplicates p3 (0.3 m apart in x)
        pos["p4"] = (0.7 + 0.003, 0.4)
    if second:
        pos = {p: (1.0 - x, y) for p, (x, y) in pos.items()}
    return pos


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    folder = tmp_path_factory.mktemp("synthetic")
    order_a, order_b = ["p1", "p2", "p3", "p4"], ["p2", "p1", "p4", "p3"]
    (folder / "metadata.xml").write_text(_meta_xml(order_a, order_b))
    lines = []
    for f in range(1, N + 1):
        pos = truth(f)
        order = order_a if f <= 50 else order_b
        groups = ";".join("{:.5f},{:.5f},1.0".format(*pos[p]) for p in order)
        ball = "NaN,NaN,NaN" if f % 4 == 0 else "{:.5f},{:.5f},0.5".format(0.5, 0.5)
        lines.append("{}:{}:{}".format(f, groups, ball))
    # p2 missing on frame id 61 (second half, so mirrored to x = 0.7)
    assert "0.70000,0.70000,1.0" in lines[60]
    lines[60] = lines[60].replace("0.70000,0.70000,1.0", "NaN,NaN,NaN")
    (folder / "tracking.txt").write_text("\n".join(lines) + "\n")
    out = folder / "out"
    stats = load.build(str(folder / "metadata.xml"), str(folder / "tracking.txt"), str(out),
                       game_id="synthetic")
    return folder, out, stats


def test_spec_order_followed(built):
    folder, _, _ = built
    meta = epts.read_metadata(str(folder / "metadata.xml"))
    assert len(meta.specs) == 2 and meta.specs[0].n_fields == 1 + 4 * 3 + 3
    raw = epts.read_tracking(meta, str(folder / "tracking.txt"))
    assert raw.n_bad_lines == 0 and raw.n_unspecified == 0
    slot = {p: i for i, p in enumerate(raw.player_ids)}
    for row in (0, 30, 55, 99):   # both specifications
        f = int(raw.frame_ids[row])
        for p, (x, y) in truth(f).items():
            if row == 60 and p == "p2":
                continue
            assert raw.px[row, slot[p]] == pytest.approx(x, abs=1e-5)
            assert raw.py[row, slot[p]] == pytest.approx(y, abs=1e-5)
    assert set(raw.pextra) == {"s"}
    assert set(raw.ball) == {"x", "y", "z"}
    assert list(raw.player_team) == [0, 0, 1, 1]


def test_meta_json(built):
    _, out, _ = built
    meta = json.loads((out / "meta.json").read_text())
    assert meta["n_frames"] == N
    assert meta["halves"] == [[0, 50], [50, 100]]
    assert meta["pitch"] == [L, W]
    assert meta["frame_convention"] == "atd_frame_id_minus_1"
    assert meta["attack_direction"] == {"1": {"0": 1, "1": -1}, "2": {"0": -1, "1": 1}}


def test_players_parquet(built):
    _, out, stats = built
    p = pd.read_parquet(out / "players.parquet")
    assert list(p.columns) == ["frame", "team", "x_m", "y_m", "vx", "vy"]
    assert p["frame"].min() == 0 and p["frame"].max() == N - 1
    per_frame = p.groupby("frame").size()
    # duplicates merged on frame ids 10..19 -> video frames 9..18 have 3 players
    assert (per_frame.loc[9:18] == 3).all()
    assert per_frame.loc[60] == 3          # p2 missing
    assert per_frame.loc[[0, 30, 70]].eq(4).all()
    assert stats["slot_merges"] == 10
    # merged position is the mean of the two
    away = p[(p.frame == 12) & (p.team == 1)]
    assert away["x_m"].iloc[0] == pytest.approx((0.7 + 0.703) / 2 * L, abs=1e-3)
    # p1 moves at 2.5 m/s along +x in half 1 (video frame 20 = frame id 21)
    row = p[(p.frame == 20) & (p.team == 0)].sort_values("x_m").iloc[0]
    assert row["vx"] == pytest.approx(STEP * L * FPS, rel=1e-3)
    assert row["vy"] == pytest.approx(0.0, abs=1e-6)
    # sorted by frame then team
    assert (np.diff(p["frame"].to_numpy()) >= 0).all()


def test_ball_raw(built):
    _, out, stats = built
    b = pd.read_parquet(out / "ball_raw.parquet")
    assert len(b) == N and b["frame"].tolist() == list(range(N))
    assert {"x_m", "y_m", "z_m", "seen"} <= set(b.columns)
    assert b["seen"].sum() == N - N // 4
    assert b.loc[b.seen, "x_m"].eq(np.float32(0.5 * L)).all()
    assert stats["ball_seen_share_all"] == 0.75
