"""Video-frame mapping: ATD frame_id and provider IdFrame, all numbers taken from games.py."""
import numpy as np

import games
from pipeline.io import frames

G = games.GAME


def test_atd_mapping_is_minus_one_and_invertible():
    ids = np.arange(1, games.source_frames("tactical") + 1)
    video = frames.atd_to_video(ids)
    assert video[0] == 0 and video[-1] == games.source_frames("tactical") - 1
    assert np.array_equal(frames.video_to_atd(video), ids)


def test_idframe_mapping_matches_games_per_half():
    for half, first_id, last_id in games.halves(G):
        probe = np.array([first_id, (first_id + last_id) // 2, last_id])
        got = frames.idframe_to_video(probe, G)
        want = [games.id_to_video(G, int(i)) for i in probe]
        assert got.tolist() == want
        # each half uses its own offset
        assert np.all(got == probe - games.frame_offset(G, half))


def test_idframe_boundaries_equal_half_video_blocks():
    blocks = games.half_video_blocks(G)
    for (half, first_id, last_id), (h2, first_v, last_v) in zip(games.halves(G), blocks):
        assert half == h2
        assert frames.idframe_to_video(np.array([first_id, last_id]), G).tolist() == \
            [first_v, last_v]
        # round trip through games.video_to_id
        assert games.video_to_id(G, first_v) == first_id
        assert games.video_to_id(G, last_v) == last_id


def test_idframe_outside_halves_is_unmapped():
    (_, _, last1), (_, first2, _) = games.halves(G)[0], games.halves(G)[1]
    gap = np.array([last1 + 1, first2 - 1])
    assert frames.idframe_to_video(gap, G).tolist() == [-1, -1]


def test_detect_convention():
    n = games.source_frames("tactical")
    assert frames.detect_convention(np.arange(1, n + 1), G) == frames.ATD
    ids = np.concatenate([np.arange(a, b + 1) for _, a, b in games.halves(G)])
    assert frames.detect_convention(ids, G) == frames.IDFRAME
    # unknown games are never treated as IdFrame
    assert frames.detect_convention(ids, "other") == frames.ATD


def test_match_halves_are_half_open_blocks():
    halves = frames.match_halves(G)
    blocks = games.half_video_blocks(G)
    assert halves == [[b[1], b[2] + 1] for b in blocks]
    n = games.source_frames("tactical")
    idx = frames.half_index(n, halves)
    graded = int((idx > 0).sum())
    assert graded == sum(b[2] - b[1] + 1 for b in blocks)
    for (start, stop), h in zip(halves, (1, 2)):
        assert idx[start] == h and idx[stop - 1] == h
        assert idx[start - 1] != h and idx[stop] != h


def test_provider_halves():
    params = {"first_half_start": "1", "first_half_end": "100",
              "second_half_start": "101", "second_half_end": "250",
              "first_extra_half_start": None}
    assert frames.provider_halves(params) == [[0, 100], [100, 250]]
    assert frames.provider_halves({}) == []
