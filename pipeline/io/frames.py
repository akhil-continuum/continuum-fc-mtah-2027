"""Tracking frame ids to video frames, and the halves in video frames.

Two counters exist for the match (see DATA.md and games.py):

- the ATD ``frame_id``, 1..199319, one record per video frame: ``video_frame = frame_id - 1``.
  This is what the delivered ATD uses (DATA.md, "Frame alignment"; starter/load_atd.py).
- the provider ``IdFrame`` counter the ground truth was derived from, mapped per half by
  ``games.id_to_video`` (``id_frame = video_frame + offsets[half]``, two offsets one frame apart).

The loader detects which counter a file uses rather than assuming: if every frame id falls inside
the IdFrame span of the halves listed in games.py, it is IdFrame; otherwise it is the ATD counter.
No frame number is written here: all of them come from games.py.
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
import games  # noqa: E402

ATD = "atd_frame_id_minus_1"
IDFRAME = "idframe_per_half_offset"


def detect_convention(frame_ids, game=None):
    """ATD or IDFRAME for an array of frame ids. IDFRAME only for games listed in games.py."""
    frame_ids = np.asarray(frame_ids)
    if game in games.GAMES and len(frame_ids):
        spans = games.halves(game)
        inside = np.zeros(len(frame_ids), dtype=bool)
        for _, first, last in spans:
            inside |= (frame_ids >= first) & (frame_ids <= last)
        if inside.all():
            return IDFRAME
    return ATD


def atd_to_video(frame_ids):
    """video_frame = frame_id - 1."""
    return np.asarray(frame_ids, dtype=np.int64) - 1


def video_to_atd(video_frames):
    return np.asarray(video_frames, dtype=np.int64) + 1


def idframe_to_video(frame_ids, game):
    """games.id_to_video, vectorised: per-half offsets, -1 outside both halves."""
    frame_ids = np.asarray(frame_ids, dtype=np.int64)
    out = np.full(len(frame_ids), -1, dtype=np.int64)
    for half, first, last in games.halves(game):
        inside = (frame_ids >= first) & (frame_ids <= last)
        out[inside] = frame_ids[inside] - games.frame_offset(game, half)
    return out


def to_video(frame_ids, convention, game=None):
    if convention == ATD:
        return atd_to_video(frame_ids)
    if convention == IDFRAME:
        return idframe_to_video(frame_ids, game)
    raise ValueError("unknown frame convention")


def match_halves(game):
    """[[start, stop), ...] in video frames from games.half_video_blocks (stop exclusive)."""
    return [[int(first), int(last) + 1] for _, first, last in games.half_video_blocks(game)]


def provider_halves(provider_params, convention=ATD, game=None):
    """Halves from EPTS provider parameters (first_half_start, ...), as video [start, stop).

    The parameters are frame ids, inclusive on both ends. Empty list when they are absent.
    """
    out = []
    for prefix in ("first_half", "second_half", "first_extra_half", "second_extra_half"):
        start = provider_params.get(prefix + "_start")
        end = provider_params.get(prefix + "_end")
        if not start or not end:
            continue
        lo, hi = to_video(np.array([int(start), int(end)]), convention, game)
        out.append([int(lo), int(hi) + 1])
    return out


def half_index(n_frames, halves):
    """int8 array over video frames: half number (1, 2, ...) or 0 outside every half."""
    out = np.zeros(n_frames, dtype=np.int8)
    for i, (start, stop) in enumerate(halves, 1):
        out[max(0, start):min(n_frames, stop)] = i
    return out
