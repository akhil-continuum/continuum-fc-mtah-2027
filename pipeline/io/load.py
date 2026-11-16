"""Load stage: EPTS pair -> players.parquet, ball_raw.parquet, meta.json (see CONTRACTS.md).

One function, ``build()``, for bench games and the match. Callers give the paths and the
per-game facts; for the match those come from games.py only (see pipeline/io/sources.py).

What it does, in order:

1. read the raw file following the metadata's DataFormatSpecifications (pipeline/io/epts.py);
2. map frame ids to video frames (pipeline/io/frames.py; detected, not assumed);
3. scale 0-1 coordinates to metres: x * pitch length, y * pitch width. The axes keep the
   provider's orientation (Metrica: origin at a corner, y growing across the pitch);
4. velocities per slot by a centred difference over +/-VEL_HALF_WIDTH frames (falling back to
   +/-1 at gaps), set to NaN above MAX_SPEED_MPS (a slot swap, not a sprint);
5. merge same-team slots closer than MERGE_RADIUS_M on the same frame (tracker duplicates):
   position and velocity are averaged, the slot is dropped;
6. attack direction per team per half from which side of the other team each team's centroid
   sits on, frame by frame, over the half (majority);
7. write the ball exactly as observed (no cleaning: that is the ball stage) to ball_raw.parquet.

Returns a dict of aggregates suitable for pipeline.report.summary.
"""
import json
import os

import numpy as np
import pandas as pd

from pipeline.io import epts, frames

# Params
MERGE_RADIUS_M = 0.5
VEL_HALF_WIDTH = 2          # frames each side for the centred difference
MAX_SPEED_MPS = 12.5        # above this a velocity is a tracking jump, not running
MIN_PLAYERS_FOR_SIDE = 5    # per team, for a frame to vote on attack direction
FLOAT = np.float32


def centred_velocity(pos, fps, half_width=VEL_HALF_WIDTH):
    """(n_frames, n_slots) positions in metres -> centred-difference velocity, NaN at gaps."""
    out = np.full(pos.shape, np.nan)
    for k in range(half_width, 0, -1):
        fwd = np.full(pos.shape, np.nan)
        back = np.full(pos.shape, np.nan)
        fwd[:-k] = pos[k:]
        back[k:] = pos[:-k]
        v = (fwd - back) * fps / (2.0 * k)
        fill = np.isnan(out) & np.isfinite(v)
        out[fill] = v[fill]
    out[np.isnan(pos)] = np.nan
    return out


def merge_duplicates(x, y, vx, vy, radius=MERGE_RADIUS_M):
    """Merge slots of one team closer than ``radius`` on the same frame.

    Inputs are (n_frames, n_slots) for one team. Slot j is merged into the first earlier slot i
    still alive and within ``radius`` (distances use the original positions). Returns
    (x, y, vx, vy, alive, n_merges) with merged slots averaged into the survivor.
    """
    n_slots = x.shape[1]
    alive = np.isfinite(x) & np.isfinite(y)
    sx, sy = np.where(alive, x, 0.0), np.where(alive, y, 0.0)
    count = alive.astype(np.float64)
    vok = alive & np.isfinite(vx) & np.isfinite(vy)
    svx, svy = np.where(vok, vx, 0.0), np.where(vok, vy, 0.0)
    vcount = vok.astype(np.float64)
    r2 = radius * radius
    n_merges = 0
    for i in range(n_slots):
        for j in range(i + 1, n_slots):
            both = alive[:, i] & alive[:, j]
            if not both.any():
                continue
            d2 = (x[:, i] - x[:, j]) ** 2 + (y[:, i] - y[:, j]) ** 2
            hit = both & (d2 < r2)
            if not hit.any():
                continue
            n_merges += int(hit.sum())
            sx[hit, i] += sx[hit, j]
            sy[hit, i] += sy[hit, j]
            count[hit, i] += count[hit, j]
            svx[hit, i] += svx[hit, j]
            svy[hit, i] += svy[hit, j]
            vcount[hit, i] += vcount[hit, j]
            alive[hit, j] = False
    with np.errstate(invalid="ignore", divide="ignore"):
        mx = np.where(alive, sx / count, np.nan)
        my = np.where(alive, sy / count, np.nan)
        mvx = np.where(alive & (vcount > 0), svx / vcount, np.nan)
        mvy = np.where(alive & (vcount > 0), svy / vcount, np.nan)
    return mx, my, mvx, mvy, alive, n_merges


def attack_directions(cx_home, cx_away, n_home, n_away, halves):
    """{half: {team: +1 or -1}}, +1 meaning the team attacks towards increasing x.

    Each frame inside a half with enough players of both teams votes for the team whose
    centroid is at lower x attacking +x. Returns (directions, agreement per half).
    """
    directions, agreement = {}, {}
    for h, (start, stop) in enumerate(halves, 1):
        sl = slice(max(0, start), stop)
        ok = (n_home[sl] >= MIN_PLAYERS_FOR_SIDE) & (n_away[sl] >= MIN_PLAYERS_FOR_SIDE)
        if not ok.any():   # small or sparse data: any frame with both teams present votes
            ok = (n_home[sl] >= 1) & (n_away[sl] >= 1)
        diff = (cx_away[sl] - cx_home[sl])[ok]
        if len(diff) == 0:
            directions[str(h)] = {"0": 0, "1": 0}
            agreement[str(h)] = None
            continue
        share = float(np.mean(diff > 0))
        home_dir = 1 if share >= 0.5 else -1
        directions[str(h)] = {"0": home_dir, "1": -home_dir}
        agreement[str(h)] = round(max(share, 1.0 - share), 4)
    return directions, agreement


def build(meta_path, raw_path, out_dir, game=None, game_id=None, n_frames=None, halves=None,
          pitch=None, source="bench"):
    """Run the load stage and write the three files to ``out_dir``. Returns aggregates."""
    meta = epts.read_metadata(meta_path)
    raw = epts.read_tracking(meta, raw_path)

    convention = frames.detect_convention(raw.frame_ids, game)
    video = frames.to_video(raw.frame_ids, convention, game)
    mapped = video >= 0
    if n_frames is None:
        n_frames = int(video[mapped].max()) + 1 if mapped.any() else 0
    in_range = mapped & (video < n_frames)
    n_out_of_range = int((~in_range).sum())

    # Keep the last record of a duplicated frame.
    keep = np.zeros(len(video), dtype=bool)
    if in_range.any():
        idx = np.flatnonzero(in_range)
        _, last_pos = np.unique(video[idx][::-1], return_index=True)
        keep[idx[::-1][last_pos]] = True
    n_duplicates = int(in_range.sum() - keep.sum())

    if halves is None:
        halves = frames.provider_halves(meta.provider_params, convention, game) or [[0, n_frames]]
    if pitch is None:
        pitch = meta.field_size
    length, width = float(pitch[0]), float(pitch[1])
    fps = float(meta.fps)

    rows = np.flatnonzero(keep)
    vf = video[rows]
    n_slots = len(raw.player_ids)
    X = np.full((n_frames, n_slots), np.nan)
    Y = np.full((n_frames, n_slots), np.nan)
    X[vf] = raw.px[rows] * length
    Y[vf] = raw.py[rows] * width
    VX = centred_velocity(X, fps)
    VY = centred_velocity(Y, fps)
    too_fast = np.hypot(VX, VY) > MAX_SPEED_MPS
    n_too_fast = int(too_fast.sum())
    VX[too_fast] = np.nan
    VY[too_fast] = np.nan

    parts, n_merges, centroid, counts = [], 0, {}, {}
    for team in (0, 1):
        cols = np.flatnonzero(raw.player_team == team)
        mx, my, mvx, mvy, alive, merges = merge_duplicates(X[:, cols], Y[:, cols],
                                                           VX[:, cols], VY[:, cols])
        n_merges += merges
        counts[team] = alive.sum(axis=1)
        with np.errstate(invalid="ignore"):
            centroid[team] = np.where(counts[team] > 0,
                                      np.nansum(np.where(alive, mx, 0.0), axis=1)
                                      / np.maximum(counts[team], 1), np.nan)
        f_idx, s_idx = np.nonzero(alive)   # row-major: sorted by frame, then slot
        parts.append(pd.DataFrame({
            "frame": f_idx.astype(np.int32),
            "team": np.full(len(f_idx), team, dtype=np.int8),
            "x_m": mx[f_idx, s_idx].astype(FLOAT),
            "y_m": my[f_idx, s_idx].astype(FLOAT),
            "vx": mvx[f_idx, s_idx].astype(FLOAT),
            "vy": mvy[f_idx, s_idx].astype(FLOAT),
            "_slot": s_idx.astype(np.int16),
        }))
    players = pd.concat(parts, ignore_index=True)
    players = players.sort_values(["frame", "team", "_slot"], kind="mergesort")
    players = players.drop(columns="_slot").reset_index(drop=True)

    directions, agreement = attack_directions(centroid[0], centroid[1], counts[0], counts[1],
                                              halves)

    # Ball, as observed. One row per video frame; NaN and seen=False where nothing was recorded.
    ball = pd.DataFrame({"frame": np.arange(n_frames, dtype=np.int32)})
    scale = {"x": length, "y": width, "z": 1.0}
    for kind, values in raw.ball.items():
        full = np.full(n_frames, np.nan) if values.dtype != object \
            else np.full(n_frames, None, dtype=object)
        full[vf] = values[rows]
        if kind in scale:
            name = "{}_m".format(kind)
            ball[name] = (full * scale[kind]).astype(FLOAT)
        else:
            name = "raw_{}".format(kind)
            ball[name] = full.astype(FLOAT) if full.dtype != object else full.astype(str)
    if "x_m" not in ball or "y_m" not in ball:
        raise ValueError("no ball x/y channels in the specification")
    ball["seen"] = np.isfinite(ball["x_m"].to_numpy()) & np.isfinite(ball["y_m"].to_numpy())

    os.makedirs(out_dir, exist_ok=True)
    players.to_parquet(os.path.join(out_dir, "players.parquet"), index=False)
    ball.to_parquet(os.path.join(out_dir, "ball_raw.parquet"), index=False)
    meta_out = {
        "game_id": game_id or game or "",
        "source": source,
        "n_frames": int(n_frames),
        "fps": fps,
        "halves": [[int(a), int(b)] for a, b in halves],
        "attack_direction": directions,
        "attack_direction_agreement": agreement,
        "attack_direction_note": "+1: team attacks towards increasing x_m; team 0 = H, 1 = A",
        "pitch": [length, width],
        "coordinates": "metres; x along the pitch length, y across; provider orientation",
        "frame_convention": convention,
        "n_track_slots": int(n_slots),
        "slots_per_team": [int((raw.player_team == t).sum()) for t in (0, 1)],
        "merge_radius_m": MERGE_RADIUS_M,
        "velocity_half_width_frames": VEL_HALF_WIDTH,
        "max_speed_mps": MAX_SPEED_MPS,
        "ball_columns": [c for c in ball.columns if c not in ("frame",)],
        "player_extra_channels": sorted(raw.pextra),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as handle:
        json.dump(meta_out, handle, indent=1, sort_keys=True)
        handle.write("\n")

    # Aggregates only.
    in_half = frames.half_index(n_frames, halves) > 0
    per_frame = np.bincount(players["frame"].to_numpy(), minlength=n_frames)
    seen = ball["seen"].to_numpy()
    hist_edges = list(range(0, 31, 2))
    hist = np.histogram(per_frame[in_half], bins=hist_edges + [10 ** 6])[0]
    stats = {
        "n_records": int(len(raw.frame_ids)),
        "n_frames": int(n_frames),
        "frame_convention": convention,
        "frame_ids_contiguous": bool(len(raw.frame_ids) > 0 and np.array_equal(
            np.sort(raw.frame_ids), np.arange(raw.frame_ids.min(),
                                              raw.frame_ids.min() + len(raw.frame_ids)))),
        "frame_ids_start_at_1": bool(len(raw.frame_ids) > 0 and raw.frame_ids.min() == 1),
        "records_out_of_range": n_out_of_range,
        "duplicate_frames": n_duplicates,
        "lines_bad_field_count": int(raw.n_bad_lines),
        "lines_without_spec": int(raw.n_unspecified),
        "n_specs": len(meta.specs),
        "n_track_slots": int(n_slots),
        "slots_without_team": int((raw.player_team < 0).sum()),
        "n_halves": len(halves),
        "frames_in_halves": int(in_half.sum()),
        "player_rows": int(len(players)),
        "players_per_frame_in_halves_mean": round(float(per_frame[in_half].mean()), 3)
        if in_half.any() else None,
        "players_per_frame_hist_edges": hist_edges,
        "players_per_frame_hist": [int(v) for v in hist],
        "slot_merges": int(n_merges),
        "velocities_dropped_too_fast": n_too_fast,
        "ball_seen_share_all": round(float(seen.mean()), 4) if n_frames else None,
        "ball_seen_share_in_halves": round(float(seen[in_half].mean()), 4)
        if in_half.any() else None,
        "attack_dir_home": {h: d["0"] for h, d in directions.items()},
        "attack_dir_agreement": agreement,
    }
    return stats
