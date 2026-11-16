"""Ball-player time alignment diagnostic (aggregates only).

    uv run python -m pipeline.events.diag_alignment match
    uv run python -m pipeline.events.diag_alignment bench --root artifacts/bench/dev_g3_clean

For players shifted by L frames relative to the ball (players at frame f + L against the ball at
frame f), reports the median ball-to-nearest-player distance on seen frames and, at the peaks of
the ball's velocity change (touch candidates before the player test), the share with a player
within 1 m and 2 m. With aligned tracks both are best at L = 0. Writes reports/diag_alignment.json
through summary(); nothing per frame is printed.
"""
import numpy as np

from pipeline.ball.contact import contact
from pipeline.ball.stage import main, read, read_meta
from pipeline.events import base, touch

LAGS = (-12, -8, -6, -4, -2, 0, 2, 4, 6, 8, 12)


def _peaks(inp):
    dv, ang, vb, va = touch.velocity_change(inp)
    thr = max(touch.DV_MPS, touch.NOISE_K * touch.noise_scale(dv))
    idx = np.flatnonzero(dv > thr)
    taken = np.zeros(inp.n, dtype=bool)
    out = []
    for i in idx[np.argsort(-dv[idx], kind="stable")]:
        lo, hi = max(0, i - touch.W), min(inp.n, i + touch.W + 1)
        if taken[lo:hi].any():
            continue
        taken[i] = True
        out.append(i)
    return np.array(sorted(out), dtype=int)


def run(root):
    meta = read_meta(root)
    ball = read(root, "ball.parquet")
    players = read(root, "players.parquet")
    inp = base.make_inputs(ball, players, meta)
    peaks = _peaks(inp)
    seen = inp.seen
    med, near1, near2 = {}, {}, {}
    for lag in LAGS:
        p = players.copy()
        p["frame"] = p["frame"] - lag          # player row at f + lag lands on ball frame f
        d = contact(p, inp.x, inp.y, inp.n).d_any
        ds = d[seen & np.isfinite(d)]
        med[str(lag)] = round(float(np.median(ds)), 3) if len(ds) else None
        dp = d[peaks]
        near1[str(lag)] = round(float(np.mean(dp <= 1.0)), 4) if len(dp) else None
        near2[str(lag)] = round(float(np.mean(dp <= 2.0)), 4) if len(dp) else None
    return {"n_peaks": int(len(peaks)), "median_nearest_m_by_lag": med,
            "peak_share_player_1m_by_lag": near1, "peak_share_player_2m_by_lag": near2}


if __name__ == "__main__":
    import sys
    sys.exit(main("diag_alignment", run, needs="ball.parquet"))
