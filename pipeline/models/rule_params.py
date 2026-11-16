"""Named weights and shapes of the rule scorer (rule_v1). One place, one rulebook ID per term.

The scorer is two sums of log-odds (pipeline/models/rule_scorer.py):

    z_dead = DEAD_BIAS + sum_k DEAD_W[k] * dead_term_k        P(D) = sigmoid(z_dead)
    z_team = sum_k TEAM_W[k] * team_term_k                    P(H | live) = sigmoid(z_team)
    P(H) = (1 - P(D)) P(H | live),  P(A) = (1 - P(D)) (1 - P(H | live))

Every term is a bounded, unit-free number per frame (0..1 for dead terms unless noted, -1..+1
with + = home for team terms), so a weight is "log-odds added at full strength". With no team
evidence z_team = 0 (H and A equal); DEAD_BIAS = log(1/2) makes a frame with no evidence at all
exactly (1/3, 1/3, 1/3).

DEFAULTS are the physically sensible start values (written before any bench run). The values in
DEAD_W / TEAM_W are the ones chosen on the bench by ``python -m pipeline.models.tune`` (bench
roots only, never the public 5 minutes); the comment on each line gives the start value. Signs
are fixed by the rule's direction and enforced as bounds by the tuner (SIGNS).
"""

# --- shapes (not tuned; chosen once from the rulebook and bench distributions) ----------------
SHAPE = {
    "gap_mid_s": 4.0,        # D12/D13: ball-gap length at which the length term is 0 (log2 scale)
    "gap_len_clip": 3.0,     # length term = clip(log2(gap / gap_mid), +-3)
    "line_near_m": 5.0,      # D3: last seen within this distance of a line ...
    "line_out_mps": 5.0,     # ... heading out: term = clip(out_speed / 5, 0, 1)
    "crowd_full": 4,         # P25: players within 3 m at the loss; term = min(n, 4) / 4
    "out_margin_m": 0.3,     # D1: seen ball beyond a line by more than this counts as outside
    "out_half_frames": 8,    # D2: outside share over a centred +-8 frame window (persistence)
    "speed_half_s": 1.0,     # D5: centred window for player speed (+-1 s)
    "speed_ref_mps": 1.6,    # D5: corrected RMS player speed where the slow term is 0
    "speed_scale_mps": 0.8,  # D5: slow term = clip((ref - speed) / scale, -2, 2) / 2
    "speed_floor_q": 2.0,    # percentile of the per-frame RMS speed taken as the noise floor
    "dip_ctx_s": 8.0,        # D5: slow-down dip = speed over +-8 s minus speed over +-1 s
    "dip_scale_mps": 1.0,    #     term = clip(dip / 1, 0, 1) (centred: past and future)
    "still_mps": 0.7,        # D9/R2-R5/P23: seen ball slower than this is stationary
    "still_half_s": 1.0,     # share of still-and-seen frames over a centred +-1 s window
    "moving_mps": 3.0,       # live: seen ball faster than this is moving
    "moving_half_s": 0.5,
    "players_full": 18,      # D10: fewer tracked players than this ...
    "players_span": 8,       # ... term = clip((18 - n) / 8, 0, 1)
    "tau_ctrl_s": 60.0,      # P1-P5: decay of the last control's ownership (start 8; bench 60)
    "tau_ctrl_next_s": 2.0,  # P1-P5: decay of the next control (look-ahead twin)
    "tau_touch_s": 3.0,      # B4: decay of the last team touch
    "near_scale_m": 2.0,     # P13: tanh((d_away - d_home) / 2)
    "near_max_m": 3.0,       # P13: only when someone is within 3 m of a seen ball
    "near_half_s": 0.5,      # P13: centred mean over +-0.5 s
    "spread_scale_m": 4.0,   # T1: (spread_home - spread_away) / 4, clipped
    "shape_half_s": 1.0,     # T1/T2: centred mean over +-1 s
}

# --- dead versus live ------------------------------------------------------------------------
DEAD_BIAS_DEFAULT = -0.693   # log(1/2): no evidence -> P(D) = 1/3

DEAD_DEFAULTS = {
    "miss": 0.5,            # D12  ball missing (neither seen nor filled)
    "miss_len": 2.0,        # D13  missing x clip(log2(gap length / 4 s), +-3) / 3: long gaps dead
    "miss_line": 1.0,       # D3   missing, last seen near a line heading out
    "miss_crowd": -0.5,     # P25  missing, lost in a crowd: weaker
    "out": 3.0,             # D1/D2 seen outside the lines, persistence share
    "exit": 3.0,            # D1-D3 between an exit event and the ball's return inside
    "pre_kickoff": 8.0,     # S4   mask from the restart stage (every frame before a half's kick-off)
    "form_kickoff": 2.0,    # R6   kick-off formation score
    "slow": 1.5,            # D5/D10/T9 collective slowness, -1..+1
    "slow_dip": 0.5,        # D5   dip in player speed against the surrounding seconds, 0..1
    "still": 1.0,           # D9/R2-R5 (trap P23) stationary seen ball
    "moving": -1.5,         # live: seen ball moving faster than a walk
    "players_low": 0.0,     # D10  few tracked players (not measurable on the bench yet)
}

# --- which team, given live ------------------------------------------------------------------
TEAM_DEFAULTS = {
    "ctrl_last": 2.0,       # P1-P5 last control's team, decaying
    "ctrl_next": 0.3,       # P1-P5 next control's team, decaying (centred twin)
    "carry": 1.5,           # P1   inside a carry span
    "pass": 1.0,            # P2   inside a pass span (kick to reception)
    "touch_last": 0.5,      # B4   last team touch, decaying
    "near": 0.3,            # P13  nearest-player team (aligned), weak
    "territory": 0.3,       # T2   both teams' centroids pushed towards a goal
    "spread": 0.2,          # T1   team with the ball is wider and deeper
}

# Sign constraints for the tuner: +1 weight >= 0, -1 weight <= 0, 0 free.
SIGNS = {
    "miss": 0, "miss_len": 1, "miss_line": 1, "miss_crowd": -1, "out": 1, "exit": 1,
    "pre_kickoff": 1, "form_kickoff": 1, "slow": 1, "slow_dip": 1, "still": 0, "moving": -1,
    "players_low": 1,
    "ctrl_last": 1, "ctrl_next": 0, "carry": 1, "pass": 1, "touch_last": 1, "near": 1,
    "territory": 0, "spread": 0,
}

# Weights fixed by law or not measurable on the bench: the tuner leaves them at the default.
FROZEN = {"pre_kickoff", "players_low"}

# Bench-chosen values: `python -m pipeline.models.tune --root artifacts/bench/metrica_g{1,2,3}_match
# --tau-ctrl 60` (2026-11-13, rebuilt noise v2, every 5th graded frame, L2 0.002 towards the
# defaults, bias held at the flat prior). Held-out-by-game 3-class log-loss 0.339 / 0.319 / 0.337
# against 0.452 / 0.454 / 0.503 with the defaults. tau_ctrl_s grid 4..120 s: held-out log-loss
# falls monotonically and is flat from 30 s on (0.3320 / 0.3317 / 0.3316 at 30 / 60 / 120).
# Fitted to 0 (rule kept, no measurable effect on this bench): miss_crowd (P25), slow_dip (D5).
DEAD_BIAS = DEAD_BIAS_DEFAULT
DEAD_W = {
    "miss": 0.452, "miss_len": 1.894, "miss_line": 0.916, "miss_crowd": 0.0, "out": 2.657,
    "exit": 2.791, "pre_kickoff": 8.0, "form_kickoff": 1.972, "slow": 3.318, "slow_dip": 0.0,
    "still": 1.429, "moving": -2.887, "players_low": 0.0,
}
TEAM_W = {
    "ctrl_last": 0.694, "ctrl_next": 0.806, "carry": 0.597, "pass": 0.838, "touch_last": 0.942,
    "near": 0.655, "territory": -0.071, "spread": 1.394,
}

MODEL_NAME = "rule_v1"
