# Features (`<root>/features.parquet`)

One row per video frame, `frame` plus the columns below, all float32 (NaN where undefined).
Built by `pipeline/features/__init__.py` from `ball.parquet`, `events.parquet` and
`players.parquet`. Team suffix `_h` = home (team 0), `_a` = away (team 1). Team level only: a
"nearest player" is whoever of that team is nearest on that frame; no track slot is followed.
Windows are centred (`_c05` = ±0.5 s, `_c1` = ±1 s). Every "time since" has a "time until"
twin, so nothing looks only backwards. A test checks that this table and the code agree.

## Ball state

| column | meaning | rule |
|---|---|---|
| `ball_seen` | 1 if the ball was observed (after B1 rejection), else 0 | B1 |
| `ball_filled` | 1 if the position is a B3 interpolation inside a short straight gap | B3 |
| `ball_conf` | ball-stage confidence 0..1 (low for B2 suspects and fills) | B2, B3 |
| `ball_speed` | m/s from the ball stage's centred fit; NaN where missing | B5 |

## Ball-missing context (D12: missingness is evidence, read in context)

| column | meaning | rule |
|---|---|---|
| `ball_gap_len_s` | length of the missing run this frame is in (0 if seen or filled) | D12, D13 |
| `ball_t_since_seen_s` | seconds since the ball was last seen (0 when seen) | D12 |
| `ball_t_until_seen_s` | seconds until it is next seen (0 when seen) | D12 |
| `ball_last_seen_x`, `ball_last_seen_y` | position at the last seen frame (current position when seen) | D3, D12 |
| `ball_last_seen_vx`, `ball_last_seen_vy` | velocity at the last seen frame | D3, D12 |
| `ball_next_seen_x`, `ball_next_seen_y` | position at the next seen frame | T5, T6 |
| `ball_last_seen_dist_line_m` | distance from the last seen position to the nearest line (negative if outside) | D3 |
| `ball_last_seen_out_speed_mps` | last seen velocity component towards that nearest line (+ = heading out) | D3 |
| `ball_loss_players_3m_h`, `ball_loss_players_3m_a` | players of each team within 3 m of the ball on the last seen frame | D12, P25 |

## Contact (control versus proximity)

| column | meaning | rule |
|---|---|---|
| `near_d_h`, `near_d_a` | distance from the ball to the team's nearest player (seen frames only) | P1, P13 |
| `near_d_diff` | `near_d_h - near_d_a` | P13 |
| `near_n1_*`, `near_n2_*`, `near_n5_*` | players of the team within 1, 2, 5 m of the ball | P1, P15 |
| `near_vel_diff_h`, `near_vel_diff_a` | speed of the ball relative to the team's nearest player (m/s); small = moving together | P1 |
| `near_d_*_c05`, `near_d_*_c1` | centred means of `near_d_*` over ±0.5 s and ±1 s (NaN frames ignored) | P1 |
| `near_n2_*_c05`, `near_n2_*_c1` | centred means of `near_n2_*` | P15 |
| `crowd_3m` | players of both teams within 3 m of the ball | P15 |
| `n_players_h`, `n_players_a` | tracked players of the team on the frame | T7, D10 |

## Events

| column | meaning | rule |
|---|---|---|
| `in_flight` | 1 inside a `flight` event span | B5, P14 |
| `t_since_control_h_s`, `t_until_control_h_s`, `t_since_control_a_s`, `t_until_control_a_s` | seconds since / until the team's last / next `control` first touch | P1 to P5 |
| `t_since_exit_s`, `t_until_exit_s` | seconds since / until the last / next `exit` event | D1 to D3 |

## Team shape (T1, T3, T4, T7; `pipeline/features/shape.py`, appended after the columns above)

Every `_diff_` column is home minus away of a per-team statistic, so swapping the teams and
flipping the pitch negates it (mirror-symmetric by construction). Statistics are means, shares and
trimmed ranges, never counts (T7). A frame is valid when both teams have at least 6 tracked
players; invalid frames are NaN before windowing. Window suffix: `_c05`, `_c1`, `_c2`, `_c4` =
centred mean over ±0.5, 1, 2, 4 s, ignoring invalid frames (NaN if the whole window is invalid).

| column | meaning | rule |
|---|---|---|
| `shape_valid` | 1 if both teams have at least 6 tracked players on the frame, else 0 | T7 |
| `shape_valid_c1` | share of valid frames over ±1 s | T7 |
| `shape_fwd_vel_diff_*` | mean player velocity along the team's own attack direction (meta.json), home minus away, m/s | T3 |
| `shape_speed_diff_*` | mean player speed, home minus away, m/s | T3, T9 |
| `shape_depth_diff_*` | trimmed x range (90th minus 10th percentile), home minus away, m | T1, T7 |
| `shape_width_diff_*` | trimmed y range (90th minus 10th percentile), home minus away, m | T1, T7 |
| `shape_depth_rel_*`, `shape_width_rel_*` | the per-frame depth / width difference minus its median over the valid frames of the same half, then windowed (a team that is always wider is not a constant lean) | T1, T7 |
| `shape_press_diff_*` | share of the team's players within 10 m of the ball's last seen position (ball seen within 2 s, else NaN), home minus away | T4 |
| `shape_approach_diff_*` | mean speed towards that ball position of those players, home minus away, m/s | T4 |

## Loose ball (P30, P3, P7; proposal P-14; `pipeline/features/loose_ball.py`, appended after the team-shape columns)

Per frame and per team: the time the team's best-placed tracked player needs to reach the ball's
path over the next 1 s (Spearman 2018 arrival model with the reaction time set to 0 on the bench,
because noisy velocities make its extrapolation harmful: a straight run at 5 m/s from the current
position, ball played within 1 m; the ball is taken as held at the end of the path). The path is the tracked ball where it has a position; where it has none, a straight line
from the last position with its velocity (capped at 35 m/s). A frame needs a ball position or one
at most 0.5 s old, and at least 6 tracked players per team (T7, whole-scene player loss);
otherwise every loose-ball value is NaN and `lb_valid` is 0. Team times are capped at 3 s. Every
`_diff_` column is home minus away or an antisymmetric transform of it (swapping the teams negates
it). Window suffix: `_c0` = the frame, `_c05`, `_c1` = centred mean over ±0.5, 1 s ignoring
invalid frames.

| column | meaning | rule |
|---|---|---|
| `lb_valid` | 1 if both team times are defined on the frame, else 0 | P30, T7 |
| `lb_extrap` | 1 if valid and the ball's start position is extrapolated (ball unseen, at most 0.5 s since its last position) | P30, D12 |
| `lb_tti_h`, `lb_tti_a` | the team's time to the ball path, s, capped at 3 (NaN if invalid) | P30 |
| `lb_tti_diff_*` | home time minus away time, s (negative: home first) | P30 |
| `lb_pfirst_diff_*` | 2 P(home first) − 1, Spearman's logistic arrival uncertainty (σ 0.45 s), in [−1, 1] | P30 |
| `lb_clear_diff_*` | +1 home first by at least 0.545 s (P ≥ 0.9), −1 away first by that much, else 0 | P30, P7 |

## Play context (T-32 step 0; `pipeline/features/play_context.py`, appended after the loose-ball columns)

Team level only. `_diff` and `_rel_` columns are home minus away (swapping the teams and turning
the pitch negates them); the others are symmetric context. Windows are centred.

| column | meaning | rule |
|---|---|---|
| `shape_fwd_vel_diff_c6`, `shape_fwd_vel_diff_c8`, `shape_speed_diff_c6`, `shape_speed_diff_c8`, `shape_press_diff_c6`, `shape_press_diff_c8`, `shape_approach_diff_c6`, `shape_approach_diff_c8`, `shape_depth_rel_c6`, `shape_depth_rel_c8`, `shape_width_rel_c6`, `shape_width_rel_c8` | the team-shape statistics above with centred windows of ±6 and ±8 s | T1, T3, T4 |
| `lb2_tti_diff_*`, `lb2_pfirst_diff_*`, `lb3_tti_diff_*`, `lb3_pfirst_diff_*` | the loose-ball arrival model with a 2 s (`lb2`) and 3 s (`lb3`) ball path (team times capped at 4 and 5 s); `_c05`, `_c1` windows | P30 |
| `pc_touch_diff_b1`, `pc_touch_diff_b2`, `pc_touch_diff_b4` | home minus away `touch` events in frames [t − w, t], w = 1, 2, 4 s | P1, P15 |
| `pc_touch_diff_a1`, `pc_touch_diff_a2`, `pc_touch_diff_a4` | the same in (t, t + w] | P1, P15 |
| `pc_touch_n_c1` | touches of both teams within ±1 s (symmetric) | P15 |
| `pc_near2_diff`, `pc_near5_diff`, `pc_near10_diff` | players within 2, 5, 10 m of the positioned ball, home minus away (NaN without a ball position) | P15 |
| `pc_near2_diff_c1`, `pc_near5_diff_c1`, `pc_near10_diff_c1` | their ±1 s means | P15 |
| `pc_crowd5` | players of both teams within 5 m of the ball | P15 |
| `pc_duel` | 1 when each team's nearest player is within 3 m of the ball and the two are within 2 m of each other | P13, P18 |
| `pc_duel_close` | closing speed of that pair inside a duel, m/s (NaN elsewhere) | P18 |
| `pc_duel_lead_diff_c05` | inside a duel, away minus home distance to the ball (clipped ±3 m; positive: home nearer), 0 elsewhere; ±0.5 s mean | P13 |
| `pc_gap_last_near_diff` | inside a ball gap: nearest-player distance away minus home (clipped ±10 m) at the last positioned frame, times the share of the gap still to come; 0 outside gaps | D12, T5 |
| `pc_gap_next_near_diff` | the same at the next positioned frame, times the share of the gap elapsed | D12, T6 |
| `pc_gap_next_n5_diff` | home minus away players within 5 m at reappearance, times the share elapsed | D12, T6 |
| `pc_zone` | distance of the ball from halfway as a share of half the pitch length (symmetric) | S |
| `pc_box_crowd` | players of both teams inside the penalty area at the ball's end (symmetric) | R |
| `pc_line_rel_c1`, `pc_line_rel_c4` | defensive line: mean distance from own goal line of the 2nd to 5th deepest players, home minus away, minus its half median; ±1, ±4 s means | T2 |
| `pc_linevel_diff_c1`, `pc_linevel_diff_c2` | mean velocity of those players along their attack direction, home minus away, m/s | T2 |

## Players near the ball (T-32 step 1; `pipeline/features/players_near_ball.py`, appended last)

The K = 4 nearest players of each team to the ball, ranked on each frame (rank k is not a person).
Ball = positioned ball, else its last position held for at most 0.5 s, else none (NaN). Player
values are in the team's own attack frame, so swapping the teams and turning the pitch exchanges
the `h` and `a` blocks and negates the four `_xh`, `_yh`, `_vxh`, `_vyh` ball columns.

| column | meaning | rule |
|---|---|---|
| `pnb_h*_d`, `pnb_a*_d` | distance of the team's k-th nearest player to the ball, m (`pnb_h1_d` ... `pnb_a4_d`) | P13, P15 |
| `pnb_h*_dx`, `pnb_a*_dx`, `pnb_h*_dy`, `pnb_a*_dy` | player minus ball along / across the team's attack direction, m | P13 |
| `pnb_h*_vto`, `pnb_a*_vto` | speed towards the ball, m/s (+: closing) | P30 |
| `pnb_h*_vatt`, `pnb_a*_vatt` | velocity along the team's attack direction, m/s (+: advancing) | T3 |
| `pnb_h*_brk`, `pnb_a*_brk` | braking of the k-th nearest player over ±0.2 s, m/s² (player followed by position continuity; T-32 variant S1g2) | P30 |
| `pnb_h*_trn`, `pnb_a*_trn` | turn of that player's velocity over ±0.2 s, rad (0 when slow); a proxy for a change of body orientation | P30 |
| `pnb_ball_ok`, `pnb_ball_seen`, `pnb_ball_speed` | ball position available (held ≤ 0.5 s), ball seen, ball speed m/s (symmetric) | B5, D12 |
| `pnb_ball_xh`, `pnb_ball_yh`, `pnb_ball_vxh`, `pnb_ball_vyh` | ball position from the centre and velocity, in the home team's attack frame | S |

## Not built yet

Team shape T2, T5, T6 and T9 (back-line height, share of players walking, convergence point). `meta.json` of bench copies made by
`bench.noise` has no `attack_direction`, which the attack-axis features will need.
