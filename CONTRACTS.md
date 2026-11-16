# Stage contracts

Every stage reads and writes these files, with the same layout for every bench game and for the
match, so stages can be built in parallel and swapped one at a time. Checked against the code on
14 Nov 2026 (T-30); where a column list below disagrees with the code, the code wins and this file
is wrong: fix it.

`<root>` is `artifacts/match/` for the competition match (agents never read it) and
`artifacts/bench/<game_id>/` for an external game (agents may read it). All frames are video
frames at 25 fps, counting from 0. Everything under `artifacts/` is gitignored.

## Root layout

| Folder | What it is |
|---|---|
| `artifacts/match/` | The competition match. Stages write here; agents never read it |
| `artifacts/bench/metrica_g<k>/` | Bench game k (1 to 3): clean tracking from `bench.metrica_io` plus the truth files below |
| `artifacts/bench/metrica_g<k>_<level>/` | The same game with simulated tracking noise, `<level>` = `low`, `match` or `high` (the level called `match` is the match-like noise level; the `legacy*` roots are the old noise, kept only for reference). The pipeline stages and the scores run on these roots |
| `artifacts/bench/_scores/` | `<version>_<level>_<truth>.json`, written by `bench.score` |

## Files by stage

| Stage | Owner | Writes | Shape |
|---|---|---|---|
| Load | data-plumber | `<root>/players.parquet` | frame, team (0 = H, 1 = A), x_m, y_m, vx, vy (one row per tracked player per frame, slot dropped after dedupe) |
| Load | data-plumber | `<root>/meta.json` | game_id, source, n_frames, fps, halves `[[start, stop), ...]`, attack_direction, attack_direction_agreement, attack_direction_note, pitch `[length, width]`, coordinates, frame_convention, n_track_slots, slots_per_team, merge_radius_m, velocity_half_width_frames, max_speed_mps, ball_columns, player_extra_channels. Bench level roots written by `bench.noise` also carry `graded_spans` and a `noise` block |
| Load | data-plumber | `<root>/ball_raw.parquet` | frame (every video frame 0..n_frames-1), x_m, y_m, seen (bool: x and y finite), plus each extra ball channel as delivered: `z_m` if a height channel exists, otherwise `raw_<channel>` (the match has `raw_s`, ball speed, unit **kph** per the ATD metadata; unusable, not read). Uncleaned; NaN where unobserved. Input to the Ball stage |
| Ball | signal-engineer | `<root>/ball.parquet` | frame, x_m, y_m, vx, vy, seen (bool), filled (bool), conf (0 to 1) |
| Events | signal-engineer | `<root>/events.parquet` | frame, kind, team, conf, end_frame, first_touch_frame, confirmed_frame, line, x_m, y_m, last_touch_team, last_touch_frame, goal_mouth, detail (14 columns; `pipeline/events/base.py` has the dtypes; integer columns use -1 for "not applicable"). `kind` is touch, flight, carry, pass, deflection, control or exit. `keeper_gather` is named in the code's docstring but no code writes it yet (T-14). `end_frame` is set on flight, carry and pass; `first_touch_frame` and `confirmed_frame` on control; `line`, `last_touch_team`, `last_touch_frame`, `goal_mouth` on exit; `detail` says how the row was found (for exit: persist, vanish or extrapolated; for touch: includes the velocity change `dv=`). The touch detector runs with a named parameter set (`pipeline/events/touch.py` `PARAM_SETS`: `w7`, the default, and `w15s` for a low-pass ball), chosen with `--touch-set` on the stage command line and reported as `touch_param_set`; the file does not record the set. In `make candidate` the set is the version's (`VERSIONS[v]["touch_set"]`, see "Touch parameter set per version" below) |
| Features | signal-engineer | `<root>/features.parquet` | frame plus float32 feature columns, one row per frame. Four groups of columns are documented in `pipeline/features/FEATURES.md` and a test checks that the file and the code agree: ball state, ball-missing context, contact and events (the original 47 columns); team shape (34 columns appended by `features/shape.py`, T-23); loose ball (`lb_*`, appended by `features/loose_ball.py`, T-28). Shape columns are home minus away and so change sign when the teams swap; windows are centred (`_c05`, `_c1`, `_c2`, `_c4` seconds) |
| Restarts | restart-engineer | `<root>/restarts.parquet` | one row per stoppage candidate, 15 columns: start_frame (-1 = not known yet, e.g. goal kick-offs until D4 exists), restart_frame, start_kind (half_start, exit, gap, slowdown, goal), restart_type (kick_off, throw_in, corner, goal_kick, free_kick, goal_kick_or_corner when the goal-line exit has no last touch and no place), p_home_restart (0.5 = undetermined; never defaulted to a team), conf, half, formation_start, formation_end (-1 if none), kick_team_llr (+ favours H), evidence (formation, formation_ball, assumed_live_before, assumed_none_found, published_anchor, and the merged sources exit, gap, slowdown joined by `+`), team_llr (restart-team log-odds, + favours H; equals kick_team_llr on kick-off rows), type_source (how type and restart frame were decided), exit_line, last_touch_team (-1 if none). The kick-off detector's own rows also carry law_kick_team_h1 (0 H, 1 A, -1 undetermined; S6). `published_anchor` appears on the match only, from `games.py`. Kick-off rows after goals take start_frame from the merged stoppage |
| Dead-ball features | restart-engineer | `<root>/features_dead.parquet` | frame, half, form_kick_off (0 to 1), mask_pre_kickoff (bool, S4: every frame of a half before its kick-off; the decoder writes D there), p_restart_kick_off; then, from the stoppage code, p_exit, p_gap_stop, p_slowdown (conf at the stoppage start frame of that source, else 0), p_restart (conf at every restart frame, kick-offs included), p_dead_candidate (max conf of the stoppage candidates covering the frame), keeper_hold (bool, P22), form_throw_in, form_corner, form_goal_kick, form_free_kick (0 to 1, centred 1 s share). The restarts stage writes it only when `ball.parquet` exists on the root; bench level roots do not get it written, the rule scorer and decoder compute the same columns in memory there |
| Rules | restart-engineer | `pipeline/rules/registry.yaml` | id, tier, encoding, statement, function, params, status (planned, implemented, measured), bench_precision, bench_coverage. Edited by hand |
| Tags | restart-engineer | `<root>/tags.parquet` | one row per video frame (n_frames rows), 20 columns: frame (int32); tag_vote (float32: sign of the unweighted sum of the per-code votes, +1 favours H, -1 A, 0 none; G1, G2, G4, G9); twelve per-code votes `vote_<code>` (int8, +1 H, -1 A, 0 none; codes build_up, progression, creating_chances, ball_in_final_third, ball_in_the_box, players_in_the_box, long_ball (first second only), defending_in_attacking_third, defending_in_middle_third, defending_in_defensive_third (vote for the other team), attacking_transition, defensive_transition (other team); opposite instances of one code cancel to 0); turnover_hint_to (int8: 0 H, 1 A, -1 none or both; the 2 s before each transition tag start, G4); restart_window (bool, each SET PIECES interval, G5 / L5); restart_type (string: throw_in, corner, goal_kick, free_kick, kick_off, other; '' outside a window); restart_team (int8 0 H, 1 A, -1 none: every throw-in, or overlapping windows that disagree; R14); goal_hint (bool, each GOALS interval widened by 2 s each side, G6); goal_conceding_team (int8, the other team than the GOALS tag's, -1 none; R13). Written by `python -m pipeline.tagging match` (T-12a) on the match only; a root without a tagging file (every bench root) gets "no tags" and no file. Teams map through `pipeline/tagging/teams.py`. `--agreement --ref-version V` / `--ref pseudo` prints the G10 table and writes nothing. No decoder stage reads this file yet (T-12c) |
| Bench truth | bench-builder | `artifacts/bench/<game_id>/` | Four label files in the submission format (frame, label), all derived by code from the provider's event data (`bench-truth` skill): `truth_first_touch.csv` (switch dated at the first touch of the new team's controlling sequence), `truth_confirmed.csv` (dated at the first controlled action), `truth.csv` (equal to `truth_first_touch.csv`; this is what `--truth default` reads), and `truth_labels.csv` (the organisers' conventions C1 to C9 as read by the reviewer, T-26; what `--truth labels` reads and the primary truth for verdicts). They share every D frame and every restart; only the owner during live play differs. Also `stoppages.csv` (the stoppages used for D, with restart team and type), `truth_meta.json` (graded spans, halves, counts) and `truth_labels_meta.json`. `bench.noise` copies all truth files to every level root |
| Probabilities | frame-modeler | `<root>/probs/<model>.npy` | float32 (n_frames, 3) = P(H), P(A), P(D), outside the halves (1/3, 1/3, 1/3). Models: `rule_v1` (rule scorer, `models/rule_scorer.py`: dead and team terms, weights fitted on the bench); `rule_v2` (team model with player-shape columns, `models/team_model.py`, weights in `team_v2_params.json`, fitted on first-touch truth; keeps rule_v1's P(D) column exactly). `rule_w1` and `rule_w2` (T-35, `team_w1_params.json`, `team_w2_params.json`, written by `models/refit_w.py`) are rule_v1's terms and rule_v2's columns refitted on the smoothed bench with `w15s` events; a params file may carry an optional `"dead"` block, which gives the model its own P(D) (both do; models without the block keep rule_v1's). `rule_p1g` (T-32, `team_p1g_params.json`, `"kind": "trees"`, written by `models/fit_t32.py write1 --variant S1g --gate G4`): per-player gradient-boosted trees (`models/player_model.py`) used inside the busy gate G4 only, base model `rule_w2` (its P(D) and its team probability outside the gate). `rule_v5` is declared in code but has no params file and is never written. The probs2 step writes every model with a tracked params file. Bench roots of a fitted game use weights fitted without that game (leave-one-game-out); the match uses weights fitted on all three bench games |
| Labels | decoder-engineer | `<root>/labels/<version>.csv` | submission format: columns frame, label, one row per frame, H, A, D, or X outside the graded spans, `\n` line endings. The file is named after the version and decodes that version's own model (`VERSIONS[v]["probs"]`, default rule_v1). With an explicit `--probs` that is not rule_v1 the name is `<version>_<model>` (so `v6s_rule_v2.csv` equals `v6s.csv`) |
| Provenance stamp | data-plumber | `<root>/labels/<version>.provenance.json` | written by `make candidate` for the match after the decode: version, code_hash (sha256 over path and content of every git-visible file under `pipeline/`, `bench/`, `starter/`, the three official evaluation scripts, `games.py`, `pyproject.toml`, `uv.lock` and `Makefile`), code_files (count), labels_sha256, probs (the probabilities model), touch_set (the touch parameter set the events were built with, T-36). `make submission` and `make check-labels` refuse labels with no stamp, with changed labels, with a probs or touch_set other than the version now declares (`probs_differs_from_version`, `touch_set_differs_from_version`; a stamp written before T-36 has no touch_set and is refused), or with a code hash that differs from the checkout |
| Submission | data-plumber | `submissions/<team_name>.csv` | the official format, validated. Gitignored |
| Bench scores | bench-builder | `artifacts/bench/_scores/<version>_<level>_<truth>.json` | `bench.score` output: per game and pooled spellF1 with bootstrap interval, short-possession recall, false splits per hour, stoppages, restart-team accuracy, possessions per minute. `<truth>` is `default`, `first_touch`, `confirmed` or `labels` |
| Diagnostics | principal-data-scientist | `reports/<module>.json` | aggregates only, written by `summary()` from `python -m pipeline.diagnostics.<name>`; agents may read these |

Load-stage details (pipeline/io/load.py): `players.parquet` is sorted by frame, then team;
float32 values; metres are the 0 to 1 coordinates times the pitch (match: `games.py` 105.3 x 68.0;
bench: the metadata FieldSize), axes in the provider's orientation; vx, vy in m/s from a centred
difference over +-2 frames, NaN at gaps and above 12.5 m/s; same-team slots closer than 0.5 m on a
frame are merged (mean). `attack_direction` is `{"<half>": {"0": +-1, "1": +-1}}` (+1 = attacks
towards increasing x_m) and `attack_direction_agreement` is the share of frames agreeing. Match
halves come from `games.half_video_blocks`, bench halves from the EPTS provider parameters.

## Stage command line

Every stage module answers `python -m pipeline.<stage> match` and
`python -m pipeline.<stage> bench [--game <id> ...]`; that is what `make match STAGE=<stage>` and
`make bench STAGE=<stage>` call. Bench games are folders `bench_data/<game_id>/` holding one EPTS
metadata `*.xml` (`*metadata*` preferred) and one tracking `*.txt` (`*tracking*` preferred). The
Metrica sample games are fetched by `make bench-data` (`pipeline/io/fetch_bench.py`, pinned to one
commit of the sample-data repository, sha256 per file). Stages that run on level roots also accept
`--root <dir> ...`. Match code wraps its entry point in `pipeline.report.match_entry` and prints
only through `pipeline.report.summary`.

Stage names for `make match STAGE=` and `make bench STAGE=`: `load`, `ball`, `events`, `features`,
`restarts`, `probs` (module `models.rule_scorer`, writes `rule_v1.npy`), `probs2` (module
`models.team_model`, writes the team models), `decode` (needs `VERSION=`). `make bench` also takes
`ROOT="dir ..."` (those roots only). `STAGE=events` takes `TOUCH_SET=<set>` (passed as
`--touch-set`; default `w7`); `TOUCH_SET` with any other stage stops with an error.

## Touch parameter set per version (T-36)

A version's events are built with one touch parameter set, declared once in its entry in
`pipeline/decode/decoders.py`, next to `probs`:

    VERSIONS["<v>"] = {..., "probs": "rule_v2", "touch_set": "w15s"}

No key means `pipeline.events.touch.DEFAULT_SET` (`w7`); every version before T-35 has no key and
builds exactly as before. `pipeline.candidate.touch_set(v)` reads it and stops the run before any
stage if the name is not in `PARAM_SETS`. `make candidate` passes it to the events stage on the
match and on the bench level roots and rebuilds every later stage (features, restarts, probs,
probs2, decode) in the same run, in order; there is no command-line override for a version. The
stage files do not record the set: after a `w15s` version's run, the level roots (and the match
root) hold `w15s` features and probabilities, so decode or score another version on them only
through its own `make candidate` run. Labels files and their stamps are per version and are not
affected. `make bench-stages TOUCH_SET=<set> [LEVELS=...]` builds ball, events, features, probs
and probs2 on the level roots for a named set without decoding; `make bench-build` always builds
`w7`. `make rebuild-check` needs nothing extra: the clone's `make candidate` reads the same
declaration from the cloned code.

## Make targets (owned by data-plumber)

| Target | Does | Prints |
|---|---|---|
| `make data-check` | the nine match files are present with the published sizes | counts |
| `make match STAGE=<stage>` | runs one stage on the match | aggregates only |
| `make bench STAGE=<stage> [GAME=<id>] [ROOT="dir ..."] [TOUCH_SET=<set>]` | runs one stage on every bench game, or on the given roots; `TOUCH_SET` with `STAGE=events` only | anything |
| `make bench-stages [TOUCH_SET=<set>] [LEVELS=...]` | ball, events (that set, default `w7`), features, probs, probs2 on the level roots; no decode, no scores | stage timings |
| `make bench-data` | fetches and verifies the Metrica sample games (pinned commit, sha256) | file counts |
| `make bench-build` | `bench-data`, then dead-ball check, truth, noise at the three levels, ball, events (`w7`), features and rule_v1 probs on every level root, and the v0 baseline per level | scores |
| `make candidate VERSION=<v> [PART=all\|match\|bench] [LEVELS="low match high"]` | the full chain for a version, events with the version's touch set. Match: load, ball, events, features, restarts, probs, probs2, decode, then the provenance stamp (with probs and touch_set). Bench: ball, events, features, probs, probs2 on the level roots, decode, `bench.score` under the `default` and `confirmed` truths (the `labels` truth is not in this chain yet; score it with `bench.score --truth labels`). A bench run needs `make bench-build` first | bench score table, match aggregates |
| `make judge VERSION=<v> BASE=<v> [SCORE=0] [DRAWS=2000]` | `bench.score` for both versions under the default, confirmed and labels truths (skipped with `SCORE=0`), then the paired bootstrap and error counts of `bench/paired.py`, written to `artifacts/bench/_judge/<v>_vs_<base>.json`; bench only | numbers |
| `make check-labels VERSION=<v>` or `TEAM=<name>` | output checks on existing labels, no re-decode | aggregates |
| `make compare-labels A=<v> B=<v> PART=match\|bench [LEVELS=...]` | two versions' label files: possessions and their ratio, shares of H/A/D, agreement, stoppages (`pipeline/compare_labels.py`) | aggregates |
| `make gate-share VERSION=<v> PART=match\|bench [LEVELS=...]` | share of graded time a gated per-player tree version (e.g. v11g) uses its trees (`pipeline/diagnostics/gate_share.py`) | aggregates |
| `make submission VERSION=<v> TEAM=<name>` | builds `submissions/<name>.csv` from stamped labels (refuses stale labels) and runs the official validator | validator output |
| `make rebuild-check VERSION=<v> TEAM=<name> [SOURCE=head\|worktree] [BENCH=1] [REPEAT=1] [KEEP=1]` | rebuilds the submission in a clean clone and compares bytes | match or mismatch |
| `make baseline`, `make score` | reproduce the organisers' baseline; official scorer on everything in `evaluation/results/` | scores |
| `make atd-audit` | channel audit of the match's two ATD files | aggregates |
| `make test` | unit tests (`tests/pipeline`); bench tests are `uv run python -m pytest tests/bench` | pass or fail |
| `make lock` | refresh `uv.lock` and `requirements.lock.txt` | none |
