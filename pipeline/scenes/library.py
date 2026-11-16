"""Ready-made scenes with known answers (``scene.expected``). Pitch 105 x 68, 25 fps.

Home (team 0) attacks +x, away (team 1) attacks -x. Every function returns a ``Scene``; the
docstring says what happens and the expected events are recorded with ``scene.expect``. Frame
numbers in the expectations are the scripted instants; detectors are checked within a small
tolerance in the tests.

| function | what happens | key answers |
|---|---|---|
| pass_scene | H dribbles, passes 17 m, H team-mate receives and carries | touch H x2, flight, pass H, carry H |
| carry_scene | H dribbles 4 s, opponents 4 m+ away | carry H, no A touch |
| deflection_scene | H pass clipped by an A defender, ball runs on at 16 m/s to H | touch A, deflection A, no control A |
| exit_scene(line) | ball kicked over ``line``, seen out, then lost | exit on that line, last touch team |
| vanish_exit_scene | ball heading out fast, lost 2 m before the line | extrapolated exit (D3) |
| projection_trap_scene | high ball drawn beyond the far touchline, back in at speed | no exit (A15) |
| flight_gap_scene(gap) | pass with the ball unseen ``gap`` frames in flight | filled iff gap <= 12 |
| interception_scene | H pass cut out by an A player who carries away | touch A, carry A, control A |
| dribble_push_scene | H pushes ball 8 m ahead past a still A defender | no A touch, no A control (P13) |
| header_duel_scene | H long ball headed by A, loose, H picks it up | touch A, no control A (A8) |
| tackle_recovery_scene | A tackle knocks ball loose, H team-mate recovers | touch A, no control A (A9) |
| false_marking_scene | "ball" still on a penalty mark, players far | B2 suspect, low conf |
| teleport_scene | one-frame 20 m jump in a rolling ball | the jump frame rejected (B1) |
"""
from pipeline.scenes import Scene

L, W = 105.0, 68.0


def _extras(sc, home=((20, 20), (25, 50)), away=((85, 15), (80, 55))):
    """Far-away bystanders so each team has several tracked players."""
    for x, y in home:
        sc.player(0, [(0, x, y)])
    for x, y in away:
        sc.player(1, [(0, x, y)])


def pass_scene():
    sc = Scene(150)
    p = sc.player(0, [(0, 40, 34), (40, 43.2, 34)])
    r = sc.player(0, [(0, 59.5, 40), (65, 59.5, 40), (125, 65.5, 40), (149, 67.9, 40)])
    sc.player(1, [(0, 50, 20)])
    sc.player(1, [(0, 72, 45)])
    _extras(sc)
    sc.ball_with(p, 0, 40)
    sc.ball([(40, None), (65, 60.0, 40.0)])
    sc.ball_with(r, 66, 149)
    sc.expect("touch", 40, team=0)
    sc.expect("touch", 65, team=0)
    sc.expect("flight", 45, team=0)
    sc.expect("pass", 40, team=0, end=65)
    sc.expect("carry", 70, team=0)
    sc.expect("control", 65, team=1, absent=True)
    return sc


def carry_scene():
    sc = Scene(125)
    p = sc.player(0, [(0, 30, 30), (100, 46, 34), (124, 49, 35)])
    sc.player(1, [(0, 40, 26), (124, 50, 29)])     # 4+ m away, shadowing
    _extras(sc)
    sc.ball_with(p, 0, 124)
    sc.expect("carry", 20, team=0)
    sc.expect("touch", 60, team=1, absent=True)
    sc.expect("control", 60, team=1, absent=True)
    return sc


def deflection_scene():
    sc = Scene(150)
    p = sc.player(0, [(0, 30, 34), (20, 31.6, 34)])
    sc.player(1, [(0, 45, 34.9)])                  # defender on the line of the pass
    r = sc.player(0, [(0, 61, 38.5), (58, 61, 38.5), (149, 67, 38.5)])
    _extras(sc)
    sc.ball_with(p, 0, 20)
    # Pass at ~25 m/s along y = 34: reaches the defender at x = 45 by frame 33.
    sc.ball([(20, None), (33, 45.0, 34.0)])
    # Toe-poke: heading turns 16 degrees, speed drops to 16 m/s, on to the H receiver.
    sc.ball([(33, 45.0, 34.0), (58, 60.5, 38.5)])
    sc.ball_with(r, 59, 149)
    sc.expect("touch", 33, team=1)
    sc.expect("deflection", 33, team=1)
    sc.expect("control", 33, team=1, absent=True)
    return sc


_EXITS = {
    # line: (kicker team, kicker start, ball target beyond the line, goal_mouth)
    "touch_y0": (0, (50, 8), (52, -6), False),
    "touch_y1": (1, (50, 60), (51, 74), False),
    "goal_x0": (1, (10, 34), (-4, 35), True),
    "goal_x1": (0, (95, 20), (109, 22), False),
}


def exit_scene(line="touch_y0"):
    """Ball kicked over ``line`` at about 14 m/s, seen 10 frames outside, then lost."""
    team, (kx, ky), (tx, ty), mouth = _EXITS[line]
    sc = Scene(200)
    k = sc.player(team, [(0, kx - 2 if line != "goal_x0" else kx + 2, ky), (30, kx, ky)])
    sc.player(1 - team, [(0, kx + 6, ky + 4 if ky < 34 else ky - 4)])
    _extras(sc)
    sc.ball_with(k, 0, 30)
    sc.ball([(30, None), (55, tx, ty)])          # 25 frames, ~14 m: ~14 m/s
    # Seen beyond the line for 10 frames after crossing, then lost for good.
    import numpy as np
    b = sc.ball_xy()
    out = np.flatnonzero((b[:, 0] < -0.3) | (b[:, 0] > L + 0.3) | (b[:, 1] < -0.3)
                         | (b[:, 1] > W + 0.3))
    first_out = int(out[0])
    sc.hide_ball(first_out + 10, 199)
    sc.expect("exit", first_out, team=None, line=line, last_touch_team=team, goal_mouth=mouth)
    return sc


def vanish_exit_scene():
    """Ball kicked towards touchline y = 0 at 15 m/s, lost 2 m before it (D3)."""
    sc = Scene(200)
    k = sc.player(0, [(0, 50, 12), (20, 51, 12)])
    sc.player(1, [(0, 58, 18)])
    _extras(sc)
    sc.ball_with(k, 0, 20)
    sc.ball([(20, None), (40, 52.0, -1.5)])      # ~13 m in 0.8 s
    import numpy as np
    b = sc.ball_xy()
    lost = int(np.flatnonzero(b[:, 1] < 2.0)[0])
    sc.hide_ball(lost, 160)
    sc.ball([(161, 52.0, 0.4), (199, 52.0, 0.4)])  # reappears at the throw-in spot
    sc.expect("exit", lost + 3, line="touch_y0", last_touch_team=0, detail="extrapolated")
    return sc


def projection_trap_scene():
    """H long ball towards the far touchline drawn up to 2.5 m outside for ~12 frames."""
    sc = Scene(175)
    p = sc.player(0, [(0, 38, 48), (30, 40, 50)])
    r = sc.player(0, [(0, 70.5, 64), (175, 70.5, 64)])
    sc.player(1, [(0, 60, 55)])
    _extras(sc)
    sc.ball_with(p, 0, 30)
    sc.ball([(30, None), (45, 55, 70.5), (52, 62, 70.0), (65, 70.0, 64.0)])
    sc.ball_with(r, 66, 174)
    sc.expect("exit", 44, absent=True)
    return sc


def flight_gap_scene(gap=8):
    """The pass of pass_scene with the ball unseen for ``gap`` frames mid-flight."""
    sc = pass_scene()
    start = 52 - gap // 2
    sc.hide_ball(start, start + gap - 1)
    sc.gap = (start, start + gap - 1)
    return sc


def interception_scene():
    sc = Scene(150)
    p = sc.player(0, [(0, 38, 34), (30, 40.4, 34)])
    sc.player(0, [(0, 62, 34)])                                  # intended receiver
    a = sc.player(1, [(0, 50, 37.5), (45, 50, 35.0), (120, 40, 32), (149, 36, 31)])
    _extras(sc)
    sc.ball_with(p, 0, 30)
    sc.ball([(30, None), (45, 49.5, 34.9)])                      # ~9 m in 0.6 s
    sc.ball_with(a, 46, 149)
    sc.expect("touch", 45, team=1)
    sc.expect("carry", 60, team=1)
    sc.expect("control", 45, team=1, first_touch=45)
    return sc


def dribble_push_scene():
    """P13: dribbler pushes the ball 8 m ahead past a still defender who never touches it."""
    sc = Scene(125)
    d = sc.player(0, [(0, 40, 30), (124, 74.7, 30)])            # 7 m/s
    sc.player(1, [(0, 51, 31.4)])                                # defender, 1.4 m off the line
    _extras(sc)
    sc.ball_with(d, 0, 25, ahead=0.5)
    sc.ball([(25, None), (40, 54.0, 30.0), (55, 55.9, 30.0)])   # push, then ball slows
    sc.ball_with(d, 56, 124, ahead=0.5)
    sc.expect("touch", 33, team=1, absent=True)
    sc.expect("control", 33, team=1, absent=True)
    sc.expect("carry", 10, team=0)
    return sc


def header_duel_scene():
    """A8: H long ball, A defender heads it loose, an H midfielder picks it up and carries."""
    sc = Scene(200)
    p = sc.player(0, [(0, 28, 34), (20, 30, 34)])
    sc.player(0, [(0, 70.3, 29.0)])                              # H attacker at the landing
    sc.player(1, [(0, 69.8, 30.3)])                              # A defender heads it
    m = sc.player(0, [(0, 82, 46), (95, 77.5, 41), (199, 85, 42)])
    _extras(sc, away=((85, 15), (60, 55)))
    sc.ball_with(p, 0, 20)
    sc.ball([(20, None), (70, 70.0, 30.0)])                      # 40 m in 2 s
    sc.ball([(70, 70.0, 30.0), (95, 77.0, 40.5)])                # loose, ~5 m/s, nobody near
    sc.ball_with(m, 96, 199)
    sc.expect("touch", 70, team=1)
    sc.expect("control", 70, team=1, absent=True)
    return sc


def tackle_recovery_scene():
    """A9: A tackles an H carrier, the ball squirts loose, an H team-mate recovers it."""
    sc = Scene(175)
    c = sc.player(0, [(0, 40, 34), (50, 48, 34), (174, 48, 34)])
    sc.player(1, [(0, 52, 38), (50, 48.6, 34.2), (174, 49, 35.5)])   # tackler, foot on the ball
    t = sc.player(0, [(0, 46, 25), (70, 47.2, 28.7), (174, 56, 28)])  # recovering team-mate
    _extras(sc)
    sc.ball_with(c, 0, 50)
    sc.ball([(50, None), (70, 47.6, 29.0)])                      # 5 m in 0.8 s, sideways
    sc.ball_with(t, 71, 174)
    sc.expect("touch", 50, team=1)
    sc.expect("control", 50, team=1, absent=True)
    return sc


def false_marking_scene():
    sc = Scene(200)
    sc.player(0, [(0, 40, 34), (199, 60, 30)])
    sc.player(1, [(0, 45, 30), (199, 62, 33)])
    _extras(sc)
    sc.ball([(0, 11.0, 34.0), (199, 11.0, 34.0)])
    sc.expect("suspect", 100)
    return sc


def teleport_scene():
    sc = Scene(100)
    sc.player(0, [(0, 30, 30), (99, 38, 30)])
    _extras(sc)
    sc.ball([(0, 31, 30), (99, 45, 30)])
    sc._ball[50] = (60.0, 45.0)
    sc.expect("rejected", 50)
    return sc


ALL = {
    "pass": pass_scene, "carry": carry_scene, "deflection": deflection_scene,
    "vanish_exit": vanish_exit_scene, "projection_trap": projection_trap_scene,
    "interception": interception_scene, "dribble_push": dribble_push_scene,
    "header_duel": header_duel_scene, "tackle_recovery": tackle_recovery_scene,
}
