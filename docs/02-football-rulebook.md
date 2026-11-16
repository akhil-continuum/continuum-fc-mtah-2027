# Football rulebook for the possession model

This is the single source of football logic for the pipeline. The reviewer edits it in plain
English; the agents turn each rule into code and tests (Part 2 says how). Nothing here was taken
from the match data. Every number is a starting value to be set on other matches.

## How to read a rule

Each rule has an ID, a statement, a confidence tier and one or more encodings.

| Tier | Meaning | How it is used |
|---|---|---|
| **LAW** | Follows from the Laws of the Game or from the challenge text. Treated as certain. | Hard constraint in the decoder, or a mask |
| **STRONG** | True in the large majority of cases (aim: 90 percent or more on other matches). | Pseudo-label, or a large cost |
| **CLUE** | Leans one way. | Feature for the frame model, or a small cost |
| **OPEN** | We do not know which convention the ground truth uses. | Reviewer answers from the public window (Q-numbers refer to `03-reviewer-playbook.md`) |

| Encoding | Meaning |
|---|---|
| MASK | Label forced on a set of frames |
| STATE | Built into the decoder's states and allowed transitions |
| EVENT | A detector that emits a timed event |
| PSEUDO | A rule that labels confident frames for self-training |
| COST | Makes a transition or label cheaper or dearer at certain frames |
| FEAT | A per-frame input to the frame model |
| SNAP | Moves a decoded boundary to an event |
| CHECK | An automatic test on the output; failing it blocks a version |

Pitch constants (metres): pitch 105.3 × 68.0; penalty area 16.5 deep × 40.32 wide; goal area 5.5 ×
18.32; penalty mark 11 from the goal line; centre circle and free-kick distance 9.15; goal 7.32
wide; corner arc 1; ball radius 0.11.

---

# Part 1. The rules

## S. Structure of the answer

| ID | Rule | Tier | Encoding |
|---|---|---|---|
| S1 | Every graded frame is H, A or D. There is no contested or unknown state. | LAW | STATE |
| S2 | While the ball is live exactly one team has it. A loose ball belongs to the team clearly about to collect it (P30); where none is clear it stays with the last team that had it. (reviewer convention, 14 Nov) | LAW | STATE (holding is free) |
| S3 | A possession never crosses half-time. Each half is decoded on its own. | LAW | STATE |
| S4 | The second half is graded from before its kick-off. Every frame until that kick-off is D. | LAW | MASK from a detected kick-off; CHECK against the published frame |
| S5 | The first half starts live at its kick-off, with the kicking-off team in possession. | LAW | STATE |
| S6 | The team that did not kick off the first half kicks off the second. | LAW | STATE (derive the first-half taker in code) |
| S7 | Teams change ends at half-time. Attack direction flips. | LAW | Derived per half from the keepers' positions |
| S8 | Dead ball has no owner. D is D whoever restarts. | LAW | Two hidden dead states both write D |
| S9 | Possessions can be shorter than one second: a deliberate play of the ball gives possession however briefly (P21). The labels are frame-accurate, not smoothed to phases of play, so a version that suppresses sub-second possessions disagrees with the definition. There is no minimum of a second or more. (reviewer convention, 14 Nov) | LAW (from the definition) | Minimum length is a small bench-chosen value |
| S10 | A stoppage is never shorter than the time it takes to restart. Start value: 2 s. | STRONG | STATE (minimum dead duration) |
| S11 | Expected totals for a full match: 35 to 50 percent of graded frames dead; 90 to 110 stoppages; 3 to 5 possessions per minute of graded time. | CLUE | CHECK (warn outside the range, do not tune to it) |

## R. Restarts: who has the ball when play resumes

**R0 (LAW, STATE).** The first live frame after any stoppage belongs to the team taking the
restart. The decoder has two dead states, `D→H` and `D→A`; each may only exit to its own team.

| ID | Restart | Who takes it | Ball becomes live when | What it looks like in tracking |
|---|---|---|---|---|
| R1 | Throw-in | Opponents of the last team to touch the ball | It is released and enters the pitch | Ball left over a touchline. One player at or just outside that line, near the exit point, almost still. Ball usually untracked until it reappears inside, moving away from the line |
| R2 | Goal kick | Defending team (attackers touched it last over the goal line, no goal) | It is kicked and clearly moves | Ball still inside the goal area. Keeper or centre-back next to it. Opponents outside the penalty area. Both teams pushed up towards halfway, or split for a short kick |
| R3 | Corner | Attacking team (defenders touched it last) | It is kicked and moves | Ball still at a corner. One attacker next to it. Eight to fourteen players inside the penalty area. Defenders at least 9.15 m from the corner |
| R4 | Free kick | The fouled team, or the defending team for offside | It is kicked and moves | Ball still on the pitch. Opponents at least 9.15 m away. Near goal: a wall, three to six defenders in a line about 9.15 m from the ball |
| R5 | Penalty | The fouled (attacking) team | It is kicked and moves forward | Ball still on the penalty mark. Only the taker and the keeper inside the area; everyone else outside the area and the arc |
| R6 | Kick-off | The team that conceded; S5 and S6 for the halves | It is kicked and moves | Ball still on the centre spot. Every player in his own half. Opponents outside the centre circle |
| R7 | Dropped ball | The team that last touched it; the keeper if inside a penalty area | It touches the ground | Rare. One player at the ball, everyone else at least 4 m away, referee present |

Restart evidence (all feed the choice between `D→H` and `D→A`):

| ID | Rule | Tier | Encoding |
|---|---|---|---|
| R8 | Last team to touch before the ball crosses a line loses the restart, except a goal. | LAW | EVENT (last touch) → COST |
| R9 | Where the ball went out fixes the restart type: touchline → throw-in; goal line outside the posts → corner or goal kick; between the posts → goal and kick-off. | LAW | EVENT |
| R10 | Over the goal line, not a goal: last touch by the defending side → corner to attackers; last touch by attackers → goal kick. | LAW | EVENT → COST |
| R11 | The restart is taken from where the ball went out (throw-in), the corner nearest the exit (corner), anywhere in the goal area (goal kick), or the place of the offence (free kick). A stationary ball far from any legal spot is not a restart. | STRONG | CHECK on restart candidates |
| R12 | The lone player standing next to a stationary ball at a legal restart spot belongs to the restarting team. | STRONG | EVENT → COST |
| R13 | After a goal the conceding team kicks off. | LAW | STATE |
| R14 | Smart Tagging SET PIECES with a team: that team restarts. Throw-ins have no team; use R8. | STRONG (check agreement) | COST |
| R15 | A penalty is a restart for the attacking team even though the ball is deep in the defenders' area. | LAW | EVENT (penalty formation) |
| R16 | Offside free kicks go to the defending team and are taken well inside their half, often by the keeper or a centre-back, with the attackers walking back. | STRONG | EVENT |
| R17 | A quick free kick or quick throw can restart in under 3 s with players still out of position. | CLUE | Keeps S10's minimum low |

## D. When the ball becomes dead

| ID | Rule | Tier | Encoding |
|---|---|---|---|
| D1 | The ball is out when all of it crosses a touchline or goal line, on the ground or in the air. The labels start D at about the crossing, slightly after it where the assistant's signal is the cue and slightly before it where the ball is plainly running out unreachable (D15). (reviewer convention, 14 Nov) | LAW | EVENT (ball exit) |
| D2 | A tracked ball outside the lines is evidence of D only if it stays out for several frames or disappears while out. A high ball near the far touchline is drawn outside the pitch by the camera projection and then comes back. | STRONG | EVENT with persistence (start: 8 frames, or vanishes) |
| D3 | A ball last seen within a few metres of a line, moving outwards fast enough to cross it, that then goes missing has probably gone out. | STRONG | EVENT (extrapolated exit) → COST |
| D4 | Goal: the ball crosses the goal line between the posts. Dead until the kick-off, usually 45 to 90 s. | LAW | EVENT + R13 |
| D5 | A foul stops play at the whistle, not at the foul. Tracking shows no whistle; the trace is players stopping their runs together, so D is dated at that collective stop. The signs are a sharp collective slow-down within about 2 s, the ball stopping or being picked up, one or more players stationary (on the ground), and a free-kick set-up afterwards. (reviewer convention, 14 Nov) | STRONG as a bundle, CLUE singly | FEAT + EVENT (slow-down) |
| D6 | Advantage: a foul with no whistle. Play continues, so no D, and the ball stays with the fouled team until the whistle (the referee waiting to see an advantage is not a stoppage). (reviewer convention, 14 Nov) | LAW | Nothing; D5 needs the slow-down |
| D7 | Offside: dead from about the whistle, not from the flag. Play stops shortly after a forward pass, attackers stop running, defenders step up and stop. (reviewer convention, 14 Nov) | STRONG | Same detector as D5 |
| D8 | Play can also stop for an injury, the ball hitting the referee, or a second ball on the pitch. Restart is a dropped ball. | CLUE | D5 detector |
| D9 | After the whistle the ball often keeps moving: kicked away, rolled to the spot, thrown in from a ball boy. Movement of the ball during a stoppage is not play. | STRONG | STATE (D persists until a legal restart, section L) |
| D10 | Long stoppages (substitutions, treatment, VAR, cooling breaks): players walk, gather at the touchline or benches, the number of tracked players changes. | STRONG | FEAT (walking share, players near benches, tracked count) |
| D11 | Goal celebration: several players of one team sprint to a corner or touchline and cluster; the other team walks to the centre. | STRONG | FEAT; confirms D4 |
| D12 | The ball going missing is evidence of D that depends on context: strong when it was last seen near a line heading out or when everyone slows; weak when it vanished in a crowd or after a long kick. | CLUE | FEAT (time since seen, last position, last velocity, players within 3 m when lost) |
| D13 | A missing ball for longer than a pass or cross can last (start: 6 s) with slow players is very likely D. | STRONG | PSEUDO candidate, measured on other matches first |
| D14 | Between the final whistle of a half and the end of the graded frames there is nothing; the graded span ends with the half. | LAW | Nothing |
| D15 | Ball out of play: D starts at about the line crossing. The exact offset has no rule to copy: it can be slightly after the crossing (around the assistant's signal) or slightly before it when the ball is plainly running out and no player can reach it. The offsets are to be measured on the bench, and the metric is forgiving at these edges, so do not spend effort below it. (reviewer convention, 14 Nov) | STRONG | EVENT (exit) with SNAP; offset a bench-chosen parameter |

## L. When the ball becomes live again

| ID | Rule | Tier | Encoding |
|---|---|---|---|
| L1 | Free kick, corner, goal kick, penalty, kick-off: the labels turn live shortly after the taker strikes the stationary ball (the lag is to be measured on the bench, not assumed zero). A keeper releasing the ball from the hands is the same: live shortly after the release. The restarting team has the ball from the first live frame (R0). (reviewer convention, 14 Nov) | LAW | EVENT (still for ≥ 1 s at a legal spot, then speed above about 3 m/s with a restart-team player within 1.5 m); SNAP by the measured lag |
| L2 | Throw-in: live at about the release. The ball is rarely tracked in the hands, so the first tracked frame inside the pitch near the exit point is slightly late. The lead before that frame is to be measured on the bench (the bench value, not a rulebook constant, decides). The same shortly-after-release convention as L1 applies. (reviewer convention, 14 Nov) | LAW + STRONG | EVENT; SNAP |
| L3 | A stoppage ends only with a legal restart. No restart candidate, no exit from D. | STRONG | COST (leaving D is very dear away from a restart candidate) |
| L4 | If the ball is not tracked at the restart, the restart shows as the formation breaking: the still players near the restart spot start moving together, and both teams' centroids accelerate. | CLUE | EVENT (formation break) |
| L5 | A Smart Tagging SET PIECES window (about 5 s wide) contains a restart. | STRONG | COST (leaving D is cheap inside it) |
| L6 | At the instant of a kick-off, corner, penalty or goal kick the formation is fixed by law (R2 to R6), so the last frame the formation still holds is just before the restart. | STRONG | SNAP |

## P. Possession while the ball is live

The definition: possession changes when the other team gains **controlled** possession. Touching
is not controlling.

### What counts as gaining control

| ID | Rule | Tier | Encoding |
|---|---|---|---|
| P1 | A player keeps the ball close (within about 1.5 m) and moving with him for a sustained time (start: 0.6 s) with no opponent closer. | STRONG | EVENT (carry) → COST, PSEUDO |
| P2 | A team makes two touches in a row by two different players (a completed pass) with no opponent touch between. | STRONG | EVENT (pass) → COST |
| P3 | A keeper catches or gathers the ball in open play: the ball stops relative to a defending player who is the deepest of his team, inside his own penalty area. This is a live switch to the keeper's team, dated at about the moment the keeper is clearly going to get it (P30), not a stoppage. (reviewer convention, 14 Nov) | STRONG | EVENT (keeper gather) → COST |
| P4 | A player receives a pass from an opponent cleanly and plays on (first touch then carry or pass). | STRONG | P1 or P2 following a touch |
| P5 | A tackler comes away with the ball (the ball then travels with him, P1). | STRONG | EVENT |
| P6 | The team taking a restart has the ball from the restart frame, whatever happens next. | LAW | STATE (R0) |
| P30 | A loose ball belongs to the team clearly about to collect it, dated before its first touch: after a misplaced pass, a failed dribble, a deflection or a rebound the label switches at the moment it is clear that one team's player will reach the ball first and no opponent can. Clues: ball direction and speed, which players are running onto it, distance of the nearest opponent. The switch is later when players of opposite teams are both close to the ball, earlier when the ball rolls into space with one team's player nearest. "Possession starts at the first touch" is too late and "once control is confirmed" later still. Numbers (lead before the touch, distances) to be measured on the bench. (reviewer convention, 14 Nov) | STRONG | FEAT (per-team time to reach the ball) → COST; SNAP before the first touch |
| P31 | No contested state: every live frame belongs to a team. A human may call a moment "nobody's ball", but the labels have no such state, so a scramble or a loose ball is resolved by P30 and otherwise held with the last team (S2). The labels are frame-accurate about who is playing or about to win the ball and are not smoothed (S9). (reviewer convention, 14 Nov) | LAW (from the definition) | STATE (no third live state) |
| P32 | A challenge that only dislodges the ball does not switch possession on its own. Possession goes to the team that next plays the ball deliberately (P21) or is clearly about to collect it (P30); if the original team collects it, nothing changed (P8). (reviewer convention, 14 Nov) | STRONG | EVENT (challenge) → switch needs P30 or P21 evidence |

### What does not change possession

| ID | Rule | Tier | Encoding |
|---|---|---|---|
| P7 | A deflection is not itself control; what decides is where the ball goes next and who is running onto it (P30). A deflection after which the original team keeps the ball, or the ball goes out of play, changes nothing. A pass that strikes an opponent and runs to that opponent's team-mates, or into space only they can reach, switches possession to the deflecting team at about the deflection. So "a deflection never changes possession" is not a rule. (reviewer convention, 14 Nov) | STRONG (was LAW from the challenge text; the labels do not follow it as a blanket) | EVENT (deflection) plus P30 evidence of who collects → switch at the deflection when the other team collects, dear otherwise |
| P8 | A block or a tackle that knocks the ball loose, when the original team has it again within a short time (start: to be measured on the bench). See P32. (reviewer convention, 14 Nov) | STRONG | EVENT |
| P9 | An incidental touch in the air that is not a deliberate header (a glance, a graze) is a deflection (P7). A header won is P17 and does give possession. (reviewer convention, 14 Nov) | STRONG | EVENT (header: touch at the end of a long flight); P17 decides ownership |
| P10 | A clearance that goes straight out of play: the clearing team has it from the clearance (P19) until the ball goes dead (D15); the other team then restarts (R8). The possession is short, followed by D. (reviewer convention, 14 Nov) | STRONG | EVENT; R8 gives the restart |
| P11 | A save that is parried or pushed away. No change on the parry alone; P7 and P30 decide who is clearly about to collect it. (reviewer convention, 14 Nov) | STRONG | P7 and P30 logic |
| P12 | The ball hitting the post, bar, corner flag or referee. | LAW | Nothing |
| P13 | A dribbler pushing the ball several metres ahead at a sprint, or a defender shielding the ball without touching it. Nearest player may be an opponent; possession has not changed. | STRONG | "Nearest player" alone never triggers a switch; P1 or P2 needed |
| P14 | A long pass or cross in flight belongs to the passing team until just before an opponent wins it (P17), not midway through the flight even when the outcome is obvious, and not after the winning touch. (reviewer convention, 14 Nov) | STRONG (was LAW, S2) | Holding through flight; ball usually missing; switch SNAP to just before the winning touch |
| P15 | A scramble: several single touches by both teams in a crowd. Stays with the last team that had it until a deliberate play (P21) or a team clearly about to collect (P30) appears. (reviewer convention, 14 Nov) | LAW (S2) | Switch dear while 4 or more players are within 3 m of the ball |

### Where the switch is dated (the timing question)

| ID | Rule | Tier | Encoding |
|---|---|---|---|
| P16 | Dating is never later than the new team's first touch of the controlling sequence, and never at the moment we became sure. For a loose ball the labels date the switch before that touch (P30); for contact events (P18, P20) at about the contact. "Once control is confirmed" and "at the second touch" are both too late. (reviewer convention, 14 Nov; answers Q3, Q6) | STRONG (was OPEN) | SNAP to P30 or to the contact |
| P17 | A header won gives possession to the winner's team from just before the header, including when the ball then rebounds off an opponent and the other team collects it, and when a team-mate collects the knock-down. The dating is not the control that follows, and not midway through the flight (P14). (reviewer convention, 14 Nov; answers Q4) | STRONG (was OPEN) | SNAP to just before the header |
| P18 | A pass cut out by an opponent switches at about the moment of the opponent's contact, not when the pass was struck, even if the ball then goes up and a team-mate secures it. Where the ball is loose and it is already clear who gets it, P30 dates the switch earlier. (reviewer convention, 14 Nov; answers Q3, Q6) | STRONG (was OPEN) | SNAP to the contact |
| P19 | A clearance is possession for the clearing team from the clearance, including a blocked clearance up to the block, and including one kicked upfield to nobody. If the original team collects it, nothing changes; if the other team wins it, the switch follows P17, P18 or P30 as the case is. (reviewer convention, 14 Nov; answers Q5) | STRONG (was OPEN) | EVENT (clearance); SNAP to the clearance |
| P20 | A tackle that sends the ball towards the tackler's team-mates, or a one-touch interception that goes on to a team-mate, switches at about the contact, even before a team-mate's touch. (reviewer convention, 14 Nov; answers Q3) | STRONG (was OPEN) | SNAP to the contact |
| P21 | A player who deliberately plays the ball (a pass under pressure, a toe-poke to a team-mate, a clearance, a header that is won) gives their team possession from that play, however brief, even if the ball is lost again almost at once. There is no minimum length and no third state (P31). What does not count on its own: a challenge that only knocks the ball away (P32). (reviewer convention, 14 Nov; answers Q1, Q2) | STRONG (was OPEN) | EVENT (deliberate play); decides how short P1 may be: no floor beyond the bench-chosen S9 value |

### Live ball that looks dead (false stoppage traps)

| ID | Rule | Tier | Encoding |
|---|---|---|---|
| P22 | Keeper holding the ball. Live, his team's possession (the switch comes at the gather, P3), for up to about 6 to 10 s. (reviewer convention, 14 Nov) Signs: ball still or missing, one defender deepest of his team inside his own area with it, opponents jogging away, team-mates pushing out. No one else is still, nothing is being set up. | STRONG | EVENT (keeper hold) blocks D; PSEUDO for the keeper's team |
| P23 | A defender standing on the ball under no pressure, or walking it out. Live. | STRONG | Still ball inside the pitch away from any restart spot, no slow-down beforehand → not D |
| P24 | Slow build-up: centre-backs passing sideways while most players walk. Live. Mean player speed alone cannot separate this from a stoppage. | STRONG | D5 needs the *drop* in speed and a restart set-up, not low speed itself |
| P25 | Ball stuck under a pile of players at a corner or scramble. Live until a whistle. | CLUE | Missing ball in a crowd is weak D evidence (D12) |

### Dead ball that looks live (false play traps)

| ID | Rule | Tier | Encoding |
|---|---|---|---|
| P26 | Ball kicked or rolled after the whistle, or passed back to the restart spot. | STRONG | D9 |
| P27 | A player walking with the ball to a corner or throw-in spot. | STRONG | Ball moving at walking pace towards a restart spot while everyone else is slow → D |
| P28 | A replacement ball thrown on from the side while the first is still tracked somewhere, or the tracker jumping between two balls. | CLUE | Ball jumps > 15 m between consecutive seen frames → treat as unreliable |
| P29 | Players sprinting during a stoppage (celebration, running to take a quick throw, protests). | CLUE | Sprint without the ball moving is not play |

## T. Reading team shape when the ball is not visible

All of these are computed per team from whoever is tracked on that frame. Track slots are not
people; never follow a slot.

| ID | Rule | Tier | Encoding |
|---|---|---|---|
| T1 | The team with the ball is wider and deeper; the team without it is compact. | CLUE | FEAT (robust width, depth, median distance to centroid, per team, and the difference) |
| T2 | The defending team's back line and keeper retreat towards their own goal; the attacking team's keeper stands high. | CLUE | FEAT (deepest player and back-line height relative to own goal, per team) |
| T3 | At a turnover the collective direction of travel flips within one to two seconds: the team that won it accelerates forwards and spreads, the team that lost it turns and runs back. | STRONG when large | EVENT (flip of team-centroid velocity along the attack axis) → switch cheaper |
| T4 | Opponents converge on the ball carrier. The point several players of one team are closing on is where the ball is, and the player there who belongs to the other team has it. | CLUE | FEAT (convergence point, team of nearest player to it) |
| T5 | If the ball was with team X before a gap, is with team X after it, and T3 saw no flip, team X had it throughout. | STRONG | Holding (free) |
| T6 | If the ball was with X before a gap and with Y after it, the switch is at the T3 flip inside the gap; with no clear flip, just before the ball reappears with Y. | STRONG | SNAP |
| T7 | Shape measures jump when a tracked player drops out. Use medians and trimmed ranges, and pass the tracked-player count to the model. | STRONG | FEAT hygiene |
| T8 | Two slots of one team within about half a metre are one player counted twice. | STRONG | Merge before computing anything |
| T9 | Both teams walking with no direction of travel and no pressing means a stoppage, not a possession. | CLUE | FEAT |

## B. Ball tracking hygiene

| ID | Rule | Tier | Encoding |
|---|---|---|---|
| B1 | A ball cannot move faster than about 40 m/s. Faster jumps are tracking errors. | LAW (physics) | Drop the point |
| B2 | A "ball" that sits exactly still on a pitch marking (penalty mark, centre spot) while players run elsewhere is probably a false detection. | CLUE | Flag; do not treat as a restart unless R5 or R6 formation holds |
| B3 | Gaps shorter than about half a second inside a straight pass can be filled by interpolation; longer gaps are left empty and marked. | STRONG | Ball stage |
| B4 | A sudden change of ball speed or direction next to a player is a touch. Without a player within about 2 m it is a bounce or an error. | STRONG | EVENT (touch) |
| B5 | A ball moving in a straight line at more than about 8 m/s with no player within 2 m is in flight between players, whatever is nearest. | STRONG | No ownership evidence from proximity during flight |
| B6 | The ball being in the air is not visible in 2-D. Long kicks that slow smoothly and land near a cluster are aerial; the first touch after them is often a header (P9). | CLUE | FEAT (flight length and duration before a touch) |

## G. Smart Tagging

Tags are automatic and noisy. They are evidence, never labels.

| ID | Rule | Tier | Encoding |
|---|---|---|---|
| G1 | BUILD UP, PROGRESSION, CREATING CHANCES, BALL IN FINAL THIRD, BALL IN THE BOX, PLAYERS IN THE BOX, LONG BALL with team T: T has the ball **if the ball is live**. | STRONG (check agreement per code) | COST on H versus A only |
| G2 | DEFENDING IN ATTACKING / MIDDLE / DEFENSIVE THIRD with team T: the other team has the ball if it is live. | STRONG (check; flip if agreement is below half) | COST on H versus A only |
| G3 | Phase tags say nothing about live versus dead. PLAYERS IN THE BOX and BALL IN FINAL THIRD can cover the wait before a corner or free kick. | STRONG | Tags never vote on D, except G5 to G7 |
| G4 | ATTACKING TRANSITION (T) starts just after T won the ball; DEFENSIVE TRANSITION (T) just after T lost it. | STRONG | COST (switch cheaper within about 2 s before the tag start) |
| G5 | SET PIECES: a restart happens inside the tagged window (L5) and the tagged team takes it (R14). | STRONG | COST |
| G6 | GOALS (T): a stoppage begins near the tag and the other team kicks off. The file has four goal tags; do not assume all four are real. Require the kick-off formation (R6) to follow. | STRONG with confirmation | EVENT + R13 |
| G7 | Fewer than half of real restarts are tagged (59 tags against 90 to 110 expected). No tag means nothing. | STRONG | Never penalise D for lacking a tag |
| G8 | Tag edges are accurate to about a second. Never snap a boundary to a tag edge. | STRONG | Tags widen or narrow costs; SNAP uses tracking events only |
| G9 | LONG BALL (T): T has it at the start; the end is a contest (P9, P14). | CLUE | COST near the start only |
| G10 | Each code's weight is its measured agreement with confident tracking frames on this match, computed by code and reported per code, never per instance. | — | Method |

## A. Ambiguous cases and the clues that resolve them

| # | Situation | Looks like | Decide with |
|---|---|---|---|
| A1 | Opponent nearest the ball for a few frames in the middle of a possession | A turnover | Was there a touch (B4)? Did the ball then travel with him (P1) or to his team-mate (P2)? If neither within about a second, hold |
| A2 | Ball missing 3 s, same team either side | Could be anything | T5: hold |
| A3 | Ball missing 5 s, other team after | Late switch | T6: place at the shape flip |
| A4 | Ball vanishes near the touchline, reappears 10 s later at the same place | Occlusion or throw-in | D3, slow-down, a still player on the line (R1) → D between, restart to the team per R8 |
| A5 | Cross into the box, ball lost, then seen with the keeper | Attack continuing | P3: change at the gather; if the ball then stays still with him, P22 not D |
| A6 | Cross into the box, ball lost, players slow, both teams reorganise | Keeper hold or foul or goal kick | Still ball in the goal area with opponents leaving the area → goal kick (R2, D). Ball with a deep defender and no formation → keeper hold (live). Ball at centre with everyone in own half next → goal |
| A7 | Shot, ball lost | Goal, save, wide, block | Kick-off formation follows → goal. Corner formation → saved or blocked wide, corner. Goal-kick formation → wide. None → live, hold until control |
| A8 | Header duel after a long ball, then a loose ball | Turnover to the header's team | P17: the header winner's team has it from just before the header, whoever collects next (reviewer convention, 14 Nov) |
| A9 | Tackle, ball squirts loose, original team picks it up | Turnover and back | P8: no change |
| A10 | Tackle, tackler's team plays one pass and loses it | Blip | P2 satisfied → it was a real short possession (S9) |
| A11 | Defender hoofs it out for a throw | Defender's possession then D | P10, P19: the clearing team has it from the clearance, then D; restart to the attackers (reviewer convention, 14 Nov) |
| A12 | Players stop but the ball rolls on with one player chasing | Offside or foul, or a tired phase | D7: stop of the attackers plus a still ball at a free-kick spot afterwards → D from the stop |
| A13 | Ball still on the edge of the box, a line of defenders 9 m away | Free kick pending | R4: D until the kick, restart to the non-wall team |
| A14 | Ball still in the centre circle | Kick-off or a midfielder standing on it | R6 formation → kick-off, D before; otherwise P23 |
| A15 | Ball "outside" the far touchline for a few frames then back in play at speed | Out of play | D2: projection artefact, hold |
| A16 | Everyone walking for 60 s, some players near the benches | Slow play | D10: D throughout |
| A17 | Quick free kick: players stop for 2 s then play | Nothing happened | D5 then L1 with a short D; allowed by S10 |
| A18 | Advantage played after a foul, one player down | Stoppage | D6: no collective slow-down, ball keeps moving with a team → live |
| A19 | Penalty saved and held | Attack continuing | R5 restart to attackers, then P3 change to the keeper |
| A20 | Goal straight from the kick-off pass back, long wait | — | R6, S5 |
| A21 | Back-pass to the keeper who dwells on it | Stoppage | P23: live, same team |
| A22 | Ball boy returns the ball quickly, throw taken while the tracker shows the ball outside | Still dead | L2 and L4: formation break dates the restart |
| A23 | Two candidate restart teams disagree (tag says H, last touch says H touched it last) | — | Law beats tag for throw-ins and corners (R8, R10); tag beats a weak last-touch detection when the ball was missing at the exit |
| A24 | Drop ball | Odd restart | R7; rare, low priority |

---

# Part 2. Turning the rules into code

## The decoder is the rulebook's skeleton

Hidden states: `H`, `A`, `D→H`, `D→A`. Output: H, A, D, D.

| From | To | Allowed | Cheap when | Dear when |
|---|---|---|---|---|
| H | A (and A → H) | yes | controlled-touch event by the other team (P1 to P5), shape flip (T3), transition tag (G4) | otherwise; dearer still in a crowd (P15) or with only a deflection after which the original team keeps the ball (P7); cheap also on time-to-ball evidence that the other team reaches a loose ball first (P30) |
| H or A | D→H or D→A | yes | ball-exit event (D1 to D4), slow-down event (D5, D7), goal (D4) | otherwise; blocked during a keeper hold (P22) |
| D→H | H only | yes | restart candidate (L1, L2, L4), set-piece tag (L5) | no restart candidate (L3) |
| D→A | A only | yes | same | same |
| D→H | A, or D→A | **never** | | |

Minimum durations by chained states: possessions a small value (S9), stoppages about 2 s (S10).
Frames with no evidence hold the current state at no cost (S2).

Which dead state: emission evidence on the whole stoppage from R8 to R16. Viterbi chooses the
branch for the entire stoppage at once, so evidence that only arrives at the restart (the
formation, the taker) still decides it.

## The path from sentence to code

Every rule goes through the same five steps. The `football-rules` skill holds the detail.

1. **Registry entry.** `pipeline/rules/registry.yaml`: id, tier, encoding, plain statement, the
   function that implements it, its parameters with start values, status (draft, tested, measured,
   in use, dropped).
2. **Detector or cost function** in `pipeline/rules/`, team-level inputs only, centred in time.
3. **Synthetic test.** A scripted toy scene (a few players and a ball moving on a pitch, written
   in code) for the rule and for its nearest trap. Example: R1 test scripts a ball crossing the
   touchline; its trap test scripts a high ball drawn outside and returning (D2).
4. **Measurement on other matches.** Precision and coverage of the detector against bench truth.
   STRONG rules below about 0.9 precision are demoted to CLUE or have their thresholds changed.
   OPEN rules are measured under both conventions.
5. **Use.** Wired into the decoder as STATE, COST, MASK or SNAP, or into the frame model as FEAT,
   or into pseudo-labels as PSEUDO. One version per change, judged on the bench.

The registry generates the "possession rules implemented" section of the technical document.

## Build order

| Order | Rules | Why first |
|---|---|---|
| 1 | S1 to S8, R0, B1, B3, T8 | Skeleton; cannot be wrong |
| 2 | D1 to D4, L1, L2, R1 to R6, R8 to R10 | Stoppages caused by the ball leaving; the bulk of all stoppages |
| 3 | B4, B5, P1, P2, P7, P13 | Control versus proximity; the short-possession problem |
| 4 | D5, D7, D9, P22 to P24, L3, L4 | Whistle stoppages and their traps |
| 5 | G1 to G8 | Tags as evidence |
| 6 | T1 to T6, D12, D13 | Long ball gaps |
| 7 | P16 to P21, P30 to P32 (answered 14 Nov; offsets measured on the bench) | Switch timing |
| 8 | D10, D11, P3, P26 to P29, B2, B6 | Remaining cases |

## Pseudo-label rules (labels chosen by code on the match)

| Label | Rule | Source rules |
|---|---|---|
| Team X | Ball seen, an X player within 1 m and moving with it, no opponent within 3 m, for at least 1 s | P1 |
| Team X | Two consecutive X touches by different players, frames between them | P2 |
| D | Ball persistently outside the lines, or vanished heading out, until a restart candidate | D1 to D3 |
| D | Kick-off, corner, penalty or goal-kick formation holding with a still ball | R2, R3, R5, R6 |
| D | Before the second-half kick-off | S4 |
| Team X | Keeper hold by X | P22 |

Their precision is measured by running the same rules on other matches.

## Output checks that block a version

- Every stoppage is followed by a possession (or the end of the half).
- Second half: D until the detected kick-off; the detected kick-off is within a second of the
  published frame; first possession belongs to the team that did not kick off the first half.
- No possession crosses half-time. No `X` on graded frames.
- Share of D, stoppage count and possessions per minute inside the S11 ranges, else a warning.
- No stoppage shorter than the S10 minimum.
