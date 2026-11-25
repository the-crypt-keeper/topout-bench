# Why it is like this

Most of what is here replaced something that looked reasonable and was not. This page records what we tried, how it failed,
and what the survivors are protecting. Dates are 2026. Numbers are the public dataset at 25 arms and 400 games unless it says
otherwise.

- [The game](#the-game) · [What to measure](#what-to-measure) · [What failed](#what-failed-and-how-we-could-tell) ·
  [Burden](#burden) · [The triangle and moves](#the-triangle-and-moves) · [Decisions](#decisions) ·
  [The shape of the tools](#the-shape-of-the-tools) · [The dataset](#the-dataset) ·
  [Rules](#rules-we-hold-ourselves-to) · [Removed](#removed-and-why) · [Open](#open)

## The game

**The board is text.** We tried screenshots first (October, an earlier harness, one model, Qwen3.8-27B): it named the falling
piece correctly in 41% of decisions, against 100% from the text board. Vision tested vision.

**One tool per button, nothing forced.** `tool_choice` is `auto`, and arguments-free verbs leave little to get wrong: forcing
is brittle in a different way on each engine. Several calls in one reply run in order, each costing time.

**Thinking is free, moving costs.** The emulator waits for the model, so a slow GPU plays the same game. Every button runs the
game at least 12 frames, a human's reaction time, so a plan has to be executed under gravity.

**A piece is one conversation.** The plan for a piece stays in context while it is steered; the next piece starts clean. That
keeps the context small, makes each piece a fresh draw from the model's policy given its board, and keeps a template that
shows thinking only after the last user turn from hiding it: there is one user turn per piece.

**100 pieces is a budget, not a goal.** It is an arbitrary number, and the headline measure barely depends on it (below).

**No forcing, no fixing the model.** Where a reply has no tool call we press `wait`, say so, and count it. A reply cut off by
its limit is tried again, and if it never fits it loses its turn. A conversation that outgrows the context starts over. These
are recorded as behaviour, never engineered away.

**One prompt.** Other instruction texts were tried in pre-repo work, on two models, and none did better. The model either knows how to
play or does not.

## What to measure

The questions sit on a grid: *how many arms* against *which level of the game*. The chain runs **models and efforts → games →
pieces → decisions**, and the whole project is choosing the right thing to say in each cell.

| | all arms (a shallow look at everything) | a few arms (a ladder, or model against model) | one arm (deep) |
|---|---|---|---|
| **games** | burden, against what it costs | every game's burden as dots | its games, together |
| **pieces** | – | triangles and moves side by side, one scale | the triangle over time, its moves |
| **decisions** | fingerprints | fingerprints on one scale | read a piece: board, thinking, presses |
| **the hat it wears** | interactive, or a still image | the same, side by side | interactive only |

The empty cell is real: we have no picture of pieces across *all* arms that says anything ([failed](#what-failed-and-how-we-could-tell)).
"The hat" follows from the cell: a still image compares things, because you cannot compare two moving pictures, and a moving
one explains one thing, because you can stop it on the piece you care about.

## What failed, and how we could tell

**Lines at the end of a game.** Coefficient of variation about 0.74 in our pre-repo work: telling two models 10% apart would
take some 850 games an arm. Worse, a survivor's line count is its final board read backwards (each piece adds 4 cells, each
line removes 10), while a topped-out game's is where it happened to stop. Two different measurements on one axis.

**Lines per turn.** It is exactly `0.4 - filled / (10 x placed)`. For a survivor it is the debris left over, squeezed against
a ceiling; for a top-out it is mostly game length dressed as a rate. Nothing in the number says which regime it is in.

**Survival.** A coin flip against an arbitrary line. 8 of 16 spans 0.28 to 0.72. Twelve of our 25 arms score 0 of 16, and those
twelve have burdens from 11.0 to 15.7, very different players. Haiku high and xhigh are both 15 of 16 while their burdens are
2.39 and 1.23.

**Averages of per-placement changes.** "Sealed cells per placement" looks like a piece-level measure. Add up every change and
you get the final count: it is the end of the game divided by its length. Only where play *lives*, or changes split *by
situation*, escapes that.

**Average dead rows over the pieces that were played.** Our first version of burden. It ranked Qwen3.8 Flash Next *low* (6.06,
nothing reached the end) above *medium* (6.70, one did): a game that dies early has no time to look bad.

**A heatmap of dead rows, all arms.** One row per arm, 21 columns. The top rows read; past the good arms it is mush, because
every game starts at the origin and most arms end up everywhere. It shows how its own average was built and little else.

**Anything against piece number.** Trajectory plots, fans, sealing over the first 24 pieces. The play decides which boards a
game sees, so a clock on an axis mixes what a model does with when its game happened to be there. Removing time is what made
the triangle readable. It returns as motion, or on one game's own page.

**Pooling a piece's replies.** Planning and execution are different populations; medians differ 7-fold (Haiku low) to 107-fold
(Haiku max). Their average describes no reply. MiMo's pooled 15.7% truncation is really 39.0% of planning and 1.7% of
execution.

**Counting faults.** A conduct report (capped, refused, stray arguments) read "everything was fine" on arm after arm. Faults
are found when bringing a model up. What tells models apart is *behaviour*: how many calls per reply, which buttons, whether
it waits. Splitting those by phase is what turned them from weak into some of the most telling columns here.

## Burden

> Dead rows averaged over every turn of the game, a turn after the game ended counting as 20.

It fixes the horizon, which repairs the early-death bias: dying is not a way to stop accumulating. Dead rows because they are
what fills a board you cannot clear; live rows have a median of 2 to 4 in every arm, and dead ones are what separate
them.

It has two arbitrary constants, so we checked whether they matter (rank agreement with the measure as used, Spearman, 25 arms):

| instead | agreement |
|---|---|
| stop at piece 30 / 50 / 75 | 0.989 / 0.996 / 0.998 |
| charge 10 / 15 / 30 for a lost turn | 0.987 / 0.998 / 0.998 |

They barely move the order. What to know about it all the same:

- **It changes character.** Where games top out it is mostly the charge for lost turns (60% to 85% for the twelve arms that never
  reach the end); at the top it is all damage. Ranking arms by unplayed turns alone agrees at 0.986, so dead rows
  mainly add resolution where survival is saturated.
- **It is one number per game**, so 16 games is n = 16. But it is a graded random variable, not a coin flip, and carries far
  more than survival does. The intervals resample whole games.
- **Overlapping intervals say nothing.** Intervals apart rule equality out. The comparison is the two sets of dots.

**Tokens are a mean on that plot** though we describe replies by quantiles. The axis is a cost, and the mean times the pieces is
the bill, tail included. The median hid Inkling's 40k budget completely: 35,686 against 36,455 at high, where the mean went
from 43,780 to 38,673.

**One point per model is a choice between two honest answers**, so the plot offers both. Best performance ignores cost. The cost
frontier weighs burden and log tokens alike; there is no right weighting. Both axes scale over every loaded arm, never the
selection, because filtering must not change what a number means.

## The triangle and moves

Cells turned out to be too fine and lines too coarse. **The row is the unit of clearability**: a row with a sealed cell cannot
complete until it is dug out, and every other row can in principle. So a board is three counts that sum to 20 (empty, live,
dead), and every board is a point in a triangle. `filled` is not an axis for this: a tower sealed on one side is full of cells
and can clear nothing.

**It is piece-level.** The share is of an arm's pieces. Weighting each game equally was proposed and rejected: it changes the
question. "Never pool survivors with top-outs" is a rule about game outcomes, and does not apply here.

**Overlapping spawns are kept.** Near a top-out the next piece appears on top of the stack and the text, with one character a
cell, prints `@` over `#`. The code used to notice its count was short and throw the board away. That is ordinary Tetris, not a
data fault: every one of the 293 top-outs ends on such a board, so the last board of every lost game was missing. Now they are kept and the piece's cells count as
occupied. No change in sealed cells is read into or out of one, because the falling piece is in that board.

Know what that does. The spawned piece is in the top row, so **every top-out's last board reads as exactly 20 rows, by
construction**: its marker sits on the line because of the rule. Without the piece those stacks are 18 (29 games), 19 (138)
or 20 (126) rows tall, so "a lost game ends at the top" is real and "exactly on the line" is not a finding. The rule also
applies to 172 earlier overlapping boards, where the game went on and only some of the four cells were stack; working out
which from the next board is the refinement, not built. It all changes an arm's burden by 0.018 at most.

**Moves keep the cell level**, where the triangle cannot see whether damage comes as erosion or catastrophe, and they show it
together with clearing, which is the only way to dig anything out. The numbers on each column are means, because debt is the sum of these changes and the rare
huge one is the point.

## Decisions

- **Mirrored**: planning up, execution down, from one line. Two tables side by side hid the pattern; one figure with the two
  halves facing each other shows it.
- **Shape for tokens, one scale for calls.** One extreme arm flattened every other when tokens shared a scale.
- **Buttons as two strips.** They were mirrored bars first; stacked strips show the mix at a glance.
- **Replies per piece is not split.** A piece has replies; a phase does not. Planning is exactly 1 for 96% of pieces (72% for
  MiMo, whose truncated plans are retried).
- **Truncated and nudged are part of the profile**, so they are columns. "Start alone" was too, and was dropped: the buttons
  strip already shows it.

## The shape of the tools

**One figure library.** The static scripts used matplotlib and the Explorer Plotly, and the Explorer's were better. Now every
figure is in `topout/figures.py`, and the Explorer and `topout-plot` both call it. A fix to one is a fix to both, and a number in
the app cannot disagree with the same number in an image. The price is that writing an image needs a Chrome.

**An app, because pictures crowd.** A static image shows everything at once, so it worsens as data grows. The Explorer makes
*selection* the control. The rail changes **what is drawn, never what a number means**: burden of "only the pieces that sealed
something" would not be burden, and we held off piece search partly for that reason.

**A backend with a local copy and a cache.** 650 MB of journals cannot stream to a browser, and a static page could not read
your own runs. So there is a server, on your dataset. Numbers and boards are digested and cached; a model's words are read from
its journal when you open that piece. Nothing derived is published: pieces recompute in seconds.

**The address is the state.** Filters, view, arm, game and piece are all in it. The rail is drawn *from* the address, and
changing the rail changes the address, so reload, Back and a pasted link all work. Choosing used to fight itself when two
things held the state; and Dash's address component reloads the page on every change unless told not to
(`refresh="callback-nav"`).

**Off the shelf where it exists.** Dash and Plotly give hover, clicks and linked plots; waitress serves several browsers at
once. Only the boards are ours, on canvas. Python builds the triangle once and the scrubber swaps its cells, so there are not
two triangles to drift apart.

**Phones get their own figures**, not a shrunken page, with a line saying when a larger screen is better.

**No facts about models in code.** Labels and ladders were guessed from names in the scripts; that broke the day Inkling needed
two ladders. The guess still exists, once, when publishing seeds `index.json`, and then a person owns it.

**The player only plays.** It writes what happened and computes no statistic, so every question can be asked again later of games
already played. Scripts that answered a question nobody asked were deleted.

**Videos are for one game at a time.** They are the only place real motion shows, and they are slow, so they come after a game
has been chosen for a reason.

## The dataset

**Raw journals, gzipped, one per game.** Not Parquet (needed only by the Hub's viewer, which is off), not a derived table (a
question we had not thought of would need the raw data anyway), not archives (nothing to share between games). gzip rather
than zstd costs about a third more than zstd's best and needs no dependency. Ids are hashes of the run and path, because a game's
index repeats across runs.

**No provenance in it.** A journal holds the served name, the settings and what happened. What hardware and engine produced it
means a lot to a lab with hundreds of recipes and nothing to someone trying one model.

**Every game has a sampler name, and an arm is 16 finished games.** Arms from before that were left out rather than patched in by
hand, except one whose name could be established exactly.

## Rules we hold ourselves to

1. **Never pool what is not one population**: survivors with top-outs, planning with execution, boards across arms on different
   scales.
2. **No clock on an axis** across games.
3. **Filtering changes what is drawn, never what a number means.** Scales and corners come from everything loaded.
4. **Same scale for anything placed side by side.**
5. **Compare distributions, not overlapping intervals.**
6. **A cost is a mean; a description is quantiles.**
7. **Say what a figure does not show.** In the caption if possible.
8. **A view nobody reads is deleted.** Git keeps it.
9. **Show it before judging it.** Several ideas here were argued about for a day and settled in a minute by a rough plot of the
   real data.
10. **Check the plot's number against something else**, and look at it in a browser: tests do not see layout.

## Removed, and why

| gone | because |
|---|---|
| `topout-regions` | plotted everything against piece number |
| `topout-placements` | pooled rates; its useful part became moves |
| the cells view | repeated the triangle at a finer grain |
| `topout-conduct` | counted faults, and nothing it could say changed how play is read |
| `topout-outcomes` | survival; burden replaced it. Its listing is `topout-plot games --table` |
| `topout-replies` | `topout-plot decisions --table`. Its reasoning character counts included encoded blocks in Haiku's journals |
| `topout-flows`, `topout-triangle-anim` | the triangle, moves and scrubber |
| `topout-burden`, `topout-decisions` | became `topout-plot` |
| the height ledger (packed, open and sealed slack) | exact, and no view used it |
| matplotlib | one figure library |

## Open

- **Skill in a situation.** Everything here mixes what a model does on a board with which boards its own play reaches. The lead:
  given the same live and dead rows, what does the next piece do? `P(clear | live rows)`, `E[lines | live, dead]`. Each
  bin should be trusted by its number of *games*, and two arms compared only where their boards overlap. Where they do not,
  that is the answer, not missing data.
- **How much would a triangle move if an arm played again?** Splitting its games in half and drawing both is the check that
  fits it. Not done.
- **A reference player.** A random policy, or a simple heuristic, on the same harness, to say what any number means in absolute
  terms.
- **Per-row sealing and unsealing**, tracked across clears, and reachability for real pieces rather than a 1x1.
- **A strict cost frontier**: the arms that nothing beats on both burden and tokens.
- **Piece search**, held off: it needs care about rule 3.
