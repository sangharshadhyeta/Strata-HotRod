# Strata-HotRod: the full notes

The README keeps the short version; this is the long one, with every measurement as it was recorded.

## What this fork changes

Upstream Strata, plus (the details and the measurements are in [SPEC.md](SPEC.md)):

- **Stand-in experts.** An expert that is not on the card can be replaced by a close one that is, so the card does not wait
  for the SSD; how far each rank may be swapped is in [the next section](#stand-in-experts-confidence-not-rank). Swap rounds move
  the experts a chat asks for onto the card in the background.
- **Guessing ahead.** A small draft head guesses the next words and the big model checks them together, with
  probabilistic acceptance of the guesses (`--no-spec-prob-accept` turns it off).
- **Three tiers of experts: card -> RAM -> SSD.** Every expert lives in exactly one place. The hottest are on the card, the
  next ones in a RAM tier sized from this PC's free memory, the rest on the SSD; swaps exchange experts between the card
  and RAM, and the look-ahead promotes the experts it predicts from the SSD into RAM.
- **Reading long prompts layer by layer.** The whole prompt passes one layer at a time, so each layer's experts are read
  from the SSD once, not once per piece of the prompt; the reads bypass the page cache (O_DIRECT), and the next layer's
  predicted experts are read ahead while the current one computes.
- **262K context with the K/V in RAM.** From 64K of context the attention's K/V streams from RAM; 32K tokens of it stay
  on the card.
- **Look-ahead on the card.** The next layers' experts are predicted on the graphics card (not recomputed by the CPU)
  while the current layer runs, and the predictions start their reads early. Since 24 Oct the reads ahead use **2 I/O
  threads, the next layer's first** (`STRATA_IO_PF_THREADS`, `STRATA_IO_EDF=0` for the old way): only about 3,000 of
  25,000 experts read ahead per answer were used, and 8 threads' wasted reads queued in front of the decode's own.
  +8% code, +15% reasoning, +8% prose, drive wait 3.8-4.7 s down to 2.8-3.0 s (two runs each, all answers right).
- **Starting the answer warm.** The experts the end of the prompt used most are put on the card before the first word
  of the answer, so decode does not start cold.
- **Picking up a conversation.** A long conversation's state is saved to the SSD, and its next turn reloads it instead of
  reading the whole conversation again: a 19.5K-token conversation comes back in 0.34 s after another one ran in between.
  The RAM tier first gives back memory for it (in whole huge pages, which Linux frees at once); before that fix 3 of 4
  returns to a 45K conversation were refused for ~100 MiB and read again from the start (68 s). The first answer after
  such a return warms the card with the experts this conversation used (the counts are saved next to it); a conversation
  that stays loaded is left to the swap rounds, which already fit the card to it. The saved conversations are kept to an
  eighth of the free disk; ones an older engine build wrote count against that and go first.
- **Memory safety.** The engine keeps a floor of RAM free before each request; when something else needs RAM, the RAM
  tier gives back its least-used experts until there is room again. The amount it keeps in reserve is measured on this
  PC and learned across restarts (`data/ram-reserve.json`), not a fixed number; the video-memory reserve is learned the
  same way (`vram-reserve.json`).
- **The app:** live speeds while it works, the experts-on-the-card map in the Monitor, and below it the decode timeline
  (Record; double-click or wheel to zoom, drag to pan): each layer's hand-offs and every SSD read, host times only.
- **Experimental, off by default:** `STRATA_GDN_LOOP=1 STRATA_GDN_LOOP_NORM=1` runs the middle linear-attention layers
  twice per token (a training-free "looped" model). Measured on 210 MMLU-Pro questions: 180 correct against 175 without
  it (not significant: only 9 questions differ), and decode 10% slower (38.5 against 42.7 tokens/s). Not worth it on a
  model that was not trained to loop, so it stays off.

## Stand-in experts: confidence, not rank

Default since 24 Oct 2026. Nothing in it is set by hand: the budget, the close-call line and when to re-check are measured.

- **One rule for every rank.** A missing pick takes the best stand-in on the card only when
  `router weight x logit gap <= budget`. A heavy pick (usually rank 0 or 1) needs a stand-in within about a logit; a light one
  (ranks 3-9) can take one much further away; a rank-0 pick in a flat routing is treated as lightly as its weight says.
- **Fragile tokens are decided exactly** (`STRATA_FRAGILE_RECHECK`, default on, `=0` off). For each token the engine adds up
  how far stand-ins moved it (weight x gap of every replaced pick, all layers). When the token's lead over the runner-up is
  below `k` times that move, the window is cut just before it and the next window decides it with the top ranks exact.
  `k` is learned: the 10th percentile of lead/move on the first 64 windows, then just above the largest move that actually
  flipped a token. One window in 16 also re-checks a token just above `k` (an explore cut), so a flip above the line is seen
  and `k` rises; `k` never falls without a flip seen in this run. On the 24 Oct tests it settled at 0.03-0.05; about 1
  re-check in 5 had flipped (the "bobed", "howed" kind of broken word), and only 1-3% of windows were cut.
- **The budget searches itself** (`STRATA_ROUTE_BUDGET_AUTO`, default on, `=0` off). Windows alternate in runs of 8 between
  budget `b` and `b x step`, so both see the same text; the faster one by 2% over 96 windows each wins, re-check costs
  included, **unless its flip rate is really higher**: with 20+ re-checks on each side, a candidate whose 90% lower bound
  on flips lies above the base's 90% upper bound loses however fast it is. Speed alone once took the budget to 0.95, and
  the text fell apart; the guard stops that climb where stand-ins start changing words.
- **Each request brakes itself** (`STRATA_FRAGILE_BRAKE`, default on, `=0` off). A request starts at half the learned budget.
  When at least 15 of its re-checks show a flip rate above 20% at 90% confidence, its budget halves; after 64 clean windows
  a long answer goes to the full learned budget. Short chats, where stand-ins move tokens about twice as much, stay low.
- **It remembers.** `data/standin-tune.txt` keeps the budget, `k` and the step, tied to the model's fingerprint, and is
  loaded at the next start (`stand-in tuning loaded: ...` in the log).

Measured 24 Oct (131K context; code / reasoning / prose tokens/s, every answer read):

| Setting | Code | Reasoning | Prose | Answers |
| --- | --- | --- | --- | --- |
| the rank rule below (until 23 Oct) | 31.8 | 39.8 | 30.9 | clean |
| fixed budget 0.25, no re-check | 45-51 | 55-57 | 40-41 | broken words in 2 of 3 prose answers |
| self-tuning + re-check, speed-only search, budget 0.29 | 42.8 | 57.0 | 36.8 | clean; at 0.95 later: broken prose, a looping thought in a short chat |
| self-tuning + re-check + guard + brake, fresh start, 3rd round | 37.3 | 48.0 | 29.1 | 27 of 27 clean, a 6-turn short chat included |

Prompt reading keeps the rank rule below (the re-check is decode only).

**Ranks 3-9 take the budget from the first token** (since 24 Oct evening, after malformed JSON and code were seen when
they took any stand-in until the request's first brake). Measured the same evening on one build: 44.7 against 46.4 tokens/s
averaged over 19 answers (about 4% slower, 10% on the 8.5K run), 12 of 12 JSON and code answers valid either way. Two other
rules stay for testing: `STRATA_ROUTE_TAIL_OPEN=1` (ranks 3-9 take any stand-in until the request brakes) and
`STRATA_ROUTE_BRAKE_STEP=1` (ranks 3-9 open, each brake moves the checked ranks up by one; in the 3-question test it gave
a wrong maths answer, 11:24 / 144 km instead of 11:36 / 156 km).

## The look-ahead learns (default since 24 Oct)

The look-ahead guesses, a few layers early, which experts the next layers will route to, so they can be read from the SSD
before they are needed. Until now the guess was the next layer's own router applied to this layer's input. It now
**learns while it runs** (`STRATA_LA_LEARN`, default on, `=0` off):

- **Three heads**, guessing 1, 2 and 4 layers ahead (the farther the guess, the more time the read has).
- **Each guess = the router's own scores + a small learned correction** (a few "directions" per layer). With the correction
  at zero it is exactly the old guess, so it cannot start worse.
- **Trained on the card, every window, from the real routing** (the router's top-32 scores as the target, a hinge keeping
  the rest below), in one batch at the window's end on a side stream.
- **The size chooses itself.** Per layer and head, versions with 4, 8 and 16 directions compete live with the plain router
  on the picks that matter (ranks below the stand-in floor that are not on the card); one replaces the leader only when it is
  better at 90% confidence, then the losers stop training. A leader that falls behind the plain router is reset to it.
- **It remembers**: `data/predictor-<model fingerprint>.bin`, loaded at the next start. Never in git.
- **Only what will be computed is read** (`STRATA_LA_TAIL_READ=2`, default; `1` = ranks 3-9 only, `0` = off): every guess,
  any rank, goes through the router's own stand-in test on the predicted scores, and a pick a stand-in will serve is not read.
  Ranks below the floor are queued first; ranks 3-9 only while the read queue holds at most one guess per I/O worker.

Live on the abliterated IQ3_S (24 Oct, share of the needed experts not on the card that the guesses held):

| Lead | Plain router | Learned |
| --- | --- | --- |
| 1 layer | 77-79% | 85-88% |
| 2 layers | 66-69% | 76-79% |
| 4 layers | 55% | 66% |

3-question test (code / reasoning / prose, sampled, every answer correct), alternated twice on one build: old defaults
59.0 and 54.4 tokens/s on the mean, the learned look-ahead 58.6 and 58.6 (about +3% and steadier); SSD wait 7.0-7.2 against
6.2-6.5 s. Offline replays and the design notes: `docs/PREDICTOR-EXPERIMENTS.md`, `tools/predictor/`.

Opt-in, measured 24 Oct and left off (same 3-question test, every answer correct):

- **Read-ahead gate** (`STRATA_IO_GATE=1`): each guess position (1, 2 or 4 layers ahead, rank 0-9) learns how often its reads
  are used, and a guess is read only when that rate is at least the drive's load ((reads in progress + queued) / workers).
  It won 3% on 19 Oct, before the learned look-ahead. Now 58.5 and 57.4 against 57.7 tokens/s, and the SSD wait rose (8.8 and
  5.9 against 5.4 s): the learned guesses and the stand-in test leave little waste, and what the gate holds back is often needed.
- **Two kernel switches from 19 Oct** (`STRATA_ROUTER_FUSED=1`: the router's GEMV and top-10 in one launch;
  `STRATA_GR_DOWN_WPC=8|4|2`: warps per staged down block) with their byte-exact bench `bench/micro/hotrod_kernel_ab.sh`:
  ported and measured 24 Oct: `bench/micro/hotrod_kernel_ab.sh` byte-identical on 3 seeds (and the split top-k); live,
  3 questions: default 59.3 and 57.7, fused router 58.8, WPC=4 57.5, WPC=2 59.4 tokens/s - within the spread, no gain.

## Stand-in experts, rank by rank (the rule until 23 Oct; prompt reading still uses it)

For every token, each layer's router picks 10 of 512 experts and ranks them, rank 0 the most important. The ones on the
card run at full speed; one that is not there has to come from RAM or the SSD, and that wait is what limits decode. A
**stand-in** is an expert already on the card that the router scored close to the missing one; using it skips the wait.
How close is measured in **logits**, the router's score: a stand-in "within 1 logit" scores at most 1 below the expert it
replaces.

What the install writes for IQ3_S (21 Oct 2026):

| Rank | Not on the card | Why |
| --- | --- | --- |
| **0, 1** (the two most important) | always the real expert, computed by the CPU from RAM, or read from the SSD first | swapping these broke answers (below) |
| **2** | a stand-in only when one scores **within 1 logit**, else the real expert | the closest margin that was safe for rank 1 in the maths test |
| **3 - 9** | the best stand-in on the card, at any distance | these carry little weight; the 50-question maths and 30-question coding test held up |

The settings, in the config's `env`: `STRATA_ROUTE_RESIDENT=1e30` (stand-ins on, any distance),
`STRATA_ROUTE_RESIDENT_RANKS=3-9` (the ranks that take any stand-in) and `STRATA_ROUTE_RESIDENT_EDGE=1` (the rank just above
them, rank 2, only within 1 logit). Measured with this rule (21 Oct, 131K context): 53.6 tokens/s on the 8.5K-prompt run,
the same as with ranks 2-9 taking any stand-in.

The same rule also works in two more places (both on by default since 21 Oct):

- **The look-ahead skips what a stand-in will cover.** The look-ahead reads ahead the experts the next layers will want.
  A predicted rank-2 expert with a stand-in already on the card within 1 logit is no longer read: the router will take
  the stand-in. On the 8.5K run that skipped about 47,000 reads, cut the experts read ahead from 16,968 to 12,009 and the
  decode's wait for the SSD from 4.7 to 4.0 s: 56.4 against 53.6 tokens/s, and 9 to 13% faster on short prompts.
  **Off by default since 24 Oct** (`STRATA_LA_EDGE_SKIP=1` turns it on): it assumes the rank-2 stand-in rule, but decode now
  follows the confidence budget, which often computes those picks exactly, so the skipped reads became misses (reasoning
  waited 11.8 against 5.2 s; with it off 30.2 / 42.9 / 30.6 against 28.4 / 32.3 / 23.1 tokens/s). It only changes what is
  read ahead, never the answer.
- **Reading the prompt uses stand-ins too.** The prompt is read with the same rule, so a short prompt reads far fewer
  experts from the SSD: 76 to 118-token prompts are read in 1.3 to 1.8 s instead of 2.3 to 2.8 s (35 to 45% faster); an
  8.5K prompt 15.1 against 15.8 s. This changes what the model computes from the prompt; the answers checked (code, a
  maths problem, prose, an 8.5K summary) read clean. `STRATA_PREFILL_STANDIN=0` turns it off.

What we found on the way (20 Oct 2026, every answer read by hand):

| Swapping the top two ranks | Result |
| --- | --- |
| at any distance | answers collapse: empty replies and loops |
| within 3, 5 or 8 logits | collapse: "Let's solve step-by step." repeated, "salt, salt, salt ..." |
| within 2 logits | dropped tokens (a missing `]` in code), broken words, and wrong sums in maths |
| within 1 logit, rank 1 only | 49 of 50 maths answers right, the same as with nothing swapped |
| within 1 logit, ranks 0 and 1 | faster (47.4 against 38.6 tokens/s on short prompts), but leaked a turn marker into an answer at 94K tokens |
| rank 1 within 1 logit, on a card just after a restart | broken answers in an agent session, so the top two are now always real |

The opt-in switches for the top two ranks stay in the engine for testing: `STRATA_ROUTE_RAM_FLOOR=1` (rank 1 within
`STRATA_ROUTE_RAM_FLOOR_MARGIN` logits, default 1) and `STRATA_ROUTE_RAM_FLOOR_RANK0=1` (rank 0 too).
