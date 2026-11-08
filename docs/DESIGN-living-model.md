# The living model: a system that slowly rewrites its router and experts, with a subconscious

*Design note, 8 Nov 2026. Builds on [DESIGN-planned-moe.md](DESIGN-planned-moe.md). Everything is learned from the user's own traffic
(self-learned only: no downloaded expert lists, no outside domain variants).*

## 1. The idea

The model on disk stays the base. Around it the system keeps **deltas it learns while serving and while idle** - router updates,
small corrections that let hot experts stand in for cold ones, earlier routing - and promotes a delta only after it passes a gate on the
user's own past traffic. Over weeks the model it runs drifts toward the efficient design: a small, conscious working set that is known
early, and everything else in a **subconscious** that is still there and comes back when it is needed.

## 2. Three levels of memory

| Level | Where | What it is | How it is used |
| --- | --- | --- | --- |
| **Conscious** | VRAM | the experts the current conversation uses, routed exactly | every token |
| **Preconscious** | RAM | the planner's likely next experts, and the absorbed experts' corrections | read ahead, swapped in at refreshes |
| **Subconscious** | NVMe | masked experts - never deleted, kept at full fidelity | **recalled only when pushed hard**; often recalled -> conscious again |

The mask is the boundary between the subconscious and the rest. A masked expert's work is done by a resident expert plus its learned
correction (§3.1), so it costs nothing - until the model needs the real thing.

### 2.1 Recall: when the model is pushed hard
A masked expert is read from the SSD and used exactly (the same path the stand-in re-check uses today) when any of these fire - all
measured, none guessed from text:
- **Uncertainty:** the token's top-two gap is small against the substitution's risk (the fragile re-check's test).
- **Distance:** the router scores the masked expert far above its substitute (gap beyond the stand-ins' edge) - the correction is being
  asked to cover too much.
- **Disagreement:** an exact re-run shows the substituted token would have flipped, or the stop re-check continued an answer.
- **Demand:** the request asks for depth (a high reasoning effort, or a request flag) - the user pushes it hard on purpose.
- **Drift:** the correction's measured error for this expert has grown since its last check.

### 2.2 Consolidation and forgetting
- **Recall strengthens:** every recall is a training sample for the expert's correction, and an expert recalled often is promoted back
  to the conscious or preconscious level (unmasked, given RAM).
- **Disuse fades:** an expert neither routed nor recalled for long moves down a level; at the periodic repack (§3.4, "sleep") its bits
  may be lowered. It is never deleted - the original stays in the base file.
- **Sleep consolidates:** the idle-time trainer replays the day's samples, refits corrections, re-runs the gate, and repacks.

## 3. The learners (in order of SSD bytes saved)

### 3.1 Absorption: hot experts learn to stand in for cold ones
Each SSD read of a cold expert gives its exact input x and output y for free. A low-rank correction (rank ~16, ~0.1% of an expert) is
fit so that `resident stand-in(x) + B·A·x ≈ y` - plain regression on one expert, no backpropagation through the model. It extends today's
stand-in learning (one scale per pair) to a matrix. When an approximation passes the gate, its expert can be masked into the subconscious.

### 3.2 Router co-adaptation
Low-rank updates to each layer's router matrix (and the per-expert bias of "Rewiring Experts on the Fly"), trained against the exact
router's choices (KL anchor) plus a byte cost per tier. Routing concentrates on what is conscious - only as far as the gate allows.

### 3.3 Routing that moves earlier
The look-ahead already predicts layers +1, +2, +4 (89 / 80 / 72% of the needed experts today, plain router 77 / 65 / 54%).
- **Confident prediction is used:** when the learned guess for layer l is confident, the engine routes on it and keeps layer l's own
  router as a check; a mismatch is a sample (as the stop re-check is).
- **The router is rewarded for agreeing with the early guess** (in §3.2's loss), so router and predictor converge from both sides.
- Over weeks the agreement rises layer by layer: routing decided 1, then 2, then 4 layers early - the pre-gated design, learned live.

### 3.4 Sleep: the periodic repack (weekly)
Masked experts' corrections are folded into the pack's side file; bits are re-allocated by measured use (more for the conscious set,
fewer for the long-unused); the planner's sets are re-clustered. Raising bits needs the original weights (the model's own download).

## 4. The gate (every delta, every promotion)
A change is promoted only when, replayed on the user's own past traffic: the divergence from the exact model stays under its bound, the
flip, stop-continue and recall rates do not rise, and a sample of answers is read by hand (degeneration and degradation reported
apart). Otherwise it is rolled back. Deltas are versioned on the NVMe; any version can be switched off with one flag.

## 5. Build order (one concept at a time, each tested before the next)
1. **Capture:** sample cold-expert inputs and outputs, and every layer's router logits, into a ring on the NVMe.
2. **Delta framework:** base + versioned deltas applied in the kernels; the gate; rollback; one flag per learner.
3. **Absorption (§3.1)** with the recall rules (§2.1): measured as SSD bytes per token, tok/s, recall rate, quality by hand.
4. **Router co-adaptation and early routing (§3.2, §3.3):** measured as early-guess agreement per layer over the weeks.
5. **Consolidation, forgetting and sleep (§2.2, §3.4).**

## 6. What tells us it works
- SSD bytes per token falling week by week, decode tok/s rising, with flips, stop-continues and hand-read quality flat.
- The recall rate settling low (the subconscious is rarely needed) but non-zero (it is still there).
- Early-routing agreement per layer rising toward the router's own choice.
