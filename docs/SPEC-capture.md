# SPEC: capture - the record every learner of the living model trains on

*Step 1 of [DESIGN-living-model.md](DESIGN-living-model.md) §5. Qwen engine first (Strata-mask), DeepSeek after. Opt-in:
`STRATA_CAPTURE=1` (off: nothing is recorded and nothing changes). Local only, never uploaded.*

## Goal
Record, while serving and with no measurable cost to decoding, the five kinds of data the learners need:

| Record | For | Size (about) |
| --- | --- | --- |
| A. Conversation record | the RLHF pairs (§7.1), the held-out panel's own-traffic split (§6) | the token ids, ~4 B per token |
| B. Preference events | DPO pairs, reward-model groups (§7.1, §7.3) | a few ids per event |
| C. Activation summaries | contrastive directions -> Steer2Edit edits (§7.2) | ~50 KB per answer |
| D. Router samples | planner, router deltas, early routing (§3.2, §3.3) | ~25 KB per sampled token |
| E. Cold-expert samples | absorption corrections (§3.1) | ~8 KB per sample |

## A. Conversation record (server side, `serve/`)
Per request: request id, time, the conversation key (the prompt cache's key: tokens up to the third `<|im_start|>`, by id), the prompt's
token ids, the answer's token ids, the finish reason, and the engine's per-request counters already printed (stop re-checks, flips,
recalls when they exist). Written by the server after the reply, outside the request's path.

## B. Preference events (server side)
- **Regenerate:** a request whose prompt token ids equal an earlier request's prompt ids (same conversation key) - the earlier answer is
  marked *replaced*; the answer the conversation later continues with (the next prompt contains its ids) is *kept*. A kept/replaced pair
  is a human preference pair. Id equality only.
- **Explicit:** an optional `preference` field on the API (`{"request": <id>, "chosen": true|false}`), recorded as given.

## C. Activation summaries (engine)
For every answer: at a fixed set of layers (every 4th, configurable), the **sum and count** of the residual stream over the answer's
tokens, accumulated **on the device** inside a kernel that already runs per layer (or one tiny kernel on the side stream), copied to the
host once at the request's end. fp32 sums, hidden-size vectors. No per-token copies.

## D. Router samples (engine)
Every Nth decode token (default 64): per layer, the router's top-32 ids and logits. Copied with the look-ahead learner's existing
per-window data where it already crosses to the host; otherwise one async copy on the side stream.

## E. Cold-expert samples (engine)
For a sampled 1-in-64 of the experts read from the SSD (not from VRAM or RAM): the expert id, its input `x` and its output `y` (fp16),
copied async after the expert's computation. Absorption fits `stand-in(x) + B·A·x ≈ y` from these.

## Storage
- A ring of segment files under `/mnt/nvme/strata-data/capture/` (owner-only, as the prompt cache), each with a header (format version,
  model and engine fingerprint, the prompt cache's rule).
- Bounded: `capture_gib` (default 4 GiB, and never more than 1/16 of the drive's free space) and `capture_max_age_d` (default 14 days);
  the oldest segment goes first.
- Written by one writer thread from a bounded queue; when the queue is full the sample is **dropped and counted**, never waited for.
- A Python reader (`serve/capture_read.py`) for the idle trainer.

## The overhead rule
The request line shows `capture: N KB written, K dropped`. Accepted only if, on the same six-prompt speed test and a long-context
agent session, decode tok/s with `STRATA_CAPTURE=1` is within the run-to-run noise of `=0` (target: < 1%); a dropped count that is
high means the queue or the sampling rates are too high, not that decoding should wait.

## Tests
- Unit: record round-trip (write, read, header checks); ring cap and age; regenerate detection (same prompt ids -> replaced/kept pair;
  different conversation -> no pair); the queue drops rather than blocks.
- Engine: activation sums equal a reference sum over the same tokens on a short prompt; router samples equal the router's own top-32 on
  the sampled tokens; a cold-expert sample's `y` equals the expert's output.
- With `STRATA_CAPTURE` unset, the server tests (364) pass unchanged and nothing is written.
