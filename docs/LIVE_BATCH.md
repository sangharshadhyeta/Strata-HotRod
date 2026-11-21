# Live batch: slots that come and go while the engine runs (design, 21 Nov 2026; not built)

## Goal
No batch by default (the best single-stream path: prefill 335-348 tok/s, decode 22-26). When requests overlap, bring up a few small slots
(about 2,048 cells each, 0.15 GiB per slot) with VRAM taken from the expert cache; when they end, give it back. Nothing restarts.

## What the measurements so far say (Qwen UD-Q4_K_XL, 12 GB card; docs/PORTING_STRATAQ.md)
- Slots as large as `--max-context` cost 40% of the expert cache and 62% of prefill; a cap of 2,048 cells costs about 7% of the cache and, with the single-GPU piece
  fix, no prefill.
- Two streams in batch mode total about 19 tok/s against 22-25 for one stream alone (one token per stream per window against MTP's 2.4). The break-even is expected at
  3 streams; the multi-stream measurement (1-8 streams, with and without `--batch-mtp`) decides how many slots are worth bringing up.
- The elastic cache works in our RAM-tier mode when the RAM-tier refusal is skipped (STRATA_VRAM_COMPLEMENT=1): the cache 1508 -> 1331 slots in 5 ms, no errors, decode unchanged.

## The command: `BATCHSET <slots> <cells>` (engine), `POST /v1/batch` (server)
Applied only at a quiet moment: no slot decoding (`!batch_on()`), no prompt loan out, no pipelined groups (the same preconditions as `VRAM`).

1. **Park what the slots hold.** Every slot that caches a conversation is copied into the main session (`copy_from_slot`) and parked in the conversation cache
   (`park_current`, 100-270 ms even for 29K tokens), so the next turn continues from the parked context and does not re-read it. A conversation that cannot be parked
   (no budget) is dropped and said so in the log. Never release a slot's memory with a conversation in it that was not parked.
2. **Release the slots.** New `Verifier::release_slots()`: destroy `exec_bm_` and `commit_bm_` (the batch graphs are captured against the old slot pointers), free `arena_b_`
   and `h_commitb_`, clear `slots_`. `init_slots` today allocates them again on every call and never frees: calling it twice would leak and run stale graphs.
   Free each slot session's device buffer (the engine does not keep the `cudaMalloc` pointers today: keep them) and its pinned host K/V.
3. **Give the memory back to the cache,** `VRAM` with the start's reserve (grow), then **take what the new slots need** (`VRAM` with free = need + reserve, shrink).
   Needs the RAM-tier refusal lifted, which the elastic test supports (it is a switch today: STRATA_VRAM_COMPLEMENT).
4. **Carve the new sessions** (`session_bytes` / `session_init` at `cells`), `init_slots`, rebuild `bs`, the groups, and print `BATCHSET slots=N cells=C` for the server.
5. **Server:** resize `slot_q`, `slot_busy`, `slot_held`, `slot_used`, `slot_live`, `slot_group` and `self.batch` / `self.slot_cells` on that line; admit to slots only
   when the engine says so. A controller (later) calls `BATCHSET N` when at least N requests overlap for a few seconds and `BATCHSET 0` after a quiet period, with hysteresis.

## Risks and open questions
- `init_slots` / graph capture while serving: untested; each (rows, S) graph is captured at first use ("captured the batch window over slots ..."), so the first windows after a
  re-carve pay a capture.
- The draft head per slot (`--batch-mtp`) holds its own state; the first version leaves it out.
- The elastic cache refuses while a prompt loan is out; BATCHSET must wait for the loan to return.
- Every static in the batch path that read `o.batch` once (`static const` values) must be checked: the count now changes at run time.

## Order
1. Multi-stream measurement (running) -> slots worth bringing up. 2. Continuity test (running) -> parking keeps the context. 3. `release_slots()` + `BATCHSET` with fixed `cells`,
tested by hand through `POST /v1/batch`. 4. The server's arrays. 5. The controller, scored with the two-measure rule.
