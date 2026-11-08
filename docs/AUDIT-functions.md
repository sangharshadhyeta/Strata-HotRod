# AUDIT: does each function actually do its job? (Qwen, 8095's live log)

*Started 8 Nov 2026. Method: for every learner, tuner and guard that is on by default, state what it should change, read its own
numbers in the live log or its saved table, and say whether it acts. A function that does not gets a cause (from the code, file:line),
one measurable objective, a method, tests and a live check; a Haiku builder implements it on its own branch from main. Evidence below is
the session that started 8 Nov 11:04 (413 requests, 828 prompt lines) unless noted.*

| # | Function | Should | Evidence | Verdict |
| --- | --- | --- | --- | --- |
| 1 | Stand-in learn (STRATA_STANDIN_LEARN=1) | learn each stand-in's correction and risk | table: 5,566 pairs, 12 ready (32 needed), 85% of samples at layer 0 | **starved** - SPEC-standin-samples.md, branch standin-samples (building) |
| 2 | Stand-in balance (STRATA_STANDIN_BALANCE=1) | keep hard-to-replace experts in VRAM, let easy ones go | 4,377 rounds, 0 kept, 0 given up | **never acts** (needs #1) |
| 3 | Fragile re-check noise control | measure the flip rate that is noise (exact vs exact), so stand-in flips are judged against it | `control 0 re-checks 0 flips` in all 413 requests; needs a row no stand-in moved, which ranks 3-9 everywhere never leave; with no samples ctrl_hi = 1, so k can never rise (verify.cpp ~5030) and the recheck brake never fires | **never runs** - SPEC-control-windows.md, branch control-windows (building) |
| 4 | Mask learn's comparison | flips with a mask move vs without, to see what the mask costs | `with a mask move 22/91, without 0/0`; "without" comes only from mask control windows, gated on the opt-in STRATA_MASK_RESIDENT (verify.cpp ~4707), so mlr_update's quality rule (~6640) never runs and p is set by the speed test alone | **no quality brake** - SPEC-control-windows.md (building) |
| 5 | Auto budget tuner (stand-in budget) | move the budget only when a test shows a real win | `won = sc >= sb * 1.02` (verify.cpp ~5245): a 2% point gap, no noise test; per-test sd 1.79 tok/s gives ~35% false wins; 11 moves in 35 tests = chance | **decides on noise** - SPEC-tuner-paired-test.md, branch tuner-paired-test (building) |
| 6 | Auto mask test (mask p, g) | the same, for the mask's settings | same rule plus one repeat; sd 0.91, ~25% false first wins; alternates (0.550,1.0)/(0.575,1.0) | **decides on noise** - SPEC-tuner-paired-test.md (building) |
| 7 | VRAM headroom at request start | keep >= 512 MiB free (driver paging, #2065) | 588 warnings in 408 of 767 sessions (lowest median 340 MiB, 15 sessions <= 50 MiB). The expert cache is sized with `reserve = vram_reserve + prefill + draft head + pipeline` (generate.cpp ~6273); the look-ahead learner (~163 MiB), verifier buffers (~75 MiB) and CUDA graphs (~94 MiB) come after. But warned requests decode at median ~31 tok/s (n = 10,574): no collapse seen | **accounting gap, no measured harm** - fix = count those allocations in the reserve (costs ~165 of 2,476 cache slots); the user's call |

## Working as intended (numbers)
- Learned look-ahead: leader hit rate d=1 86.5% vs plain 74.7%, d=2 76.6% vs 61.4%, d=4 67.3% vs 49.7%; never stalled the main stream.
- Decode expert cache: 95.4% hit; KV streaming 98.1% of block reads in VRAM; route tail skip and look-ahead tail read act (121k skips,
  122k guesses left to stand-ins).
- Mask: 136,668 masked picks seen, 114,130 moved (whether that is a quality cost is item 4).
- PCIe balance: re-measured the CPU pool (93 us per missed expert) and moved pcie_frac 0.55 -> 0.40.
- Stop re-check: runs (1 kept this request; 16 of 586 continued earlier).

## Next rounds
DeepSeek's engine (ds4: lines), the server's own functions (prompt cache, MCP, effort handling), and the RAM tier (promotion: 11-37
promoted per request, hits 8-21 - is that the intended scale?).
