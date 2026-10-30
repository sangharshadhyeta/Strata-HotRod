# Strata-HotRod on DeepSeek-V4.1-Flash

Everything Strata-HotRod built for Qwen3.8-Flash-Next, ported to upstream Strata's DeepSeek engine (`strata-dsv41`, `src/dsv41`, from
Project Maya's maya-ds4). Branch `upstream-dsv41`. **Built and unit-tested; not yet run on the model** (the files are still downloading).

The model: Starw1's DeepSeek-V4.1-Flash-1017 uncensored, the GSQ-RCO IQ2_XXS backbone (138.6 GB) with DeepSeek's own MXFP8 Engram
(202.8 GB), the DSpark drafter (8.0 GB) and Unsloth's 5.8 MB metadata shard. Setup knows it as `--family deepseek41 --model
Starw1-2.00bpw-MXFP8`. The PC: RTX 3060 12 GB, i5-12400F, 31 GB RAM, PCIe 3.0 NVMe.

## How this engine differs from our Qwen engine

An expert that is not on the card is computed **on the CPU** here (read from RAM or the SSD in place); ours fetches it to the card. So
the ideas carry over, but each is reworked:

| Qwen feature | On DeepSeek | Switch (default) |
| --- | --- | --- |
| Stand-ins by confidence | in the router kernel (`route_publish_kernel`): an off-card pick takes the best card-resident expert the token did not pick; ranks 0-1 exact, rank 2 within 1 logit, from rank 3 `w x gap <= budget`; the weight kept | `STRATA_STANDIN=1` (off) |
| Exact re-check of a close call | the token is run again at the same position without stand-ins (each position's ring row, compressor slot and index keys are written, not added to, so no snapshot is needed) | `STRATA_FRAGILE_RECHECK` (on with stand-ins), `STRATA_FRAGILE_K0` |
| Self-tuning budget, brake, learned k | `ds4_standin_tune.hpp`: interleaved A/B (400 tokens an arm), quality gate, clean ceiling, per-request brake; `<pack>/standin-tune.txt` | `STRATA_ROUTE_BUDGET_AUTO`, `STRATA_FRAGILE_BRAKE` (on with stand-ins) |
| Learned look-ahead 1 / 2 / 4 layers | `ds4_lalearn.cu`: the router's guess on this layer's FFN input plus a live-trained low-rank correction; `<pack>/hotrod-predictor.bin` | `STRATA_LA_LEARN` (on when the file exists) |
| Read ahead only exact picks | the learned guesses (1, 2 and 4 ahead) read into the RAM store, the ranks below the floor only | `STRATA_LA_TAIL_READ=2` (with learner + stand-ins) |
| K/V streaming | the compressed K/V rows in host-mapped RAM, read over PCIe by the sparse attention; the index keys stay on the card | `STRATA_KV_HOST` (on above 131K context) |
| Learned VRAM reserve | `<pack>/vram-reserve.txt`, keyed by GPU, VRAM, context, K/V placement, drafter | without `--reserve-mib` |
| Monitor: timeline, experts map | the engine writes both files (`ds4_timeline.hpp`, `residency_write`); the server and page ported | on |
| Server: drop a stalled client | `CLIENT_TIMEOUT_S = 60` | on |
| Three tiers, RAM tier from free memory | upstream's own: the VRAM cache and the RAM store (`--ssd-cache`, auto-sized, gives RAM back when the PC runs short) | upstream |
| Layer-by-layer prompts | upstream's chunked flow | upstream |
| Conversations resumed | upstream's slots (on disk, survive restarts) | upstream |
| Draft and check | upstream's DSpark drafter (downloaded; upstream measured +2-13% on a 3060 with `--dspark on`) | `--dspark auto` |

## Tests (no model needed)

| Test | Runs on | State |
| --- | --- | --- |
| `ds4_timeline_test` | CPU | passes |
| `ds4_standin_tune_test` | CPU | passes (the fake engine's knee 0.477, out of its flip region; brake; k; save/load) |
| `ds4_standin_test` | GPU | built, run when the card is free of 8095 |
| `ds4_lalearn_test` | GPU | built, run when the card is free of 8095 |
| `tests/terminal`: timeline, residency, drain, terminal, dsv41 setup and planner | CPU | pass (4 need the `textual` package) |

## When the files are in

1. `scl enable gcc-toolset-13 -- ./setup.sh --family deepseek41 --model Starw1-2.00bpw-MXFP8 --gguf-dir /mnt/nvme/models/deepseek41-flash/GSQ-RCO-IQ2_XXS`
   with our `strata-dsv41` in `engine/` (`--gguf-dir`: setup never deletes the person's files; it checks them against Starw1's pins).
2. The 16-token smoke test (Paris), then the 3-question speed at upstream's defaults, answers read by hand.
3. Each switch above against that baseline, one at a time; a switch becomes a default only when it wins and the answers stay clean.
4. ThinkVest on this engine instead of gemma-4.

## Rules kept

Answers read by hand (degeneration and degradation reported apart). Never compile during a speed leg. One model on the GPU at a time:
a DeepSeek test takes the card from Qwen, and 8095 goes back to its defaults after. No regex, no pattern matching on tokens.
