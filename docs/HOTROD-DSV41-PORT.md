# Strata-HotRod on DeepSeek-V4.1-Flash

Everything Strata-HotRod built for Qwen3.8-Flash-Next, ported to upstream Strata's DeepSeek engine (`strata-dsv41`, `src/dsv41`, from
Project Maya's maya-ds4). Branch `upstream-dsv41`, release `v0.2.7.1-hotrod.1`. **Run and measured on the model** (31 Oct 2026).

The model: Starw1's DeepSeek-V4.1-Flash-1017 uncensored, the GSQ-RCO IQ2_XXS backbone (138.6 GB; its experts IQ1_M / IQ2_XXS /
IQ2_XS per layer, 1.94 bits a weight) with DeepSeek's own MXFP8 Engram (202.8 GB), the DSpark drafter (8.0 GB, Q8_0, a separate file:
the backbone has no drafter tensors) and Unsloth's 5.8 MB metadata shard. Setup knows it as `--family deepseek41 --model
Starw1-2.00bpw-MXFP8`. The PC: RTX 3060 12 GB, i5-12400F, 31 GB RAM, PCIe 3.0 NVMe.

## Speed on this PC (answer tokens/s, the same 3 questions, 32K context, every answer read)

| Setting | tokens/s |
| --- | ---: |
| Upstream as it ships (the learner off) | 5.05 |
| **The default here: the VRAM ring + stand-ins** (`DS4_RING=32 DS4_RING_PIN=1 STRATA_STANDIN=1`) | **5.2 / 5.3 / 5.4** in three runs, against 4.8-5.0 in the same sessions |
| The learner alone (on once its file exists) | 4.9-5.03 (no measurable gain) |
| The ring alone (plain / pinned copies) | 3.4 / 3.6 |
| Stand-ins alone (with the re-check, the tuner, the exact read-ahead) | 3.9-4.2 |
| The DSpark drafter capped at 0.05 / 0.1 of its experts on the card | 4.96 / 4.97 (drafts kept 80-91%) |
| The drafter uncapped | 4.72 |
| K/V streaming at 32K (`STRATA_KV_HOST=1`) | 4.82 |
| Upstream's `DS4_STORE_ZC=1` | 4.32 |

Why: 270 of the 15,360 routed experts (2.65 GB) fit the card next to 6.2 GB of dense weights, so ~75% of the picks are computed on the
CPU lane (from the RAM store or in place from the files); the PCIe link sits ~99% idle. The ring copies the learner's guesses for the
next 1-2 layers into card slots on its own stream and thread and a pick uses a slot only if its copy has landed; with the stand-ins the
CPU lane's share falls to 34-37%. Each token still needs ~2 GB of expert weights from RAM and the NVMe, which is what the ring cannot
remove (a copy reads the same bytes the CPU would). Upstream's own 3060 figure (9-11 tokens/s) is on 46 GB of RAM.

CPU cost per expert by type (the timeline, 27,195 layer rows): IQ1_M 572 us (74 us/MB, the smallest experts and the slowest), IQ2_XXS
435 us (48 us/MB), IQ2_XS 475 us (46 us/MB). `DS4_RING_COST` uses this live.

## What is ported and how it maps

An expert that is not on the card is computed **on the CPU** here (read from RAM or the SSD in place); the Qwen engine fetches it to
the card. So the ideas carry over, but each is reworked:

| Qwen feature | On DeepSeek | Switch (engine default) |
| --- | --- | --- |
| Stand-ins by confidence | in the router kernel: ranks 0-1 exact, rank 2 within 1 logit, from rank 3 `w x gap <= budget`; the weight kept | `STRATA_STANDIN=1` (off) |
| Exact re-check of a close call | the token run again at the same position without stand-ins | `STRATA_FRAGILE_RECHECK` (on with stand-ins) |
| Self-tuning budget, brake | `ds4_standin_tune.hpp`, `<pack>/standin-tune.txt` | on with stand-ins |
| Learned look-ahead 1 / 2 / 4 layers | `ds4_lalearn.cu`, `<pack>/hotrod-predictor.bin` (learned on your PC; not shipped) | `STRATA_LA_LEARN` (on when the file exists) |
| Read ahead only exact picks | the learned guesses into the RAM store | `STRATA_LA_TAIL_READ=2` |
| **VRAM ring** (new here) | the guesses copied ahead to card slots, used only if landed; `DS4_RING_PIN`, `DS4_RING_RANKS`, `DS4_RING_COST`, `DS4_RING_STICKY` / `DS4_RING_MOVE` (experimental: copies kept across tokens are reused ~6x, but fills starve) | `DS4_RING=<slots>` (off) |
| Drafter's card share (new here) | DSpark's hottest experts on the card, the rest from RAM like the trunk's | `DS4_DSPARK_VRAM=<f>` (all) |
| K/V streaming | the compressed K/V rows in host-mapped RAM | `STRATA_KV_HOST` (on above 131K) |
| Learned VRAM reserve | `<pack>/vram-reserve.txt` | without `--reserve-mib` |
| Request-end reports | stand-ins, re-checks, the budget, the learner, the ring - printed and saved after every answer | on |
| Monitor, stalled-client timeout | ported | on |
| Three tiers, chunked prompts, saved conversations, DSpark | upstream's own | upstream |

Every switch, flag and server key: [FEATURES-deepseek.md](FEATURES-deepseek.md).

## Running it

1. `scl enable gcc-toolset-13 -- ./setup.sh --family deepseek41 --model Starw1-2.00bpw-MXFP8 --gguf-dir <the folder with the files>`
   (setup never deletes your files; it checks them against Starw1's pins). The pack is a few MB beside the files: no weights copied.
2. In the config setup writes, the default set: `"env": { "DS4_RING": "32", "DS4_RING_PIN": "1", "STRATA_STANDIN": "1" }` (a blank
   value means the engine's default). The drafter: a pack made with `--dspark-gguf`, `--dspark on`, and `DS4_DSPARK_VRAM=0.05`.
3. The learned VRAM reserve counts the ring's slots as planned use (they are taken after the expert placement), so it does not grow
   because of them: 270 experts stay on the card start after start. A reserve raised by an engine before this fix (e.g. 1244 MiB)
   comes down only slowly; set its line in `<pack>/vram-reserve.txt` back once (926 MiB on a 3060 at 32K).

## Tests

| Test | Runs on | State |
| --- | --- | --- |
| `ds4_timeline_test`, `ds4_standin_tune_test` | CPU | pass |
| `ds4_standin_test`, `ds4_lalearn_test` | GPU | pass (31 Oct) |
| `tests/terminal` | CPU | pass (4 need the `textual` package) |

## Rules kept

Answers read by hand (degeneration and degradation reported apart). Never compile during a speed leg. One model on the GPU at a time.
No regex, no pattern matching on tokens. A switch becomes a default only when it wins its A/B and the answers stay clean.
