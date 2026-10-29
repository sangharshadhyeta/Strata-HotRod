# Porting Strata-HotRod's additions to the DeepSeek-V4.1 engine

The model: Starw1's DeepSeek-V4.1-Flash-1017 uncensored, GSQ-RCO IQ2_XXS backbone (138.6 GB) + DeepSeek's MXFP8 Engram (202.8 GB),
in `/mnt/nvme/models/deepseek41-flash/GSQ-RCO-IQ2_XXS`. The PC: RTX 3060 12 GB, i5-12400F, 31 GB RAM, PCIe 3.0 NVMe.
The engine: upstream Strata's `strata-dsv41` (`src/dsv41`, from Project Maya's maya-ds4), this worktree, branch `upstream-dsv41`.

## How this engine differs from our Qwen engine

| | Qwen engine (ours) | DeepSeek engine (upstream) |
| --- | --- | --- |
| an expert not on the card | read from RAM / SSD **to the card**, or a stand-in | computed **on the CPU** ("the CPU lane"), read from RAM or the SSD in place |
| RAM tier | ours (card -> RAM -> SSD, learned reserves) | `ExpertStore` (`--ssd-cache`, auto; LRU / LFU slots, unbuffered reads) |
| guessing ahead | router + a learned low-rank correction, 1 / 2 / 4 layers ahead | the next layer's gate on this layer's FFN input (plain router); **off with the store**: measured slower (62% right, the SSD the narrow part) |
| a missing expert left out | stand-in by confidence (`w x gap <= budget`, ranks 0-2 exact, flip re-check) | `STRATA_DS4_DISK_SKIP_W` (opt-in): a light expert not in RAM **dropped**, no re-check; and "faster mode": 5 of 6 experts on cards <= 16 GB |
| long prompts | layer by layer from the SSD | the chunked "flow" (same idea, exists) |
| conversations kept | saved state | `ds4-slots` (exists) |
| speculative decoding | MTP | DSpark drafter (off on a 3060: auto) |
| server | ours: 60 s client timeout, always-on timeline | neither |

What already exists there is not ported again (memory rule: read what the tree has first): the layer-by-layer prompt path, the RAM tier,
the saved conversations.

## Steps, one at a time (each: build, test on the model, measure, then the next)

0. **Baseline.** Build `strata-dsv41` (sm_86). Pack with `tools/dsv41/pack_gguf.py --engram <MXFP8>` + the shipped `profile-2.00.bin`;
   `--pack-info`; the 16-token smoke test (Paris). Then the 3-question speed at upstream's defaults, answers read by hand.
1. **Server.** Upstream's `serve/server.py` is the one that speaks DSML (tool calls) to this engine: port our `CLIENT_TIMEOUT_S = 60`
   handler timeout, and the always-on decode timeline (the engine writes its windows; the Monitor page reads them).
2. **Learned look-ahead.** The guess the device already makes (layer L+1's gate on L's FFN input) + our low-rank correction trained live
   (`native_router.cu` learner: top-32 centred MSE + hinge, AdamW, batched on a side stream, rank by live contest, 2-failure reset),
   heads 1 / 2 / 4 layers ahead, saved per model fingerprint. Goal: lift the right-rate from upstream's 62% to where reading ahead with
   the store pays (our Qwen: 81 -> 91% one layer ahead). Then `DS4_AHEAD` back on with the store, measured.
3. **Stand-ins.** On decode, a CPU-lane pick that must come from the SSD (not on the card, not in the store), at rank >= the floor
   (`STRATA_ROUTE_RESIDENT_RANKS`, 3 here: ranks 0-1 exact, rank 2 within `STRATA_ROUTE_RESIDENT_EDGE`), is replaced by the best-scoring
   expert already on the card when `w x gap <= budget` (gap on the router's selection score, bias included); its weight kept. The device
   publishes the scores of the top 16 for that. A token a stand-in could have flipped is computed again exactly. Replaces DISK_SKIP_W's drop.
4. **Self-tuning budget + brake**, saved per model fingerprint (`standin-tune-<fp>.txt`), as on Qwen.
5. **Read ahead only exact picks** (`STRATA_LA_TAIL_READ=2`): the store reads ahead only the guesses step 3 will compute exactly.
6. **Setup / README**: the family in our `setup.py`, the start script, the Monitor for this engine.

Later, measured first: K/V streaming (this model's K/V is compressed; at long contexts it takes card memory from the experts), the
5-vs-6 experts default against stand-ins.

## Rules kept

Answers read by hand (degeneration and degradation reported apart). Never compile during a speed leg. 8095 serves one model at a time:
a DeepSeek test takes the card from Qwen, and 8095 goes back to its defaults after. No regex, no pattern matching on tokens.
