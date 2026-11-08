# Planned MoE: a model whose expert needs are known ahead, for ~100 tok/s on one consumer GPU

*Design note, 8 Nov 2026. Target: Strata-HotRod's PC class - RTX 3060 12 GB, 31 GB RAM, one NVMe (~2.4 GB/s measured), PCIe ~25 GB/s.*

## 1. The budget

100 tok/s leaves 10 ms per token. Whatever a token needs must already be close:

| Source | Bandwidth | Bytes per token at 100 tok/s |
| --- | --- | --- |
| NVMe | ~2.4 GB/s | ~24 MB |
| RAM -> GPU (PCIe) | ~25 GB/s | ~250 MB |
| VRAM | ~360 GB/s | ~3.6 GB (the compute's read) |

A perfect predictor hides latency; it does not move bytes faster. So the model must be built so that **the bytes a token needs
that are not already in VRAM are tiny**, and so that **the system knows them early**. Our own numbers show the gap today: Qwen3.8-Flash-Next
(78 GB pack, ~1.8x RAM+VRAM) hits 97% of experts in cache and decodes at ~33 tok/s; DeepSeek-V4.1-Flash (341 GB, ~8x) decodes at ~5 tok/s.
The learned look-ahead already guesses 89 / 80 / 72% of the needed experts 1 / 2 / 4 layers ahead (plain router 77 / 65 / 54%) - and still,
on Qwen, removing ~20% of the SSD bytes (the resident mask) did not move decode speed: the remaining cost is spread across PCIe, the
CPU path and the miss tail.

## 2. What the research says (Nov 2026)

- **Routing can be made exact ahead of time.** Pre-gated MoE (ISCA 2024) trains layer N's gate to choose layer N+1's experts; with the
  experts in CPU memory it runs within ~23% of an oracle and needs ~4.2x less GPU memory. Read-ME (NeurIPS 2024) decouples the router
  from the backbone so routing is computed before the layers run (lookahead scheduling, expert-aware batching and caching); it builds the
  MoE from a dense model with ~1B tokens of training. Pre-attention routers reach 93-98% accuracy on DeepSeek-V2-Lite, Qwen3-30B and
  Phi-mini-MoE, against 54-79% for cross-layer predictors.
- **Locality can be trained, but it is not free.** ReMoE adds reuse and working-set losses with a KL anchor to the pretrained router;
  they concentrate routing and raise load imbalance. A pre-registered study on 137M-340M MoEs ("Cacheable by Design?") cut misses by up
  to 60%, but every setting broke its 1% perplexity gate, and the tax did not shrink with scale; trained locality plus training-free
  cache-aware rerouting reached ~80% fewer misses at <=3.4% perplexity. StickyMoE reports the loss as near free, but only on small
  single-domain models. Natural routing consistency differs widely between models ("Not All Models Suit Expert Offloading", ICLR 2026).
- **Coarser routing can cost nothing or help.** Task-level MoE beat token-level routing by +3.6 BLEU at comparable inference cost;
  SeqTopK moves the expert budget from the token to the sequence; DoMoE's domain tables raise the expert hit ratio 1.22x.
- **Systems confirm where the time goes.** Expert loading is reported at over 70% of decode time in SSD-serving runtimes; SSD-LLaMA runs
  trillion-parameter MoEs at 1+ tok/s on a consumer PC. A caution: one 2026 system found a 64.7%-recall lookahead bought 0.3% end to end -
  prediction only pays when the bytes it schedules are few enough to arrive in time.

**Conclusion:** per-token stickiness fights quality; *segment-level* working sets (a conversation or a turn uses a subset) plus *exact*
pre-gated routing inside the subset is the direction with evidence on its side.

## 3. The design

### 3.1 Architecture
- **A conversation set per layer (the planner).** A small planner head reads the context and picks, per layer, a subset S of the routed
  experts (e.g. 16-32 of 256). It runs at the start of a turn and every K tokens (K ~ 256), with hysteresis: a refresh may change at most
  M experts per layer, so a refresh moves a bounded, known number of bytes.
- **Exact routing inside S, decided k layers early.** The token router is restricted to S and pre-gated: layer l's experts for a token are
  chosen at layer l-k (k = 2-4, from Pre-gated MoE / pre-attention routers). Nothing is guessed on the critical path.
- **A shared expert always in VRAM**, small, fine-grained routed experts (better caching, closer substitutes), a hybrid linear-attention
  stack with a few full-attention layers (small KV at 256K), and an MTP head for speculative decoding - draft tokens share S, so drafting
  adds no misses.

### 3.2 Tiers at run time
- **VRAM: the current S** (exact routing inside it: no SSD, no RAM on the decode path).
- **RAM: the planner's likely next sets** (prefetched from the SSD in the background between refreshes).
- **SSD: everything else.**

### 3.3 The numbers (illustrative, to be measured)
300B total, ~3B active (top-6 small experts + shared), 3 bits/weight -> ~112 GB pack, ~100 GB of routed experts. With S at ~8% of the routed
experts, S is ~8 GB and fits VRAM beside the dense weights and KV. Per token the GPU reads ~1.1 GB from VRAM -> ~3 ms at 360 GB/s;
with attention and overheads ~5-8 ms -> **~125-200 tok/s ceiling, ~100 tok/s with margin**, before MTP's gain. A refresh changing 64
experts per layer at ~0.4 MB each is ~1.2 GB from RAM over PCIe -> ~50 ms every 256 tokens (2% of the time), and the SSD refills RAM in the
background at 2.4 GB/s.

### 3.4 Training objectives
1. The language-model loss.
2. **Coverage**: the unrestricted router's top-k should fall inside S (the planner learns the conversation's needs).
3. **Refresh cost**: lambda x (bytes changed between consecutive sets) - hardware-aware, in bytes, not in "switches".
4. **Load balance across segments**, not across tokens - experts are used evenly over the corpus while each conversation stays narrow;
   this is the point where segment-level design avoids the token-level locality tax.
5. Pre-gating trained by distillation from the ordinary router, then jointly.

## 4. How we could get there (cheapest first)

1. **Measure first (no training, days).** From our routing traces, compute each turn's working set: how many experts per layer cover 90%
   of a turn's routing mass. Globally Qwen needs ~81% of its experts for 90% (measured 1 Oct); per *turn* it may be far fewer. If a turn
   needs <=10-15% of the experts, the design fits the hardware; if it needs 50%, it does not, whatever the predictor.
2. **Engine-only approximation (no training).** Build S at inference with what we have: the look-ahead learner as planner, the mask as the
   restriction to S, stand-ins as the in-S substitutes - with the top-rank logit check (which fixed the early stops) and the stop
   re-check. Measure quality by hand and new-bytes per token.
3. **Upcycle an existing open MoE (rented GPUs, far below pretraining).** Add the planner and the pre-gate to a Qwen-Next-class model and
   fine-tune with a KL anchor to its own router (Read-ME needed ~1B tokens for its conversion; ReMoE uses the same anchor).
4. **Train from scratch** with all five objectives: research scale, not ours.

## 5. Risks
- The locality tax: the 137M-340M study says miss reduction and quality are coupled; segment-level sets with segment-level balance are
  the bet that it is smaller there - unproven.
- Agent work switches topics within a turn (code, logs, tools): S may need refreshes more often than every K tokens.
- Prefill is not helped (a prompt touches many experts); the prompt cache (save before a switch, restore in seconds) covers the return.
- SSD endurance: background refills of RAM write nothing; only the prompt cache writes.

## Sources
- [Pre-gated MoE (ISCA 2024)](https://arxiv.org/html/2308.12066v3), [camera-ready](https://www.microsoft.com/en-us/research/uploads/prodnew/2024/05/isca24_pregated_moe_camera_ready.pdf)
- [Read-ME (NeurIPS 2024)](https://arxiv.org/html/2410.19123v1)
- [Pre-Attention Expert Prediction and Prefetching](https://www.alphaxiv.org/abs/2511.10676)
- [ReMoE: expert reuse through router fine-tuning](https://arxiv.org/html/2605.27081)
- [Cacheable by Design? (pre-registered negative result)](https://arxiv.org/pdf/2608.18261)
- [Not All Models Suit Expert Offloading (ICLR 2026)](https://iclr.cc/virtual/2026/poster/10011732)
- [Route Expert by Sequence, Not by Token (SeqTopK)](https://arxiv.org/html/2511.06494)
- [Task-level Mixture-of-Experts](https://www.arxiv.org/pdf/2110.03742)
- [DoMoE (IJCAI 2026)](https://www.ijcai.org/proceedings/2026/657)
- [SSD-LLaMA](https://arxiv.org/html/2609.18110v1), [Six MoE runtimes serving experts from SSD](https://originshq.com/blog/moe-ssd-expert-serving-runtimes/)
- [Who Should Own the Expert Cache? (the 64.7%-recall caution)](https://arxiv.org/html/2608.12103v1)

*Several 2026 entries are preprints seen through search summaries; read the papers before relying on their numbers.*
