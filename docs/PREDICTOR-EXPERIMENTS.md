# Expert predictor: every experiment so far, and the one small predictor to build (24 Oct 2026)

Machine: RTX 3060 12 GB, i5-12400F, 31 GB RAM, NVMe on a PCIe 3.0 link (3.1 GB/s, one expert read ~0.55 ms).
Model: Qwen3.8-Flash-Next IQ3_S (since 24 Oct the abliterated one on 8095). Sources: tools/predictor/README.md and
docs/perf-ideas.md on branch router-predictor-trace (/home/Projects/Strata-pred), docs/NEXT-STEPS-2026-10-22.md on main,
the job logs of 24 Oct, solid.cpp's docs/plan.md.

## The job
Ranks 0 to lo-1 (lo = STRATA_ROUTE_RESIDENT_RANKS, 3 today) are computed exactly; the ones not on the card are computed on the
CPU and, when not in the RAM tier, read from the NVMe first. That read is today's largest decode stage (cpu-wait 13-14 ms of a
~57 ms window). A predictor helps only by getting those reads started early enough, without extra reads that slow the
drive for the reads that matter.

## Accuracy (offline, trace1.bin: 24,708 decode tokens x 48 layers, exact routing; true rank 0-1 inside predicted top-4)
| predictor | lead 1 layer | lead 2 | lead 3 | notes |
|---|---|---|---|---|
| router reuse (next layer's real router on this layer's input; the engine's default) | 74.4 | 64.3 | 59.3 | free, already on the GPU |
| our ridge, 25k tokens | 89.2 / new text 74.7 | 85.2 / 67.5 | 82.5 / 63.5 | does not generalise |
| published linear d=1 (aswinkumar99), zero-shot | **89.4 / new text 85.5** | - | - | transfers NVFP4 -> IQ3_S |
| published + fine-tune on our trace | 93.5 / new text 85.9 | - | - | +0.4 on new text |
| online from the real router (Adam, test-then-train) | ~+0 | **72-75 (+8-10)** | | the only live learning that paid |
| online from zero / solid.cpp's SGD rule | below reuse all stream | | | |
| online fine-tune of the published model | 74.5 vs 89.8 static | | | live weight updates hurt |
| MLP vs linear (published study) | 76.4 vs 77.2 cov@10 | | | no gain from depth |
| layer 0 (no earlier layer): token table | 57 | | | needs a cross-token signal |
| next token's experts (SeqMoE, GRU, recency) | 5-56 | | | dropped |
| MTP hidden state as an input (published study) | out | | | |
Confidence: the published model is calibrated (p 0.70+ -> 99% right); raw router softmax is under-confident.

## In the engine
| experiment | date | result | why |
|---|---|---|---|
| shadow mode, CPU thread | 18 Oct | covers 77/89/93% (2/4/8 guesses) but ready only 0.2-0.5 ms before routing | lead, not recall |
| GPU node with the published matrix | 18 Oct | 82.5/93.7/96.9%, lead 0.5-1 ms for 57% of layers | 123 MB VRAM, ~5% node cost |
| ring of VRAM slots / mid-window swap (lend) | 18-19 Oct | 31-41 vs 43.9 tok/s | host-staged copy 1.0-1.6 ms > lead |
| learned 4-ahead CPU predictor | 18 Oct | 39.8-40.4 vs 43.9; removed | CPU cost (memory bandwidth) 16-19 ms/window |
| predictor rows as read-ahead (4-8 guesses) | 19 Oct | 38.7; 95% unused, floods the drive | waste reads cost more than hits gain |
| read-ahead gated by its own learned use-rate vs drive load (0040da13) | 19 Oct | **WIN 42.4 -> 43.6/43.9** | never ported to main |
| confidence gate STRATA_LA_PMIN | 19 Oct | no gain | fixed threshold |
| predictor-fed CPU read-ahead | 19 Oct | 49.0 vs 49.6 (alternated) | no net gain |
| masking resident experts before top-k (STRATA_LA_MASK) | 18 Oct | kept | CPU-pick accuracy 51 -> 62-69% |
| GPU look-ahead replaces CPU router recompute | 20 Oct | **WIN, default** (48.1 -> 47.2 ms/window) | |
| look-ahead edge skip | 21 Oct | +5% then; opt-in since 24 Oct (fights the self-tuning budget) | |
| STRATA_LA_MODEL (published, perf branch port) | 23-24 Oct | 61.2 first run, then 59-62 vs 60-63: neutral | more guesses right but more arrive late |
| + 3 I/O threads | 24 Oct | 59.6 vs 61.2 | more late reads |
| tail skip (rank >= lo not read ahead) | 24 Oct | default look-ahead already guesses ranks 0-2 only: 0 skips | with the model: 41 skips |
| solid.cpp live-trained prerouter | Oct | eviction +, admission -, paired recheck tie / -7% on Qwen | late and lossy, CPU thread |

## Where the time goes now (24 Oct, arrival counters, one 8.5K answer and the reasoning answer)
layers 2+: landed in time 558 / 1742, still being read 190 / 1167, still queued 22 / 482, never predicted 330 / 1846;
layers 0-1: never predicted 25 / 156. Read ahead ~15,500, used ~4,500 (29%).
So: half of the misses were never predicted, a third were predicted right but late, and 70% of reads are wasted.

## The one small predictor (design for branch predictor-online)
Built from what won, avoiding what lost:
1. **On the GPU, in the existing look-ahead node** (no CPU thread: the CPU versions lost on cost).
2. **Base = the target layers' own routers** (already in VRAM, cost nothing, the best start for live learning).
3. **Two heads per layer: lead 1 and lead 2.** Lead 2 doubles the time for a read, which is what the late third needs.
4. **A small learned correction per head** (low-rank or scale+bias, size chosen by tools/predictor/online_cheap.out),
   trained continuously on the GPU from each window's real routing (labels are free) - and from the prompt's prefill, which
   routes every prompt token exactly: the topic is learned before the answer starts.
5. **Two parts**: a long-term correction saved per model fingerprint (like the stand-in tuning file), and a per-request
   topic correction reset at each request and warmed on its prefill.
6. **Only ranks < lo, where lo is whatever the user sets** (STRATA_ROUTE_RESIDENT_RANKS: 3-9 -> ranks 0-2, 6-9 -> ranks 0-5;
   after a brake the tail ranks are computed exactly, so they are predicted too). Never a fixed rank count: the heads are trained
   on the full routed top-10 order so any lo works without retraining, and **only guesses worth a read**: a guess is read when its learned hit rate beats the drive's measured
   load (the 19 Oct self-regulating rule, the one gate that won) - no fixed K, no fixed threshold.
7. Layer 0-1 stay out of scope for now (no in-token lead; the token table reached 57%).
Gate to build: the offline replay shows the small correction keeps most of the +8-10 points at lead 2; then a prefill trace
checks the topic part; then the engine A/B on the 8.5K speed run, alternated, every answer read.

## The repeat (24 Oct 14:15, 8.5K run / code / reasoning / prose, tok/s, every answer correct)
defaults 67.1 / 43.4 / 53.0 / 32.4 and 66.6 / 43.9 / 53.2 / 28.7; published predictor + tail skip 63.8 / 42.5 / 39.6 / 32.8 and
66.4 / 46.1 / 55.7 / 31.2. The published model stays opt-in: neutral at best on this box.

## What Aswin's study teaches for ours (compact/heads/197 + 199, read only, none of their code run)
1. **The metric that matters is precision among NON-resident experts** (p@1/2/4 and miss-recall@K), not cov@10: most routed
   experts are resident already. Ours must be trained and scored on the picks the card does not hold (our STRATA_LA_MASK result agrees).
2. **Loss**: regress the router's own top-32 logits (centred MSE) plus a hinge keeping the other 480 below the 32nd minus 0.1. That
   teaches the order inside the top and keeps the tail down; better suited than plain cross-entropy on 10 labels.
3. **State is nearly everything.** Temporal counters, a recurrent state and a cross-token head add 0-5 points at lead 2-3 and ~0 at
   lead 1 (e.g. layer 34 lead 2: 63.5 -> 68.7; layer 12 lead 2: 80.5 -> 80.3). A linear map is as good as an MLP. So: small and linear.
4. **Size**: a 1M-parameter low-rank shape per layer is their deployment target; our low-rank correction is in that range.
5. **Calibration** (Platt per layer, 199_platt.py): turns scores into probabilities, which the "read only if worth it" gate needs.
6. **Their replay shows the same wall as ours**: learned prefetch precision 14-20%, most of the rest late; the large wins came from
   eviction + substitution (ref198: 38 misses/token vs LRU 68). Their reuse heads (will this resident be used within 32/128 tokens?)
   are the learned part that pays: keep the right experts, not only fetch new ones - solid.cpp found the same (eviction won).
7. **History resets per request** - matches our per-request topic part.

So the small predictor gets two outputs per layer: (a) routing for lead 1-2 among non-resident experts (low-rank correction on the
real router, centred-MSE + hinge loss, Platt-calibrated, gated by drive load), (b) a reuse score for the card's and the RAM tier's
residents (which to keep), the part both prior studies found to pay.
