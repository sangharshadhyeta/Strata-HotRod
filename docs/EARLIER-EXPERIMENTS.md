# Earlier experiments (a separate llama.cpp-based fork)

These results come from an earlier, separate project: a fork of llama.cpp with an MoE expert cache (solid.cpp), run in
August 2026. It is a different engine on different models, so **its numbers are not comparable with the Strata-HotRod
numbers** in the README. They are kept here because several of them explain why HotRod does things the way it does.

Machine: RTX 3060 12 GB, Linux. Main model: Gemma-4-26B-A4B. All numbers are from the earlier project's own notes
(`docs/moe-cache-colibri-notes.md`); a number marked "noise" was retracted there.

## What was measured

| Date | Experiment | Result | Kept? |
|---|---|---|---|
| 13 Aug | VRAM expert cache over CPU-offloaded experts (`-ngl 99 -ncmoe 99`, cache 2048, one 100-token answer) | decode 27.0 -> 41.6 tok/s (+54%), hit rate 62.1% (32,421 of 52,200). Hit rate was 44.2% on a 20-token run: the cache needs warming. | yes |
| 13 Aug | Why the first runs showed no gain | two defaults blocked the cache silently: the maximum batch it accepts defaulted to 1 (a 2-token prefill was rejected, no log line), and a 3 GB VRAM reserve left 47 MB on a 12 GB card against a 136 MB minimum pool | fixed by setting both |
| 15 Aug | Usage history written between runs (5 GiB memory cap, about 2x oversubscribed, 5 interleaved rounds) | first request +5.6% (4 of 5 rounds), nothing once warm. An earlier claim of "+21% to +50%" came from three single runs and was **noise**; it was retracted. | yes (small, free) |
| 15 Aug | Read-ahead hints to the kernel | +4.7% cold, about -8% warm, 3.4 GiB more page-cache pressure | no, off by default |
| 15 Aug | Pinning the hottest CPU experts with `mlock` (365 experts, 640 of the 5 GiB cap) | generation 2.03 / 7.59 / 10.30 / 11.40 tok/s with pinning against 2.20 / 8.73 / 11.74 without; more reclaim events (83,058 against 69,150) | no, off by default |
| 15 Aug | KV cache at q8_0 instead of f16 (65,536 context, 4 slots) | KV 2,180 -> 1,158 MiB; the memory fitter then put 2 more expert layers on the GPU (7 -> 9) with no code change; tok/s unchanged within noise (medians 77.2 and 81.5); 6 of 6 short prompts byte-identical; needle test 3 of 3 at about 24.7K tokens | documented, not the default |
| 15 Aug | KV allocated lazily through CUDA virtual memory | works behind a flag after a gating bug was fixed (it had captured the weights buffer); total VRAM use rose because the expert cache spent the saving | off by default |
| Oct | A pre-router trained live on a CPU thread (see `PREDICTOR-EXPERIMENTS.md`) | better eviction, worse admission; a paired re-check was a tie, or 7% slower, on Qwen | no |

## What carried over to HotRod

- **Pin in segments, not everything.** Pinning a slice of the hot experts lost here, because pinned pages cannot be
  reclaimed and starve everything else. HotRod's `STRATA_PIN_SEGS` page-locks its RAM tier in segments and gives them
  back, and gained 9% on the 3060 PC. This is a different mechanism on a different engine, not a re-run of the same test.
- **A live-learned predictor needs the model's own router as its base and has to run on the GPU.** The CPU-thread version
  here lost. HotRod's learned look-ahead starts from the router and learns a small correction (see
  `PREDICTOR-EXPERIMENTS.md`).
- **Interleaved A/B runs and a written retraction.** One-run "wins" turned out to be noise more than once, so every
  comparison since uses alternated pairs.

## Not included

The same project later got a "hot-rod" branch (KV streaming, sparse flash attention). Its own handoff notes of 1 Oct
2026 say it was written on a machine with no GPU and **never built or run**, so no result from it is listed here.
